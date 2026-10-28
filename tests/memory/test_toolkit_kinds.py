"""The per-kind shapes fingerprints compare: file shapes, dialogue observation shapes and shell output shapes."""

import io
import json
import sys

import pytest

from unify.memory.analysis import shapes, shellout, transitions

INVOICES = (
    "vendor_id,invoice_no,amount,due_date,status\n"
    "V-17,INV-0042,1250.50,2026-10-14,open\n"
    "V-03,INV-0043,99,2026-10-21,paid\n"
)


def _dumps(s):
    return json.dumps(s, sort_keys=True)


# --- shapes ----------------------------------------------------------------------------------------------


def test_csv_shape_has_columns_and_types_and_no_values():
    s = shapes.shape("finance/ap/invoices-2026-10.csv", INVOICES.encode())
    assert s["format"] == "csv" and s["delimiter"] == "," and s["header"] is True
    assert s["columns"] == ["vendor_id", "invoice_no", "amount", "due_date", "status"]
    assert s["types"] == ["str", "str", "float", "date", "str"]
    assert s["encoding"] == "utf-8" and s["stats"]["rows"] == 2
    text = _dumps(s)
    for value in ("V-17", "INV-0042", "1250.50", "2026-10-14", "open", "paid"):
        assert value not in text


def test_tab_delimited_file_named_csv_is_told_apart_and_signatures_ignore_row_counts():
    tsv = INVOICES.replace(",", "\t").encode()
    s_tab = shapes.shape("inventory.csv", tsv)
    s_comma = shapes.shape("inventory.csv", INVOICES.encode())
    assert s_tab["delimiter"] == "\t" and s_tab["columns"] == s_comma["columns"]
    assert shapes.signature(s_tab) != shapes.signature(s_comma)
    longer = INVOICES + "V-04,INV-0044,7.25,2026-11-01,open\n"
    assert shapes.signature(shapes.shape("a.csv", longer.encode())) == shapes.signature(
        s_comma,
    )
    assert shapes.conforms(s_tab, {"format": "csv", "delimiter": ","}) == [
        "delimiter: expected ',', found '\\t'",
    ]
    assert (
        shapes.conforms(s_comma, {"delimiter": ",", "columns": s_comma["columns"]})
        == []
    )


def test_headerless_and_mixed_columns():
    s = shapes.shape("r.csv", b"1,2026-01-01,x\n2,2026-01-02T10:00:00Z,7\n")
    assert s["header"] is False and s["columns"] is None
    assert s["types"] == ["int", "datetime", "mixed:int+str"]


def test_json_key_tree_without_values():
    data = {
        "vendor_id": "V-1",
        "amount": -12.5,
        "positive": True,
        "lines": [{"sku": "A", "n": 2}],
    }
    s = shapes.shape("finance/ap/credit_notes.json", json.dumps(data).encode())
    assert s["format"] == "json"
    assert s["tree"] == {
        "amount": "float",
        "lines": [{"n": "int", "sku": "str"}],
        "positive": "bool",
        "vendor_id": "str",
    }
    assert "V-1" not in _dumps(s)
    jl = shapes.shape("x.jsonl", b'{"a": 1}\n{"a": 2}\n{"a": "x"}\n')
    assert jl["format"] == "jsonl" and jl["tree"] == [{"a": "int"}, {"a": "str"}]


YAML = """\
# runbook config
name: payables
schedule:
  day: 14
  enabled: true
steps:
  - read: invoices
  - pay: vendors
"""


def test_yaml_key_tree_with_and_without_pyyaml(monkeypatch):
    s = shapes.shape("conf/run.yaml", YAML.encode())
    assert s["format"] == "yaml"
    assert s["tree"]["schedule"] == {"day": "int", "enabled": "bool"}
    assert isinstance(s["tree"]["steps"], list)
    basic = shapes.yaml_tree_basic(YAML)
    assert basic == {
        "name": "str",
        "schedule": {"day": "int", "enabled": "bool"},
        "steps": ["list"],
    }
    monkeypatch.setitem(sys.modules, "yaml", None)
    assert shapes.shape("conf/run.yaml", YAML.encode())["tree"] == basic


def test_xlsx_sheets_and_unparsed_fallback(monkeypatch):
    openpyxl = pytest.importorskip("openpyxl")
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Rates"
    ws.append(["currency", "rate", "effective"])
    ws.append(["EUR", 1.07, "2026-10-01"])
    wb.create_sheet("Notes")
    buf = io.BytesIO()
    wb.save(buf)
    data = buf.getvalue()
    s = shapes.shape("rates.xlsx", data)
    assert s == {
        "format": "xlsx",
        "sheets": [
            {"name": "Rates", "header": ["currency", "rate", "effective"]},
            {"name": "Notes", "header": None},
        ],
    }
    assert "EUR" not in _dumps(s)
    assert shapes.shape("blob-without-extension", data)["format"] == "xlsx"
    monkeypatch.setitem(sys.modules, "openpyxl", None)
    assert shapes.shape("rates.xlsx", data) == {"format": "xlsx-unparsed"}


def test_text_and_binary():
    s = shapes.shape("finance/runbooks/payables.md", "# Payables\n\nRun it.\n".encode())
    assert s == {
        "format": "text",
        "encoding": "utf-8",
        "stats": {"lines": 3, "bytes": 20},
    }
    assert shapes.shape("a.txt", "caf\xe9\n".encode("latin-1"))["encoding"] == "latin-1"
    assert shapes.shape("img.png", b"\x89PNG\0\0")["format"] == "binary"
    assert shapes.signature(shapes.shape("a.md", b"x\ny\n")) == shapes.signature(
        shapes.shape("b.md", b"z\n"),
    )


def test_shape_is_deterministic_and_bounded():
    big = ("a,b\n" + "1,2\n" * 400_000).encode()
    assert len(big) > shapes.PARSE_LIMIT
    s1, s2 = shapes.shape("big.csv", big), shapes.shape("big.csv", big)
    assert (
        s1 == s2 and s1["types"] == ["int", "int"] and s1["stats"]["lines"] == 400_001
    )


# --- observation shapes ----------------------------------------------------------------------------------


def test_observation_shapes():
    a = transitions.observation_shape(
        "Status: health 9\nInventory: wood 1\nFacing: tree",
    )
    b = transitions.observation_shape(
        "Status: health 3\nInventory: wood 7, stone 2\nFacing: water",
    )
    assert a == b == "text:Status:|Inventory:|Facing:"
    assert (
        transitions.observation_shape('{"grid": [[1, 2]]}')
        == 'tree:{"grid": [["int"]]}'
    )
    assert transitions.observation_shape({"ok": True}) == 'tree:{"ok": "bool"}'
    assert transitions.observation_shape("correct") == transitions.observation_shape(
        "incorrect",
    )


# --- shell outputs ---------------------------------------------------------------------------------------


def test_shell_output_shapes():
    failed = "F\n=== FAILURES ===\nE  assert 1 == 2\n=== 1 failed, 2 passed in 0.31s ==="
    s0 = shellout.output_shape(failed)
    assert s0["format"] == "lines" and s0["last"] == "=== 9 a, 9 a a 9.9a ==="
    assert "0.31" not in json.dumps(s0)
    s2 = shellout.output_shape('[{"name": "a", "version": "1"}]')
    assert s2["format"] == "json" and s2["tree"] == [{"name": "str", "version": "str"}]
    assert shellout.output_shape("") == {"format": "empty", "lines": 0}
    t = shellout.output_shape("a.txt  12\nb.txt  7\n")
    assert t["format"] == "table" and t["delimiter"] == "whitespace" and t["width"] == 2
    assert shellout.signature(s0) == shellout.signature(
        shellout.output_shape("...\n=== 3 passed in 0.12s ==="),
    )

"""The per-kind toolkit: file shapes, dialogue transitions, shell output shapes and recorded blobs."""

import io
import json
import sys

import pytest

from unify.memory_v2.analysis import recorded, shapes, shellout, transitions
from unify.memory_v2.episodes import Action

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


# --- transitions -----------------------------------------------------------------------------------------


def _crafter_row():
    def act(i, text, obs, status="ok"):
        return {
            "index": i,
            "cell": -1,
            "channel": "dialogue:user",
            "method": "reply",
            "args": [text],
            "kwargs": {},
            "response": obs,
            "status": status,
            "effect": "unknown",
            "error": None,
            "kind": "dialogue",
        }

    return {
        "episode_id": "c1",
        "request": ["Status: health 9\nInventory: none\nFacing: tree"],
        "actions": [
            act(0, "do", "Status: health 9\nInventory: wood 1\nFacing: tree"),
            {
                "index": 1,
                "channel": "world",
                "method": "noop",
                "args": [],
                "kwargs": {},
                "response": {},
                "status": "ok",
            },  # a tool row (no kind): ignored
            act(2, "do", "Status: health 9\nInventory: wood 2\nFacing: tree"),
            act(
                3,
                "make_wood_pickaxe",
                "Status: health 9\nInventory: wood 2\nFacing: tree",
            ),
            act(4, "noop", None, "unrecorded"),
        ],
    }


def _wood(obs):
    for line in (obs or "").splitlines():
        if line.startswith("Inventory:"):
            parts = line.split()
            return int(parts[parts.index("wood") + 1]) if "wood" in parts else 0
    return None


def test_transitions_pair_actions_with_before_and_after():
    ts = transitions.transitions([_crafter_row()])
    assert [(t.index, t.action) for t in ts] == [
        (0, "do"),
        (2, "do"),
        (3, "make_wood_pickaxe"),
    ]
    assert ts[0].before.startswith("Status") and "wood 1" in ts[0].after
    assert ts[1].before == ts[0].after
    tbl = transitions.table(ts, value=lambda t: _wood(t.after) - _wood(t.before))
    assert tbl["do"] == {"1": [("c1", 0), ("c1", 2)]}
    assert (
        transitions.conflicts(ts, value=lambda t: _wood(t.after) - _wood(t.before))
        == {}
    )
    assert set(transitions.conflicts(ts)) == {"do"}  # raw next observations differ


def test_check_reports_every_contradiction():
    ts = transitions.transitions([_crafter_row()])

    def predict(
        t,
    ):  # "do adds one wood; crafting needs 3 wood and otherwise changes nothing"
        if t.action == "do":
            return _wood(t.before) + 1
        return _wood(t.before) - (3 if _wood(t.before) >= 3 else 0)

    res = transitions.check(ts, predict, lambda t: _wood(t.after))
    assert res.ok and res.checked == 3
    bad = transitions.check(
        ts,
        lambda t: 0,
        lambda t: _wood(t.after),
        in_scope=lambda t: t.action == "do",
    )
    assert not bad.ok and bad.out_of_scope == 1
    assert [(e, i) for e, i, _, _ in bad.contradictions] == [("c1", 0), ("c1", 2)]
    raised = transitions.check(ts, lambda t: 1 / 0, lambda t: 1)
    assert raised.contradictions[0][2] == "raised ZeroDivisionError"
    rows = transitions.fixture(ts)
    assert json.loads(json.dumps(rows))[0]["action"] == "do"


def test_transitions_from_action_objects_and_observation_shapes():
    acts = [
        Action(
            -1,
            "dialogue:user",
            "reply",
            ["submit"],
            {},
            "incorrect",
            "ok",
            kind="dialogue",
        ),
    ]
    [t] = transitions.from_actions("a1", acts)
    assert (t.index, t.action, t.after) == (0, "submit", "incorrect")
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


def _shell(i, cmd, code, tail):
    return {
        "index": i,
        "cell": 0,
        "channel": "shell:uv",
        "method": "run",
        "args": [cmd],
        "kwargs": {},
        "response": {"exit_code": code, "tail": tail},
        "status": "ok" if code == 0 else "error",
        "effect": "unknown",
        "error": None,
        "kind": "shell",
    }


def test_shell_output_shapes_and_groups():
    row = {
        "episode_id": "s1",
        "actions": [
            _shell(
                0,
                "uv run pytest tests/test_dates.py -x",
                1,
                "F\n=== FAILURES ===\nE  assert 1 == 2\n=== 1 failed, 2 passed in 0.31s ===",
            ),
            _shell(1, "uv run pytest", 0, "...\n=== 3 passed in 0.12s ==="),
            _shell(
                2,
                "uv pip list --format json",
                0,
                '[{"name": "a", "version": "1"}]',
            ),
            {
                "index": 3,
                "channel": "shell:uv",
                "method": "spawn",
                "args": ["uv"],
                "kwargs": {},
                "response": None,
                "status": "unrecorded",
                "kind": "shell",
            },
        ],
    }
    recs = shellout.records([row])
    assert [r.index for r in recs] == [0, 1, 2] and recs[0].exit_code == 1
    s0 = shellout.output_shape(recs[0].tail)
    assert s0["format"] == "lines" and s0["last"] == "=== 9 a, 9 a a 9.9a ==="
    assert "0.31" not in json.dumps(s0)
    s2 = shellout.output_shape(recs[2].tail)
    assert s2["format"] == "json" and s2["tree"] == [{"name": "str", "version": "str"}]
    assert shellout.output_shape("") == {"format": "empty", "lines": 0}
    t = shellout.output_shape("a.txt  12\nb.txt  7\n")
    assert t["format"] == "table" and t["delimiter"] == "whitespace" and t["width"] == 2
    assert shellout.exit_codes(recs) == {"shell:uv": [0, 1]}
    groups = shellout.by_shape(recs)
    assert sorted(len(v) for v in groups.values()) == [1, 1, 1]
    assert shellout.signature(s0) == shellout.signature(
        shellout.output_shape(recs[1].tail),
    )


# --- recorded blobs --------------------------------------------------------------------------------------


def test_load_blob_and_worktree_queries(tmp_path):
    import hashlib

    data = INVOICES.encode()
    sha = hashlib.sha256(data).hexdigest()
    (tmp_path / sha).write_bytes(data)
    missing = "f" * 64
    (tmp_path / "index.json").write_text(
        json.dumps({"exported": [sha], "skipped": {missing: "over the per-blob cap"}}),
    )
    assert recorded.load_blob(sha, tmp_path) == data
    with pytest.raises(KeyError, match="per-blob cap"):
        recorded.load_blob(missing, tmp_path)
    with pytest.raises(ValueError):
        recorded.load_blob("../index.json", tmp_path)
    row = {
        "episode_id": "o1",
        "actions": [
            {
                "index": 0,
                "channel": "venmo",
                "method": "me",
                "args": [],
                "response": {},
            },
            {
                "index": 1,
                "channel": "worktree:workspace",
                "method": "read",
                "args": ["finance/ap/invoices.csv"],
                "kind": "worktree",
                "response": {
                    "blob_before": sha,
                    "blob_after": sha,
                    "size": len(data),
                    "shape": shapes.shape("invoices.csv", data),
                },
            },
        ],
    }
    assert [a["index"] for a in recorded.actions_of_kind(row, "tool")] == [0]
    [f] = recorded.worktree_files(row)
    assert f["path"] == "finance/ap/invoices.csv" and f["blob_before"] == sha
    assert recorded.blob_ids(row["actions"][1]) == [sha]
    assert (
        recorded.blob_ids({"kind": "worktree", "response": {"blob_before": "../x"}})
        == []
    )

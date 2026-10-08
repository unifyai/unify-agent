"""Scope is shape, not observed values (spec §F3a, D20): G2's held-out-value check.

The first part needs no sandbox: the perturbations are derived from types alone and are deterministic.
The second runs the gate (bubblewrap) on items that whitelist recorded values and on items that check
shape only.
"""

import csv
import io
import json
import shutil
from pathlib import Path

import pytest

from unify.memory_v2.admission import cover_problem, is_rejection
from unify.memory_v2.blobs import BlobStore
from unify.memory_v2.episodes import Action
from unify.memory_v2.evidence import EvidenceStore
from unify.memory_v2.gate import Gate
from unify.memory_v2.gitio import Repo
from unify.memory_v2.held_out import (
    _read_results,
    MAX_COVERS_PER_ITEM,
    MAX_FIELDS_PER_COVER,
    Case,
    Plan,
    run_plan,
    beyond_number,
    fresh_string,
    literal_affixes,
    perturb_value,
    plan,
)
from unify.memory_v2.sandbox_run import PytestOutcome
from tests.memory_v2.test_episodes import _ep

needs_bwrap = pytest.mark.skipif(
    shutil.which("bwrap") is None,
    reason="bubblewrap required",
)

STOCK = (
    b"sku,colour,qty,price,restocked\n"
    b"A-1,red,4,12.50,2026-09-30\n"
    b"B-2,blue,10,3.25,2026-10-02\n"
)
# Three ticket families (an enumeration of prefixes), one constant agent-id prefix, and an order relation
# (opened <= closed) inside every row.
TICKETS = (
    b"ticket_id,agent_id,opened,closed,minutes\n"
    b"NET-0042,A1243,2026-09-01,2026-09-05,30\n"
    b"DSK-0043,A1246,2026-09-02,2026-09-02,45\n"
    b"APP-0044,A1488,2026-09-03,2026-09-04,15\n"
)
# A mixed-sign column: credits negative, debits positive (a sign convention, which is shape).
LEDGER = (
    b"entry,kind,amount\n"
    b"L-1,credit,-50.00\n"
    b"L-2,debit,120.00\n"
    b"L-3,debit,30.00\n"
)
SEVEN = {f"f{i}": i for i in range(1, 8)}


def _search(colour, limit, status="ok", response=None, error=None):
    return Action(
        0,
        "shop",
        "search",
        [],
        {"colour": colour, "limit": limit},
        response if response is not None or status != "ok" else {"items": []},
        status,
        "read",
        error,
    )


def _wt_read(index, path, blob, size):
    return Action(
        index,
        "worktree:workspace",
        "read",
        [path],
        {},
        {"blob_before": blob, "blob_after": blob, "size": size},
        "ok",
        "read",
        kind="worktree",
    )


def _actions(blob, tickets=None, ledger=None, mauve=None):
    acts = [
        _search("red", 5, response={"items": [{"sku": "A-1"}]}),
        _search("blue", 10),
        _search("mauve", 5, status="error", error="unknown colour"),
        Action(
            3,
            "shell:make",
            "run",
            ["make check"],
            {},
            {"exit_code": 0, "tail": "3 checks passed"},
            "ok",
            kind="shell",
        ),
        Action(
            4,
            "worktree:workspace",
            "read",
            ["store/2026/stock.csv"],
            {},
            {"blob_before": blob, "blob_after": blob, "size": len(STOCK)},
            "ok",
            "read",
            kind="worktree",
        ),
        Action(
            5,
            "dialogue:user",
            "reply",
            ["look"],
            {},
            {"room": "hall", "items": ["key", "lamp"], "steps": 3},
            "ok",
            kind="dialogue",
        ),
        Action(
            6,
            "worktree:workspace",
            "write",
            ["store/2026/stock.csv"],
            {},
            None,
            "error",
            "write",
            "permission denied",
            kind="worktree",
        ),
    ]
    if tickets is None or ledger is None:
        return acts
    return acts + [
        _wt_read(7, "desk/2026/tickets.csv", tickets, len(TICKETS)),
        # an auth failure on the same method: its keywords equal an accepted call's
        _search("red", 5, status="error", error="token expired"),
        # a rejection of another method
        Action(9, "shop", "buy", [], {"colour": "mauve"}, None, "error", error="no"),
        Action(
            10,
            "dialogue:user",
            "reply",
            ["count"],
            {},
            dict(SEVEN),
            "ok",
            kind="dialogue",
        ),
        _wt_read(11, "books/2026/ledger.csv", ledger, len(LEDGER)),
        # a refused write that records the unchanged file (before == after)
        _failed_write(12, blob, blob),
        # a refused write that would have changed one colour
        _failed_write(13, blob, mauve),
    ]


def _failed_write(index, before, after):
    return Action(
        index,
        "worktree:workspace",
        "write",
        ["store/2026/stock.csv"],
        {},
        {"blob_before": before, "blob_after": after, "size": len(STOCK)},
        "error",
        "write",
        "permission denied",
        kind="worktree",
    )


# Declared semantic types (D21): month and share (a probability) may be restricted to their domains once
# declared; hours is untagged.
PERIODS = (
    b"period_id,month,share,hours\n" b"P-1,9,0.25,100.00\n" b"P-2,10,0.50,120.00\n"
)
# Every month observed (a twelve-row monthly file); zero-padded months; a wide observation whose declared
# field comes after more than MAX_FIELDS_PER_COVER other fields.
MONTHS = b"month,sales\n" + b"".join(b"%d,%d\n" % (m, 10 * m) for m in range(1, 13))
PADDED = b"month,share\n09,0.25\n10,0.50\n"
# The first recorded value of each column is in the declared domain; a later one is not.
RANGES = b"amount,pct\n5.00,50\n-3.00,110\n"
WIDE = {**{f"f{i:02d}": i for i in range(1, 21)}, "zmonth": 5}
_ROWS = STOCK.split(b"\n")
# refused writes whose after-blob only reorders the rows, appends a copy of a row, or appends a new row
REORDERED = b"\n".join([_ROWS[0], _ROWS[2], _ROWS[1], b""])
APPENDED_COPY = STOCK + _ROWS[1] + b"\n"
APPENDED_NEW = STOCK + b"C-3,green,4,3.25,2026-09-30\n"


def _all_actions(store):
    """The shared recording: actions 0-13 of :func:`_actions`, 14-16 (refused writes by row change), 17."""
    stock = store.put(STOCK)
    acts = _actions(
        stock,
        store.put(TICKETS),
        store.put(LEDGER),
        store.put(STOCK.replace(b",blue,", b",mauve,")),
    )
    acts += [
        _failed_write(14, stock, store.put(REORDERED)),
        _failed_write(15, stock, store.put(APPENDED_COPY)),
        _failed_write(16, stock, store.put(APPENDED_NEW)),
        _wt_read(17, "plan/2026/periods.csv", store.put(PERIODS), len(PERIODS)),
        _wt_read(18, "plan/2026/months.csv", store.put(MONTHS), len(MONTHS)),
        _wt_read(19, "plan/2026/padded.csv", store.put(PADDED), len(PADDED)),
        Action(
            20,
            "dialogue:user",
            "reply",
            ["rates"],
            {},
            {"pct": "40%", "month": None},
            "ok",
            kind="dialogue",
        ),
        Action(
            21,
            "dialogue:user",
            "reply",
            ["wide"],
            {},
            dict(WIDE),
            "ok",
            kind="dialogue",
        ),
        _wt_read(22, "plan/2026/ranges.csv", store.put(RANGES), len(RANGES)),
        Action(
            23,
            "dialogue:user",
            "reply",
            ["rates"],
            {},
            {"pct": 40, "month": 5},
            "ok",
            kind="dialogue",
        ),
        # two feedback observations: a constant message-kind tag and a constant number (identity and
        # format), a varying attempt and a varying flag
        _dl(24, {"type": "SubmitFeedback", "version": 3, "attempt": 1, "ok": True}),
        _dl(25, {"type": "SubmitFeedback", "version": 3, "attempt": 2, "ok": False}),
        # SEVEN and WIDE again with other values, so their fields vary across covers
        _dl(26, {k: 10 * v for k, v in SEVEN.items()}),
        _dl(27, {**{k: 100 + v for k, v in WIDE.items()}, "zmonth": 7}),
        _dl(28, {"room": "yard", "items": ["rope", "map"], "steps": 5}),
    ]
    return acts


def _dl(index, obs):
    return Action(
        index,
        "dialogue:user",
        "reply",
        ["look"],
        {},
        obs,
        "ok",
        kind="dialogue",
    )


@pytest.fixture
def recorded(tmp_path):
    store = BlobStore(tmp_path / "blobs")
    return store, _all_actions(store)


def _covers(acts, *idx):
    return [("h1", i, acts[i]) for i in idx]


# --- perturbations from types alone (no sandbox) ---------------------------------------------------------


def test_fresh_string_keeps_character_classes_and_is_unseen():
    seen = {"INV-0042", "INV-0043"}
    out = fresh_string("INV-0042", b"seed", seen)
    assert out is not None and out not in seen and len(out) == len("INV-0042")
    assert out[3] == "-" and out[:3].isupper() and out[4:].isdigit()
    assert fresh_string("INV-0042", b"seed", seen) == out  # deterministic
    assert fresh_string("INV-0042", b"other", seen) != out
    # hex-like strings stay hex-like (a format, which is shape)
    hexy = fresh_string("a3f9-0c", b"seed", set())
    assert all(c in "0123456789abcdef-" for c in hexy)
    assert fresh_string("--", b"seed", set()) is None  # nothing to vary


def test_literal_format_parts_stay_and_only_the_varying_part_changes():
    assert literal_affixes(["E1243", "E1246", "E1488"]) == (
        "E",
        "",
    )  # cut to a class-run boundary
    assert literal_affixes(["NET-0042", "DSK-0043", "APP-0044"]) == ("", "")
    assert literal_affixes(["a@x.org", "bb@x.org"]) == ("", "@x.org")
    assert literal_affixes(["E1243"]) == ("", "")  # one value: nothing is literal
    out = fresh_string("A1243", b"seed", set(), ["A1243", "A1246", "A1488"])
    assert out[0] == "A" and out[1:].isdigit() and len(out) == 5 and out != "A1243"
    # an enumeration of prefixes is not a format: the prefix varies too
    seen = {"NET-0042", "DSK-0043", "APP-0044"}
    outs = {
        fresh_string("NET-0042", bytes([i]), seen, sorted(seen))[:3] for i in range(8)
    }
    assert outs - {"NET", "DSK", "APP"}
    # with a single observed value the non-alphanumeric skeleton stays and every run varies
    one = fresh_string("ab-12.x", b"seed", set(), ["ab-12.x"])
    assert one[2] == "-" and one[5] == "." and one != "ab-12.x"


def _rows_runner(rows):
    """A stand-in for the box: writes *rows* (by case id) as the results; nothing runs."""

    def runner(argv, *, ro, rw, cwd, timeout_s, env=None):
        (out,) = rw
        (out / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

    return runner


def test_notes_count_beyond_five_and_take_only_known_error_names(tmp_path):
    fam = ("dialogue", "dialogue:user", "reply")
    names = [f"f{i}" for i in range(1, 12)]
    cases = [
        Case(
            ("h1", 10),
            None,
            fam,
            "dialogue",
            {"kind": "dialogue", "observation": {}},
        ),
    ]
    cases += [
        Case(("h1", 10), n, fam, "dialogue", {"kind": "dialogue", "observation": {}})
        for n in names
    ]
    p = Plan(cases=cases, exempt={fam: set(names[:7])})
    rows = [{"id": 0, "outcome": "handled", "calls": 0}]
    rows += [
        {"id": i, "outcome": "refused", "calls": 0} for i in range(1, 8)
    ]  # 7 allowed
    rows += [
        {"id": 8, "outcome": "error", "calls": 0, "raised": "KeyError"},
        {"id": 9, "outcome": "error", "calls": 0, "raised": "LEAKED-VALUE-1234567890"},
        {"id": 10, "outcome": "error", "calls": 0, "raised": "Sneaky"},
        {"id": 11, "outcome": "refused", "calls": 0},
    ]
    v = run_plan(
        "env/dialogue_user:parse",
        p,
        tree=tmp_path,
        python=Path("/nonexistent"),
        work=tmp_path / "w",
        runner=_rows_runner(rows),
    )
    assert v.refused == ["f11"]
    allowed = [n for n in v.notes if "allowed by a covered recorded rejection" in n]
    assert len(allowed) == 6 and allowed[-1].startswith("and 2 more")
    raised = [n for n in v.notes if "raised before any environment call" in n]
    assert raised == [
        "held-out values raised before any environment call on f8 (KeyError); "
        "only MemoryInputError is judged",
    ]
    assert not any("LEAKED" in n or "Sneaky" in n for n in v.notes)


@pytest.mark.parametrize(
    "cover, varied",
    [(14, set()), (15, set()), (16, {"sku", "colour"})],
    ids=["reordered-rows", "appended-copy", "appended-new-row"],
)
def test_a_file_rejection_compares_rows_by_content_not_position(
    recorded,
    cover,
    varied,
):
    """N8: a reorder or an appended copy varies no field; an appended row varies only its new values."""
    store, acts = recorded
    p = plan(
        "env/worktree_workspace:r",
        _covers(acts, 4, cover),
        seen=acts,
        blob=store.get,
    )
    fam = ("worktree", "worktree:workspace", "store/#/*.csv")
    assert p.exempt.get(fam, set()) == varied
    if not varied:
        assert any("changes no field" in n and "exempts nothing" in n for n in p.notes)


def test_the_manifest_takes_declared_types_from_the_fixed_list_only():
    from unify.memory_v2.manifest import SEMANTIC_TYPES, ManifestError, parse_manifest

    assert set(SEMANTIC_TYPES) == {
        "month",
        "day_of_month",
        "hour",
        "probability",
        "percentage",
        "nonneg_count",
        "nonneg_money",
        "currency_code",
    }
    item = {
        "item": "env/worktree_workspace:read_periods",
        "kind": "env_function",
        "covers": [["h1", 17]],
        "field_types": {"month": "month", "share": "probability"},
    }
    (parsed,) = parse_manifest({"items": [item]}).items
    assert parsed.field_types == {"month": "month", "share": "probability"}
    assert (
        parse_manifest({"items": [{**item, "field_types": None}]}).items[0].field_types
        == {}
    )
    for bad in (
        {"month": "fiscal_month"},  # not on the list
        {"month": 3},
        ["month"],
        {"": "month"},
        {"a\nb": "month"},
        {"x" * 201: "month"},
    ):
        with pytest.raises(ManifestError):
            parse_manifest({"items": [{**item, "field_types": bad}]})
    note = {
        "item": "env/x/NOTES.md#a",
        "kind": "env_note",
        "field_types": {"a": "month"},
    }
    with pytest.raises(ManifestError):
        parse_manifest({"items": [note]})


def test_the_manifest_takes_an_input_form_from_the_fixed_list_only():
    from unify.memory_v2.manifest import (
        INPUT_KINDS,
        ManifestError,
        describe_input_kinds,
        parse_manifest,
    )

    assert set(INPUT_KINDS) == {"path", "text", "bytes", "observation", "env"}
    import unify.memory_v2.manifest as manifest_module

    rules = manifest_module.__doc__
    assert "{input_forms}" not in rules
    assert ", ".join(f"``{name}``" for name in INPUT_KINDS) in rules
    for name in INPUT_KINDS:
        assert f"{name} (" in describe_input_kinds()
    item = {
        "item": "env/worktree_workspace:parse_load_log",
        "kind": "env_function",
        "covers": [["h1", 4]],
    }
    for kind in INPUT_KINDS:
        (parsed,) = parse_manifest({"items": [{**item, "input": kind}]}).items
        assert parsed.input == kind
    assert parse_manifest({"items": [item]}).items[0].input is None
    for bad in ("file", "Text", "", 3, ["text"], {"text": 1}):
        with pytest.raises(ManifestError):
            parse_manifest({"items": [{**item, "input": bad}]})
    note = {"item": "env/x/NOTES.md#a", "kind": "env_note", "input": "text"}
    with pytest.raises(ManifestError):
        parse_manifest({"items": [note]})


def test_a_declared_field_gets_unseen_in_domain_values_and_one_out_of_domain(recorded):
    store, acts = recorded
    p = plan(
        "env/worktree_workspace:read_periods",
        _covers(acts, 17),
        seen=acts,
        blob=store.get,
        field_types={"month": "month", "share": "probability", "absent": "hour"},
    )
    by = {}
    for c in p.cases:
        if c.field:
            row = list(csv.reader(io.StringIO(c.file.decode())))[1]
            col = ["period_id", "month", "share", "hours"].index(c.field)
            by.setdefault((c.field, c.side), []).append(row[col])
    months_in = [int(v) for v in by[("month", "in")]]
    assert months_in and all(1 <= m <= 12 and m not in (9, 10) for m in months_in)
    assert {1, 12} <= set(months_in)  # both ends of the unseen domain
    assert by[("month", "out")] == ["0", "13"]  # both sides of the domain
    shares_in = by[("share", "in")]
    assert all(len(v.split(".")[1]) == 2 and 0 < float(v) < 1 for v in shares_in)
    assert not set(shares_in) & {"0.25", "0.50"}
    assert by[("share", "out")] == ["-0.10", "1.10"]
    assert set(by) >= {
        ("hours", None),
        ("period_id", None),
    }  # untagged fields keep the shape rule
    assert any("absent" in n and "not checked" in n for n in p.notes)
    again = plan(
        "env/worktree_workspace:read_periods",
        _covers(acts, 17),
        seen=acts,
        blob=store.get,
        field_types={"month": "month", "share": "probability", "absent": "hour"},
    )
    assert [c.file for c in again.cases] == [c.file for c in p.cases]  # deterministic
    wrong = plan(
        "env/worktree_workspace:read_periods",
        _covers(acts, 17),
        seen=acts,
        blob=store.get,
        field_types={"period_id": "month", "hours": "hour"},
    )
    # hours holds 100.00 and 120.00: a value outside the declared domain contradicts the declaration
    assert wrong.mismatched == {"hours": "hour"}
    # "P-1" is not written as a month at all: noted, and the field keeps the untagged shape check
    assert "period_id" not in wrong.mismatched
    assert any("period_id" in n and "not written as" in n for n in wrong.notes)
    assert any(c.field == "period_id" and c.side is None for c in wrong.cases)


def _declared(acts, store, cover, types, item="env/worktree_workspace:r"):
    p = plan(item, _covers(acts, cover), seen=acts, blob=store.get, field_types=types)
    out = {}
    for c in p.cases:
        if c.side:
            if c.file is not None:
                rows = list(csv.DictReader(io.StringIO(c.file.decode())))
                changed = [r[c.field] for r in rows]
                v = next(
                    x
                    for x, o in zip(changed, _first_col(c, store, acts, cover))
                    if x != o
                )
            elif "observation" in c.payload:
                v = c.payload["observation"][c.field]
            else:
                v = c.payload["kwargs"][c.field]
            out.setdefault((c.field, c.side), []).append(v)
    return p, out


def _first_col(c, store, acts, cover):
    blob = acts[cover].response["blob_before"]
    return [r[c.field] for r in csv.DictReader(io.StringIO(store.get(blob).decode()))]


def test_a_fully_observed_domain_checks_only_the_out_of_domain_side(recorded):
    """I1: all twelve months recorded: no unseen in-domain value exists, which is not a contradiction."""
    store, acts = recorded
    p, by = _declared(acts, store, 18, {"month": "month"})
    assert not p.mismatched and ("month", "in") not in by
    assert by[("month", "out")] == ["0", "13"]
    assert any(
        "month" in n and "every in-domain value is already recorded" in n
        for n in p.notes
    )


def test_every_recorded_value_must_fit_the_declared_type(recorded):
    """A declaration contradicted by any recorded value of the field fails, not only by the first one."""
    store, acts = recorded
    p, _ = _declared(acts, store, 22, {"amount": "nonneg_money", "pct": "percentage"})
    assert p.mismatched == {"amount": "nonneg_money", "pct": "percentage"}


def test_a_field_checked_on_one_cover_is_not_noted_as_unwritten_on_another(recorded):
    store, acts = recorded
    p = plan(
        "env/dialogue_user:r",
        _covers(acts, 20, 23),
        seen=acts,
        blob=store.get,
        field_types={"month": "month"},
    )
    assert any(c.field == "month" and c.side == "out" for c in p.cases)
    assert not any("month" in n and "not written as" in n for n in p.notes)


def test_representations_are_normalised_or_skipped_never_failed(recorded):
    store, acts = recorded
    # zero-padded months are written back with the same width
    p, by = _declared(acts, store, 19, {"month": "month", "share": "probability"})
    assert not p.mismatched
    assert all(
        len(v) == 2 and v.isdigit() and 1 <= int(v) <= 12 for v in by[("month", "in")]
    )
    assert by[("month", "out")] == ["00", "13"]
    # "40%" and a null are representations the check does not handle: noted, not failed
    p2, by2 = _declared(
        acts,
        store,
        20,
        {"pct": "percentage", "month": "month"},
        "env/dialogue_user:r",
    )
    assert not p2.mismatched and not by2
    assert any("pct" in n and "not written as" in n for n in p2.notes)
    assert any("month" in n and "not written as" in n for n in p2.notes)


def test_a_bad_currency_code_is_one_upper_casing_cannot_fix(recorded):
    from unify.memory_v2.held_out import declared_values, seed_of

    got = declared_values(
        "currency_code",
        "USD",
        ["USD", "EUR"],
        seed_of("i", "h1", 0, "c"),
        {"USD"},
    )
    assert got.ins and all(
        len(v) == 3 and v.isupper() and v not in ("USD", "EUR") for v in got.ins
    )
    import re as _re

    assert got.outs and not any(_re.fullmatch("[A-Z]{3}", v.upper()) for v in got.outs)


def test_declared_fields_come_first_within_the_field_limit(recorded):
    store, acts = recorded
    p, by = _declared(acts, store, 21, {"zmonth": "month"}, "env/dialogue_user:r")
    assert ("zmonth", "in") in by and ("zmonth", "out") in by
    assert not any("zmonth" in n and "not in any covered input" in n for n in p.notes)
    untagged = {c.field for c in p.cases if c.field and c.side is None}
    assert len(untagged) == MAX_FIELDS_PER_COVER


def test_a_declared_tool_keyword_is_checked_on_the_recorded_call(recorded):
    store, acts = recorded
    p, by = _declared(acts, store, 0, {"limit": "nonneg_count"}, "env/shop:search")
    assert all(
        isinstance(v, int) and v >= 0 and v not in (5, 10) for v in by[("limit", "in")]
    )
    assert by[("limit", "out")] == [-1]
    assert {c.param for c in p.cases if c.field == "limit"} == {"limit"}


def test_declared_cases_route_other_exceptions_to_notes_and_need_a_handled_baseline(
    tmp_path,
):
    fam = ("dialogue", "dialogue:user", "reply")

    def case(field, side):
        return Case(
            ("h1", 10),
            field,
            fam,
            "dialogue",
            {"kind": "dialogue", "observation": {}},
            side=side,
            semantic="currency_code" if field else None,
        )

    p = Plan(cases=[case(None, None), case("code", "in"), case("code", "out")])
    rows = [
        {"id": 0, "outcome": "handled", "calls": 0},
        {"id": 1, "outcome": "error", "calls": 0, "raised": "KeyError"},
        {"id": 2, "outcome": "refused", "calls": 0},
    ]
    v = run_plan(
        "env/dialogue_user:r",
        p,
        tree=tmp_path,
        python=Path("/x"),
        work=tmp_path / "w",
        runner=_rows_runner(rows),
    )
    assert not v.failures and any("code (KeyError)" in n for n in v.notes)
    # an out-of-domain case on a cover whose baseline errors is not judged
    rows2 = [
        {"id": 0, "outcome": "error", "calls": 0},
        {"id": 1, "outcome": "handled", "calls": 0},
        {"id": 2, "outcome": "handled", "calls": 0},
    ]
    v2 = run_plan(
        "env/dialogue_user:r",
        p,
        tree=tmp_path,
        python=Path("/x"),
        work=tmp_path / "w2",
        runner=_rows_runner(rows2),
    )
    assert not v2.failures
    # ... but it leaves a note that the out-of-domain side never ran
    assert any("code" in n and "out-of-domain check did not run" in n for n in v2.notes)


def test_numbers_and_dates_go_beyond_the_observed_range():
    from decimal import Decimal

    vals = [Decimal(4), Decimal(10)]
    assert beyond_number(vals, b"s") > 10
    assert beyond_number([Decimal(-3), Decimal(-7)], b"s") < -7  # sign kept
    # in a mixed-sign field each value keeps its own sign and goes past its own side's range
    mixed = [-50, 120, 30]
    assert perturb_value(-50, mixed, b"s", set()) < -50
    assert perturb_value(30, mixed, b"s", set()) > 120
    # a zero takes the side of the field's other values; alone it counts as non-negative
    assert perturb_value(0, [-50, 0, -20], b"s", set()) < -50
    nz = perturb_value("-0.00", ["-50.00", "-0.00"], b"s", set())
    assert nz.startswith("-") and float(nz) < -50
    assert perturb_value(0, [0], b"s", set()) > 0
    assert perturb_value(0, [0, 7, -3], b"s", set()) > 7
    neg = perturb_value("-50.00", ["-50.00", "120.00", "30.00"], b"s", set())
    assert neg.startswith("-") and len(neg.split(".")[1]) == 2 and float(neg) < -50
    assert perturb_value(5, [5, 10], b"s", set()) > 10
    assert isinstance(perturb_value(5, [5, 10], b"s", set()), int)
    f = perturb_value(2.5, [2.5, 0.5], b"s", set())
    assert isinstance(f, float) and f > 2.5
    p = perturb_value("12.50", ["12.50", "3.25"], b"s", set())
    assert p.count(".") == 1 and len(p.split(".")[1]) == 2 and float(p) > 12.5
    d = perturb_value("2026-09-30", ["2026-09-30", "2026-10-02"], b"s", set())
    assert d > "2026-10-02" and len(d) == 10
    import datetime

    datetime.date.fromisoformat(d)
    for skipped in (True, None, "", "true"):
        assert perturb_value(skipped, [skipped], b"s", set()) is None


def test_plan_is_deterministic_and_seeded_by_item_and_cover(recorded):
    store, acts = recorded
    covers = _covers(acts, 0, 1, 4, 5)
    a = plan("env/shop:search", covers, seen=acts, blob=store.get)
    b = plan("env/shop:search", covers, seen=acts, blob=store.get)
    assert [c.payload for c in a.cases] == [c.payload for c in b.cases]
    assert [c.file for c in a.cases] == [c.file for c in b.cases]
    other = plan("env/shop:find", covers, seen=acts, blob=store.get)
    changed = [c.payload for c in other.cases] != [c.payload for c in a.cases]
    assert changed or [c.file for c in other.cases] != [c.file for c in a.cases]


def test_tool_perturbs_recorded_keywords_and_keeps_the_recorded_call(recorded):
    store, acts = recorded
    p = plan("env/shop:search", _covers(acts, 0), seen=acts, blob=store.get)
    base = [c for c in p.cases if c.field is None]
    fields = {c.field: c for c in p.cases if c.field is not None}
    assert len(base) == 1 and set(fields) == {"colour", "limit"}
    assert base[0].payload["kwargs"] == {"colour": "red", "limit": 5}
    colour = fields["colour"].payload
    assert colour["action"]["kwargs"] == {"colour": "red", "limit": 5}  # the replay key
    assert colour["kwargs"]["limit"] == 5
    assert colour["kwargs"]["colour"] not in {"red", "blue", "mauve"}
    assert (
        fields["limit"].payload["kwargs"]["limit"] > 10
    )  # beyond every recorded limit
    assert fields["colour"].param == "colour"


def test_worktree_csv_perturbation_keeps_the_header_and_the_columns(recorded):
    store, acts = recorded
    p = plan(
        "env/worktree_workspace:read_stock",
        _covers(acts, 4),
        seen=acts,
        blob=store.get,
    )
    base = [c for c in p.cases if c.field is None]
    assert len(base) == 1
    assert base[0].file == STOCK
    original = list(csv.reader(io.StringIO(STOCK.decode())))
    fields = [c for c in p.cases if c.field is not None]
    assert {c.field for c in fields} == {"sku", "colour", "qty", "price", "restocked"}
    for c in fields:
        rows = list(csv.reader(io.StringIO(c.file.decode())))
        assert rows[0] == original[0]  # header unchanged
        assert [len(r) for r in rows] == [len(r) for r in original]
        diff = [
            (i, j)
            for i, (r, o) in enumerate(zip(rows, original))
            for j, (x, y) in enumerate(zip(r, o))
            if x != y
        ]
        assert len(diff) == 1 and diff[0][0] >= 1
        assert c.name.endswith(".csv")
    qty = next(c for c in fields if c.field == "qty")
    assert int(list(csv.reader(io.StringIO(qty.file.decode())))[1][2]) > 10


def test_a_perturbed_date_moves_the_other_dates_of_its_record_by_the_same_offset(
    recorded,
):
    import datetime

    store, acts = recorded
    p = plan(
        "env/worktree_workspace:read_tickets",
        _covers(acts, 7),
        seen=acts,
        blob=store.get,
    )
    original = list(csv.reader(io.StringIO(TICKETS.decode())))
    fields = {
        c.field: list(csv.reader(io.StringIO(c.file.decode())))
        for c in p.cases
        if c.field
    }
    rows = fields["opened"]
    changed = [
        (i, j)
        for i, (r, o) in enumerate(zip(rows, original))
        for j, (x, y) in enumerate(zip(r, o))
        if x != y
    ]
    assert {j for _, j in changed} == {2, 3} and len({i for i, _ in changed}) == 1
    (i,) = {i for i, _ in changed}
    d = datetime.date.fromisoformat
    assert d(rows[i][3]) - d(rows[i][2]) == d(original[i][3]) - d(original[i][2])
    assert d(rows[i][2]) > d("2026-09-03")  # still beyond the observed span
    # numbers and other strings still change one at a time
    for name in ("minutes", "ticket_id", "agent_id"):
        diff = [
            (i, j)
            for i, (r, o) in enumerate(zip(fields[name], original))
            for j, (x, y) in enumerate(zip(r, o))
            if x != y
        ]
        assert len(diff) == 1
    assert fields["agent_id"][1][1].startswith("A")


def test_dialogue_perturbs_json_leaves_and_keeps_structure(recorded):
    store, acts = recorded
    p = plan("env/dialogue_user:parse", _covers(acts, 5), seen=acts, blob=store.get)
    fields = {c.field: c.payload["observation"] for c in p.cases if c.field}
    # one observation from one episode is no support for constancy: every field varies
    assert set(fields) == {"room", "items[]", "steps"}
    for obs in fields.values():
        assert set(obs) == {"room", "items", "steps"} and len(obs["items"]) == 2
    assert fields["steps"]["steps"] > 3
    assert fields["room"]["room"] not in {"hall"}
    assert not any("not perturbed" in n for n in p.notes)


SUBMIT = [
    {"type": "SubmitFeedback", "version": 3, "attempt": n, "ok": n % 2 == 0}
    for n in range(1, 5)
]
DEMOS = [
    {"type": "DemosFeedback", "version": 3, "demos": n, "shown": [n, n + 1]}
    for n in range(1, 4)
]


def _arc_pool():
    """ARC-like feedback: Submit and Demos messages on one channel, across three episodes."""
    msgs = [_dl(i, m) for i, m in enumerate(SUBMIT + DEMOS)]
    eids = ["a1", "a2", "a3", "a1", "a2", "a3", "a1"]
    return msgs, list(zip(eids, msgs))


def test_a_tag_constant_within_its_message_shape_across_the_store_is_identity():
    """Submit and Demos messages share a channel; type is constant within each field-name set."""
    msgs, pool = _arc_pool()
    p = plan(
        "env/dialogue_user:parse_feedback",
        [("a1", 0, msgs[0]), ("a2", 1, msgs[1])],
        seen=msgs,
        blob=lambda sha: b"",
        pool=pool,
    )
    assert {c.field for c in p.cases if c.field} == {"attempt"}  # ok: a boolean
    (note,) = [n for n in p.notes if "not perturbed" in n]
    assert "type" in note and "version" in note
    assert (
        "SubmitFeedback" not in note and "3" not in note
    )  # names fields, never values
    d = plan(
        "env/dialogue_user:parse_demos",
        [("a1", 4, msgs[4])],
        seen=msgs,
        blob=lambda sha: b"",
        pool=pool,
    )
    assert {c.field for c in d.cases if c.field} == {"demos", "shown[]"}
    # without the field-name split, type would hold two values: both shapes' types are kept here
    assert all(c.field != "type" for c in p.cases + d.cases)


def test_constancy_needs_support_across_episodes_and_resists_chosen_covers():
    def room(i, r):  # each observation distinct in content (turn), so each counts
        return _dl(i, {"room": r, "steps": 3, "turn": i})

    covers = [("b1", 0, room(0, "kitchen")), ("b2", 1, room(1, "kitchen"))]
    # the covers agree, but a same-shape observation elsewhere in the store differs: room varies
    wider = [(e, a) for e, _, a in covers] + [("b3", room(2, "hall"))]
    p = plan(
        "env/dialogue_user:r",
        covers,
        seen=[],
        blob=lambda sha: b"",
        pool=wider,
    )
    fields = {c.field for c in p.cases if c.field}
    assert (
        "room" in fields and "steps" not in fields
    )  # steps: 3 observations, 3 episodes, one value
    # three observations from one episode: no support, every field varies
    one_ep = [("b1", room(i, "kitchen")) for i in range(3)]
    q = plan(
        "env/dialogue_user:r",
        covers[:1],
        seen=[],
        blob=lambda sha: b"",
        pool=one_ep,
    )
    assert {c.field for c in q.cases if c.field} == {"room", "steps", "turn"}
    # two observations from two episodes: below the minimum, every field varies
    two = [(e, a) for e, _, a in covers]
    r = plan("env/dialogue_user:r", covers, seen=[], blob=lambda sha: b"", pool=two)
    assert {c.field for c in r.cases if c.field} == {"room", "steps", "turn"}
    # the default pool is the covers (and seen actions, here none)
    default = plan("env/dialogue_user:r", covers, seen=[], blob=lambda sha: b"")
    assert {c.field for c in default.cases if c.field} == {"room", "steps", "turn"}
    # the same observation in three episodes is one observation: no support
    same = [(e, room(0, "kitchen")) for e in ("b1", "b2", "b3")]
    t = plan(
        "env/dialogue_user:r",
        covers[:1],
        seen=[],
        blob=lambda sha: b"",
        pool=same,
    )
    assert {c.field for c in t.cases if c.field} == {"room", "steps", "turn"}


def test_the_same_file_read_in_two_episodes_is_one_observation(recorded):
    """N2: a shared expenses file read three times in two episodes does not make currency constant."""
    store, _ = recorded
    sha = store.put(_expenses([5, 7]))
    reads = [
        (eid, _wt_read(i, f"exp/2026/e{i}.csv", sha, 1))
        for i, eid in enumerate(["d1", "d2", "d2"])
    ]
    covers = [(reads[0][0], 0, reads[0][1])]
    p = plan("env/worktree_workspace:r", covers, seen=[], blob=store.get, pool=reads)
    assert "currency" in {c.field for c in p.cases}
    assert not any("not perturbed" in n for n in p.notes)
    # one more, different file: two distinct observations, still below the minimum of three
    other = store.put(_expenses([9, 11]))
    two = reads + [("d2", _wt_read(3, "exp/2026/e3.csv", other, 1))]
    q = plan("env/worktree_workspace:r", covers, seen=[], blob=store.get, pool=two)
    assert "currency" in {c.field for c in q.cases}


def _expenses(rows):
    return b"id,amount,currency\n" + b"".join(
        b"X-%d,%d.00,USD\n" % (i, amount) for i, amount in enumerate(rows)
    )


def test_a_capped_pool_keeps_no_field_constant(recorded, monkeypatch):
    """N1: an unread observation could vary a field, so a cap makes constancy fail safe."""
    import unify.memory_v2.held_out as held_out

    store, _ = recorded
    msgs, pool = _arc_pool()
    capped = plan(
        "env/dialogue_user:parse_feedback",
        [("a1", 0, msgs[0])],
        seen=msgs,
        blob=store.get,
        pool=pool,
        pool_capped=True,
    )
    assert {"type", "version", "attempt"} <= {c.field for c in capped.cases}
    assert any("stopped at its read cap" in n for n in capped.notes)
    # three distinct expense files from two episodes: currency is USD in all of them
    reads = [
        (eid, _wt_read(i, f"exp/2026/e{i}.csv", store.put(_expenses(rows)), 1))
        for i, (eid, rows) in enumerate(
            [("c1", [5, 7]), ("c2", [9, 11]), ("c2", [13, 15])],
        )
    ]
    covers = [(reads[0][0], 0, reads[0][1])]
    kept = plan("env/worktree_workspace:r", covers, seen=[], blob=store.get, pool=reads)
    assert "currency" not in {c.field for c in kept.cases}
    assert any("currency" in n and "not perturbed" in n for n in kept.notes)
    # the file parse cap leaves a file of the family unread: currency is perturbed again
    monkeypatch.setattr(held_out, "MAX_HOST_PARSES", 2)
    cut = plan("env/worktree_workspace:r", covers, seen=[], blob=store.get, pool=reads)
    assert "currency" in {c.field for c in cut.cases}
    assert any("parsed its cap of 2 recorded files" in n for n in cut.notes)


def test_the_gate_reads_the_stores_episodes_first_in_store_order(tmp_path, monkeypatch):
    """N1: the manifest's episodes cannot use up the read cap before the store's are read."""
    from types import SimpleNamespace

    import unify.memory_v2.held_out as held_out

    ev = EvidenceStore(tmp_path / "e.sqlite")
    recorded_eps = {
        eid: [_dl(0, {"room": eid}), _dl(1, {"room": eid + "x"})]
        for eid in ("s2", "s1", "n1")
    }
    for eid in ("s1", "s2"):  # store order: s1 before s2
        ev.index_episode(_ep(episode_id=eid, actions=recorded_eps[eid]), "1" * 40)
    lookup = lambda eid, i: (
        recorded_eps[eid][i]
        if eid in recorded_eps and i < len(recorded_eps[eid])
        else None
    )
    gate = Gate(
        Repo.init_bare(tmp_path / "m.git"),
        ev,
        BlobStore(tmp_path / "b"),
        action_lookup=lookup,
    )
    named = SimpleNamespace(source_episodes=["n1"], covers=[("n1", 0)])
    run = SimpleNamespace(pools={}, man=SimpleNamespace(items=[named]))
    cover = [("n1", 0, recorded_eps["n1"][0])]
    pool, capped = gate._pool(run, cover)
    assert [e for e, _ in pool] == ["s1", "s1", "s2", "s2", "n1", "n1"] and not capped
    monkeypatch.setattr(held_out, "MAX_POOL_ACTIONS", 3)
    pool, capped = gate._pool(SimpleNamespace(pools={}, man=run.man), cover)
    assert [e for e, _ in pool] == ["s1", "s1"] and capped


def test_a_tool_keyword_is_constant_only_across_every_recorded_call_in_the_pool(
    recorded,
):
    store, acts = recorded
    # an enumeration of prefixes (three distinct ticket ids) still varies
    t = plan("env/worktree_workspace:r", _covers(acts, 7), seen=acts, blob=store.get)
    assert "ticket_id" in {c.field for c in t.cases}
    calls = [_search("red", 5), _search("blue", 5), _search("green", 5)]
    pool = list(zip(["k1", "k2", "k3"], calls))
    k = plan(
        "env/shop:search",
        [("k1", 0, calls[0])],
        seen=calls[:1],
        blob=store.get,
        pool=pool,
    )
    assert {c.field for c in k.cases if c.field} == {"colour"}
    assert any("limit" in n and "not perturbed" in n for n in k.notes)
    # one more recorded call elsewhere with another limit: limit varies, and its range spans that call
    pool2 = pool + [("k4", _search("red", 50))]
    k2 = plan(
        "env/shop:search",
        [("k1", 0, calls[0])],
        seen=calls[:1],
        blob=store.get,
        pool=pool2,
    )
    lim = next(c for c in k2.cases if c.field == "limit")
    assert lim.payload["kwargs"]["limit"] > 50


def test_the_declared_input_decides_the_first_argument_form(recorded):
    """path: a mounted file; text: the decoded text; bytes: the mounted file, read as bytes."""
    store, acts = recorded
    item = "env/worktree_workspace:read_stock"
    by = {
        kind: plan(item, _covers(acts, 4), seen=acts, blob=store.get, input_kind=kind)
        for kind in (None, "path", "text", "bytes")
    }
    legacy = [c for c in by[None].cases if c.field is None][0]
    assert legacy.payload["form"] == "path" and legacy.file == STOCK
    assert [c.payload for c in by["path"].cases] == [c.payload for c in by[None].cases]
    text = [c for c in by["text"].cases if c.field is None][0]
    assert text.payload["form"] == "text" and text.file is None
    assert text.payload["observation"] == STOCK.decode()
    assert all(isinstance(c.payload["observation"], str) for c in by["text"].cases)
    raw = [c for c in by["bytes"].cases if c.field is None][0]
    assert raw.payload["form"] == "bytes" and raw.file == STOCK
    assert raw.payload["path"].startswith("/cases/files/")
    # the same fields are perturbed whatever the form
    assert {c.field for c in by["text"].cases} == {c.field for c in by["path"].cases}
    # a form a cover's kind cannot give fails the item (C1)
    env = plan(item, _covers(acts, 4), seen=acts, blob=store.get, input_kind="env")
    assert env.cases == []
    assert env.unfit == ["declares input env, which a worktree cover cannot give"]
    # a dialogue observation declared as an observation is passed as before
    d = plan(
        "env/dialogue_user:p",
        _covers(acts, 24, 25),
        seen=acts,
        blob=store.get,
        input_kind="observation",
    )
    legacy_d = plan(
        "env/dialogue_user:p",
        _covers(acts, 24, 25),
        seen=acts,
        blob=store.get,
    )
    assert d.cases and all(c.payload["form"] == "observation" for c in d.cases)
    assert [c.payload for c in d.cases] == [c.payload for c in legacy_d.cases]


def test_a_tool_item_declared_on_observations_gets_the_response_perturbed(recorded):
    store, _ = recorded
    calls = [
        _search(
            "red",
            5,
            response={"items": [{"sku": "A-1", "qty": 4}], "kind": "page"},
        ),
        _search(
            "blue",
            10,
            response={"items": [{"sku": "B-2", "qty": 9}], "kind": "page"},
        ),
    ]
    third = _search(
        "green",
        5,
        response={"items": [{"sku": "C-3", "qty": 1}], "kind": "page"},
    )
    p = plan(
        "env/shop:parse_page",
        [("h1", i, a) for i, a in enumerate(calls)],
        seen=calls,
        blob=store.get,
        input_kind="observation",
        pool=[("h1", calls[0]), ("h1", calls[1]), ("h2", third)],
    )
    assert {c.field for c in p.cases if c.field} == {"items[].sku", "items[].qty"}
    for c in p.cases:
        assert c.payload["form"] == "observation" and c.param is None
        assert c.payload["kwargs"] in (
            {"colour": "red", "limit": 5},
            {"colour": "blue", "limit": 10},
        )
    sku = next(c for c in p.cases if c.field == "items[].sku" and c.cover == ("h1", 0))
    assert sku.payload["observation"]["items"][0]["sku"] not in ("A-1", "B-2")
    assert (
        sku.payload["observation"]["kind"] == "page"
    )  # constant in 3 responses, 2 episodes
    # a rejection of the call exempts keywords, never response fields
    assert all(c.family[0] == "tool_response" for c in p.cases)


def test_a_declared_form_a_cover_cannot_give_fails_the_item(recorded):
    """C1: a declared form that no covered input can be given in would check nothing."""
    store, acts = recorded

    def unfit(cover_idx, kind, covers=None):
        return plan(
            "env/x:f",
            covers or _covers(acts, *cover_idx),
            seen=acts,
            blob=store.get,
            input_kind=kind,
        ).unfit

    assert unfit([5], "path") == [
        "declares input path, which a dialogue cover cannot give",
    ]
    assert unfit([4], "env") == [
        "declares input env, which a worktree cover cannot give",
    ]
    assert unfit([0], "bytes") == [
        "declares input bytes, which a tool cover cannot give",
    ]
    assert unfit([5], "text") == [
        "declares input text, which a dialogue cover with a non-text observation cannot give",
    ]
    plain = _search("red", 5, response="3 results")
    assert unfit(None, "observation", [("h1", 0, plain)]) == [
        "declares input observation, which a tool cover without a JSON response cannot give",
    ]
    # forms the covers can give pass; a plain-text observation gives nothing to perturb: a note
    assert (
        unfit([4], "text") == []
        and unfit([0], "env") == []
        and unfit([5], "observation") == []
    )
    text_obs = _dl(0, "Inventory: wood 1")
    p = plan(
        "env/x:f",
        [("h1", 0, text_obs)],
        seen=[],
        blob=store.get,
        input_kind="observation",
    )
    assert p.unfit == [] and p.cases == []
    assert any("give nothing to perturb as observation" in n for n in p.notes)
    # a failing form is reported by run_plan as a failure, before anything runs
    v = run_plan(
        "env/x:f",
        plan("env/x:f", _covers(acts, 5), seen=acts, blob=store.get, input_kind="path"),
        tree=Path("/nonexistent"),
        python=Path("/x"),
        work=Path("/nonexistent/w"),
        runner=_rows_runner([]),
    )
    assert v.failures == ["declares input path, which a dialogue cover cannot give"]


def test_shell_covers_are_skipped_with_a_note(recorded):
    store, acts = recorded
    p = plan("env/shell_make:parse", _covers(acts, 3), seen=acts, blob=store.get)
    assert p.cases == []
    assert any("shell" in n and "not supported yet" in n for n in p.notes)


def test_a_rejection_exempts_only_the_fields_it_varied(recorded):
    store, acts = recorded
    fam = ("tool", "shop", "search")
    # colour=mauve was never accepted; limit=5 was: only colour is exempt
    p = plan("env/shop:search", _covers(acts, 0, 2), seen=acts, blob=store.get)
    assert {c.cover for c in p.cases} == {("h1", 0)}
    assert p.exempt == {fam: {"colour"}}
    # an auth failure whose keywords equal an accepted call's exempts nothing, and says so
    auth = plan("env/shop:search", _covers(acts, 0, 8), seen=acts, blob=store.get)
    assert not auth.exempt.get(fam)
    assert any("varies no field" in n and "exempts nothing" in n for n in auth.notes)
    # a rejection of another method exempts nothing on this one
    other = plan("env/shop:search", _covers(acts, 0, 9), seen=acts, blob=store.get)
    assert not other.exempt.get(fam) and not other.exempt.get(("tool", "shop", "buy"))
    assert any("cannot be compared" in n for n in other.notes)
    # a rejected write without a recorded file cannot be compared: nothing exempt, noted
    wt = plan(
        "env/worktree_workspace:r",
        _covers(acts, 4, 6),
        seen=acts,
        blob=store.get,
    )
    assert not any(wt.exempt.values())
    assert any("does not record a changed file" in n for n in wt.notes)
    # a refused write that records the unchanged file exempts nothing: not sku, colour or qty
    same = plan(
        "env/worktree_workspace:r",
        _covers(acts, 4, 12),
        seen=acts,
        blob=store.get,
    )
    assert not any(same.exempt.values())
    assert any("does not record a changed file" in n for n in same.notes)
    # a refused write exempts exactly the field its own before and after differ in
    changed = plan(
        "env/worktree_workspace:r",
        _covers(acts, 4, 13),
        seen=acts,
        blob=store.get,
    )
    assert changed.exempt == {
        ("worktree", "worktree:workspace", "store/#/*.csv"): {"colour"},
    }


def test_covers_per_item_are_bounded(recorded):
    store, acts = recorded
    many = [(f"h{i}", 0, acts[0]) for i in range(MAX_COVERS_PER_ITEM + 5)]
    p = plan("env/shop:search", many, seen=acts, blob=store.get)
    assert len({c.cover for c in p.cases}) == MAX_COVERS_PER_ITEM


def test_a_recorded_rejection_with_its_error_is_an_admissible_cover(recorded):
    store, acts = recorded
    assert is_rejection(acts[2]) and is_rejection(acts[6])
    assert not is_rejection(acts[0])
    assert cover_problem(acts[2], "shop", store.has) is None
    assert cover_problem(acts[6], "worktree_workspace", store.has) is None
    silent = Action(0, "shop", "search", [], {}, None, "error")
    assert cover_problem(silent, "shop", store.has) is not None


# --- through the gate ------------------------------------------------------------------------------------

SKELETON = '''__all__ = ["{name}"]


class MemoryInputError(ValueError):
    """The input differs in shape from what this function was built from."""


def {name}({params}):
    """{doc}

    Effect: read
    """
{body}
'''
WHITELIST = """    if colour not in ("red", "blue"):
        raise MemoryInputError("colour must be one of the recorded colours")
    return apis.shop.search(colour=colour, limit=limit)["items"]"""
SHAPE_ONLY = """    if not isinstance(colour, str) or not colour:
        raise MemoryInputError("colour must be a non-empty string")
    if not isinstance(limit, int):
        raise MemoryInputError("limit must be an integer")
    return apis.shop.search(colour=colour, limit=limit)["items"]"""
CALL_FIRST = """    try:
        return apis.shop.search(colour=colour, limit=limit)["items"]
    except LookupError as exc:
        raise MemoryInputError("the environment has no such call") from exc"""
UNCAUGHT = """    return apis.shop.search(colour=colour, limit=limit)["items"]"""

STOCK_WHITELIST = """    rows = list(csv.DictReader(open(path, newline="")))
    for r in rows:
        if r["colour"] not in {"red", "blue"}:
            raise MemoryInputError("unknown colour")
    return rows"""
STOCK_RANGE = """    rows = list(csv.DictReader(open(path, newline="")))
    for r in rows:
        if int(r["qty"]) > 10:
            raise MemoryInputError("qty out of range")
    return rows"""
STOCK_SHAPE = """    import datetime
    from decimal import Decimal, InvalidOperation
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ["sku", "colour", "qty", "price", "restocked"]:
            raise MemoryInputError("unexpected columns")
        rows = list(reader)
    for r in rows:
        try:
            int(r["qty"]); Decimal(r["price"]); datetime.date.fromisoformat(r["restocked"])
        except (ValueError, InvalidOperation) as exc:
            raise MemoryInputError("a cell is not of its column's type") from exc
    return rows"""


def _module(name, params, body, doc="Search the shop by colour.", imports=""):
    return imports + SKELETON.format(name=name, params=params, doc=doc, body=body)


def _green(target, *, python, ro, rw, cwd, timeout_s=300.0, env=None):
    return PytestOutcome(passed={"t::a"}, returncode=0)


@pytest.fixture
def shop(tmp_path, recorded):
    store, acts = recorded
    mem = Repo.init_bare(tmp_path / "mem.git")
    ev = EvidenceStore(tmp_path / "e.sqlite")
    ev.index_episode(_ep(episode_id="h1"), "1" * 40)
    lookup = lambda eid, i: acts[i] if eid == "h1" and 0 <= i < len(acts) else None
    gate = Gate(mem, ev, store, action_lookup=lookup, pytest_runner=_green)
    return mem, gate


def _default_input(channel):
    for prefix, kind in (
        ("worktree_", "path"),
        ("dialogue_", "observation"),
        ("shell_", "text"),
    ):
        if channel.startswith(prefix):
            return kind
    return "env"


def _check(mem, gate, channel, module, covers, fn, field_types=None, input_kind=None):
    """Commit *module* and gate it; the item declares *input_kind* in its manifest and docstring."""
    input_kind = input_kind or _default_input(channel)
    if "    Input: " not in module:
        module = module.replace(
            "    Effect: read\n",
            f"    Effect: read\n    Input: {input_kind}\n",
            1,
        )
    parent = mem.head()
    test = f"env/{channel}/tests/test_{fn}.py"
    with mem.temp_checkout("main") as wt:
        for rel, text in {
            f"env/{channel}/__init__.py": module,
            test: f"from env.{channel} import {fn}\n\ndef test_it():\n    assert {fn}\n",
        }.items():
            (wt / rel).parent.mkdir(parents=True, exist_ok=True)
            (wt / rel).write_text(text)
        cand = mem.commit_all(wt, "pass", {"Pass": "p1"})
    man = {
        "items": [
            {
                "item": f"env/{channel}:{fn}",
                "kind": "env_function",
                "source_episodes": ["h1"],
                "tests": [test],
                "covers": [["h1", i] for i in covers],
                "input": input_kind,
                **({"field_types": field_types} if field_types else {}),
            },
        ],
        "skeleton": [f"env/{channel}"],
    }
    return gate.check(parent, cand, man)


@needs_bwrap
def test_an_item_with_a_seen_value_whitelist_fails_g2(shop):
    mem, gate = shop
    res = _check(
        mem,
        gate,
        "shop",
        _module("search", "apis, colour, limit=20", WHITELIST),
        [0, 1],
        "search",
    )
    assert not res.checks["G2"]
    bad = [r for r in res.reasons if "held-out value refused" in r]
    assert bad == ["G2: env/shop:search held-out value refused: colour"], res.reasons
    # the reason names the field, never a value
    assert not any(v in r for r in bad for v in ("red", "blue", "mauve"))


@needs_bwrap
def test_the_same_item_with_a_shape_only_check_passes_g2(shop):
    mem, gate = shop
    res = _check(
        mem,
        gate,
        "shop",
        _module("search", "apis, colour, limit=20", SHAPE_ONLY),
        [0, 1],
        "search",
    )
    assert res.checks["G2"], res.reasons


@needs_bwrap
def test_an_item_covering_an_environment_rejection_may_refuse(shop):
    mem, gate = shop
    res = _check(
        mem,
        gate,
        "shop",
        _module("search", "apis, colour, limit=20", WHITELIST),
        [0, 1, 2],
        "search",
    )
    assert res.checks["G2"], res.reasons
    assert any(
        r.startswith("note:")
        and "refusal of colour allowed by a covered recorded rejection" in r
        for r in res.reasons
    ), res.reasons
    only = _check(
        mem,
        gate,
        "shop",
        _module("search", "apis, colour, limit=20", WHITELIST),
        [2],
        "search",
    )
    assert not only.checks["G2"]
    assert any("only recorded rejections" in r for r in only.reasons)


@needs_bwrap
@pytest.mark.parametrize(
    "body",
    [CALL_FIRST, UNCAUGHT],
    ids=["refuses-after-miss", "miss-propagates"],
)
def test_a_tool_replay_miss_is_not_a_violation(shop, body):
    mem, gate = shop
    res = _check(
        mem,
        gate,
        "shop",
        _module("search", "apis, colour, limit=20", body),
        [0, 1],
        "search",
    )
    assert res.checks["G2"], res.reasons


@needs_bwrap
@pytest.mark.parametrize(
    "body, field",
    [(STOCK_WHITELIST, "colour"), (STOCK_RANGE, "qty"), (STOCK_SHAPE, None)],
    ids=["whitelist", "observed-range", "shape-only"],
)
def test_worktree_readers_are_checked_on_perturbed_files(shop, body, field):
    mem, gate = shop
    mod = _module(
        "read_stock",
        "path",
        body,
        doc="Read a stock CSV.",
        imports="import csv\n\n",
    )
    res = _check(mem, gate, "worktree_workspace", mod, [4], "read_stock")
    refused = [r for r in res.reasons if "held-out value refused" in r]
    if field is None:
        assert res.checks["G2"] and not refused, res.reasons
    else:
        assert refused == [
            f"G2: env/worktree_workspace:read_stock held-out value refused: {field}",
        ], res.reasons
    # a rejected write without a recorded file cannot show which field it varied: no exemption, a note
    if field is not None:
        res2 = _check(mem, gate, "worktree_workspace", mod, [4, 6], "read_stock")
        assert not res2.checks["G2"], res2.reasons
        assert any("does not record a changed file" in r for r in res2.reasons)


@needs_bwrap
def test_a_shell_item_is_skipped_with_a_note(shop):
    mem, gate = shop
    body = """    if text.strip() not in ("3 checks passed",):
        raise MemoryInputError("unseen output")
    return int(text.split()[0])"""
    res = _check(
        mem,
        gate,
        "shell_make",
        _module("parse", "text", body, doc="Parse the check count."),
        [3],
        "parse",
    )
    assert res.checks["G2"], res.reasons
    assert any(
        r.startswith("note:") and "shell" in r and "not supported yet" in r
        for r in res.reasons
    )


TICKET_PREFIX = """    for r in csv.DictReader(open(path, newline="")):
        if not r["ticket_id"].startswith(("NET-", "DSK-", "APP-")):
            raise MemoryInputError("unknown ticket family")
    return True"""
AGENT_FORMAT = """    for r in csv.DictReader(open(path, newline="")):
        a = r["agent_id"]
        if not (a.startswith("A") and a[1:].isdigit()):
            raise MemoryInputError("agent ids are A followed by digits")
    return True"""
DATE_ORDER = """    import datetime
    for r in csv.DictReader(open(path, newline="")):
        if datetime.date.fromisoformat(r["opened"]) > datetime.date.fromisoformat(r["closed"]):
            raise MemoryInputError("a ticket closes before it opens")
    return True"""


@needs_bwrap
@pytest.mark.parametrize(
    "body, field",
    [(TICKET_PREFIX, "ticket_id"), (AGENT_FORMAT, None), (DATE_ORDER, None)],
    ids=["prefix-enumeration-flagged", "constant-prefix-format", "date-order-in-row"],
)
def test_literal_formats_and_record_relations_are_not_flagged(shop, body, field):
    mem, gate = shop
    mod = _module(
        "read_tickets",
        "path",
        body,
        doc="Read a tickets CSV.",
        imports="import csv\n\n",
    )
    res = _check(mem, gate, "worktree_workspace", mod, [7], "read_tickets")
    refused = [r for r in res.reasons if "held-out value refused" in r]
    if field is None:
        assert res.checks["G2"] and not refused, res.reasons
    else:
        assert refused == [
            f"G2: env/worktree_workspace:read_tickets held-out value refused: {field}",
        ], res.reasons


LEDGER_SIGNS = """    from decimal import Decimal
    for r in csv.DictReader(open(path, newline="")):
        amount = Decimal(r["amount"])
        if r["kind"] == "credit" and amount >= 0:
            raise MemoryInputError("credits are negative")
        if r["kind"] == "debit" and amount <= 0:
            raise MemoryInputError("debits are positive")
    return True"""


@needs_bwrap
def test_a_sign_convention_in_a_mixed_sign_column_is_not_flagged(shop):
    mem, gate = shop
    mod = _module(
        "read_ledger",
        "path",
        LEDGER_SIGNS,
        doc="Read a ledger CSV.",
        imports="import csv\n\n",
    )
    res = _check(mem, gate, "worktree_workspace", mod, [11], "read_ledger")
    assert res.checks["G2"], res.reasons
    assert not [r for r in res.reasons if "held-out value refused" in r]


@needs_bwrap
@pytest.mark.parametrize("cover", [8, 9], ids=["auth-failure", "other-method"])
def test_an_unrelated_rejection_exempts_nothing(shop, cover):
    mem, gate = shop
    res = _check(
        mem,
        gate,
        "shop",
        _module("search", "apis, colour, limit=20", WHITELIST),
        [0, 1, cover],
        "search",
    )
    assert [r for r in res.reasons if "held-out value refused" in r] == [
        "G2: env/shop:search held-out value refused: colour",
    ], res.reasons
    assert any("exempts nothing" in r for r in res.reasons)


KEYED = """    APPS = {"red": 1, "blue": 2}
    APPS[colour]
    return apis.shop.search(colour=colour, limit=limit)["items"]"""
RENAMED = """    if shade not in ("red", "blue"):
        raise MemoryInputError("unknown shade")
    return apis.shop.search(colour=shade, limit=limit)["items"]"""
STALLS = """    import time
    try:
        time.sleep(30)
    except BaseException:
        return []"""
REFUSES_ALL = """    raise MemoryInputError("never")"""


@needs_bwrap
@pytest.mark.parametrize(
    "params, body, note",
    [
        ("apis, colour, limit=20", KEYED, "colour (KeyError)"),
        (
            "apis, shade, limit=20",
            RENAMED,
            "not taken under the same name, not checked: colour",
        ),
        ("apis, colour, limit=20", STALLS, "limit and were stopped"),
        ("apis, colour, limit=20", REFUSES_ALL, "refuses 1 of its own covered inputs"),
    ],
    ids=[
        "keyerror-whitelist",
        "renamed-parameter",
        "swallowed-timeout",
        "refused-baseline",
    ],
)
def test_unjudged_outcomes_are_notes_not_failures(shop, params, body, note):
    mem, gate = shop
    res = _check(mem, gate, "shop", _module("search", params, body), [0], "search")
    assert res.checks["G2"], res.reasons
    assert any(r.startswith("note:") and note in r for r in res.reasons), res.reasons


@needs_bwrap
def test_a_dialogue_item_is_checked_and_reasons_are_capped_at_five(shop):
    mem, gate = shop
    body = f"""    if obs != {SEVEN!r}:
        raise MemoryInputError("not the recorded observation")
    return obs"""
    res = _check(
        mem,
        gate,
        "dialogue_user",
        _module("parse", "obs", body, doc="Parse the counts."),
        [10],
        "parse",
    )
    refused = [r for r in res.reasons if "held-out value refused" in r]
    assert len(refused) == 5 and not res.checks["G2"], res.reasons
    assert all(
        r.startswith("G2: env/dialogue_user:parse held-out value refused: f")
        for r in refused
    )


@needs_bwrap
@pytest.mark.parametrize(
    "old_ok_covers",
    [False, True],
    ids=["new-ok-covers", "only-rejection-is-new"],
)
def test_g5_does_not_count_a_rejection_cover_as_new_coverage(shop, old_ok_covers):
    """Controller ruling: library growth must cover new successful recorded observations."""
    mem, gate = shop
    if old_ok_covers:
        for idx in (0, 1):
            gate.ev.add_cover("env/shop:search", "h1", idx)
    res = _check(
        mem,
        gate,
        "shop",
        _module("search", "apis, colour, limit=20", SHAPE_ONLY),
        [0, 1, 2],
        "search",
    )
    assert res.checks["G2"], res.reasons
    assert res.checks["G5"] is (not old_ok_covers), res.reasons


@needs_bwrap
@pytest.mark.parametrize(
    "body, field",
    [(STOCK_WHITELIST, "colour"), (STOCK_RANGE, "qty")],
    ids=["colour", "qty"],
)
def test_a_refused_write_of_the_unchanged_file_exempts_nothing(shop, body, field):
    """The reviewer's probe: a permission-denied write recording the file as it was varies no field."""
    mem, gate = shop
    mod = _module(
        "read_stock",
        "path",
        body,
        doc="Read a stock CSV.",
        imports="import csv\n\n",
    )
    res = _check(mem, gate, "worktree_workspace", mod, [4, 12], "read_stock")
    assert [r for r in res.reasons if "held-out value refused" in r] == [
        f"G2: env/worktree_workspace:read_stock held-out value refused: {field}",
    ], res.reasons
    assert any("does not record a changed file" in r for r in res.reasons)
    # the write that changed the colour exempts the colour only
    res2 = _check(mem, gate, "worktree_workspace", mod, [4, 13], "read_stock")
    if field == "colour":
        assert res2.checks["G2"], res2.reasons
        assert any("refusal of colour allowed" in r for r in res2.reasons)
    else:
        assert not res2.checks["G2"], res2.reasons


PERIOD_HEAD = """    from decimal import Decimal
    rows = list(csv.DictReader(open(path, newline="")))
    for r in rows:
"""
PERIOD_OK = PERIOD_HEAD + """        if not 1 <= int(r["month"]) <= 12:
            raise MemoryInputError("month out of its domain")
        if not 0 <= Decimal(r["share"]) <= 1:
            raise MemoryInputError("share is a probability")
    return rows"""
PERIOD_SEEN = PERIOD_HEAD + """        if int(r["month"]) not in (9, 10):
            raise MemoryInputError("month not one of the recorded months")
        if not 0 <= Decimal(r["share"]) <= 1:
            raise MemoryInputError("share is a probability")
    return rows"""
PERIOD_LOOSE = PERIOD_HEAD + """        int(r["month"]); Decimal(r["share"])
    return rows"""
PERIOD_HOURS = PERIOD_OK.replace(
    "    return rows",
    """        if Decimal(r["hours"]) > 120:
            raise MemoryInputError("hours above the recorded maximum")
    return rows""",
)
_ITEM = "G2: env/worktree_workspace:read_periods"


@needs_bwrap
@pytest.mark.parametrize(
    "body, reasons",
    [
        (PERIOD_OK, []),
        (PERIOD_SEEN, [f"{_ITEM} declared month field month refuses in-domain values"]),
        (
            PERIOD_LOOSE,
            [
                f"{_ITEM} declared month field month does not refuse out-of-domain values",
                f"{_ITEM} declared probability field share does not refuse out-of-domain values",
            ],
        ),
        (PERIOD_HOURS, [f"{_ITEM} held-out value refused: hours"]),
    ],
    ids=[
        "domain-checks",
        "seen-month-whitelist",
        "no-domain-check",
        "untagged-hours-range",
    ],
)
def test_declared_types_are_checked_on_both_sides(shop, body, reasons):
    mem, gate = shop
    mod = _module(
        "read_periods",
        "path",
        body,
        doc="Read a periods CSV.",
        imports="import csv\n\n",
    )
    res = _check(
        mem,
        gate,
        "worktree_workspace",
        mod,
        [17],
        "read_periods",
        field_types={"month": "month", "share": "probability"},
    )
    assert sorted(r for r in res.reasons if r.startswith("G2:")) == sorted(
        reasons,
    ), res.reasons
    assert res.checks["G2"] is (not reasons)


LIMIT_CHECKED = """    if not isinstance(limit, int) or limit < 0:
        raise MemoryInputError("limit is a non-negative count")
    return apis.shop.search(colour=colour, limit=limit)["items"]"""
MONTHS_OK = """    for r in csv.DictReader(open(path, newline="")):
        if not 1 <= int(r["month"]) <= 12:
            raise MemoryInputError("month out of its domain")
    return True"""


@needs_bwrap
@pytest.mark.parametrize(
    "body, reasons",
    [
        (LIMIT_CHECKED, []),
        (
            UNCAUGHT,
            [
                "G2: env/shop:search declared nonneg_count field limit does not refuse out-of-domain values",
            ],
        ),
    ],
    ids=["checked", "unchecked"],
)
def test_a_declared_unbounded_tool_keyword_through_the_gate(shop, body, reasons):
    mem, gate = shop
    res = _check(
        mem,
        gate,
        "shop",
        _module("search", "apis, colour, limit=20", body),
        [0, 1],
        "search",
        field_types={"limit": "nonneg_count"},
    )
    assert sorted(r for r in res.reasons if r.startswith("G2:")) == reasons, res.reasons


@needs_bwrap
def test_a_correct_item_over_a_fully_observed_domain_passes(shop):
    mem, gate = shop
    mod = _module(
        "read_months",
        "path",
        MONTHS_OK,
        doc="Read monthly sales.",
        imports="import csv\n\n",
    )
    res = _check(
        mem,
        gate,
        "worktree_workspace",
        mod,
        [18],
        "read_months",
        field_types={"month": "month"},
    )
    assert res.checks["G2"], res.reasons
    assert any("every in-domain value is already recorded" in r for r in res.reasons)


RANGES_READER = """    for r in csv.DictReader(open(path, newline="")):
        float(r["amount"]); int(r["pct"])
    return True"""


@needs_bwrap
def test_a_declaration_contradicted_by_a_later_recorded_value_fails_g2(shop):
    mem, gate = shop
    mod = _module(
        "read_ranges",
        "path",
        RANGES_READER,
        doc="Read ranges.",
        imports="import csv\n\n",
    )
    res = _check(
        mem,
        gate,
        "worktree_workspace",
        mod,
        [22],
        "read_ranges",
        field_types={"amount": "nonneg_money", "pct": "percentage"},
    )
    item = "G2: env/worktree_workspace:read_ranges"
    assert sorted(r for r in res.reasons if r.startswith("G2:")) == [
        f"{item} declared nonneg_money field amount does not match its recorded values",
        f"{item} declared percentage field pct does not match its recorded values",
    ], res.reasons


# --- fix 13: constant fields are identity; items declare their input --------------------------------------

TAG_CHECK = """    if not isinstance(obs, dict) or obs.get("type") != "SubmitFeedback":
        raise MemoryInputError("not a submit feedback observation")
    return obs["attempt"]"""
ATTEMPT_WHITELIST = """    if obs.get("type") != "SubmitFeedback" or obs["attempt"] not in (1, 2):
        raise MemoryInputError("not a recorded attempt")
    return obs["attempt"]"""


@needs_bwrap
@pytest.mark.parametrize(
    "body, refused",
    [(TAG_CHECK, []), (ATTEMPT_WHITELIST, ["attempt"])],
    ids=["constant-tag-by-equality", "whitelist-over-two-values"],
)
def test_a_constant_tag_is_identity_and_a_varying_whitelist_is_still_flagged(
    shop,
    body,
    refused,
):
    """The tag is constant in every same-shape message of the store (4 messages, 3 episodes)."""
    mem, gate = shop
    extra = {
        "h2": [
            _dl(0, {"type": "SubmitFeedback", "version": 3, "attempt": 3, "ok": True}),
        ],
        "h3": [
            _dl(0, {"type": "SubmitFeedback", "version": 3, "attempt": 4, "ok": False}),
        ],
    }
    for eid, recorded_actions in extra.items():
        gate.ev.index_episode(_ep(episode_id=eid, actions=recorded_actions), "1" * 40)
    base = gate.lookup
    gate.lookup = lambda eid, i: (
        extra[eid][i]
        if eid in extra and 0 <= i < len(extra[eid])
        else None if eid in extra else base(eid, i)
    )
    mod = _module("parse_feedback", "obs", body, doc="Parse submit feedback.")
    res = _check(mem, gate, "dialogue_user", mod, [24, 25], "parse_feedback")
    got = [r for r in res.reasons if "held-out value refused" in r]
    assert got == [
        f"G2: env/dialogue_user:parse_feedback held-out value refused: {f}"
        for f in refused
    ], res.reasons
    assert res.checks["G2"] is (not refused)
    assert any(
        r.startswith("note:") and "not perturbed: type, version" in r
        for r in res.reasons
    ), res.reasons


ROOM_WHITELIST = """    if obs["room"] != "hall":
        raise MemoryInputError("not the recorded room")
    return obs"""


@needs_bwrap
def test_a_declared_form_the_covers_cannot_give_fails_g2(shop):
    """C1: a dialogue whitelist declared as taking a path would otherwise check nothing."""
    mem, gate = shop
    mod = _module("parse", "obs", ROOM_WHITELIST, doc="Parse a room.")
    res = _check(mem, gate, "dialogue_user", mod, [5], "parse", input_kind="path")
    assert res.checks["G1"], res.reasons
    assert not res.checks["G2"]
    assert (
        "G2: env/dialogue_user:parse declares input path, which a dialogue cover cannot give"
        in res.reasons
    ), res.reasons
    # declared as an observation, the whitelist is caught on its one cover
    ok = _check(mem, gate, "dialogue_user", mod, [5], "parse", input_kind="observation")
    assert [r for r in ok.reasons if "held-out value refused" in r] == [
        "G2: env/dialogue_user:parse held-out value refused: room",
    ], ok.reasons


STOCK_TEXT_HEAD = """    if not isinstance(data, str):
        raise MemoryInputError("data must be the file's text")
    lines = data.splitlines()
    if not lines or lines[0] != "sku,colour,qty,price,restocked":
        raise MemoryInputError("unexpected columns")
    rows = [ln.split(",") for ln in lines[1:] if ln]
"""
STOCK_TEXT_SHAPE = STOCK_TEXT_HEAD + "    return rows"
STOCK_TEXT_WHITELIST = STOCK_TEXT_HEAD + """    for r in rows:
        if r[1] not in ("red", "blue"):
            raise MemoryInputError("unknown colour")
    return rows"""


@needs_bwrap
@pytest.mark.parametrize(
    "body, input_kind, refused, baseline_refused",
    [
        (STOCK_TEXT_SHAPE, "text", [], False),
        (STOCK_TEXT_WHITELIST, "text", ["colour"], False),
        # declared as a path, the same text reader refuses its own covered input: nothing is checked
        (STOCK_TEXT_SHAPE, "path", [], True),
    ],
    ids=["text-shape-only", "text-whitelist", "declared-path"],
)
def test_a_text_reader_is_called_with_the_files_text(
    shop,
    body,
    input_kind,
    refused,
    baseline_refused,
):
    """A parse_load_log(data)-style reader gets the decoded text when it declares input text."""
    mem, gate = shop
    mod = _module("read_stock_text", "data", body, doc="Parse a stock CSV's text.")
    res = _check(
        mem,
        gate,
        "worktree_workspace",
        mod,
        [4],
        "read_stock_text",
        input_kind=input_kind,
    )
    assert res.checks["G1"], res.reasons
    got = [r for r in res.reasons if "held-out value refused" in r]
    assert got == [
        f"G2: env/worktree_workspace:read_stock_text held-out value refused: {f}"
        for f in refused
    ], res.reasons
    own = any("refuses 1 of its own covered inputs" in r for r in res.reasons)
    assert own is baseline_refused, res.reasons


def test_a_result_line_nested_too_deep_to_parse_is_dropped_not_raised(tmp_path):
    # 9deefbfd1 caught only ValueError here, so json.loads' RecursionError escaped the held-out check
    deep = "[" * 100_000 + "]" * 100_000
    f = tmp_path / "results.jsonl"
    f.write_text(deep + "\n" + json.dumps({"id": 0, "outcome": "handled"}) + "\n")
    assert _read_results(f) == [{"id": 0, "outcome": "handled"}]

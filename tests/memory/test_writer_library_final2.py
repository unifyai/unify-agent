"""Identical parts answered with a pointer to where the same agent saw them, and finish validating the manifest with
the gate's own parser (the repair-reserve pricing is a known issue: KNOWN-ISSUES.md). Built on test_writer_pass's and
test_writer_library_tools' helpers."""

from unify.memory import writer_pass
from tests.memory import writer_library_items as wl
from tests.memory.test_writer_pass import _call
from tests.memory.test_writer_library_tools import EPS, _episode, _pass

# --- identical parts ---------------------------------------------------------------------------------------


def test_an_identical_part_is_answered_with_where_it_was_shown(tmp_path):
    turns = [
        _call("a", "read_episode", {"episode": "e1", "part": "request"}),
        _call(
            "b",
            "read_episode",
            {"episode": "e2", "part": "request"},
        ),  # the same bytes as e1's
    ]
    out, model = _pass(tmp_path, turns, max_calls=10)
    assert model.outputs["a"] == '== request ==\n"req"'
    assert model.outputs["b"].startswith(
        "== request ==\nidentical to e1 request (sha256 ",
    )
    assert model.outputs["b"].endswith('read e1 parts=["request"] to see it again')
    assert out.coverage["opened"] == ["e1"]


def test_what_one_agent_was_shown_never_counts_for_another():
    from unify.memory import writer_tools as wt

    helper_shown: dict = {}
    writer_shown: dict = {}
    log = wt.ReadLog()
    args = {"episode": "e1", "parts": ["request"]}
    wt.read_episode(
        EPS["e1"],
        "e1",
        args,
        blob=lambda s: b"",
        log=log,
        shown=helper_shown,
    )
    got = wt.read_episode(
        EPS["e2"],
        "e2",
        {"episode": "e2", "parts": ["request"]},
        blob=lambda s: b"",
        log=log,
        shown=writer_shown,
    )
    assert got.text == '== request ==\n"req"'  # the writer was never shown e1's request


def test_a_part_shown_only_in_part_earns_no_identical_credit(tmp_path):
    long = (
        "x" * 9000
    )  # over one view: e3's request is shown in part, so it is not "shown complete"
    # the same long observation (an identical request would now be one shared block, shown whole)
    EPS["e3"], EPS["e4"] = _episode("e3", request=("first", long)), _episode(
        "e4",
        request=("second", long),
    )
    try:
        turns = [
            _call("a", "read_episode", {"episode": "e3", "part": "observation:0"}),
            _call("b", "read_episode", {"episode": "e4", "part": "observation:0"}),
        ]
        # three unrelated episodes keep the long text in under half the batch: not shared text
        for e in ("e5", "e6", "e7"):
            EPS[e] = _episode(e, request=(f"other {e}",))
        out, model = _pass(
            tmp_path,
            turns,
            eids=("e3", "e4", "e5", "e6", "e7"),
            max_calls=10,
        )
    finally:
        for e in ("e3", "e4", "e5", "e6", "e7"):
            EPS.pop(e, None)
    assert "identical to" not in model.outputs["b"]


# --- finish validates the manifest ------------------------------------------------------------------------


def test_finish_names_the_gates_parse_error(tmp_path, monkeypatch):
    monkeypatch.setattr(
        writer_pass,
        "_read_manifest",
        lambda box: (True, {**wl.MANIFEST, "items": "x"}, None),
    )
    turns = [
        _call("f", "finish", {"summary": "s"}),
    ]
    _, model = _pass(tmp_path, turns, eids=("e2",), max_calls=10)
    assert (
        model.outputs["f"]
        == wl.FINISH_CELL + "not finished: G1: malformed manifest: items must be a list"
    )


def test_an_empty_manifest_at_finish_records_nothing_to_store(tmp_path, monkeypatch):
    empty = {**wl.MANIFEST, "items": [], "support": []}
    monkeypatch.setattr(writer_pass, "_read_manifest", lambda box: (True, empty, None))
    turns = [
        _call("f", "finish", {"summary": "nothing to store"}),
    ]
    out, _ = _pass(tmp_path, turns, eids=("e2",), max_calls=10)
    assert out.finished and out.nothing_to_store


# --- the first repair round is priced on the write phase ----------------------------------------------------

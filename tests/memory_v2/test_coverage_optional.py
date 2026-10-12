"""UNIFY_MEMORY_LIBRARY_COVERAGE (MAIN 11 Oct; prepared, ``required`` until the lead decides; the subtractive note
38c0bd35): ``optional`` lets the writer finish without covering; an episode left neither covered nor dismissed is
recorded as unread (the pass's coverage and, with the pool, an ``unread`` entry), never dropped; ``dismiss`` stays a
tool; the pass reports coverage measures that nothing reads back. ``required`` is today's behaviour, byte for byte.
"""

from unify.memory_v2 import pool, library_prompts as pv, writer_pass
from unify.memory_v2.integration import switch
from unify.memory_v2.writer_pass import PassConfig, writer_tools
from unify.settings import SETTINGS
from tests.memory_v2.test_pool import EMPTY, _finish_with, _Stores
from tests.memory_v2.test_writer_pass import _call, needs_bwrap
from tests.memory_v2.test_writer_library_tools import _pass

FINISH = [_call("f", "finish", {"summary": "s"})]


def test_the_switch_is_required_by_default_and_takes_only_its_two_values():
    assert SETTINGS.UNIFY_MEMORY_LIBRARY_COVERAGE == "required"
    assert PassConfig().coverage_rule == "required"
    assert switch.parse_library_coverage("") == "required"
    assert switch.parse_library_coverage(" Optional ") == "optional"
    assert switch.PARSERS[switch.LIBRARY_COVERAGE] is switch.parse_library_coverage
    try:
        switch.parse_library_coverage("off")
    except ValueError as exc:
        assert "UNIFY_MEMORY_LIBRARY_COVERAGE" in str(exc)
    else:
        raise AssertionError("an unknown value must be refused")


def test_the_brief_is_the_same_under_both_values_and_names_no_reading_rule():
    for pool_on in (False, True):
        text = pv.write_brief_now(pool=pool_on)
        assert (
            "neither covered nor dismissed" not in text and "required_parts" not in text
        )
        assert "every episode in the batch" not in text


def test_dismiss_stays_a_tool_under_both_values():
    names = {t["function"]["name"] for t in writer_tools(library_mode=True)}
    assert "dismiss" in names and "read_episode" in names


def test_finish_is_accepted_without_reading_under_both_values(tmp_path, monkeypatch):
    """The batch map names no reading list, so finish never waits for reading and no episode is unread."""
    _finish_with(monkeypatch, EMPTY)
    for rule in ("required", "optional"):
        (tmp_path / rule).mkdir()
        out, model = _pass(
            tmp_path / rule,
            FINISH,
            eids=("e2",),
            max_calls=3,
            coverage_rule=rule,
        )
        assert out.finished and "f" not in model.outputs
        assert (out.coverage or {}).get("unread", []) == []


@needs_bwrap
def test_no_unread_entry_enters_the_pool(tmp_path, monkeypatch):
    uid = pool.entry_id("unread", "e2")
    with monkeypatch.context() as m:
        _finish_with(m, EMPTY)
        st = _Stores(tmp_path / "o")
        st.run(FINISH, "p1", coverage_rule="optional")
    assert uid not in st.pool().entries


@needs_bwrap
def test_a_dismissed_or_deferred_episode_is_not_unread(tmp_path, monkeypatch):
    turns = [
        _call(
            "d",
            "dismiss",
            {"episode": "e2", "reason": "nothing here", "defer": True},
        ),
        *FINISH,
    ]
    with monkeypatch.context() as m:
        _finish_with(m, EMPTY)
        st = _Stores(tmp_path)
        out, _ = st.run(turns, "p1", coverage_rule="optional")
    assert out.coverage["unread"] == []
    assert pool.entry_id("unread", "e2") not in st.pool().entries
    assert out.coverage_measures["unread"] == 0


def test_the_measures_name_unread_episodes_that_carry_evidence():
    coverage = {
        "episodes": 3,
        "missing": ["a", "b"],
        "bytes_required": 1000,
        "bytes_read": 30,
    }
    rows = [
        {
            "episode_id": "a",
            "functions": [{"name": "f"}],
            "structural_events": [{"kind": "cell_error", "cell": 1}],
            "pool_entries": [],
            "staging": {"candidates": [{"name": "g"}]},
        },
        {
            "episode_id": "b",
            "functions": [],
            "structural_events": [],
            "staging": {"candidates": []},
        },
        {"episode_id": "c", "functions": [{"name": "h"}]},  # read: not unread
    ]
    m = writer_pass._coverage_measures(coverage, ["c"], rows, [])
    assert m["unread_with_evidence"] == {
        "a": ["function", "cell_error", "staging"],
    }
    assert (
        m["read_share"] == 0.03
        and m["empty_pass"] is True
        and m["episodes_opened"] == 1
    )
    assert (
        writer_pass._coverage_measures(coverage, ["c"], rows, ["x"])["empty_pass"]
        is False
    )
    assert (
        writer_pass._coverage_measures({"missing": []}, [], [], [])["read_share"]
        is None
    )


def test_a_measures_failure_never_fails_the_pass(tmp_path, monkeypatch):
    _finish_with(monkeypatch, EMPTY)

    def broken(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(writer_pass, "_coverage_measures", broken)
    out, _ = _pass(
        tmp_path,
        FINISH,
        eids=("e2",),
        max_calls=3,
        coverage_rule="optional",
    )
    assert out.finished
    assert out.coverage_measures in (None, {"error": "RuntimeError: boom"})

"""UNIFY_MEMORY_V21_COVERAGE (MAIN 11 Oct; prepared, ``required`` until the lead decides; the subtractive note
38c0bd35): ``optional`` lets the writer finish without covering; an episode left neither covered nor dismissed is
recorded as unread (the pass's coverage and, with the pool, an ``unread`` entry), never dropped; ``dismiss`` stays a
tool; the pass reports coverage measures that nothing reads back. ``required`` is today's behaviour, byte for byte.
"""

from unify.memory_v2 import pool, prompts_v21 as pv, sol_pass
from unify.memory_v2.integration import switch
from unify.memory_v2.sol_pass import PassConfig, sol_tools
from unify.settings import SETTINGS
from tests.memory_v2.test_pool import EMPTY, _finish_with, _Stores
from tests.memory_v2.test_sol_pass import _call, needs_bwrap
from tests.memory_v2.test_sol_v21_tools import _pass

FINISH = [_call("f", "finish", {"summary": "s"})]


def test_the_switch_is_required_by_default_and_takes_only_its_two_values():
    assert SETTINGS.UNIFY_MEMORY_V21_COVERAGE == "required"
    assert PassConfig().coverage_rule == "required"
    assert switch.parse_v21_coverage("") == "required"
    assert switch.parse_v21_coverage(" Optional ") == "optional"
    assert switch.PARSERS[switch.V21_COVERAGE] is switch.parse_v21_coverage
    try:
        switch.parse_v21_coverage("off")
    except ValueError as exc:
        assert "UNIFY_MEMORY_V21_COVERAGE" in str(exc)
    else:
        raise AssertionError("an unknown value must be refused")


def test_the_required_brief_is_unchanged_and_the_optional_one_states_the_contract():
    for pool_on in (False, True):
        req = pv.write_brief_now(pool=pool_on)
        assert req == pv.write_brief_now(pool=pool_on, coverage="required")
        assert (
            "`finish` is refused while any episode is neither covered nor dismissed. Scan the episodes cheaply"
            in req
        )
        assert "You are done when every episode in the batch is covered" in req
        opt = pv.write_brief_now(pool=pool_on, coverage="optional")
        assert pv.FINISH_COVERAGE_OPTIONAL + "Scan the episodes cheaply" in opt
        assert "is refused while any episode" not in opt
        assert "You are done when every item you wrote has passed the gate" in opt
        # only the two sentences differ
        assert opt.replace(pv.FINISH_COVERAGE_OPTIONAL, "") == req.replace(
            "`finish` is refused while any episode is neither covered nor dismissed. ",
            "",
        ).replace(
            "every episode in the batch is "
            + (
                "covered, dismissed with a reason, or attached to a pool entry"
                if pool_on
                else "covered or dismissed with a reason"
            )
            + ", and ",
            "",
        )
    low = pv.FINISH_COVERAGE_OPTIONAL.lower()
    assert " must " not in low and " only " not in low and "measure" not in low
    assert "share" not in low and "evidence" not in low and "task" not in low
    assert pv.benchmark_words(opt) == [] and pv.example_checks(opt) == []


def test_dismiss_stays_a_tool_under_both_values():
    names = {t["function"]["name"] for t in sol_tools(v21=True)}
    assert "dismiss" in names and "read_episode" in names


def test_finish_needs_coverage_only_when_required(tmp_path, monkeypatch):
    _finish_with(monkeypatch, EMPTY)
    (tmp_path / "r").mkdir()
    (tmp_path / "o").mkdir()
    req, rmodel = _pass(tmp_path / "r", FINISH, eids=("e2",), max_calls=3)
    assert rmodel.outputs["f"].startswith("not finished: 1 episode(s)")
    assert not req.finished and req.coverage_measures is None
    assert "unread" not in (req.coverage or {})
    opt, omodel = _pass(
        tmp_path / "o",
        FINISH,
        eids=("e2",),
        max_calls=3,
        coverage_rule="optional",
    )
    # accepted at its first call: the pass ended there, so no refusal was ever fed back to the model
    assert opt.finished and "f" not in omodel.outputs and "fin" not in omodel.outputs
    assert opt.coverage["unread"] == ["e2"] and opt.coverage["missing"] == ["e2"]
    m = opt.coverage_measures
    assert m["unread"] == 1 and m["episodes"] == 1 and m["episodes_opened"] == 0
    assert m["read_share"] == 0.0 and m["empty_pass"] is True


@needs_bwrap
def test_an_unread_episode_is_kept_in_the_pool_never_dropped(tmp_path, monkeypatch):
    uid = pool.entry_id("unread", "e2")
    with monkeypatch.context() as m:
        _finish_with(m, EMPTY)
        st = _Stores(tmp_path / "o")
        out, _ = st.run(FINISH, "p1", coverage_rule="optional")
    entry = st.pool().entries[uid]
    assert entry.kind == "unread" and entry.status == "open"
    assert entry.provenance["episodes"] == ["e2"] and entry.data["parts_read"] == 0
    assert out.pool["unread_open"] == 1
    with monkeypatch.context() as m:
        _finish_with(m, EMPTY)
        st2 = _Stores(tmp_path / "r")
        out2, _ = st2.run(FINISH + FINISH, "p1")
    assert uid not in st2.pool().entries and "unread_open" not in out2.pool


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
            "signals": [{"kind": "cell_error", "cell": 1}],
            "s0_clusters": [{"key": "k"}],
            "pool_entries": [],
            "staging": {"candidates": [{"name": "g"}]},
        },
        {
            "episode_id": "b",
            "functions": [],
            "signals": [],
            "staging": {"candidates": []},
        },
        {"episode_id": "c", "functions": [{"name": "h"}]},  # read: not unread
    ]
    m = sol_pass._coverage_measures(coverage, ["c"], rows, [])
    assert m["unread_with_evidence"] == {
        "a": ["function", "cell_error", "s0", "staging"],
    }
    assert (
        m["read_share"] == 0.03
        and m["empty_pass"] is True
        and m["episodes_opened"] == 1
    )
    assert (
        sol_pass._coverage_measures(coverage, ["c"], rows, ["x"])["empty_pass"] is False
    )
    assert (
        sol_pass._coverage_measures({"missing": []}, [], [], [])["read_share"] is None
    )


def test_a_measures_failure_never_fails_the_pass(tmp_path, monkeypatch):
    _finish_with(monkeypatch, EMPTY)

    def broken(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(sol_pass, "_coverage_measures", broken)
    out, _ = _pass(
        tmp_path,
        FINISH,
        eids=("e2",),
        max_calls=3,
        coverage_rule="optional",
    )
    assert out.finished and out.coverage["unread"] == ["e2"]
    assert out.coverage_measures == {"error": "RuntimeError: boom"}

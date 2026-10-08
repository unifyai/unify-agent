from unify.memory_v2.evidence import EvidenceStore
from unify.memory_v2.trigger import Trigger
from tests.memory_v2.test_episodes import _ep


def test_incremental_then_lift_then_maintenance(tmp_path):
    ev = EvidenceStore(tmp_path / "e.sqlite")
    tr = Trigger(ev, maintenance_every=3, mode="d6")
    ev.index_episode(_ep(episode_id="a"), "1" * 40)
    [r] = tr.after_episode("a")
    assert (r.kind, r.channel, r.episodes, r.lift) == (
        "incremental",
        "venmo",
        ["a"],
        False,
    )
    ev.index_episode(_ep(episode_id="b"), "2" * 40)
    [r] = tr.after_episode("b")
    assert (
        r.episodes == ["a", "b"] and r.lift
    )  # not marked done: both pending, lifting allowed
    tr.mark_done(r)
    ev.index_episode(_ep(episode_id="c"), "3" * 40)
    reqs = tr.after_episode("c")
    assert [x.kind for x in reqs] == ["incremental", "maintenance"] and reqs[
        0
    ].episodes == ["c"]


def test_drift_queues_channel_without_new_episodes(tmp_path):
    ev = EvidenceStore(tmp_path / "e.sqlite")
    tr = Trigger(ev, maintenance_every=100, mode="d6")
    ev.index_episode(_ep(episode_id="a"), "1" * 40)
    tr.mark_done(tr.after_episode("a")[0])
    tr.queue_drift("venmo")
    ev.index_episode(_ep(episode_id="b", actions=[]), "2" * 40)
    [r] = tr.after_episode("b")
    assert r.channel == "venmo" and r.kind == "incremental"


# --- the batched mode (the default) ----------------------------------------------------------------------

from decimal import Decimal

import pytest

from unify.memory_v2.episodes import Action, Cell
from unify.memory_v2.experience import count_text, experience_pieces, experience_tokens


def _big(eid, n_chars):
    """An episode whose experience is dominated by one cell output of *n_chars* characters."""
    return _ep(
        episode_id=eid,
        cells=[Cell(0, "print(x)", "w " * (n_chars // 2))],
        transcript=[],
        actions=[],
    )


def test_batched_is_the_default_and_fires_on_experience_tokens(tmp_path):
    ev = EvidenceStore(tmp_path / "e.sqlite")
    a, b = _big("a", 1200), _ep(
        episode_id="b",
        actions=[],
        transcript=[],
    )  # b: no channel
    budget = experience_tokens(a)[0] + experience_tokens(b)[0] + 1
    tr = Trigger(ev, experience_budget=budget)
    assert tr.mode == "batched"
    assert Trigger(ev).budget == 150_000
    ev.index_episode(a, "1" * 40)
    assert tr.after_episode("a") == []
    ev.index_episode(b, "2" * 40)
    assert tr.after_episode("b") == []  # one token short
    ev.index_episode(_big("c", 1200), "3" * 40)
    [r] = tr.after_episode("c")
    assert (r.kind, r.channel, r.episodes, r.lift) == (
        "batched",
        None,
        ["a", "b", "c"],
        True,
    )
    assert r.experience_tokens == sum(ev.experience_of(e)[0] for e in "abc") >= budget
    assert tr.after_episode("c") == [r]  # due until marked done
    tr.mark_done(r)
    assert tr.pending() == []
    ev.index_episode(_big("d", 200), "4" * 40)
    assert tr.after_episode("d") == [] and [e for e, _ in tr.pending()] == ["d"]


def test_batched_has_no_maintenance_or_per_channel_passes_and_drift_rides_along(
    tmp_path,
):
    ev = EvidenceStore(tmp_path / "e.sqlite")
    tr = Trigger(ev, maintenance_every=1, experience_budget=10**9)
    ev.index_episode(_ep(episode_id="a"), "1" * 40)
    tr.queue_drift("venmo")
    assert tr.after_episode("a") == []
    small = Trigger(ev, experience_budget=1)
    small.queue_drift("venmo")
    [r] = small.after_episode("a")
    small.mark_done(r)
    assert small._drift == set()


def test_idle_trigger_is_off_by_default_and_needs_three_new_trajectories(tmp_path):
    ev = EvidenceStore(tmp_path / "e.sqlite")
    off = Trigger(ev, experience_budget=10**9)
    on = Trigger(ev, experience_budget=10**9, idle_after_s=600)
    for i, eid in enumerate("abc"):
        ev.index_episode(_ep(episode_id=eid), str(i) * 40)
        assert on.after_episode(eid, at=100.0 * i) == []
        off.after_episode(eid, at=100.0 * i)
        if eid != "c":
            assert on.idle(10_000) == []  # fewer than three new
    assert off.idle(10_000) == []
    assert on.idle(200 + 599) == []  # not idle long enough
    [r] = on.idle(200 + 600)
    assert r.kind == "batched" and r.episodes == ["a", "b", "c"]


def test_d6_mode_is_kept_and_modes_are_validated(tmp_path):
    ev = EvidenceStore(tmp_path / "e.sqlite")
    with pytest.raises(ValueError):
        Trigger(ev, mode="stream")
    with pytest.raises(ValueError):
        Trigger(ev, experience_budget=0)
    ev.index_episode(_ep(episode_id="a"), "1" * 40)
    [r] = Trigger(ev, mode="d6").after_episode("a")
    assert (r.kind, r.channel) == ("incremental", "venmo")


def test_pass_budget_is_experience_times_usd_per_token(tmp_path):
    ev = EvidenceStore(tmp_path / "e.sqlite")
    assert Trigger(ev).pass_budget_usd() == Decimal("0.10950000")
    assert (
        str(
            Trigger(
                ev,
                experience_budget=1000,
                usd_per_token="0.000001",
            ).pass_budget_usd(),
        )
        == "0.001000"
    )


# --- experience tokens -----------------------------------------------------------------------------------


def _msg(role, content, **extra):
    return {"type": "message", "message": {"role": role, "content": content, **extra}}


def test_experience_counts_recorded_content_once_whatever_the_transcript_resends():
    replies = [
        _msg("assistant", "I paid Bob back."),
        _msg("assistant", "Done; anything else?"),
    ]
    plain = _ep(
        episode_id="p",
        transcript=[_msg("user", "Pay my Venmo friends back"), *replies],
    )
    resent = _ep(
        episode_id="r",
        transcript=[
            _msg("user", "Pay my Venmo friends back"),
            replies[0],
            # a later model call re-sends the earlier context, and a compaction summarises it
            _msg("user", "Pay my Venmo friends back"),
            replies[0],
            {
                "type": "message",
                "in_context_replay": True,
                "message": {
                    "role": "assistant",
                    "content": "I paid Bob back. (replayed)",
                },
            },
            _msg("assistant", "Summary of earlier turns", compaction=True),
            replies[1],
        ],
    )
    assert experience_tokens(plain) == experience_tokens(resent)
    assert experience_pieces(plain) == experience_pieces(resent)
    n, how = experience_tokens(plain)
    assert how in ("tiktoken:o200k_base", "bytes/4") and n > 0


def test_experience_counts_cells_observations_and_shapes_not_file_bodies():
    body = "vendor_id,amount\n" + "V-1,2.5\n" * 5000
    wt = Action(
        0,
        "worktree:workspace",
        "read",
        ["inv.csv"],
        {},
        {
            "blob_before": "a" * 64,
            "blob_after": "a" * 64,
            "size": len(body),
            "shape": {"format": "csv", "columns": ["vendor_id", "amount"]},
        },
        "ok",
        kind="worktree",
    )
    obs = "Status: health 9\nInventory: wood 1"
    dl = Action(-1, "dialogue:user", "reply", ["do"], {}, obs, "ok", kind="dialogue")
    sh = Action(
        1,
        "shell:uv",
        "run",
        ["uv run pytest"],
        {},
        {"exit_code": 1, "tail": "1 failed"},
        "error",
        kind="shell",
    )
    ep = _ep(
        request=["Collect wood", obs],
        transcript=[],
        cells=[Cell(0, "print(1)", "1\n")],
        actions=[_ep().actions[0], wt, dl, sh],
    )
    pieces = experience_pieces(ep)
    assert (
        pieces.count(obs) == 1
    )  # the dialogue observation is the next user message: counted once
    assert "1 failed" in pieces and "print(1)" in pieces
    assert not any("V-1" in p for p in pieces)  # the file's shape, not its body
    assert '{"blob": "' + "z" * 10 in "".join(pieces)  # the tool response
    assert experience_tokens(ep)[0] == sum(count_text(p) for p in pieces)


def test_evidence_records_experience_per_episode(tmp_path):
    ev = EvidenceStore(tmp_path / "e.sqlite")
    ep = _ep(episode_id="a")
    ev.index_episode(ep, "1" * 40)
    assert ev.experience_of("a") == experience_tokens(ep)
    with pytest.raises(KeyError):
        ev.experience_of("nope")


def test_experience_falls_back_to_bytes_over_four(monkeypatch):
    import sys

    from unify.memory_v2 import experience

    monkeypatch.setitem(sys.modules, "tiktoken", None)
    experience._encoder.cache_clear()
    try:
        assert experience.counter() == "bytes/4"
        assert experience.count_text("abcde") == 2 and experience.count_text("") == 0
        ep = _ep(request=["abcd"], transcript=[], cells=[], actions=[])
        assert experience_tokens(ep) == (1, "bytes/4")
    finally:
        experience._encoder.cache_clear()

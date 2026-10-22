from unify.memory.evidence import EvidenceStore
from tests.memory.test_episodes import _ep


# --- the batched mode (the default) ----------------------------------------------------------------------


import pytest

from unify.memory.episodes import Action, Cell
from unify.memory.experience import count_text, experience_pieces, experience_tokens
from tests.memory import store_reads


def _big(eid, n_chars):
    """An episode whose experience is dominated by one cell output of *n_chars* characters."""
    return _ep(
        episode_id=eid,
        cells=[Cell(0, "print(x)", "w " * (n_chars // 2))],
        transcript=[],
        actions=[],
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
    assert store_reads.experience_of(ev, "a") == experience_tokens(ep)
    with pytest.raises(KeyError):
        store_reads.experience_of(ev, "nope")


def test_experience_falls_back_to_bytes_over_four(monkeypatch):
    import sys

    from unify.memory import experience

    monkeypatch.setitem(sys.modules, "tiktoken", None)
    experience._encoder.cache_clear()
    try:
        assert experience.counter() == "bytes/4"
        assert experience.count_text("abcde") == 2 and experience.count_text("") == 0
        ep = _ep(request=["abcd"], transcript=[], cells=[], actions=[])
        assert experience_tokens(ep) == (1, "bytes/4")
    finally:
        experience._encoder.cache_clear()

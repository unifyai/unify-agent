from unify.memory_v2.evidence import EvidenceStore
from tests.memory_v2.test_episodes import _ep


def test_seq_and_channels_and_cursor(tmp_path):
    ev = EvidenceStore(tmp_path / "e.sqlite")
    assert ev.index_episode(_ep(episode_id="a"), "1" * 40) == 1
    assert ev.index_episode(_ep(episode_id="b"), "2" * 40) == 2
    assert ev.episode_ids_since("venmo", 0) == ["a", "b"]
    ev.set_cursor("venmo", 1)
    assert ev.episode_ids_since("venmo", ev.cursor("venmo")) == ["b"]


def test_covers(tmp_path):
    ev = EvidenceStore(tmp_path / "e.sqlite")
    ev.index_episode(_ep(episode_id="a"), "1" * 40)
    ev.add_cover("env/venmo:login", "a", 0)
    assert ev.covered() == {("a", 0)}
    assert ev.covers() == {("env/venmo:login", "a", 0)}


def test_add_pass_notes_appends_to_the_recorded_reasons(tmp_path):
    import json

    import pytest

    ev = EvidenceStore(tmp_path / "e.sqlite")
    ev.record_pass(
        {"pass_id": "p", "passed": 1, "reasons": json.dumps(["note: a"]), "usd": "0"},
    )
    ev.add_pass_notes("p", ["deadline reached", "note: 2 unpriced calls"])
    ev.add_pass_notes("p", [])
    row = ev.db.execute(
        "SELECT reasons, passed FROM passes WHERE pass_id='p'",
    ).fetchone()
    assert json.loads(row[0]) == [
        "note: a",
        "deadline reached",
        "note: 2 unpriced calls",
    ]
    assert row[1] == 1
    with pytest.raises(KeyError):
        ev.add_pass_notes("missing", ["x"])

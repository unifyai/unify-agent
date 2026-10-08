from unify.memory_v2.blobs import BlobStore
from unify.memory_v2.episodes import (
    Action,
    Cell,
    CostRow,
    Episode,
    EpisodeWriter,
    episode_dir,
    load_episode,
)
from unify.memory_v2.gitio import Repo
from unify.memory_v2.redact import Redactor

KEY = "sk-or-v1-" + "cd" * 32  # pragma: allowlist secret


def _ep(**kw):
    base = dict(
        episode_id="e-0001",
        started_at="2026-10-08T01:00:00Z",
        ended_at="2026-10-08T01:01:00Z",
        build="f64bfca2c",
        model="openai/gpt-6-luna",
        effort="low",
        regime="dense",
        memory_main="0" * 40,
        worktree_before=None,
        worktree_after=None,
        request=["Pay my Venmo friends back"],
        transcript=[{"role": "user", "content": "hi " + KEY}],
        cells=[Cell(0, "print(apis.venmo.login(username='a'))", "{'token': 'x'}")],
        actions=[
            Action(
                0,
                "venmo",
                "login",
                [],
                {"username": "a"},
                {"blob": "z" * 30000},
                "ok",
                "read",
            ),
        ],
        costs=[CostRow("actor", "openai/gpt-6-luna", 10, 2, "0.000123")],
        fingerprints={},
    )
    base.update(kw)
    return Episode(**base)


def test_write_then_load_roundtrip_with_cap_and_redaction(tmp_path):
    repo = Repo.init_bare(tmp_path / "episodes.git")
    blobs = BlobStore(tmp_path / "blobs")
    w = EpisodeWriter(repo, blobs, Redactor(), response_cap=1000)
    sha = w.write(_ep())
    assert "Episode: e-0001" in repo.run("log", "-1", "--format=%B", sha)
    raw = repo.run("show", f"{sha}:2026/10/e-0001/actions.jsonl")
    assert '"blob":' in raw and "z" * 2000 not in raw  # capped into the blob store
    assert KEY not in repo.run("show", f"{sha}:2026/10/e-0001/transcript.jsonl")
    ep = load_episode(repo, sha, episode_dir(_ep()), blobs)
    assert ep.actions[0].response == {
        "blob": "z" * 30000,
    }  # restored from the blob store
    assert ep.costs[0].usd == "0.000123"


def test_episodes_are_append_only_linear(tmp_path):
    repo = Repo.init_bare(tmp_path / "episodes.git")
    w = EpisodeWriter(repo, BlobStore(tmp_path / "b"), Redactor())
    s1 = w.write(_ep(episode_id="e1"))
    s2 = w.write(_ep(episode_id="e2"))
    assert repo.log_shas()[-2:] == [s1, s2]


def test_action_kind_defaults_to_tool_and_is_validated():
    import pytest

    a = Action(0, "venmo", "login", [], {})
    assert a.kind == "tool"
    for k in ("tool", "shell", "worktree", "dialogue"):
        assert Action(0, "c", "m", [], {}, kind=k).kind == k
    with pytest.raises(ValueError):
        Action(0, "c", "m", [], {}, kind="email")


def test_old_rows_without_kind_load_as_tool_and_kind_roundtrips(tmp_path):
    import json

    repo = Repo.init_bare(tmp_path / "episodes.git")
    blobs = BlobStore(tmp_path / "blobs")
    w = EpisodeWriter(repo, blobs, Redactor())
    shell = Action(1, "sh", "run", ["ls"], {}, {"exit": 0}, "ok", "read", kind="shell")
    sha = w.write(_ep(actions=[_ep().actions[0], shell]))
    rel = episode_dir(_ep())
    ep = load_episode(repo, sha, rel, blobs)
    assert [a.kind for a in ep.actions] == ["tool", "shell"]
    # A row written before kinds existed has no "kind" key.
    old = json.loads(repo.show(sha, f"{rel}/actions.jsonl").decode().splitlines()[0])
    old.pop("kind")
    assert Action(**old).kind == "tool"

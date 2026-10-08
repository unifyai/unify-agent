"""Package prerequisites for the actor integration (integration Task 16)."""

import subprocess

import pytest

from tests.memory_v2.test_episodes import _ep
from unify.memory_v2 import gitio
from unify.memory_v2.blobs import BlobStore
from unify.memory_v2.episodes import (
    Action,
    Cell,
    EpisodeWriter,
    episode_dir,
    load_episode,
)
from unify.memory_v2.evidence import EvidenceStore
from unify.memory_v2.fingerprint import Generations
from unify.memory_v2.gitio import GitError, Repo
from unify.memory_v2.redact import Redactor
from unify.memory_v2.sol_pass import export_for_sol


def test_kind_language_replies_roundtrip(tmp_path):
    repo = Repo.init_bare(tmp_path / "e.git")
    blobs = BlobStore(tmp_path / "b")
    ep = _ep(
        cells=[Cell(0, "ls", "a\n", None, "bash")],
        replies=["done"],
        actions=[
            Action(
                0,
                "dialogue:user",
                "reply",
                ["done"],
                {},
                "next",
                "ok",
                "unknown",
                None,
                "dialogue",
            ),
        ],
    )
    sha = EpisodeWriter(repo, blobs, Redactor()).write(ep)
    back = load_episode(repo, sha, episode_dir(ep), blobs)
    assert back.cells[0].language == "bash"
    assert back.replies == ["done"]
    assert back.actions[0].kind == "dialogue"


def test_old_rows_default_to_tool_and_python():
    assert Action(0, "venmo", "login", [], {}).kind == "tool"
    assert Cell(0, "x = 1", "").language == "python"


def test_episode_without_replies_file_loads_with_none(tmp_path):
    repo = Repo.init_bare(tmp_path / "e.git")
    blobs = BlobStore(tmp_path / "b")
    ep = _ep()
    sha = EpisodeWriter(repo, blobs, Redactor()).write(ep)
    rel = episode_dir(ep)
    with repo.temp_checkout() as wt:
        (wt / rel / "replies.json").unlink()
        old = repo.commit_all(wt, "drop replies", {})
    repo.fast_forward("main", old, expected_old=sha)
    assert load_episode(repo, old, rel, blobs).replies == []


def test_replies_are_redacted(tmp_path):
    repo = Repo.init_bare(tmp_path / "e.git")
    blobs = BlobStore(tmp_path / "b")
    ep = _ep(replies=["your token is tok-abcdefgh123"])
    sha = EpisodeWriter(repo, blobs, Redactor({"t": "tok-abcdefgh123"})).write(ep)
    text = repo.show(sha, f"{episode_dir(ep)}/replies.json").decode()
    assert "tok-abcdefgh123" not in text


def test_episode_ref(tmp_path):
    ev = EvidenceStore(tmp_path / "ev.sqlite")
    ep = _ep()
    ev.index_episode(ep, "a" * 40)
    assert ev.episode_ref(ep.episode_id) == ("a" * 40, ep.started_at)
    with pytest.raises(KeyError):
        ev.episode_ref("missing")


def test_generations_json_roundtrip():
    g = Generations()
    g.observe({"venmo.login": {"shapes": ["{a:int}"], "errors": []}})
    g2 = Generations.from_json(g.to_json())
    assert g2.observe({"venmo.login": {"shapes": ["{b:int}"], "errors": []}}) == {
        "venmo",
    }
    assert g2.generation("venmo") == 1
    assert Generations.from_json(g2.to_json()).generation("venmo") == 1


def test_export_for_sol_carries_action_kind(tmp_path):
    ep = _ep(actions=[Action(0, "venmo", "me", [], {}, {"id": 1}, "ok", "read")])
    export_for_sol(lambda eid: ep, ["e-0001"], tmp_path)
    assert '"kind": "tool"' in (tmp_path / "e-0001.json").read_text()


def test_global_git_config_is_ignored(tmp_path, monkeypatch):
    hooks = tmp_path / "hooks"
    hooks.mkdir()
    marker = tmp_path / "ran"
    (hooks / "pre-commit").write_text(f"#!/bin/sh\ntouch {marker}\n")
    (hooks / "pre-commit").chmod(0o755)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    (tmp_path / ".gitconfig").write_text(f"[core]\n\thooksPath = {hooks}\n")
    Repo.init_bare(tmp_path / "m.git")  # commits the init commit
    assert not marker.exists()


def test_git_timeout_becomes_git_error(monkeypatch):
    def slow(*a, **kw):
        assert kw.get("timeout") == gitio.GIT_TIMEOUT_S
        raise subprocess.TimeoutExpired(a[0], kw["timeout"])

    monkeypatch.setattr(gitio.subprocess, "run", slow)
    with pytest.raises(GitError, match="timed out"):
        gitio._git(["status"])

"""P1 Task 7 (Amendment B), the writer and the loader under UNIFY_MEMORY_V21 (RUNTIME's checklist points 1–4).

With ``EpisodeWriter(v21=True)`` the record is format 2: every large value the recorders now keep whole (each
observation in ``request``, an action's args, kwargs and response) is stored as a content-addressed blob after
redaction, a recorded value that looks like a reference is escaped, and :func:`load_episode` resolves format 2
strictly. A missing or corrupt blob fails closed, in both formats."""

import json

import pytest

from unify.memory_v2 import batch_map as bm
from unify.memory_v2.blobs import BlobStore
from unify.memory_v2.episodes import (
    Action,
    EpisodeRecordError,
    EpisodeWriter,
    episode_dir,
    load_episode,
)
from unify.memory_v2.gitio import Repo
from unify.memory_v2.redact import Redactor
from tests.memory_v2.test_episodes import KEY, _ep

RED = Redactor(secrets={"OPENROUTER_API_KEY": KEY})
BIG_OBS = "o" * 70000 + KEY + "OBS_END"
BIG_ARG = "a" * 20000 + KEY + "ARG_END"
BIG_RESP = {"rows": "r" * 30000 + KEY + "RESP_END"}


def _v21_ep(**kw):
    base = dict(
        request=["Pay my friends back", BIG_OBS],
        actions=[
            Action(
                0,
                "svc",
                "send",
                [BIG_ARG],
                {"note": BIG_ARG},
                BIG_RESP,
                "ok",
                "write",
            ),
            Action(
                0,
                "svc",
                "echo",
                [{"__literal__": "x"}],
                {},
                {"__capped__": 1},
                "ok",
                "read",
            ),
        ],
    )
    base.update(kw)
    return _ep(**base)


def _write(tmp_path, ep, v21=True):
    repo = Repo.init_bare(tmp_path / "episodes.git")
    blobs = BlobStore(tmp_path / "blobs")
    sha = EpisodeWriter(repo, blobs, RED, v21=v21).write(ep)
    return repo, blobs, sha


def _record_files(repo, sha, rel):
    names = (
        "meta.json",
        "request.json",
        "actions.jsonl",
        "cells.jsonl",
        "transcript.jsonl",
    )
    return {n: repo.show(sha, f"{rel}/{n}").decode() for n in names}


def test_v21_record_stores_large_values_as_blobs_and_loads_them_whole(tmp_path):
    ep = _v21_ep()
    repo, blobs, sha = _write(tmp_path, ep)
    files = _record_files(repo, sha, episode_dir(ep))
    assert json.loads(files["meta.json"])["record_format"] == 2
    # each large value is a reference (its head as an excerpt, its tail only in the blob)
    assert (
        "__capped__" in files["request.json"] and "OBS_END" not in files["request.json"]
    )
    assert files["actions.jsonl"].count('{"__capped__": {"blob"') == 3
    assert (
        "ARG_END" not in files["actions.jsonl"]
        and "RESP_END" not in files["actions.jsonl"]
    )
    got = load_episode(repo, sha, episode_dir(ep), blobs)
    redact = lambda s: s.replace(KEY, RED.text(KEY))  # noqa: E731
    assert got.request == ["Pay my friends back", redact(BIG_OBS)]
    send, echo = got.actions
    assert send.args == [redact(BIG_ARG)] and send.kwargs == {"note": redact(BIG_ARG)}
    assert send.response == {"rows": redact(BIG_RESP["rows"])}
    assert echo.args == [{"__literal__": "x"}] and echo.response == {
        "__capped__": 1,
    }  # escaped, round-trips


def test_no_key_reaches_any_record_file_or_blob(tmp_path):
    ep = _v21_ep()
    repo, _, sha = _write(tmp_path, ep)
    for text in _record_files(repo, sha, episode_dir(ep)).values():
        assert KEY not in text
    assert not any(
        KEY.encode() in p.read_bytes()
        for p in (tmp_path / "blobs").rglob("*")
        if p.is_file()
    )


def _format1_ep():
    """Format 1 reads any dict with a "__capped__" key as a reference (as at 4675a3c45), so its episodes
    leave out the colliding echo action that format 2 escapes."""
    return _v21_ep(actions=_v21_ep().actions[:1])


def test_format_1_is_unchanged_when_off(tmp_path):
    ep = _format1_ep()
    repo, blobs, sha = _write(tmp_path, ep, v21=False)
    files = _record_files(repo, sha, episode_dir(ep))
    assert "record_format" not in json.loads(files["meta.json"])
    assert (
        '"__capped__"' not in files["request.json"]
    )  # format 1 caps action responses only
    got = load_episode(repo, sha, episode_dir(ep), blobs)
    assert got.actions[0].response["rows"].endswith("RESP_END")


@pytest.mark.parametrize("v21", [True, False])
@pytest.mark.parametrize("damage", ["missing", "corrupt"])
def test_a_missing_or_corrupt_blob_fails_closed(tmp_path, v21, damage):
    ep = _v21_ep() if v21 else _format1_ep()
    repo, blobs, sha = _write(tmp_path, ep, v21=v21)
    stored = [p for p in (tmp_path / "blobs").rglob("*") if p.is_file()]
    assert stored
    for p in stored:
        if damage == "missing":
            p.unlink()
        else:
            p.write_bytes(b"tampered")
    with pytest.raises(
        EpisodeRecordError,
        match=f"episode {ep.episode_id}: blob [0-9a-f]{{64}} {damage}",
    ):
        load_episode(repo, sha, episode_dir(ep), blobs)


def test_part_text_and_coverage_sizes_use_the_resolved_value(tmp_path):
    ep = _v21_ep()
    repo, blobs, sha = _write(tmp_path, ep)
    got = load_episode(repo, sha, episode_dir(ep), blobs)
    text = bm.part_text(got, "observation:0")
    assert text == json.dumps(BIG_OBS.replace(KEY, RED.text(KEY)))
    assert len(text.encode()) > 70000


def _export(repo, sha, rel, dest):
    dest.mkdir()
    for name in ("meta.json", "transcript.jsonl", "actions.jsonl", "memory.diff"):
        (dest / name).write_bytes(repo.show(sha, f"{rel}/{name}"))
    return dest


def test_offline_use_analysis_refuses_or_resolves_a_format_2_record(tmp_path):
    from unify.memory_v2.analysis.use import use_from_episode_dir

    ep = _v21_ep()
    repo, blobs, sha = _write(tmp_path, ep)
    d = _export(repo, sha, episode_dir(ep), tmp_path / "export")
    with pytest.raises(ValueError, match="record_format 2: pass blobs to resolve"):
        use_from_episode_dir(d)
    assert isinstance(use_from_episode_dir(d, blobs=blobs), dict)
    old = _format1_ep()
    repo1, _, sha1 = _write(tmp_path / "f1", old, v21=False)
    assert isinstance(
        use_from_episode_dir(_export(repo1, sha1, episode_dir(old), tmp_path / "e1")),
        dict,
    )

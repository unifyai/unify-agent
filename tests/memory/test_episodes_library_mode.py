"""The episode writer and loader (rev 2: the library layout is always on).

The record is format 2: every large value the recorders keep whole (each observation in ``request``, an action's args,
kwargs and response) is stored as a content-addressed blob after redaction, a recorded value that looks like a
reference is escaped, and :func:`load_episode` resolves format 2 strictly. A missing or corrupt blob fails closed."""

import json

import pytest

from unify.memory import batch_map as bm
from unify.memory.blobs import BlobStore
from unify.memory.episodes import (
    Action,
    EpisodeRecordError,
    EpisodeWriter,
    episode_dir,
    load_episode,
)
from unify.memory.gitio import Repo
from unify.memory.redact import Redactor
from tests.memory.test_episodes import KEY, _ep

RED = Redactor(secrets={"OPENROUTER_API_KEY": KEY})
BIG_OBS = "o" * 70000 + KEY + "OBS_END"
BIG_ARG = "a" * 20000 + KEY + "ARG_END"
BIG_RESP = {"rows": "r" * 30000 + KEY + "RESP_END"}


def _library_ep(**kw):
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


def _write(tmp_path, ep):
    """The episode as rev 2's writer records it: always record format 2 (format 1 was the library-off format)."""
    repo = Repo.init_bare(tmp_path / "episodes.git")
    blobs = BlobStore(tmp_path / "blobs")
    sha = EpisodeWriter(repo, blobs, RED).write(ep)
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


def test_library_record_stores_large_values_as_blobs_and_loads_them_whole(tmp_path):
    ep = _library_ep()
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
    ep = _library_ep()
    repo, _, sha = _write(tmp_path, ep)
    for text in _record_files(repo, sha, episode_dir(ep)).values():
        assert KEY not in text
    assert not any(
        KEY.encode() in p.read_bytes()
        for p in (tmp_path / "blobs").rglob("*")
        if p.is_file()
    )


@pytest.mark.parametrize("damage", ["missing", "corrupt"])
def test_a_missing_or_corrupt_blob_fails_closed(tmp_path, damage):
    ep = _library_ep()
    repo, blobs, sha = _write(tmp_path, ep)
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
    ep = _library_ep()
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
    from unify.memory.analysis.use import use_from_episode_dir

    ep = _library_ep()
    repo, blobs, sha = _write(tmp_path, ep)
    d = _export(repo, sha, episode_dir(ep), tmp_path / "export")
    with pytest.raises(ValueError, match="record_format 2: pass blobs to resolve"):
        use_from_episode_dir(d)
    assert isinstance(use_from_episode_dir(d, blobs=blobs), dict)


@pytest.mark.parametrize("bad", [3, "x"])
def test_an_unknown_record_format_fails_closed(tmp_path, monkeypatch, bad):
    import unify.memory.episodes as episodes_mod
    from unify.memory.analysis.use import use_from_episode_dir

    ep = _library_ep()
    monkeypatch.setattr(episodes_mod, "RECORD_FORMAT", bad)  # the writer stamps it
    repo, blobs, sha = _write(tmp_path, ep)
    monkeypatch.undo()
    with pytest.raises(
        EpisodeRecordError,
        match=f"episode {ep.episode_id}: unknown record_format {bad!r}",
    ):
        load_episode(repo, sha, episode_dir(ep), blobs)
    d = _export(repo, sha, episode_dir(ep), tmp_path / "export")
    with pytest.raises(ValueError, match="unknown record_format"):
        use_from_episode_dir(d, blobs=blobs)


def test_offline_use_refuses_references_without_a_readable_meta(tmp_path):
    from unify.memory.analysis.use import use_from_episode_dir

    ep = _library_ep()
    repo, blobs, sha = _write(tmp_path, ep)
    d = _export(repo, sha, episode_dir(ep), tmp_path / "export")
    (d / "meta.json").unlink()
    with pytest.raises(ValueError, match="no readable meta.json"):
        use_from_episode_dir(d, blobs=blobs)
    (d / "meta.json").write_text("{not json")
    with pytest.raises(ValueError, match="no readable meta.json"):
        use_from_episode_dir(d, blobs=blobs)

"""The ``similar`` tool: an incremental, deterministic per-episode index of normalised code units, built at
recording time, and the tool's replies (matches with counts and excerpts, refusals for unknown episodes or cells).
"""

import subprocess
from types import SimpleNamespace

import pytest

from unify.memory import similar, workspace_trees
from unify.memory.episodes import Cell
from unify.memory.redact import Redactor

CODE = (
    "path = os.path.join(base, name)\n"
    "rows = [line.split(',') for line in open(path).read().splitlines() if line]\n"
    "totals = {}\n"
    "for key, value in rows:\n"
    "    totals[key] = totals.get(key, 0) + int(value)\n"
)
# the same code with other names and constants
RENAMED = (
    "p = os.path.join(folder, fname)\n"
    "recs = [ln.split(';') for ln in open(p).read().splitlines() if ln]\n"
    "acc = {}\n"
    "for k, v in recs:\n"
    "    acc[k] = acc.get(k, 1) + int(v)\n"
)
UNRELATED = "import json\ndata = json.loads(raw)\nprint(sorted(data.keys())[:5], len(data), type(data).__name__)\n"


def _ep(eid, *codes, errors=()):
    return SimpleNamespace(
        episode_id=eid,
        worktree_before=None,
        cells=[
            Cell(i, c, "out", "boom" if i in errors else None)
            for i, c in enumerate(codes)
        ],
    )


EPISODES = {
    # different import sets around the same code: `from os import path` here, `import os` there
    "e1": _ep("e1", "from os import path\nimport os", CODE),
    "e2": _ep("e2", "import os, json", CODE),
    "e3": _ep("e3", RENAMED, UNRELATED, errors=(0,)),
    "e4": _ep("e4", UNRELATED),
}


def _indexed(tmp_path, eids=("e1", "e2", "e3", "e4")):
    root = tmp_path / "index"
    for e in eids:
        similar.index_episode(root, e, EPISODES[e].cells)
    return similar.Context(
        index_dir=root,
        load=EPISODES.__getitem__,
        recorded=lambda: list(EPISODES),
    )


# --- the index -----------------------------------------------------------------------------------------------------


def test_the_index_is_built_once_per_content_and_is_byte_stable(tmp_path):
    root = tmp_path / "a"
    got = similar.index_episode(root, "e1", EPISODES["e1"].cells)
    assert got["status"] == "written" and got["units"] > 0
    path = root / "episodes" / "e1.json"
    first = path.read_bytes()
    mtime = path.stat().st_mtime_ns
    assert similar.index_episode(root, "e1", EPISODES["e1"].cells)["status"] == "kept"
    assert path.read_bytes() == first and path.stat().st_mtime_ns == mtime
    # the same cells give the same bytes anywhere
    similar.index_episode(tmp_path / "b", "e1", EPISODES["e1"].cells)
    assert (tmp_path / "b" / "episodes" / "e1.json").read_bytes() == first
    # changed content is indexed again
    changed = [*EPISODES["e1"].cells, Cell(2, UNRELATED, "")]
    assert similar.index_episode(root, "e1", changed)["status"] == "written"
    assert path.read_bytes() != first
    # other episodes are untouched: one file each
    similar.index_episode(root, "e2", EPISODES["e2"].cells)
    assert sorted(p.name for p in (root / "episodes").iterdir()) == [
        "e1.json",
        "e2.json",
    ]


def test_indexing_stops_at_its_time_limit_and_is_completed_later(tmp_path):
    ticks = iter(range(0, 10_000, 5))
    entry = similar.build_entry(
        "e3",
        EPISODES["e3"].cells,
        time_limit_s=1,
        clock=lambda: next(ticks),
    )
    assert entry["complete"] is False and entry["cells_indexed"] < 2
    root = tmp_path / "i"
    ticks2 = iter(range(0, 10_000, 5))
    got = similar.index_episode(
        root,
        "e3",
        EPISODES["e3"].cells,
        time_limit_s=1,
        clock=lambda: next(ticks2),
    )
    assert got["status"] == "partial"
    # a partial entry is not kept: the next build completes it
    assert (
        similar.index_episode(root, "e3", EPISODES["e3"].cells)["status"] == "written"
    )
    assert similar.read_entry(root, "e3")["complete"] is True


def test_the_recording_hook_indexes_redacted_cells(tmp_path):
    secret = "s3cr3t-value-0123456789abcdef"  # pragma: allowlist secret
    ep = _ep(
        "e9",
        f"token = '{secret}'\nheaders = {{'auth': token, 'kind': 'x'}}\nprint(headers, len(headers))\n",
    )
    got = similar.record_episode(tmp_path, ep, Redactor({"API_TOKEN": secret}))
    assert got["status"] == "written"
    text = (similar.index_dir(tmp_path) / "episodes" / "e9.json").read_text()
    assert secret not in text and "<secret:API_TOKEN>" in text


def test_an_episode_id_is_never_a_path(tmp_path):
    with pytest.raises(ValueError):
        similar.index_episode(tmp_path, "../x", [])


# --- the tool ------------------------------------------------------------------------------------------------------


def test_the_same_code_matches_across_episodes_with_different_imports(tmp_path):
    ctx = _indexed(tmp_path)
    out = similar.handle({"episode": "e1", "cell": 1}, ctx)
    assert out.startswith("Code units of episode e1 cell 1:")
    whole = next(
        ln for ln in out.splitlines() if ln.startswith("lines 1-5 (whole cell")
    )
    assert "in 2 other episode(s), 2 place(s)" in whole
    assert "e2 cell 1 lines 1-5" in out and "e3 cell 0 lines 1-5 (cell errored)" in out
    # excerpts as written in each episode
    assert (
        "    | path = os.path.join(base, name)" in out
        and "    | p = os.path.join(folder, fname)" in out
    )
    # the statements inside the whole cell match only inside its places: not listed again
    assert "smaller unit(s) match only inside the places above" in out
    assert "e4" not in out.split("Searched")[1]


def test_code_given_directly_is_looked_for_in_every_indexed_episode(tmp_path):
    ctx = _indexed(tmp_path)
    out = similar.handle({"code": UNRELATED}, ctx)
    assert (
        out.startswith("Code units of the given code:")
        and "Searched 4 other indexed episodes" in out
    )
    assert "in 2 other episode(s)" in out and "e3 cell 1" in out and "e4 cell 0" in out


def test_unindexed_and_unmatched_are_stated(tmp_path):
    ctx = _indexed(tmp_path, eids=("e1", "e3"))
    out = similar.handle({"episode": "e1", "cell": 1}, ctx)
    assert "Not indexed (2 recorded episodes, not searched): e2, e4." in out
    lonely = similar.handle(
        {"code": "q = [x * x for x in range(10) if x % 2]\nprint(q, sum(q))\n"},
        ctx,
    )
    assert "No other episode holds: lines 1-2" in lonely


def test_an_unknown_episode_or_cell_is_a_refusal(tmp_path):
    ctx = _indexed(tmp_path)
    assert (
        similar.handle({"episode": "zz", "cell": 0}, ctx)
        == "refused: no recorded episode zz."
    )
    assert (
        similar.handle({"episode": "e1", "cell": 9}, ctx)
        == "refused: episode e1 has no cell 9; its cells are 0-1."
    )
    # without a loader, the index answers alone
    bare = similar.Context(index_dir=ctx.index_dir)
    assert (
        similar.handle({"episode": "zz", "cell": 0}, bare)
        == "refused: no recorded episode zz."
    )
    assert similar.handle({"episode": "e1", "cell": 1}, bare) == similar.handle(
        {"episode": "e1", "cell": 1},
        ctx,
    )
    for bad in (
        {},
        {"episode": "e1"},
        {"cell": 1},
        {"episode": "e1", "cell": 1, "code": "x"},
        {"episode": "e1", "cell": True},
    ):
        assert (
            similar.handle(bad, ctx)
            == "refused: give `episode` and `cell` together, or `code` alone."
        ), bad
    assert similar.handle({"code": "def (:"}, ctx).startswith(
        "refused: the code does not parse as Python (",
    )
    # empty values are absent ones: the form models send when they fill every field of the schema
    by_cell = similar.handle({"episode": "e1", "cell": 1}, ctx)
    assert similar.handle({"episode": "e1", "cell": 1, "code": ""}, ctx) == by_cell
    assert similar.handle({"episode": "e1", "cell": 1, "code": None}, ctx) == by_cell
    by_code = similar.handle({"code": "x = 1"}, ctx)
    assert similar.handle({"episode": "", "cell": 0, "code": "x = 1"}, ctx) == by_code
    assert similar.handle({"episode": "", "cell": "", "code": ""}, ctx).startswith(
        "refused: ",
    )


def test_the_reply_is_deterministic(tmp_path):
    a = similar.handle({"episode": "e1", "cell": 1}, _indexed(tmp_path / "a"))
    b = similar.handle({"episode": "e1", "cell": 1}, _indexed(tmp_path / "b"))
    assert a == b


# --- the hooks -----------------------------------------------------------------------------------------------------


def test_the_request_indexes_the_episode_it_just_recorded(tmp_path, monkeypatch):
    from unify.memory.integration import request

    errors = []
    run = SimpleNamespace(
        paths=SimpleNamespace(state_dir=tmp_path),
        _recorded=(EPISODES["e1"], Redactor()),
        _error=lambda stage, exc, progress: errors.append((stage, exc)),
    )
    monkeypatch.setattr(request, "memory_library_on", lambda: False)
    request.RequestRun._index_similar(run, lambda _t: None)
    assert not similar.index_dir(tmp_path).exists() and run._recorded is None
    run._recorded = (EPISODES["e1"], Redactor())
    monkeypatch.setattr(request, "memory_library_on", lambda: True)
    request.RequestRun._index_similar(run, lambda _t: None)
    assert (
        similar.read_entry(similar.index_dir(tmp_path), "e1")["complete"] is True
        and not errors
    )
    # a failure is logged, never raised
    run._recorded = (SimpleNamespace(episode_id="../bad", cells=[]), None)
    request.RequestRun._index_similar(run, lambda _t: None)
    assert [s for s, _ in errors] == ["similar index"]


def test_the_writer_pass_hands_the_tool_its_index_loader_and_episode_list(tmp_path):
    from unify.memory.writer_pass import WriterPass

    fake = SimpleNamespace(
        cfg=SimpleNamespace(similar_index=str(tmp_path / "index")),
        ev=SimpleNamespace(experience_since=lambda seq: [("e1", 3), ("e2", 4)]),
        load=EPISODES.__getitem__,
    )
    ctx = WriterPass.similar_context(fake)
    assert (
        ctx.index_dir == tmp_path / "index"
        and ctx.load is fake.load
        and list(ctx.recorded()) == ["e1", "e2"]
    )
    none = WriterPass.similar_context(
        SimpleNamespace(cfg=SimpleNamespace(similar_index=None), ev=None, load=None),
    )
    assert none.index_dir is None and similar.load_index(none.index_dir) == {}


# --- recorded office workspaces, without the miner -----------------------------------------------------------------


def _git(*args, cwd):
    subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        env={
            "PATH": "/usr/bin:/bin", "HOME": str(cwd), "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
            "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
        },
    )  # fmt: skip


def test_office_workspaces_are_exported_once_from_the_worktree_repo(tmp_path):
    repo = tmp_path / "wt"
    repo.mkdir()
    _git("init", "-q", cwd=repo)
    (repo / "claims.csv").write_text("a,1\nb,2\n")
    _git("add", "claims.csv", cwd=repo)
    _git("commit", "-q", "-m", "before", cwd=repo)
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    eps = {
        "o1": SimpleNamespace(worktree_before=sha),
        "o2": SimpleNamespace(worktree_before=None),
    }

    def load(eid):
        return eps[eid]

    cache = tmp_path / "trees"
    got = workspace_trees.trees_for(
        load,
        ["o1", "o2", "missing", "../o1"],
        str(repo / ".git"),
        cache,
    )
    assert (
        list(got) == ["o1"] and (got["o1"] / "claims.csv").read_text() == "a,1\nb,2\n"
    )
    (got["o1"] / "marker").write_text("kept")  # a second request reuses the export
    again = workspace_trees.trees_for(load, ["o1"], str(repo / ".git"), cache)
    assert (again["o1"] / "marker").exists()
    assert workspace_trees.trees_for(load, ["o1"], None, cache) == {}
    assert (
        workspace_trees.export_tree(str(repo / ".git"), "0" * 40, cache / "bad") is None
    )
    assert not (cache / "bad").exists() and not (cache / "bad.part").exists()

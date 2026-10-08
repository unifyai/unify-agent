from pathlib import Path

import pytest

from unify.memory_v2.gitio import GitError, Repo


def test_bare_repo_commit_via_temp_checkout(tmp_path):
    r = Repo.init_bare(tmp_path / "mem.git")
    base = r.head()
    with r.temp_checkout() as wt:
        (wt / "env").mkdir()
        (wt / "env" / "a.py").write_text("x = 1\n")
        sha = r.commit_all(wt, "add a", {"Pass": "p1", "Episode": ["e1", "e2"]})
    r.fast_forward("main", sha, expected_old=base)
    assert r.head() == sha
    msg = r.run("log", "-1", "--format=%B", sha)
    assert "Pass: p1" in msg and "Episode: e1" in msg and "Episode: e2" in msg
    assert r.changed_paths(base, sha) == ["env/a.py"]


def test_fast_forward_refuses_non_ancestor(tmp_path):
    r = Repo.init_bare(tmp_path / "m.git")
    base = r.head()
    with r.temp_checkout() as wt:
        (wt / "a").write_text("1")
        s1 = r.commit_all(wt, "a", {})
    with r.temp_checkout() as wt:
        (wt / "b").write_text("2")
        s2 = r.commit_all(wt, "b", {})
    r.fast_forward("main", s1, expected_old=base)
    with pytest.raises(GitError):
        r.fast_forward("main", s2, expected_old=base)  # main moved
    assert r.head() == s1


def test_notes_append_and_read(tmp_path):
    r = Repo.init_bare(tmp_path / "e.git")
    r.add_note(r.head(), '{"x":1}')
    r.add_note(r.head(), '{"x":2}')
    assert r.notes(r.head()) == ['{"x":1}', '{"x":2}']


def test_snapshot_mode_keeps_git_out_of_work_tree(tmp_path):
    wt = tmp_path / "work"
    wt.mkdir()
    (wt / "f.txt").write_text("a\nb\n")
    s = Repo.init_snapshot(tmp_path / "work.git", wt)
    first = s.snapshot("before")
    (wt / "f.txt").write_text("a\nB\n")
    second = s.snapshot("after")
    assert not (wt / ".git").exists()
    assert s.blame_lines(second, "f.txt") == [first, second]


def test_init_bare_accepts_a_relative_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    repo = Repo.init_bare(Path("rel") / "memory.git")
    assert repo.git_dir.is_absolute()
    assert repo.git_dir == tmp_path / "rel" / "memory.git"


def test_commit_all_never_lets_a_gitignore_hide_a_file(tmp_path):
    r = Repo.init_bare(tmp_path / "m.git")
    with r.temp_checkout() as wt:
        (wt / ".gitignore").write_text("*\n")  # ignores everything, itself included
        (wt / "a.py").write_text("x = 1\n")
        sha = r.commit_all(wt, "a", {})
    assert sorted(r.run("ls-tree", "-r", "--name-only", sha).split()) == [
        ".gitignore",
        "a.py",
    ]

"""The writer's /memory is taken as regular files and directories only: git metadata, links, hard links and
special files are refused by check and finish with one message, and the copy into the harness's checkout never
carries a .git entry."""

import os

from unify.memory_v2 import writer_pass


def _tree(root):
    (root / "memory" / "grids").mkdir(parents=True)
    (root / "memory" / "grids" / "fill.py").write_text("def fill():\n    return 1\n")
    (root / ".pass").mkdir()
    (root / ".pass" / "manifest.json").write_text("{}")
    return root


def test_a_plain_tree_has_no_layout_problem(tmp_path):
    assert writer_pass._layout_problems(_tree(tmp_path)) == []


def test_git_entries_links_hard_links_and_specials_are_named(tmp_path):
    box = _tree(tmp_path)
    (box / ".gitattributes").write_text("* filter=x\n")
    (box / "memory" / ".gitmodules").write_text("")
    (box / "memory" / ".git").mkdir()
    os.symlink("/etc/passwd", box / "memory" / "link.py")
    os.link(box / "memory" / "grids" / "fill.py", box / "memory" / "grids" / "twin.py")
    os.mkfifo(box / "memory" / "pipe")
    found = writer_pass._layout_problems(box)
    text = "\n".join(found)
    assert ".gitattributes: a path starting with .git" in text
    assert "memory/.gitmodules: a path starting with .git" in text
    assert "memory/.git: a path starting with .git" in text
    assert "memory/link.py: a symbolic link" in text
    assert "memory/grids/twin.py: a hard link (2 links)" in text
    assert "memory/grids/fill.py: a hard link (2 links)" in text
    assert "memory/pipe: not a regular file or directory" in text
    msg = writer_pass._layout_refusal(found)
    assert msg.startswith("layout: the library is taken as regular files and directories only")
    assert writer_pass._GATE_LINE.match(msg)


def test_the_pass_folder_is_not_part_of_the_candidate(tmp_path):
    box = _tree(tmp_path)
    os.symlink("manifest.json", box / ".pass" / "alias.json")
    assert writer_pass._layout_problems(box) == []


def test_the_copy_never_carries_git_entries(tmp_path):
    box = _tree(tmp_path / "box")
    (box / ".gitignore").write_text("*\n")
    (box / "memory" / ".git").mkdir()
    (box / "memory" / ".git" / "config").write_text("[core]\n")
    dst = tmp_path / "dst"
    dst.mkdir()
    writer_pass._mirror(box, dst, skip_top=frozenset({".pass"}))
    names = {p.name for p in dst.rglob("*")}
    assert not any(n.startswith(".git") for n in names)
    assert (dst / "memory" / "grids" / "fill.py").read_text().startswith("def fill")

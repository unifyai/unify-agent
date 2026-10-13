"""The library copy never follows a link: a box bound read-write to the copied directory can run while the host copies
it (a helper's snapshot of the writer's library), so an entry swapped for a link after it was listed is recreated as
the link or left out, a FIFO is left out unread, and no outside byte reaches the copy.
"""

import os
import stat

from unify.memory import writer_pass

SECRET = b"outside-secret-bytes"


def _outside(tmp_path):
    out = tmp_path / "outside"
    out.mkdir()
    (out / "secret.txt").write_bytes(SECRET)
    return out


def _all_bytes(root):
    found = b""
    for dirpath, _, files in os.walk(root, followlinks=False):
        for f in files:
            p = os.path.join(dirpath, f)
            if not os.path.islink(p):
                found += open(p, "rb").read()
    return found


def _swap_after_stat(monkeypatch, name, swap):
    """Run *swap* right after the walk stats *name*, as a box writing beside the copy could."""
    real_stat = os.stat
    done = []

    def stat_then_swap(path, *a, **kw):
        st = real_stat(path, *a, **kw)
        if path == name and kw.get("dir_fd") is not None and not done:
            done.append(True)
            swap()
        return st

    monkeypatch.setattr(os, "stat", stat_then_swap)


def test_an_ordinary_tree_is_copied_with_its_modes_and_pass_is_skipped(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    (src / "memory" / "pkg").mkdir(parents=True)
    (src / "memory" / "pkg" / "a.py").write_text("x = 1\n")
    os.chmod(src / "memory" / "pkg" / "a.py", 0o640)
    (src / ".pass").mkdir()
    (src / ".pass" / "manifest.json").write_text("{}")
    (src / "__pycache__").mkdir()
    dst.mkdir()
    assert writer_pass._mirror(src, dst, skip_top=frozenset({".pass"})) == []
    assert (dst / "memory" / "pkg" / "a.py").read_text() == "x = 1\n"
    assert stat.S_IMODE(os.stat(dst / "memory" / "pkg" / "a.py").st_mode) == 0o640
    assert not (dst / ".pass").exists() and not (dst / "__pycache__").exists()


def test_a_directory_swapped_for_a_link_mid_walk_is_never_followed(
    tmp_path,
    monkeypatch,
):
    out = _outside(tmp_path)
    src, dst = tmp_path / "src", tmp_path / "dst"
    (src / "a").mkdir(parents=True)
    (src / "a" / "ok.py").write_text("ok")
    dst.mkdir()

    def swap():
        os.rename(src / "a", src / "a-moved")
        os.symlink(out, src / "a")

    _swap_after_stat(monkeypatch, "a", swap)
    left = writer_pass._mirror(src, dst)
    assert "a" in left
    assert SECRET not in _all_bytes(dst) and not (dst / "a" / "secret.txt").exists()


def test_a_file_swapped_for_a_link_mid_walk_is_never_followed(tmp_path, monkeypatch):
    out = _outside(tmp_path)
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    (src / "f.py").write_text("fine")
    dst.mkdir()

    def swap():
        os.unlink(src / "f.py")
        os.symlink(out / "secret.txt", src / "f.py")

    _swap_after_stat(monkeypatch, "f.py", swap)
    left = writer_pass._mirror(src, dst)
    assert "f.py" in left and SECRET not in _all_bytes(dst)


def test_a_fifo_is_left_out_unread(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    os.mkfifo(src / "pipe")  # nothing writes it: reading it would block
    (src / "b.py").write_text("b")
    dst.mkdir()
    assert writer_pass._mirror(src, dst) == ["pipe"]
    assert (dst / "b.py").read_text() == "b" and not (dst / "pipe").exists()


def test_a_link_to_the_process_environment_stays_a_link(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    os.symlink("/proc/self/environ", src / "env")
    os.symlink(_outside(tmp_path), src / "dir")
    dst.mkdir()
    assert writer_pass._mirror(src, dst) == []
    assert (
        os.path.islink(dst / "env") and os.readlink(dst / "env") == "/proc/self/environ"
    )
    assert os.path.islink(dst / "dir")
    assert _all_bytes(dst) == b""

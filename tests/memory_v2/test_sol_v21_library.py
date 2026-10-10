"""Sol's box under v2.1 (spec §4, §7.2): the generated helper and index, never committed; no library.json.

P3 Task 7, built on test_sol_pass's own helpers (``Turns``, ``_sol``, ``_run``): the plan's ``_pass``/``_scripted``
sketches do not exist there. Amendment C: the generated paths are read-only binds in the writer's cells.
"""

from __future__ import annotations

import errno

from tests.memory_v2.test_layout import _tree
from tests.memory_v2.test_library_helper import _commit, _py
from tests.memory_v2.test_sol_pass import Turns, _run, _sol, needs_bwrap
from unify.memory_v2.gitio import Repo
from unify.memory_v2.sandbox_run import PYTHON, run_confined
from unify.memory_v2.sol_pass import SolPass, _drop_unchanged


def test_the_box_gets_the_generated_files_and_the_writer_can_use_them(tmp_path):
    mem = Repo.init_bare(tmp_path / "memory")
    sha = _commit(
        mem,
        {
            "memory/text/parse.py": 'def tokens(s):\n    """Split."""\n    return s.split()\n',
        },
        "seed",
    )
    box = _tree(tmp_path / "box")
    sol = SolPass.__new__(SolPass)
    sol.mem = mem
    generated = sol._stage_library_v21(box, sha)
    assert {"memory/__init__.py", "INDEX.md", "links.json"} <= set(generated)
    assert (
        _py(box, "import memory\nprint(memory.index('text').splitlines()[0])")
        == "## memory.text — Text helpers.\n"
    )
    msg = SolPass._library_message_v21(box)
    assert msg.startswith("Current library index (/memory/INDEX.md")
    assert "memory.text.dates:parse_date" in msg
    assert "never write" not in msg  # Amendment C: blocked, not instructed


def test_unchanged_generated_files_are_dropped_and_edited_ones_kept(tmp_path):
    box = _tree(tmp_path / "box")
    sol = SolPass.__new__(SolPass)
    sol.mem = Repo.init_bare(tmp_path / "memory")
    generated = sol._stage_library_v21(box, sol.mem.head())
    (box / "INDEX.md").write_text("edited by the writer\n")
    _drop_unchanged(box, generated)
    assert (
        box / "INDEX.md"
    ).read_text() == "edited by the writer\n"  # the gate refuses it (P4)
    assert not (box / "memory/__init__.py").exists() and not (box / ".memory").exists()
    assert not (box / "links.json").exists() and (box / "memory/text/dates.py").exists()


def test_v21_first_message_carries_the_index_and_no_library_json(tmp_path, monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("library.json is v2's; v2.1 reads the tree")

    monkeypatch.setattr(SolPass, "_stage_context", refuse)
    _, _, sol = _sol(tmp_path, Turns([]), v21=True, max_calls=1)
    _run(sol)
    first = sol.messages[1]["content"]
    assert "Current library index (/memory/INDEX.md" in first
    assert "Functions on this pass's channels" not in first


@needs_bwrap
def test_a_writer_cell_cannot_write_a_generated_path(tmp_path):
    """Amendment C: memory/__init__.py, INDEX.md, links.json and .memory/ are read-only binds over the writable
    box; a write fails at once with the ordinary OS error, and the rest of the box stays writable.
    """
    box = _tree(tmp_path / "box")
    sol = SolPass.__new__(SolPass)
    sol.mem = Repo.init_bare(tmp_path / "memory")
    sol._stage_library_v21(box, sol.mem.head())
    binds = SolPass._generated_binds(box)
    assert set(binds.values()) == {
        "/memory/memory/__init__.py",
        "/memory/INDEX.md",
        "/memory/links.json",
        "/memory/.memory",
    }
    code = (
        "import errno\n"
        "out = []\n"
        "for p in ('INDEX.md', 'links.json', 'memory/__init__.py', '.memory/items.json', '.memory/new.json'):\n"
        "    try:\n"
        "        open(p, 'a').close()\n"
        "        out.append('wrote')\n"
        "    except OSError as e:\n"
        "        out.append(errno.errorcode[e.errno])\n"
        "open('memory/text/new.py', 'w').write('x = 1\\n')\n"
        "print(out)\n"
    )
    r = run_confined(
        [str(PYTHON), "-c", code],
        rw={box: "/memory"},
        cwd="/memory",
        late_ro=binds,
    )
    assert r.stdout.strip() == str([errno.errorcode[errno.EROFS]] * 5), r.stderr
    assert (box / "memory/text/new.py").read_text() == "x = 1\n"

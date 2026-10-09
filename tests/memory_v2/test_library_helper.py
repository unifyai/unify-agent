"""The generated ``memory/__init__.py`` (spec §4.2, §6): index, show and find, run as a cell runs them."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tests.memory_v2.test_layout import NOTE, PARSE, _tree
from unify.memory_v2 import library_export as lx
from unify.memory_v2 import memory_helper
from unify.memory_v2.catalogue import write_files
from unify.memory_v2.gitio import Repo
from unify.memory_v2.layout import INDEX_FILE, LINKS_FILE, RESERVED_INIT
from unify.memory_v2.library_helper import (
    FIND_FILE,
    ITEMS_FILE,
    MATCHER_FILE,
    SHAPES_FILE,
)

DATE = "memory.text.dates:parse_date"


def _py(root: Path, code: str) -> str:
    """Run *code* in a fresh isolated interpreter with *root* first on its path (as a cell imports it)."""
    out = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            f"import sys\nsys.path.insert(0, {str(root)!r})\n{code}",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert out.returncode == 0, out.stderr
    return out.stdout


def _library(tmp_path: Path, **kw) -> Path:
    root = _tree(tmp_path / "lib")
    write_files(root, lx.generated_v21(root, **kw))
    return root


def _commit(mem: Repo, files: dict, message: str) -> str:
    base = mem.head()
    with mem.temp_checkout() as wt:
        for rel, text in files.items():
            (wt / rel).parent.mkdir(parents=True, exist_ok=True)
            (wt / rel).write_text(text)
        sha = mem.commit_all(wt, message, {})
    mem.fast_forward("main", sha, expected_old=base)
    return sha


def test_generated_files_are_the_seven_and_deterministic(tmp_path):
    root = _tree(tmp_path / "lib")
    files = lx.generated_v21(root)
    assert set(files) == {
        RESERVED_INIT,
        INDEX_FILE,
        LINKS_FILE,
        ITEMS_FILE,
        FIND_FILE,
        MATCHER_FILE,
        SHAPES_FILE,
    }
    assert lx.generated_v21(root) == files
    assert (
        files[RESERVED_INIT]
        == (Path(lx.__file__).parent / "library_helper.py").read_bytes()
    )


def test_index_and_show_from_the_generated_helper(tmp_path):
    root = _library(tmp_path, history=({DATE: ["abc123def456 add dates"]}, True))
    code = (
        "import json, memory, memory.text.dates as d\n"
        "try:\n    memory.show('memory.text.dates:nope')\nexcept LookupError as exc:\n    missing = str(exc)\n"
        "print(json.dumps([memory.index(), memory.index('text'), memory.show('"
        + DATE
        + "'),\n"
        "    memory.show('memory.text.dates.week'), memory.show('notes/text/dates.md'), memory.show(d.parse_date),\n"
        "    missing, memory.__file__]))"
    )
    full, text, fn, week, note, by_object, missing, where = json.loads(_py(root, code))
    assert full == (root / "INDEX.md").read_text()
    assert (
        text.startswith("## memory.text — Text helpers.\n") and "memory.web" not in text
    )
    assert (
        fn.splitlines()[0]
        == "memory.text.dates:parse_date(s: str) -> str  [experimental]"
    )
    assert "def parse_date(s: str) -> str:" in fn and "Notes: notes/text/dates.md" in fn
    assert "Links to: notes/text/dates.md\nLinked from: notes/text/dates.md\n" in fn
    assert "Verification: no record yet\nUse: no record yet\n" in fn
    assert "History (newest first):\n  abc123def456 add dates\n" in fn
    assert (
        week.splitlines()[0] == "memory.text.dates:week(s)  [experimental]"
        and "History: none recorded" in week
    )
    assert (
        note.startswith("notes/text/dates.md  [experimental]  Dates\n")
        and "Dates are YYYY-MM-DD." in note
    )
    assert by_object == fn
    assert missing.startswith("memory: no item 'memory.text.dates:nope'")
    assert where == str(root / "memory" / "__init__.py")


def test_show_explains_a_hidden_item(tmp_path):
    root = _library(
        tmp_path,
        status_of=lambda i: "suspect" if i.endswith(":week") else "experimental",
    )
    out = json.loads(
        _py(
            root,
            "import json, memory\nprint(json.dumps(memory.show('memory.text.dates:week')))",
        ),
    )
    assert out.splitlines()[:2] == [
        "memory.text.dates:week(s)  [suspect]",
        "Not in the index: its status is suspect.",
    ]
    assert "memory.text.dates:week(" not in (root / "INDEX.md").read_text()


def test_find_ranks_like_v2s_matcher(tmp_path, monkeypatch):
    value = {"date": "2026-10-09", "zone": "UTC"}
    shapes = {
        DATE: [memory_helper.value_shape({"date": "2025-01-01", "zone": "CET"})],
        "memory.text.parse:tokens": [
            memory_helper.value_shape({"date": "2025-01-01", "zone": "CET", "n": 1}),
        ],
    }
    root = _library(
        tmp_path,
        shapes=lambda item, digest: (shapes[item], False) if item in shapes else None,
    )
    got = json.loads(
        _py(
            root,
            f"import json, memory\nprint(json.dumps([list(f) for f in memory.find({value!r})]))",
        ),
    )
    entries = json.loads((root / FIND_FILE).read_text())["functions"]
    monkeypatch.setattr(memory_helper, "_catalog", lambda: {"functions": entries})
    want = [
        [":".join(f.name.rsplit(".", 1)), *list(f)[1:]]
        for f in memory_helper.find(value)
    ]
    assert got == want
    assert [(g[0], g[3]) for g in got] == [
        (DATE, "exact"),
        ("memory.text.parse:tokens", "structure"),
    ]
    assert (
        json.loads(
            _py(
                root,
                "import json, memory\nprint(json.dumps(memory.find('plain text')))",
            ),
        )
        == []
    )


def test_the_matcher_is_v2s_without_its_catalogue():
    src = lx.matcher_source().decode("utf-8")
    compile(src, "matcher.py", "exec")
    assert (
        "def match(" in src and "def value_shape(" in src and "MAX_RESULTS = 5" in src
    )
    for gone in (
        "recorded for the next consolidation",
        "def catalog(",
        "def describe(",
        "def find(",
    ):
        assert gone not in src


def test_history_lists_the_commits_that_changed_each_item(tmp_path):
    mem = Repo.init_bare(tmp_path / "memory")
    a = _commit(
        mem,
        {"memory/text/parse.py": PARSE, "notes/text/dates.md": NOTE},
        "add parse and a note",
    )
    b = _commit(
        mem,
        {
            "memory/text/parse.py": PARSE.replace(
                "return len(s.split())",
                "return len(s.split()) or 0",
            ),
        },
        "words: count empty text as 0",
    )
    hist, complete = lx.item_history(mem, b)
    assert complete
    assert hist["memory.text.parse:words"] == [
        f"{b[:12]} words: count empty text as 0",
        f"{a[:12]} add parse and a note",
    ]
    assert hist["memory.text.parse:tokens"] == [f"{a[:12]} add parse and a note"]
    assert hist["notes/text/dates.md"] == [f"{a[:12]} add parse and a note"]
    assert lx.item_history(mem, b, scan=1)[1] is False


def test_history_subjects_reach_the_actor_redacted(tmp_path):
    """show() prints each item's history to the actor; a commit subject may come from any merge path, so a
    key-shaped string in it never reaches the export (RUNTIME, P3 Task 3 review)."""
    key = "sk-or-v1-" + "ab" * 32
    mem = Repo.init_bare(tmp_path / "memory")
    a = _commit(mem, {"memory/text/parse.py": PARSE}, f"add parse ({key})")
    hist, _ = lx.item_history(mem, a)
    assert hist["memory.text.parse:words"] == [
        f"{a[:12]} add parse (<redacted:key-shaped>)",
    ]
    out = lx.generated_v21(_tree(tmp_path / "lib"), history=(hist, True))
    assert not any(key.encode() in data for data in out.values())


def test_write_generated_still_writes_v2s_files(tmp_path):
    from unify.memory_v2.catalogue import generated, write_generated

    (tmp_path / "env/x").mkdir(parents=True)
    (tmp_path / "env/x/__init__.py").write_text(
        'def f(a):\n    """F."""\n    return a\n',
    )
    assert write_generated(tmp_path) == generated(tmp_path)
    assert (tmp_path / "memory.py").read_bytes() == (
        Path(memory_helper.__file__)
    ).read_bytes()

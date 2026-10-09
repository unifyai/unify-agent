"""The v2.1 layout (spec §4.1–4.3): path classes, ids, front matter, links as written, discovery, imports."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from unify.memory_v2 import layout

PARSE = '''def tokens(s):
    """Split text on whitespace."""
    return s.split()


def words(s):
    """Count the words of a text."""
    return len(s.split())
'''
DATES = '''"""Date readers."""

from memory.text.parse import tokens


def parse_date(s: str) -> str:
    """Parse a date written as YYYY-MM-DD.

    Notes: notes/text/dates.md
    """
    return "-".join(tokens(s.replace("-", " ")))


def _pad(n):
    return f"{n:02d}"


def week(s):
    """The ISO week of a date."""
    return 1
'''
REPORT = (
    "import memory.text.dates as d\n\n\n"
    'def summary(s):\n    """One line about a date."""\n    return d.parse_date(s)\n'
)
FETCH = (
    "from ..text import tokens\n\n\n"
    'def get(u):\n    """Fetch nothing; split the address."""\n    return tokens(u)\n'
)
NOTE = (
    "---\ntitle: Dates\ndescription: How dates are written.\n"
    "uses: [memory.text.dates:parse_date, memory.text.dates:gone]\n---\nDates are YYYY-MM-DD.\n"
)
#: The fixture library every v2.1 test of P3 reuses.
LIB = {
    "memory/text/__init__.py": '"""Text helpers. More detail follows."""\nfrom .parse import tokens\n',
    "memory/text/parse.py": PARSE,
    "memory/text/dates.py": DATES,
    "memory/text/report.py": REPORT,
    "memory/web/fetch.py": FETCH,
    "memory/text/tests/test_dates.py": (
        "from memory.text.dates import parse_date\n\n\n"
        "def test_parse_date():\n    assert parse_date('2026-10-09') == '2026-10-09'\n"
    ),
    "memory/text/tests/data/sample.json": '{"date": "2026-10-09"}\n',
    "notes/text/dates.md": NOTE,
}
FUNCTIONS = [
    "memory.text.dates:parse_date",
    "memory.text.dates:week",
    "memory.text.parse:tokens",
    "memory.text.parse:words",
    "memory.text.report:summary",
    "memory.web.fetch:get",
]


def _tree(root: Path, files: dict | None = None) -> Path:
    for rel, text in (LIB if files is None else files).items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    return root


@pytest.mark.parametrize(
    "path, kind",
    [
        ("memory/text/dates.py", "module"),
        ("memory/text/__init__.py", "package_init"),
        ("memory/text/tests/test_dates.py", "test"),
        ("memory/text/tests/helpers.py", "test_helper"),
        ("memory/text/tests/data/sample.json", "fixture"),
        ("memory/text/tests/data/deep/x.csv", "fixture"),
        ("notes/text/dates.md", "note"),
        ("memory/__init__.py", "generated"),
        ("INDEX.md", "generated"),
        ("links.json", "generated"),
        (".memory/items.json", "generated"),
        ("memory/text/sub/x.py", None),
        ("memory/text/notes.txt", None),
        ("memory/Text/x.py", None),
        (
            "memory/find/x.py",
            None,
        ),  # would replace memory.find in the package's namespace
        ("memory/text/tests.py", None),
        ("memory/tests/x.py", None),
        ("memory/text/tests/__init__.py", None),
        ("notes/dates.md", None),
        ("notes/text/Dates.md", None),
        ("env/x/__init__.py", None),
        ("README.md", None),
    ],
)
def test_classify(path, kind):
    assert layout.classify(path) == kind


def test_exported_keeps_library_code_and_notes_only():
    paths = [
        "memory/text/__init__.py",
        "memory/text/dates.py",
        "memory/text/tests/test_dates.py",
        "memory/text/tests/data/sample.json",
        "memory/text/__pycache__/dates.cpython-312.pyc",
        "memory/__init__.py",
        "INDEX.md",
        "notes/text/dates.md",
        "README.md",
    ]
    assert [p for p in paths if layout.exported(p)] == [
        "memory/text/__init__.py",
        "memory/text/dates.py",
        "notes/text/dates.md",
    ]


def test_item_ids_and_paths():
    assert layout.FUNCTION_ID.match("memory.text.dates:parse_date")
    assert not layout.FUNCTION_ID.match("memory.text.dates:_pad")
    assert not layout.FUNCTION_ID.match("memory.text:parse_date")
    assert layout.NOTE_ID.match("notes/text/dates.md")
    assert layout.item_path("memory.text.dates:parse_date") == "memory/text/dates.py"
    assert layout.item_path("notes/text/dates.md") == "notes/text/dates.md"
    assert layout.module_of("memory/text/dates.py") == "memory.text.dates"
    assert layout.module_of("memory/text/__init__.py") == "memory.text"
    assert layout.module_of("memory/text/tests/test_dates.py") is None
    assert layout.module_path("memory.text") == "memory/text/__init__.py"
    assert layout.module_path("memory.text.dates") == "memory/text/dates.py"
    assert (
        layout.package_of_path("memory/text/tests/test_dates.py") == "memory.text.tests"
    )


def test_front_matter_parses_the_three_keys():
    meta = layout.parse_front_matter(
        '---\ntitle: "Dates: how"\ndescription: How dates are written.\n'
        "uses: [memory.text.dates:parse_date, 'memory.text.parse:tokens']\n---\nBody line.\n",
    )
    assert (meta.title, meta.description) == ("Dates: how", "How dates are written.")
    assert meta.uses == ["memory.text.dates:parse_date", "memory.text.parse:tokens"]
    assert meta.body == "Body line.\n"
    assert layout.parse_front_matter("---\ntitle: a\ndescription: b\n---\n").uses == []


@pytest.mark.parametrize(
    "text, message",
    [
        ("no front matter\n", "starts with a '---' line"),
        ("---\ntitle: a\n", "no closing '---' line"),
        (
            "---\ntitle: a\ndescription: b\nauthor: c\n---\n",
            "expected one of title, description, uses",
        ),
        ("---\ntitle: a\ntitle: b\ndescription: c\n---\n", "title is given twice"),
        ("---\ntitle: a\n---\n", "non-empty description"),
        (
            "---\ntitle: a\ndescription: b\nuses: memory.x.y:z\n---\n",
            "list on one line",
        ),
        ("---\ntitle: a\ndescription: b\nuses: [a, , b]\n---\n", "empty entry"),
    ],
)
def test_front_matter_is_strict(text, message):
    with pytest.raises(layout.FrontMatterError, match=message):
        layout.parse_front_matter(text)


def test_notes_line_tokens_as_written():
    doc = "Summary.\n\nNotes: notes/text/dates.md, notes/text/zones.md\n    Notes: bad link\n"
    assert layout.notes_links(doc) == [
        "notes/text/dates.md",
        "notes/text/zones.md",
        "bad link",
    ]
    assert layout.notes_links("No notes here.") == []


def test_discover_finds_public_functions_notes_and_package_summaries(tmp_path):
    lib = layout.discover(_tree(tmp_path))
    assert [f.item_id for f in lib.functions] == FUNCTIONS
    first = lib.functions[0]
    assert (first.module, first.name, first.path, first.lineno) == (
        "memory.text.dates",
        "parse_date",
        "memory/text/dates.py",
        6,
    )
    assert first.signature == "parse_date(s: str) -> str"
    assert first.summary == "Parse a date written as YYYY-MM-DD."
    assert first.notes == ["notes/text/dates.md"]
    assert lib.packages == {"text": "Text helpers.", "web": ""}
    assert [
        (n.item_id, n.title, n.description, n.uses, n.error) for n in lib.notes
    ] == [
        (
            "notes/text/dates.md",
            "Dates",
            "How dates are written.",
            ["memory.text.dates:parse_date", "memory.text.dates:gone"],
            "",
        ),
    ]
    assert lib.errors == []


def test_discover_reports_unreadable_files(tmp_path):
    root = _tree(
        tmp_path,
        {
            "memory/text/broken.py": "def x(:\n",
            "notes/text/bad.md": "no front matter\n",
        },
    )
    lib = layout.discover(root)
    assert lib.functions == [] and [n.item_id for n in lib.notes] == [
        "notes/text/bad.md",
    ]
    assert lib.notes[0].error.startswith("a note starts with")
    assert any(e.startswith("memory/text/broken.py: SyntaxError") for e in lib.errors)


def test_function_bodies_are_unparsed_definitions(tmp_path):
    bodies = layout.function_bodies(_tree(tmp_path))
    assert sorted(bodies) == FUNCTIONS
    tree = ast.parse(PARSE)
    assert bodies["memory.text.parse:tokens"] == ast.unparse(tree.body[0])


def test_import_graph_resolves_absolute_relative_and_package_imports(tmp_path):
    root = _tree(tmp_path)
    graph = layout.import_graph(root)
    assert graph == {
        "memory.text": {"memory.text.parse"},
        "memory.text.dates": {"memory.text", "memory.text.parse"},
        "memory.text.parse": set(),
        "memory.text.report": {"memory.text", "memory.text.dates"},
        "memory.web.fetch": {"memory.text"},
    }
    assert layout.dependants(graph, {"memory.text.parse"}) == set(graph)
    assert layout.dependants(graph, {"memory.text.report"}) == {"memory.text.report"}
    assert layout.imported_closure(graph, "memory.text.report") == {
        "memory.text.report",
        "memory.text",
        "memory.text.dates",
        "memory.text.parse",
    }
    refs = layout.imports(
        b"from memory.text.parse import words\nimport memory.web.fetch\n",
        "memory.text.tests",
        layout.library_modules(root),
    )
    assert refs == ({"memory.text.parse:words"}, {"memory.text", "memory.web.fetch"})
    assert layout.imports(b"def x(:\n", "memory.text", set()) is None
    merged = layout.merge_graphs({"a": {"b"}}, {"a": {"c"}, "d": set()})
    assert merged == {"a": {"b", "c"}, "d": set()}


@pytest.mark.parametrize(
    "path",
    [
        "memory/text/tests/data/../../x.py",
        "memory/text/tests/data/./x.json",
        "memory/text/tests/data//x.json",
        "memory//text/x.py",
        "memory/text/../web/fetch.py",
        "notes/text/../dates.md",
        "memory\\text\\dates.py",
        "memory/text/da\x00tes.py",
        "./memory/text/dates.py",
    ],
)
def test_classify_refuses_dot_empty_backslash_and_nul_components(path):
    assert layout.classify(path) is None


def test_discovery_never_reads_through_a_link_or_outside_the_tree(tmp_path):
    root = _tree(tmp_path / "lib")
    outside = tmp_path / "outside"
    (outside / "pkg").mkdir(parents=True)
    (outside / "evil.py").write_text(
        'def leaked():\n    """Outside the library."""\n    return 1\n',
    )
    (outside / "pkg" / "x.py").write_text(
        'def also():\n    """Outside too."""\n    return 2\n',
    )
    (outside / "note.md").write_text(
        "---\ntitle: Leak\ndescription: Outside.\n---\nbody\n",
    )
    (root / "memory/text/evil.py").symlink_to(outside / "evil.py")
    (root / "memory/linked").symlink_to(outside / "pkg")
    (root / "notes/text/leak.md").symlink_to(outside / "note.md")
    lib = layout.discover(root)
    assert [f.item_id for f in lib.functions] == FUNCTIONS
    assert [n.item_id for n in lib.notes] == ["notes/text/dates.md"]
    for rel in ("memory/text/evil.py", "memory/linked/x.py", "notes/text/leak.md"):
        assert f"{rel}: refused: a link or a path outside the library" in lib.errors
    assert sorted(layout.function_bodies(root)) == FUNCTIONS
    assert (
        set(layout.import_graph(root))
        == set(layout.library_modules(root))
        == {
            "memory.text",
            "memory.text.dates",
            "memory.text.parse",
            "memory.text.report",
            "memory.web.fetch",
        }
    )

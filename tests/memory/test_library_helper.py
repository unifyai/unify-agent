"""The generated ``memory/__init__.py``: the earlier calls (index, show, find) as a cell runs them, the generated
files, and the library calls' audit events. The contract itself (``memory.get``, ``memory.items``, ``Item``) is
tested in ``test_library_items_api``."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tests.memory.test_layout import NOTE, PARSE, _tree
from unify.memory import library_export as lx
from unify.memory import memory_helper
from unify.memory.catalogue import write_files
from unify.memory.gitio import Repo
from unify.memory.layout import INDEX_FILE, LINKS_FILE, RESERVED_CONTRACT, RESERVED_INIT, RESERVED_STUB
from unify.memory.library_helper import (
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
    write_files(root, lx.generated_library_mode(root, **kw))
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


def test_generated_files_are_the_nine_and_deterministic(tmp_path):
    root = _tree(tmp_path / "lib")
    files = lx.generated_library_mode(root)
    assert set(files) == {
        RESERVED_INIT,
        RESERVED_CONTRACT,
        RESERVED_STUB,
        INDEX_FILE,
        LINKS_FILE,
        ITEMS_FILE,
        FIND_FILE,
        MATCHER_FILE,
        SHAPES_FILE,
    }
    assert lx.generated_library_mode(root) == files
    assert files[RESERVED_CONTRACT] == lx.CONTRACT_SOURCE.read_bytes()
    assert files[RESERVED_STUB] == lx.stub_source()
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
        == "memory.text.dates:parse_date(s: str) -> str  [on-trial]"
    )
    assert "def parse_date(s: str) -> str:" in fn and "Notes: notes/text/dates.md" in fn
    assert "Links to: notes/text/dates.md\nLinked from: notes/text/dates.md\n" in fn
    assert "Verification: no record yet\nUse: no record yet\n" in fn
    assert "History (newest first):\n  abc123def456 add dates\n" in fn
    assert (
        week.splitlines()[0] == "memory.text.dates:week(s)  [on-trial]"
        and "History: none recorded" in week
    )
    assert (
        note.startswith("notes/text/dates.md  [on-trial]  Dates\n")
        and "Dates are YYYY-MM-DD." in note
    )
    assert by_object == fn
    assert missing.startswith(
        "memory.show('memory.text.dates:nope'): no item named 'memory.text.dates:nope'.\n"
        "Expected an item id, a function name, or a note slug, e.g. memory.get(",
    )
    assert where == str(root / "memory" / "__init__.py")


def test_show_explains_a_hidden_item(tmp_path):
    root = _library(
        tmp_path,
        status_of=lambda i: "retired" if i.endswith(":week") else "on-trial",
    )
    out = json.loads(
        _py(
            root,
            "import json, memory\nprint(json.dumps(memory.show('memory.text.dates:week')))",
        ),
    )
    assert out.splitlines()[:2] == [
        "memory.text.dates:week(s)  [retired]",
        "Not in the index: its status is retired.",
    ]
    assert "memory.text.dates:week(" not in (root / "INDEX.md").read_text()


def test_find_ranks_like_v2s_matcher(tmp_path, monkeypatch):
    """Ranking is v2's matcher's for single-parameter shapes; ``Found.param`` names the parameter that matched."""
    value = {"date": "2026-10-09", "zone": "UTC"}
    shapes = {
        DATE: [
            {
                **memory_helper.value_shape({"date": "2025-01-01", "zone": "CET"}),
                "param": "s",
            },
        ],
        "memory.text.parse:tokens": [
            {
                **memory_helper.value_shape(
                    {"date": "2025-01-01", "zone": "CET", "n": 1},
                ),
                "param": "s",
            },
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
    assert all("input" not in e for e in entries)
    # v2's matcher on the same shapes without the parameter key: what v2 would rank
    stripped = [
        {
            **e,
            "input_shapes": [
                {k: v for k, v in d.items() if k != "param"}
                for d in e.get("input_shapes", [])
            ],
        }
        for e in entries
    ]
    monkeypatch.setattr(memory_helper, "_catalog", lambda: {"functions": stripped})
    want = [
        [":".join(f.name.rsplit(".", 1)), f.signature, "s", *list(f)[3:]]
        for f in memory_helper.find(value)
    ]
    assert got == want
    assert [(g[0], g[2], g[3]) for g in got] == [
        (DATE, "s", "exact"),
        ("memory.text.parse:tokens", "s", "structure"),
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
    key-shaped string in it never reaches the export."""
    key = "sk-or-v1-" + "ab" * 32
    mem = Repo.init_bare(tmp_path / "memory")
    a = _commit(mem, {"memory/text/parse.py": PARSE}, f"add parse ({key})")
    hist, _ = lx.item_history(mem, a)
    assert hist["memory.text.parse:words"] == [
        f"{a[:12]} add parse (<redacted:key-shaped>)",
    ]
    out = lx.generated_library_mode(_tree(tmp_path / "lib"), history=(hist, True))
    assert not any(key.encode() in data for data in out.values())


def test_write_generated_still_writes_v2s_files(tmp_path):
    from unify.memory.catalogue import generated, write_generated

    (tmp_path / "env/x").mkdir(parents=True)
    (tmp_path / "env/x/__init__.py").write_text(
        'def f(a):\n    """F."""\n    return a\n',
    )
    assert write_generated(tmp_path) == generated(tmp_path)
    assert (tmp_path / "memory.py").read_bytes() == (
        Path(memory_helper.__file__)
    ).read_bytes()


def test_the_derived_effect_shows_beside_the_status_in_the_index_and_show(tmp_path):
    from unify.memory.item_records import empty_record

    week = "memory.text.dates:week"
    records = {
        DATE: {**empty_record(DATE, "function"), "effect": "read"},
        week: {
            **empty_record(week, "function"),
            "effect": "write",
            "status_reason": "it refused inputs it was not built for in 2 episodes and no call to it has "
            "returned yet; a refusal is not a failure",
        },
    }
    root = _library(tmp_path, records=records)
    index = (root / "INDEX.md").read_text()
    assert (
        "`memory.text.dates:week(s)` — The ISO week of a date. [on-trial, effect: write]"
        in index
    )
    assert "[on-trial, effect: read]" in index
    # an item without a derived effect keeps the plain status
    assert "Split text on whitespace. [on-trial]\n" in index
    out = json.loads(
        _py(root, f"import json, memory\nprint(json.dumps(memory.show({week!r})))"),
    )
    assert out.splitlines()[:2] == [
        "memory.text.dates:week(s)  [on-trial, effect: write]",
        "Status: it refused inputs it was not built for in 2 episodes and no call to it has returned yet; "
        "a refusal is not a failure.",
    ]


def test_show_gives_the_evidence_level_and_the_findings_that_set_it(tmp_path):
    from unify.memory.item_records import empty_record

    week = "memory.text.dates:week"
    reasons = [
        "examples: no call of it returns on a value from a recorded episode",
        "exact assertion: its tests assert no exact value",
    ]
    records = {
        DATE: {**empty_record(DATE, "function"), "level": "reproduces-its-case"},
        week: {**empty_record(week, "function"), "level": "unchecked", "level_reasons": reasons},
    }
    root = _library(tmp_path, records=records)
    index = (root / "INDEX.md").read_text()
    assert "The ISO week of a date. [on-trial, evidence: unchecked]\n" in index
    out = json.loads(
        _py(root, f"import json, memory\nprint(json.dumps(memory.show({week!r})))"),
    )
    assert out.splitlines()[:4] == [
        "memory.text.dates:week(s)  [on-trial, evidence: unchecked]",
        "Evidence level: unchecked, set by:",
        "  " + reasons[0],
        "  " + reasons[1],
    ]
    date = json.loads(
        _py(root, f"import json, memory\nprint(json.dumps(memory.show({DATE!r})))"),
    )
    assert date.splitlines()[:2] == [
        "memory.text.dates:parse_date(s: str) -> str  [on-trial, evidence: reproduces-its-case]",
        "Evidence level: reproduces-its-case",
    ]


def test_show_reads_old_status_and_level_names_in_todays_names(tmp_path):
    """A row written before the names changed (the bytes of an older copy's items.json) shows today's names."""
    root = _library(tmp_path)
    items = json.loads((root / ITEMS_FILE).read_text())
    row = items["items"]["memory.text.dates:week"]
    row.update(status="suspect", status_reason=None, level="verified", level_reasons=[])
    (root / ITEMS_FILE).write_text(json.dumps(items))
    out = json.loads(
        _py(root, "import json, memory\nprint(json.dumps(memory.show('memory.text.dates:week')))"),
    )
    assert out.splitlines()[:3] == [
        "memory.text.dates:week(s)  [retired, evidence: reproduces-its-case]",
        "Not in the index: its status is retired.",
        "Evidence level: reproduces-its-case",
    ]


def test_the_helpers_own_copy_of_the_names_matches_item_states():
    """The generated helper imports only the standard library, so it keeps its own copy of the names."""
    from unify.memory import item_states, library_helper

    assert library_helper._NEW_ITEM_STATUS == item_states.ON_TRIAL
    assert set(library_helper._HIDDEN_STATUSES) == set(item_states.HIDDEN)
    assert library_helper._OLD_STATUSES == item_states.OLD_STATUSES
    assert library_helper._OLD_LEVELS == item_states.OLD_LEVELS


GUARD = '''class MemoryInputError(ValueError):
    pass


def double(x):
    """Double a whole number."""
    if not isinstance(x, int):
        raise MemoryInputError("not an int")
    if x < 0:
        raise ValueError("negative")
    return x * 2


async def later(x):
    """Add one, later."""
    return x + 1
'''
CALLS_PRELUDE = """
import asyncio, inspect, json
events = []
sys.addaudithook(lambda e, a: events.append(list(a)) if e == "memory.call" else None)
import memory
"""


def _guarded(tmp_path: Path) -> Path:
    from tests.memory.test_layout import LIB

    root = _tree(tmp_path / "lib", {**LIB, "memory/text/guard.py": GUARD})
    write_files(root, lx.generated_library_mode(root))
    return root


def test_each_library_call_raises_one_event_with_its_outcome(tmp_path):
    root = _guarded(tmp_path)
    out = _py(
        root,
        CALLS_PRELUDE + """
from memory.text.guard import double, later, MemoryInputError
assert double(2) == 4 and double(2) == 4
try:
    double("a private value")
except MemoryInputError as exc:
    assert str(exc) == "not an int" and type(exc) is MemoryInputError
    tb = exc.__traceback__
    while tb.tb_next is not None:
        tb = tb.tb_next
    assert tb.tb_frame.f_code.co_filename.endswith("guard.py")  # raised where it was, unchanged
try:
    double(-1)
except ValueError as exc:
    assert type(exc) is ValueError and str(exc) == "negative"
assert asyncio.run(later(1)) == 2
assert inspect.iscoroutinefunction(later) and double.__name__ == "double"
assert double.__doc__ == "Double a whole number." and str(inspect.signature(double)) == "(x)"
from memory.text.dates import parse_date
assert parse_date("2026-10-09") == "2026-10-09"
print(json.dumps(events))
""",
    )
    events = json.loads(out)
    item = "memory.text.guard:double"
    assert [(e[0], e[2]) for e in events] == [
        (item, "ok"),
        (item, "ok"),
        (item, "refused"),
        (item, "error"),
        ("memory.text.guard:later", "ok"),
        (
            "memory.text.parse:tokens",
            "ok",
        ),  # a library call made by another library function
        (DATE, "ok"),
    ]
    assert (
        events[0][1] == events[1][1] != events[2][1]
    )  # equal arguments, equal digests
    assert all(len(e[1]) == 16 and int(e[1], 16) >= 0 for e in events)
    assert "private" not in out


def test_a_failed_call_raises_one_failure_event_with_its_type_and_first_line_and_no_values(tmp_path):
    root = _guarded(tmp_path)
    out = _py(
        root,
        """
import json
failed = []
sys.addaudithook(lambda e, a: failed.append(list(a)) if e == "memory.call_failed" else None)
import memory
from memory.text.guard import double, MemoryInputError
assert double(2) == 4
for arg in ("a private value", -1):
    try:
        double(arg)
    except (MemoryInputError, ValueError):
        pass
try:
    memory.show({"private-key-name": "private value"}, )
except Exception:
    pass
print(json.dumps(failed))
""",
    )
    failed = json.loads(out)
    item = "memory.text.guard:double"
    # an ok call raises none; each failed call raises one, after its memory.call event
    assert failed[:2] == [
        [item, "MemoryInputError", "not an int", ""],
        [item, "ValueError", "negative", ""],
    ]
    # a helper's failure carries the arguments' types and sizes, never their values (its message is the
    # exception's own first line, which the harness redacts when it lists it)
    (show,) = failed[2:]
    assert show[:2] == ["memory:show", "ArgumentError"] and show[3] == "(dict[1])", show
    assert show[2].startswith("memory.show({") and show[2].endswith("(dict).") and "\n" not in show[2]
    assert "private" not in json.dumps([e[3] for e in failed])


def test_the_failure_event_keeps_the_first_line_and_shapes_hold_no_values():
    from unify.memory import library_helper

    class Multi(Exception):
        def __str__(self):
            return "\n  first line of it\nsecond line with a value"

    class Unprintable(Exception):
        def __str__(self):
            raise RuntimeError("no text")

    assert library_helper._first_line(Multi()) == "first line of it"
    assert library_helper._first_line(Unprintable()) == ""
    assert library_helper._first_line(ValueError("x" * 500)) == "x" * library_helper.FAILURE_TEXT
    shape = library_helper.args_shape(("a value", [1, 2], {"k": "v"}, 3.5, object()), {"limit": 10, "name": "n"})
    assert shape == "(str[7], list[2], dict[1], float, object; limit=int, name=str[1])"
    assert library_helper.args_shape(tuple(range(10)), {}) == "(" + ", ".join(["int"] * 8) + ", +2)"


def test_the_digest_never_runs_argument_code_and_a_failing_hook_changes_nothing(
    tmp_path,
):
    root = _guarded(tmp_path)
    out = _py(
        root,
        """
import json
def loud(event, args):
    if event == "memory.call":
        raise RuntimeError("hook failed")
sys.addaudithook(loud)
import memory
from memory.text.guard import double, MemoryInputError
ran = []
class Odd:
    def __repr__(self):
        ran.append("repr")
        return "odd"
    def __len__(self):
        ran.append("len")
        return 1
    def __iter__(self):
        ran.append("iter")
        return iter(())
try:
    double(Odd())
except MemoryInputError:
    pass
assert double(3) == 6
assert memory.args_digest((Odd(), [Odd()], {"k": Odd()}), {}) == memory.args_digest((Odd(), [Odd()], {"k": Odd()}), {})
print(json.dumps(ran))
""",
    )
    assert json.loads(out) == []


def test_lookups_work_on_wrapped_functions_and_tests_are_left_unwrapped(tmp_path):
    root = _guarded(tmp_path)
    out = _py(
        root,
        CALLS_PRELUDE + """
from memory.text.guard import double
import memory.text.tests.test_dates as t
assert getattr(double, "__memory_item__") == "memory.text.guard:double"
assert not hasattr(t.test_parse_date, "__memory_item__")
text = memory.show(double)
assert "def double(x):" in text and "[on-trial" in text
assert memory.index("text")
memory.find({"date": "2026-10-09"})
print(json.dumps(events))
""",
    )
    assert json.loads(out) == []  # lookups are not calls


def test_the_harness_import_of_the_helper_installs_no_finder():
    from unify.memory import library_helper

    assert library_helper.__name__ != "memory"
    assert not any(getattr(f, "memory_call_finder", False) for f in sys.meta_path)

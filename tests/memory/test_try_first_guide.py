"""What the actor reads about finding and trying stored procedures: the guide's nudge, the pointer form of the
memory section, and the index line that shows an item's applicability signature.

The guide informs and never forces: it says to look for stored procedures at the start of a request and that
trying a matching one is cheaper than working it out again; it says nothing about checking a task's examples.

The index line's item records are what production writes (:mod:`tests.memory.gate_records`): a pass lands a captured
procedure, the library gate checks it, and the records are ``lifecycle.next_records`` with the gate's verification and
the capture store's landing. Only a landed procedure carries an applicability signature; building the records runs
code confined, so those tests need bubblewrap.
"""

from __future__ import annotations

import atexit
import functools
import json
import re
import shutil
import tempfile
from pathlib import Path

import pytest

from tests.memory.gate_records import Built, Landing, build, needs_box
from tests.memory.test_episodes import _ep
from tests.memory.test_layout import LIB, _tree
from unify.common import in_code_switch
from unify.memory import library_export as lx
from unify.memory import library_prompts as lp
from unify.memory.catalogue import write_files
from unify.memory.episodes import Action

# ==== ASSUMED NAMES (the integrator adapts these to the code, here only) ======================================
#: The guide texts the actor reads (``library_prompts``): with tools in code off, and in code (which name the
#: library's calls without their arguments).
OFF_GUIDES = ("GUIDE_LIBRARY", "GUIDE_POINTER")
IN_CODE_GUIDES = ("GUIDE_LIBRARY_IN_CODE", "GUIDE_POINTER_IN_CODE", "GUIDE_LIBRARIAN")
GUIDES = OFF_GUIDES + IN_CODE_GUIDES
#: How the guides with tools in code off name the two helpers.
MATCH_CALL = "memory.match("
RUN_CALL = "memory.run("
#: How the in-code guides name the calls that find and open an item: by name, with no arguments. ``memory.ask`` is
#: the librarian, so only the librarian's guide names it.
ASK_NAME = "`memory.ask`"
GET_NAME = "`memory.get`"
ITEMS_NAME = "`memory.items`"
#: The opening's library sentence for the actor (``unify/actor/in_code_tools.py``, ``opening``): the pointer
#: section under tools in code must not say it again.
OPENING_MEMORY_LINE = "The library is importable in your cells as `memory`."
#: The line that follows an item's line in ``INDEX.md`` with its applicability signature (design I.4.2).
INPUT_LINE = re.compile(r"^\s*input: (?P<shape>.+)$")
#: The phrase in the index head that says what the ``input:`` line holds.
INDEX_HEAD_INPUT = "`input:`"


def _export(root: Path) -> str:
    """``INDEX.md`` of a copy at *root* of the library the gate checked, exported with the item records production
    writes for it: a landed procedure's record carries its capture's applicability signature, which the export reads
    through ``saved_against_of`` (with the one identifier rule applied as the export applies it)."""
    built = _built()
    _tree(root, built.files)
    files = lx.generated_library_mode(root, records=built.records, saved_against=lx.saved_against_of(built.records))
    write_files(root, files)
    return files["INDEX.md"].decode("utf-8")


# ==============================================================================================================

DATE = "memory.text.dates:parse_date"
WEEK = "memory.text.dates:week"
ALIAS = "3f2c8a10-9b7e-4c51-a2d4-6e0f1b9c7d35"
#: The procedure the pass lands from its capture, as ``land_capture`` names it.
STAMP = "memory.stamps.procedures:stamp"
#: What its episode was given: a date and its zone, and rows under a key that is a printed identifier.
RECORDED = {"date": "2026-10-09", "zone": "UTC", ALIAS: [{"item": "case-0093", "amount": 3}]}
STAMP_PROCEDURE = '''"""Write a recorded date with its zone."""

import json


def read_input(text):
    """The JSON object of the request."""
    return json.loads(text)


def run(value):
    """The date and its zone, as one stamp."""
    return value["date"] + " " + value["zone"]
'''
STAMP_SUMMARY = (
    "## Asked\nWrite the order's date with its zone.\n## Approach\n- Read the JSON and joined the date and zone.\n"
    "## Worked\n- the stamp\n## Failed\nnone\n## Outcome\nnot shown\n"
)
#: The library stored before the pass: the layout tests' library, its note without the link to a function it lacks
#: (the gate refuses a library with a dangling link).
BASE = {**LIB, "notes/text/dates.md": LIB["notes/text/dates.md"].replace(", memory.text.dates:gone", "")}


@functools.lru_cache(maxsize=1)
def _built() -> Built:
    """The pass that lands the capture of episode ``ep-stamp`` as :data:`STAMP`, the gate's check and the item
    records; once per process."""
    root = Path(tempfile.mkdtemp(prefix="try-first-world-"))
    atexit.register(shutil.rmtree, root, True)
    request = json.dumps(RECORDED)
    episode = _ep(
        episode_id="ep-stamp",
        request=[request],
        transcript=[{"role": "user", "content": request}],
        cells=[],
        actions=[
            Action(-1, "dialogue:user", "answer", ["2026-10-09 UTC"], {}, "Accepted.", "ok", "unknown", kind="dialogue"),
        ],
    )
    return build(
        root,
        episodes={"ep-stamp": episode},
        base=BASE,
        landings=[Landing("ep-stamp", STAMP_PROCEDURE, STAMP_SUMMARY, "stamps", "stamp")],
    )


def _guide(name: str) -> str:
    return getattr(lp, name)


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.:;])\s+|\n+", text) if s.strip()]


# --- the guide's nudge ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", OFF_GUIDES)
def test_the_guide_names_match_and_run(name):
    text = _guide(name)
    assert MATCH_CALL in text and RUN_CALL in text


@pytest.mark.parametrize("name", IN_CODE_GUIDES)
def test_the_in_code_guide_names_its_calls_without_arguments(name):
    text = _guide(name)
    assert GET_NAME in text and ITEMS_NAME in text
    assert (ASK_NAME in text) is (name == "GUIDE_LIBRARIAN"), "memory.ask is named only where the librarian answers"
    assert not re.search(r"memory\.[a-z_]+\(", text), "a library call shown with its arguments"
    assert MATCH_CALL not in text and RUN_CALL not in text
    assert lp.MEMORY_HELP_LINE in text


@pytest.mark.parametrize("name", GUIDES)
def test_the_guide_nudges_to_look_for_stored_procedures_at_the_start_of_a_request(name):
    text = _guide(name)
    hits = [
        s
        for s in _sentences(text)
        if re.search(r"\b(start|begin\w*)\b", s, re.I)
        and re.search(r"\brequest\b", s, re.I)
        and re.search(r"stored procedure|memory\.(?:match|ask|items)", s, re.I)
    ]
    assert hits, "no sentence says to look for stored procedures at the start of a request"


@pytest.mark.parametrize("name", GUIDES)
def test_the_guide_says_trying_a_matching_procedure_is_cheaper_than_working_it_out_again(name):
    text = _guide(name)
    hits = [
        s
        for s in _sentences(text)
        if re.search(r"\bcheaper\b|\bcosts?\b", s, re.I) and re.search(r"\bagain\b|\brediscover", s, re.I)
    ]
    assert hits, "no sentence compares the cost of trying a stored procedure with working it out again"


@pytest.mark.parametrize("name", GUIDES)
def test_the_guide_says_nothing_about_a_tasks_examples_or_any_benchmark(name):
    text = _guide(name)
    assert lp.example_checks(text) == []
    assert lp.benchmark_words(text) == []
    assert not re.search(r"\bexamples?\b", text, re.I)
    assert not re.search(r"input[- ]output pairs?|given examples", text, re.I)


@pytest.mark.parametrize("name", GUIDES)
def test_the_nudge_informs_and_never_forces(name):
    """Trying a stored procedure is optional: the sentences about ``match`` and ``run`` (in code: ``ask`` and ``get``)
    require nothing, and the guide still says the library is optional."""
    text = _guide(name)
    about = [
        s
        for s in _sentences(text)
        if any(
            word in s
            for word in ("memory.match", "memory.run", "memory.ask", "memory.items", "memory.get", "stored procedure")
        )
    ]
    assert about
    for s in about:
        assert not re.search(r"\b(must|always|required?|never skip|do not answer)\b", s, re.I), s
    assert "optional" in text


@pytest.mark.parametrize("name", GUIDES)
def test_the_guide_never_tells_the_agent_to_store_or_avoid_ids_or_answers(name):
    text = _guide(name).lower()
    for phrase in ("task id", "store the answer", "store answers"):
        assert phrase not in text


def test_the_in_code_guide_is_what_the_actor_reads_with_tools_in_code(monkeypatch):
    monkeypatch.setattr(in_code_switch, "tools_in_code", lambda: True)
    assert lp.guide_library() == lp.GUIDE_LIBRARY_IN_CODE and lp.guide_pointer() == lp.GUIDE_POINTER_IN_CODE
    assert GET_NAME in lp.guide_library() and ASK_NAME not in lp.guide_library() and MATCH_CALL not in lp.guide_library()
    monkeypatch.setattr(in_code_switch, "tools_in_code", lambda: False)
    assert MATCH_CALL in lp.guide_library() and RUN_CALL in lp.guide_library()


# --- the pointer section -------------------------------------------------------------------------------------


def _copies(tmp_path: Path) -> list[Path]:
    """Two copies of the library of different sizes: an empty one and one with many functions and notes."""
    empty = tmp_path / "empty"
    empty.mkdir()
    write_files(empty, lx.generated_library_mode(empty))
    files = dict(LIB)
    for i in range(40):
        files[f"memory/extra/m{i:02d}.py"] = f'def f{i}(x):\n    """Item {i}."""\n    return x\n'
    big = _tree(tmp_path / "big", files)
    write_files(big, lx.generated_library_mode(big))
    return [empty, big]


@pytest.mark.parametrize("in_code", [True, False])
def test_the_pointer_section_has_the_same_bytes_at_any_library_size(tmp_path, monkeypatch, in_code):
    from unify.memory.integration.prompt import render_memory_library_mode

    monkeypatch.setattr(in_code_switch, "tools_in_code", lambda: in_code)
    texts = [render_memory_library_mode(c, actor_index="pointer")[0] for c in _copies(tmp_path)]
    assert texts[0] == texts[1]
    assert texts[0]  # shown even for an empty library
    assert "memory.extra" not in texts[0] and "f00" not in texts[0]


@pytest.mark.parametrize("in_code", [True, False])
def test_the_pointer_section_names_match_and_run_and_says_memory_is_bound(tmp_path, monkeypatch, in_code):
    from unify.memory.integration.prompt import render_memory_library_mode

    monkeypatch.setattr(in_code_switch, "tools_in_code", lambda: in_code)
    text = render_memory_library_mode(_copies(tmp_path)[1], actor_index="pointer")[0]
    if in_code:
        assert ITEMS_NAME in text and GET_NAME in text and ASK_NAME not in text
        assert MATCH_CALL not in text and RUN_CALL not in text
    else:
        assert MATCH_CALL in text and RUN_CALL in text
    assert re.search(r"\bmemory\b`?[^.\n]*\balready\b|\balready\b[^.\n]*`memory`", text), text
    assert lp.example_checks(text) == [] and lp.benchmark_words(text) == []


def test_under_tools_in_code_the_pointer_section_follows_the_shared_opening_and_repeats_none_of_it(
    tmp_path,
    monkeypatch,
):
    from types import SimpleNamespace

    from unify.common import tool_registry as R
    from unify.memory.integration import hooks
    from unify.memory.integration.prompt import render_memory_library_mode

    monkeypatch.setattr(in_code_switch, "tools_in_code", lambda: True)
    section = render_memory_library_mode(_copies(tmp_path)[1], actor_index="pointer")[0]
    opening = R.preamble(
        "actor",
        {"environment": "Your box: /workspace and a private /tmp are writable.", "memory": OPENING_MEMORY_LINE},
    )
    # nothing the opening says is said again in the section
    for sentence in _sentences(opening):
        if len(sentence) >= 30:
            assert sentence not in section, sentence
    assert OPENING_MEMORY_LINE not in section
    # the opening already says the library is importable as `memory`; the section says it is bound, not again
    assert "importable in your cells as `memory`" not in section
    assert "## How you work here" not in section
    # the section comes last, after the opening and the role's text
    monkeypatch.setattr(hooks, "enabled", lambda: True)
    monkeypatch.setattr(hooks, "_run", lambda: SimpleNamespace(index=section))
    full = hooks.system_prompt(opening + "\n\n# Role\nYou act for the user.\n")
    assert full.startswith(opening)
    assert full.endswith(section) and full.count(section) == 1


def test_the_default_actor_index_stays_inline():
    from unify.memory.integration.switch import parse_library_actor_index
    from unify.settings import SETTINGS

    assert parse_library_actor_index("") == "inline"
    assert parse_library_actor_index(getattr(SETTINGS, "UNIFY_MEMORY_LIBRARY_ACTOR_INDEX", "") or "") == "inline"


# --- the index line ------------------------------------------------------------------------------------------


def _input_lines(index: str, item: str) -> list[str]:
    """The ``input:`` lines that belong to *item*'s entry (its line and the indented lines under it)."""
    module, name = item.split(":")
    lines = index.splitlines()
    out: list[str] = []
    for i, line in enumerate(lines):
        if line.startswith(f"- `{module}:{name}("):
            for nxt in lines[i + 1 :]:
                if not nxt.strip() or nxt.startswith(("- ", "## ")):
                    break
                m = INPUT_LINE.match(nxt)
                if m:
                    out.append(m.group("shape"))
    return out


@needs_box
def test_an_items_index_line_shows_its_applicability_signature(tmp_path):
    index = _export(tmp_path / "lib")
    assert isinstance(_built().records[STAMP]["signature"], dict)  # carried from its capture by the landing
    shown = _input_lines(index, STAMP)
    assert len(shown) == 1
    assert "date" in shown[0] and "zone" in shown[0]
    # a function that was never landed has no recorded input, so no such line
    for item in (DATE, WEEK):
        assert _built().records[item]["signature"] is None and _input_lines(index, item) == []
    assert INDEX_HEAD_INPUT in index.split("\n## ", 1)[0]  # the head says what the line holds


@needs_box
def test_the_index_with_signatures_is_byte_stable(tmp_path):
    a = _export(tmp_path / "a")
    b = _export(tmp_path / "b")
    assert a == b
    assert a == _export(tmp_path / "a")


@needs_box
def test_no_identifier_or_recorded_value_appears_in_a_signature_line(tmp_path):
    """A signature is structure: no recorded value, and no printed identifier even where one was a key."""
    root = tmp_path / "lib"
    index = _export(root)
    shown = "\n".join(_input_lines(index, STAMP))
    assert shown
    assert ALIAS not in index and "case-0093" not in index and "2026-10-09 UTC" not in index
    assert str(root) not in index

r"""Symbolic: ``UNIFY_ESCAPE_DRIFT_CHECK`` refuses a library write that retypes a session literal with one extra escape.

On the lean-all office attempts of 6 Oct (research artifact
``runtime-20261006/escaped-newline-v1/DIAGNOSIS.md``) the storage review
copied code from the session's own cells into
``FunctionManager_add_functions`` with every backslash doubled: the cell's
``''.join(n+'\n' for n in qual)`` was stored as ``"".join(name + "\\n" ...)``
and ``r'\bERROR\b'`` as ``r"\\bERROR\\b"``. The harness stores the source
byte for byte, so the function wrote a literal backslash-n into every later
output file and its regex matched nothing; 17 failed sessions used such a
function. 16 of 41 review writes with an escape doubled it; 0 of 74 task
cells with an escape did.

With the switch on, while a storage review runs, a string literal in the
source being stored (plain, raw, or an f-string's text) is refused when
removing one level of backslash escaping from it gives a literal of the
session's own code cells and the literal itself is in none of them (nor is
the one-level-less value elsewhere in the same source). The refusal names
both spellings; nothing else changes. Without the session's cells, or with
the switch off, everything is stored as shipped. No model is called except
the scripted one in the last test.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from tests import cache_discipline_helpers as h
from tests.helpers import _handle_project
from unify.function_manager import escape_drift as ed
from unify.function_manager.function_manager import FunctionManager
from unify.settings import SETTINGS

# The session's own working cell (source text, as the model typed it).
CELL = r"""names = ['Bruno Silva', 'Chloe Martin']
Path('overtime.txt').write_text(''.join(n+'\n' for n in names), encoding='utf-8')
"""
CELL_RE = r"""import re
text = open('app.log').read()
codes = sorted(set(re.findall(r'\bE\d+\b', text)))
"""
CELL_PATH = r"""root = 'C:\\temp'
print(root)
"""

# What the review stores: the same code, one escape level up (doubled) ...
DOUBLED = r"""def write_names(names, output_path):
    from pathlib import Path
    Path(output_path).write_text("".join(name + "\\n" for name in names), encoding="utf-8")
"""
# ... and as the cell had it.
CLEAN = r"""def write_names(names, output_path):
    from pathlib import Path
    Path(output_path).write_text("".join(name + "\n" for name in names), encoding="utf-8")
"""
DOUBLED_RE = r"""def error_codes(text):
    import re
    return sorted(set(re.findall(r"\\bE\\d+\\b", text)))
"""
CLEAN_RE = r"""def error_codes(text):
    import re
    return sorted(set(re.findall(r"\bE\d+\b", text)))
"""
DOUBLED_F = r"""def lines(names):
    return "".join(f"{name}\\n" for name in names)
"""
WINDOWS = r"""def temp_root():
    return 'C:\\temp'
"""
# Escaping newlines on purpose: both levels are in the source.
ESCAPER = r"""def one_line(text):
    return text.replace("\n", "\\n")
"""


def _call(call_id: str, code: str) -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": "execute_code",
                    "arguments": json.dumps({"code": code, "language": "python"}),
                },
            },
        ],
    }


def _trajectory(*cells: str) -> list[dict]:
    out: list[dict] = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "Write the names to overtime.txt."},
    ]
    for i, code in enumerate(cells):
        out.append(_call(f"c{i}", code))
        out.append({"role": "tool", "tool_call_id": f"c{i}", "content": "ok"})
    out.append({"role": "assistant", "content": "Done."})
    return out


@pytest.fixture
def drift(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_STORE_VERIFY", "")


def _add(source: str, *cells: str) -> str:
    fm = FunctionManager(include_primitives=False)
    name = source.split("def ", 1)[1].split("(", 1)[0]
    with ed.reviewing(_trajectory(*cells)):
        return fm.add_functions(implementations=[source], raise_on_error=False)[name]


def _stored(name: str):
    fm = FunctionManager(include_primitives=False)
    return fm.list_functions(include_implementations=True).get(name)


# ── the rule ────────────────────────────────────────────────────────────


def test_the_literals_of_a_source_include_raw_and_f_string_text():
    found = {(lit.value, lit.raw) for lit in ed.literals(CELL_RE + DOUBLED_F)}
    assert ("\\bE\\d+\\b", True) in found
    assert ("\\n", False) in found  # the f-string's text after the field
    assert ed.literals("this is not python(") == []


def test_one_level_less_is_one_level_of_backslash_escaping():
    plain = lambda v: ed.one_level_less(ed.Literal(v, raw=False, text=""))
    raw = lambda v: ed.one_level_less(ed.Literal(v, raw=True, text=""))
    assert plain("\\n") == "\n"
    assert plain("\\r\\n") == "\r\n"
    assert plain("a\\tb") == "a\tb"
    assert plain("\\\\d") == "\\d"
    assert plain("\\x41\\u00e9") == "Aé"
    assert plain("\\d") == "\\d"  # not an escape: left as it is
    assert raw("\\\\bERROR\\\\b") == "\\bERROR\\b"
    assert raw("\\n") == "\\n"  # a raw single backslash is not doubled


def test_the_offline_traces_are_flagged_with_both_spellings():
    seen = ed.cell_literals([CELL, CELL_RE])
    assert ed.drifted(DOUBLED, seen) == [('"\\\\n"', "'\\n'")]
    assert ed.drifted(DOUBLED_RE, seen) == [('r"\\\\bE\\\\d+\\\\b"', "r'\\bE\\d+\\b'")]
    assert ed.drifted(CLEAN, seen) == []
    assert ed.drifted(CLEAN_RE, seen) == []


# ── add_functions while a review runs ────────────────────────────────────


@_handle_project
def test_a_doubled_newline_is_refused_and_named(drift):
    status = _add(DOUBLED, CELL)
    assert status == (
        "error: 'write_names' was not stored, because its source writes "
        "`\"\\\\n\"` where the session's own code wrote `'\\n'` (one extra "
        "backslash at each escape)."
    ), status
    assert _stored("write_names") is None


@_handle_project
def test_the_cells_own_newline_is_stored(drift):
    assert _add(CLEAN, CELL) == "added"
    assert _stored("write_names") is not None


@_handle_project
def test_a_doubled_raw_regex_is_refused(drift):
    status = _add(DOUBLED_RE, CELL_RE)
    assert status.startswith("error: 'error_codes' was not stored, because"), status
    assert '`r"\\\\bE\\\\d+\\\\b"`' in status and "`r'\\bE\\d+\\b'`" in status
    assert _add(CLEAN_RE, CELL_RE) == "added"


@_handle_project
def test_a_doubled_f_string_newline_is_refused(drift):
    status = _add(DOUBLED_F, CELL)
    assert status.startswith("error: 'lines' was not stored, because"), status
    assert '`f"{name}\\\\n"`' in status


@_handle_project
def test_a_backslash_the_cells_also_wrote_is_stored(drift):
    # 'C:\\temp' in both: the same value, so nothing drifted.
    assert _add(WINDOWS, CELL_PATH) == "added"
    # A doubled-looking literal the cell itself wrote is the cell's value.
    cell = CELL + "flat = text.replace('\\n', '\\\\n')\n"
    assert _add(ESCAPER, cell) == "added"


@_handle_project
def test_a_source_with_both_levels_is_stored(drift):
    # The cells wrote only '\n'; the function escapes newlines on purpose.
    assert _add(ESCAPER, CELL) == "added"


@_handle_project
def test_without_the_sessions_cells_nothing_is_checked(drift):
    fm = FunctionManager(include_primitives=False)
    assert fm.add_functions(implementations=[DOUBLED]) == {"write_names": "added"}
    # A review whose session ran no code cell.
    assert _add(DOUBLED_RE) == "added"


@_handle_project
def test_raise_on_error_raises_with_the_refusal(drift):
    fm = FunctionManager(include_primitives=False)
    with ed.reviewing(_trajectory(CELL)):
        with pytest.raises(ValueError, match="one extra backslash at each escape"):
            fm.add_functions(implementations=DOUBLED)


@_handle_project
def test_a_patch_that_doubles_an_escape_is_refused(drift, monkeypatch):
    fm = FunctionManager(include_primitives=False)
    with ed.reviewing(_trajectory(CELL)):
        assert fm.add_functions(implementations=[CLEAN]) == {"write_names": "added"}
        out = fm.patch_function(
            name="write_names",
            old='name + "\\n"',
            new='name + "\\\\n"',
            why="separator",
        )
    assert out == {
        "name": "write_names",
        "error": (
            "'write_names' was not stored, because its source writes "
            "`\"\\\\n\"` where the session's own code wrote `'\\n'` (one extra "
            "backslash at each escape)."
        ),
    }
    assert _stored("write_names")["implementation"] == CLEAN


# ── through the actor's storage review ───────────────────────────────────


async def _review(monkeypatch, session_cell: str, review_calls: list) -> list[str]:
    """Run a scripted session whose one cell is *session_cell*; its review makes *review_calls*.

    Returns the review's tool results (as the model read them).
    """
    from unify.actor import code_act_actor as caa
    from unify.actor import review_gate

    monkeypatch.setattr(caa, "_library_counts", lambda *_a, **_k: (1, 0))
    actor = caa.CodeActActor()
    reviewed = {"calls": 0}
    try:
        with h.scripted([]) as provider:

            def _reply():
                system = provider.requests[-1]["messages"][0]["content"]
                if system == review_gate.GATE_SYSTEM_PROMPT:
                    return h.completion(content='{"review": true, "reason": "x"}')
                if system.startswith("You are a skill librarian."):
                    reviewed["calls"] += 1
                    if reviewed["calls"] == 1:
                        return h.completion(calls=review_calls)
                    return h.completion(content="done")
                actor_turns = sum(
                    1
                    for r in provider.requests
                    if not r["messages"][0]["content"].startswith("You are a skill")
                )
                if actor_turns == 1:
                    return h.completion(
                        calls=[("execute_code", {"code": session_cell})],
                    )
                return h.completion(content="Bruno\nChloe")

            provider.replies = [_reply] * 30
            handle = await actor.act("Print the names one per line.", persist=False)
            await asyncio.wait_for(handle.result(), 60)
            await asyncio.wait_for(handle._completion_event.wait(), 60)
    finally:
        await actor.close()
    reviews = [
        r
        for r in provider.requests
        if r["messages"][0]["content"].startswith("You are a skill librarian.")
    ]
    assert len(reviews) >= 2
    return [str(m["content"]) for m in reviews[-1]["messages"] if m["role"] == "tool"]


@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_the_review_is_refused_a_doubled_write_of_its_sessions_cell(
    drift,
    monkeypatch,
):
    session_cell = (
        "names = ['Bruno', 'Chloe']\nprint(''.join(n+'\\n' for n in names))\n"
    )
    results = await _review(
        monkeypatch,
        session_cell,
        [("FunctionManager_add_functions", {"implementations": [DOUBLED]})],
    )
    results = [t for t in results if "write_names" in t]
    assert any("one extra backslash at each escape" in t for t in results), results
    assert _stored("write_names") is None


# ── with UNIFY_STORE_FROM_SESSION: a name stores the cell's own source ────

# The session defines the function in a cell and runs it.
SESSION_DEF = r"""def join_names(names):
    return "".join(name + "\n" for name in names)
"""
SESSION_CELL = SESSION_DEF + "\nprint(join_names(['Bruno', 'Chloe']))\n"
# The same function retyped by the review, one escape level up.
RETYPED = SESSION_DEF.replace('"\\n"', '"\\\\n"')


@pytest.fixture
def by_name(drift):
    return None


def test_the_pair_sources_are_one_escape_level_apart():
    assert '"\\n"' in SESSION_DEF and '"\\\\n"' in RETYPED
    assert ed.drifted(RETYPED, ed.cell_literals([SESSION_CELL]))
    assert ed.drifted(SESSION_DEF, ed.cell_literals([SESSION_CELL])) == []


@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_both_on_a_function_stored_by_name_is_the_cells_and_passes(
    by_name,
    monkeypatch,
):
    results = await _review(
        monkeypatch,
        SESSION_CELL,
        [("FunctionManager_add_functions", {"implementations": ["join_names"]})],
    )
    assert any('"join_names": "added"' in t for t in results), results
    assert not any("one extra backslash" in t for t in results), results
    assert _stored("join_names")["implementation"] == SESSION_DEF


@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_both_on_a_retyped_source_with_one_more_escape_is_refused(
    by_name,
    monkeypatch,
):
    results = await _review(
        monkeypatch,
        SESSION_CELL,
        [("FunctionManager_add_functions", {"implementations": [RETYPED]})],
    )
    assert any(
        "'join_names' was not stored, because its source writes "
        '`"\\\\n"` where the session\'s own code wrote `"\\n"`' in t
        for t in results
    ), results
    assert _stored("join_names") is None

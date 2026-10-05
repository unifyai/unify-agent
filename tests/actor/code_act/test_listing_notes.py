"""Symbolic: notes on the shortlist's lines (``UNIFY_LISTING_PROVENANCE``, ``UNIFY_LESSON_STATUS``, ``UNIFY_LISTING_USAGE``).

On the 5 Oct Continual-ARC paper-protocol run the shortlist's first entry,
in every list, was a guidance entry written after a failed first instance,
shown with its first line as if it were a rule, and nothing in the list
said where an entry came from or how its session ended. The switches add,
under request records: an ``origin:`` line on every listed entry (the rare
identifiers its request shares with this one, and that session's outcome
or that it is unknown); a lesson from an unaccepted or unchecked session
listed as unverified without its first line; and a function's calls with
how the sessions of its last three ended. They inform; nothing is hidden or
asked. Embeddings come from the concept fake of ``test_shortlist_lift``;
requests are captured at unillm's transport, so nothing leaves the process.
"""

from __future__ import annotations

import re

import pytest

from tests.actor.code_act.test_shortlist_lift import (  # noqa: F401 (fixture)
    EARLIER,
    GENERIC,
    GENERIC_TITLE,
    ROTATE,
    ROTATE_AGAIN,
    ROTATE_DOC,
    UNRELATED,
    _act,
    _block,
    _entries,
    _in_task,
    _rotate_source,
    _seed,
    _stream,
    computed,
)
from tests.helpers import _handle_project
from unify.actor import library_shortlist as ls
from unify.actor import shortlist_lift
from unify.function_manager import task_origin
from unify.settings import SETTINGS


@pytest.fixture
def switches(monkeypatch):
    def set_(
        *,
        lift="",
        origin=False,
        provenance=False,
        lesson=False,
        usage=False,
        gate="",
        guidance_origin=False,
        origin_provenance=False,
        review_outcome=False,
    ):
        monkeypatch.setattr(SETTINGS, "UNIFY_LIBRARY_SHORTLIST", True)
        monkeypatch.setattr(SETTINGS, "UNIFY_SHORTLIST_LIFT", lift)
        monkeypatch.setattr(SETTINGS, "UNIFY_SHORTLIST_GATE", gate)
        monkeypatch.setattr(SETTINGS, "UNIFY_TASK_ORIGIN", origin)
        monkeypatch.setattr(SETTINGS, "UNIFY_TRY_FIRST", False)
        monkeypatch.setattr(SETTINGS, "UNIFY_LISTING_PROVENANCE", provenance)
        monkeypatch.setattr(SETTINGS, "UNIFY_LESSON_STATUS", lesson)
        monkeypatch.setattr(SETTINGS, "UNIFY_LISTING_USAGE", usage)
        monkeypatch.setattr(SETTINGS, "UNIFY_GUIDANCE_ORIGIN", guidance_origin)
        monkeypatch.setattr(SETTINGS, "UNIFY_ORIGIN_PROVENANCE", origin_provenance)
        monkeypatch.setattr(SETTINGS, "UNIFY_REVIEW_OUTCOME", review_outcome)
        monkeypatch.setattr(SETTINGS, "UNIFY_SIMILAR_REQUEST_IDENTIFIERS", True)
        # Rare identifiers are rare among the stream's logged requests.
        monkeypatch.setattr(
            SETTINGS,
            "UNIFY_SIMILAR_REQUEST_CORPUS",
            "stream" if origin else "",
        )
        monkeypatch.setattr(SETTINGS, "UNIFY_DISCOVERY_GATE", False)
        monkeypatch.setattr(SETTINGS, "UNIFY_LIBRARY_SNAPSHOT", True)
        monkeypatch.setattr(SETTINGS, "UNIFY_BUILTIN_GUIDANCE", False)
        monkeypatch.setattr(SETTINGS, "UNIFY_STORE_CHECK", "")
        monkeypatch.setattr(SETTINGS, "UNIFY_STORE_VERIFY", "")

    return set_


# ── UNIFY_LISTING_PROVENANCE ─────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.timeout(240)
@_handle_project
async def test_provenance_puts_an_origin_line_under_every_listed_entry(
    switches,
    computed,
):
    switches(origin=True, provenance=True, review_outcome=True)
    seed_and_judge = lambda actor: (  # noqa: E731
        _seed(actor, under=ROTATE),
        _in_task(
            ROTATE,
            lambda: task_origin.record_outcome(False, source=task_origin.REVIEW),
        ),
    )
    (again,) = await _stream([ROTATE_AGAIN], seed=seed_and_judge)
    block = _block(again)
    lines = block.splitlines()[1:]
    assert len(lines) == 4  # two entries, each with its origin line
    rejected = "that session's answer was rejected, as judged by its review"
    following = {lines[i].split(" ", 2)[1]: lines[i + 1] for i in (0, 2)}
    assert following["function"] == (
        "  origin: stored while handling a request that also named `p-3d61a`; "
        + rejected
    )
    assert following["guidance"] == (
        "  origin: written while handling a request that also named `p-3d61a`; "
        + rejected
    )
    (other,) = await _stream([UNRELATED])
    assert (
        "  origin: stored while handling another request (no rare identifier in "
        "common); that session's answer was rejected" in _block(other)
    )


@pytest.mark.asyncio
@pytest.mark.timeout(240)
@_handle_project
async def test_provenance_says_when_the_origin_or_outcome_is_unknown(
    switches,
    computed,
):
    switches(origin=True, provenance=True)
    # Stored before the switches recorded anything: no request.
    (first,) = await _stream([ROTATE], seed=lambda actor: _seed(actor))
    assert _block(first).count("  origin: no request recorded; outcome unknown") == 2
    # Stored under this same request, no outcome kept.
    from unify import db

    db.clear()
    (same,) = await _stream([ROTATE_AGAIN], seed=lambda a: _seed(a, under=ROTATE_AGAIN))
    assert (
        _block(same).count(
            "handling this same request; that session's outcome is unknown",
        )
        == 2
    )


@pytest.mark.asyncio
@pytest.mark.timeout(240)
@_handle_project
async def test_provenance_replaces_the_marked_parenthesis(switches, computed):
    switches(origin=True, origin_provenance=True)
    (marked,) = await _stream([ROTATE_AGAIN], seed=lambda a: _seed(a, under=ROTATE))
    assert "(stored while handling a request that also named `p-3d61a`)" in marked
    switches(origin=True, origin_provenance=True, provenance=True)
    (listed,) = await _stream([ROTATE_AGAIN])
    function_line = next(
        ln for ln in _block(listed).splitlines() if ln.startswith("- function")
    )
    assert "(stored while" not in function_line
    assert "  origin: stored while handling a request that also named `p-3d61a`" in (
        listed
    )


@pytest.mark.asyncio
@pytest.mark.timeout(240)
@_handle_project
async def test_provenance_in_the_gated_list(switches, computed):
    gate = "similar_request:0.175"
    switches(origin=True, provenance=True, gate=gate, guidance_origin=True)
    (again,) = await _stream([ROTATE_AGAIN], seed=lambda a: _seed(a, under=ROTATE))
    lines = again[again.index("Stored functions and guidance") :].split("\n\n")[0]
    lines = lines.splitlines()[1:]
    assert lines[0].startswith("- function `rotate_table(table)`")
    assert "[similar_request" in lines[0] and "used 0×]" in lines[0]
    assert lines[1].startswith("  origin: stored while handling a request that")
    assert lines[2].startswith("- guidance ") and "[similar_request" in lines[2]
    assert lines[3].startswith("  origin: written while handling a request that")


@pytest.mark.asyncio
@pytest.mark.timeout(120)
@_handle_project
async def test_listing_switches_refuse_to_start_without_request_records(
    switches,
    computed,
):
    for name in ("provenance", "lesson", "usage"):
        switches(**{name: True})
        with pytest.raises(ValueError, match="UNIFY_TASK_ORIGIN"):
            await _act(ROTATE)


# ── UNIFY_LESSON_STATUS ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("outcome", "status"),
    [
        (False, "(unverified: written after a session whose answer was not accepted)"),
        (None, "(unverified: written in a session whose outcome is unknown)"),
        (True, None),
    ],
)
@pytest.mark.asyncio
@pytest.mark.timeout(240)
@_handle_project
async def test_lesson_status_marks_unverified_guidance_without_its_first_line(
    switches,
    computed,
    outcome,
    status,
):
    switches(origin=True, lesson=True)

    def seed(actor):
        _seed(actor, under=ROTATE)
        if outcome is not None:
            _in_task(ROTATE, lambda: task_origin.record_outcome(outcome))

    (again,) = await _stream([ROTATE_AGAIN], seed=seed)
    line = next(ln for ln in _block(again).splitlines() if ln.startswith("- guidance"))
    if status is None:
        assert line.endswith(f"`{GENERIC_TITLE}`: {GENERIC}")
    else:
        assert line.endswith(f"`{GENERIC_TITLE}` {status}")
        assert GENERIC not in again
    # Functions are not lessons.
    assert f"rotate_table(table)`: {ROTATE_DOC}" in again


@pytest.mark.asyncio
@pytest.mark.timeout(240)
@_handle_project
async def test_lesson_status_leaves_built_in_guidance_as_shipped(
    switches,
    computed,
    monkeypatch,
):
    switches(origin=True, lesson=True)
    monkeypatch.setattr(SETTINGS, "UNIFY_BUILTIN_GUIDANCE", True)
    (first,) = await _stream([ROTATE])
    builtin = [ln for ln in _entries(_block(first)) if ln.startswith("- guidance")]
    assert builtin and not any("unverified" in ln for ln in builtin)


def test_lesson_status_with_several_writers(switches, monkeypatch):
    switches(origin=True, lesson=True)
    row = {
        "metadata": {
            task_origin.REQUESTS_FIELD: [
                task_origin.bounded_text(EARLIER[0]),
                task_origin.bounded_text(EARLIER[1]),
            ],
        },
    }
    assert task_origin.lesson_status(row) == (
        "unverified: written in a session whose outcome is unknown"
    )
    _in_task(EARLIER[0], lambda: task_origin.record_outcome(True))
    assert "outcome is unknown" in task_origin.lesson_status(row)
    _in_task(EARLIER[1], lambda: task_origin.record_outcome(True))
    assert task_origin.lesson_status(row) is None
    _in_task(EARLIER[1], lambda: task_origin.record_outcome(False))
    assert "was not accepted" in task_origin.lesson_status(row)


# ── UNIFY_LISTING_USAGE ──────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.timeout(240)
@_handle_project
async def test_usage_shows_calls_and_how_their_sessions_ended(switches, computed):
    switches(origin=True, usage=True)

    def seed(actor):
        _seed(actor, under=ROTATE)
        fm = actor.function_manager
        row = fm._library_rows()[0]
        # Four calls in three sessions: the first failed, then unknown, then failed twice.
        for request, outcome in (
            (EARLIER[0], False),
            (EARLIER[1], None),
            (EARLIER[2], False),
            (EARLIER[2], False),
        ):
            _in_task(request, lambda: fm._note_function_use(row))
            if outcome is not None:
                _in_task(request, lambda: task_origin.record_outcome(outcome))

    (again,) = await _stream([ROTATE_AGAIN], seed=seed)
    line = next(ln for ln in _block(again).splitlines() if ln.startswith("- function"))
    assert line.startswith(f"- function `rotate_table(table)`: {ROTATE_DOC} [")
    assert line.endswith(
        " [used 4×; of its last 3 recorded calls, 2 ran in a session whose answer "
        "was not accepted, 1 with the outcome unknown]",
    )


def test_usage_note_without_calls_or_records(switches):
    switches(origin=True, usage=True)
    assert task_origin.usage_note("never", 0) == "not called yet"
    assert task_origin.usage_note("old", 7) == (
        "used 7×; the sessions of its calls were not recorded"
    )


def test_usage_records_nothing_while_off(switches):
    from unify.function_manager.function_manager import FunctionManager

    switches(origin=True)
    fm = FunctionManager(include_primitives=False)
    _in_task(ROTATE, lambda: fm.add_functions(implementations=_rotate_source()))
    row = fm._library_rows()[0]
    _in_task(ROTATE, lambda: fm._note_function_use(row))
    path = task_origin.request_log_path()
    if path.exists():
        import sqlite3

        with sqlite3.connect(path) as conn:
            tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master")}
        assert "function_calls" not in tables


# ── composition ──────────────────────────────────────────────────────────


def test_lift_and_notes_with_the_bound_core_list(switches, computed, monkeypatch):
    """``UNIFY_CORE_BIND_LISTED``: the lifted list's functions are bound and the header says how to call."""
    from unify.function_manager.function_manager import FunctionManager
    from unify.guidance_manager.guidance_manager import GuidanceManager

    switches(lift="recent:3", origin=True, provenance=True)
    fm = FunctionManager(include_primitives=False)
    gm = GuidanceManager()
    _in_task(ROTATE, lambda: fm.add_functions(implementations=_rotate_source()))
    _in_task(ROTATE, lambda: gm.add_guidance(title=GENERIC_TITLE, content=GENERIC))
    for request in EARLIER:  # each earlier task start ranks and is kept
        _in_task(request, lambda: ls.shortlist_block(fm, gm, request))
        shortlist_lift.log_request(ls.request_text(request))
    bound: list[list[str]] = []

    def bind(names):
        bound.append(list(names))
        return {name: False for name in names}

    block = _in_task(
        ROTATE_AGAIN,
        lambda: ls.shortlist_block(fm, gm, ROTATE_AGAIN, bind=bind),
    )
    assert block.startswith(ls._LIFT_HEADER_CALL)
    assert ls.CALL_FORM in block.splitlines()[0]
    assert bound == [["rotate_table"]]
    assert (
        "  origin: stored while handling a request that also named `p-3d61a`" in block
    )
    assert ls.shortlisted_names(block) == {
        "functions": ["rotate_table"],
        "guidance": [re.search(r"- guidance (\S+)", block).group(1)],
    }

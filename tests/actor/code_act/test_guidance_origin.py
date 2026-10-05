"""Symbolic: ``UNIFY_GUIDANCE_ORIGIN`` records the request a guidance entry was written for.

The request-gated shortlist (``UNIFY_SHORTLIST_GATE``) scores stored entries
by how close the current request is to the requests they were written for,
and only functions recorded those. ScienceWorld's libraries hold only
guidance, so the gated arm of the 5 Oct lean screen showed no list there at
all and solved 6 of 12 episodes, against 10-11 of 12 for lean variants whose
ungated list showed guidance lines in 11-12 sessions per run. In the
retrieval audit (research artifact retrieval-matching-audit-v1) giving
guidance its origins took the gate on ScienceWorld same-task visits from 0%
to 93% found, at 14% of unrelated visits getting a list.

With the switch a guidance entry added or updated while handling a request
records it as a function does, in a new nullable ``origin`` column that no
guidance read returns, and the gated shortlist scores those entries like
functions. Requests are captured at unillm's transport, so nothing leaves the
process.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3

import pytest

from tests import cache_discipline_helpers as h  # noqa: F401  (scripted transport)
from tests.actor.code_act.test_shortlist_gate import (
    AGAIN,
    FIRST,
    OTHERS,
    UNRELATED,
    _act,
    _first_user,
    _in_task,
    _source,
    embed_calls,  # noqa: F401  (fixture)
)
from tests.helpers import _handle_project
from unify import db
from unify.actor import library_shortlist as ls
from unify.function_manager import task_origin
from unify.guidance_manager.guidance_manager import GuidanceManager
from unify.guidance_manager.types.guidance import Guidance
from unify.settings import ProductionSettings, SETTINGS

GATE = "similar_request:0.175"


@pytest.fixture
def switches(monkeypatch):
    def set_(*, guidance_origin=True, origin=True, gate=GATE, patch=False):
        monkeypatch.setattr(SETTINGS, "UNIFY_GUIDANCE_ORIGIN", guidance_origin)
        monkeypatch.setattr(SETTINGS, "UNIFY_TASK_ORIGIN", origin)
        monkeypatch.setattr(SETTINGS, "UNIFY_TRY_FIRST", False)
        monkeypatch.setattr(SETTINGS, "UNIFY_LIBRARY_SHORTLIST", True)
        monkeypatch.setattr(SETTINGS, "UNIFY_SHORTLIST_GATE", gate)
        monkeypatch.setattr(SETTINGS, "UNIFY_DISCOVERY_GATE", False)
        monkeypatch.setattr(SETTINGS, "UNIFY_LIBRARY_SNAPSHOT", True)
        monkeypatch.setattr(SETTINGS, "UNIFY_BUILTIN_GUIDANCE", False)
        monkeypatch.setattr(SETTINGS, "UNIFY_FUNCTION_PATCH", patch)
        monkeypatch.setattr(SETTINGS, "UNIFY_STORE_CHECK", "")
        monkeypatch.setattr(SETTINGS, "UNIFY_STORE_VERIFY", "")

    return set_


def _origin(guidance_id: int):
    row = db.query_one(
        "SELECT origin FROM guidance WHERE guidance_id = ?",
        (int(guidance_id),),
    )
    return db.loads(row["origin"]) if row["origin"] else None


def _add(gm: GuidanceManager, title: str, content: str) -> int:
    return int(gm.add_guidance(title=title, content=content)["details"]["guidance_id"])


# ── the store ────────────────────────────────────────────────────────────


@_handle_project
def test_an_old_store_gains_an_empty_origin_column(tmp_path, switches):
    switches(guidance_origin=False)
    old = tmp_path / "old-store.sqlite"
    with sqlite3.connect(old) as conn:
        # The schema as shipped before the column existed.
        conn.executescript(db.SCHEMA)
        conn.execute(
            "INSERT INTO guidance (title, content, function_ids, stale_reasons,"
            " created_at) VALUES ('Old note', 'Kept as it was.', '[]', '[]',"
            " '2026-10-01T00:00:00+00:00')",
        )
        columns = {row[1] for row in conn.execute("PRAGMA table_info(guidance)")}
    assert "origin" not in columns
    previous = os.environ.get("UNIFY_STORE_PATH")
    os.environ["UNIFY_STORE_PATH"] = str(old)
    db.reset_store()
    try:
        columns = {row["name"] for row in db.query("PRAGMA table_info(guidance)")}
        assert "origin" in columns
        (entry,) = GuidanceManager().filter()
        assert (entry.title, entry.content) == ("Old note", "Kept as it was.")
        assert _origin(entry.guidance_id) is None
        # Opening again changes nothing.
        db.reset_store()
        assert len(GuidanceManager().filter()) == 1
    finally:
        if previous is None:
            os.environ.pop("UNIFY_STORE_PATH", None)
        else:
            os.environ["UNIFY_STORE_PATH"] = previous
        db.reset_store()


@_handle_project
def test_an_entry_records_each_request_it_was_written_for(switches):
    switches()
    gm = GuidanceManager()
    gid = _in_task(FIRST, lambda: _add(gm, "Doubling", "Double the number."))
    assert _origin(gid) == {
        "origin_tasks": [task_origin.task_key(FIRST)],
        "origin_requests": [task_origin.bounded_text(FIRST)],
    }
    _in_task(
        AGAIN,
        lambda: gm.update_guidance(guidance_id=gid, content="Double it: 2 * x."),
    )
    assert _origin(gid)["origin_requests"] == [
        task_origin.bounded_text(FIRST),
        task_origin.bounded_text(AGAIN),
    ]
    # Outside a keyed task nothing is recorded, and nothing is lost.
    gm.update_guidance(guidance_id=gid, title="Doubling numbers")
    assert len(_origin(gid)["origin_tasks"]) == 2


@_handle_project
@pytest.mark.parametrize("guidance_origin, origin", [(False, True), (True, False)])
def test_off_nothing_is_recorded(switches, guidance_origin, origin):
    switches(guidance_origin=guidance_origin, origin=origin, gate="")
    gm = GuidanceManager()
    gid = _in_task(FIRST, lambda: _add(gm, "Doubling", "Double the number."))
    _in_task(FIRST, lambda: gm.update_guidance(guidance_id=gid, content="2 * x"))
    assert _origin(gid) is None


@_handle_project
def test_no_guidance_read_shows_the_origin(switches, embed_calls):  # noqa: F811
    switches(patch=True)
    gm = GuidanceManager()
    gid = _in_task(FIRST, lambda: _add(gm, "Doubling", "Double the number 21."))
    _in_task(
        FIRST,
        lambda: gm.patch_guidance(
            id_or_title=gid,
            old="21",
            new="22",
            why="the number changed",
        ),
    )
    reads = [
        gm.get_guidance(guidance_id=gid),
        *gm.filter(),
        *gm.search(references={"content": "double"}),
        *gm._rows(gm._scope(None)),
        *gm._shortlist_rows("double the number", 5),
    ]
    for shown in reads:
        text = json.dumps(shown, default=lambda o: o.model_dump(mode="json"))
        assert "origin" not in text, text
    # History, which no read returns, keeps the whole row as it was (as
    # function_history keeps a function's metadata with its origins).
    (history,) = db.query("SELECT previous FROM guidance_history")
    assert db.loads(history["previous"])["origin"]
    # A filter cannot reach it either: reads go through a view without it.
    refused = gm.filter(filter="origin IS NOT NULL")
    assert not any(isinstance(entry, Guidance) for entry in refused)


# ── the gated shortlist ──────────────────────────────────────────────────


def _block(text: str):
    for header in (ls._GATED_HEADER_WITH_GUIDANCE, ls._GATED_HEADER):
        if header in text:
            return text[text.index(header) :].split("\n\n", 1)[0]
    return None


def _seed_guidance(actor):
    gm = actor.guidance_manager
    _in_task(
        FIRST,
        lambda: _add(gm, "Puzzle p-3d61a", "Swap the first and last rows.\nMore."),
    )
    for name, factor, request in OTHERS:
        _in_task(request, lambda: _add(gm, f"Puzzle {name}", f"Scale by {factor}."))


GUIDANCE_LINE = re.compile(
    r"^- guidance \d+ `Puzzle p-3d61a`: Swap the first and last rows\. "
    r"\[similar_request (\d\.\d\d)\]$",
)


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_guidance_stored_for_the_same_puzzle_is_listed(
    switches,
    embed_calls,
):  # noqa: F811
    switches()
    requests, at_start = await _act(AGAIN, seed=_seed_guidance, embed_calls=embed_calls)
    first = _first_user(requests[0])
    block = _block(first)
    assert block is not None, first
    assert block.startswith(ls._GATED_HEADER_WITH_GUIDANCE)
    (line,) = block.splitlines()[1:]
    assert 0.175 <= float(GUIDANCE_LINE.match(line).group(1)) < 1
    assert at_start == []


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_an_unrelated_request_gets_no_list(switches, embed_calls):  # noqa: F811
    switches()
    requests, _ = await _act(UNRELATED, seed=_seed_guidance, embed_calls=embed_calls)
    assert _block(_first_user(requests[0])) is None


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_off_a_guidance_only_library_gets_no_list(
    switches,
    embed_calls,
):  # noqa: F811
    # As shipped with the gate: guidance has no origin, so nothing can pass.
    switches(guidance_origin=False)
    requests, _ = await _act(AGAIN, seed=_seed_guidance, embed_calls=embed_calls)
    assert _block(_first_user(requests[0])) is None


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_an_actor_refuses_it_without_request_records(switches):
    from unify.actor import code_act_actor as caa

    switches(origin=False, gate="")
    actor = caa.CodeActActor()
    try:
        with pytest.raises(ValueError, match="UNIFY_GUIDANCE_ORIGIN"):
            await actor.act(AGAIN, persist=False)
    finally:
        await actor.close()


@_handle_project
def test_functions_and_guidance_share_the_five_places(switches):
    from unify.function_manager.function_manager import FunctionManager

    switches()
    fm, gm = FunctionManager(), GuidanceManager()
    for n in range(4):
        _in_task(
            FIRST,
            lambda n=n: fm.add_functions(implementations=_source(f"f{n}", n + 2)),
        )
    for n in range(3):
        _in_task(FIRST, lambda n=n: _add(gm, f"Note {n}", f"Note number {n}."))
    rows = _in_task(
        FIRST,
        lambda: fm._gated_shortlist_rows(0.175, 5, gm._origin_rows()),
    )
    assert len(rows) == 5
    kinds = [row["kind"] for row in rows]
    assert set(kinds) == {"function", "guidance"}
    guidance = [row for row in rows if row["kind"] == "guidance"]
    assert all(row["similar_request"] == 1.0 for row in rows)
    assert all("metadata" not in row for row in rows)
    # Without functions, only guidance.
    only = _in_task(
        FIRST,
        lambda: fm._gated_shortlist_rows(0.175, 5, gm._origin_rows(), functions=False),
    )
    assert [row["kind"] for row in only] == ["guidance"] * 3
    assert guidance


@_handle_project
def test_the_review_scores_functions_as_the_shortlist_with_guidance(
    switches,
    monkeypatch,
):
    """``UNIFY_REVIEW_GENERALISE`` shows the score the gated shortlist showed.

    Guidance origins weigh the requests' words in the shortlist, so the
    review's list of functions stored for similar requests weighs them too.
    """
    from unify.actor import code_act_actor as caa
    from unify.function_manager.function_manager import FunctionManager

    switches()
    monkeypatch.setattr(SETTINGS, "UNIFY_REVIEW_GENERALISE", True)
    fm, gm = FunctionManager(), GuidanceManager()
    _in_task(FIRST, lambda: fm.add_functions(implementations=_source("double", 2)))
    for name, factor, request in OTHERS:
        _in_task(
            request,
            lambda: fm.add_functions(implementations=_source(name, factor)),
        )
        _in_task(request, lambda: _add(gm, f"About {name}", f"Use {name}."))
    _in_task(UNRELATED, lambda: _add(gm, "Dinners", "Plan the week first."))
    gated = _in_task(
        AGAIN,
        lambda: fm._gated_shortlist_rows(0.175, 5, gm._origin_rows()),
    )
    shown = {r["name"]: r["similar_request"] for r in gated if r["kind"] == "function"}
    assert "double" in shown
    reviewed = _in_task(
        AGAIN,
        lambda: fm._similar_request_functions(0.175, 5, gm._origin_rows()),
    )
    # Guidance takes some of the shortlist's places; the functions both
    # list carry the same score.
    scores = {r["name"]: r["similar_request"] for r in reviewed}
    assert {name: scores.get(name) for name in shown} == shown
    # Weighed over the functions alone, the score would differ.
    alone = _in_task(AGAIN, lambda: fm._similar_request_functions(0.175, 5))
    assert {r["name"]: r["similar_request"] for r in alone}["double"] != shown["double"]
    # The review's note asks with the guidance rows, and only while the
    # switch is on.
    asked = []

    class _Fake:
        def _similar_request_functions(self, *args):
            asked.append(args)
            return []

    _in_task(AGAIN, lambda: caa._review_generalise_note(_Fake(), gm))
    assert len(asked[-1]) == 3 and asked[-1][2] == gm._origin_rows()
    monkeypatch.setattr(SETTINGS, "UNIFY_GUIDANCE_ORIGIN", False)
    _in_task(AGAIN, lambda: caa._review_generalise_note(_Fake(), gm))
    assert len(asked[-1]) == 2


def test_the_guidance_line_is_labelled():
    row = {
        "guidance_id": 7,
        "title": "Heating water",
        "content": "Use the stove.\nThen wait.",
        "similar_request": 0.312,
    }
    assert ls._gated_guidance_line(row) == (
        "- guidance 7 `Heating water`: Use the stove. [similar_request 0.31]"
    )


def test_the_setting_defaults_off_and_parses_booleans():
    assert ProductionSettings().UNIFY_GUIDANCE_ORIGIN is False
    assert ProductionSettings(UNIFY_GUIDANCE_ORIGIN="1").UNIFY_GUIDANCE_ORIGIN is True

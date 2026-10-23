"""The library's answer at a request's start, and the one-primary answer form.

The request's librarian is asked about the request once, before the actor's first turn (the same ask as
``memory.ask()``), and its answer ends the actor's first message: the entry that fits best with its call, then at
most one fallback, labelled for when the first fails, or that nothing fits; bounded in size; nothing is asked of an
empty library. The librarian's answer names at most two entries: never a menu.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from unify.memory import librarian as L
from unify.memory.library_helper import ITEMS_FILE


def _copy(tmp_path, items: dict):
    copy = tmp_path / "copy"
    (copy / ITEMS_FILE).parent.mkdir(parents=True, exist_ok=True)
    (copy / ITEMS_FILE).write_text(json.dumps({"items": items}))
    return copy


def _state(asked: list, answer: dict):
    async def opener():
        raise AssertionError("the stub asks without a session")

    return SimpleNamespace(request_text="Sort the rows by id.", open_librarian=opener, librarian=None, asked=asked,
                           answer=answer)  # fmt: skip


def _run(tmp_path, monkeypatch, items: dict, answer: dict):
    from unify.actor import tool_handlers as H
    from unify.memory.integration import request as R
    from unify.memory.writer_tools import Reply

    asked: list = []

    async def ask_library(ctx, args):
        asked.append(args["question"])
        return Reply(json.dumps(answer))

    monkeypatch.setattr(H, "ask_library", ask_library)
    monkeypatch.setattr(R, "current", lambda: SimpleNamespace(copy_root=_copy(tmp_path, items)))
    return asyncio.run(L.request_start_block(_state(asked, answer))), asked


def test_the_block_names_the_primary_with_its_call_and_one_fallback_for_when_it_fails(tmp_path, monkeypatch):
    answer = {"fits": [
        {"id": "capture:" + "a" * 16, "call": "memory.get('capture:" + "a" * 16 + "')(memory.request())",
         "why": "it re-runs sorting rows by an integer column"},
        {"id": "memory.rows.sort:by_id", "call": "memory.get('memory.rows.sort:by_id')(rows)", "why": "a general sort"},
    ]}  # fmt: skip
    block, asked = _run(tmp_path, monkeypatch, {"memory.rows.sort:by_id": {"kind": "function"}}, answer)
    assert asked == ["Sort the rows by id."]  # the request itself, as memory.ask() asks it
    lines = block.splitlines()
    assert lines[0] == L.START_HEAD
    assert lines[1].startswith("- capture:") and "call: memory.get(" in lines[2]
    assert lines[3].startswith("- only if the first fails: memory.rows.sort:by_id")
    assert len(block) <= L.START_BLOCK_CHARS


def test_nothing_fits_says_so_and_an_empty_library_asks_nothing(tmp_path, monkeypatch):
    block, asked = _run(tmp_path, monkeypatch, {"notes/a/b.md": {"kind": "note"}}, {"fits": [], "nearest": []})
    assert block.endswith("nothing in the library fits this request.") and asked
    block, asked = _run(tmp_path, monkeypatch, {}, {"fits": []})
    assert block == "" and asked == []


def test_the_block_stays_within_its_bound():
    long = {"fits": [{"id": "x" * 3000, "call": "memory.get('x')()", "why": "w" * 5000}] * 2}
    assert len(L.start_block(long)) <= L.START_BLOCK_CHARS


def test_an_answer_names_at_most_a_primary_and_one_fallback():
    ids = ["memory.a.b:c", "memory.a.b:d", "memory.a.b:e"]
    fits = [{"id": i, "call": f"memory.get({i!r})()", "why": "w"} for i in ids]
    got, why = L.check_answer({"fits": fits}, ids)
    assert got is None and "at most 2" in why
    got, _ = L.check_answer({"fits": fits[:2]}, ids)
    assert got == {"fits": fits[:2]}


def test_an_answer_is_bounded_in_all_including_its_reasons_and_nearest():
    ids = [f"memory.a.b:f{i}" for i in range(6)]
    long = "w " * 3000
    got, _ = L.check_answer({"fits": [], "nearest": [{"id": i, "differs": long} for i in ids]}, ids)
    assert len(got["nearest"]) == L.MAX_NEAREST and len(json.dumps(got)) <= L.ANSWER_CHARS
    fits = [{"id": i, "call": f"memory.get({i!r})()", "why": long} for i in ids[:2]]
    got, _ = L.check_answer({"fits": fits}, ids)
    assert [f["id"] for f in got["fits"]] == ids[:2] and len(json.dumps(got)) <= L.ANSWER_CHARS
    assert all(f["call"] == f"memory.get({f['id']!r})()" for f in got["fits"])  # calls are never cut

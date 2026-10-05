"""Symbolic: ``UNIFY_ENTRY_RECORD`` (one record for both kinds) and ``UNIFY_SEARCH_IDENTIFIERS``.

Every piece of evidence the library kept was built for functions only:
guidance recorded no use, its origin was kept only under a separate switch,
and a note written after a failed session read like one that had worked. One
record now covers both kinds: the requests an entry was written for, how
those sessions ended, how often other requests like it came, and the
sessions that called, read or (as the storage review judged) relied on it.
Links between functions and notes are many to many, in a table of their own.
A search whose query names an identifier finds the entries recorded under a
request naming it.
"""

from __future__ import annotations

import pytest

from tests.helpers import _handle_project
from unify import db
from unify.actor import code_act_actor as caa
from unify.function_manager import entry_links, entry_record, task_origin
from unify.function_manager.function_manager import FunctionManager
from unify.guidance_manager.guidance_manager import GuidanceManager
from unify.settings import SETTINGS

REQUEST = "Rename the files in batch inv_2024q3 to ISO dates and zip them."
AGAIN = "Rename the files in batch inv_2024q3 to ISO dates, zip them, upload the zip."
OTHER = "Summarise the customer feedback from last week in three bullets."


@pytest.fixture(autouse=True)
def fake_embeddings(monkeypatch):
    """Searches rank with a fake embedder: nothing reaches a model."""
    import hashlib

    import numpy as np

    import unify.common.embeddings as embeddings
    import unify.common.semantic_search as semantic_search

    def fake(texts):
        out = []
        for text in texts:
            seed = int(hashlib.sha256(str(text).encode()).hexdigest()[:8], 16)
            v = np.random.default_rng(seed).standard_normal(16).astype(np.float32)
            out.append(v / np.linalg.norm(v))
        return np.stack(out)

    monkeypatch.setattr(embeddings, "embed", fake)
    monkeypatch.setattr(semantic_search, "embed", fake)


@pytest.fixture
def switches(monkeypatch):
    def set_(*, record=True, origin=True, identifiers=False, review_outcome=False):
        monkeypatch.setattr(SETTINGS, "UNIFY_ENTRY_RECORD", record)
        monkeypatch.setattr(SETTINGS, "UNIFY_TASK_ORIGIN", origin)
        monkeypatch.setattr(SETTINGS, "UNIFY_TRY_FIRST", False)
        monkeypatch.setattr(SETTINGS, "UNIFY_SEARCH_IDENTIFIERS", identifiers)
        monkeypatch.setattr(SETTINGS, "UNIFY_REVIEW_OUTCOME", review_outcome)
        monkeypatch.setattr(SETTINGS, "UNIFY_SIMILAR_REQUEST_IDENTIFIERS", True)
        monkeypatch.setattr(SETTINGS, "UNIFY_BUILTIN_GUIDANCE", False)
        monkeypatch.setattr(SETTINGS, "UNIFY_STORE_CHECK", "")
        monkeypatch.setattr(SETTINGS, "UNIFY_STORE_VERIFY", "")

    return set_


def _in_task(request, fn):
    token = task_origin.enter(request)
    try:
        return fn()
    finally:
        task_origin.leave(token)


def _source(name: str, doc: str) -> str:
    return f'def {name}(x: str) -> str:\n    """{doc}"""\n    return x\n'


def _note(gm, title, content, function_ids=None):
    return int(
        gm.add_guidance(title=title, content=content, function_ids=function_ids)[
            "details"
        ]["guidance_id"],
    )


# ── one origin for both kinds ────────────────────────────────────────────


@_handle_project
def test_guidance_records_its_request_under_the_record_switch_alone(switches):
    switches()
    gid = _in_task(
        REQUEST,
        lambda: _note(GuidanceManager(), "Batch renames", "Zip after renaming."),
    )
    row = db.query_one("SELECT origin FROM guidance WHERE guidance_id = ?", (gid,))
    origin = db.loads(row["origin"])
    assert origin[task_origin.FIELD] == [task_origin.task_key(REQUEST)]
    # No read returns it.
    assert "origin" not in GuidanceManager().get_guidance(guidance_id=gid).model_dump()


@_handle_project
def test_off_nothing_is_recorded(switches):
    switches(record=False, origin=False)
    gm = GuidanceManager()
    gid = _in_task(REQUEST, lambda: _note(gm, "Batch renames", "Zip after renaming."))
    gm.get_guidance(guidance_id=gid)
    row = db.query_one("SELECT origin FROM guidance WHERE guidance_id = ?", (gid,))
    assert row["origin"] is None
    assert not task_origin.request_log_path().exists()
    assert (
        db.query_one(
            "SELECT 1 AS found FROM sqlite_master WHERE name = ?",
            (entry_links.TABLE,),
        )
        is None
    )
    assert "record" not in gm.get_guidance(guidance_id=gid).model_dump()


# ── use: reads, calls, relied on ─────────────────────────────────────────


@_handle_project
def test_a_read_is_one_use_per_session_and_a_reviews_read_is_none(switches):
    switches()
    gm = GuidanceManager()
    gid = _in_task(REQUEST, lambda: _note(gm, "Batch renames", "Zip after renaming."))
    _in_task(
        AGAIN,
        lambda: (gm.get_guidance(guidance_id=gid), gm.get_guidance(guidance_id=gid)),
    )

    def in_review():
        with entry_record.reviewing():
            gm.get_guidance(guidance_id=gid)

    _in_task(OTHER, in_review)
    uses = entry_record.uses_of([("guidance", str(gid))])[("guidance", str(gid))]
    assert uses.keys("read") == [task_origin.text_key(task_origin.bounded_text(AGAIN))]
    record = gm.get_guidance(guidance_id=gid).record
    assert record.endswith(
        "unverified: written in a session whose outcome is unknown; read in 1 "
        "session (1 with the outcome unknown)",
    )


@_handle_project
def test_status_is_verified_by_an_accepted_writer_or_a_later_accepted_use(switches):
    switches()
    gm = GuidanceManager()
    gid = _in_task(REQUEST, lambda: _note(gm, "Batch renames", "Zip after renaming."))
    row = {
        "guidance_id": gid,
        "metadata": db.loads(
            db.query_one("SELECT origin FROM guidance WHERE guidance_id = ?", (gid,))[
                "origin"
            ],
        ),
    }
    assert entry_record.status("guidance", row).startswith(
        "unverified: written in a session whose outcome",
    )
    # A later session relied on it and was accepted.
    _in_task(
        AGAIN,
        lambda: entry_record.record_use("guidance", gid, "relied", review=True),
    )
    _in_task(AGAIN, lambda: task_origin.record_outcome(True))
    uses = entry_record.uses_of([("guidance", str(gid))])[("guidance", str(gid))]
    assert entry_record.status("guidance", row, uses) == (
        "verified: relied on in a later session whose answer was accepted"
    )
    # Its own session not accepted: unverified, saying so.
    _in_task(REQUEST, lambda: task_origin.record_outcome(False))
    assert entry_record.status("guidance", row) == (
        "unverified: written after a session whose answer was not accepted"
    )


def test_the_reviews_relied_on_line_is_parsed():
    text = 'Stored nothing new.\n{"answer_outcome": "confirmed", "relied_on": ["rename_batch", "guidance 5", "guidance_12", "not a name!"]}'
    assert entry_record.parse_relied(text) == [
        ("function", "rename_batch"),
        ("guidance", "5"),
        ("guidance", "12"),
    ]
    assert entry_record.parse_relied('{"relied_on": []}') == []
    assert entry_record.parse_relied("no line") is None


@_handle_project
def test_relied_on_is_kept_under_the_session_only_with_the_switch(switches):
    switches()
    _in_task(
        REQUEST,
        lambda: entry_record.record_relied('{"relied_on": ["rename_batch"]}'),
    )
    uses = entry_record.uses_of([("function", "rename_batch")])[
        ("function", "rename_batch")
    ]
    assert len(uses.keys("relied")) == 1
    switches(record=False)
    assert (
        _in_task(OTHER, lambda: entry_record.record_relied('{"relied_on": ["x"]}'))
        is None
    )


def test_the_review_is_asked_for_relied_on_only_with_the_switch(switches):
    switches(record=False)
    _, review_note, _ = caa._origin_link_notes(
        [],
        outcome=None,
        answer=None,
        lessons=False,
    )
    assert entry_record.REVIEW_SECTION not in review_note
    switches()
    _, review_note, _ = caa._origin_link_notes(
        [],
        outcome=None,
        answer=None,
        lessons=False,
    )
    assert entry_record.REVIEW_SECTION in review_note


@_handle_project
def test_a_function_search_row_carries_its_record(switches):
    switches()
    fm = FunctionManager()
    # An identifier is rare when not every logged request names it.
    _in_task(OTHER, lambda: None)
    _in_task(
        REQUEST,
        lambda: fm.add_functions(
            implementations=_source("rename_batch", "Rename a batch of files."),
        ),
    )
    _in_task(
        REQUEST,
        lambda: entry_record.record_use("function", "rename_batch", "call"),
    )
    rows = _in_task(AGAIN, lambda: fm.search_functions(query="rename files", n=3))
    row = next(r for r in rows if r["name"] == "rename_batch")
    assert row["record"].startswith(
        "stored while handling a request that also named `inv_2024q3`; that "
        "session's outcome is unknown",
    )
    assert row["record"].endswith(
        "called in 1 session (1 with the outcome unknown)",
    )


# ── links: many to many, in a table of their own ─────────────────────────


@_handle_project
def test_links_are_many_to_many_and_follow_every_write(switches):
    switches()
    fm, gm = FunctionManager(), GuidanceManager()
    # A link written before the switch (only in function_ids) is carried in.
    switches(record=False, origin=False)
    fm.add_functions(
        implementations=[
            _source("rename_batch", "Rename."),
            _source("zip_batch", "Zip."),
        ],
    )
    ids = {r["name"]: int(r["function_id"]) for r in fm.filter_functions()}
    old = _note(gm, "Old link", "From before.", [ids["rename_batch"]])
    switches()
    assert entry_links.links() == {(ids["rename_batch"], old)}
    # One note guiding two functions, and a second note on one of them.
    both = _note(
        gm,
        "Batch jobs",
        "Rename, then zip.",
        [ids["rename_batch"], ids["zip_batch"]],
    )
    one = _note(gm, "Zip names", "Keep ISO dates.", [ids["zip_batch"]])
    assert entry_links.functions_of(both) == sorted(
        [ids["rename_batch"], ids["zip_batch"]],
    )
    assert entry_links.notes_of(ids["zip_batch"]) == [both, one]
    gm.update_guidance(guidance_id=one, function_ids=[ids["rename_batch"]])
    assert entry_links.notes_of(ids["zip_batch"]) == [both]
    gm.delete_guidance(guidance_id=old)
    fm.delete_function(function_id=ids["zip_batch"], delete_dependents=False)
    assert entry_links.links() == {
        (ids["rename_batch"], both),
        (ids["rename_batch"], one),
    }


# ── identifier-aware search ──────────────────────────────────────────────


@_handle_project
def test_a_query_naming_an_identifier_finds_what_was_recorded_under_it(switches):
    switches(identifiers=True)
    fm, gm = FunctionManager(), GuidanceManager()
    _in_task(
        REQUEST,
        lambda: fm.add_functions(
            implementations=_source("prepare_archive", "Prepare an archive."),
        ),
    )
    gid = _in_task(REQUEST, lambda: _note(gm, "Archive prep", "Check the batch first."))
    _in_task(
        OTHER,
        lambda: fm.add_functions(
            implementations=_source("summarise_feedback", "Summarise feedback."),
        ),
    )
    _in_task(OTHER, lambda: _note(gm, "Feedback", "Three bullets."))
    rows = fm.search_functions(query="what did we do for inv_2024q3", n=1)
    assert rows[0]["name"] == "prepare_archive"
    assert rows[0]["record"].startswith(
        "stored while handling a request that also named `inv_2024q3`",
    )
    found = gm.search(references={"content": "inv_2024q3"}, k=1)
    assert found[0].guidance_id == gid
    assert found[0].record.startswith(
        "written while handling a request that also named `inv_2024q3`",
    )


@_handle_project
def test_without_the_switch_a_query_with_an_identifier_ranks_as_shipped(switches):
    switches(record=False, identifiers=False)
    fm = FunctionManager()
    fm.add_functions(implementations=_source("prepare_archive", "Prepare an archive."))
    rows = fm.search_functions(query="inv_2024q3", n=1)
    assert "record" not in rows[0]


def test_record_switches_refuse_to_start_without_request_records(switches):
    switches(origin=False)
    with pytest.raises(ValueError, match="UNIFY_ENTRY_RECORD needs UNIFY_TASK_ORIGIN"):
        task_origin.require_origin_link_prerequisites()
    switches(record=False, origin=False, identifiers=True)
    with pytest.raises(
        ValueError,
        match="UNIFY_SEARCH_IDENTIFIERS needs UNIFY_TASK_ORIGIN",
    ):
        task_origin.require_origin_link_prerequisites()

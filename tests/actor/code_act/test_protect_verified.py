"""Symbolic: ``UNIFY_PROTECT_VERIFIED``: a session not known to be accepted never replaces a verified entry.

On the 5 Oct Continual-ARC paper-protocol run a failed visit twice replaced a
rule a solved visit had written with generic advice. With the switch each
entry records which session wrote its current content; an overwrite, patch,
update or deletion of a verified entry (written in an accepted session, or
used in a later one) from a session whose answer is not known to be accepted
is not applied, and the writer is told why.
"""

from __future__ import annotations

import pytest

from unify import db
from unify.function_manager import entry_record, task_origin, verified_guard
from unify.function_manager.function_manager import FunctionManager
from unify.guidance_manager.guidance_manager import GuidanceManager
from unify.settings import SETTINGS

SOLVED = "Rename the files in batch inv_2024q3 to ISO dates and zip them."
LATER = "Rename the files in batch inv_2024q4 to ISO dates and zip them."
OTHER = "Summarise the customer feedback from last week in three bullets."


@pytest.fixture(autouse=True)
def fresh_store(monkeypatch, tmp_path):
    monkeypatch.setenv("UNIFY_STORE_PATH", str(tmp_path / "store.sqlite"))
    monkeypatch.setenv("UNIFY_HOME", str(tmp_path / "home"))
    db.reset_store()
    yield
    db.reset_store()


@pytest.fixture
def switches(monkeypatch):
    def set_(*, protect=True, origin=True, record=False):
        monkeypatch.setattr(SETTINGS, "UNIFY_PROTECT_VERIFIED", protect)
        monkeypatch.setattr(SETTINGS, "UNIFY_TASK_ORIGIN", origin)
        monkeypatch.setattr(SETTINGS, "UNIFY_ENTRY_RECORD", record)
        monkeypatch.setattr(SETTINGS, "UNIFY_TRY_FIRST", False)
        monkeypatch.setattr(SETTINGS, "UNIFY_FUNCTION_PATCH", True)
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


def _source(doc: str) -> str:
    return f'def rename_batch(x: str) -> str:\n    """{doc}"""\n    return x\n'


def _note(gm, content):
    return int(
        gm.add_guidance(title="Batch renames", content=content)["details"][
            "guidance_id"
        ],
    )


def _content(gid):
    return db.query_one("SELECT content FROM guidance WHERE guidance_id = ?", (gid,))[
        "content"
    ]


def _accept(request, solved=True):
    _in_task(request, lambda: task_origin.record_outcome(solved))


def test_a_verified_note_is_not_changed_or_deleted_by_an_unaccepted_session(switches):
    switches()
    gm = GuidanceManager()
    gid = _in_task(SOLVED, lambda: _note(gm, "Rename to ISO dates, then zip."))
    _accept(SOLVED)
    for request in (OTHER, LATER):  # outcome unknown, then not accepted
        if request == LATER:
            _accept(LATER, solved=False)
        with pytest.raises(ValueError, match="is kept as it is"):
            _in_task(
                request,
                lambda: gm.update_guidance(guidance_id=gid, content="Generic advice."),
            )
        with pytest.raises(ValueError, match="is kept as it is"):
            _in_task(
                request,
                lambda: gm.patch_guidance(
                    id_or_title=gid,
                    old="then zip",
                    new="then upload",
                    why="x",
                ),
            )
        with pytest.raises(ValueError, match="not deleted now"):
            _in_task(request, lambda: gm.delete_guidance(guidance_id=gid))
    assert _content(gid) == "Rename to ISO dates, then zip."


def test_the_writing_session_or_an_accepted_one_may_change_it(switches):
    switches()
    gm = GuidanceManager()
    gid = _in_task(SOLVED, lambda: _note(gm, "Rename, then zip."))
    _accept(SOLVED)
    _in_task(
        SOLVED,
        lambda: gm.update_guidance(
            guidance_id=gid,
            content="Rename to ISO dates, then zip.",
        ),
    )
    _accept(LATER)
    _in_task(
        LATER,
        lambda: gm.update_guidance(
            guidance_id=gid,
            content="Rename, zip, check the zip.",
        ),
    )
    assert _content(gid) == "Rename, zip, check the zip."
    origin = db.loads(
        db.query_one("SELECT origin FROM guidance WHERE guidance_id = ?", (gid,))[
            "origin"
        ],
    )
    assert origin[verified_guard.FIELD] == task_origin.text_key(
        task_origin.bounded_text(LATER),
    )


def test_an_unverified_note_may_be_changed(switches):
    switches()
    gm = GuidanceManager()
    gid = _in_task(
        SOLVED,
        lambda: _note(gm, "Rename, then zip."),
    )  # outcome never known
    _in_task(OTHER, lambda: gm.update_guidance(guidance_id=gid, content="Changed."))
    assert _content(gid) == "Changed."


def test_a_verified_function_is_kept_and_the_writer_is_told(switches):
    switches()
    fm = FunctionManager()
    _in_task(
        SOLVED,
        lambda: fm.add_functions(implementations=_source("Rename a batch.")),
    )
    _accept(SOLVED)
    result = _in_task(
        OTHER,
        lambda: fm.add_functions(implementations=_source("Generic."), overwrite=True),
    )
    assert result["rename_batch"].startswith(
        "kept: function `rename_batch` is kept as it is",
    )
    (row,) = fm.filter_functions(filter="name == 'rename_batch'")
    assert row["docstring"] == "Rename a batch."
    fid = int(row["function_id"])
    with pytest.raises(ValueError, match="not deleted now"):
        _in_task(OTHER, lambda: fm.delete_function(function_id=fid))


def test_a_later_accepted_use_verifies_an_entry(switches):
    switches(record=True)
    gm = GuidanceManager()
    gid = _in_task(
        OTHER,
        lambda: _note(gm, "Rename, then zip."),
    )  # writer's outcome unknown
    _in_task(
        LATER,
        lambda: entry_record.record_use("guidance", gid, "relied", review=True),
    )
    _accept(LATER)
    with pytest.raises(
        ValueError,
        match="relied on in a later session whose answer was accepted",
    ):
        _in_task(
            SOLVED,
            lambda: gm.update_guidance(guidance_id=gid, content="Changed."),
        )


def test_off_nothing_is_stamped_or_refused(switches):
    switches(protect=False, origin=False)
    gm = GuidanceManager()
    gid = _in_task(SOLVED, lambda: _note(gm, "Rename, then zip."))
    gm.update_guidance(guidance_id=gid, content="Changed.")
    assert (
        db.query_one("SELECT origin FROM guidance WHERE guidance_id = ?", (gid,))[
            "origin"
        ]
        is None
    )
    assert _content(gid) == "Changed."


def test_it_refuses_to_start_without_request_records(switches):
    switches(origin=False)
    with pytest.raises(
        ValueError,
        match="UNIFY_PROTECT_VERIFIED needs UNIFY_TASK_ORIGIN",
    ):
        task_origin.require_origin_link_prerequisites()


# ── versioned: changes kept beside verified content until their session is accepted ──


@pytest.fixture
def versioned(switches, monkeypatch):
    switches(record=True)
    monkeypatch.setattr(SETTINGS, "UNIFY_PROTECT_VERIFIED", "versioned")


def _origin(gid):
    return db.loads(
        db.query_one("SELECT origin FROM guidance WHERE guidance_id = ?", (gid,))[
            "origin"
        ],
    )


def test_versioned_keeps_the_change_beside_verified_content(versioned):
    gm = GuidanceManager()
    gid = _in_task(SOLVED, lambda: _note(gm, "Rename to ISO dates, then zip."))
    _accept(SOLVED)
    out = _in_task(
        OTHER,
        lambda: gm.update_guidance(guidance_id=gid, content="Generic advice."),
    )
    assert out["outcome"] == "guidance kept; change stored as an unverified version"
    assert (
        "replaces the content if this session's answer is accepted"
        in out["details"]["note"]
    )
    assert _content(gid) == "Rename to ISO dates, then zip."
    (version,) = verified_guard.versions(_origin(gid))
    assert version["fields"] == {"content": "Generic advice."}
    # A patch is kept the same way and says so.
    out = _in_task(
        OTHER,
        lambda: gm.patch_guidance(
            id_or_title=gid,
            old="then zip",
            new="then upload",
            why="x",
        ),
    )
    assert out["outcome"].startswith("guidance kept") and out["details"]["edits"]
    assert (
        len(verified_guard.versions(_origin(gid))) == 1
    )  # one per session, the latest
    # Reads show the canonical content; the record names the version.
    read = gm.get_guidance(guidance_id=gid)
    assert read.content == "Rename to ISO dates, then zip."
    assert read.record.endswith(
        "1 unverified version from later sessions kept beside it",
    )
    # Deleting stays refused.
    with pytest.raises(ValueError, match="not deleted now"):
        _in_task(OTHER, lambda: gm.delete_guidance(guidance_id=gid))


def test_an_accepted_outcome_promotes_the_version(versioned):
    gm = GuidanceManager()
    gid = _in_task(SOLVED, lambda: _note(gm, "Rename to ISO dates, then zip."))
    _accept(SOLVED)
    _in_task(
        LATER,
        lambda: gm.update_guidance(guidance_id=gid, content="Rename, zip, check."),
    )
    assert _content(gid) == "Rename to ISO dates, then zip."
    _accept(LATER)  # the review (or checker) later judges this session accepted
    assert _content(gid) == "Rename, zip, check."
    origin = _origin(gid)
    assert verified_guard.versions(origin) == []
    assert origin[verified_guard.FIELD] == task_origin.text_key(
        task_origin.bounded_text(LATER),
    )
    assert task_origin.task_key(LATER) in origin[task_origin.FIELD]
    history = db.query(
        "SELECT reason FROM guidance_history WHERE guidance_id = ?",
        (gid,),
    )
    assert [r["reason"] for r in history][-1] == verified_guard.PROMOTED_REASON


def test_a_rejected_session_s_version_is_never_applied(versioned):
    gm = GuidanceManager()
    gid = _in_task(SOLVED, lambda: _note(gm, "The specific rule."))
    _accept(SOLVED)
    _in_task(
        OTHER,
        lambda: gm.update_guidance(guidance_id=gid, content="Generic advice."),
    )
    _accept(OTHER, solved=False)
    assert _content(gid) == "The specific rule."
    assert len(verified_guard.versions(_origin(gid))) == 1


def test_a_function_version_is_kept_then_promoted(versioned):
    fm = FunctionManager()
    _in_task(
        SOLVED,
        lambda: fm.add_functions(implementations=_source("Rename a batch.")),
    )
    _accept(SOLVED)
    result = _in_task(
        LATER,
        lambda: fm.add_functions(
            implementations=_source("Rename a batch, with ISO dates."),
            overwrite=True,
        ),
    )
    assert result["rename_batch"].startswith(
        "kept; function `rename_batch` keeps its content",
    )
    (row,) = fm.filter_functions(filter="name == 'rename_batch'")
    assert row["docstring"] == "Rename a batch."
    _accept(LATER)
    (row,) = fm.filter_functions(filter="name == 'rename_batch'")
    assert row["docstring"] == "Rename a batch, with ISO dates."


def test_the_mode_parses_and_refuse_is_the_old_switch():
    from unify.settings import ProductionSettings

    assert (
        ProductionSettings(UNIFY_PROTECT_VERIFIED="1").UNIFY_PROTECT_VERIFIED
        == "refuse"
    )
    assert (
        ProductionSettings(UNIFY_PROTECT_VERIFIED="versioned").UNIFY_PROTECT_VERIFIED
        == "versioned"
    )
    assert ProductionSettings().UNIFY_PROTECT_VERIFIED == ""
    with pytest.raises(ValueError):
        ProductionSettings(UNIFY_PROTECT_VERIFIED="sometimes")

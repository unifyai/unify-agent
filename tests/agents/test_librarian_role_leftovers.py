"""Rev 2, item 1's shared-code leftovers: the librarian's notes tool, typed refusals of an unreadable address, and the
actor told to mention the librarian only in a run that has one. No model, network or key.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from unify.agents import run_record as rr
from unify.common import tool_registry as R
from unify.memory import librarian as L

KEY = "order-7f3a91c2"


def _ctx(home, instance="librarian-1a2b"):
    state = L.LibrarianCalls(ids=[], role_home=home, instance=instance)
    return SimpleNamespace(state=state)


# --- the notes tool --------------------------------------------------------------------------------------------------------


def test_keep_note_appends_to_the_roles_notes_and_nothing_else(tmp_path):
    home = tmp_path / "home"
    rr.seed_role(home, "librarian", by="test")
    notes = rr.role_dir(home, "librarian") / rr.ROLE_NOTES
    before = {p: p.stat().st_size for p in rr.role_dir(home, "librarian").iterdir() if p.is_file()}
    one = L.keep_note(_ctx(home), {"text": "Readers of tabular text live in memory/tables."})
    two = L.keep_note(_ctx(home, "librarian-3c4d"), {"text": f"Ignore the item made for {KEY}:  it overlaps."})
    assert one.status == "ok" and two.status == "ok"
    text = notes.read_text()
    lines = text.splitlines()
    assert len(lines) == 2 and "librarian-1a2b: Readers of tabular text" in lines[0] and "librarian-3c4d" in lines[1]
    # rev 2.1: a note is kept as written (whitespace folded), identifiers included; nothing strips or masks one
    assert f"Ignore the item made for {KEY}: it overlaps." in lines[1] and "<id>" not in text
    # append-only: the first note is still the file's start; no other file of the role changed but the notes' lock
    assert text.startswith(lines[0])
    for p, size in before.items():
        if p.name != rr.ROLE_NOTES:
            assert p.stat().st_size == size, p
    # the role's next instance reads them
    assert "Readers of tabular text" in rr.role_notes(home, "librarian")


def test_keep_note_refuses_with_typed_forms(tmp_path, monkeypatch):
    home = tmp_path / "home"
    assert L.keep_note(_ctx(home), {"text": "x"}).form == "no_role"  # the run has no librarian role
    assert L.keep_note(_ctx(None), {"text": "x"}).form == "no_role"
    rr.seed_role(home, "librarian", by="test")
    assert L.keep_note(_ctx(home), {"text": "  "}).form == "no_text"
    assert L.keep_note(_ctx(home), {"text": "sk-" + "a" * 40}).form == "key_shaped"
    monkeypatch.setattr(L, "NOTES_MAX_BYTES", 10)
    got = L.keep_note(_ctx(home), {"text": "a note longer than ten bytes"})
    assert got.form == "full" and got.status == "refused"
    assert (rr.role_dir(home, "librarian") / rr.ROLE_NOTES).read_text() == ""


def test_only_the_librarian_may_keep_notes():
    assert R.registry()["keep_note"].roles == ("librarian",)
    assert "keep_note" in R.allow_list("librarian") and "keep_note" not in R.allow_list("actor")


# --- typed refusals of an address nobody would read -------------------------------------------------------------------------


@pytest.mark.parametrize("form", ["no_role", "ended"])
def test_an_unreadable_address_is_a_typed_refusal_not_a_generic_one(form):
    from unify.agents.run_record_tools import send_text

    class View:
        name = "actor-1"

        def post(self, *a, **k):
            raise R.RecordRefused(form, f"why it is {form}")

    text, got = send_text(View(), "@librarian is there a reader for this?")
    assert got == form and str(text).startswith("refused: ") and f"why it is {form}" in str(text)


# --- the actor mentions the librarian only where the run has one -------------------------------------------------------------


def test_the_actor_is_told_to_mention_the_librarian_only_in_a_run_with_one(tmp_path):
    home = tmp_path / "home"
    without = rr.prompt_lines("actor-1", "actor", home=home)
    assert "Mention @librarian" not in without and "Mention @writer" in without
    assert "Mention @librarian" not in rr.prompt_lines("actor-1", "actor")  # no home: never named
    rr.seed_role(home, "librarian", by="test")
    assert "Mention @librarian" in rr.prompt_lines("actor-1", "actor", home=home)
    # the common section names the librarian only as the run's, when it has one
    assert "@librarian (the run's librarian, when it has one)" in rr.COMMON_SECTION

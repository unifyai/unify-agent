"""Rev 2.4 (P2, R6): the standing-note digest at every request start, and use counts on every card.

The digest lists the library's preference, rule and caution notes (and notes written before the ``kind`` field, as
``note``), with what each applies to and who stated it when its front matter says so; lessons, retired and withdrawn
notes are left out. No fit judgement: it is shown whatever the request. It is ordered by status, then trust (positive
less negative uses), then recency (the export's ``changed_order``), and held to a display limit; every note stays
reachable through ``memory.items(kind='note')`` and ``memory.get``. A persistent session re-sends it only when it
changes. Cards state ``used N times: M positive, K negative`` beside the status.

Real export (``library_export.generated_library_mode`` with the serving evidence), real package (a cell imports the
generated ``memory``), real digest and hooks; the library is a fixture tree, no model is called and nothing is
matched on words.
"""

from __future__ import annotations

import asyncio
import inspect
import json

from tests.memory.test_capture_examples import _cell
from tests.memory.test_layout import LIB, _tree
from unify.memory import library_export as lx
from unify.memory import library_helper as lh
from unify.memory import library_search, serving, standing_notes
from unify.memory.catalogue import write_files
from unify.memory.integration import hooks


def _note(title: str, description: str, body: str, **front: str) -> str:
    extra = "".join(f"{k}: {v}\n" for k, v in front.items())
    return f"---\ntitle: {title}\ndescription: {description}\n{extra}---\n## Rule\n{body}\n"


NOTES = {
    "notes/team/report-format.md": _note(
        "Report format",
        "Reports are tab-separated.",
        "Send reports as tab-separated files.",
        kind="preference",
        stated_by="the requester",
        applies_to="reports for the requester",
    ),
    "notes/team/approvals.md": _note(
        "Approvals",
        "Changes above the limit need a second approver.",
        "Ask a second approver first.",
        kind="rule",
        stated_by="the team lead",
        applies_to="every change above the limit",
    ),
    "notes/team/retry.md": _note(
        "Retries",
        "Retrying a refused change repeats the refusal.",
        "Do not retry.",
        kind="caution",
    ),
    "notes/team/tried.md": _note(
        "Tried",
        "Sorting by name first failed.",
        "It failed because of ties.",
        kind="lesson",
    ),
    "notes/team/old.md": _note("Old rule", "An old rule.", "Retired.", kind="rule"),
}
STATUS = {"notes/team/approvals.md": "adopted", "notes/team/old.md": "retired"}
#: Newest change first: item_history's keys are in that order.
HISTORY = [
    "notes/team/retry.md",
    "notes/team/report-format.md",
    "notes/text/dates.md",
    "notes/team/approvals.md",
    "notes/team/tried.md",
    "notes/team/old.md",
]
COUNTS = {
    "notes/team/report-format.md": {"clean": 2, "failed": 0, "neutral": 0},
    "notes/team/retry.md": {"clean": 0, "failed": 1, "neutral": 1},
}


def _status_of(item: str) -> str:
    return STATUS.get(item, "on-trial")


def _serving(counts: dict) -> dict:
    found = {i: {"counts": c} for i, c in counts.items()}
    statuses = {i: _status_of(i) for i in HISTORY}
    return {
        "evidence": serving.evidence_doc(found, {}, statuses),
        "withdrawn_items": {},
    }


def _copy(
    tmp_path,
    notes: dict | None = None,
    history: list | None = None,
    counts: dict | None = None,
):
    tree = _tree(tmp_path / "copy", {**LIB, **(NOTES if notes is None else notes)})
    hist = {
        i: [f"{n:012d} change {i}"]
        for n, i in enumerate(HISTORY if history is None else history)
    }
    files = lx.generated_library_mode(
        tree,
        status_of=_status_of,
        history=(hist, True),
        serving=_serving(COUNTS if counts is None else counts),
    )
    write_files(tree, files)
    return tree, files


def test_the_export_records_each_items_place_in_the_order_of_its_newest_change(
    tmp_path,
):
    tree, files = _copy(tmp_path)
    rows = json.loads(files[lh.ITEMS_FILE])["items"]
    assert [rows[i]["changed_order"] for i in HISTORY] == list(range(len(HISTORY)))
    assert (
        rows["memory.text.dates:week"]["changed_order"] is None
    )  # beyond the scanned history


def test_the_digest_shows_standing_notes_with_scope_and_owner_in_trust_then_recency_order(
    tmp_path,
):
    tree, _ = _copy(tmp_path)
    text = standing_notes.digest_text(tree)
    assert text.splitlines() == [
        standing_notes.DIGEST_HEAD,
        "- notes/team/approvals.md: [rule, adopted] Approvals: Changes above the limit need a second approver. "
        "(applies to: every change above the limit; stated by: the team lead)",
        "- notes/team/report-format.md: [preference, on-trial] Report format: Reports are tab-separated. "
        "(applies to: reports for the requester; stated by: the requester)",
        # a note written before the kind field: shown, with no scope or owner
        "- notes/text/dates.md: [note, on-trial] Dates: How dates are written.",
        # trust below the others: one negative use
        "- notes/team/retry.md: [caution, on-trial] Retries: Retrying a refused change repeats the refusal.",
        "Showing 4 of 4 standing notes. " + standing_notes.DIGEST_REACH,
    ]
    # a lesson and a retired note are not standing notes
    assert "notes/team/tried.md" not in text and "notes/team/old.md" not in text


def test_recency_orders_notes_of_equal_status_and_trust(tmp_path):
    tree, _ = _copy(tmp_path, counts={})
    ids = [n["id"] for n in standing_notes.standing_notes(tree)]
    assert ids == [
        "notes/team/approvals.md",
        "notes/team/retry.md",
        "notes/team/report-format.md",
        "notes/text/dates.md",
    ]


def test_the_display_limit_never_hides_a_note_from_memory(tmp_path):
    many = {
        f"notes/rules/r{i:02d}.md": _note(
            f"Rule {i}",
            "x" * 400,
            "Follow it.",
            kind="rule",
            applies_to="y" * 100,
        )
        for i in range(12)
    }
    tree, _ = _copy(tmp_path, notes=many, history=sorted(many))
    lines = standing_notes.digest_text(tree).splitlines()
    entries = [line for line in lines if line.startswith("- ")]
    assert len(entries) == standing_notes.DIGEST_ENTRIES == 8
    for line in entries:
        head, _, rest = line.partition(": ")
        assert (
            head.startswith("- notes/rules/r")
            and len(rest) <= standing_notes.ENTRY_CHARS
        )
    # the twelve rules and the fixture library's note written before the kind field
    assert lines[-1] == "Showing 8 of 13 standing notes. " + standing_notes.DIGEST_REACH
    # a display limit only: the package lists and opens every note, the ones the digest does not show too
    got = _cell(
        tree,
        "print(json.dumps({'ids': [i.id for i in memory.items(kind='note')], "
        "'last': str(memory.get('notes/rules/r11.md'))}))\n",
    )
    assert set(many) <= set(got["ids"]) and got["last"].startswith(
        "notes/rules/r11.md  [on-trial",
    )


def test_a_library_without_standing_notes_has_no_digest(tmp_path):
    tree, _ = _copy(
        tmp_path,
        notes={"notes/team/tried.md": NOTES["notes/team/tried.md"]},
    )
    tree.joinpath("notes/text/dates.md").unlink()
    write_files(tree, lx.generated_library_mode(tree, status_of=_status_of))
    assert standing_notes.digest_text(tree) == ""
    assert standing_notes.digest_text(None) == ""


def test_front_matter_without_the_new_fields_is_read_gracefully():
    assert standing_notes.front_fields(LIB["notes/text/dates.md"]) == {
        "title": "Dates",
        "description": "How dates are written.",
        "uses": "[memory.text.dates:parse_date, memory.text.dates:gone]",
    }
    assert standing_notes.front_fields("no front matter") == {}
    assert standing_notes.front_fields("---\ntitle: never closed\n") == {}


def test_a_persistent_session_resends_the_digest_only_when_it_changes(tmp_path):
    tree, _ = _copy(tmp_path)
    digest = standing_notes.StandingDigest(lambda: tree)
    first = digest.update()
    assert first and first.startswith(standing_notes.DIGEST_HEAD)

    class Handle:
        def __init__(self) -> None:
            self.sent: list[str] = []

        async def submit(self, text: str) -> None:
            self.sent.append(text)

    handle = Handle()
    hooks.digest_on_submit(handle, digest)
    asyncio.run(handle.submit("the next request"))
    assert handle.sent == [
        "the next request",
    ]  # unchanged: nothing repeated, the cached prefix holds
    # the library changes: the next request opens with the new digest, once
    STATUS["notes/team/retry.md"] = "adopted"
    try:
        _copy(tmp_path)
    finally:
        del STATUS["notes/team/retry.md"]
    asyncio.run(handle.submit("another request"))
    asyncio.run(handle.submit("a third request"))
    assert (
        handle.sent[1].endswith("\n\n---\n\nanother request")
        and handle.sent[1] != first
    )
    assert "- notes/team/retry.md: [caution, adopted]" in handle.sent[1]
    assert handle.sent[2] == "a third request"


def test_the_actor_opens_each_request_with_the_digest():
    from unify.actor.code_act_actor import CodeActActor

    source = inspect.getsource(CodeActActor.act)
    assert "_memory.standing_digest(" in source and "digest.update()" in source
    assert "_memory.digest_on_submit(handle, digest)" in source
    assert hooks.digest_on_submit(None, None) is None  # inert without a digest


# --- R6: use counts on every card ------------------------------------------------------------------------------------


def test_use_counts_text():
    assert (
        lh.use_counts_text({"clean": 2, "failed": 1, "neutral": 1})
        == "used 4 times: 2 positive, 1 negative"
    )
    assert lh.use_counts_text({"clean": 1}) == "used 1 time: 1 positive, 0 negative"
    assert lh.use_counts_text({}) == lh.use_counts_text(None) == "not used yet"


def test_cards_show_use_counts_beside_the_status(tmp_path):
    tree, files = _copy(tmp_path)
    got = _cell(
        tree,
        "print(json.dumps({'note': str(memory.get('notes/team/report-format.md')), "
        "'fn': str(memory.get('memory.text.dates:week'))}))\n",
    )
    assert got["note"].startswith(
        "notes/team/report-format.md  [on-trial, used 2 times: 2 positive, 0 negative, ",
    )
    assert got["fn"].startswith("memory.text.dates:week(s)  [on-trial, not used yet, ")
    catalogue = lx.catalogue_text(tree, status_of=_status_of, serving=_serving(COUNTS))
    retry = catalogue.split("### notes/team/retry.md\n", 1)[1].split("###", 1)[0]
    assert "Status: on-trial, used 2 times: 0 positive, 1 negative; " in retry
    week = catalogue.split("### memory.text.dates:week\n", 1)[1].split("###", 1)[0]
    assert "; status on-trial, not used yet; " in week
    rows = {
        r["id"]: r["card"]
        for r in library_search.rows(
            tree,
            status_of=_status_of,
            serving=_serving(COUNTS),
        )
    }
    assert (
        "| used 2 times: 2 positive, 0 negative" in rows["notes/team/report-format.md"]
    )

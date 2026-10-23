"""A recorded procedure (a capture) is reusable the moment it is stored, and nothing the agents read discourages its use.

No writer pass has to run before a capture is listed, found by the librarian and callable; its card names what it is
and what it rests on (the work an earlier request accepted), never a trial status, a "not reviewed" label or a "may
hold only for that case" warning. That a capture merged into an item stays offered is covered in
``test_capture_withdrawal``; that its calls are counted as use, in ``test_record_contract``.
"""

from __future__ import annotations

import json

from tests.memory.test_capture_examples import ROWS, TABLE, _cell, _copy, _put, _signature
from tests.memory.test_layout import _tree
from unify.memory import captures, librarian, library_prompts, library_search
from unify.memory import library_export as lx
from unify.memory.catalogue import write_files
from unify.memory.item_records import support, support_text

#: Wording that marks a stored entry as second-class or tells an agent to wait for or prefer curated items.
DISCOURAGING = (
    "not yet reviewed",
    "unreviewed",
    "pending",
    "may hold only for that case",
    "has not been used",
    "on-trial",
    "unchecked",
)


def _served_copy(tmp_path, outcome: str = "none"):
    """A library copy as a request's export writes it, with the capture store's snapshot and the librarian's files
    (its search index included), and no writer pass ever run; one capture's episode had *outcome*."""
    store = captures.store_dir(tmp_path / "state")
    table = _put(store, TABLE, outcome=outcome, signature=_signature(ROWS, 7), asked="Add up an order table.")
    tree = _tree(tmp_path / "lib")
    snapshot = captures.export_files(store)
    write_files(tree, lx.generated_library_mode(tree, librarian=True, captures=snapshot))
    for rel, data in snapshot.items():
        (tree / rel).parent.mkdir(parents=True, exist_ok=True)
        (tree / rel).write_bytes(data)
    return tree, f"capture:{table}"


def test_capture_served_before_any_pass(tmp_path):
    """Straight from the capture store's export, with no pass ever run: listed by ``memory.items()``, in the
    librarian's library listing and in its search results, accepted as the request-start primary, and
    ``memory.get(id)`` returns a callable that runs it on a request's text."""
    root, table = _served_copy(tmp_path)
    request = "TABLE\n" + json.dumps(ROWS)
    got = _cell(
        root,
        f"ex = memory.get({table!r})\n"
        f"print(json.dumps({{'listed': [i.id for i in memory.items()], 'callable': callable(ex), "
        f"'run': ex({request!r})}}))\n",
    )
    assert table in got["listed"] and got["callable"] is True
    assert got["run"] == 7  # the item's call ran it on the request's text and returned its result
    listing = librarian.library_items(root)
    assert listing is not None and table in listing[0] and table not in listing[1]
    found = library_search.search_text(library_search.read_index(root), "add up the amounts of an order table")
    assert table in found, found[:2000]
    # the librarian may name it as the request-start primary: its answer is accepted and the block names it first
    answer = {"fits": [{"id": table, "call": f"memory.get({table!r})()", "why": "it re-runs adding up an order table"}]}
    checked, why = librarian.check_answer(answer, listing[0], listing[1], listing[2])
    assert checked is not None, why
    block = librarian.start_block(checked)
    assert block.splitlines()[1].startswith(f"- {librarian.BEST_FIT_LABEL}: {table}:")


def test_capture_card_has_no_discouraging_wording(tmp_path):
    """The card of a capture whose episode was accepted, its status text, the librarian's briefs and the actor's guide
    hold none of the wording that marks a stored entry as second-class."""
    words = ("not yet reviewed", "unreviewed", "pending", "may hold only for that case", "has not been used")
    root, table = _served_copy(tmp_path, outcome="positive")
    got = _cell(root, f"print(json.dumps(str(memory.get({table!r}))))\n")
    head = got.split("\n\n", 1)[0]
    assert head.startswith(f"{table}  [recorded procedure"), head
    assert "accepted" in head, head  # the card says what the capture rests on: accepted work
    status = json.loads((root / captures.EXPORT_DIR / table.split(":", 1)[1] / captures.RECORD_FILE).read_text())
    texts = {
        "card": got,
        "status text": str(status.get("status_text") or ""),
        "librarian brief": librarian.BRIEF,
        "librarian brief with evidence": librarian.BRIEF_EVIDENCE,
        "actor guide": library_prompts.GUIDE_LIBRARIAN,
    }
    found = {name: [w for w in words if w in text.lower()] for name, text in texts.items()}
    assert not any(found.values()), found


def test_a_capture_card_and_run_report_carry_no_discouraging_wording(tmp_path):
    root, ids = _copy(tmp_path)
    table = f"capture:{ids['table']}"
    got = _cell(
        root,
        f"print(json.dumps({{'card': str(memory.get({table!r})), 'value': memory.get({table!r})('TABLE\\n[]'), "
        f"'run': repr(memory._tracked_run({table!r}, ('TABLE\\n[]',), {{}}))}}))\n",
    )
    head = got["card"].split("\n\n", 1)[0]  # the card's lines above the procedure's own source
    assert not [w for w in DISCOURAGING if w in head.lower()], head
    # the item's call returns its result; the report of the tracked call it is made through says nothing more of it
    assert got["value"] == 0 and "run of " + table + ": returned" in got["run"] and "result: 0" in got["run"]
    # every run report says a result is unchecked against the request, for any item; nothing else is said of a capture
    assert not [w for w in DISCOURAGING if w != "unchecked" and w in got["run"].lower()], got["run"]
    # the fixture's episode recorded no outcome: the card claims no acceptance
    assert "re-runs the work of an earlier request (no outcome of that request is on record)" in head
    assert "Calling it with no arguments runs it on the current request" in head or "Calling it runs it" in head
    assert "rests on: recorded procedure" in got["run"]


def test_the_texts_agents_read_never_rank_captures_below_curated_items():
    for text in (library_prompts.GUIDE_LIBRARIAN, librarian.BRIEF, librarian.BRIEF_EVIDENCE):
        assert "may hold only for that case" not in text and "not yet reviewed" not in text
        assert "wait for" not in text
    assert "an item's kind (a curated item, a note or a recorded procedure) is never a reason" in librarian.BRIEF


def test_support_says_a_recorded_procedure_rests_on_accepted_work():
    rec = {
        "episodes": ["e1", "e2"],
        "cases": [{"episode": "e1", "outcome": "positive"}, {"episode": "e2", "outcome": "none"}],
        "track": {},
    }
    assert support(rec) == {"episodes": 2, "confirmed_uses": 0, "accepted": 1}
    assert support_text("recorded procedure", support(rec)) == "recorded procedure from 2 episodes, accepted in 1"

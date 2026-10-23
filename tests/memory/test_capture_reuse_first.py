"""A recorded procedure (a capture) is reusable the moment it is stored, and nothing the agents read discourages its use.

No writer pass has to run before a capture is listed, found by the librarian and callable; its card names what it is
and what it rests on (the work an earlier request accepted), never a trial status, a "not reviewed" label or a "may
hold only for that case" warning. That a capture merged into an item stays offered is covered in
``test_capture_withdrawal``; that its calls are counted as use, in ``test_use_telemetry`` and ``test_metering``.
"""

from __future__ import annotations

import json

from tests.memory.test_capture_examples import ROWS, _cell, _copy
from unify.memory import librarian, library_prompts, trust
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


def test_a_fresh_capture_is_served_before_any_writer_pass(tmp_path):
    """Straight from the capture store's export, with no pass ever run: listed, in the librarian's library, and its
    call runs it on a request's text."""
    root, ids = _copy(tmp_path)
    table = f"capture:{ids['table']}"
    request = "TABLE\n" + json.dumps(ROWS)
    got = _cell(
        root,
        f"ex = memory.get({table!r})\n"
        f"print(json.dumps({{'listed': [i.id for i in memory.items()], 'card': str(ex), "
        f"'run': repr(memory.run({table!r}, {request!r}))}}))\n",
    )
    assert table in got["listed"]
    assert "run of " + table + ": returned" in got["run"] and "result: 7" in got["run"]
    listing = librarian.library_items(root)
    assert listing is not None and table in listing[0] and table not in listing[1]


def test_a_capture_card_and_run_report_carry_no_discouraging_wording(tmp_path):
    root, ids = _copy(tmp_path)
    table = f"capture:{ids['table']}"
    got = _cell(
        root,
        f"print(json.dumps({{'card': str(memory.get({table!r})), "
        f"'run': repr(memory.run({table!r}, 'TABLE\\n[]'))}}))\n",
    )
    head = got["card"].split("\n\n", 1)[0]  # the card's lines above the procedure's own source
    assert not [w for w in DISCOURAGING if w in head.lower()], head
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
    assert trust.record_line({}) == "no later use recorded yet"


def test_support_says_a_recorded_procedure_rests_on_accepted_work():
    rec = {
        "episodes": ["e1", "e2"],
        "cases": [{"episode": "e1", "outcome": "positive"}, {"episode": "e2", "outcome": "none"}],
        "track": {},
    }
    assert support(rec) == {"episodes": 2, "confirmed_uses": 0, "accepted": 1}
    assert support_text("recorded procedure", support(rec)) == "recorded procedure from 2 episodes, accepted in 1"

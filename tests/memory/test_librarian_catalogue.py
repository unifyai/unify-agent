"""The librarian's generated files (design J(b).3): ``.memory/catalogue.md``, the complete catalogue the librarian's
system prompt carries, and ``.memory/dispatch.json``, the readers and ``applies`` that ``memory.ask`` runs.

Both are written by the harness at each export, with no model call, only while ``UNIFY_MEMORY_LIBRARIAN=on``:

* one card per stored function, in module-path order (never a score), each with its id, signature, the first
  paragraph of its docstring, what it was saved against (the identifier-stripped ``## Asked`` and ``## Approach``
  lines of its episodes' ``summary.md``), its evidence (level, status, reader, whether it has ``applies``, input
  line) and the notes it links to;
* the notes after the functions, and the captured procedures last, at level ``unchecked``;
* no answer value, no raw request text, no printed identifier; the same bytes for the same commit.

The item records are what production writes (``librarian_world``: a landing, the library gate's check and
``lifecycle.next_records``), so a function's reader is the one the records name: a landed procedure's is the reader
it was landed with, another function's is the gate's when the gate found exactly one, and a function the gate found
several readers of has none.

No model is called. Building the world's records runs code confined, so those tests need bubblewrap.
"""

from __future__ import annotations

import json
import re

import pytest

from tests.memory import librarian_world as w
from tests.memory.librarian_world import needs_box
from unify.memory import captures
from unify.memory.library_index import signature_text
from unify.memory.signatures import strip_identifiers
from unify.settings import SETTINGS

# ==== ASSUMED NAMES (the integrator adapts these to the code, here only) ======================================
#: The generated files and the call that generates them: ``librarian_world.CATALOGUE_FILE``,
#: ``librarian_world.DISPATCH_FILE`` and ``librarian_world.generate`` / ``generate_off``.
CATALOGUE = w.CATALOGUE_FILE
DISPATCH = w.DISPATCH_FILE
#: The setting (design J(b).2) and the one it requires.
SWITCH = "UNIFY_MEMORY"  # the librarian is on whenever memory is
#: The two field labels the spec names on every function card (J(b).3, in bold there).
SAVED_AGAINST_LABEL = "saved against"
EVIDENCE_LABEL = "evidence"
#: The keys of one dispatch row (J(b).3) and where the rows sit in the file (a list, or a list under this key).
#: Rev 2: the structural catalogue's row (``library_export.dispatch_data``).
DISPATCH_KEYS = {"id", "kind", "form", "signature", "applies_text", "effects", "reads_input", "constant", "occurrences",
                 "outcomes", "provenance"}  # fmt: skip
DISPATCH_ROWS_KEY = "procedures"


def card_heads(text: str) -> list[tuple[int, int, str]]:
    """Each card of the catalogue as (line number, heading level, id): a card opens with a Markdown heading line
    that names its id (a function's ``memory.<package>.<module>:<name>``, a note's ``notes/...md`` path, a captured
    procedure's ``capture:<16 hex digits>``)."""
    ids = re.compile(r"(memory(?:\.[a-z_][a-z0-9_]*)+:[A-Za-z_][A-Za-z0-9_]*|notes/[^\s`*]+\.md|capture:[0-9a-f]{16})")
    out = []
    for n, line in enumerate(text.splitlines()):
        m = re.match(r"(#+)\s", line)
        if m and (found := ids.search(line)):
            out.append((n, len(m.group(1)), found.group(1)))
    return out


def dispatch_rows(data) -> list[dict]:
    return data if isinstance(data, list) else data[DISPATCH_ROWS_KEY]


# ==============================================================================================================


def set_switch(monkeypatch, value: str) -> None:
    """Set the switch on the live settings; while the setting does not exist yet the generating call's own
    argument still says which mode it is."""
    if hasattr(SETTINGS, SWITCH):  # while the setting does not exist yet, the test fails on what it checks
        monkeypatch.setattr(SETTINGS, SWITCH, value)


@pytest.fixture(autouse=True)
def librarian_on(monkeypatch):
    set_switch(monkeypatch, "on")


def _card(text: str, item: str) -> str:
    """The text of *item*'s card: from its heading to the next heading of the same or a higher level."""
    lines = text.splitlines()
    heads = card_heads(text)
    mine = [h for h in heads if h[2] == item]
    assert len(mine) == 1, (item, [h[2] for h in heads])
    start, level, _ = mine[0]
    end = len(lines)
    for n in range(start + 1, len(lines)):
        m = re.match(r"(#+)\s", lines[n])
        if m and len(m.group(1)) <= level:
            end = n
            break
    return "\n".join(lines[start:end])


def _flat(text: str) -> str:
    return " ".join(text.split())


def _world(tmp_path, captured: bool = True, files=None):
    tree = w.write_tree(tmp_path / "lib", files)
    caps = w.capture_files() if captured else {}
    return tree, w.generate(tree, records=w.records(), summaries=w.SUMMARIES, captures=caps), caps


def _capture_ids(caps: dict) -> list[str]:
    return sorted({"capture:" + p.split("/")[2] for p in caps})


# --- the catalogue's layout ----------------------------------------------------------------------------------


@needs_box
def test_one_card_per_function_in_module_path_order_then_notes_then_captures(tmp_path):
    _, files, caps = _world(tmp_path)
    text = files[CATALOGUE].decode("utf-8")
    order = [h[2] for h in card_heads(text)]
    assert order == w.FUNCTIONS + [w.NOTE] + _capture_ids(caps), order
    # a private function has no card
    assert "_helper" not in text


@needs_box
def test_a_card_holds_its_id_signature_first_docstring_paragraph_and_notes(tmp_path):
    _, files, _ = _world(tmp_path)
    text = files[CATALOGUE].decode("utf-8")
    card = _card(text, w.TOTAL)
    assert "`memory.orders.procedures:total_amount(value)`" in card  # as the landing named it
    # the first paragraph, both its lines, and nothing of the second
    assert "Sum the amount column of an order table." in card
    assert "Each row is a record keyed by column name." in card
    assert "SECOND-PARAGRAPH-MARK" not in text
    assert w.NOTE in card  # the notes it links to
    for item in w.FUNCTIONS:
        assert _card(text, item)  # every function has exactly one card


@needs_box
def test_a_card_shows_what_it_was_saved_against_without_identifiers(tmp_path):
    _, files, _ = _world(tmp_path)
    text = files[CATALOGUE].decode("utf-8")
    card = _flat(_card(text, w.TOTAL))
    assert SAVED_AGAINST_LABEL in card.lower()
    assert _flat(strip_identifiers(w.TOTAL_ASKED)) in card
    assert _flat(strip_identifiers(w.TOTAL_APPROACH)) in card
    ledger = _flat(_card(text, w.RECONCILE))
    assert _flat(strip_identifiers(w.LEDGER_ASKED)) in ledger
    assert _flat(strip_identifiers(w.LEDGER_APPROACH)) in ledger


@needs_box
def test_a_card_shows_its_evidence_level_status_reader_applies_and_input_line(tmp_path):
    _, files, _ = _world(tmp_path)
    text = files[CATALOGUE].decode("utf-8")
    recs = w.records()
    card = _card(text, w.TOTAL)
    assert EVIDENCE_LABEL in card.lower()
    # rev 2's gate is labels only: it runs no reader, so the writer's parsed fixture is not re-derived and the
    # procedure's calls are not grounded on it; the landed procedure stays unchecked with that finding, and its card
    # shows the record's level and status and the reader it was landed with
    assert recs[w.TOTAL]["level"] == "unchecked"
    assert any("[examples:grounded]" in r for r in recs[w.TOTAL]["level_reasons"])
    assert f"Evidence: level unchecked; status adopted; reader {w.READ_TOTAL}; applies: yes" in card
    # its input line: the capture's applicability signature, which its record carries from the landing
    assert f"Input: {signature_text(recs[w.TOTAL]['signature'])}" in card
    ledger = _card(text, w.RECONCILE)
    assert f"level {recs[w.RECONCILE]['level']}; status on-trial;" in ledger
    # never landed: no signature, so no input line
    assert recs[w.RECONCILE]["signature"] is None and "Input:" not in ledger


@needs_box
def test_captured_procedures_come_last_at_level_unchecked(tmp_path):
    _, files, caps = _world(tmp_path)
    text = files[CATALOGUE].decode("utf-8")
    for cid in _capture_ids(caps):
        card = _card(text, cid)
        assert "unchecked" in card, card
    first_capture = min(n for n, _, i in card_heads(text) if i.startswith("capture:"))
    note_line = next(n for n, _, i in card_heads(text) if i == w.NOTE)
    function_lines = [n for n, _, i in card_heads(text) if i in w.FUNCTIONS]
    assert max(function_lines) < note_line < first_capture


@needs_box
def test_the_catalogue_carries_no_answer_raw_request_or_printed_identifier(tmp_path):
    _, files, _ = _world(tmp_path)
    text = files[CATALOGUE].decode("utf-8")
    assert w.ANSWER not in text
    assert w.RAW_REQUEST not in text and "attached order sheet" not in text
    for ident in w.IDENTIFIERS:
        assert ident not in text, ident
    # the summaries' other sections are not cards' fields
    assert "The total came to" not in text and "accepted:" not in text


@needs_box
def test_the_catalogue_is_byte_stable_for_the_same_commit_and_names_no_host_path(tmp_path):
    _, one, caps = _world(tmp_path / "a")
    _, two, _ = _world(tmp_path / "b")
    assert one[CATALOGUE] == two[CATALOGUE]
    assert one[DISPATCH] == two[DISPATCH]
    tree = tmp_path / "a" / "lib"
    again = w.generate(tree, records=w.records(), summaries=w.SUMMARIES, captures=caps)
    assert again[CATALOGUE] == one[CATALOGUE] and again[DISPATCH] == one[DISPATCH]
    for blob in (one[CATALOGUE], one[DISPATCH]):
        assert str(tmp_path).encode() not in blob


@needs_box
def test_the_order_is_never_a_score(tmp_path):
    """Use counts, statuses and levels change nothing about where a card sits."""
    tree = w.write_tree(tmp_path / "lib")
    recs = w.records()
    recs[w.WORDS]["use"] = {"2" * 40: {"episodes": 50, "uses": 90, "seen": 90}}
    recs[w.WORDS]["status"] = "adopted"
    recs[w.WORDS]["level"] = "confirmed-by-outcomes"
    text = w.generate(tree, records=recs, summaries=w.SUMMARIES)[CATALOGUE].decode("utf-8")
    assert [h[2] for h in card_heads(text) if h[2] in w.FUNCTIONS] == w.FUNCTIONS


def test_an_empty_library_has_an_empty_catalogue_and_dispatch_file(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    files = w.generate(empty)
    assert card_heads(files[CATALOGUE].decode("utf-8")) == []
    assert dispatch_rows(json.loads(files[DISPATCH])) == []


# --- the dispatch file ------------------------------------------------------------------------------------------


@needs_box
def test_the_dispatch_file_is_one_structural_row_per_listed_procedure(tmp_path):
    _, files, caps = _world(tmp_path)
    rows = dispatch_rows(json.loads(files[DISPATCH]))
    for row in rows:
        assert set(row) == DISPATCH_KEYS, row
        assert row["kind"] in ("function", "capture-shape") and isinstance(row["applies_text"], str)
        assert isinstance(row["effects"], list) and isinstance(row["constant"], bool)
    by_id = {r["id"]: r for r in rows}
    total = by_id[w.TOTAL]
    assert total["signature"] == signature_text(w.records()[w.TOTAL]["signature"])
    assert total["applies_text"].startswith("def ")  # its applies, named by its record, as text
    # rev 2: every stored function is a row, listed by its structure (nothing runs to decide which are); one with no
    # applies has an empty applies text and no input line
    assert by_id[w.COUNT]["kind"] == "function" and by_id[w.COUNT]["applies_text"] == ""
    for item in (w.TOKENS, w.WORDS, w.RECONCILE, w.READ_ORDERS, w.READ_LEDGER, w.READ_PAYMENTS):
        assert by_id[item]["kind"] == "function"
    # a capture is a row of its shape, its applies as text when it states one
    grid, marked = (f"capture:{w.capture_id(s)}" for s in (w.CAPTURE_GRID, w.CAPTURE_MARKED))
    assert by_id[grid]["kind"] == by_id[marked]["kind"] == "capture-shape"
    assert by_id[grid]["applies_text"].startswith("def applies(") and by_id[marked]["applies_text"] == ""
    # rows carry no request text, answer, identifier or host path
    blob = files[DISPATCH].decode("utf-8")
    for planted in (w.ANSWER, w.RAW_REQUEST, *w.IDENTIFIERS, str(tmp_path)):
        assert planted not in blob


# --- each function's reader, as the records production writes name it -------------------------------------------


def _dispatched(files: dict) -> dict:
    data = json.loads(files[DISPATCH])
    return {"rows": {r["id"]: r for r in dispatch_rows(data)}}


def _evidence(card: str) -> str:
    (line,) = [x for x in card.splitlines() if x.startswith("Evidence: ")]
    return line


@needs_box
def test_a_landed_procedure_is_read_by_the_reader_it_was_landed_with(tmp_path):
    """The landed procedure's record names its capture and the reader it was landed with, and carries the capture's
    signature and its applies; the dispatch file and its card use the landing's reader (rev 2's gate derives
    none)."""
    total = w.records()[w.TOTAL]
    assert total["landed_from"] == {"capture": w.landed_capture(), "reader": w.READ_TOTAL}
    # rev 2: the gate runs no reader (re-deriving a parsed fixture would), so it records none
    assert total["reader"] == []
    landed = captures.landed_items(w.built().store)[w.TOTAL]
    assert total["signature"] == landed["signature"] and isinstance(total["signature"], dict)
    assert total["applies"] == landed["applies"] == w.TOTAL_APPLIES
    _, files, _ = _world(tmp_path)
    got = _dispatched(files)["rows"][w.TOTAL]
    assert got["signature"] == signature_text(total["signature"]) and got["applies_text"].startswith("def ")
    card = _card(files[CATALOGUE].decode("utf-8"), w.TOTAL)
    assert _evidence(card).endswith(f"; reader {w.READ_TOTAL}; applies: yes")


# --- off ---------------------------------------------------------------------------------------------------------


@needs_box
def test_off_generates_neither_file_and_the_other_files_keep_their_bytes(tmp_path, monkeypatch):
    from unify.memory.layout import INDEX_FILE, LINKS_FILE, RESERVED_CONTRACT, RESERVED_INIT, RESERVED_STUB
    from unify.memory.library_helper import FIND_FILE, ITEMS_FILE, MATCHER_FILE, SHAPES_FILE

    set_switch(monkeypatch, "off")
    tree = w.write_tree(tmp_path / "lib")
    off = w.generate_off(tree, records=w.records(), summaries=w.SUMMARIES, captures=w.capture_files())
    assert set(off) == {
        RESERVED_INIT,
        RESERVED_CONTRACT,
        RESERVED_STUB,
        INDEX_FILE,
        LINKS_FILE,
        ITEMS_FILE,
        FIND_FILE,
        MATCHER_FILE,
        SHAPES_FILE,
    }
    set_switch(monkeypatch, "on")
    on = w.generate(tree, records=w.records(), summaries=w.SUMMARIES)
    # the files both modes generate are the same bytes, except the package with its stub (which gain memory.ask) and
    # the index the actor no longer reads
    for path in set(off) & set(on) - {"memory/__init__.py", RESERVED_STUB, "INDEX.md"}:
        assert on[path] == off[path], path

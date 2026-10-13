"""The librarian's generated files (design J(b).3): ``.memory/catalogue.md``, the complete catalogue the librarian's
system prompt carries, and ``.memory/dispatch.json``, the readers and ``applies`` that ``memory.ask`` runs.

Both are written by the harness at each export, with no model call, only while ``UNIFY_MEMORY_LIBRARIAN=on``:

* one card per stored function, in module-path order (never a score), each with its id, signature, the first
  paragraph of its docstring, what it was saved against (the identifier-stripped ``## Asked`` and ``## Approach``
  lines of its episodes' ``summary.md``), its evidence (level, status, reader, whether it has ``applies``, input
  line) and the notes it links to;
* the notes after the functions, and the captured procedures last, at level ``unchecked``;
* no answer value, no raw request text, no printed identifier; the same bytes for the same commit.

No model is called and no box is started.
"""

from __future__ import annotations

import json
import re

import pytest

from tests.memory import librarian_world as w
from unify.memory.library_index import signature_text
from unify.memory.signatures import strip_identifiers
from unify.settings import SETTINGS

# ==== ASSUMED NAMES (the integrator adapts these to the code, here only) ======================================
#: The generated files and the call that generates them: ``librarian_world.CATALOGUE_FILE``,
#: ``librarian_world.DISPATCH_FILE`` and ``librarian_world.generate`` / ``generate_off``.
CATALOGUE = w.CATALOGUE_FILE
DISPATCH = w.DISPATCH_FILE
#: The setting (design J(b).2) and the one it requires.
SWITCH = "UNIFY_MEMORY_LIBRARIAN"
#: The two field labels the spec names on every function card (J(b).3, in bold there).
SAVED_AGAINST_LABEL = "saved against"
EVIDENCE_LABEL = "evidence"
#: The keys of one dispatch row (J(b).3) and where the rows sit in the file (a list, or a list under this key).
DISPATCH_KEYS = {"id", "reader", "applies", "input"}
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
    monkeypatch.setattr(SETTINGS, "UNIFY_TOOLS_IN_CODE", "on")
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


def test_one_card_per_function_in_module_path_order_then_notes_then_captures(tmp_path):
    _, files, caps = _world(tmp_path)
    text = files[CATALOGUE].decode("utf-8")
    order = [h[2] for h in card_heads(text)]
    assert order == w.FUNCTIONS + [w.NOTE] + _capture_ids(caps), order
    # a private function has no card
    assert "_helper" not in text


def test_a_card_holds_its_id_signature_first_docstring_paragraph_and_notes(tmp_path):
    _, files, _ = _world(tmp_path)
    text = files[CATALOGUE].decode("utf-8")
    card = _card(text, w.TOTAL)
    assert "total_amount(rows: list) -> int" in card
    # the first paragraph, both its lines, and nothing of the second
    assert "Sum the amount column of an order table." in card
    assert "Each row is a record keyed by column name." in card
    assert "SECOND-PARAGRAPH-MARK" not in text
    assert w.NOTE in card  # the notes it links to
    for item in w.FUNCTIONS:
        assert _card(text, item)  # every function has exactly one card


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


def test_a_card_shows_its_evidence_level_status_reader_applies_and_input_line(tmp_path):
    _, files, _ = _world(tmp_path)
    text = files[CATALOGUE].decode("utf-8")
    card = _card(text, w.TOTAL)
    assert EVIDENCE_LABEL in card.lower()
    assert "reproduces-its-case" in card and "stable" in card
    assert w.READ_ORDERS in card  # its reader
    assert "applies" in card
    assert _flat(signature_text(w.TOTAL_SIGNATURE)) in _flat(card)
    ledger = _card(text, w.RECONCILE)
    assert "unchecked" in ledger and "experimental" in ledger
    assert _flat(signature_text(w.LEDGER_SIGNATURE)) in _flat(ledger)


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


def test_the_catalogue_carries_no_answer_raw_request_or_printed_identifier(tmp_path):
    _, files, _ = _world(tmp_path)
    text = files[CATALOGUE].decode("utf-8")
    assert w.ANSWER not in text
    assert w.RAW_REQUEST not in text and "attached order sheet" not in text
    for ident in w.IDENTIFIERS:
        assert ident not in text, ident
    # the summaries' other sections are not cards' fields
    assert "The total came to" not in text and "accepted:" not in text


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


def test_the_order_is_never_a_score(tmp_path):
    """Use counts, statuses and levels change nothing about where a card sits."""
    tree = w.write_tree(tmp_path / "lib")
    recs = w.records()
    recs[w.WORDS]["use"] = {"2" * 40: {"episodes": 50, "uses": 90, "seen": 90}}
    recs[w.WORDS]["status"] = "stable"
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


def test_the_dispatch_file_lists_each_procedure_with_a_reader_or_applies(tmp_path):
    _, files, caps = _world(tmp_path)
    rows = dispatch_rows(json.loads(files[DISPATCH]))
    for row in rows:
        assert set(row) == DISPATCH_KEYS, row
        assert isinstance(row["applies"], bool)
        assert row["reader"] is None or (isinstance(row["reader"], str) and row["reader"])
        assert isinstance(row["input"], str)
    by_id = {r["id"]: r for r in rows}
    total = by_id[w.TOTAL]
    assert total["reader"] == w.READ_ORDERS and total["applies"] is True
    assert total["input"] == signature_text(w.TOTAL_SIGNATURE)
    # a function with neither a reader nor applies is not a row
    for item in (w.TOKENS, w.WORDS, w.COUNT):
        assert item not in by_id
    grid, marked = (f"capture:{w.capture_id(s)}" for s in (w.CAPTURE_GRID, w.CAPTURE_MARKED))
    assert by_id[grid]["applies"] is True and by_id[grid]["reader"]
    assert by_id[marked]["applies"] is False and by_id[marked]["reader"]
    assert by_id[grid]["reader"] != by_id[marked]["reader"]
    # rows carry no request text, answer, identifier or host path
    blob = files[DISPATCH].decode("utf-8")
    for planted in (w.ANSWER, w.RAW_REQUEST, *w.IDENTIFIERS, str(tmp_path)):
        assert planted not in blob


# --- off ---------------------------------------------------------------------------------------------------------


def test_off_generates_neither_file_and_the_other_files_keep_their_bytes(tmp_path, monkeypatch):
    from unify.memory.layout import INDEX_FILE, LINKS_FILE, RESERVED_INIT
    from unify.memory.library_helper import FIND_FILE, ITEMS_FILE, MATCHER_FILE, SHAPES_FILE

    set_switch(monkeypatch, "off")
    tree = w.write_tree(tmp_path / "lib")
    off = w.generate_off(tree, records=w.records(), summaries=w.SUMMARIES, captures=w.capture_files())
    assert set(off) == {RESERVED_INIT, INDEX_FILE, LINKS_FILE, ITEMS_FILE, FIND_FILE, MATCHER_FILE, SHAPES_FILE}
    set_switch(monkeypatch, "on")
    on = w.generate(tree, records=w.records(), summaries=w.SUMMARIES)
    # the files both modes generate are the same bytes, except the package and the index the actor no longer reads
    for path in set(off) & set(on) - {"memory/__init__.py", "INDEX.md"}:
        assert on[path] == off[path], path

"""What a recorded procedure's card, its call and the librarian say of it can be done with what an agent has.

A procedure that acts through its episode's tools (``run(state, env)``) is not run by any call: a call cannot pass it
tools, and no cell has ``exec`` or a ``state``. Its card, a call's refusal and the librarian's brief say so, show its
code with ``memory.get(id).text``, and say that the same tool calls made in a cell with the agent's own tools do its
work; the librarian's answer cannot offer a call of it. A card prints no code at all and at most ``CARD_CHARS``
characters, ending with the call that prints the code. The package's ``memory.ask`` prints one best fit and at most one
fallback, never a menu, with the words the request-start block uses.
"""

from __future__ import annotations

import inspect
from pathlib import Path

from tests.memory.test_capture_examples import _cell
from tests.memory.test_capture_trace import VALUE_PROC
from tests.memory.test_record_contract import _export, _store_two_captures
from unify.memory import captures as cp
from unify.memory import librarian, library_ask
from unify.memory import library_helper as lh
from unify.memory.library_export import _capture_reader


def test_an_acting_capture_is_not_run_by_any_call_and_its_card_says_how_its_work_is_done(tmp_path):
    _, store, cid, acting = _store_two_captures(tmp_path / "home")
    root, _ = _export(tmp_path / "copy", store, None)
    aid = "capture:" + acting
    got = _cell(
        root,
        "seen = []\n"
        "sys.addaudithook(lambda e, a: seen.append(e) if e == 'memory.call' else None)\n"
        f"item = memory.get({aid!r})\n"
        "calls = [err(item), err(item, 'a request'), err(item, {'a': 1}, {'b': 2}), err(item, state=1)]\n"
        "print(json.dumps({'card': str(item), 'calls': calls, 'seen': seen, 'text': item.text,\n"
        "                  'builtins_exec': 'exec' in dir(__builtins__) if isinstance(__builtins__, dict) else None}))\n",
    )
    assert [c[0] for c in got["calls"]] == ["WrongCall"] * 4 and got["seen"] == []
    for text in [got["card"], *(c[1] for c in got["calls"])]:
        assert f"print(memory.get({aid!r}).text)" in text and "memory.request()" in text
        assert "exec(" not in text and "ns['run']" not in text
    assert "a call does not run it here" in got["card"]
    assert got["text"].startswith('"""Write the summary file')
    # the same words in the catalogue line the librarian reads, and in the help of a call
    assert "no call runs it" in _capture_reader({"entry_point": {"form": "state_env"}}, None)
    assert "exec(" not in _capture_reader({"entry_point": {"form": "state_env"}}, None)
    doc = inspect.getdoc(lh.Item.__call__)
    assert "of the value or state form" in doc and "is not run by any call" in doc
    assert "exec(" not in lh.BY_HAND


def test_a_capture_card_prints_no_code_is_bounded_and_ends_with_the_call_that_prints_the_code(tmp_path):
    from tests.memory.test_capture_trace import _ep as capture_ep
    from tests.memory.test_capture_trace import _staging
    from unify.memory.integration.consolidate import open_stores
    from unify.memory.integration.paths import Paths

    stores = open_stores(Paths.under(tmp_path / "home"))
    state = Path(stores.paths.state_dir)
    long_proc = VALUE_PROC + "".join(f"\n\n# step {i}: {'x' * 70}" for i in range(200)) + "\n"
    assert len(long_proc) > 4 * lh.CARD_CHARS
    row = cp.capture_trace_step(_staging(state, "e9", {"procedure.py": long_proc}), capture_ep("e9"), cp.store_dir(state))
    root, _ = _export(tmp_path / "copy", cp.store_dir(state), None)
    iid = "capture:" + row["capture_id"]
    got = _cell(root, f"i = memory.get({iid!r})\nprint(json.dumps([str(i), i.text]))\n")
    card, text = got
    assert text == long_proc
    assert card.rstrip("\n").endswith(f"Its code: print(memory.get({iid!r}).text)")
    assert "def " not in card and "# step" not in card and len(card) <= lh.CARD_CHARS


def test_the_librarian_cannot_offer_a_call_of_an_acting_capture(tmp_path):
    _, store, cid, acting = _store_two_captures(tmp_path / "home")
    root, _ = _export(tmp_path / "copy", store, None)
    ids, failed, recipes = librarian.library_items(root)
    acts = librarian.acting_captures(root)
    aid, vid = "capture:" + acting, "capture:" + cid
    assert acts == {aid}
    for call in (f"memory.get({aid!r})()", f"memory.get({aid!r})(memory.request())", f"f = memory.get({aid!r})\nf()"):
        got, why = librarian.check_answer({"fits": [{"id": aid, "call": call, "why": "w"}]}, ids, failed, recipes, acts)
        assert got is None and f"print(memory.get({aid!r}).text)" in why, call
    shown = {"fits": [{"id": aid, "call": f"print(memory.get({aid!r}).text)", "why": "w"}]}
    assert librarian.check_answer(shown, ids, failed, recipes, acts)[0] == shown
    value = {"fits": [{"id": vid, "call": f"memory.get({vid!r})()", "why": "w"}]}
    assert librarian.check_answer(value, ids, failed, recipes, acts)[0] == value
    # a refused call names each form's call, and the brief names the call per form
    _, why = librarian.check_answer({"fits": [{"id": vid, "call": "do it", "why": "w"}]}, ids, failed, recipes, acts)
    assert "that takes the request" in why and "print(memory.get('<id>').text)" in why
    for brief in (librarian.BRIEF, librarian.BRIEF_EVIDENCE):
        assert "says no call runs it" in brief and "the call its card gives that prints its code" in brief


def test_ask_prints_one_best_fit_and_one_fallback_with_the_blocks_words():
    assert (library_ask.BEST_FIT_LABEL, library_ask.FALLBACK_LABEL) == (librarian.BEST_FIT_LABEL, librarian.FALLBACK_LABEL)

    def item(iid, reason):
        return lh.Item(id=iid, kind="function", summary="", applies="", evidence="unchecked", status="on-trial",
                       example="", text="", links=(), history=(), reason=reason)  # fmt: skip

    a, b = "memory.text.dates:parse_date", "capture:" + "a" * 16
    shown = repr(library_ask.Answer([item(a, "it parses dates"), item(b, "it re-runs the work")],
                                    calls={a: f"memory.get({a!r})(x)", b: f"memory.get({b!r})()"}))  # fmt: skip
    assert f"Best fit: {a} (function): it parses dates" in shown
    assert f"Only if the first fails: {b} (function): it re-runs the work" in shown
    assert "apply:" not in shown and "item(s)" not in shown
    block = librarian.start_block({"fits": [{"id": a, "call": "c1", "why": "w1"}, {"id": b, "call": "c2", "why": "w2"}]})
    assert f"- Best fit: {a}: w1" in block and f"- Only if the first fails: {b}: w2" in block
    doc = inspect.getdoc(library_ask.ask)
    assert "every item" not in doc and "at most one fallback" in doc
    assert "every item" not in (inspect.getdoc(library_ask.Answer) or "")
    assert "every item it judges applicable" not in lh.__doc__ and "at most one fallback" in lh.__doc__


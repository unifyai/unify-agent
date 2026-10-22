"""The generalisation gate: a landed procedure carries a recipe, ``apply(request) -> result``, and lands only when the
recipe, run confined, gives the expected output on every case its episode observed.

Each world is built by the code that builds it in production (:mod:`tests.memory.gate_records`): the capture step on
the episode's staged procedure, ``land_capture`` (which writes the recipe, the source case and the shown cases it is
given through the fixture tool), then the library gate's check. Two kinds of request: a
request that shows one example call of a pricing rule with its result, and a text request with one worked example.
Keyless; the boxes need bubblewrap and prlimit.
"""

from __future__ import annotations

import json

import pytest

from tests.memory import gate_records as gr
from tests.memory.test_episodes import _ep
from unify.memory.episodes import Action
from unify.memory.signals import Signal

pytestmark = [gr.needs_box, pytest.mark.timeout(900)]


def _cut(text: str, part: str) -> list[int]:
    raw, want = text.encode("utf-8"), part.encode("utf-8")
    start = raw.index(want)
    return [start, start + len(want)]


def _episode(eid: str, request: str, answer, regime: str = "dense"):
    """A recorded episode: the request, and the dialogue action that handed *answer* back."""
    return _ep(
        episode_id=eid,
        regime=regime,
        request=[request],
        transcript=[{"role": "user", "content": request}],
        cells=[],
        actions=[Action(-1, "dialogue:user", "answer", [answer], {}, "Thanks.", "ok", "unknown", kind="dialogue")],
    )


# --- an API-style request: one example call with its result ----------------------------------------------------

EXAMPLE_CALL = '{"quantity": 2, "unit": 5}'
PRICING = (
    "The price service multiplies the quantity by the unit price.\n"
    f"Example call: price({EXAMPLE_CALL}) returned 10\n"
    'Call it for: {"quantity": 4, "unit": 3}\n'
)
#: The request's example call as a shown case: its arguments, and the result the request showed for them.
SHOWN_CALL = {
    "input": {"source": "request:0", "slice": _cut(PRICING, EXAMPLE_CALL)},
    "expected": {"source": "request:0", "slice": _cut(PRICING, "returned 10")},
}
SHOWN_CALL["expected"]["slice"][0] += len("returned ")  # the result alone

_HEAD = '''"""Price an order line read from the request."""


class MemoryInputError(ValueError):
    """Not an order line."""


def read_input(text):
    """The last JSON object of the request: the order line to price."""
    found = [v for v in text.data if isinstance(v, dict) and {"quantity", "unit"} <= set(v)]
    if not found:
        raise MemoryInputError("the request holds no order line")
    return found[-1]
'''

#: Derives every value from the request: quantity times unit price.
DERIVES = _HEAD + '''

def run(value):
    """Quantity times unit price."""
    return value["quantity"] * value["unit"]
'''

#: Copies the unit price of the request it was learned from: right there, wrong on the example call.
HARDCODES = _HEAD + '''

def run(value):
    """Quantity times the unit price of the request it was learned from."""
    return value["quantity"] * 3
'''


def _pricing(procedure: str, *, cases=(SHOWN_CALL,), regime: str = "dense", signals=(), tmp_path):
    ep = _episode("p1", PRICING, 12, regime)
    landing = gr.Landing("p1", procedure, "## Asked\nPrice an order line.\n", "orders", "line_price", tuple(cases))
    return gr.build(tmp_path, episodes={"p1": ep}, base={}, landings=[landing], signals=signals, require_pass=False)


PROC = "memory.orders.procedures:line_price"
RECIPE = "memory.orders.procedures:apply_line_price"
SHOWN_FILE = "memory/orders/tests/data/line_price.case1.shown1.case.json"
SOURCE_FILE = "memory/orders/tests/data/line_price.case1.source.case.json"


def test_a_recipe_that_derives_every_value_from_the_request_lands_on_its_source_and_shown_cases(tmp_path):
    built = _pricing(DERIVES, tmp_path=tmp_path)
    assert built.result.passed, built.result.reasons
    assert PROC not in built.result.items_refused and RECIPE not in built.result.items_refused
    proc = next(i for i in built.manifest["items"] if i["item"] == PROC)
    assert proc["recipe"] == RECIPE and proc["cases"] == [SOURCE_FILE, SHOWN_FILE]
    shown = json.loads(built.files[SHOWN_FILE])
    assert (shown["input"], shown["expected"], shown["kind"]) == (EXAMPLE_CALL, 10, "shown")
    # recorded per item: how many cases, how many passed, of which kinds; the catalogue card shows them
    assert built.result.verification[PROC]["recipe"] == {
        "item": RECIPE,
        "cases": 2,
        "passed": 2,
        "kinds": {"source": 1, "shown": 1},
    }
    from unify.memory.item_records import recipe_text

    assert recipe_text(built.records[PROC]) == f"{RECIPE}; 2 of 2 cases passed (1 shown, 1 source)"


def _labels(res, item: str) -> list[str]:
    """Rev 2: an item's quality findings are labels in its verification record, never refusals."""
    return list((res.verification.get(item) or {}).get("label_findings", []))


def test_a_recipe_that_hardcodes_a_source_constant_fails_the_shown_case_and_is_labelled(tmp_path):
    built = _pricing(HARDCODES, tmp_path=tmp_path)
    res = built.result
    assert res.passed and not res.items_refused, res.reasons
    assert all(any("does not give the expected output" in r for r in _labels(res, i)) for i in (PROC, RECIPE))
    (reason,) = [r for r in _labels(res, PROC) if "does not give the expected output" in r]
    assert SHOWN_FILE in reason and "a shown case of p1" in reason
    # the difference report names the expected output and the actual one
    assert "expected: 10" in reason and "actual:   6" in reason
    assert "the value: expected 10, got 6" in reason
    assert res.verification[PROC]["recipe"]["passed"] == 1 and res.verification[PROC]["recipe"]["cases"] == 2


def test_with_no_reward_signal_and_no_shown_case_the_source_case_alone_lands_it(tmp_path):
    built = _pricing(HARDCODES, cases=(), regime="none", tmp_path=tmp_path)
    assert built.result.passed, built.result.reasons
    assert built.result.verification[PROC]["recipe"] == {
        "item": RECIPE,
        "cases": 1,
        "passed": 1,
        "kinds": {"source": 1, "shown": 0},
    }


def test_a_procedure_from_a_failed_episode_is_labelled(tmp_path):
    contrary = Signal("p1.provenance.1", "p1", "provenance", "correct", "2026-10-08T01:02:00Z", regime="dense")
    built = _pricing(DERIVES, signals=[contrary], tmp_path=tmp_path)
    res = built.result
    # rev 2: the failed episode is a label on the procedure, never a refusal
    assert res.passed and PROC not in res.items_refused, res.reasons
    assert any("failed; a failed episode never lands a procedure" in r for r in _labels(res, PROC))
    assert any(f"{SOURCE_FILE} is the source case of p1, which failed" in r for r in _labels(res, PROC))


def test_a_case_not_re_derivable_from_the_recording_is_refused(tmp_path):
    built = _pricing(DERIVES, tmp_path=tmp_path)
    invented = json.loads(built.files[SHOWN_FILE])
    invented["expected"] = 11
    text = json.dumps(invented, sort_keys=True, ensure_ascii=False, indent=1) + "\n"
    rebuilt = _pricing_with_file(SHOWN_FILE, text, tmp_path=tmp_path / "again")
    res = rebuilt.result
    assert not res.passed and PROC in res.items_refused
    assert any("differs from the case" in r for r in res.item_reasons[PROC]), res.item_reasons[PROC]


def _pricing_with_file(path: str, text: str, *, tmp_path):
    ep = _episode("p1", PRICING, 12)
    landing = gr.Landing("p1", DERIVES, "## Asked\nPrice an order line.\n", "orders", "line_price", (SHOWN_CALL,))
    return gr.build(
        tmp_path, episodes={"p1": ep}, base={}, landings=[landing], files={path: text}, require_pass=False,
    )


def test_a_procedure_without_a_recipe_is_labelled(tmp_path):
    ep = _episode("p1", PRICING, 12)
    landing = gr.Landing("p1", DERIVES, "## Asked\nPrice an order line.\n", "orders", "line_price", ())

    def without_recipe(manifest: dict) -> dict:
        items = [
            {k: v for k, v in i.items() if not (i["item"] == PROC and k in ("recipe", "cases"))}
            for i in manifest["items"]
        ]
        return {**manifest, "items": items}

    built = gr.build(
        tmp_path, episodes={"p1": ep}, base={}, landings=[landing], edit_manifest=without_recipe, require_pass=False,
    )
    res = built.result
    # rev 2: a label on the procedure, never a refusal
    assert res.passed and PROC not in res.items_refused, res.reasons
    assert any("is a procedure without a recipe" in r for r in _labels(res, PROC))


# --- a text request with one worked example --------------------------------------------------------------------

INITIALS = "Write the initials of the name.\nExample: Ada Lovelace -> A.L.\nName: Grace Brewster Hopper\n"
INITIALS_PROC = '''"""The initials of the name the request ends with."""


class MemoryInputError(ValueError):
    """No name."""


def read_input(text):
    """The name: the last non-empty line, after any label."""
    lines = [ln.strip() for ln in str(text).splitlines() if ln.strip()]
    if not lines:
        raise MemoryInputError("the request holds no name")
    return lines[-1].split(":")[-1].strip()


def run(value):
    """Each word's first letter, each followed by a full stop."""
    return "".join(word[0] + "." for word in value.split())
'''


def test_a_text_request_with_one_worked_example_lands_when_the_recipe_gives_its_output(tmp_path):
    ep = _episode("t1", INITIALS, "G.B.H.", regime="none")
    shown = {
        "input": {"source": "request:0", "slice": _cut(INITIALS, "Ada Lovelace")},
        "expected": {"source": "request:0", "slice": _cut(INITIALS, "A.L.")},
    }
    landing = gr.Landing("t1", INITIALS_PROC, "## Asked\nInitials.\n", "names", "initials", (shown,))
    built = gr.build(tmp_path, episodes={"t1": ep}, base={}, landings=[landing])
    proc = "memory.names.procedures:initials"
    assert built.result.verification[proc]["recipe"]["passed"] == 2
    assert built.result.verification[proc]["recipe"]["kinds"] == {"source": 1, "shown": 1}
    card = built.records[proc]
    from unify.memory.item_records import verification_summary

    assert verification_summary(card)["recipe"].startswith("memory.names.procedures:apply_initials; 2 of 2")

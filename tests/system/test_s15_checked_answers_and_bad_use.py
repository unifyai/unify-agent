"""S15: an answer made from a function on the request's declared input, stored as the request's procedure, recalled on
the next request; a rejected answer equal to what the recalled procedure returned is a bad use of it, and the
librarian is asked again on the message that shows the verdict.

Design (rev 2.3): the declared input and examples as data beside a message (never parsed from text); the answer from a
function, checked first on those examples; the function an accepted answer was made with stored as the request's
procedure by AST nodes only, read by the capture step in place of the fork's, reading the declared input on a later
request; a bad use counted inside the request by value identity; the re-ask on new evidence, its line above the
message, with the failed item named to the librarian. Real components throughout; only the model and the world are
scripted, and no real model is called.
"""

from __future__ import annotations

import json

import pytest

from tests.actor.code_act.sandbox_world import needs_bwrap
from tests.memory.conformance import harness as H
from tests.scripted_model import Always, ScriptedModel, cell, reply
from tests.system import sysworld as SW
from tests.system.drive import MEMORY_MODES

FN = "def solve(words):\n    return [w[::-1].upper() for w in words]\n"
EXAMPLES = [{"input": ["ab"], "output": ["BA"]}]


def _last_user(call) -> str:
    users = [m for m in call.messages if m.get("role") == "user"]
    return SW.flat(users[-1].get("content")) if users else ""


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(1800)
@pytest.mark.parametrize("mode", MEMORY_MODES)
async def test_an_accepted_answer_from_a_function_is_stored_recalled_and_a_rejected_reuse_is_a_bad_use(system, mode):
    s = system
    s.writer.plan = "keep"

    # ── request 1: the input and examples come beside a message; the actor answers from its function ──
    sink1: dict = {}

    def ask_for_input(call):
        return reply("Please send the input.")

    def answer_from(call):
        sink1["msg"] = _last_user(call)
        code = FN + SW.probe_json(
            "_try('examples', lambda: tools.examples())\n_try('answered', lambda: str(tools.answer_from(solve)))\n"
        )
        return reply(calls=[cell(code)])

    def after_verdict(call):
        sink1["out"] = SW.marked(SW.last_tools(call))
        return reply("Good.")

    host1 = H.replies(
        [{"message": "Here is the input.", "input": ["stone", "river"], "examples": EXAMPLES}],
        [{"checker": {"label": "pass"}}, {"message": SW.ACCEPTED_MESSAGE}],
    )
    model1 = ScriptedModel(actor=[ask_for_input, answer_from, after_verdict],
                           librarian=Always(lambda c: SW.librarian_answer([], nearest=[])))  # fmt: skip
    r1 = await s.request(SW.words_request(["stone", "river"]), model1, host1)
    assert "Here is the input." in sink1["msg"] and "stone" not in sink1["msg"].split("Here is the input.")[1]
    assert sink1["out"]["examples"] == EXAMPLES
    assert str(sink1["out"]["answered"]).startswith("answered"), sink1["out"]
    s.settle()

    use1 = json.loads(s.episode_file(r1.episode_id, "memory_use.json"))
    (row,) = use1["answered_from"]
    assert row["fn"] == "solve" and row["source"] == "declared" and row["answered"] is True
    assert row["checked"] == {"passed": 1, "total": 1} and row["stored"] is True, row
    cid = s.capture_of(r1.episode_id)
    item = f"capture:{cid}"
    cap = s.captures()[cid]
    assert cap["record"]["reads"] == "declared_input" and cap["record"]["checked"] == {"passed": 1, "total": 1}
    source = (s.paths.state_dir / "memory-captures" / cid / "procedure.py").read_text()
    assert "def solve(words)" in source and "def run(value)" in source and "probe" not in source

    # ── request 2: recalled with one call on the new declared input; its value answered and rejected ──
    sink2: dict = {}
    librarian_calls: list = []

    def librarian(call):
        librarian_calls.append(_last_user(call))
        if "Tried and failed in this request" in librarian_calls[-1]:
            # a librarian that keeps to the rule: a failed item is offered again only by reconsider
            return SW.librarian_answer([], nearest=[{"id": item, "differs": "its answer was not accepted"}])
        return SW.librarian_answer([SW.fit(item)])

    def first(call):
        sink2["start"] = _last_user(call)
        return reply("Please send the input.")

    def use_it(call):
        code = SW.probe_json(f"_try('call', lambda: memory.get({item!r})())\n") + "tools.answer(out['call'])\n"
        return reply(calls=[cell(code)])

    def after_rejection(call):
        sink2["msg"] = _last_user(call)
        sink2["out"] = SW.marked(SW.last_tools(call))
        return reply("I will try another way.")

    host2 = H.replies(
        [{"message": "Here is the input.", "input": ["oak", "elm"]}],
        [{"checker": {"label": "fail"}}, {"message": "That answer was not accepted."}],
    )
    model2 = ScriptedModel(actor=[first, use_it, after_rejection], librarian=Always(librarian))
    r2 = await s.request(SW.words_request(["oak", "elm"]), model2, host2)
    assert sink2["out"]["call"] == ["KAO", "MLE"], sink2["out"]  # one call, on the declared input, in the child
    s.settle()
    use2 = json.loads(s.episode_file(r2.episode_id, "memory_use.json"))
    seen = {k: use2.get(k) for k in ("answer_attributions", "reasks", "answered_from")}
    # the verdict's message came with the librarian's answer on it, above the message
    from unify.memory.librarian import REASK_HEAD

    # the line comes above the message it is on (the run record shows the message as one entry)
    at = sink2["msg"].find(REASK_HEAD)
    assert -1 < at < sink2["msg"].find("That answer was not accepted."), (
        sink2["msg"], seen)
    assert any("Tried and failed in this request" in q and item in q for q in librarian_calls), librarian_calls[-1:]
    rejected = [a for a in use2["answer_attributions"] if a["label"] == "fail"]
    assert rejected and rejected[0]["items"] == [item] and rejected[0]["short"] is False, use2["answer_attributions"]
    (reask,) = use2["reasks"]
    assert "outcome" in reask["events"] and "failed_use" in reask["events"]
    assert s.item_use(item).get("answered_rejected", 0) >= 1


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(900)
@pytest.mark.parametrize("mode", ["off"])
async def test_without_memory_an_answer_from_a_function_is_checked_first_and_withheld_on_a_failing_example(system,
                                                                                                         mode):
    s = system
    sink: dict = {}
    wrong = "def wrong(words):\n    return [w.upper() for w in words]\n"

    def ask_for_input(call):
        return reply("Please send the input.")

    def try_both(call):
        code = FN + wrong + SW.probe_json(
            "_try('withheld', lambda: str(tools.answer_from(wrong)))\n"
            "_try('answered', lambda: str(tools.answer_from(solve)))\n"
        )
        return reply(calls=[cell(code)])

    def done(call):
        sink["out"] = SW.marked(SW.last_tools(call))
        return reply("Good.")

    host = H.replies([{"message": "Here is the input.", "input": ["stone"], "examples": EXAMPLES}],
                     [{"message": SW.ACCEPTED_MESSAGE}])  # fmt: skip
    r = await s.request(SW.words_request(["stone"]), ScriptedModel(actor=[ask_for_input, try_both, done]), host)
    out = sink["out"]
    assert out["withheld"].startswith("not answered: fn did not pass this request's examples (passed 0 of 1)"), out
    assert str(out["answered"]).startswith("answered"), out
    results = [line for line in r.lines if line.get("type") == "response"]
    assert any("ENOTS" in json.dumps(line) for line in results), results


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(900)
@pytest.mark.parametrize("mode", ["off", "on"])
async def test_the_first_turn_sees_what_the_declared_file_declares(system, mode, monkeypatch, tmp_path):
    # a driver's first message is the command line, never a stdin line: its declaration comes in a file (--declared)
    import unify.cli as cli

    path = tmp_path / "declared.json"
    path.write_text(json.dumps({"input": ["stone", "river"], "examples": EXAMPLES}))
    real = cli._parse_args
    monkeypatch.setattr(cli, "_parse_args", lambda argv: real([*argv[:-1], "--declared", str(path), argv[-1]]))
    s = system
    sink: dict = {}

    def first(call):
        code = FN + SW.probe_json(
            "_try('examples', lambda: tools.examples())\n_try('answered', lambda: str(tools.answer_from(solve)))\n"
        )
        return reply(calls=[cell(code)])

    def after(call):
        sink["out"] = SW.marked(SW.last_tools(call))
        return reply("Good.")

    model = ScriptedModel(actor=[first, after], librarian=Always(lambda c: SW.librarian_answer([], nearest=[])))
    r = await s.request(SW.words_request(["stone", "river"]), model, H.replies([{"message": SW.ACCEPTED_MESSAGE}]))
    assert sink["out"]["examples"] == EXAMPLES
    assert str(sink["out"]["answered"]).startswith("answered"), sink["out"]
    assert any("ENOTS" in json.dumps(line) for line in r.lines if line.get("type") == "response")

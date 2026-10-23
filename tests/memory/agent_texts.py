"""Every text an agent of a memory run is shown, rendered as the harness composes it, for the guards that read them
(``test_no_world_vocabulary.py``, ``test_no_storage_steering.py``, ``test_gate_claims.py``).

:func:`rendered` gives, by source name: each agent's tool JSON, stub, opening, ``tools.list()`` and every
``tools.help(name)`` (for the actor and a teammate with and without memory), every refusal template as each agent
reads it, the run record's common section and each role's roster lines, the writer's brief under every input it can
have, its first messages (an attention pass with and without a group), the reference files beside its inputs, the
helper's brief, the fork's and a segment fork's instructions, the guides, the librarian's prompts, the kinship judge's
brief and a card it is shown, the land_capture reply lines, the re-pin notice, and the gate's own lines as real merges
write them (refusal reasons, notes and item labels).
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from types import SimpleNamespace

#: The flags every optional tool needs, so every tool's text is rendered.
EVERY = frozenset({"pool", "similar", "helper"})


def _tools(out: dict[str, str]) -> None:
    from unify.common import tool_registry as R

    for agent in R.AGENTS:
        names = R.allow_list(agent, EVERY)
        out[f"rendered:tools:{agent}"] = json.dumps(R.json_tools(agent)) + "\n" + R.render_stub(names, agent)
        out[f"rendered:preamble:{agent}"] = R.preamble(agent, {"environment": "Your box.", "flags": EVERY})
        for memory in (True, False) if agent in ("actor", "teammate") else (True,):
            tag = "" if memory else ":off"
            out[f"rendered:list:{agent}{tag}"] = R.list_text(names, agent, memory=memory)
            for name in names:
                out[f"rendered:help:{agent}{tag}:{name}"] = R.help_text(name, agent, memory=memory)
    templates = []
    for spec in R.registry().values():
        templates += list(spec.refusals.values())
        for texts in [*spec.role_text.values(), spec.off_text]:
            templates += list((texts or {}).get("refusals", {}).values())
    out["rendered:refusals"] = (
        json.dumps(R.refusal_codes()) + "\n" + json.dumps(R.GENERAL_REFUSALS) + "\n" + "\n".join(templates)
    )
    out["rendered:common_section"] = R.COMMON_SECTION


def _writer(out: dict[str, str]) -> None:
    from unify.memory import attention_tools as at
    from unify.memory import helpers
    from unify.memory import land_capture as lc
    from unify.memory import library_prompts as lp

    names = ("helper", "similar")
    for values in itertools.product((False, True), repeat=len(names)):
        out[f"rendered:write:{values}"] = lp.write_brief_now(**dict(zip(names, values)))
    for similar in (False, True):
        out[f"rendered:helper:{similar}"] = helpers.helper_brief(similar)
    for name, text in lp.input_pointer_files().items():
        out[f"rendered:inputs:{name}"] = text
    handed = [SimpleNamespace(entry=f"e{i}", kind=k) for i, k in enumerate(
        ("new_experience", "own_check_failed", "use_went_badly", "late_evidence", "repeated_use", "deferred", "message"))]
    for n in (1, len(handed)):
        out[f"rendered:attention_first:{n}"] = at.first_message(
            SimpleNamespace(pass_id="p1"), SimpleNamespace(attention=handed[:n]), "2700")
    group = {"related_closed": [{"entry": "e9"}], "library_items": ["memory.a.b:c"]}
    for n in (1, 2):
        out[f"rendered:attention_first:group:{n}"] = at.first_message(
            SimpleNamespace(pass_id="p1", group=group), SimpleNamespace(attention=handed[:n]), "2700")
    out["rendered:land_capture_reply"] = "\n".join(
        (lc.REPLY_RECIPE.format(recipe="memory.p.procedures:apply_f", cases="c1"), lc.REPLY_NO_RESULT_FIXTURE,
         lc.REPLY_SECTIONS))  # fmt: skip


def _others(out: dict[str, str], tmp: Path) -> None:
    from unify.agents import run_record as rr
    from unify.memory import kinship, librarian
    from unify.memory import library_prompts as lp
    from unify.memory.integration import fork, in_episode

    out["rendered:fork"] = fork.fork_message("`request`: (a request)") + "\n" + fork.fork_instruction()
    out["rendered:segment_instruction"] = fork.segment_instruction(2, 5) + "\n" + fork.segment_instruction(None, None)
    out["rendered:guide"] = lp.GUIDE_LIBRARIAN
    out["rendered:librarian"] = librarian.system_prompt() + "\n" + lp.GUIDE_LIBRARIAN
    out["rendered:librarian_evidence"] = (
        librarian.system_prompt(evidence=True) + "\n" + librarian.evidence_question_message("q", "m", 1)
    )
    for role in rr.AGENT_ROLES:
        out[f"rendered:record_lines:{role}"] = rr.prompt_lines(f"{role}-1", role, spawner="actor-1")
    out["rendered:repin_notice"] = in_episode.notice(
        "a" * 40, ["notes/a.md"], ["notes/b.md"], [("notes/c.md", "retired")], True)
    cap = tmp / "judge-capture"
    cap.mkdir(parents=True, exist_ok=True)
    (cap / "procedure.py").write_text('def run(value):\n    """Count the rows."""\n    return len(value)\n')
    profile = SimpleNamespace(sequence=(), input=frozenset({"rows:list"}), outcome=None)
    messages, _ = kinship.question((profile, cap), [("e1", profile, cap)])
    out["rendered:judge"] = "\n".join(m["content"] for m in messages)


def gate_texts(tmp: Path) -> dict[str, str]:
    """The gate's own lines as real merges write them: for each row of the gate's behaviour table, the merge's
    reasons (refusals and notes) and every item's labels."""
    from tests.memory import library_example_world as lw
    from tests.memory.test_gate_behaviour_table import ROWS
    from unify.memory import gate_policy as gp
    from unify.memory.gate import RED_TEST_HINT

    lines = [gp.NOT_RUN_GATE, RED_TEST_HINT, gp.FAILED_SOURCE_LABEL, gp.NO_RECIPE_LABEL]
    for n, (name, row) in enumerate(ROWS.items()):
        res, _, _ = row(lw.make_world(tmp / f"gate-{n}"))
        lines += list(res.reasons)
        for rec in res.verification.values():
            lines += list(rec.get("label_findings") or [])
    return {"rendered:gate": "\n".join(lines)}


def rendered(tmp: Path) -> dict[str, str]:
    """Every rendered text (module docstring), by source name."""
    out: dict[str, str] = {}
    _tools(out)
    _writer(out)
    _others(out, Path(tmp))
    out.update(gate_texts(Path(tmp)))
    return out

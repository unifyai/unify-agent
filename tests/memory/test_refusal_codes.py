"""Refusal form codes: every refusal text a memory tool returns is rendered from a spec's refusal forms, the codes
are listed by tool and written at a pass's start, the briefs state the operational limits from their constants,
the pass keeps the manifest of its last accepted finish, and an uncapped repair message states no round count.
"""

import asyncio
import json
import re
import string
from decimal import Decimal
from pathlib import Path

import pytest

from unify.memory import helpers, library_prompts, tool_spec, views, writer_pass
from unify.memory.gate import GateResult
from unify.memory.repair import REPAIR_ROUNDS, repair_message
from unify.memory.trigger import PassRequest
from tests.memory.test_writer_library_tools import EPS
from tests.memory.test_writer_pass import Turns, _call, _write, _writer, needs_bwrap
from tests.memory.writer_library_items import FINISH_CELL

PKG = Path(writer_pass.__file__).parent

#: The modules whose replies reach an agent as tool results (handlers, readers and the pass loop).
TOOL_MODULES = (
    "writer_tools.py",
    "writer_pass.py",
    "helpers.py",
    "similar.py",
    "views.py",
    "fixtures.py",
    "replay_verify.py",
    "pool.py",
    "batch_map.py",
    "staging.py",
)

#: A string literal (any prefix) that starts with "refused:".
_REFUSED_LITERAL = re.compile(r"""(?<![\w])[rRbBuUfF]{0,2}(?:"|')refused:""")


def _fields(template: str) -> dict:
    return {name: "x" for _, name, _, _ in string.Formatter().parse(template) if name}


def test_no_refusal_text_is_written_outside_the_tool_spec():
    found = []
    for name in TOOL_MODULES:
        for i, line in enumerate((PKG / name).read_text().splitlines(), 1):
            if _REFUSED_LITERAL.search(line):
                found.append(f"{name}:{i}: {line.strip()}")
    assert found == []
    # the templates live in the shared registry
    assert _REFUSED_LITERAL.search((PKG.parent / "common" / "tool_registry.py").read_text())


def test_refusal_codes_list_every_form_of_every_tool_and_each_renders():
    codes = tool_spec.refusal_codes()
    for spec in tool_spec.registry():
        assert set(spec.refusals) <= set(codes[spec.name]), spec.name
    assert codes[tool_spec.GENERAL] == sorted(tool_spec.GENERAL_REFUSALS)

    templates = {(s.name, f): t for s in tool_spec.registry() for f, t in s.refusals.items()}
    for tool, forms in codes.items():
        for form in forms:
            assert form.isidentifier(), (tool, form)
            template = templates.get((tool, form), tool_spec.GENERAL_REFUSALS.get(form))
            text = tool_spec.refusal(tool, form, **_fields(template))
            assert (
                isinstance(text, tool_spec.Refusal) and tool_spec.form_of(text) == form
            )
            assert text.startswith(("refused: ", *tool_spec.KEPT_STARTS)), (tool, form)
    with pytest.raises(KeyError):
        tool_spec.refusal("read", "no_such_form")


def test_helper_refusals_carry_their_own_form_code(tmp_path):
    out = helpers._Staging(tmp_path / "s" / "call-0").stage("../x.py", "x")
    assert tool_spec.form_of(out) == "path"
    assert helpers._refusal_form(out)["form"] == "path"
    done = tool_spec.refusal(tool_spec.GENERAL, "task_done")
    assert (
        done == "not run: task done"
        and helpers._refusal_form(done)["form"] == "task_done"
    )
    assert tool_spec.form_of("staged notes.md (3 bytes)") is None


def test_grep_problems_are_raised_and_answered_as_the_grep_form(tmp_path):
    root = tmp_path / "inputs"
    root.mkdir()
    with pytest.raises(views.PatternProblem, match="invalid pattern"):
        views.grep("(", "/inputs", {"/inputs": root})
    with pytest.raises(views.PatternProblem, match="invalid pattern"):
        views.grep_bounded("(", "/inputs", {"/inputs": root}, timeout_s=10)


def test_the_pass_writes_its_refusal_codes_at_its_start(tmp_path):
    model = Turns([_call("a", "read", {"path": "/inputs"})])
    _, _, writer = _writer(
        tmp_path,
        model,
        max_calls=3,
        pass_record_root=str(tmp_path / "out"),
    )
    writer.load = EPS.__getitem__
    asyncio.run(writer.run(PassRequest("incremental", "svc", ["e1"], False), "p1"))
    path = tmp_path / "out" / "p1" / "refusal-codes.json"
    assert path.is_file() and not path.is_symlink()
    assert json.loads(path.read_text()) == tool_spec.refusal_codes()
    # every refusal recorded in the tool events is one of the listed codes
    codes = json.loads(path.read_text())
    for row in writer.tool_events:
        if row["refusal"] is not None:
            assert (
                row["refusal"]["form"] in codes.get(row["tool"], []) + codes["*"]
            ), row


def test_an_existing_refusal_codes_file_is_never_written_over(tmp_path):
    folder = tmp_path / "out" / "p1"
    folder.mkdir(parents=True)
    (folder / "refusal-codes.json").write_text("earlier\n")
    model = Turns([_call("a", "read", {"path": "/inputs"})])
    _, _, writer = _writer(
        tmp_path,
        model,
        max_calls=3,
        pass_record_root=str(tmp_path / "out"),
    )
    writer.load = EPS.__getitem__
    asyncio.run(writer.run(PassRequest("incremental", "svc", ["e1"], False), "p1"))
    assert (folder / "refusal-codes.json").read_text() == "earlier\n"


def test_the_briefs_state_the_operational_limits_from_their_constants():
    """Rev 2's in-code brief states the limits from their constants; the fork's staging quota is the fork's text
    (test_writer_tool_surface) and the compaction is stated by the note that follows a restart
    (``writer_pass.RESTART_POINTER``), so neither is in the writer's brief."""
    finish_cap = f"up to {writer_pass.FINISH_G1_REFUSALS} times in a pass"
    capped_rounds = f"At most {REPAIR_ROUNDS} repair rounds follow the first refused `tools.finish`"
    uncapped_rounds = "There is no set number of repair rounds"
    full = library_prompts.write_brief_now(finish_capped=True)
    for sentence in (finish_cap, capped_rounds):
        assert full.count(sentence) == 1, sentence
    assert uncapped_rounds not in full
    # an uncapped pass: neither cap is stated
    plain = library_prompts.write_brief_now()
    assert plain.count(uncapped_rounds) == 1
    assert finish_cap not in plain and capped_rounds not in plain
    for text in (full, plain):
        assert "budget" not in text.lower() and "<<" not in text


def test_the_pass_states_the_finish_cap_as_it_is_configured(tmp_path):
    _, _, writer = _writer(
        tmp_path,
        Turns([]),
        max_usd=Decimal("1"),
    )
    # rev 2 always compacts, so the brief's inputs carry no compaction flag
    assert "compact" not in writer._write_inputs()
    assert writer._write_inputs()["finish_capped"] is True
    _, _, uncapped = _writer(tmp_path / "u", Turns([]), max_usd=None)
    assert uncapped._write_inputs()["finish_capped"] is False


def test_accepted_manifest_is_the_raw_dict_or_none(tmp_path):
    assert writer_pass._accepted_manifest(tmp_path) is None
    (tmp_path / ".pass").mkdir()
    (tmp_path / ".pass" / "manifest.json").write_text("[1]")
    assert writer_pass._accepted_manifest(tmp_path) is None
    (tmp_path / ".pass" / "manifest.json").write_text(
        '{"items": [], "kept": {"a": "b"}}',
    )
    assert writer_pass._accepted_manifest(tmp_path) == {"items": [], "kept": {"a": "b"}}


@needs_bwrap
def test_final_manifest_is_set_by_an_accepted_finish_only(tmp_path):
    manifest = {"items": [], "summary": "nothing to keep"}
    model = Turns(
        [
            _call("f0", "finish", {"summary": "too early"}),
            _call(
                "w",
                "execute_code",
                {"code": _write("/memory/.pass/manifest.json", json.dumps(manifest))},
            ),
            _call("f1", "finish", {"summary": "nothing to keep"}),
        ],
    )
    _, _, writer = _writer(tmp_path, model, max_calls=6)
    writer.load = EPS.__getitem__
    assert writer.final_manifest is None
    asyncio.run(writer.run(PassRequest("incremental", "svc", ["e1"], False), "p1"))
    # the first finish (no manifest yet) is refused; the second is accepted (the pass ends there, so the model
    # never sees its reply)
    # rev 2: finish is called from a cell, which prints the interpreter's notices before the refusal
    assert model.outputs["f0"].startswith(FINISH_CELL + "not finished: ")
    assert writer.final_manifest == manifest


def test_an_uncapped_repair_message_states_no_round_count():
    res = GateResult(
        False,
        {},
        ["G3: x"],
        ["G3"],
        items_refused={"memory.a.b:f": ["G3"]},
    )
    capped = repair_message(1, "/inputs/gate/result-0.md", res, REPAIR_ROUNDS)
    assert f"repair round 1 of {REPAIR_ROUNDS}" in capped
    uncapped = repair_message(1, "/inputs/gate/result-0.md", res, None)
    assert "repair round 1." in uncapped
    assert not re.search(r"round \d+ of \d+", uncapped)


def test_a_refusal_survives_copy_and_pickle_as_a_model_call_copies_its_messages():
    import copy
    import pickle

    from unify.memory.tool_spec import Refusal, refusal

    r = refusal("similar", "args")
    assert isinstance(r, Refusal)
    for twin in (
        copy.deepcopy([{"content": r}])[0]["content"],
        pickle.loads(pickle.dumps(r)),
    ):
        assert twin == r and twin.tool == "similar" and twin.form == "args"

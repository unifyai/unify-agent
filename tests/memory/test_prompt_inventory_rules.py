"""The prompt texts every agent reads when tools are called from code (``UNIFY_TOOLS_IN_CODE=on``).

With the setting on, each agent has one model-API tool, ``execute_code``, and calls every other tool in its cells as
``tools.<name>(...)``. The briefs then name tools in that form and leave each tool's details to ``tools.help(name)``.
Three things must hold for those texts:

* every rule a check enforces is still stated where the agent acts on it (:data:`RULES`: one stable phrase per
  inventory row, and the text it must be in);
* no brief names a tool its agent cannot call (:data:`ALLOWED`, the per-agent allow-lists);
* no text says "stored function", and none names a model-API tool that is no longer offered.

With the setting off every text is today's, byte for byte; the existing wording tests cover that arm.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import pytest

from unify.common import in_code_switch

#: Names every agent's ``tools`` module has besides its allow-list: listing, help, the shell, and the two exceptions a
#: call raises (the shared opening that starts every brief names them).
COMMON = frozenset({"list", "help", "sh", "Refused", "NotRun"})

#: Each agent's in-code tools (the design's per-agent allow-lists); the writer's optional tools by the flag that
#: offers them.
WRITER_ALWAYS = frozenset(
    {
        "read",
        "grep",
        "overview",
        "read_episode",
        "fixture",
        "replay",
        "land_capture",
        "check",
        "finish",
        "send",
        "messages",
        "wait",
        "who",
        "reply",
        "stop",
    },
)
WRITER_OPTIONAL = {"pool": "pool", "helper": "delegate", "similar": "similar", "edit": ("write", "edit")}
ALLOWED = {
    "curator": frozenset(
        {
            "read",
            "grep",
            "overview",
            "read_episode",
            "fixture",
            "send",
            "messages",
            "wait",
            "who",
            "reply",
            "stop",
            "write",
            "edit",
            "check",
            "finish",
        },
    ),
    "helper": frozenset(
        {"read", "grep", "overview", "read_episode", "similar", "stage", "send", "messages", "wait", "who", "ask", "done"},
    ),
    "actor": frozenset(
        {"read", "grep", "install", "send", "messages", "wait", "who", "reply", "stop", "delegate", "answer"},
    ),
    "fork": frozenset(),
}

#: The model-API tools each agent had before, none of which it has with tools called from code.
MEMORY_JSON_TOOLS = frozenset(
    {
        "read",
        "grep",
        "overview",
        "read_episode",
        "pool",
        "fixture",
        "replay",
        "land_capture",
        "check",
        "finish",
        "similar",
        "delegate",
        "send",
        "messages",
        "wait",
        "who",
        "reply",
        "stop",
        "write",
        "edit",
        "stage",
        "done",
        "ask",
    },
)
ACTOR_JSON_TOOLS = frozenset({"final_response", "send_response", "store_skills"})
#: A backticked word that is a manifest key, not a tool: the writer's manifest has a `pool` block.
MANIFEST_KEYS = frozenset({"pool"})

#: Every inventory row of kind "rule a check enforces" (and the facts the briefs state from a checked constant),
#: with the text the agent reads it in and a stable phrase of the rule. Rows a refusal states are checked in the
#: refusal's own text.
RULES: tuple[tuple[str, str, str], ...] = (
    ("B86", "writer", "Functions go in `memory/<package>/<module>.py`"),
    ("B87", "writer", "Each package has an `__init__.py` whose docstring is one paragraph"),
    ("B88", "writer", "an `Example:` section of one line naming one of the function's tests"),
    ("B89", "writer", "must name an item that exists"),
    ("B90", "writer", "plain pytest, importing only the library and the standard library"),
    ("B93", "writer", '{"items": [{"item": "<id>", "kind": "function" or "note"'),
    ("B94", "writer", "A note's entry takes `item`, `kind` and `source_episodes`"),
    ("B95", "writer", "its `tests` name at least one test file"),
    ("B96", "writer", "A field with a fixed domain is declared in `field_types` (form: /inputs/README.md)"),
    ("B96", "inputs_readme", "declares a field with a fixed domain"),
    ("B98", "writer", "`covers` takes typed covers only"),
    ("B99", "writer", "`support` lists helper files"),
    ("B101", "writer", "keeps a moved item's history"),
    ("B104", "writer", "An item's `adopts` lists the drafts it drew on"),
    ("B109", "writer", "`pool` records decisions on pool entries"),
    ("B113", "writer", "A procedure is verified by replaying its episode"),
    ("B115", "writer", 'raises `MemoryInputError("<what was expected>")` on an input it was not built for'),
    ("B127", "writer", "add a test that fails on the old version and passes on yours"),
    ("B128", "writer", "This pass never removes library items and never deletes or weakens a test"),
    ("B129", "writer", "name it in `tests_changed` with the reason"),
    ("B130", "writer", "in this order of trust"),
    ("B131", "writer", "At least one call returns on arguments loaded from fixtures made with `tools.fixture`"),
    ("B132", "writer", "one episode is enough"),
    ("B133", "writer", "A call that refuses or raises counts for nothing"),
    ("B134", "writer", "equals an exact trusted value"),
    ("B135", "writer", "Never compute an expected value with the function's own expression"),
    ("B136", "writer", "never assert on the clock"),
    ("B136", "curator", "never assert on the clock"),
    ("B138", "writer", "recorded-inputs file"),
    ("B139", "writer", "more recorded inputs of the same kind to a copy of that file"),
    ("B141", "writer", "Never store credentials, session tokens or the results of calls with side effects"),
    ("B144", "writer", "% of small mutations of each changed function"),
    ("B145", "writer", "a breach of a pass-wide rule (the layout, lost tests, fewer items) refuses the round"),
    ("B147", "writer", "G1, layout and provenance"),
    ("B147", "curator", "G1, layout and provenance"),
    ("B148", "writer", "G2, evidence"),
    ("B148", "curator", "G2, evidence"),
    ("B149", "writer", "G3, tests"),
    ("B149", "curator", "G3, tests"),
    ("B150", "writer", "G4, the index"),
    ("B150", "curator", "G4, the index"),
    ("B151", "writer", "a store pass never ends with fewer items"),
    ("B152", "writer", "G6, safety"),
    ("B152", "curator", "G6, safety"),
    ("B157", "writer", "a breach refuses the pass"),
    ("B158", "writer", "model and check calls"),
    ("B162", "writer", "is refused with the reason up to"),
    ("B190", "curator", "keeping each old name as an alias"),
    ("B193", "curator", "with a one-line reason in `retired`"),
    ("B198", "curator", '"deleted": ["<retired id>"]'),
    ("B202", "curator", "keeps the library's existing tests green"),
    ("B204", "curator", "needs a test that fails on the parent and passes on yours"),
    ("B205", "curator", "stay used by a remaining function's tests"),
    ("B206", "curator", "New tests check exact expected values, in this order of trust"),
    ("B210", "curator", "Never store credentials or session tokens"),
    ("B211", "curator", "After `tools.finish(summary)` a deterministic gate checks the candidate"),
    ("B212", "curator", "is refused with the items that import it"),
    ("B213", "curator", "each alias and retirement names an item"),
    ("T58", "writer", "every fixture came through `tools.fixture` unchanged"),
    ("T60", "curator", "every fixture came through `tools.fixture` unchanged"),
    ("B247", "helper", "a `tools.done` without them is refused"),
    ("B268", "fork", "at most 200 files and 4 MiB in all are kept"),
    ("B300", "refusals", "add a test call that returns on values loaded from a fixture of a recorded episode"),
    ("B302", "refusals", "make one call that returns take each structured argument"),
    ("T74a", "refusals", "a link is on the way to"),
    ("A2", "actor", "Your answer is your final reply: a message without a tool call."),
    ("A8", "actor_structured", "answer by calling `tools.answer(answer)`"),
    ("H3", "install_help", "Never install from a cell"),
    ("J16", "compress_context", "becomes the **only**"),
    ("M12", "library_guide", "A function raises `MemoryInputError` on an input it was not built for."),
    ("M19", "library_head", "Generated by the harness from the library."),
    ("N1", "refusals", "The step limit for this request is reached"),
    ("N2", "refusals", "no more tools can be called for it"),
    ("P7", "refusals", "write long content to a file and post its path"),
)

#: Rule rows whose text another part of the build writes (the shared opening, each tool's help, the actor's
#: execute_code description): not checked here.
ELSEWHERE = {
    "A27": "the shared opening's environment sentence",
    "A43": "execute_code's description",
    "J3": "execute_code's description",
    "J7": "execute_code's description",
    "J9": "execute_code's description",
    "J10": "execute_code's description",
    "H5": 'tools.help("read")',
    "H6": 'tools.help("grep")',
    "P2": "the actor's own text",
    "P3": 'tools.help("wait")',
    "H11": 'tools.help("wait")',
}


@pytest.fixture
def on(monkeypatch):
    monkeypatch.setattr(in_code_switch, "tools_in_code", lambda: True)


@pytest.fixture
def off(monkeypatch):
    monkeypatch.setattr(in_code_switch, "tools_in_code", lambda: False)


def _writer_briefs() -> dict[str, tuple[str, frozenset[str]]]:
    """The writer's briefs with no optional input and with every one, each with the flags it was built with."""
    from unify.memory import library_prompts as lp

    full = {"staging": True, "pool": True, "helper": True, "similar": True, "edit_tools": True}
    return {
        "writer base": (lp.write_brief_now(), frozenset()),
        "writer full": (
            lp.write_brief_now(**full, compact=True, finish_capped=True),
            frozenset({"pool", "helper", "similar", "edit"}),
        ),
    }


def _writer_allowed(flags: frozenset[str]) -> frozenset[str]:
    names = set(WRITER_ALWAYS)
    for flag, tools in WRITER_OPTIONAL.items():
        if flag in flags:
            names.update((tools,) if isinstance(tools, str) else tools)
    return frozenset(names)


def _actor_prompt(**kw) -> str:
    from unify.actor import prompt_builders as pb
    from unify.actor.core_surface import PromptSurface

    return pb.build_code_act_prompt(
        environments={},
        core=PromptSurface(functions=False, guidance=False, **kw),
        session_sections=False,
    )


def _sources(*modules: str) -> str:
    import importlib

    return "\n".join(Path(importlib.import_module(m).__file__).read_text(encoding="utf-8") for m in modules)


def _texts() -> dict[str, str]:
    """Every text the rows are checked in, as the agents read them with tools called from code."""
    from unify.actor import core_surface
    from unify.common._async_tool import cache_discipline, context_compression
    from unify.memory import helpers, library_index
    from unify.memory import library_prompts as lp
    from unify.memory.integration import fork

    writers = _writer_briefs()
    files = lp.input_pointer_files(staging=True, pool=True)
    return {
        "writer": writers["writer full"][0],
        "writer base": writers["writer base"][0],
        "curator": lp.curate_brief_now(compact=True, finish_capped=True, edit_tools=True),
        "helper": helpers.helper_brief(True),
        "fork": fork.fork_instruction(),
        "actor": _actor_prompt(),
        "actor_structured": _actor_prompt(structured=True),
        "library_guide": lp.guide_library(),
        "library_head": library_index.INDEX_HEAD_IN_CODE,
        "install_help": core_surface.install_doc(),
        "compress_context": context_compression.compress_context_tool().__doc__ or "",
        "compaction": cache_discipline.compression_fork_instruction(),
        "inputs_readme": files[lp.INPUTS_README_FILE],
        "refusals": "\n".join(
            [
                lp.EPISODES_FIX,
                lp.GROUNDED_FIX,
                _sources(
                    "unify.memory.writer_pass",
                    "unify.common._async_tool.loop",
                    "unify.agents.record",
                ),
            ],
        ),
    }


@pytest.mark.parametrize("row,where,phrase", RULES, ids=[f"{r}-{w}" for r, w, _ in RULES])
def test_every_checked_rule_is_still_stated_where_the_agent_acts(on, row, where, phrase):
    text = " ".join(_texts()[where].split())
    assert " ".join(phrase.split()) in text, f"{row}: {phrase!r} is not in the {where} text"


def test_the_rows_stated_elsewhere_are_not_rows_this_file_checks():
    assert not set(ELSEWHERE) & {row for row, _, _ in RULES}


def test_the_writer_brief_states_each_flagged_rule_only_with_its_input(on):
    base = _texts()["writer base"]
    assert "An item's `adopts` lists" not in base and "`pool` records decisions" not in base


def _tool_calls(text: str) -> set[str]:
    return set(re.findall(r"\btools\.([A-Za-z_]+)", text))


def test_no_brief_names_a_tool_its_agent_cannot_call(on):
    from unify.memory import helpers

    for name, (brief, flags) in _writer_briefs().items():
        extra = _tool_calls(brief) - _writer_allowed(flags) - COMMON
        assert not extra, f"{name} names {sorted(extra)}"
    texts = _texts()
    for agent in ("curator", "actor", "fork"):
        extra = _tool_calls(texts[agent]) - ALLOWED[agent] - COMMON
        assert not extra, f"{agent} names {sorted(extra)}"
    assert not _tool_calls(texts["actor_structured"]) - ALLOWED["actor"] - COMMON
    for similar in (True, False):
        allowed = ALLOWED["helper"] - (set() if similar else {"similar"})
        extra = _tool_calls(helpers.helper_brief(similar)) - allowed - COMMON
        assert not extra, f"helper (similar={similar}) names {sorted(extra)}"


def _bare_tool_names(text: str, names: frozenset[str]) -> set[str]:
    """Backticked bare tool names: `name` or `name(...)`, never `tools.name`."""
    return {m for m in re.findall(r"`([a-z_]+)(?:`|\()", text) if m in names}


def test_no_text_says_stored_function_or_names_a_tool_no_longer_offered(on):
    texts = _texts()
    memory = ("writer", "writer base", "curator", "helper", "fork", "compaction")
    for name in memory:
        assert "stored function" not in texts[name].lower(), name
        bare = _bare_tool_names(texts[name], MEMORY_JSON_TOOLS) - (MANIFEST_KEYS if "writer" in name else set())
        assert not bare, f"{name} names {sorted(bare)} as a model-API tool"
    for name in ("library_guide", "library_head", "install_help", "compaction"):
        assert "stored function" not in texts[name].lower(), name
    for name in ("actor", "actor_structured"):
        from unify.actor import prompt_builders as pb

        # the role (its first paragraph) is the actor's own text, rewritten in its own change; with the tools in
        # code it is the role that states the one answer rule
        role = pb._lean_role_in_code()
        assert role in texts[name], name
        body = texts[name].split(role, 1)[-1]
        assert "stored function" not in body.lower(), name
        assert not _bare_tool_names(texts[name], ACTOR_JSON_TOOLS), name
        assert "### Tools" not in texts[name] and "### Sandbox Objects" not in texts[name]
    from unify.memory import helpers

    for similar in (True, False):
        assert not _bare_tool_names(helpers.helper_brief(similar), MEMORY_JSON_TOOLS)


def test_the_team_record_section_is_left_to_the_shared_opening(on):
    from unify.agents import binding

    view = binding.Binding.__new__(binding.Binding)
    assert view.prompt_section() == ""


def test_with_the_setting_off_every_text_is_todays(off):
    from unify.actor import core_surface
    from unify.agents import binding
    from unify.common._async_tool import cache_discipline, context_compression, in_code_texts
    from unify.memory import helpers, library_index, messaging
    from unify.memory import library_prompts as lp
    from unify.memory.integration import fork

    assert lp.guide_library() == lp.GUIDE_LIBRARY
    assert lp.candidate_contract() == lp.CANDIDATE_CONTRACT
    assert messaging.record_text() == messaging.RECORD_TEXT
    assert lp.EPISODES_RULE in lp.write_brief_now()
    assert "# Tools\n" in lp.write_brief_now() and "# Tools\n" in lp.curate_brief_now()
    # in the persistent interpreter (the default) the brief starts with the shared sandbox paragraph
    assert helpers.helper_brief(True).endswith(
        "\n\n"
        + helpers.HELPER_BRIEF.replace("{similar}", helpers.SIMILAR_LINE).replace(
            "<<execute_code>>",
            helpers.EXECUTE_CODE_LINE,
        ),
    )
    assert fork.fork_instruction() == fork.INSTRUCTION
    assert cache_discipline.compression_fork_instruction() == cache_discipline.COMPRESSION_FORK_INSTRUCTION
    assert context_compression.compress_context_tool() is context_compression.compress_context
    assert core_surface.install_doc() == core_surface._INSTALL_DOC
    assert in_code_texts.send_response_description() == in_code_texts.SEND_RESPONSE_DESCRIPTION
    assert in_code_texts.unknown_tool_text("x").startswith("⚠️ Error: Tool 'x' is not available.")
    assert library_index.prompt_view_head(library_index.INDEX_HEAD + "x") == library_index.INDEX_HEAD + "x"
    view = binding.Binding.__new__(binding.Binding)
    assert view.prompt_section() == binding.PROMPT_SECTION


def test_the_in_code_texts_that_follow_a_switch(on):
    from unify.common._async_tool import in_code_texts
    from unify.memory import library_index
    from unify.memory.gate import GateResult
    from unify.memory.repair import repair_message

    res = GateResult(False, {}, ["G3: x"], ["G3"], items_refused={"memory.a.b:c": ["G3"]})
    msg = repair_message(1, "/inputs/gate/result-0.md", res)
    assert "read it with tools.read(path)" in msg and "call tools.finish(summary) again" in msg
    assert "notifications" not in in_code_texts.send_response_description()
    assert "`tools.list()`" in in_code_texts.unknown_tool_text("x")
    view = library_index.prompt_view_head(library_index.INDEX_HEAD + "\n## memory.a\n")
    assert view == library_index.INDEX_HEAD_IN_CODE + "\n## memory.a\n"
    assert "examples from one episode" not in view and "never edit it" not in view


def test_the_pointer_files_hold_the_tables_the_brief_points_to(on):
    from unify.memory import library_prompts as lp

    files = lp.input_pointer_files(staging=True, pool=True)
    keys = files[lp.BATCH_MAP_KEYS_FILE]
    for key, _ in lp.row_keys(staging=True, pool=True):
        assert f"- `{key}` (" in keys, key
    readme = files[lp.INPUTS_README_FILE]
    assert "Pool entry kinds" in readme and "Pool entry kinds" not in lp.input_pointer_files()[lp.INPUTS_README_FILE]
    brief = _texts()["writer"]
    assert "/inputs/batch_map_keys.md" in brief and "/inputs/README.md" in brief
    assert "`episode_id` (str)" not in brief


def test_prompt_sizes_off_and_on(monkeypatch):
    """Each role's prompt size in characters, with the setting off and on (printed and warned for the report)."""
    from unify.agents import binding
    from unify.common._async_tool import cache_discipline
    from unify.memory import helpers, library_index
    from unify.memory import library_prompts as lp
    from unify.memory.integration import fork

    def sizes() -> dict[str, int]:
        writers = _writer_briefs()
        return {
            "writer base brief": len(writers["writer base"][0]),
            "writer full brief": len(writers["writer full"][0]),
            "curator brief": len(lp.curate_brief_now(compact=True, finish_capped=True, edit_tools=True)),
            "helper brief (no similar)": len(helpers.helper_brief(False)),
            "helper brief (similar)": len(helpers.helper_brief(True)),
            "fork instruction": len(fork.fork_instruction()),
            "actor system prompt (no clock, workspace or library)": len(_actor_prompt()),
            "actor library guide + index head": len(lp.guide_library())
            + len(library_index.prompt_view_head(library_index.INDEX_HEAD)),
            "actor team record section": len(binding.Binding.__new__(binding.Binding).prompt_section()),
            "compaction instruction": len(cache_discipline.compression_fork_instruction()),
        }

    monkeypatch.setattr(in_code_switch, "tools_in_code", lambda: False)
    before = sizes()
    monkeypatch.setattr(in_code_switch, "tools_in_code", lambda: True)
    after = sizes()
    # the memory briefs now start with the shared opening every agent reads first; the role's own text is what the
    # in-code rewrite trimmed, so that is what is compared, and the opening's share is reported beside it
    opening = _opening_sizes()
    lines = [
        f"{k}: off {before[k]}, on {after[k]} ({after[k] - before[k]:+d})"
        + (f", of which the opening {opening[k]}" if k in opening else "")
        for k in before
    ]
    report = "PROMPT SIZES (characters)\n" + "\n".join(lines)
    print(report)
    warnings.warn(report, stacklevel=1)
    for k in before:
        assert after[k] - opening.get(k, 0) <= before[k], k


def _opening_sizes() -> dict[str, int]:
    """With the setting on, the characters the shared opening and its blank line add to each memory brief."""
    from unify.memory import helpers
    from unify.memory import library_prompts as lp

    texts = {
        "writer base brief": _writer_briefs()["writer base"][0],
        "writer full brief": _writer_briefs()["writer full"][0],
        "curator brief": lp.curate_brief_now(compact=True, finish_capped=True, edit_tools=True),
        "helper brief (no similar)": helpers.helper_brief(False),
        "helper brief (similar)": helpers.helper_brief(True),
    }
    out = {}
    for name, text in texts.items():
        start = "\n\nYou are a helper" if name.startswith("helper") else "\n\n# Role\n"
        head, sep, _ = text.partition(start)
        assert sep and head.startswith("## How you work here"), name
        out[name] = len(head) + 2
    return out

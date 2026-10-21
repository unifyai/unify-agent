"""The memory prompts: the actor's guide, the writer's store-pass (WRITE) brief and its curate-pass (CURATE) brief.

Each text serves one role and states its role, inputs, outputs, rules and when it is done. None carries another
role's instructions, any benchmark's vocabulary, or an instruction to check a task's examples:
``tests/memory/test_prompts_library_mode.py`` pins that, and ``test_prompts_library_contradictions.py`` pins
:data:`RULES` (each rule stated in its owners' prompts only) and :data:`V2_RETIRED` (no retired rule survives).

Every number a brief states is read from the constant that enforces it (:func:`write_brief_now`,
:func:`curate_brief_now`), so a brief cannot drift from its gate. A change to any of these texts is a reviewed
change. Read-only access is enforced by the mounts, not by wording.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from decimal import Decimal
from typing import Mapping

# --- the actor's guide -----------------------------------------------------------------------

#: The bullet optional pull helpers are inserted before. Constant bytes.
FILES_LINE = "- The library is also plain files at the location below, so `grep` and file reads work on it.\n"
#: :data:`FILES_LINE` when tools are called from code (the bare ``grep`` is ``tools.grep`` there).
FILES_LINE_IN_CODE = (
    "- The library is also plain files at the location below, so `tools.grep` and `tools.read` work on it.\n"
)

def guide_library_mode(extra_lines: tuple[str, ...] = ()) -> str:
    """The guide, with *extra_lines* (optional pull helpers, constant per run) before :data:`FILES_LINE`; with tools
    called from code (``UNIFY_TOOLS_IN_CODE=on``), :data:`GUIDE_LIBRARY_IN_CODE`."""
    guide = guide_library()
    if not extra_lines:
        return guide
    line = FILES_LINE_IN_CODE if FILES_LINE_IN_CODE in guide else FILES_LINE
    return guide.replace(line, "".join(extra_lines) + line, 1)


def guide_library() -> str:
    """The inline-index guide the actor reads: :data:`GUIDE_LIBRARY`, or :data:`GUIDE_LIBRARY_IN_CODE` when tools are
    called from code."""
    return GUIDE_LIBRARY_IN_CODE


def _in_code() -> bool:
    from unify.common import in_code_switch

    return in_code_switch.tools_in_code()


#: The one line that points the in-code texts to the library's own documentation: each call's arguments, results
#: and errors are in its help, never in a prompt.
MEMORY_HELP_LINE = "`help(memory)` shows the library's calls and types."

#: The in-code look-first bullets, first under "Finding items": when to look, and where. With the librarian off the
#: actor looks with ``memory.items`` (in the index, when the prompt carries it); with it on, ``memory.ask`` is the
#: librarian, and :data:`ASK_LINE_IN_CODE` says when to ask and that an answer is a lead, not proof. The switch is
#: fixed for a run, so each guide's bytes are too.
LOOK_LINE_IN_CODE = (
    "- At the start of a request, and before you work a step out yourself, look for an item that applies in the "
    "index at the end of this prompt or with `memory.items`; the index lists every function and note.\n"
)
ASK_LINE_IN_CODE = (
    "- At the start of a request, and before you work a step out yourself, ask the library what applies with "
    "`memory.ask`. What it names can still be wrong for your case.\n"
)
#: The in-code try-first bullet, first under "Using items": open what you will use, then call it.
USE_LINE_IN_CODE = (
    "- When an item applies, open it with `memory.get` and call it before you work the step out yourself. Trying "
    "a fitting item costs one call, and working it out again usually costs many. Its result is unchecked unless "
    "its record says otherwise.\n"
)
#: The in-code refusal and status bullets, after the try-first bullet.
STATUS_LINES_IN_CODE = (
    "- A function raises `MemoryInputError` on an input it was not built for. Then do that step with your own "
    "code.\n"
    "- `on-trial` items passed the library's checks and have had little use; `adopted` items have also been "
    "used without errors. "
    "A function's brackets also give its evidence level: `evidence: unchecked` (stored, but some check of its "
    "evidence did not hold; its record lists which), `evidence: reproduces-its-case` (its tests reproduce its "
    "recorded cases and every evidence check holds) or `evidence: confirmed-by-outcomes` (outcomes after use "
    "confirm it).\n"
)
#: The opening of every in-code guide: the bound name, the help pointer and that using the library is optional.
_OPENING_IN_CODE = (
    "### Memory Library\n"
    "\n"
    "A Python library of tested functions and linked notes, built from earlier work, is already available "
    "in your cells as `memory`, no import needed. " + MEMORY_HELP_LINE + " Using the library is optional: use an "
    "item when it fits, and otherwise work as usual.\n"
    "\n"
)

#: The inline-index guide when tools are called from code (the librarian off): it says when to use which library call
#: and how they combine (look for what applies, open what you will use, call it) and names each call without its
#: arguments; the calls' own help holds their arguments, results and errors. The index head says what each line holds.
GUIDE_LIBRARY_IN_CODE = (
    _OPENING_IN_CODE
    + "Finding items:\n"
    + LOOK_LINE_IN_CODE
    + FILES_LINE_IN_CODE
    + "\n"
    "Using items:\n"
    + USE_LINE_IN_CODE
    + STATUS_LINES_IN_CODE
)

#: The constant pointer to the librarian (``UNIFY_MEMORY_LIBRARIAN=on``): the same bytes for every library, in place
#: of the index. It says who answers ``memory.ask``; the answer's form is in the call's help.
LIBRARIAN_POINTER = (
    "Your library of procedures and notes from earlier work is the `memory` package in your cells. A librarian, "
    "who searches the library by what each item was saved for and reads each item it names, answers `memory.ask`."
)

#: The actor's memory section with the librarian on (the librarian requires the tools in code): the in-code guide
#: with the index bullet replaced by :data:`LIBRARIAN_POINTER`, and nothing after it: no index view, whatever the
#: size of the library, so the library never changes the actor's cached prefix.
GUIDE_LIBRARIAN = (
    _OPENING_IN_CODE
    + "Finding items:\n"
    + "- "
    + LIBRARIAN_POINTER
    + "\n"
    + ASK_LINE_IN_CODE
    + "- `memory.items` browses every function and note in the library.\n"
    + FILES_LINE_IN_CODE
    + "\n"
    "Using items:\n"
    + USE_LINE_IN_CODE
    + STATUS_LINES_IN_CODE
)


# --- the writer's briefs: store pass (WRITE) and curate pass (CURATE) ---------------------------------------------
#
# Each brief is built from stable text plus the configuration of the run (which optional inputs exist, which tools
# are registered for the role), so it is byte-stable for a whole run. Nothing that changes between passes (the
# library index, the pool, use counts, drafts, why a curate pass runs) is inlined: those live in files the brief
# points to. The Tools section is rendered from the tool spec registry (:func:`.tool_spec.tools_section`), so it
# names exactly the tools the role gets.

#: The format of ``summary.md``, the episode-end fork's account of its episode, as the fork's instruction states it.
#: The writer reads each summary first and opens the episode's record only where it needs more; each line's pointer
#: (a part name and a quote copied from that part) is checked host-side, as a measure only (:mod:`.summaries`).
SUMMARY_FORMAT = (
    "`summary.md` has five headed sections, in this order:\n"
    "`## Asked`: what the request asked, in general terms: the kind of task, what it was given and what it had to "
    "produce. Name a specific value only where the task depends on it.\n"
    "`## Approach`: the steps taken, in order, one line each.\n"
    "`## Worked`: what worked, one line each.\n"
    "`## Failed`: what failed and why, one line each; write `none` if nothing failed.\n"
    "`## Outcome`: how the episode ended: what was handed back, and what the environment said about it. If nothing "
    "in the record shows whether it was right, write `not shown`; never guess.\n\n"
    "Every line under Approach, Worked, Failed and Outcome ends with a pointer into the record: a part name in "
    'backticks and a short quote copied exactly from that part, for example (`cell:3` "ValueError: invalid literal '
    'for int()"). Part names are `request`, `observation:<n>`, `cell:<n>`, `action:<n>` and `diff`. Keep it short: '
    "the writer reads it first and opens the record only where it needs more."
)

#: The keys of a batch-map row as :func:`.batch_map.build_batch_map` writes them, each with its type and meaning.
#: The WRITE brief renders this table, and a test checks it against a built row key for key.
BATCH_ROW_KEYS: tuple[tuple[str, str], ...] = (
    ("episode_id", "str"),
    ("memory_main", "str: the library commit the episode ran on"),
    ("regime", "str: the feedback regime the run declared for the episode"),
    (
        "request",
        "str: the start of the request text, a shared block shown as `[shared:<id>]`; the whole request is the "
        "episode's `request` part",
    ),
    ("request_bytes", "int: the whole request's size in bytes"),
    ("observations", "int: how many observations followed the request"),
    (
        "observation_copies",
        "dict: observation index -> the earlier observation it repeats byte for byte",
    ),
    ("cells", "int: how many code cells ran"),
    ("actions", "int: how many tool calls were made"),
    ("calls", "dict: `channel.method` -> count"),
    (
        "errors",
        "list of `{action, error, next_call}`: actions that returned an error, and the call after each",
    ),
    (
        "structural_events",
        "list of `{kind, cell, action}`, `kind` one of `cell_error`, `action_error`, `retry_after_error`: what the "
        "harness sees without reading any environment text; an empty list means none of these happened, not that "
        "the work succeeded",
    ),
    (
        "end_parts",
        "list of str: the parts where the episode ended (its last cell that ran without an error, its last action "
        "and its last observation): where an outcome, if any, was shown",
    ),
    (
        "functions",
        "list of `{name, signature, source, cell, lineno, cell_error, called_later}` (plus `path` when `source` is "
        '`"diff"`): the functions the working model defined; `source` is `"cell"` or `"diff"`, `cell` the defining '
        "cell's index (null for the diff), `cell_error` whether that cell raised, `called_later` how many later "
        "cells call it",
    ),
    (
        "diff_unparsed",
        "list of str: work-tree `.py` files whose added lines did not parse",
    ),
)

#: What the writer is told about an episode whose drafting did not finish: its ``staging.status`` is then
#: ``no reader output: <why>`` (:data:`.staging.NO_OUTPUT`), never ``ok``, and nothing of it is handed on.
STAGING_UNFINISHED = (
    "A row whose `staging.status` is not `ok` has no reader output and says why; read that episode yourself."
)
#: The writer's caution about summaries (the WRITE brief's step 3, with staging on).
SUMMARY_NOT_EVIDENCE = (
    "A summary is a reader's account, not evidence: before a test or an item rests on what a summary says, read "
    "the part it points to."
)

#: Row keys the pass adds when an optional input exists (described only then).
#: The keys of a batch-map row's ``staging`` entry (:func:`.staging.export`, plus ``candidates`` and ``capture``), in
#: the order the row key's text names them. Which reader staged the files is not among them: that stays in the
#: host-side records.
STAGING_ROW_FIELDS = ("status", "files", "refused", "path", "summary", "blocks", "candidates", "capture")
STAGING_ROW_KEY = (
    "staging",
    "dict: `status`, `files`, `refused`, `path`, `summary` (the path of the episode's `summary.md`, or null when "
    "there is none), `blocks` (the drafting reply's block counts, when it left them), `candidates` (list of "
    "`{name, file, cases}`) and `capture` (`{id, land}` when the episode's capture can be landed: its id and the one "
    "call that lands it, its procedure then under /inputs/captures/<id>/ only; else null): what was staged for the "
    "episode under /inputs/staging/<episode>/; a `status` other than `ok` (`no reader output: <why>`) means nothing "
    "was handed on, and says why",
)
CAPTURE_ROW_KEY = (
    "capture",
    "dict or null: `id`, `reproduces` (`reproduces`, `differs` or `unknown`: whether the captured procedure gave "
    "the result the episode handed back), `outcome` (the signal the episode carried, or `failed`), `level`, "
    "`status`, `same_as` (stored captures that gave the same result on this request) and `path`: the attempt the "
    "harness captured from the episode, under /inputs/captures/<id>/",
)
POOL_ROW_KEY = (
    "pool_entries",
    "list of `{id, kind, title}`: the open pool entries that cite the episode",
)
RELATED_ROW_KEY = (
    "related_closed",
    "`{count, entries}`: the closed pool entries whose code shares a normalised code unit with the episode's cells "
    "or with a draft staged from it, or that cite the episode; `entries` lists the first "
    "{page} as `{id, kind, title, status, reason, closed_in, evidence_episodes, shared_code, shared_episodes}`, "
    "`shared_code` mapping what matched (`cells` or the staged draft's ref) to the number of shared units; "
    "`error` instead of a count when the links could not be computed",
)

#: The captured attempts in a store pass's inputs (:mod:`.captures`), named whenever staging is, in its bullet.
CAPTURES_TEXT = (
    "Beside it, /inputs/captures/<id>/ holds attempts the harness captured when their episode ended: `procedure.py` (the "
    "procedure that produced the result the episode handed back), `broke.md` (what broke), `signature.json` (the "
    "structure of its input and result) and `record.json` (its uses and outcomes). When an episode's capture can be "
    "landed, its procedure is here only, not in /inputs/staging/. The harness has already run each "
    "procedure on its recorded request, confined; the episode's batch-map row `capture` says whether it gave the same "
    "result, what signal the episode carried, and which stored captures give the same result there. "
    "{land}A capture is not in the library: it becomes library items only as you write them. A failed capture is "
    "knowledge too: a note of what broke, or a test marked as a known failure.\n"
)


#: The landing tool's sentence in :data:`CAPTURES_TEXT`, when the agent has the tool.
LAND_TEXT = (
    "`{tools}land_capture(capture_id, package, name)` turns one into library items you can change before you "
    "finish; its row's `staging.capture.land` is that call. "
)


def captures_input(in_code: bool, land: bool = True) -> str:
    """:data:`CAPTURES_TEXT`, with the landing tool's sentence when the agent has the tool (*land*), named as the
    agent calls it (in code: ``tools.``)."""
    sentence = LAND_TEXT.replace("{tools}", "tools." if in_code else "") if land else ""
    return CAPTURES_TEXT.replace("{land}", sentence)

#: What each pool entry kind holds (only the kinds :data:`.pool.KINDS` defines are named).
POOL_KIND_TEXT: dict[str, str] = {
    "candidate": "a function with its cases",
    "lesson": "notes",
    "cluster": "code that recurs across episodes",
    "draft": "work the gate refused, with its result",
    "deferred": "episodes a pass kept open to revisit",
    "unread": "episodes no pass has read",
    "capture": "a procedure the harness captured from a completed attempt, with its record",
    "failure": "an attempt that failed, with what broke",
}

_RUN_TESTS = (
    'Run the tests as the gate does, from /memory: `subprocess.run([sys.executable, "-m", "pytest", "memory", "-q", '
    '"-p", "no:cacheprovider", "--rootdir", "/memory"], cwd="/memory", env=<<pytest_env>>)` (without `--rootdir`, '
    "the empty config file makes /dev the root and the library's imports fail)."
)

_TRUST = (
    "in this order of trust: the environment's own response; a value used in an episode whose record shows it "
    "worked; agreement across episodes. A value you make up is not trusted, and is allowed only in refusal and "
    "shape tests."
)

_DETERMINISM = "Tests give the same result on every run: never assert on the clock, a random draw or a hash order."

_HONESTY = (
    "Never claim a read, a test run or a result that did not happen, and never shape a function or a test to pass "
    "a known check by special-casing its values."
)

#: The example-evidence rule exactly as the gate checks it (``example_evidence.evidence_check``, whose
#: ``MIN_EXAMPLE_EPISODES`` a test keeps equal to this): the brief states it in these words, and the example refusals end
#: with the fix that satisfies them. Values from more episodes are recorded and shown beside the function, never
#: required.
EXAMPLE_EPISODES = 1
EPISODES_FIX = (
    "add a test call that returns on values loaded from a fixture of a recorded episode; a refused or raised call "
    "counts for nothing"
)
#: ``[examples:grounded]``: what satisfies it, with the loading idiom of the library's own tests
#: (``DATA = Path(__file__).parent / "data"``), so the refusal names a concrete change and not only the rule.
GROUNDED_FIX = (
    "make one call that returns take each structured argument (a list, a mapping or a path; plain numbers and "
    "strings may stay literal) from a fixture made with `fixture`, or compute it in the test from one. The "
    "fixture is a recorded episode's value saved under the package's `tests/data/`, and the test calls the "
    'function on what it loads from it, such as `rows = json.loads((Path(__file__).parent / "data" / '
    '"rows.json").read_text())` then `assert function(rows, 2) == <trusted value>`. A call on values written '
    "in the test itself does not count, however close to the recorded ones. When the recorded value is text, "
    "store a function that reads such text into a structured value and make the fixture with "
    '`fixture(..., parser="<that function\'s id>")`; the gate runs the function on the recorded bytes again to '
    "check the fixture"
)
#: ``[examples:unjudged]``: why the check with unseen fixture values did not complete -> what lets it complete.
UNJUDGED_FIX = {
    "budget": (
        "the check runs the listed tests that open an input fixture of every function in this gate run within "
        "{budget_s:.0f} s in all; make those tests quicker"
    ),
    "failed": "its listed tests that open an input fixture must pass, within their time limit, when run on their own",
    "record": (
        "its tests must end normally (no exit, no killed process) and make at most {max_calls} recorded calls of "
        "library functions"
    ),
    "unmatched": (
        "each call of it in its tests must still be reached when the fixture values change: what a test computes "
        "from a fixture before the call (parsing, slicing, checks) must work on any values of the same shape"
    ),
}

_G5_WRITE = (
    "a store pass never ends with fewer items, and every fixture a function's calls ran on is still used by a "
    "function's calls"
)
#: The "when to use" section every stored function and note carries and the README every folder holds, as the writer
#: and the curator are told them (:mod:`.when_to_use`: the gate checks the shape, never the words).
WHEN_TO_USE_TEXT = (
    "- Every function and note carries a section saying when to use it, written for a reader who has a new "
    "request and must decide whether this item fits it. In a function's docstring it is a `When to use:` section; "
    "in a note it is the body's `## When to use` heading. Either holds four labelled lines, each with its text: "
    "`Saved for:` what the request it was saved from asked for, summarised; `Recognise:` how to tell that a new request is that situation; `Not for:` the "
    "requests that look alike but need something else; `Example request:` one request it fits, in a requester's "
    "words. For example:\n"
    "  When to use:\n"
    "      Saved for: totalling the amounts of a list of payment rows per currency.\n"
    "      Recognise: the request gives rows with an amount and a currency and asks for totals.\n"
    "      Not for: rows that must first be converted to one currency, or totals per date.\n"
    "      Example request: Sum these payments per currency.\n"
    "  The gate refuses a new or changed function or note without the section or one of its four lines. Keep it "
    "true when you change what the item does.\n"
    "- Every folder, `memory/<package>/` and `notes/<topic>/`, holds a short `README.md` you keep: what the "
    "folder holds and how its items relate. The gate refuses the items of a folder the pass changes when the "
    "folder has no README.md, or an empty one.\n"
)


def _fill(template: str, values: Mapping[str, object]) -> str:
    out = template
    for key, value in values.items():
        out = out.replace(f"<<{key}>>", str(value))
    left = sorted(set(re.findall(r"<<[a-z_]+>>", out)))
    if left:
        raise ValueError(f"unfilled placeholders: {left}")
    return out


def _pct(share: Decimal) -> str:
    """A share as a whole-or-decimal percentage string: Decimal('0.6') -> '60'."""
    return format((Decimal(share) * 100).normalize(), "f")


def _env(env: Mapping[str, str]) -> str:
    return json.dumps({**dict(env), "PYTHONDONTWRITEBYTECODE": "1"})


def _num(value: float) -> str:
    """A number as the brief states it: ``60.0`` -> ``60``."""
    return format(Decimal(str(value)).normalize(), "f")


@dataclass(frozen=True)
class Limits:
    """The operational limits a brief states once, as information (each from the constant that enforces it)."""

    view_bytes: int
    cell_s: float
    file_bytes: int
    total_bytes: int
    entries: int
    step_guard: int
    check_floor_s: float
    cells_per_turn: int | None = None
    readers_per_turn: int | None = None
    #: finish's refusals of a manifest that does not parse before the final check judges it instead (None: no cap)
    finish_refusals: int | None = None
    #: repair rounds after the first refused `finish` (None: no set number; only the other limits end them)
    repair_rounds: int | None = None
    #: with no set number of rounds: the consecutive rounds after which a refusal repeated for the same item and
    #: reason gets no further round (repair.STUCK_REPEATS; None: not stated)
    stuck_repeats: int | None = None
    #: compaction (the agent loop's, writer_pass.compact_by_summary): the share of the model's context window, in
    #: percent, a call's prompt reaches before the conversation is summarised (None: compaction off), and whether
    #: the request messages are kept before the summary (UNIFY_COMPACTION_KEEP_PREFIX=on)
    compact_threshold_percent: str | None = None
    #: and the fixed prompt size at which it compacts first (context_compression.compaction_at_tokens)
    compact_at_tokens: int | None = None
    compact_keep_prefix: bool = False
    #: what the episode-end fork (and each helper call) can stage: files and bytes
    stage_files: int | None = None
    stage_bytes: int | None = None


def _tool_names(agent: str, flags: frozenset[str]) -> frozenset[str]:
    """The tools *agent* calls in code with the configuration *flags* on."""
    from unify.common.tool_registry import allow_list

    return frozenset(allow_list(agent, flags))


def _join(parts: list[str]) -> str:
    if len(parts) <= 1:
        return "".join(parts)
    return ", ".join(parts[:-1]) + " and " + parts[-1]


def _pool_kinds() -> str:
    from .pool import KINDS

    return "; ".join(
        f"`{k}` ({POOL_KIND_TEXT[k]})" if k in POOL_KIND_TEXT else f"`{k}`"
        for k in KINDS
    )


def row_keys(staging: bool = False, pool: bool = False) -> tuple[tuple[str, str], ...]:
    """The batch-map row keys a pass with these inputs writes, in the order the brief lists them."""
    return (
        BATCH_ROW_KEYS
        + ((STAGING_ROW_KEY, CAPTURE_ROW_KEY) if staging else ())
        + ((POOL_ROW_KEY, _related_row_key()) if pool else ())
    )


def _related_row_key() -> tuple[str, str]:
    from .pool import RELATED_PAGE

    return RELATED_ROW_KEY[0], RELATED_ROW_KEY[1].replace("{page}", str(RELATED_PAGE))


def write_brief(
    *,
    limits: Limits,
    mutation_min: Decimal,
    sample_k: int,
    lint_min: int,
    pytest_env: Mapping[str, str],
    tool_names: frozenset[str] = frozenset(),
    flags: frozenset[str] = frozenset(),
) -> str:
    """The WRITE brief, after the shared opening for the configuration *flags* (the flags the writer's allow-list is
    built with). *staging* and *pool* add their input bullets, row keys and manifest fields; *tool_names* are the tools
    the role gets, so a sentence that relies on an optional tool (``similar``, ``delegate``) appears only when that
    tool exists. Every input is constant for a run, so the brief is byte-stable for the run.
    """
    return _opened(
        "writer",
        flags,
        limits,
        _write_brief_in_code(
            limits=limits,
            mutation_min=mutation_min,
            sample_k=sample_k,
            lint_min=lint_min,
            pytest_env=pytest_env,
            tool_names=tool_names,
        ),
    )


def _opened(agent: str, flags: frozenset[str], limits: Limits, brief: str) -> str:
    """*brief* after the shared opening every agent reads first when tools are called from code
    (:func:`.agent_profiles.in_code_opening`); both are constant for a run, so the system message is a stable
    prefix."""
    from .agent_profiles import in_code_opening

    return in_code_opening(agent, frozenset(flags), cell_s=limits.cell_s) + "\n\n" + brief


def limits_now(
    cell_s: float | None = None,
    compact: bool = False,
    finish_capped: bool = False,
) -> Limits:
    """The limits from the constants that enforce them (imported here: they import this module's callers).
    *compact*: the pass compacts its conversation; *finish_capped*: finish's manifest refusals are capped.
    """
    from . import repair as _repair
    from . import views, writer_pass
    from .integration import fork_worker
    from unify.common._async_tool import context_compression as _cc

    return Limits(
        view_bytes=views.VIEW_BYTES,
        cell_s=writer_pass.PassConfig.cell_timeout_s if cell_s is None else cell_s,
        file_bytes=writer_pass.QUOTA_FILE_BYTES,
        total_bytes=writer_pass.QUOTA_TOTAL_BYTES,
        entries=writer_pass.QUOTA_ENTRIES,
        step_guard=writer_pass.STEP_GUARD,
        check_floor_s=writer_pass.CHECK_FLOOR_S,
        cells_per_turn=getattr(writer_pass, "_MAX_CELLS_PER_TURN", None),
        readers_per_turn=getattr(writer_pass, "READERS_PER_TURN", None),
        finish_refusals=writer_pass.FINISH_G1_REFUSALS if finish_capped else None,
        # a capped pass also caps its repair rounds (repair.may_repair); repair_message states the same count
        repair_rounds=_repair.REPAIR_ROUNDS if finish_capped else None,
        # an uncapped pass stops offering rounds for a refusal repeated this many rounds in a row
        stuck_repeats=None if finish_capped else _repair.STUCK_REPEATS,
        compact_threshold_percent=(
            _num(_cc.COMPACTION_THRESHOLD * 100) if compact else None
        ),
        compact_at_tokens=_cc.compaction_at_tokens() if compact else None,
        compact_keep_prefix=bool(compact and False),
        stage_files=fork_worker.QUOTA_FILES,
        stage_bytes=fork_worker.QUOTA_BYTES,
    )


def tool_flags(helper: bool = False, similar: bool = False) -> frozenset[str]:
    """The tool spec configuration flags of a pass: ``helper``, a helper lane exists, so ``delegate`` is offered;
    ``similar``, the similar-code index exists."""
    return frozenset(({"helper"} if helper else set()) | ({"similar"} if similar else set()))


def write_brief_now(
    cell_s: float | None = None,
    helper: bool = False,
    similar: bool = False,
    finish_capped: bool = False,
) -> str:
    """The WRITE brief from the constants that enforce each number and the tools registered for the role (*helper*:
    a helper lane exists; *similar*: the pass has a similar-code index; *finish_capped*: the pass has a USD cap)."""
    from . import code_lint, qa
    from .gate import _PYTEST_ENV

    flags = tool_flags(helper, similar)
    return write_brief(
        limits=limits_now(cell_s, compact=True, finish_capped=finish_capped),
        mutation_min=qa.MUTATION_MIN,
        sample_k=qa.SAMPLE_K,
        lint_min=code_lint.LINT_MIN,
        pytest_env=_PYTEST_ENV,
        tool_names=_tool_names("writer", flags),
        flags=flags,
    )


# --- the briefs when tools are called from code (UNIFY_TOOLS_IN_CODE=on) ---------------------------------------------
#
# Every agent then has one model-API tool, `execute_code`, and calls every other tool in its cells as
# `tools.<name>(...)`; `tools.list()` and `tools.help(name)` describe them, so a brief names a tool where the role uses
# it and leaves its details to the help. Every rule a gate check enforces is stated once, where the agent acts on it.
# Tables that only describe input files (the batch-map row keys, the `field_types` form, the pool's entry kinds) are
# written next to the inputs (:func:`input_pointer_files`) and the brief points to them.

#: The drafting contract when tools are called from code: the reason the writer generalises drafts is its own brief's.
CANDIDATE_CONTRACT_IN_CODE = (
    "Draft from what you saw: code that worked, the cases it ran on (values as they were recorded), and notes. "
    "Where you can, write a reusable function or lesson in general form, with the values that vary as "
    "parameters. Where your work is specific to this case, still write it down as it is: the library writer builds "
    "from these drafts and may generalise them with other episodes."
)


#: The example-evidence rule when tools are called from code: the same rule, with the wider evidence said once.
EPISODES_RULE_IN_CODE = (
    "the calls of a function that return take fixture values from a recorded episode: one episode is enough. Calls "
    "on same-shaped values from other episodes, in this batch or not, are shown beside the function, never "
    "required. A call that refuses or raises counts for nothing"
)

_REPAIR_IN_CODE = (
    "# Repair\n"
    "If the gate refuses anything, its full result (every check, every failing test's output and every note) is "
    "written to /inputs/gate/result-<round>.md and you get a repair round: fix what it refused, keep the manifest "
    "current, and call `tools.finish(summary)` again. The limits below say how many repair rounds a pass has. "
    "The items that pass land, and the rest is kept as a draft for a later pass, with its failure.\n"
)


def _check_names_in_code(g5: str, g3_mutations: str) -> str:
    """What each check the gate names means, G1 to G6, stated once; *g3_mutations* is how many small mutations the
    tests must catch, as the role's brief states it."""
    return (
        "Its results and refusals name the check: G1, layout and provenance (only `memory/` and `notes/` change and "
        "never the generated files, the manifest is well formed, every link names an item that exists, every "
        "fixture came through `tools.fixture` unchanged, a parsed one re-made by its reader, every new or changed "
        "function and note has its `When to use` section, and every folder the pass changes has its README.md); "
        "G2, evidence (each "
        "function's listed tests call it, and the calls they make of it return or raise `MemoryInputError`; what "
        "else those calls show sets its evidence level: `unchecked` when a call that returns uses a value from a recorded "
        "episode in any form but some evidence check does not hold, `reproduces-its-case` when its structured "
        "arguments come from fixtures or through a stored reader, with those values replaced by unseen values of "
        "the same type a call it accepted neither refuses nor raises, its docstring's `Example:` names one of its "
        "tests, its typed covers are recorded evidence of their type, and the literal rule holds); G3, tests (plain "
        "pytest, the tests pass on the candidate, a behaviour change has a test that fails on the parent, and no "
        "test is lost or weakened without a reason; an exact assertion against a trusted value on a recorded "
        "fixture for each changed function, no test that recomputes the function, and passing on the drawn "
        "recorded inputs and catching "
        + g3_mutations
        + " also set the evidence level); G4, the index (it never refuses: it notes when the index has outgrown the view the working model "
        "is shown); G5, items (" + g5 + "); G6, safety (no links, executables, submodules, git, pytest or "
        "interpreter configuration files, bytecode or key-shaped strings).\n"
    )


_GATE_IN_CODE = (
    "# The gate\n"
    "After `tools.finish(summary)` a deterministic gate checks the candidate. An item that breaks a rule is refused "
    "together with the items that import it, and the rest can land; a breach of a pass-wide rule (the layout, lost "
    "tests, fewer items) refuses the round. "
    + _check_names_in_code(
        _G5_WRITE, "at least <<mutation_min_pct>>% of small mutations of each changed function"
    )
)

_LIMITS_IN_CODE = (
    "# Limits\n"
    "These are safety limits, stated as facts; none is a target. A reply that reads a file or a record shows at "
    "most <<view_bytes>> bytes and ends with a marker naming the next offset; nothing is cut silently. A cell runs "
    "for at most <<cell_s>> s. Files under /memory are limited to <<file_mib>> MiB each, <<total_mib>> MiB in all "
    "and <<entries>> files and directories; a breach refuses the pass. The pass ends at its wall clock (its length "
    "in seconds is in the first message<<helper_time>>) or after <<step_guard>> model and check calls, whichever "
    "comes first, and a `tools.check` starts only while at least <<check_floor>> s of the wall clock remain."
    "<<repair_cap>><<finish_cap>>\n"
)


def _limits_text_in_code(limits: Limits, helper_time: bool) -> str:
    """The limits when tools are called from code. The per-turn counts are stated by the 'not run' answers when they
    apply, and the compaction by the note that follows a restart (``writer_pass.RESTART_POINTER``)."""
    return _fill(
        _LIMITS_IN_CODE,
        {
            "view_bytes": limits.view_bytes,
            "cell_s": _num(limits.cell_s),
            "file_mib": _num(limits.file_bytes / 1024**2),
            "total_mib": _num(limits.total_bytes / 1024**2),
            "entries": limits.entries,
            "helper_time": (
                "; the helpers you start work within the same wall clock and stop when it is reached"
                if helper_time
                else ""
            ),
            "step_guard": limits.step_guard,
            "check_floor": _num(limits.check_floor_s),
            "repair_cap": (
                f" At most {limits.repair_rounds} repair rounds follow the first refused `tools.finish` in this "
                "pass."
                if limits.repair_rounds is not None
                else " There is no set number of repair rounds; only these limits end them."
            ),
            "finish_cap": (
                " While /memory/.pass/manifest.json is missing or does not parse, `tools.finish` is refused with the "
                f"reason up to {limits.finish_refusals} times in a pass; after that, `tools.finish` ends the pass and "
                "the final check refuses the manifest by name."
                if limits.finish_refusals is not None
                else ""
            ),
        },
    )


#: The writer's library bullet with tools in code: which library calls to use for what, each named without its
#: arguments (their help holds those).
_LIBRARY_BULLET_IN_CODE = (
    "- /memory: a writable copy of the library. There is no git history in the box; `memory.get` opens an item "
    "with its recent changes. /memory/INDEX.md is the library index, generated by the harness like "
    "/memory/links.json and /memory/memory/__init__.py. In every cell `memory` is already bound (no import "
    "needed): browse it with `memory.items` and open an item with `memory.get` before you add or change one. "
    "Names you bound earlier keep the code they were bound to; after you edit a library file, import it again. "
    "For: knowing what exists before you add anything.\n"
)

_WHAT_TO_WRITE_IN_CODE = (
    "# What to write\n"
    "- Functions go in `memory/<package>/<module>.py`, in packages you choose by what the code does. Each "
    "package has an `__init__.py` whose docstring is one paragraph on what lives there. A function's id is "
    "`memory.<package>.<module>:<name>`; a name starting with `_` is private and is not an item.\n"
    "- A procedure is a function that takes the value a stored reader gives on a request and returns the result "
    "that was handed back.\n"
    "- A procedure comes with its recipe: one function in its module, named apply_ followed by the procedure's "
    "name (apply_total for total), that is handed a new request as the harness hands it (its text, with .text and "
    ".data), reads from it every value the job needs, calls the procedure and the helpers, and returns what the job "
    "hands back. A value copied from the request you learned from is wrong on the next request; derive it. The "
    "procedure's manifest entry adds \"role\": \"procedure\", \"recipe\" (the recipe's id; the recipe is an item "
    "too) and \"cases\": its case files, each an input with its known output, made with `tools.fixture(...)` "
    "given expected: the source request with the value its episode handed back (source \"request:0\", expected "
    "\"answer\"), and every input the request or the environment showed with the output it showed for it, a later "
    "observation included. The gate runs the recipe, confined, on every case and lands the procedure only if it "
    "gives every expected output; a failing case comes back with the difference. An episode whose recipe does not "
    "pass, or that failed, lands no procedure (a note of what was tried may still land). "
    "`tools.land_capture(...)` writes the recipe and the source case; add the shown cases.\n"
    "- A function's docstring has a one-line summary; `Args:`, `Returns:` and `Raises:` sections "
    "(`MemoryInputError` on an input it was not built for); an `Example:` section of one line naming one of "
    "the function's tests, `memory/<package>/tests/test_<module>.py::test_<case>`, with no doctest and no "
    "`Input:` line; a `When to use:` section (below); and, optionally, a `Notes:` line naming note paths.\n"
    "- Notes go in `notes/<topic>/<slug>.md`: free-form markdown (a fact, a pitfall, a how-to, or how to combine "
    "functions) with front matter `title`, `description` (one line, shown in the index) and `uses` (the ids of the "
    "functions it uses). A link is written only in a note's `uses` or a function's `Notes:` line, and must name an "
    "item that exists. A new item starts as `on-trial`; later passes revise or remove it. A function's linked "
    "note says its preconditions and its known failures.\n"
    + WHEN_TO_USE_TEXT
    + "- Tests go in `memory/<package>/tests/test_<module>.py`: plain pytest, importing only the library and the "
    "standard library, with fixtures under `memory/<package>/tests/data/`. `tests/` takes no `__init__.py`: "
    "pytest imports each test file by its path. " + _RUN_TESTS + "\n"
    "- The manifest is /memory/.pass/manifest.json (never committed):\n"
    '  {"items": [{"item": "<id>", "kind": "function" or "note", "source_episodes": [...], "tests": [...], '
    '"covers": [...]<<item_extra>>}], "support": [], <<manifest_extra>>"why": "<one line>", "summary": '
    '"<what changed>"} (optional: "tests_changed", "renames")\n'
    "  - A note's entry takes `item`, `kind` and `source_episodes`: "
    '`{"item": "notes/<topic>/<slug>.md", "kind": "note", "source_episodes": [...]}`; `tests`, `covers` and '
    "`field_types` are for functions.\n"
    "  - A function takes any arguments; its examples are the calls its listed tests make, so its `tests` name at "
    "least one test file.\n"
    "  - A field with a fixed domain is declared in `field_types` (form: /inputs/README.md).\n"
    '  - `covers` takes typed covers only, e.g. `{"episode": e, "type": "cell", "cell": i}` (also `"diff"`, or '
    '`"episode"` with a `runner` for a whole procedure). A recorded action is not a cover.\n'
    "  - `support` lists helper files: a test helper under a package's `tests/` directory, or a shared module "
    "`memory/<package>/<module>.py` that the package's items import, holding private (`_`) functions and classes "
    "such as the package's `MemoryInputError` (a public function is an item).\n"
    "<<manifest_extra_docs>>"
    '  - `renames` (`{"<old id>": "<new id>"}`, optional) keeps a moved item\'s history.\n'
    "  - `why` is one line on why the change helps. The harness writes it, the episodes and the item ids into "
    "the commit.\n"
)

_WRITE_RULES_IN_CODE = (
    "# Rules\n"
    "1. Store anything that repeats or is likely to repeat, in any domain: readers, calculations, call "
    "sequences, and whole-job procedures built from smaller tested functions. A procedure is verified by "
    "replaying its episode: a work-tree procedure must reproduce the recorded file changes, a tool procedure the "
    "recorded calls, and a dialogue procedure the recorded reply the environment then accepted. A procedure "
    "of any other kind cannot be verified, so it stays a draft.\n"
    "2. Start from the working model's own code when it already wrote it (its cells and its work-tree diff). "
    "Generalise it: values that vary become parameters, and the function raises "
    '`MemoryInputError("<what was expected>")` on an input it was not built for. Check and refuse shapes, not '
    "observed values. The gate replaces the fixture values the tests load with unseen values of the same type and "
    "reruns the tests: the function must not then raise `MemoryInputError` or another exception on a call it "
    "accepted before. A field with a fixed domain is declared in `field_types`.\n"
    "3. Judge outcomes from the records. The harness never reads environment text for a verdict: "
    "`structural_events` lists only cell errors, action errors and retries, and an empty list says nothing about "
    "success. Whether the work succeeded is in the environment's own replies, if anywhere; `end_parts` names "
    "where each episode ended. Prefer work from episodes whose records show it working. Work that only failed "
    'becomes a note ("X was tried and failed because Y; Z worked"), not a function, unless its tests check '
    "values from episodes where it worked. When no record shows an outcome, rely on errors and on the gate's "
    "checks.\n"
    "4. Prefer extending an existing item to adding a near-duplicate. When you change what an item returns on "
    "its recorded inputs, add a test that fails on the old version and passes on yours, and say why in `why`. "
    "This pass never removes library items and never deletes or weakens a test: the gate refuses a candidate "
    "with fewer items or tests. If an existing test must change, name it in `tests_changed` with the reason. "
    "A pass may also change only notes, only tests, or the expected values in the tests of a function you did not "
    "change; such tests need only pass.\n"
    "5. Write the tests first, against trusted expected values, "
    + _TRUST
    + " At least one call returns on arguments loaded from fixtures made with `tools.fixture`, or computed in the "
    "test from one; the gate checks the argument changes when the fixture does. An item whose evidence does not "
    "reach that is still stored, at a lower evidence level, and later evidence can raise it. Across its examples, "
    + EPISODES_RULE_IN_CODE
    + ". Each function needs at least one test asserting "
    "that its result on a recorded fixture equals an exact trusted value, and one test that it refuses an input of "
    "another shape. A procedure's test checks the "
    "effect the environment later confirmed, not only that it ran. Never compute an expected value with the "
    "function's own expression. "
    + _DETERMINISM
    + " For a function you add or change, run each new test and see it fail before your change, then pass.\n"
    "6. Each function has a recorded-inputs file, "
    "`memory/<package>/tests/data/<module>.<function>.inputs.jsonl`, with one line per recorded input made with "
    "`tools.fixture(..., append=True)`, and a test parametrised over its lines. The gate appends <<sample_k>> more "
    "recorded inputs of the same kind to a copy of that file, and the function's tests must pass on every one.\n"
    "7. Code never holds a literal equal to a value that varies across its recorded inputs (text of "
    "<<lint_min>> or more characters, or a number with 2 or more significant digits): make it a parameter.\n"
    "8. Never store credentials, session tokens or the results of calls with side effects. "
    + _HONESTY
    + "\n"
)


#: The attention pass's section of the brief.
ATTENTION_TEXT = (
    "# Your entries\n"
    "You are given entries that need your attention; decide each with `tools.decide`. Open entries listed beside "
    "them may share a rule with yours; look with `tools.open_entries` and `tools.find`, and `tools.take` those that "
    "do. /inputs/attention/index.json lists your entries, each with the material it owes a read (its summary, its "
    "procedure, its use, its signal, its note or its message); the harness checks what you opened, never what you "
    "say you read. One decision can close several entries: that is how you record that they are one rule. A "
    "decision is one of new_item, changed_item, merged_into (naming the item), note, retire, nothing_to_keep and "
    "defer (the work stays open as a new entry). An entry you do not decide comes back to a later pass.\n"
)
#: What an item is, under ``UNIFY_MEMORY_IMPURE_ITEMS=on`` (the brief's What to write).
IMPURE_TEXT = (
    "- An item is written for a kind of situation, with what varies as its inputs. It may take the current state "
    "and act through the environment's tools. When you land or change one, say which episodes it is meant to cover "
    "and which of its values are inputs and which are constants, with the source of each constant. Notes and "
    "checklists are items too.\n"
)
#: The manifest fields of an item's stated provenance (``UNIFY_MEMORY_IMPURE_ITEMS=on``).
IMPURE_MANIFEST_DOCS = (
    "  - Every item may state, and every item you land or change states: `covers_episodes`, the episodes it is meant "
    "to cover; `inputs`, what varies (for example `state.data[0].lines`); `constants`, each value it holds fixed "
    "with its source (`{\"value\": \"tax rate 0.2\", \"source\": \"<episode>/action:3\"}`); and for a function "
    "`effects`, `none`, `reads`, `writes` or `unknown` (what its calls do to the environment). A note that is a "
    "checklist says `\"form\": \"checklist\"`. They are kept on the item's label beside its evidence.\n"
)
#: The stated tests (``UNIFY_MEMORY_WRITER_TESTS``).
TESTS_TEXT = "Running code is a test you state first: say what you expect, then run `tools.test`."

#: Rev 2's gate (``UNIFY_MEMORY_GATE_LABELS_ONLY``, :mod:`.gate_policy`): what a finding does and what the gate runs.
LABELS_ONLY_GATE = (
    "# The gate\n"
    "After `tools.finish(summary)` a deterministic gate checks the candidate. A finding about an item is recorded on "
    "its label and the item lands with it; a finding about the whole pass is a note. Only these refuse: safety "
    "(G6), a change outside `memory/` and `notes/` or to the generated files, a manifest that does not parse or "
    "leaves a change undeclared, a fixture that did not come through `tools.fixture` unchanged, a cited episode that "
    "is not recorded, and code that does not parse or import. The gate runs code only for an item that is a pure "
    "function of its input (it reads its input and calls no tools, no files and no network): its listed tests, its "
    "recipe on its cases and its own recorded calls. For any other item its label says what was not run and why. "
    "Its labels name the check: G1, layout and provenance (the paths, the manifest, the links, the fixtures, each "
    "new or changed function and note's `When to use` section and each changed folder's README.md); G2, evidence "
    "(each function's listed tests call it, and what those calls show sets its evidence level); G3, tests (plain "
    "pytest, the tests pass, a behaviour change has a test that fails on the parent, no test is lost or weakened "
    "without a reason, and an exact assertion against a trusted value on a recorded fixture for each changed "
    "function, never one that recomputes it); G4, the index (a note when it has outgrown the view the working "
    "model is shown); G5, items (no fewer items, and a deleted item's recorded jobs kept by a remaining one); G6, "
    "safety (no links, executables, submodules, git, pytest or interpreter configuration files, bytecode or "
    "key-shaped strings).\n"
)
#: Rev 2's repair: only a hard check gives a repair round.
LABELS_ONLY_REPAIR = (
    "# Repair\n"
    "If a check refuses anything, its full result (every check, every failing test's output and every note) is "
    "written to /inputs/gate/result-<round>.md and you get a repair round: fix what it refused, keep the manifest "
    "current, and call `tools.finish(summary)` again. The limits below say how many repair rounds a pass has. "
    "Everything else lands, each item with its label; the labels are in the same result, for this pass or a later "
    "one to improve.\n"
)


def _swap(text: str, old: str, new: str) -> str:
    """*text* with *old*, which it holds exactly once, replaced by *new* (a brief whose wording moved fails loudly)."""
    if text.count(old) != 1:
        raise ValueError(f"the WRITE brief holds {text.count(old)} copies of {old[:60]!r}, not one")
    return text.replace(old, new)


def with_labels_only(body: str, recipe_rule: str) -> str:
    """The in-code WRITE brief under rev 2's gate (:data:`LABELS_ONLY_GATE`): every sentence that says the gate refuses
    what it now labels, or runs what it now leaves alone, says what it does instead."""
    body = _swap(
        body,
        "The gate runs the recipe, confined, on every case and records what each case gave on the procedure's "
        "label; the procedure lands either way. ",
        "The gate runs the recipe on every case when the procedure is a pure function of its input, and records "
        "what each case gave on its label; the procedure lands either way. ",
    )
    body = _swap(
        body,
        "Each quality finding about an item (its recipe's cases, the unseen values, its examples, its tests and "
        "its behaviour) is recorded on its label and lands with it; safety, paths, syntax and the library's other "
        "tests still refuse.\n\n",
        "",
    )
    body = _swap(
        body,
        "A procedure is verified by replaying its episode: a work-tree procedure must reproduce the recorded file "
        "changes, a tool procedure the recorded calls, and a dialogue procedure the recorded reply the environment "
        "then accepted. ",
        "A procedure is judged by its own tests and its recipe's cases. ",
    )
    body = _swap(
        body,
        "The gate replaces the fixture values the tests load with unseen values of the same type and reruns the "
        "tests: the function must not then raise `MemoryInputError` or another exception on a call it accepted "
        "before. ",
        "",
    )
    body = _swap(
        body,
        "the gate refuses a candidate with fewer items or tests.",
        "a candidate with fewer items or tests is labelled so.",
    )
    body = _swap(
        body,
        " The gate appends <<sample_k>> more recorded inputs of the same kind to a copy of that file, and the "
        "function's tests must pass on every one.",
        "",
    )
    body = _swap(
        body,
        "  The gate refuses a new or changed function or note without the section or one of its four lines.",
        "  A new or changed function or note without the section or one of its four lines is labelled so.",
    )
    body = _swap(
        body,
        "The gate refuses the items of a folder the pass changes when the folder has no README.md, or an empty one.",
        "The items of a folder the pass changes are labelled when the folder has no README.md, or an empty one.",
    )
    body = _swap(body, _GATE_IN_CODE, LABELS_ONLY_GATE)
    body = _swap(body, _REPAIR_IN_CODE, LABELS_ONLY_REPAIR)
    body = _swap(
        body,
        "has passed the gate or is kept as a draft with its failure.",
        "has landed with its label, or is kept as a draft with the check that refused it.",
    )
    return body


def with_switches(body: str) -> str:
    """The WRITE brief *body* for an attention pass with items of any effect and stated tests: the attention pass's
    section after the role; items of any effect (the recipe's cases and every other quality finding are labels; rule
    7's literal rule becomes the stated inputs and constants; rule 8 keeps credentials and tokens, and no longer bars
    the results of calls with side effects; a procedure of another kind is labelled, never left a draft); rev 2's gate
    (:func:`with_labels_only`); the stated tests."""
    body = _swap(body, "\n# Inputs\n", "\n" + ATTENTION_TEXT + "\n# Inputs\n")
    body = _swap(
        body,
        "- /inputs/batch_map.json:",
        "- /inputs/attention/: your entries (index.json) and, per entry, what the harness holds for it outside the "
        "episodes (use.json, signal.json, note.json). For: what each entry owes a read.\n- /inputs/batch_map.json:",
    )
    recipe_rule = (
        "The gate runs the recipe, confined, on every case and lands the procedure only if it gives every expected "
        "output; a failing case comes back with the difference. An episode whose recipe does not pass, or that "
        "failed, lands no procedure (a note of what was tried may still land). "
    )
    body = _swap(
        body,
        recipe_rule,
        "The gate runs the recipe, confined, on every case and records what each case gave on the procedure's "
        "label; the procedure lands either way. ",
    )
    body = _swap(body, "- A function's docstring has a one-line summary;", IMPURE_TEXT + "- A function's docstring has a one-line summary;")
    body = _swap(
        body,
        "A procedure of any other kind cannot be verified, so it stays a draft.",
        "A procedure of any other kind, or one that acts through the environment's tools, is kept with the "
        "evidence it has (its tests replay the recorded calls, or it was judged by reading), stated on its label.",
    )
    body = _swap(
        body,
        "7. Code never holds a literal equal to a value that varies across its recorded inputs (text of "
        "<<lint_min>> or more characters, or a number with 2 or more significant digits): make it a parameter.\n",
        "7. Say which of an item's values are inputs and which are constants, each constant with its source "
        "(`inputs`, `constants`): a value copied from one episode is a constant only where its source says it "
        "holds for the kind of situation the item is for.\n",
    )
    body = _swap(
        body,
        "8. Never store credentials, session tokens or the results of calls with side effects. ",
        "8. Never store credentials or session tokens; your own code holds no key, opens no network and writes "
        "nothing outside /memory. ",
    )
    body = _swap(
        body,
        "# Repair\n",
        "Each quality finding about an item (its recipe's cases, the unseen values, its examples, its tests and "
        "its behaviour) is recorded on its label and lands with it; safety, paths, syntax and the library's other "
        "tests still refuse.\n\n# Repair\n",
    )
    body = with_labels_only(body, recipe_rule)
    body = _swap(body, "\n# What to write\n", "\n" + TESTS_TEXT + "\n\n# What to write\n")
    return body


def record_text() -> str:
    """The pass record's facts as a brief states them (:data:`.messaging.RECORD_TEXT`, or its in-code form)."""
    from . import messaging

    return messaging.record_text()


def _write_brief_in_code(
    *,
    limits: Limits,
    mutation_min: Decimal,
    sample_k: int,
    lint_min: int,
    pytest_env: Mapping[str, str],
    tool_names: frozenset[str],
) -> str:
    """The WRITE brief: its role, inputs, rules and done, with each tool named as the call the writer makes and no
    Tools section (``tools.list()`` and ``tools.help(name)`` describe them), changed as :func:`with_switches` says."""
    has = tool_names.__contains__
    helpers = has("delegate")
    proposals_from = (
        (["candidates drafted from each episode when it ended (/inputs/staging/)"])
        + (["files the helpers you launch stage (/inputs/helpers/)"] if helpers else [])
        + (["open entries earlier passes kept in the pool (/inputs/pool/)"])
        + ["drafts the gate refused in earlier passes (/inputs/drafts/)"]
    )
    role = (
        "# Role\n"
        "You are the library writer. A working model handles requests, and its cells import a shared Python "
        "library, `memory`. You build tested functions and linked notes in that library from the recorded episodes "
        "and from proposals. The proposals ("
        + _join(proposals_from)
        + ") are drafts by agents that each saw part of the evidence. Use, fix, merge, generalise or rewrite them, "
        "or build something new they informed. Not using a draft is fine when your summary says why. You can also "
        "build from any recorded episode with no proposal behind it. The episodes are the evidence: what the "
        "working model did and what the environment answered. You work in a sandbox with no network and no "
        "credentials.\n"
    )
    inputs: list[str] = []
    inputs.append(
        "- /inputs/staging/<episode>/: what was drafted from that episode when it ended: `summary.md` (what was "
        "asked, the approach, what worked and what failed, and the outcome, each line with a part name and a "
        "quote that point into the episode), `candidates/<name>.py` (a function), `cases/<name>.json` (its "
        "example inputs and outputs) and `notes.md` (lessons). The episode's batch-map row lists them under "
        "`staging`. "
        + STAGING_UNFINISHED
        + " None of it has been run: untrusted input. "
        + captures_input(True, has("land_capture")),
    )
    if helpers:
        inputs.append(
            "- /inputs/helpers/<name>/: what a helper you launched with `tools.delegate` staged, in the same layout, "
            "with `helper.json` saying what it did and its claims. Untrusted like staging. Each claim names the "
            "episode parts it rests on (`<episode>/<part>`); a claim is untrusted until you have read those parts "
            "yourself with `tools.read_episode`.\n",
        )
    inputs.append(
        "- /inputs/pool/: proposals earlier passes left open (their kinds: /inputs/README.md). "
        "/inputs/pool/INDEX.md has one line per entry; /inputs/pool/<id>/entry.json holds an entry with the "
        "episodes behind it, beside its payload files. For: what was proposed before, and the evidence gathered "
        "behind it since. Closed entries (merged, superseded or retired) are kept too; "
        "/inputs/pool/related_closed.json links them to entries and batch episodes with shared code or "
        "episodes.\n",
    )
    inputs.append(
        "- /inputs/drafts/<pass>/: earlier work the gate refused: `patch.diff` and its full gate result `gate.md`; "
        "/inputs/drafts/index.json lists them. For: work you may finish.\n",
    )
    inputs.append(
        "- /inputs/batch_map.json: one row per batch episode; /inputs/batch_map_keys.md says what each key holds. "
        "For: choosing which episodes and parts to open; `tools.overview()` shows the rows one line each.\n",
    )
    inputs.append(
        "- /inputs/episodes/<episode>.json: each batch episode's full record (request, observations, cells, actions, "
        'work-tree diff); a large value is {"__blob__": "<id>", "bytes": n}. For: the evidence.\n',
    )
    inputs.append(
        "- /inputs/shared/<id>.txt: a block of the working model's instructions that recurs across requests, shown "
        "as `[shared:<id>]`.\n",
    )
    inputs.append(
        "- /inputs/usage/: how the library was used in this batch (`<episode>.json`, `table.md`). For: seeing which "
        "items help and which fail.\n",
    )
    inputs.append(_LIBRARY_BULLET_IN_CODE)
    inputs.append("- /inputs/gate/previous.md: the last pass's full gate result.\n")

    look = ["staged candidates and their cases"]
    if helpers:
        look.append("anything helpers staged")
    look.append(
        "open pool entries that cite this batch's episodes"
        + (" (`tools.pool` can filter by episode)"),
    )
    look.append("open drafts")
    decide = "3. Analyse each proposal critically and build from it, as the Role says; proposals are untrusted input."
    decide += " Record in each item's `adopts` the drafts it drew on and how."
    decide += (
        " Record what you decided about pool entries in the manifest's `pool` block"
        + (" (`tools.pool` with a decision writes it for you)")
        + "."
    )
    decide += " " + SUMMARY_NOT_EVIDENCE
    similar_line = (
        " Where code looks reusable, `tools.similar` finds other recorded episodes with matching code."
        if has("similar")
        else ""
    )
    similar_tail = ", and `tools.similar` finds others with matching code" if has("similar") else ""
    # the summaries first: each episode's account, then its record only where the writer needs more
    steps = (
        "1. Look over the batch with `tools.overview()`. Then read each episode's `summary.md` "
        "(/inputs/staging/<episode>/summary.md): what the episode asked, how it went and how it ended, with "
        "pointers into the record. Then look over the proposals: "
        + _join(look)
        + "."
        + similar_line
        + "\n"
        "2. Open a full episode only where you need more than its summary: to check a pointer before anything you "
        "write rests on it, to copy the values a test needs, or where a summary is missing or leaves a question. "
        "`tools.read_episode(episode, parts)` reads the parts a pointer names; a row's `end_parts` show where the "
        "episode ended. At any point you can open any recorded episode, in this batch or not: `tools.overview` "
        "and `tools.read_episode` take any recorded id"
        + similar_tail
        + ".\n"
    )
    procedure = (
        "# How to work\n"
        "This is a procedure, not a quota: read as much or as little as the material needs.\n"
        + steps
        + decide
        + "\n"
        "4. Write tests first, against trusted values (rule 5), with fixtures made by `tools.fixture`; see each new "
        "test fail before the code exists, then pass.\n"
        + (
            "5. When it helps, hand a reading or drafting task to a helper with `tools.delegate`. It returns at once; "
            "the helper messages you on the record and ends with its claims and files under /inputs/helpers/<name>/. "
            "You decide when you have enough: `tools.stop` ends a helper, and finishing ends every helper still "
            "running. Using helpers is your choice.\n"
            if helpers
            else ""
        )
        + f"{6 if helpers else 5}. `tools.check()` the manifest as often as you like, then `tools.finish(summary)`. "
        "If the gate refuses something, repair it (below).\n"
    )
    manifest_extra = ""
    manifest_docs = ""
    item_extra = ""
    item_extra = ', "adopts": [...]'
    forms = (
        (["`staging:<episode>/candidates/<name>.py`"])
        + (["`helper:<helper name>/candidates/<name>.py`"] if helpers else [])
        + (["`pool:<id>`"])
    )
    manifest_docs += (
        '  - An item\'s `adopts` lists the drafts it drew on, each as `{"ref": <ref>, "how": <how>}`. The ref is '
        + _join(forms)
        + ". How is `as_is` (used unchanged), `edited`, `merged` (with other drafts or a library item), "
        "`generalised` (parametrised or generalised), `rewritten`, or `informed` (it only informed new work). "
        "Leave it out when the item drew on none. The manifest itself has no `adopts`.\n"
    )
    manifest_extra += '"pool": {...}, '
    manifest_docs += (
        "  - `pool` records decisions on pool entries"
        + (
            " (`tools.pool` with op='decide' writes it; `tools.help(\"pool\")` lists the decisions)"
        )
        + '. `"reopened": {"<id>": "<reason>"}` opens a closed entry again.\n'
    )
    body = (
        role
        + "\n# Inputs\n"
        + "".join(inputs)
        + "\n"
        + record_text()
        + "\n"
        + procedure
        + "\n"
        + _WHAT_TO_WRITE_IN_CODE
        + "\n"
        + _WRITE_RULES_IN_CODE
        + "\n"
        + _GATE_IN_CODE
        + "\n"
        + _REPAIR_IN_CODE
        + "\n"
        + _limits_text_in_code(limits, helper_time=helpers)
        + "\n"
        "# Done\n"
        "You are done when you have built the best library items you can from this batch, with the proposals as "
        "material and the episodes as evidence, and every item "
        "you wrote has passed the gate or is kept as a draft with its failure. Then call `tools.finish(summary)`. "
        "Writing nothing is a valid decision when your summary says why.\n"
    )
    item_extra += ', "effects": "<none|reads|writes|unknown>", "covers_episodes": [...], "inputs": [...], "constants": [...]'
    manifest_docs += IMPURE_MANIFEST_DOCS
    return _fill(
        with_switches(body),
        {
            "manifest_extra": manifest_extra,
            "item_extra": item_extra,
            "manifest_extra_docs": manifest_docs,
            "mutation_min_pct": _pct(mutation_min),
            "sample_k": sample_k,
            "lint_min": lint_min,
            "pytest_env": _env(pytest_env),
        },
    )


#: The files :func:`input_pointer_files` writes next to a store pass's inputs, by name under /inputs.
BATCH_MAP_KEYS_FILE = "batch_map_keys.md"
INPUTS_README_FILE = "README.md"


def input_pointer_files(staging: bool = False, pool: bool = False) -> dict[str, str]:
    """The reference files a store pass's /inputs holds when tools are called from code, by name: what each
    batch-map row key holds (:func:`row_keys`, the same table the brief carries today), and the manifest's
    `field_types` form with the semantic types and, with the pool on, the pool's entry kinds. The brief points to
    both; each is constant for a run."""
    from .manifest import describe_semantic_types

    keys = "".join(f"- `{k}` ({doc})\n" for k, doc in row_keys(staging, pool))
    readme = (
        "# Inputs: reference\n\n"
        "## The manifest's `field_types`\n\n"
        '`field_types` (`{"<fixture path>#<field>": "<type>"}`) declares a field with a fixed domain: the path is a '
        "fixture of the function's package under `memory/<package>/tests/data/`; the field is a JSON or YAML key path "
        "(`a.b`, `items[]`), a CSV column (`#3` for the third column of a file without a header) or, in a JSON-lines "
        "file made with `append=True`, a key path under `[].input`; the type is one of "
        + describe_semantic_types()
        + ". Leave it out when no field has a fixed domain.\n"
        + (
            "\n## Pool entry kinds\n\n" + _pool_kinds() + ".\n"
            if pool
            else ""
        )
    )
    return {
        BATCH_MAP_KEYS_FILE: (
            "# Batch-map row keys\n\n"
            '/inputs/batch_map.json is `{"version": 1, "episodes": [row, ...], "shared_blocks": [{"id", "chars", '
            '"file"}, ...]}`, facts and pointers only. Each row holds:\n\n' + keys
        ),
        INPUTS_README_FILE: readme,
    }


#: Words no writer brief or first message may hold: economy cues and the retired reading-quota machinery.
BANNED_WORDS = (
    "cheaply",
    "budget",
    "coverage",
    "required_parts",
    "experience_tokens",
    "covered or dismissed",
)

#: The most an inlined list may hold: items (lines starting "- ", or table rows) and bytes. A list that grows with
#: the library, the pool or the batch goes to a file the prompt points to. The Tools section, rendered once per run
#: from the tool registry, is stable text and is not such a list.
MAX_INLINE_LIST = 10
MAX_INLINE_LIST_BYTES = 1000


# --- guards -----------------------------------------------------------------

#: Names and terms of particular benchmarks and environments (case-insensitive, whole words).
_BENCHMARK = re.compile(
    r"\b(apis|venmo|arc|grids?|office|payroll|crafter|alfworld|appworld|scienceworld|travelplanner|webshop|"
    r"minecraft|sokoban|submission)\b",
    re.IGNORECASE,
)
#: An environment's verdict text, which the harness never matches: upper case only.
_VERDICT = re.compile(r"\bCORRECT\b")

#: Instructions to check a task's examples. A function docstring's ``Example:`` line names one of its tests, not a
#: task's example, and matches none of these.
_EXAMPLE_CHECKS = (
    re.compile(
        r"\b(check|verify|validate|confirm|test|try|run)\w*\b[^.\n]{0,60}?\bexamples?\b",
        re.I,
    ),
    re.compile(r"\bexamples?\b[^.\n]{0,20}\bfirst\b", re.I),
    re.compile(
        r"\b(training|train|worked|given|task'?s?|request'?s?|input[- ]output)\s+(examples?|pairs?)\b",
        re.I,
    ),
)


def benchmark_words(text: str) -> list[str]:
    return [m.group(0) for m in _BENCHMARK.finditer(text)] + [
        m.group(0) for m in _VERDICT.finditer(text)
    ]


def example_checks(text: str) -> list[str]:
    return [m.group(0) for rx in _EXAMPLE_CHECKS for m in rx.finditer(text)]


#: Phrases (lower case) that only the WRITE brief may hold, only the CURATE brief, and only the two briefs.
WRITE_ONLY = (
    "batch_map.json",
    "/inputs/staging",
    "/inputs/pool",
    "/inputs/helpers",
    "`delegate`",
    "`similar`",
    "`replay`",
    "`pool`",
)
WRITER_ONLY = ("manifest", "the gate", "finish(")

#: Each rule the prompts state: (rule, topic, owners, the phrase that states it). The phrase must be in every
#: owner's text and in no other prompt (tests/memory/test_prompts_library_contradictions.py).
RULES: tuple[tuple[str, str, tuple[str, ...], str], ...] = (
    (
        "status meanings",
        "library item status",
        ("guide",),
        "`on-trial` items are candidates",
    ),
    (
        "find matches structure only",
        "how find matches",
        ("guide",),
        "It cannot match plain text",
    ),
    (
        "match compares structure only",
        "how match matches",
        ("guide",),
        "It compares structure only",
    ),
    (
        "an input refusal means do the step yourself",
        "input refusals",
        ("guide",),
        "Then do that step with your own code",
    ),
    (
        "store anything that repeats",
        "what to store",
        ("write",),
        "Store anything that repeats",
    ),
    (
        "procedures are replayed",
        "procedure verification",
        ("write",),
        "A procedure is verified by replaying its episode",
    ),
    (
        "start from the actor's code",
        "where code comes from",
        ("write",),
        "Start from the working model's own code",
    ),
    (
        "outcomes are judged from the records",
        "outcome evidence",
        ("write",),
        "Judge outcomes from the records",
    ),
    ("failure-only work is a note", "notes", ("write",), "becomes a note"),
    (
        "WRITE never removes",
        "pass roles",
        ("write",),
        "This pass never removes library items",
    ),
    ("trust order", "trusted sources", ("write", "curate"), "in this order of trust"),
    (
        "exact trusted assertion",
        "test assertions",
        ("write",),
        "equals an exact trusted value",
    ),
    (
        "no self-oracle",
        "test assertions",
        ("write",),
        "Never compute an expected value with the function's own expression",
    ),
    (
        "drawn inputs appended",
        "drawn test inputs",
        ("write",),
        "more recorded inputs of the same kind to a copy of that file",
    ),
    ("literal rule", "literals", ("write",), "make it a parameter"),
    (
        "deterministic tests",
        "qa determinism",
        ("write", "curate"),
        "never assert on the clock",
    ),
    ("no credentials", "credentials", ("write", "curate"), "Never store credentials"),
    ("why trailer", "commit trailer", ("write", "curate"), '"why"'),
    ("repair rounds", "repair rounds", ("write", "curate"), "you get a repair round"),
    ("drafts", "drafts", ("write", "curate"), "kept as a draft"),
    (
        "merge keeps aliases",
        "aliases",
        ("curate",),
        "keeping each old name as an alias",
    ),
    (
        "standing tests green",
        "existing tests",
        ("curate",),
        "keeps the library's existing tests green",
    ),
    (
        "retired fixtures stay used",
        "retirement",
        ("curate",),
        "stay used by a remaining function's tests",
    ),
    ("two episodes decide", "evidence count", ("curate",), "one episode never decides"),
)

#: Instructions and surfaces of the earlier memory design that are retired: none may appear in any text the actor or
#: the writer reads in library mode.
V2_RETIRED = (
    "do not guess",
    "Do not write job functions",
    "Do not write workflow notes",
    "Tend the library too",
    "env/<channel>",
    "memory_channels",
    "library.json",
    "previous_gate.json",
    "unify_memory_testkit",
    "memlab",
    "NOTES.md",
    "workflows/",
    "catalog()",
    "check the example first",
    "candidates, not authority",
    "recorded for the next consolidation",
    "proposals/",
    "memory.diff",
)

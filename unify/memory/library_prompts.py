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

#: The look-first bullet, first under "Finding items" in every guide: at the start of a request, list the stored
#: procedures whose recorded input fits what the agent holds. It informs; nothing runs.
MATCH_LINE = (
    "- At the start of a request, `memory.match(request)` lists the stored procedures whose recorded input fits the "
    "request's text; pass any data you were given as `inputs=` too. It compares structure only, "
    "so a match can be wrong, and it runs nothing. Each row gives a procedure's id, evidence level, status and the "
    "goal it was saved for; attempts that were tried on input like yours and failed are listed apart.\n"
)
#: The try-first bullet, first under "Using items" in every guide: a nudge, never a requirement.
RUN_LINE = (
    "- When a listed procedure fits, try it before working the step out yourself: `memory.run(\"<id>\", ...)` runs "
    "it on the arguments you pass and reports whether it returned, its result, whether a second run agreed and "
    "whether its stated conditions held; a failure is reported, not raised. Trying a matching stored procedure "
    "costs one call, and rediscovering it usually costs many. Its result is unchecked unless its record says "
    "otherwise.\n"
)

GUIDE_LIBRARY = (
    "### Memory Library\n"
    "\n"
    "A Python library of tested functions and linked notes, built from earlier work, is already available "
    "in your cells as `memory`, no import needed. Before you write code for a step, look in the index for an item "
    "that does it. Using the library is optional: use an item when it fits, and otherwise work as usual.\n"
    "\n"
    "Finding items:\n"
    + MATCH_LINE
    + "- The index at the end of this prompt lists every function (its id, signature, one-line summary and "
    "status, then a call quoted from the function's tests, when it has one) and every note (its path, "
    "description and the functions it uses). When the index is long, it shows package headings only.\n"
    '- `print(memory.index("<name>"))` prints the full section of a package or notes topic, and '
    "`print(memory.index())` the whole index.\n"
    "- `print(memory.show(\"<id>\"))` prints one item in full: a function's source and docstring, or a note's "
    "text, then its links, status, test and use records, and recent changes.\n"
    "- `memory.find(value)` lists the functions whose tests call them on arguments shaped like a structured "
    "value you hold (a dict, a list of records, JSON text, or a CSV file with a header), and which parameter "
    "matched. It cannot match plain text: for text, use the index or grep.\n"
    + FILES_LINE
    + "\n"
    "Using items:\n"
    + RUN_LINE
    + "- Import a function by its id: `memory.text.dates:parse_date` is "
    "`from memory.text.dates import parse_date`.\n"
    "- A function raises `MemoryInputError` on an input it was not built for. Then do that step with your own "
    "code.\n"
    "- `on-trial` items are candidates: they passed the library's checks when they were added and have "
    "had little use. `adopted` items passed them and have been used without errors. Items found to fail are "
    "`retired`: left out of the index, and `memory.show` says why. "
    "A function's brackets also give its evidence level: `evidence: unchecked` (stored, but some check of its "
    "evidence did not hold; `memory.show` lists which), `evidence: reproduces-its-case` (its tests reproduce its "
    "recorded cases and every evidence check holds) or `evidence: confirmed-by-outcomes` (outcomes after use "
    "confirm it).\n"
    "\n"
    "Write your own code and files in your session as usual.\n"
)

#: Where the guide states each part of a role prompt (the guide is prose, so the parts are phrases, not headings).
GUIDE_PARTS: tuple[tuple[str, str], ...] = (
    ("role", "is already available in your cells as `memory`"),
    ("inputs", "The index at the end of this prompt lists"),
    ("outputs", "Write your own code and files in your session as usual"),
    ("rules", "Then do that step with your own code"),
    ("done", "Using the library is optional"),
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
    return GUIDE_LIBRARY_IN_CODE if _in_code() else GUIDE_LIBRARY


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
LOOK_LINE_POINTER_IN_CODE = (
    "- At the start of a request, and before you work a step out yourself, look for an item that applies with "
    "`memory.items`, which browses every function and note in the library.\n"
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

#: The guide when the actor's prompt points to the index instead of carrying it
#: (``UNIFY_MEMORY_LIBRARY_ACTOR_INDEX=pointer``). Constant bytes, naming nothing the library holds, so the cached
#: prefix never changes as the library grows; the index is reached in a cell.
#: How the pointer guides say the index is reached (both forms).
POINTER_INDEX_LINE = (
    "- `print(memory.index())` prints the library index: every function (its id, signature, one-line summary, "
    "status and effect: whether it reads, writes or is unknown, then a call quoted from the function's tests, "
    "when it has one, and for a stored procedure the structure of the input it was saved against) and every note "
    "(its path, description and the functions it uses). "
    '`print(memory.index("<name>"))` prints the section of a package or notes topic.\n'
)
#: How the pointer guides say ``memory`` is reached: the worker binds it in every cell (``hooks.worker_globals``).
POINTER_BOUND = (
    "A Python library of tested functions, stored procedures and linked notes, built from earlier work, is already "
    "available in your cells as `memory`, no import needed. It may be empty. "
)

GUIDE_POINTER = (
    "### Memory Library\n"
    "\n"
    + POINTER_BOUND
    + "Before you write code for a step, look in the library for an "
    "item that does it. Using the library is optional: use an item when it fits, and otherwise work as usual.\n"
    "\n"
    "Finding items:\n"
    + MATCH_LINE
    + POINTER_INDEX_LINE
    +     "- `print(memory.show(\"<id>\"))` prints one item in full: a function's source and docstring, or a note's "
    "text, then its links, status, test and use records, and recent changes.\n"
    "- `memory.find(value)` lists the functions whose tests call them on arguments shaped like a structured "
    "value you hold (a dict, a list of records, JSON text, or a CSV file with a header), and which parameter "
    "matched. It cannot match plain text: for text, use the index or grep.\n"
    + FILES_LINE
    + "\n"
    "Using items:\n"
    + RUN_LINE
    + "- Import a function by its id: `memory.text.dates:parse_date` is "
    "`from memory.text.dates import parse_date`.\n"
    "- A function raises `MemoryInputError` on an input it was not built for. Then do that step with your own "
    "code.\n"
    "- `on-trial` items are candidates: they passed the library's checks when they were added and have "
    "had little use. `adopted` items passed them and have been used without errors. Items found to fail are "
    "`retired`: left out of the index, and `memory.show` says why. "
    "A function's brackets also give its evidence level: `evidence: unchecked` (stored, but some check of its "
    "evidence did not hold; `memory.show` lists which), `evidence: reproduces-its-case` (its tests reproduce its "
    "recorded cases and every evidence check holds) or `evidence: confirmed-by-outcomes` (outcomes after use "
    "confirm it).\n"
    "\n"
    "Write your own code and files in your session as usual.\n"
)

#: :data:`GUIDE_POINTER` when tools are called from code (``UNIFY_TOOLS_IN_CODE=on``): the actor's memory section,
#: rendered after the shared opening (which already says how cells and tools work), so it carries only the
#: library's own guidance: when to use which library call, each named without its arguments. Constant bytes, naming
#: nothing the library holds.
GUIDE_POINTER_IN_CODE = (
    "### Memory Library\n"
    "\n"
    + POINTER_BOUND
    + MEMORY_HELP_LINE
    + " Using the library is optional: use an item when it fits, and otherwise work as usual.\n"
    "\n"
    "Finding items:\n"
    + LOOK_LINE_POINTER_IN_CODE
    + FILES_LINE_IN_CODE
    + "\n"
    "Using items:\n"
    + USE_LINE_IN_CODE
    + STATUS_LINES_IN_CODE
)


def guide_pointer() -> str:
    """The pointer guide the actor reads (``UNIFY_MEMORY_LIBRARY_ACTOR_INDEX=pointer``): :data:`GUIDE_POINTER`, or
    :data:`GUIDE_POINTER_IN_CODE` when tools are called from code."""
    return GUIDE_POINTER_IN_CODE if _in_code() else GUIDE_POINTER


#: The constant pointer to the librarian (``UNIFY_MEMORY_LIBRARIAN=on``): the same bytes for every library, in place
#: of the index. It says who answers ``memory.ask``; the answer's form is in the call's help.
LIBRARIAN_POINTER = (
    "Your library of procedures and notes from earlier work is the `memory` package in your cells. A librarian, "
    "who has read every item, what each was built for and how each has been used, answers `memory.ask`."
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

#: What a drafting agent (the episode-end fork, a helper) is told about its drafts: draft what it saw; fitting the
#: drafts to the shared library and generalising them is the writer's work.
CANDIDATE_CONTRACT = (
    "Draft from what you saw: code that worked, the cases it ran on (values as they were recorded), and notes. "
    "Where you can, write a reusable function or lesson in general form, with the values that vary as "
    "parameters. Where your work is specific to this case, still write it down as it is: the library writer, who "
    "alone commits to the shared library, builds from these drafts and may generalise them with other episodes."
)

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

_REPAIR = (
    "# Repair\n"
    "If the gate refuses anything, its full result (every check, every failing test's output and every note) is "
    "written to /inputs/gate/result-<round>.md and you get a repair round: fix what it refused, keep the manifest "
    "current, and call `finish` again. The limits below say how many repair rounds a pass has. "
    "The items that pass land, and the rest is kept as a draft for a later pass, with its failure.\n"
)

#: The example-evidence rule exactly as the gate checks it (``example_evidence.evidence_check``, whose
#: ``MIN_EXAMPLE_EPISODES`` a test keeps equal to this): the brief states it in these words, and the example refusals end
#: with the fix that satisfies them. Values from more episodes are recorded and shown beside the function, never
#: required.
EXAMPLE_EPISODES = 1
EPISODES_RULE = (
    "the calls of a function that return take fixture values from a recorded episode: one episode is enough. Calls "
    "that return on same-shaped values recorded in other episodes, in this batch or not, are recorded and shown "
    "beside the function (they need no exact expected value), never required. A call that refuses or raises counts "
    "for nothing"
)
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

#: What each check the gate names in its refusals and results means (the same names: G1 to G6).
_CHECK_NAMES = (
    "Its results and refusals name the check: G1, layout and provenance (only `memory/` and `notes/` change and "
    "never the generated files, the manifest is well formed, every link names an item that exists, every "
    "fixture came through `fixture` unchanged, a parsed one re-made by its reader, every new or changed function "
    "and note has its `When to use` section, and every folder the pass changes has its README.md); G2, evidence (each "
    "function's listed tests call it, and the calls they make of it return or raise `MemoryInputError`; what else "
    "those calls show sets its evidence level: `unchecked` when a call that returns uses a value from a recorded "
    "episode in any form but some evidence check does not hold, `reproduces-its-case` when its structured "
    "arguments come from fixtures or through a stored reader, with those values replaced by unseen values of the "
    "same type a call it accepted neither refuses nor raises, its docstring's `Example:` names one of its tests, "
    "its typed covers are recorded evidence of their type, and the literal rule holds); G3, tests (plain pytest, "
    "the tests pass on the candidate, a behaviour change has a test that fails on the parent, and no test is lost "
    "or weakened without a reason; an exact assertion against a trusted value on a recorded fixture for each "
    "changed function, no test that recomputes the function, and passing on the drawn recorded inputs and "
    "catching enough small mutations also set the evidence level); G4, the index (it never "
    "refuses: it notes when the index has outgrown the view the working model is shown); G5, items (<<g5>>); "
    "G6, safety (no links, executables, submodules, git, pytest or interpreter configuration files, bytecode or "
    "key-shaped strings).\n"
)

_G5_WRITE = (
    "a store pass never ends with fewer items, and every fixture a function's calls ran on is still used by a "
    "function's calls"
)
_G5_CURATE = (
    "the fixtures a deleted function's calls ran on are used by a remaining function's calls, and each alias and "
    "retirement names an item"
)

_GATE = (
    "# The gate\n"
    "After `finish` a deterministic gate checks the candidate: the layout and the links; that every fixture "
    "came through `fixture` unchanged; that the calls of each function that return take fixture values from "
    "a recorded episode; its tests on the candidate, and on the parent where behaviour "
    "changed; that a call it accepted neither refuses nor raises when the fixture values its tests load are "
    "replaced by unseen values of the same type; the appended recorded inputs; that its tests catch at least "
    "<<mutation_min_pct>>% of small mutations of each changed function; the literal rule; and that no test is "
    "lost or weakened without a reason. An item that breaks a rule is refused together with the items that import it, "
    "and the rest can land; a breach of a pass-wide rule (the layout, lost tests, fewer items) refuses the "
    "round. " + _CHECK_NAMES.replace("<<g5>>", _G5_WRITE)
)

_CURATE_GATE = (
    "# The gate\n"
    "After `finish` a deterministic gate checks the candidate: the layout and the links, the aliases and "
    "retirements, every test on the candidate (and on the parent where behaviour changed), and that the "
    "fixtures a deleted function's tests called it on are still used by a remaining function's tests. An item "
    "that breaks a rule is refused together with "
    "the items that import it, and the rest can land. "
    + _CHECK_NAMES.replace("<<g5>>", _G5_CURATE)
)

_LIMITS = (
    "# Limits\n"
    "These are the run's safety limits, stated as facts so that none surprises you; none is a target, and there "
    "is no token, call or spending allowance to manage. A reply that reads a file or a record shows at most "
    "<<view_bytes>> bytes and ends with a marker naming the next offset; nothing is cut silently. A cell runs for "
    "at most <<cell_s>> s<<per_turn>>. Files under /memory are limited to <<file_mib>> MiB each, <<total_mib>> MiB "
    "in all and <<entries>> files and directories; a breach refuses the pass. A module in /memory named like a "
    "standard-library module shadows that module in your cells for the rest of the pass. The pass ends at its wall "
    "clock (its "
    "length in seconds is in the first message<<helper_time>>) or after <<step_guard>> model and check calls, "
    "whichever comes first, and a `check` starts only while at least <<check_floor>> s of the wall clock remain."
    "<<repair_cap>><<finish_cap>><<compaction>>\n"
)

#: The writer's library bullet with tools in code off under ``UNIFY_CELL_STATE=fresh``: each cell is a fresh script,
#: which imports ``memory`` itself.
_LIBRARY_BULLET_FRESH = (
    "- /memory: a writable copy of the library. There is no git history in the box; an item's recent changes "
    'are in `memory.show("<id>")`. /memory/INDEX.md is the library index, generated by the harness like '
    "/memory/links.json and /memory/memory/__init__.py. In a cell, `import memory` gives `memory.index()`, "
    '`memory.index("<package>")`, `memory.show("<id>")` and `memory.find(value)`, as the working model has them. '
    "For: knowing what exists before you add anything.\n"
)

_LIBRARY_BULLET = (
    "- /memory: a writable copy of the library. There is no git history in the box; an item's recent changes "
    'are in `memory.show("<id>")`. /memory/INDEX.md is the library index, generated by the harness like '
    "/memory/links.json and /memory/memory/__init__.py. In every cell `memory` is already bound (no import "
    'needed): `memory.index()`, `memory.index("<package>")`, `memory.show("<id>")` and `memory.find(value)`, as '
    "the working model has them. Names you bound earlier keep the code they were bound to; after you edit a "
    "library file, import it again. For: knowing what exists before you add anything.\n"
)

#: The "when to use" section every stored function and note carries and the README every folder holds, as the writer
#: and the curator are told them (:mod:`.when_to_use`: the gate checks the shape, never the words).
WHEN_TO_USE_TEXT = (
    "- Every function and note carries a section saying when to use it, written for a reader who has a new "
    "request and must decide whether this item fits it. In a function's docstring it is a `When to use:` section; "
    "in a note it is the body's `## When to use` heading. Either holds four labelled lines, each with its text: "
    "`Saved for:` what the request it was saved from asked for, summarised in general terms (no names, ids or "
    "values of that request); `Recognise:` how to tell that a new request is that situation; `Not for:` the "
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


_WHAT_TO_WRITE = (
    "# What to write\n"
    "- Functions go in `memory/<package>/<module>.py`, in packages you choose by what the code does. Each "
    "package has an `__init__.py` whose docstring is one paragraph on what lives there. A function's id is "
    "`memory.<package>.<module>:<name>`; a name starting with `_` is private and is not an item.\n"
    "- A procedure is a function that takes the value a stored reader gives on a request and returns the result "
    "that was handed back.\n"
    "- A function's docstring has a one-line summary; `Args:`, `Returns:` and `Raises:` sections "
    "(`MemoryInputError` on an input it was not built for); an `Example:` section of one line naming one of "
    "the function's tests, `memory/<package>/tests/test_<module>.py::test_<case>`, with no doctest and no "
    "`Input:` line; a `When to use:` section (below); and, optionally, a `Notes:` line naming note paths.\n"
    "- Notes go in `notes/<topic>/<slug>.md`: free-form markdown of any kind (a fact, a pitfall, a how-to, or "
    "how to combine functions), starting with front matter: `title`, `description` (one line, shown in the "
    "index) and `uses` (the ids of the functions it uses). A link is written only in a note's `uses` or a "
    "function's `Notes:` line, and must name an item that exists. A note states its rule, procedure, fact or "
    "pitfall in general terms where it can; a rule for one kind of request is worth keeping when that kind of "
    "request comes back. A new item starts as `on-trial`; later passes and the curate pass revise, generalise "
    "or remove it as evidence arrives, so storing a rule that worked is not a final commitment. A function's "
    "linked note says its preconditions and its known failures.\n"
    + WHEN_TO_USE_TEXT
    + "- Tests go in `memory/<package>/tests/test_<module>.py`: plain pytest, importing only the library and the "
    "standard library, with fixtures under `memory/<package>/tests/data/`. `tests/` takes no `__init__.py`: "
    "pytest imports each test file by its path. " + _RUN_TESTS + "\n"
    "- The manifest is /memory/.pass/manifest.json (never committed):\n"
    '  {"items": [{"item": "<id>", "kind": "function" or "note", "source_episodes": [...], "tests": [...], '
    '"covers": [...]<<item_extra>>}], "support": [], "tests_changed": {}, "renames": {}, <<manifest_extra>>"why": '
    '"<one line>", "summary": "<what changed>"}\n'
    "  - A note's entry takes `item`, `kind` and `source_episodes`: "
    '`{"item": "notes/<topic>/<slug>.md", "kind": "note", "source_episodes": [...]}`; `tests`, `covers` and '
    "`field_types` are for functions.\n"
    "  - A function takes any arguments; its examples are the calls its listed tests make, so its `tests` name at "
    "least one test file.\n"
    '  - `field_types` (`{"<fixture path>#<field>": "<type>"}`) declares a field with a fixed domain: the path is '
    "a fixture of the function's package under `memory/<package>/tests/data/`; the field is a JSON or YAML key path "
    "(`a.b`, `items[]`), a CSV column (`#3` for the third column of a file without a header) or, in a JSON-lines "
    "file made with `append=true`, a key path under `[].input`; the type is one of <<semantic_types>>. Leave it out "
    "when no field has a fixed domain.\n"
    "  - `covers` takes typed covers only, the recorded work a function's tests check: "
    '`{"episode": e, "type": "cell", "cell": i}` or `{"episode": e, "type": "diff"}` for the working model\'s own '
    'code; `{"episode": e, "type": "episode", "runner": "worktree" | "tool" | "dialogue", ...}` for a whole '
    "procedure. A recorded action is not a cover: a function's evidence is its tests' calls on fixtures.\n"
    "  - `support` lists helper files: a test helper under a package's `tests/` directory, or a shared module "
    "`memory/<package>/<module>.py` that the package's items import, holding private (`_`) functions and classes "
    "such as the package's `MemoryInputError` (a public function is an item). The harness records "
    "fixtures itself.\n"
    "<<manifest_extra_docs>>"
    '  - `renames` (`{"<old item id>": "<new item id>"}`) declares an item you moved or renamed: it keeps its '
    "history (use, status, provenance) under the new id.\n"
    "  - `why` is one line on why the change helps. The harness writes it, the episodes and the item ids into "
    "the commit.\n"
)

_WRITE_RULES = (
    "# Rules\n"
    "1. Store anything that repeats or is likely to repeat, in any domain: readers, calculations, call "
    "sequences, and whole-job procedures built from smaller tested functions. A procedure is verified by "
    "replaying its episode: a work-tree procedure must reproduce the recorded file changes, a tool procedure the "
    "recorded calls, and a dialogue procedure the recorded reply the environment then accepted. A procedure "
    "of any other kind cannot be verified, so it stays a draft.\n"
    "2. Start from the working model's own code when it already wrote it (its cells and its work-tree diff). "
    "Generalise it: values that vary become parameters, and the function raises "
    '`MemoryInputError("<what was expected>")` on an input it was not built for. Check shapes, not observed '
    "values. The gate replaces the fixture values the tests load with unseen values of the same type and reruns "
    "the tests: the function must not then raise `MemoryInputError` or another exception on a call it accepted "
    "before. Refuse shapes, not values; a field with a fixed domain is declared in `field_types`."
    "<<replay_rule>>\n"
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
    + " A function takes any arguments, and its examples are the calls its tests make. At least one call returns on "
    "arguments loaded from fixtures made with `fixture`, or computed in the test from such a fixture: the test's "
    "code is the transformation, and the gate checks that the argument changes when the fixture changes. An item "
    "whose evidence does not reach that is still stored, at a lower evidence level, and later evidence can raise "
    "it. "
    "Across its examples, "
    + EPISODES_RULE
    + ". Each function needs at least one test asserting "
    "that its result on a recorded fixture equals an exact trusted value, and one test that it refuses an input of "
    "another shape. A procedure's test checks the "
    "effect the environment later confirmed, not only that it ran. Never compute an expected value with the "
    "function's own expression. "
    + _DETERMINISM
    + " For a function you add or change, run each new test and see it fail before your change, then pass.\n"
    "6. Each function has a recorded-inputs file, "
    "`memory/<package>/tests/data/<module>.<function>.inputs.jsonl`, with one line per recorded input made with "
    "`fixture(..., append=true)`, and a test parametrised over its lines. The gate appends <<sample_k>> more "
    "recorded inputs of the same kind to a copy of that file, and the function's tests must pass on every one.\n"
    "7. Code never holds a literal equal to a value that varies across its recorded inputs (text of "
    "<<lint_min>> or more characters, or a number with 2 or more significant digits): make it a parameter.\n"
    "8. Never store credentials, session tokens or the results of calls with side effects. "
    + _HONESTY
    + "\n"
)

#: The curator's library bullet with tools in code off: its interpreter binds ``memory``.
_CURATE_LIBRARY_BULLET = (
    "- /memory: a writable copy of the library. There is no git history in the box. In every cell `memory` is "
    'already bound (no import needed): `memory.index()`, `memory.find(value)`, and `memory.show("<id>")`, which '
    "gives an item's status, its test record, its use record per library commit and its recent changes. Names you "
    "bound earlier keep the code they were bound to; after you edit a library file, import it again. "
    "/memory/INDEX.md is the library index, generated by the harness.\n"
)
#: The same under ``UNIFY_CELL_STATE=fresh``: each cell is a fresh script, which imports ``memory`` itself.
_CURATE_LIBRARY_BULLET_FRESH = (
    "- /memory: a writable copy of the library. There is no git history in the box. `import memory` works in your "
    'cells: `memory.index()`, `memory.find(value)`, and `memory.show("<id>")`, which gives an item\'s status, its '
    "test record, its use record per library commit and its recent changes. /memory/INDEX.md is the library index, "
    "generated by the harness.\n"
)

_CURATE = (
    "# Role\n"
    "You are the library curator. The library, `memory`, is a Python package of tested functions and linked "
    "notes that a working model imports. You keep it small, correct and easy to navigate: merge overlapping "
    "items, generalise variants, repair or roll back items retired by their outcomes, retire wrong or unused ones, "
    "fix links and "
    "regroup packages. You work in a sandbox with no network and no credentials, on the library alone.\n"
    "\n"
    "# Inputs\n"
    "- <<trigger_path>>: why this pass runs (its reasons, and the library commit it starts from). For: where to "
    "begin.\n"
    "- <<overlaps_path>>: the overlap candidates the harness found: functions with matching argument "
    "shapes and overlapping input fixtures, functions whose bodies share a common generalisation (given with it), and "
    "notes that use the same functions. For: what to merge or generalise.\n"
    "- /inputs/curate/same_shape.json: groups of functions whose bodies have the same structure. For: more "
    "merge and generalisation candidates.\n"
    "- <<suspects_path>>: the items retired by their outcomes, each with why it was retired (errors or negative "
    "outcomes after use, in at least two episodes, or a dependency so retired), the episodes involved, the first "
    "version whose behaviour "
    "changed, and, when an earlier version has a clean use record, that version as the proposed rollback. An "
    "input refusal (a `MemoryInputError` raised on an input the item was not built for) is counted and reported, "
    "never a failure. For: what to repair or roll back.\n"
    "- /inputs/curate/rollback/<n>/: the files (module and tests) of each proposed rollback version; "
    "/inputs/curate/rollback/index.json names each one's item, directory and commit. For: rolling back; these "
    "files stand in for git history, which the box does not have.\n"
    "- /inputs/curate/use.json: how each item was used (imported, called, shown, found), with errors, per library "
    "commit. /inputs/curate/history.json: each item's recent changes, as commit and subject. "
    "/inputs/curate/aliases.json: the old names kept as aliases, the commit that added each, and whether this "
    "pass may remove it. For: judging what is used, what changed and what may go.\n"
    "- /inputs/episodes/<episode>.json: the recorded episodes the retired items were used in. For: the evidence "
    "behind a retirement; `read_episode` and `overview` show them part by part.\n"
    "<<library_bullet>>"
    "- /inputs/gate/: the full result of each `check` and each repair round of this pass, as it happens.\n"
    "\n"
    "# Tools\n"
    "<<tools>>"
    "\n"
    "# What to do\n"
    "- Merge overlapping items into one, keeping each old name as an alias: `old_name = new_name` in its "
    "module, listed in the manifest's `aliases`.\n"
    "- Generalise variants into one parametric function.\n"
    "- Repair an item retired by its outcomes, or roll it back to the proposed earlier version from "
    "/inputs/curate/rollback/.\n"
    "- Retire an item that is wrong, or unused across many episodes, together with its tests, with a one-line "
    "reason in `retired`.\n"
    "- Fix broken links and add missing ones between notes and functions.\n"
    "- Regroup packages when that makes the index clearer, keeping each moved function's old id as an alias.\n"
    "- Keep each item's `When to use` section and each folder's README.md true to what they hold now: a merged or "
    "generalised item's section covers the requests of the items it replaces. "
    + WHEN_TO_USE_TEXT[2:]
    + "- Leave alone what is fine, and say why in `kept`: changing nothing is a valid result when no candidate is "
    "really one job.\n"
    "- Write the manifest, /memory/.pass/manifest.json (never committed):\n"
    '  {"items": [{"item": "<id>", "kind": "function" or "note", "source_episodes": [...], "tests": [...], '
    '"covers": [...]}], "deleted": ["<retired id>"], '
    '"retired": {"<retired id>": "<one-line reason>"}, "deleted_tests": ["<test file removed with them>"], '
    '"aliases": {"<old id>": "<new id>"}, "renames": {"<old id>": "<new id>"}, '
    '"kept": {"<item id, or INDEX.md>": "<why it stays as it is>"}, '
    '"tests_changed": {"<test>": "<reason>"}, "why": "<one line>", "summary": "<what changed>"}\n'
    "  `renames` moves an item to a new id without keeping the old name (its history follows it); `aliases` "
    "keeps the old name forwarding to the new one.\n"
    "  `items` lists every item you added or changed, with the tests and fixtures of the items it replaces. "
    "`kept` names each "
    "item a reason in your inputs points at that you leave as it is, with why. "
    + _RUN_TESTS
    + "\n"
    "\n"
    "# Rules\n"
    "1. Every change keeps the library's existing tests green, or retires a test with its reason in "
    "`tests_changed`. The tests of a merged item stay and run through its alias.\n"
    "2. A change in what a function returns on its recorded inputs needs a test that fails on the parent and "
    "passes on yours; a repair always changes behaviour.\n"
    "3. The fixtures a retired function's tests called it on stay used by a remaining function's tests.\n"
    "4. New tests check exact expected values, " + _TRUST + " " + _DETERMINISM + "\n"
    "5. Act on records that span at least two episodes: one episode never decides.\n"
    "6. Fixture files stay where they are: a test may read a fixture from any package's `tests/data/`.\n"
    "7. A reason in your inputs is dealt with when this pass changes an item it names, or keeps that item with a "
    "reason in `kept`; otherwise it is raised again by a later curate pass.\n"
    "8. Never store credentials or session tokens. " + _HONESTY + "\n"
    "\n" + _CURATE_GATE + "\n" + _REPAIR + "\n" + "<<limits>>" + "\n"
    "# Done\n"
    "You are done when each overlap candidate and each item retired by its outcomes in your inputs has been "
    "merged, generalised, "
    "repaired, rolled back or retired, or kept as it is with the reason in `kept`, and your changes have "
    "passed the gate or are kept as a draft with their failure. Then call `finish(summary)`.\n"
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


def _limits_text(limits: Limits, helper_time: bool) -> str:
    per_turn = []
    if limits.cells_per_turn is not None:
        per_turn.append(f"{limits.cells_per_turn} cells")
    if limits.readers_per_turn is not None:
        per_turn.append(f"{limits.readers_per_turn} reading calls")
    return _fill(
        _LIMITS,
        {
            "view_bytes": limits.view_bytes,
            "cell_s": _num(limits.cell_s),
            "per_turn": (
                ", and at most " + " and ".join(per_turn) + " run in one turn"
                if per_turn
                else ""
            ),
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
                f" At most {limits.repair_rounds} repair rounds follow the first refused `finish` in this pass."
                if limits.repair_rounds is not None
                else " There is no set number of repair rounds; only these limits end them."
                + (
                    f" A refusal the gate repeats for the same item and reason in {limits.stuck_repeats} rounds in a "
                    "row gets no further round of its own (the repair message names it), and once every refusal "
                    "of a round is such a repeat, the final merge follows."
                    if limits.stuck_repeats is not None
                    else ""
                )
            ),
            "finish_cap": (
                f" While /memory/.pass/manifest.json is missing or does not parse, `finish` is refused with the reason "
                f"up to {limits.finish_refusals} times in a pass; after that, `finish` ends the pass and the final "
                "check refuses the manifest by name."
                if limits.finish_refusals is not None
                else ""
            ),
            "compaction": (
                f" When a model call's prompt reaches {limits.compact_threshold_percent}% of the model's context "
                "window"
                + (
                    f" or {limits.compact_at_tokens:,} tokens, whichever comes first"
                    if limits.compact_at_tokens
                    else ""
                )
                + ", you are asked to summarise the work so far, and the conversation restarts from these "
                "instructions"
                + (", the pass's request" if limits.compact_keep_prefix else "")
                + " and your summary; nothing under /memory, /inputs or /outputs changes, so `read` shows any of it "
                "again."
                if limits.compact_threshold_percent is not None
                else ""
            ),
        },
    )


def _stage_quota_text(limits: Limits) -> str:
    """The staging quota sentence, from the fork's own constants (empty when they are not given)."""
    if limits.stage_files is None or limits.stage_bytes is None:
        return ""
    return (
        f" What is staged for one episode is at most {limits.stage_files} files and "
        f"{_num(limits.stage_bytes / 1024**2)} MiB in all; a file past that is not staged."
    )


def _tool_names(role: str, flags: frozenset[str]) -> frozenset[str]:
    from .tool_spec import tools_for

    return frozenset(s.name for s in tools_for(role, flags))


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
    staging: bool = False,
    pool: bool = False,
    tools: str = "",
    tool_names: frozenset[str] = frozenset(),
    pool_kinds: str = "",
    in_code: bool | None = None,
    flags: frozenset[str] = frozenset(),
) -> str:
    """The WRITE brief. *staging* (the episode-end fork is on) and *pool* (the candidate pool is on) add their input
    bullets, row keys and manifest fields; *tools* is the rendered Tools section and *tool_names* the tools the role
    gets, so a sentence that relies on an optional tool (``pool``, ``similar``, ``delegate``, ``replay``) appears
    only when that tool exists. Every input is constant for a run, so the brief is byte-stable for the run.
    *in_code* (None: ``UNIFY_TOOLS_IN_CODE``) gives the brief for tools called from code, after the shared opening
    for the configuration *flags* (the flags the writer's allow-list is built with).
    """
    if in_code if in_code is not None else _in_code():
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
                staging=staging,
                pool=pool,
                tool_names=tool_names,
            ),
        )
    from .manifest import describe_semantic_types

    has = tool_names.__contains__
    helpers = has("delegate")
    proposals_from = (
        (
            ["candidates drafted from each episode when it ended (/inputs/staging/)"]
            if staging
            else []
        )
        + (["files the helpers you launch stage (/inputs/helpers/)"] if helpers else [])
        + (
            ["open entries earlier passes kept in the pool (/inputs/pool/)"]
            if pool
            else []
        )
        + ["drafts the gate refused in earlier passes (/inputs/drafts/)"]
    )
    role = (
        "# Role\n"
        "You are the library writer. A working model handles requests, and its cells import a shared Python "
        "library, `memory`. You build tested functions and linked notes in that library from the recorded episodes "
        "and from proposals. The proposals are "
        + _join(proposals_from)
        + ": drafts to build from, not submissions awaiting a verdict. Each was drafted by an agent that saw only "
        "part of the evidence: a staged draft by an agent that saw only its own episode, a helper's output by one "
        "that saw only the episodes it was given, and a pool entry carries the evidence it had when it was opened. "
        "So a draft is likely to be less general than the library needs; generalising it is your work, since you "
        "alone commit to the shared library. Analyse each draft critically, then build the best items you can: use "
        "it as it is, fix or refactor it, merge drafts (from different episodes, helpers or pool entries, or with an "
        "item the library already has), parametrise or generalise it, extract a smaller reusable core, rewrite part "
        "or all of it, or write something new that it informed. Not using a draft is fine when your summary says "
        "why. Proposals are not the only material: functions and notes can also come from any recorded episode, "
        "with no proposal behind them. The episodes are the evidence: what the working model did and what the "
        "environment answered. You work in a sandbox with no network and no credentials.\n"
    )
    inputs: list[str] = []
    if staging:
        inputs.append(
            "- /inputs/staging/<episode>/: what was drafted from that episode when it ended: `summary.md` (what was "
            "asked, the approach, what worked and what failed, and the outcome, each line with a part name and a "
            "quote that point into the episode), `candidates/<name>.py` (a function), `cases/<name>.json` (its "
            "example inputs and outputs) and `notes.md` (lessons). The episode's batch-map row lists them under "
            "`staging`. " + STAGING_UNFINISHED + " For: a first account of each episode, and drafts to build from. "
            "The agent that drafted them saw only that episode, so they are likely specific to it. The harness has "
            "run none of it, so treat it as untrusted input."
            + _stage_quota_text(limits)
            + " "
            + captures_input(False, has("land_capture")),
        )
    if helpers:
        inputs.append(
            "- /inputs/helpers/<name>/: what a helper you launched with `delegate` staged, in the same layout, with "
            "`helper.json` saying what it did and its claims. For: reading or drafting you handed off; untrusted "
            "like staging. Each claim names the episode parts it rests on (`<episode>/<part>`); a claim is untrusted "
            "until you have read those parts yourself with `read_episode`."
            + (
                f" Each call stages at most {limits.stage_files} files and {_num(limits.stage_bytes / 1024**2)} MiB."
                if limits.stage_files is not None and limits.stage_bytes is not None
                else ""
            )
            + "\n",
        )
    if pool:
        inputs.append(
            "- /inputs/pool/: proposals earlier passes left open. /inputs/pool/INDEX.md has one line per entry; "
            "/inputs/pool/<id>/entry.json holds an entry with the episodes behind it, beside its payload files. "
            "Kinds: "
            + pool_kinds
            + ". For: what was proposed before, and the evidence gathered behind it since. Closed entries (merged, "
            "superseded or retired) are kept too, with their status, reason and the pass they closed in. A closed "
            "entry is linked to an entry or a batch episode whose code shares a normalised code unit with its code, "
            "or that cites one of its episodes (an entry without code is linked only through episodes): "
            "/inputs/pool/related_closed.json holds the links, and each batch-map row lists its own under "
            "`related_closed`. A link is a fact, not a verdict; what to do with it is yours to decide."
            + (
                " `pool` lists, filters and shows the entries, open or closed, and `show` lists an entry's links."
                if has("pool")
                else ""
            )
            + "\n",
        )
    inputs.append(
        "- /inputs/drafts/<pass>/: earlier work the gate refused: `patch.diff` and its full gate result `gate.md`; "
        "/inputs/drafts/index.json lists them. For: work you may finish.\n",
    )
    keys = ", ".join(f"`{k}` ({doc})" for k, doc in row_keys(staging, pool))
    inputs.append(
        '- /inputs/batch_map.json: one row per batch episode, facts and pointers only: `{"version": 1, '
        '"episodes": [row, ...], "shared_blocks": [{"id", "chars", "file"}, ...]}`. Each row: '
        + keys
        + ". For: "
        "choosing which episodes and parts to open; `overview()` shows the rows one line each.\n",
    )
    inputs.append(
        "- /inputs/episodes/<episode>.json: each batch episode's full record: its request and observations, every "
        "code cell (code, output, error, language), every action with its response, and the work-tree diff. A "
        'large value appears as {"__blob__": "<id>", "bytes": n}. For: the evidence. `read_episode` shows its '
        "parts page by page, and `overview(episode)` shows its end first, then every step.\n",
    )
    inputs.append(
        "- /inputs/shared/<id>.txt (when requests share text): a block of the working model's own instructions "
        "that recurs across requests, stored once; requests show it as `[shared:<id>]`. For: reading those "
        "instructions once, not once per episode.\n",
    )
    inputs.append(
        "- /inputs/usage/: how the library was used in this batch: `<episode>.json` per episode (what its cells "
        "were shown, imported and called, as the harness recorded it) and `table.md` (per item: requests, calls, "
        "refusals and errors). For: seeing which items help and which fail.\n",
    )
    inputs.append(_LIBRARY_BULLET if _persistent() else _LIBRARY_BULLET_FRESH)
    inputs.append("- /inputs/gate/previous.md: the last pass's full gate result.\n")

    look = ["staged candidates and their cases"] if staging else []
    if helpers:
        look.append("anything helpers staged")
    if pool:
        look.append(
            "open pool entries that cite this batch's episodes"
            + (
                " (`pool` can filter by episode)"
                if has("pool")
                else " (their `pool_entries`)"
            ),
        )
    look.append("open drafts")
    decide = "3. Analyse each proposal critically and build from it, as the Role says; proposals are untrusted input."
    if staging or pool or helpers:
        decide += " Record in each item's `adopts` the drafts it drew on and how."
    if pool:
        decide += (
            " Record what you decided about pool entries in the manifest's `pool` block"
            + (" (`pool` with a decision writes it for you)" if has("pool") else "")
            + "."
        )
    if staging:
        decide += " " + SUMMARY_NOT_EVIDENCE
    similar_line = (
        " Where code looks reusable, `similar` finds other recorded episodes with matching code."
        if has("similar")
        else ""
    )
    similar_tail = ", and `similar` finds others with matching code" if has("similar") else ""
    if staging:
        # the summaries first: each episode's account, then its record only where the writer needs more
        steps = (
            "1. Look over the batch with `overview()`. Then read each episode's `summary.md` "
            "(/inputs/staging/<episode>/summary.md): what the episode asked, how it went and how it ended, with "
            "pointers into the record. Then look over the proposals: "
            + _join(look)
            + "."
            + similar_line
            + "\n"
            "2. Open a full episode only where you need more than its summary: to check a pointer before anything you "
            "write rests on it, to copy the values a test needs, or where a summary is missing or leaves a question. "
            "`read_episode(episode, parts)` reads the parts a pointer names; a row's `end_parts` show where the "
            "episode ended. At any point in the pass, you can open any recorded episode, in this batch or not and "
            "from earlier passes too: `overview` and `read_episode` take any recorded id"
            + similar_tail
            + ".\n"
        )
    else:
        steps = (
            "1. Look over the batch with `overview()`, then over the proposals: "
            + _join(look)
            + "."
            + similar_line
            + "\n"
            "2. Read the episodes behind what may be worth keeping, starting from each one's `end_parts`, which show "
            "where it ended and so where an outcome, if any, was shown. At any point in the pass, you can open any "
            "recorded episode, in this batch or not and from earlier passes too, whenever you need to investigate or "
            "dig deeper: `overview` and `read_episode` take any recorded id"
            + similar_tail
            + ".\n"
        )
    procedure = (
        "# How to work\n"
        "This is a procedure, not a quota: read as much or as little as the material needs.\n"
        + steps
        + decide
        + "\n"
        "4. Write tests first, against trusted values (rule 5), with fixtures made by `fixture`; see each new test "
        "fail before the code exists, then pass.\n"
        + (
            "5. When it helps, hand a reading or drafting task to a helper with `delegate`. It returns at once and "
            "the helper works while you go on; it messages you on the record as it goes (`messages` reads them, "
            "`wait` waits for the next, `reply` answers) and ends with a summary and claims, each with the episode "
            "parts it rests on, and its files under /inputs/helpers/<name>/. Read those parts yourself before you "
            "rely on a claim. You decide when you have enough: `stop` ends a helper, and finishing ends every "
            "helper still running, keeping what it staged. Using helpers is your choice.\n"
            if helpers
            else ""
        )
        + f"{6 if helpers else 5}. `check` the manifest as often as you like, then `finish`. If the gate refuses "
        "something, repair it (below).\n"
    )
    manifest_extra = ""
    manifest_docs = ""
    item_extra = ""
    if staging or pool or helpers:
        item_extra = ', "adopts": [...]'
        forms = (
            (["`staging:<episode>/candidates/<name>.py`"] if staging else [])
            + (["`helper:<helper name>/candidates/<name>.py`"] if helpers else [])
            + (["`pool:<id>`"] if pool else [])
        )
        manifest_docs += (
            '  - An item\'s `adopts` lists the drafts it drew on, each as `{"ref": <ref>, "how": <how>}`. The ref is '
            + _join(forms)
            + ". How is `as_is` (used unchanged), `edited`, `merged` (with other drafts or a library item), "
            "`generalised` (parametrised or generalised), `rewritten`, or `informed` (it only informed new work). "
            "Leave it out when the item drew on none. The manifest itself has no `adopts`.\n"
        )
    if pool:
        manifest_extra += '"pool": {...}, '
        manifest_docs += (
            '  - `pool` records decisions on pool entries: `{"merged": {"<id>": "<item>"}, "superseded": '
            '{"<id>": "<id or item>"}, "retired": {"<id>": "<reason>"}}`. An entry is merged into an item this '
            "manifest lists, superseded by another entry or an item, or retired with a reason. `retired` closes an "
            "entry for good: it is wrong or not reusable. To keep an entry open for a later pass, defer it "
            "(`pool` with decision `defer`, which writes the block's `defer` rows). "
            '`"reopened": {"<id>": "<reason>"}` opens a closed entry again.\n'
        )
    replay_rule = (
        " A function lifted from the working model's cells can be verified by listing in its manifest entry the "
        "code it `replaces` and calling `replay`: the instances that replay become its test."
        if has("replay")
        else ""
    )
    body = (
        role
        + "\n# Inputs\n"
        + "".join(inputs)
        + "\n# Tools\n"
        + (tools + "\n" if tools else "")
        + "\n"
        + _record_text()
        + "\n"
        + procedure
        + "\n"
        + _WHAT_TO_WRITE
        + "\n"
        + _WRITE_RULES
        + "\n"
        + _GATE
        + "\n"
        + _REPAIR
        + "\n"
        + _limits_text(limits, helper_time=helpers)
        + "\n"
        "# Done\n"
        "You are done when you have built the best library items you can from this batch, with the proposals as "
        "material and the episodes as evidence, and every item "
        "you wrote has passed the gate or is kept as a draft with its failure. Then call `finish(summary)`. "
        "Writing nothing is a valid decision when your summary says why.\n"
    )
    return _sandbox_first("writer") + _fill(
        body,
        {
            "manifest_extra": manifest_extra,
            "item_extra": item_extra,
            "manifest_extra_docs": manifest_docs,
            "replay_rule": replay_rule,
            "mutation_min_pct": _pct(mutation_min),
            "sample_k": sample_k,
            "lint_min": lint_min,
            "pytest_env": _env(pytest_env),
            "semantic_types": describe_semantic_types(),
        },
    )


def _record_text() -> str:
    """The pass record's facts (:data:`.messaging.RECORD_TEXT`), stated in every brief of an agent on a record."""
    from .messaging import RECORD_TEXT

    return RECORD_TEXT


def curate_brief(
    *,
    limits: Limits,
    pytest_env: Mapping[str, str],
    overlaps_path: str,
    suspects_path: str,
    trigger_path: str,
    tools: str = "",
    in_code: bool | None = None,
    flags: frozenset[str] = frozenset(),
) -> str:
    """The CURATE brief with its numbers, input paths and Tools section filled in; *in_code* (None:
    ``UNIFY_TOOLS_IN_CODE``) gives the brief for tools called from code, after the shared opening for the
    configuration *flags*."""
    if in_code if in_code is not None else _in_code():
        return _opened(
            "curator",
            flags,
            limits,
            _curate_brief_in_code(
                limits=limits,
                pytest_env=pytest_env,
                overlaps_path=overlaps_path,
                suspects_path=suspects_path,
                trigger_path=trigger_path,
            ),
        )
    return _sandbox_first("curator") + _fill(
        _CURATE,
        {
            "tools": (tools + "\n" if tools else "") + "\n" + _record_text(),
            "library_bullet": _CURATE_LIBRARY_BULLET if _persistent() else _CURATE_LIBRARY_BULLET_FRESH,
            "limits": _limits_text(limits, helper_time=False),
            "pytest_env": _env(pytest_env),
            "overlaps_path": overlaps_path,
            "suspects_path": suspects_path,
            "trigger_path": trigger_path,
        },
    )


def _persistent() -> bool:
    from unify.common.box_interpreter import persistent

    return persistent()


def _sandbox_first(agent: str) -> str:
    """With tools in code off, a memory agent gets no shared opening: its brief starts with the shared sandbox
    paragraph instead (:func:`unify.common.tool_registry.sandbox_paragraph`: its interpreter keeps its state, when it
    restarts, and that ``memory`` is bound), then a blank line. Under the ``UNIFY_CELL_STATE=fresh`` opt-out each
    cell is a fresh script, whose brief and tool text already say so: nothing is added, so the opt-out's prompt is the
    earlier one byte for byte."""
    from unify.common import tool_registry

    if not _persistent():
        return ""
    return tool_registry.sandbox_paragraph(agent, ("memory",), True) + "\n\n"


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
        compact_keep_prefix=bool(compact and _cc.keep_prefix_enabled()),
        stage_files=fork_worker.QUOTA_FILES,
        stage_bytes=fork_worker.QUOTA_BYTES,
    )


def tool_flags(
    staging: bool = False,
    pool: bool = False,
    helper: bool = False,
    similar: bool = False,
    edit: bool = False,
) -> frozenset[str]:
    """The tool spec configuration flags of a pass (``fork``: staging exists; ``pool``: the candidate pool is on;
    ``helper``: a helper lane exists, so ``delegate`` is offered; ``similar``: the similar-code index exists;
    ``edit``: the ``write`` and ``edit`` file tools are on).
    """
    return frozenset(
        ({"fork"} if staging else set())
        | ({"pool"} if pool else set())
        | ({"helper"} if helper else set())
        | ({"similar"} if similar else set())
        | ({"edit"} if edit else set()),
    )


def write_brief_now(
    staging: bool = False,
    pool: bool = False,
    cell_s: float | None = None,
    helper: bool = False,
    similar: bool = False,
    compact: bool = False,
    finish_capped: bool = False,
    edit_tools: bool = False,
) -> str:
    """The WRITE brief from the constants that enforce each number and the tools registered for the role."""
    from . import code_lint, qa
    from .gate import _PYTEST_ENV
    from .tool_spec import tools_section

    flags = tool_flags(staging, pool, helper, similar, edit_tools)
    return write_brief(
        limits=limits_now(cell_s, compact=compact, finish_capped=finish_capped),
        mutation_min=qa.MUTATION_MIN,
        sample_k=qa.SAMPLE_K,
        lint_min=code_lint.LINT_MIN,
        pytest_env=_PYTEST_ENV,
        staging=staging,
        pool=pool,
        tools=tools_section("write", flags),
        tool_names=_tool_names("write", flags),
        pool_kinds=_pool_kinds() if pool else "",
        flags=flags,
    )


def curate_brief_now(
    cell_s: float | None = None,
    compact: bool = False,
    finish_capped: bool = False,
    edit_tools: bool = False,
) -> str:
    """The CURATE brief from the enforcing constants, the curate input paths and the curate tools (with the
    ``write`` and ``edit`` file tools when *edit_tools*)."""
    from . import curate
    from .gate import _PYTEST_ENV
    from .tool_spec import tools_section

    flags = tool_flags(edit=edit_tools)
    return curate_brief(
        limits=limits_now(cell_s, compact=compact, finish_capped=finish_capped),
        pytest_env=_PYTEST_ENV,
        overlaps_path=curate.OVERLAPS_INPUT,
        suspects_path=curate.SUSPECTS_INPUT,
        trigger_path=curate.TRIGGER_INPUT,
        tools=tools_section("curate", flags),
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


def candidate_contract() -> str:
    """The drafting contract a fork or helper reads: :data:`CANDIDATE_CONTRACT`, or its in-code form."""
    return CANDIDATE_CONTRACT_IN_CODE if _in_code() else CANDIDATE_CONTRACT


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

_CURATE_GATE_IN_CODE = (
    "# The gate\n"
    "After `tools.finish(summary)` a deterministic gate checks the candidate (G1 to G6 below); an item that breaks "
    "a rule is refused with the items that import it, and the rest can land. "
    + _check_names_in_code(_G5_CURATE, "enough small mutations")
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
    staging: bool,
    pool: bool,
    tool_names: frozenset[str],
) -> str:
    """The WRITE brief when tools are called from code: the same role, inputs, rules and done, with each tool named
    as the call the writer makes and no Tools section (``tools.list()`` and ``tools.help(name)`` describe them)."""
    has = tool_names.__contains__
    helpers = has("delegate")
    proposals_from = (
        (["candidates drafted from each episode when it ended (/inputs/staging/)"] if staging else [])
        + (["files the helpers you launch stage (/inputs/helpers/)"] if helpers else [])
        + (["open entries earlier passes kept in the pool (/inputs/pool/)"] if pool else [])
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
    if staging:
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
    if pool:
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

    look = ["staged candidates and their cases"] if staging else []
    if helpers:
        look.append("anything helpers staged")
    if pool:
        look.append(
            "open pool entries that cite this batch's episodes"
            + (" (`tools.pool` can filter by episode)" if has("pool") else " (their `pool_entries`)"),
        )
    look.append("open drafts")
    decide = "3. Analyse each proposal critically and build from it, as the Role says; proposals are untrusted input."
    if staging or pool or helpers:
        decide += " Record in each item's `adopts` the drafts it drew on and how."
    if pool:
        decide += (
            " Record what you decided about pool entries in the manifest's `pool` block"
            + (" (`tools.pool` with a decision writes it for you)" if has("pool") else "")
            + "."
        )
    if staging:
        decide += " " + SUMMARY_NOT_EVIDENCE
    similar_line = (
        " Where code looks reusable, `tools.similar` finds other recorded episodes with matching code."
        if has("similar")
        else ""
    )
    similar_tail = ", and `tools.similar` finds others with matching code" if has("similar") else ""
    if staging:
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
    else:
        steps = (
            "1. Look over the batch with `tools.overview()`, then over the proposals: "
            + _join(look)
            + "."
            + similar_line
            + "\n"
            "2. Read the episodes behind what may be worth keeping, starting from each one's `end_parts`, which show "
            "where it ended and so where an outcome, if any, was shown. At any point you can open any recorded "
            "episode, in this batch or not: `tools.overview` and `tools.read_episode` take any recorded id"
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
    if staging or pool or helpers:
        item_extra = ', "adopts": [...]'
        forms = (
            (["`staging:<episode>/candidates/<name>.py`"] if staging else [])
            + (["`helper:<helper name>/candidates/<name>.py`"] if helpers else [])
            + (["`pool:<id>`"] if pool else [])
        )
        manifest_docs += (
            '  - An item\'s `adopts` lists the drafts it drew on, each as `{"ref": <ref>, "how": <how>}`. The ref is '
            + _join(forms)
            + ". How is `as_is` (used unchanged), `edited`, `merged` (with other drafts or a library item), "
            "`generalised` (parametrised or generalised), `rewritten`, or `informed` (it only informed new work). "
            "Leave it out when the item drew on none. The manifest itself has no `adopts`.\n"
        )
    if pool:
        manifest_extra += '"pool": {...}, '
        manifest_docs += (
            "  - `pool` records decisions on pool entries"
            + (
                " (`tools.pool` with op='decide' writes it; `tools.help(\"pool\")` lists the decisions)"
                if has("pool")
                else ""
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
    return _fill(
        body,
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


_CURATE_IN_CODE = (
    "# Role\n"
    "You are the library curator. The library, `memory`, is a Python package of tested functions and linked "
    "notes that a working model imports. You keep it small, correct and easy to navigate: merge overlapping "
    "items, generalise variants, repair or roll back items retired by their outcomes, retire wrong or unused ones, "
    "fix links and "
    "regroup packages. You work in a sandbox with no network and no credentials, on the library alone.\n"
    "\n"
    "# Inputs\n"
    "- <<trigger_path>>: why this pass runs (its reasons, and the library commit it starts from). For: where to "
    "begin.\n"
    "- <<overlaps_path>>: the overlap candidates the harness found: functions with matching argument "
    "shapes and overlapping input fixtures, functions whose bodies share a common generalisation (given with it), and "
    "notes that use the same functions. For: what to merge or generalise.\n"
    "- /inputs/curate/same_shape.json: groups of functions whose bodies have the same structure. For: more "
    "merge and generalisation candidates.\n"
    "- <<suspects_path>>: the items retired by their outcomes, each with why it was retired, the episodes "
    "involved and, when an "
    "earlier version has a clean use record, that version as the proposed rollback. An input refusal (a "
    "`MemoryInputError` raised on an input the item was not built for) is counted and reported, never a failure. "
    "For: what to repair or roll back.\n"
    "- /inputs/curate/rollback/<n>/: the files (module and tests) of each proposed rollback version; "
    "/inputs/curate/rollback/index.json names each one's item, directory and commit. For: rolling back (the box has "
    "no git history).\n"
    "- /inputs/curate/use.json (use per item and library commit), /inputs/curate/history.json (recent changes) and "
    "/inputs/curate/aliases.json (old names and whether this pass may remove them). For: judging what is used, what "
    "changed and what may go.\n"
    "- /inputs/episodes/<episode>.json: the recorded episodes the retired items were used in. For: the evidence "
    "behind a retirement; `tools.read_episode` and `tools.overview` show them part by part.\n"
    "- /memory: a writable copy of the library. In every cell `memory` is already bound (no import needed): "
    "browse it with `memory.items` and open an item, with its records and recent changes, with `memory.get`. "
    "Names you bound earlier keep the code they were bound to; after you edit a library file, import it again. "
    "There is no git history in the box. "
    "/memory/INDEX.md and /memory/memory/__init__.py are generated by the harness and need no reading.\n"
    "- /inputs/gate/: the full result of each `tools.check` and each repair round of this pass, as it happens.\n"
    "\n"
    "<<record>>"
    "\n"
    "# What to do\n"
    "- Merge overlapping items into one, keeping each old name as an alias: `old_name = new_name` in its "
    "module, listed in the manifest's `aliases`.\n"
    "- Generalise variants into one parametric function.\n"
    "- Repair an item retired by its outcomes, or roll it back to the proposed earlier version from "
    "/inputs/curate/rollback/.\n"
    "- Retire an item that is wrong, or unused across many episodes, together with its tests, with a one-line "
    "reason in `retired`.\n"
    "- Fix broken links and add missing ones between notes and functions.\n"
    "- Regroup packages when that makes the index clearer, keeping each moved function's old id as an alias.\n"
    "- Keep each item's `When to use` section and each folder's README.md true to what they hold now: a merged or "
    "generalised item's section covers the requests of the items it replaces. "
    + WHEN_TO_USE_TEXT[2:]
    + "- Leave alone what is fine, and say why in `kept`: changing nothing is a valid result when no candidate is "
    "really one job.\n"
    "- Write the manifest, /memory/.pass/manifest.json (never committed):\n"
    '  {"items": [{"item": "<id>", "kind": "function" or "note", "source_episodes": [...], "tests": [...], '
    '"covers": [...]}], "deleted": ["<retired id>"], '
    '"retired": {"<retired id>": "<one-line reason>"}, "deleted_tests": ["<test file removed with them>"], '
    '"aliases": {"<old id>": "<new id>"}, "renames": {"<old id>": "<new id>"}, '
    '"kept": {"<item id, or INDEX.md>": "<why it stays as it is>"}, '
    '"tests_changed": {"<test>": "<reason>"}, "why": "<one line>", "summary": "<what changed>"}\n'
    "  `renames` moves an item to a new id without keeping the old name (its history follows it); `aliases` "
    "keeps the old name forwarding to the new one.\n"
    "  `items` lists every item you added or changed, with the tests and fixtures of the items it replaces. "
    + _RUN_TESTS
    + "\n"
    "\n"
    "# Rules\n"
    "1. Every change keeps the library's existing tests green, or retires a test with its reason in "
    "`tests_changed`. The tests of a merged item stay and run through its alias.\n"
    "2. A change in what a function returns on its recorded inputs needs a test that fails on the parent and "
    "passes on yours; a repair always changes behaviour.\n"
    "3. The fixtures a retired function's tests called it on stay used by a remaining function's tests.\n"
    "4. New tests check exact expected values, " + _TRUST + " " + _DETERMINISM + "\n"
    "5. Act on records that span at least two episodes: one episode never decides.\n"
    "6. Fixture files stay where they are: a test may read a fixture from any package's `tests/data/`.\n"
    "7. A reason in your inputs is settled when this pass changes the item it names or keeps it in `kept`; "
    "otherwise a later curate pass raises it again.\n"
    "8. Never store credentials or session tokens. " + _HONESTY + "\n"
    "\n" + _CURATE_GATE_IN_CODE + "\n" + _REPAIR_IN_CODE + "\n" + "<<limits>>" + "\n"
    "# Done\n"
    "You are done when each overlap candidate and each item retired by its outcomes in your inputs has been "
    "merged, generalised, "
    "repaired, rolled back or retired, or kept as it is with the reason in `kept`, and your changes have "
    "passed the gate or are kept as a draft with their failure. Then call `tools.finish(summary)`.\n"
)


def _curate_brief_in_code(
    *,
    limits: Limits,
    pytest_env: Mapping[str, str],
    overlaps_path: str,
    suspects_path: str,
    trigger_path: str,
) -> str:
    """The CURATE brief when tools are called from code."""
    return _fill(
        _CURATE_IN_CODE,
        {
            "record": record_text(),
            "limits": _limits_text_in_code(limits, helper_time=False),
            "pytest_env": _env(pytest_env),
            "overlaps_path": overlaps_path,
            "suspects_path": suspects_path,
            "trigger_path": trigger_path,
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
CURATE_ONLY = ("overlap", "roll it back", "alias", "retire an item")
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

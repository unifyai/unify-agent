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

GUIDE_LIBRARY = (
    "### Memory Library\n"
    "\n"
    "A Python library of tested functions and linked notes, built from earlier work, is importable "
    "in your cells as `memory`. Before you write code for a step, look in the index for an item that does it. "
    "Using the library is optional: use an item when it fits, and otherwise work as usual.\n"
    "\n"
    "Finding items:\n"
    "- The index at the end of this prompt lists every function (its id, signature, one-line summary and "
    "status) and every note (its path, description and the functions it uses). When the index is long, it "
    "shows package headings only.\n"
    '- `print(memory.index("<package>"))` prints one package\'s full section, and `print(memory.index())` the '
    "whole index.\n"
    "- `print(memory.show(\"<id>\"))` prints one item in full: a function's source and docstring, or a note's "
    "text, then its links, status, test and use records, and recent changes.\n"
    "- `memory.find(value)` lists the functions built on inputs shaped like a structured value you hold (a "
    "dict, a list of records, JSON text, or a CSV file with a header). It cannot match plain text: for text, "
    "use the index or grep.\n" + FILES_LINE + "\n"
    "Using items:\n"
    "- Import a function by its id: `memory.text.dates:parse_date` is "
    "`from memory.text.dates import parse_date`.\n"
    "- A function raises `MemoryInputError` on an input it was not built for. Then do that step with your own "
    "code.\n"
    "- `experimental` items are candidates: they passed the library's checks when they were added and have "
    "had little use. `stable` items passed them and have been used without errors. Items found to fail are "
    "left out of the index, and `memory.show` says why.\n"
    "\n"
    "Write your own code and files in your session as usual.\n"
)

#: Where the guide states each part of a role prompt (the guide is prose, so the parts are phrases, not headings).
GUIDE_PARTS: tuple[tuple[str, str], ...] = (
    ("role", "is importable in your cells as `memory`"),
    ("inputs", "The index at the end of this prompt lists"),
    ("outputs", "Write your own code and files in your session as usual"),
    ("rules", "Then do that step with your own code"),
    ("done", "Using the library is optional"),
)


def guide_library_mode(extra_lines: tuple[str, ...] = ()) -> str:
    """The guide, with *extra_lines* (optional pull helpers, constant per run) before :data:`FILES_LINE`."""
    if not extra_lines:
        return GUIDE_LIBRARY
    return GUIDE_LIBRARY.replace(FILES_LINE, "".join(extra_lines) + FILES_LINE, 1)


#: The guide when the actor's prompt points to the index instead of carrying it
#: (``UNIFY_MEMORY_LIBRARY_ACTOR_INDEX=pointer``). Constant bytes, naming nothing the library holds, so the cached
#: prefix never changes as the library grows; the index is reached in a cell.
GUIDE_POINTER = (
    "### Memory Library\n"
    "\n"
    "A Python library of tested functions and linked notes, built from earlier work, is importable "
    "in your cells as `memory`. It may be empty. Before you write code for a step, look in the library for an "
    "item that does it. Using the library is optional: use an item when it fits, and otherwise work as usual.\n"
    "\n"
    "Finding items:\n"
    "- `print(memory.index())` prints the library index: every function (its id, signature, one-line summary, "
    "status and effect: whether it reads, writes or is unknown) and every note (its path, description and the functions it uses). "
    '`print(memory.index("<package>"))` prints one package\'s section.\n'
    "- `print(memory.show(\"<id>\"))` prints one item in full: a function's source and docstring, or a note's "
    "text, then its links, status, test and use records, and recent changes.\n"
    "- `memory.find(value)` lists the functions built on inputs shaped like a structured value you hold (a "
    "dict, a list of records, JSON text, or a CSV file with a header). It cannot match plain text: for text, "
    "use the index or grep.\n" + FILES_LINE + "\n"
    "Using items:\n"
    "- Import a function by its id: `memory.text.dates:parse_date` is "
    "`from memory.text.dates import parse_date`.\n"
    "- A function raises `MemoryInputError` on an input it was not built for. Then do that step with your own "
    "code.\n"
    "- `experimental` items are candidates: they passed the library's checks when they were added and have "
    "had little use. `stable` items passed them and have been used without errors. Items found to fail are "
    "left out of the index, and `memory.show` says why.\n"
    "\n"
    "Write your own code and files in your session as usual.\n"
)


# --- the writer's briefs: store pass (WRITE) and curate pass (CURATE) ---------------------------------------------
#
# Each brief is built from stable text plus the configuration of the run (which optional inputs exist, which tools
# are registered for the role), so it is byte-stable for a whole run. Nothing that changes between passes (the
# library index, the pool, use counts, drafts, why a curate pass runs) is inlined: those live in files the brief
# points to. The Tools section is rendered from the tool spec registry (:func:`.tool_spec.tools_section`), so it
# names exactly the tools the role gets.

#: The input forms of a function's first parameter, in ``manifest.INPUT_KINDS`` order, described without any
#: environment's vocabulary.
INPUT_FORMS_LIBRARY: dict[str, str] = {
    "path": "a path to a work-tree file",
    "text": "the text of a file or of an output",
    "bytes": "the raw bytes of a file",
    "observation": "an observation value, as recorded",
    "env": "the object through which the request's tools are called",
}

#: What a drafting agent (the episode-end fork, a helper) is told about its drafts: draft what it saw; fitting the
#: drafts to the shared library and generalising them is the writer's work.
CANDIDATE_CONTRACT = (
    "Draft from what you saw: code that worked, the cases it ran on (values as they were recorded), and notes. "
    "Where you can, write a reusable function or lesson in general form, with the values that vary as "
    "parameters. Where your work is specific to this case, still write it down as it is: the library writer, who "
    "alone commits to the shared library, builds from these drafts and may generalise them with other episodes."
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

#: Row keys the pass adds when an optional input exists (described only then).
STAGING_ROW_KEY = (
    "staging",
    "dict: `status`, `source`, `files`, `refused`, `path` and `candidates` (list of `{name, file, cases}`): what "
    "was staged for the episode under /inputs/staging/<episode>/",
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

#: What each pool entry kind holds (only the kinds :data:`.pool.KINDS` defines are named).
POOL_KIND_TEXT: dict[str, str] = {
    "candidate": "a function with its cases",
    "lesson": "notes",
    "cluster": "code that recurs across episodes",
    "draft": "work the gate refused, with its result",
    "deferred": "episodes a pass kept open to revisit",
    "unread": "episodes no pass has read",
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

#: What each check the gate names in its refusals and results means (the same names: G1 to G6).
_CHECK_NAMES = (
    "Its results and refusals name the check: G1, layout and provenance (only `memory/` and `notes/` change and "
    "never the generated files, the manifest is well formed, every link names an item that exists, and every "
    "fixture came through `fixture` unchanged); G2, evidence (each function's covers are recorded evidence of "
    "their type, a function given a same-shaped input from another episode returns a value or raises "
    "`MemoryInputError`, and the literal rule); G3, tests (plain pytest, an exact assertion against a trusted "
    "value on a recorded fixture for each changed function, no test that recomputes the function, the tests pass "
    "on the candidate and on the drawn recorded inputs and catch enough small mutations, a behaviour change has a "
    "test that fails on the parent, and no test is lost or weakened without a reason); G4, the index (it never "
    "refuses: it notes when the index has outgrown the view the working model is shown); G5, items (<<g5>>); "
    "G6, safety (no links, executables, submodules, git, pytest or interpreter configuration files, bytecode or "
    "key-shaped strings).\n"
)

_G5_WRITE = "a store pass never ends with fewer items, and every recorded input a function covered stays covered"
_G5_CURATE = (
    "every recorded input a deleted function covered stays covered by a remaining item, and each alias and "
    "retirement names an item"
)

_GATE = (
    "# The gate\n"
    "After `finish` a deterministic gate checks the candidate: the layout and the links; that every fixture "
    "came through `fixture` unchanged; each function's covers; its tests on the candidate, and on the parent "
    "where behaviour changed; the appended recorded inputs; that its tests catch at least <<mutation_min_pct>>% "
    "of small mutations of each changed function; that a function given a same-shaped input from another "
    "episode returns a value or raises `MemoryInputError`; the literal rule; and that no test is lost or "
    "weakened without a reason. An item that breaks a rule is refused together with the items that import it, "
    "and the rest can land; a breach of a pass-wide rule (the layout, lost tests, fewer items) refuses the "
    "round. " + _CHECK_NAMES.replace("<<g5>>", _G5_WRITE)
)

_CURATE_GATE = (
    "# The gate\n"
    "After `finish` a deterministic gate checks the candidate: the layout and the links, the aliases and "
    "retirements, every test on the candidate (and on the parent where behaviour changed), and that every "
    "recorded input a deleted function covered stays covered. An item that breaks a rule is refused together with "
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

_LIBRARY_BULLET = (
    "- /memory: a writable copy of the library. There is no git history in the box; an item's recent changes "
    'are in `memory.show("<id>")`. /memory/INDEX.md is the library index, generated by the harness like '
    "/memory/links.json and /memory/memory/__init__.py. In a cell, `import memory` gives `memory.index()`, "
    '`memory.index("<package>")`, `memory.show("<id>")` and `memory.find(value)`, as the working model has them. '
    "For: knowing what exists before you add anything.\n"
)

_WHAT_TO_WRITE = (
    "# What to write\n"
    "- Functions go in `memory/<package>/<module>.py`, in packages you choose by what the code does. Each "
    "package has an `__init__.py` whose docstring is one paragraph on what lives there. A function's id is "
    "`memory.<package>.<module>:<name>`; a name starting with `_` is private and is not an item.\n"
    "- A function's docstring has a one-line summary; `Args:`, `Returns:` and `Raises:` sections "
    "(`MemoryInputError` on an input it was not built for); an `Example:` section with a doctest over a "
    "fixture; and, optionally, a `Notes:` line naming note paths.\n"
    "- Notes go in `notes/<topic>/<slug>.md`: free-form markdown of any kind (a fact, a pitfall, a how-to, or "
    "how to combine functions), starting with front matter: `title`, `description` (one line, shown in the "
    "index) and `uses` (the ids of the functions it uses). A link is written only in a note's `uses` or a "
    "function's `Notes:` line, and must name an item that exists. A note states its rule, procedure, fact or "
    "pitfall in general terms where it can; a rule for one kind of request is worth keeping when that kind of "
    "request comes back. A new item starts as `experimental`; later passes and the curate pass revise, generalise "
    "or remove it as evidence arrives, so storing a rule that worked is not a final commitment. A function's "
    "linked note says when to use it, its preconditions and its known failures.\n"
    "- Tests go in `memory/<package>/tests/test_<module>.py`: plain pytest, importing only the library and the "
    "standard library, with fixtures under `memory/<package>/tests/data/`. `tests/` takes no `__init__.py`: "
    "pytest imports each test file by its path. " + _RUN_TESTS + "\n"
    "- The manifest is /memory/.pass/manifest.json (never committed):\n"
    '  {"items": [{"item": "<id>", "kind": "function" or "note", "input": "<form>", '
    '"source_episodes": [...], "tests": [...], "covers": [...]<<item_extra>>}], "support": [], "tests_changed": {}, '
    '"renames": {}, <<manifest_extra>>"why": "<one line>", "summary": "<what changed>"}\n'
    "  - A note's entry takes `item`, `kind` and `source_episodes`: "
    '`{"item": "notes/<topic>/<slug>.md", "kind": "note", "source_episodes": [...]}`; `input` and `covers` '
    "are for functions.\n"
    "  - `input` is the form a function's first parameter takes, one of: <<input_forms>>.\n"
    "  - `covers` are the recorded evidence a function's tests check: `[episode, action index]` for a recorded "
    'action; `{"episode": e, "type": "cell", "cell": i}` or `{"episode": e, "type": "diff"}` for '
    'the working model\'s own code; `{"episode": e, "type": "episode", "runner": "worktree" | "tool" '
    '| "dialogue", ...}` for a whole procedure.\n'
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
    "values.<<replay_rule>>\n"
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
    "with fewer items or tests. If an existing test must change, name it in `tests_changed` with the reason.\n"
    "5. Write the tests first, against trusted expected values, "
    + _TRUST
    + " Each function needs at least one test asserting that its result on a recorded fixture equals an exact "
    "trusted value, and one test that it refuses an input of another shape. A procedure's test checks the "
    "effect the environment later confirmed, not only that it ran. Never compute an expected value with the "
    "function's own expression. "
    + _DETERMINISM
    + " Run each new test and see it fail before the code exists, then pass.\n"
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

_CURATE = (
    "# Role\n"
    "You are the library curator. The library, `memory`, is a Python package of tested functions and linked "
    "notes that a working model imports. You keep it small, correct and easy to navigate: merge overlapping "
    "items, generalise variants, repair or roll back suspect items, retire wrong or unused ones, fix links and "
    "regroup packages. You work in a sandbox with no network and no credentials, on the library alone.\n"
    "\n"
    "# Inputs\n"
    "- <<trigger_path>>: why this pass runs (its reasons, and the library commit it starts from). For: where to "
    "begin.\n"
    "- <<overlaps_path>>: the overlap candidates the harness found: functions with matching recorded input "
    "shapes and overlapping covers, functions whose bodies share a common generalisation (given with it), and "
    "notes that use the same functions. For: what to merge or generalise.\n"
    "- /inputs/curate/same_shape.json: groups of functions whose bodies have the same structure. For: more "
    "merge and generalisation candidates.\n"
    "- <<suspects_path>>: the suspect items, each with why it is suspect (errors or negative outcomes after use, "
    "in at least two episodes, or a suspect dependency), the episodes involved, the first version whose behaviour "
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
    "- /inputs/episodes/<episode>.json: the recorded episodes the suspect items were used in. For: the evidence "
    "behind a suspect; `read_episode` and `overview` show them part by part.\n"
    "- /memory: a writable copy of the library. There is no git history in the box. `import memory` works in your "
    'cells: `memory.index()`, `memory.find(value)`, and `memory.show("<id>")`, which gives an item\'s status, its '
    "test record, its use record per library commit and its recent changes. /memory/INDEX.md is the library index, "
    "generated by the harness.\n"
    "- /inputs/gate/: the full result of each `check` and each repair round of this pass, as it happens.\n"
    "\n"
    "# Tools\n"
    "<<tools>>"
    "\n"
    "# What to do\n"
    "- Merge overlapping items into one, keeping each old name as an alias: `old_name = new_name` in its "
    "module, listed in the manifest's `aliases`.\n"
    "- Generalise variants into one parametric function.\n"
    "- Repair a suspect item, or roll it back to the proposed earlier version from /inputs/curate/rollback/.\n"
    "- Retire an item that is wrong, or unused across many episodes, together with its tests, with a one-line "
    "reason in `retired`.\n"
    "- Fix broken links and add missing ones between notes and functions.\n"
    "- Regroup packages when that makes the index clearer, keeping each moved function's old id as an alias.\n"
    "- Leave alone what is fine, and say why in `kept`: changing nothing is a valid result when no candidate is "
    "really one job.\n"
    "- Write the manifest, /memory/.pass/manifest.json (never committed):\n"
    '  {"items": [{"item": "<id>", "kind": "function" or "note", "input": "<form>", '
    '"source_episodes": [...], "tests": [...], "covers": [...]}], "deleted": ["<retired id>"], '
    '"retired": {"<retired id>": "<one-line reason>"}, "deleted_tests": ["<test file removed with them>"], '
    '"aliases": {"<old id>": "<new id>"}, "renames": {"<old id>": "<new id>"}, '
    '"kept": {"<item id, or INDEX.md>": "<why it stays as it is>"}, '
    '"tests_changed": {"<test>": "<reason>"}, "why": "<one line>", "summary": "<what changed>"}\n'
    "  `renames` moves an item to a new id without keeping the old name (its history follows it); `aliases` "
    "keeps the old name forwarding to the new one.\n"
    "  `items` lists every item you added or changed, with the covers of the items it replaces. `kept` names each "
    "item a reason in your inputs points at that you leave as it is, with why. "
    + _RUN_TESTS
    + "\n"
    "\n"
    "# Rules\n"
    "1. Every change keeps the library's existing tests green, or retires a test with its reason in "
    "`tests_changed`. The tests of a merged item stay and run through its alias.\n"
    "2. A change in what a function returns on its recorded inputs needs a test that fails on the parent and "
    "passes on yours; a repair always changes behaviour.\n"
    "3. Every recorded input a retired function covered stays covered by a remaining function.\n"
    "4. New tests check exact expected values, " + _TRUST + " " + _DETERMINISM + "\n"
    "5. Act on records that span at least two episodes: one episode never decides.\n"
    "6. Fixture files stay where they are: a test may read a fixture from any package's `tests/data/`.\n"
    "7. A reason in your inputs is dealt with when this pass changes an item it names, or keeps that item with a "
    "reason in `kept`; otherwise it is raised again by a later curate pass.\n"
    "8. Never store credentials or session tokens. " + _HONESTY + "\n"
    "\n" + _CURATE_GATE + "\n" + _REPAIR + "\n" + "<<limits>>" + "\n"
    "# Done\n"
    "You are done when each overlap candidate and suspect item in your inputs has been merged, generalised, "
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
    #: the conversation size in characters that moves old tool results out (None: compaction off), and the
    #: messages at the end that are never moved
    compact_at_chars: int | None = None
    compact_keep_last: int = 0
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
            ),
            "finish_cap": (
                f" While /memory/.pass/manifest.json is missing or does not parse, `finish` is refused with the reason "
                f"up to {limits.finish_refusals} times in a pass; after that, `finish` ends the pass and the final "
                "check refuses the manifest by name."
                if limits.finish_refusals is not None
                else ""
            ),
            "compaction": (
                f" When the conversation's text grows past {limits.compact_at_chars} characters, the oldest tool "
                "results (never the system message, the request, your own messages or the last "
                f"{limits.compact_keep_last} messages) are moved to files under /outputs/compacted/, and each is "
                'replaced by a note naming its file; `read(path="/outputs/compacted/<file>")` shows it again.'
                if limits.compact_at_chars is not None
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
        + ((STAGING_ROW_KEY,) if staging else ())
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
    input_forms: Mapping[str, str] = INPUT_FORMS_LIBRARY,
    staging: bool = False,
    pool: bool = False,
    tools: str = "",
    tool_names: frozenset[str] = frozenset(),
    pool_kinds: str = "",
) -> str:
    """The WRITE brief. *staging* (the episode-end fork is on) and *pool* (the candidate pool is on) add their input
    bullets, row keys and manifest fields; *tools* is the rendered Tools section and *tool_names* the tools the role
    gets, so a sentence that relies on an optional tool (``pool``, ``similar``, ``delegate``, ``replay``) appears
    only when that tool exists. Every input is constant for a run, so the brief is byte-stable for the run.
    """
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
            "- /inputs/staging/<episode>/: what was drafted from that episode when it ended: "
            "`candidates/<name>.py` (a function), `cases/<name>.json` (its example inputs and outputs) and "
            "`notes.md` (lessons). The episode's batch-map row lists them under `staging`. For: drafts to build from. "
            "The agent that drafted them saw only that episode, so they are likely specific to it. The harness has "
            "run none of it, so treat it as untrusted input."
            + _stage_quota_text(limits)
            + "\n",
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
    inputs.append(_LIBRARY_BULLET)
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
    procedure = (
        "# How to work\n"
        "This is a procedure, not a quota: read as much or as little as the material needs.\n"
        "1. Look over the batch with `overview()`, then over the proposals: "
        + _join(look)
        + "."
        + (
            " Where code looks reusable, `similar` finds other recorded episodes with matching code."
            if has("similar")
            else ""
        )
        + "\n"
        "2. Read the episodes behind what may be worth keeping, starting from each one's `end_parts`, which show "
        "where it ended and so where an outcome, if any, was shown. At any point in the pass, you can open any "
        "recorded episode, in this batch or not and from earlier passes too, whenever you need to investigate or "
        "dig deeper: `overview` and `read_episode` take any recorded id"
        + (", and `similar` finds others with matching code" if has("similar") else "")
        + ".\n"
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
    return _fill(
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
            "input_forms": ", ".join(f"`{k}` ({v})" for k, v in input_forms.items()),
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
) -> str:
    """The CURATE brief with its numbers, input paths and Tools section filled in."""
    return _fill(
        _CURATE,
        {
            "tools": (tools + "\n" if tools else "") + "\n" + _record_text(),
            "limits": _limits_text(limits, helper_time=False),
            "pytest_env": _env(pytest_env),
            "overlaps_path": overlaps_path,
            "suspects_path": suspects_path,
            "trigger_path": trigger_path,
        },
    )


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
        compact_at_chars=writer_pass.COMPACT_AT_CHARS if compact else None,
        compact_keep_last=writer_pass.COMPACT_KEEP_LAST,
        stage_files=fork_worker.QUOTA_FILES,
        stage_bytes=fork_worker.QUOTA_BYTES,
    )


def tool_flags(
    staging: bool = False,
    pool: bool = False,
    helper: bool = False,
    similar: bool = False,
) -> frozenset[str]:
    """The tool spec configuration flags of a pass (``fork``: staging exists; ``pool``: the candidate pool is on;
    ``helper``: a helper lane exists, so ``delegate`` is offered; ``similar``: the similar-code index exists).
    """
    return frozenset(
        ({"fork"} if staging else set())
        | ({"pool"} if pool else set())
        | ({"helper"} if helper else set())
        | ({"similar"} if similar else set()),
    )


def write_brief_now(
    staging: bool = False,
    pool: bool = False,
    cell_s: float | None = None,
    helper: bool = False,
    similar: bool = False,
    compact: bool = False,
    finish_capped: bool = False,
) -> str:
    """The WRITE brief from the constants that enforce each number and the tools registered for the role."""
    from . import code_lint, qa
    from .gate import _PYTEST_ENV
    from .tool_spec import tools_section

    flags = tool_flags(staging, pool, helper, similar)
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
    )


def curate_brief_now(
    cell_s: float | None = None,
    compact: bool = False,
    finish_capped: bool = False,
) -> str:
    """The CURATE brief from the enforcing constants, the curate input paths and the curate tools."""
    from . import curate
    from .gate import _PYTEST_ENV
    from .tool_spec import tools_section

    return curate_brief(
        limits=limits_now(cell_s, compact=compact, finish_capped=finish_capped),
        pytest_env=_PYTEST_ENV,
        overlaps_path=curate.OVERLAPS_INPUT,
        suspects_path=curate.SUSPECTS_INPUT,
        trigger_path=curate.TRIGGER_INPUT,
        tools=tools_section("curate"),
    )


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

#: Instructions to check a task's examples. A function docstring's ``Example:`` section is a
#: doctest over a fixture, not a task's example, and matches none of these.
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
        "`experimental` items are candidates",
    ),
    (
        "find matches structure only",
        "how find matches",
        ("guide",),
        "It cannot match plain text",
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
        "retired covers stay covered",
        "retirement",
        ("curate",),
        "stays covered by a remaining function",
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

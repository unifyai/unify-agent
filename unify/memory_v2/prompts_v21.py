"""The memory v2.1 prompts (spec §12): the actor's GUIDE, the writer's WRITE brief and its CURATE brief.

Each text serves one role and states its role, inputs, outputs, rules and when it is done (spec P10). None carries
another role's instructions, any benchmark's vocabulary, or an instruction to check a task's examples (the 6 Oct
rule): ``tests/memory_v2/test_prompts_v21.py`` pins that, and ``test_prompts_v21_contradictions.py`` pins
:data:`RULES` (each rule stated in its owners' prompts only) and :data:`V2_RETIRED` (no v2 rule survives).

Every number a brief states is read from the constant that enforces it (:func:`write_brief_now`,
:func:`curate_brief_now`), so a brief cannot drift from its gate. The texts are the lead's to review: a change to
any of them is a reviewed change. They are the texts as the lead stored them (P7 Amendment E), with Amendment B's
corrections (``observation:<i>`` parts; ``check`` runs the items' tests and drawn inputs) and without read-only
instructions: read-only is enforced by the mount (P3 Amendment C), not by wording.
"""

from __future__ import annotations

import json
import re
from decimal import Decimal
from typing import Mapping

# --- the actor's guide (spec §12.1) -----------------------------------------------------------------------

#: The bullet the optional pull helpers (P8) are inserted before. Constant bytes.
FILES_LINE = "- The library is also plain files at the location below, so `grep` and file reads work on it.\n"

GUIDE_V21 = (
    "### Memory Library\n"
    "\n"
    "A read-only Python library of tested functions and linked notes, built from earlier work, is importable "
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

#: Where the GUIDE states each part of spec §12 (the GUIDE is prose, so the parts are phrases, not headings).
GUIDE_PARTS: tuple[tuple[str, str], ...] = (
    ("role", "is importable in your cells as `memory`"),
    ("inputs", "The index at the end of this prompt lists"),
    ("outputs", "Write your own code and files in your session as usual"),
    ("rules", "Then do that step with your own code"),
    ("done", "Using the library is optional"),
)


def guide_v21(extra_lines: tuple[str, ...] = ()) -> str:
    """The GUIDE, with *extra_lines* (P8's optional pull helpers, constant per run) before :data:`FILES_LINE`."""
    if not extra_lines:
        return GUIDE_V21
    return GUIDE_V21.replace(FILES_LINE, "".join(extra_lines) + FILES_LINE, 1)


# --- the writer's briefs (spec §12.2, §12.3) -----------------------------------------------------------------

#: The input forms of a function's first parameter, in ``manifest.INPUT_KINDS`` order, described without any
#: environment's vocabulary (v2's ``env`` text named one benchmark's object).
INPUT_FORMS_V21: dict[str, str] = {
    "path": "a path to a work-tree file",
    "text": "the text of a file or of an output",
    "bytes": "the raw bytes of a file",
    "observation": "an observation value, as recorded",
    "env": "the object through which the request's tools are called",
}

_RUN_TESTS = (
    'Run the tests as the gate does: `subprocess.run([sys.executable, "-m", "pytest", "memory", "-q", '
    '"-p", "no:cacheprovider"], env=<<pytest_env>>)`.'
)

_READ_TOOLS = (
    "- `read(path, offset)` and `grep(pattern, path, offset)` read files under /inputs, /memory and /outputs. A "
    "reply shows at most <<view_bytes>> bytes and ends with a marker naming the next offset; nothing is cut "
    "silently.\n"
)

_CHECK_FINISH = (
    "- `check(manifest)` runs the gate's static checks, the items' own tests and the recorded inputs the gate "
    "draws, on your current files, at most <<max_checks>> times per pass. It changes nothing.\n"
    "- `finish(summary)` ends the round.\n"
)

_REPAIR = (
    "# Repair\n"
    "If the gate refuses anything, its full result (every check, every failing test's output and every note) is "
    "written to /inputs/gate/result-<round>.md and you get a repair round, up to <<repair_rounds>> of them while "
    "the pass budget lasts. Fix what it refused, keep the manifest current, and call `finish` again. When the "
    "rounds or the budget run out, the items that pass land and the rest is kept as a draft for a later pass.\n"
)

_TRUST = (
    "in this order of trust: the environment's own response; a value a successful episode used; agreement "
    "across episodes. Values you make up are allowed only in refusal and shape tests."
)

_DETERMINISM = "Tests give the same result on every run: never assert on the clock, a random draw or a hash order."

_WRITE = (
    "# Role\n"
    "You are the library writer. A working model handles requests, and it imports a shared Python library, "
    "`memory`, read-only. You read a batch of its recorded episodes in full, find work that repeats or is likely "
    "to repeat, and turn that work into tested functions and linked notes in the library. You work in a sandbox "
    "with no network.\n"
    "\n"
    "# Inputs\n"
    "- /inputs/batch_map.json: one row per episode in this batch, with pointers and facts, not summaries: the "
    "episode id, the library commit it ran on, its request text, step counts, the library items it used, the "
    "tools and functions it called, each error with the call that followed it, each recorded signal with its "
    "regime, the functions the working model defined (and whether each ran cleanly), the parts you must read "
    "(`required_parts`) and the sizes of the raw records.\n"
    "- /inputs/episodes/<episode>.json: each episode's full record: its request and observations, every code "
    "cell (code, output, error, language), every action with its response, the work-tree diff and the "
    'library-use record. A large value appears as {"__blob__": "<id>", "bytes": n}; read_episode shows it '
    "whole, page by page.\n"
    "- /memory: a writable checkout of the library with its full git history (`git log` and `git show` work "
    'there). `import memory` works in your cells: `memory.index()`, `memory.show("<id>")` and '
    "`memory.find(value)`. /memory/INDEX.md, /memory/links.json and /memory/memory/__init__.py are generated by "
    "the harness.\n"
    "- /inputs/drafts/<pass>/: earlier work the gate refused: `patch.diff` and its full gate result `gate.md`. "
    "You may finish a draft.\n"
    "- /inputs/gate/previous.md: the last pass's full gate result.\n"
    "<<extra_inputs>>"
    "\n"
    "# Tools\n"
    + _READ_TOOLS
    + "- `read_episode(episode, ...)` shows named parts of an episode: `request`, `observation:<i>`, `cell:<i>`, "
    "`action:<i>` or "
    "`diff`. Only this tool counts as reading an episode.\n"
    "- `dismiss(episode, reason)` marks an episode as holding nothing to store, with a one-line reason.\n"
    "- `fixture(episode, action, dest, ...)` copies recorded bytes into a test fixture under "
    "`memory/<package>/tests/data/` and records where they came from. The gate accepts only fixtures made this "
    "way and left unchanged.\n"
    "- `execute_code(code)` runs Python as a script in a fresh process, with /memory as the working directory; "
    "only files persist between calls. Long output is shown head first, and the whole output is saved under "
    "/outputs. " + _RUN_TESTS + "\n" + _CHECK_FINISH + "\n"
    "# What to read\n"
    "An episode is covered when every part in its `required_parts` has reached you in full through "
    "`read_episode`: its request, every observation (an identical one counts once), every signal segment, "
    "every cell or diff hunk that defines a function, and every reply the working model sent and every action "
    "that changed something, with the cell it came from. "
    "`finish` is refused while any episode is neither covered nor dismissed. Read before you decide what to "
    "store; dismiss an episode only with a reason.\n"
    "\n"
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
    "function's `Notes:` line, and must name an item that exists.\n"
    "- Tests go in `memory/<package>/tests/test_<module>.py`: plain pytest, importing only the library and the "
    "standard library, with fixtures under `memory/<package>/tests/data/`.\n"
    "- The manifest is /memory/.pass/manifest.json (never committed):\n"
    '  {"items": [{"item": "<id>", "kind": "function" or "note", "input": "<form>", '
    '"source_episodes": [...], "tests": [...], "covers": [...]}], "support": [], "tests_changed": {}, '
    '"why": "<one line>", "summary": "<what changed>"}\n'
    "  - `input` is the form a function's first parameter takes, one of: <<input_forms>>.\n"
    "  - `covers` are the recorded evidence a function's tests check: `[episode, action index]` for a recorded "
    'action; `{"episode": e, "type": "cell", "cell": i}` or `{"episode": e, "type": "diff"}` for '
    'the working model\'s own code; `{"episode": e, "type": "episode", "runner": "worktree" | "tool" '
    '| "dialogue", ...}` for a whole procedure.\n'
    "  - `support` lists helper files your tests import, under a `tests/` directory. The harness records "
    "fixtures itself.\n"
    "  - `why` is one line on why the change helps. The harness writes it, the episodes and the item ids into "
    "the commit.\n"
    "\n"
    "# Rules\n"
    "1. Store anything that repeats or is likely to repeat, in any domain: readers, calculations, call "
    "sequences, and whole-job procedures built from smaller tested functions. A procedure is verified by "
    "replaying its episode: a work-tree procedure must reproduce the recorded file changes, a tool procedure the "
    "recorded calls, and a dialogue procedure the recorded action that a positive signal followed. A procedure "
    "of any other kind cannot be verified, so it stays a draft.\n"
    "2. Start from the working model's own code when it already wrote it (its cells and its work-tree diff). "
    "Generalise it: values that vary become parameters, and the function raises "
    '`MemoryInputError("<what was expected>")` on an input it was not built for. Check shapes, not observed '
    "values: a function may restrict a field's values only by declaring the field's type in the item's "
    "`field_types`, one of: <<semantic_types>>.\n"
    "3. Use every signal the records hold: a verdict the working model was shown, an error, a retry, an early "
    "end, a change in a returned count or score, or no signal at all. Prefer work from episodes with positive "
    'signals. Work whose episodes show only negative signals becomes a note ("X was tried and failed because Y; '
    'Z worked"), not a function, unless its tests check values from successful episodes. Where an episode has '
    "no signal, rely on errors and on the gate's checks; never invent a signal.\n"
    "4. Prefer extending an existing item to adding a near-duplicate. When you change what an item returns on "
    "its recorded inputs, add a test that fails on the old version and passes on yours, and say why in `why`. "
    "This pass never removes or merges items and never deletes or weakens a test: the gate refuses a candidate "
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
    "8. Never store credentials, session tokens or the results of calls with side effects.\n"
    "\n"
    "# The gate\n"
    "After `finish` a deterministic gate checks the candidate: the layout and the links; that every fixture "
    "came through `fixture` unchanged; each function's covers; its tests on the candidate, and on the parent "
    "where behaviour changed; the appended recorded inputs; that its tests catch at least <<mutation_min_pct>>% "
    "of small mutations of each changed function; that a function given a same-shaped input from another "
    "episode returns a value or raises `MemoryInputError`; the literal rule; and that no test is lost or "
    "weakened without a reason. An item that breaks a rule is refused together with the items that import it, "
    "and the rest can land; a breach of a pass-wide rule (the layout, lost tests, fewer items) refuses the "
    "round.\n"
    "\n" + _REPAIR + "\n"
    "# Done\n"
    "You are done when every episode in the batch is covered or dismissed, and every item you wrote has passed "
    "the gate or is kept as a draft with its failure. Then call `finish(summary)`.\n"
)

_CURATE = (
    "# Role\n"
    "You are the library curator. The library, `memory`, is a git-tracked Python package of tested functions "
    "and linked notes that a working model imports read-only. You keep it small, correct and easy to navigate. "
    "You work in a sandbox with no network, on the library alone.\n"
    "\n"
    "# Inputs\n"
    "- /memory: a writable checkout of the library with its full git history (`git log`, `git log -L` and "
    "`git show` work there). `import memory` works in your cells: `memory.index()`, `memory.find(value)`, and "
    '`memory.show("<id>")`, which gives an item\'s status, its test record, its use record per library commit '
    "and its recent changes.\n"
    "- <<trigger_path>>: why this pass runs: overlap candidates, suspect items, or an index over its budget of "
    "<<index_tokens>> tokens.\n"
    "- <<overlaps_path>>: the overlap candidates the harness found: functions with matching recorded input "
    "shapes and overlapping covers, functions whose bodies share a common generalisation (given with it), and "
    "notes that use the same functions.\n"
    "- <<suspects_path>>: the suspect items, each with why it is suspect (input refusals, errors or negative "
    "signals after use, in at least two episodes, or a suspect dependency), the first version whose behaviour "
    "changed, and, when an earlier version has a clean use record, that version as the proposed rollback.\n"
    "- /inputs/gate/previous.md: the last pass's full gate result.\n"
    "\n"
    "# Tools\n"
    + _READ_TOOLS
    + "- `execute_code(code)` runs Python as a script in a fresh process, with /memory as the working directory; "
    "only files persist between calls. " + _RUN_TESTS + "\n" + _CHECK_FINISH + "\n"
    "# What to do\n"
    "- Merge overlapping items into one, keeping each old name as an alias: `old_name = new_name` in its "
    "module, listed in the manifest's `aliases`.\n"
    "- Generalise variants into one parametric function.\n"
    "- Repair a suspect item, or roll it back to the proposed earlier version (`git show <commit>:<path>`).\n"
    "- Retire an item that is wrong, or unused across many episodes, together with its tests.\n"
    "- Fix broken links and add missing ones between notes and functions.\n"
    "- Regroup packages when that makes the index clearer, keeping each moved function's old id as an alias.\n"
    "- Write the manifest, /memory/.pass/manifest.json (never committed):\n"
    '  {"items": [{"item": "<id>", "kind": "function" or "note", "input": "<form>", '
    '"source_episodes": [...], "tests": [...], "covers": [...]}], "deleted": ["<retired id>"], '
    '"deleted_tests": ["<test file removed with them>"], "aliases": {"<old id>": "<new id>"}, '
    '"tests_changed": {"<test>": "<reason>"}, "why": "<one line>", "summary": "<what changed>"}\n'
    "  `items` lists every item you added or changed, with the covers of the items it replaces.\n"
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
    "7. Never store credentials or session tokens.\n"
    "\n" + _REPAIR + "\n"
    "# Done\n"
    "You are done when each overlap candidate and suspect item in your inputs has been merged, generalised, "
    "repaired, rolled back or retired, or left as it is with the reason in your summary, and your changes have "
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


def write_brief(
    *,
    view_bytes: int,
    max_checks: int,
    repair_rounds: int,
    mutation_min: Decimal,
    sample_k: int,
    lint_min: int,
    pytest_env: Mapping[str, str],
    semantic_types: str,
    input_forms: Mapping[str, str] = INPUT_FORMS_V21,
    extra_inputs: tuple[str, ...] = (),
) -> str:
    """The WRITE brief with its numbers filled in. *extra_inputs* are whole input bullets that an optional switch
    adds (P8; constant per run)."""
    return _fill(
        _WRITE,
        {
            "extra_inputs": "".join(extra_inputs),
            "view_bytes": view_bytes,
            "max_checks": max_checks,
            "repair_rounds": repair_rounds,
            "mutation_min_pct": _pct(mutation_min),
            "sample_k": sample_k,
            "lint_min": lint_min,
            "pytest_env": _env(pytest_env),
            "semantic_types": semantic_types,
            "input_forms": ", ".join(f"`{k}` ({v})" for k, v in input_forms.items()),
        },
    )


def curate_brief(
    *,
    view_bytes: int,
    max_checks: int,
    repair_rounds: int,
    pytest_env: Mapping[str, str],
    index_tokens: int,
    overlaps_path: str,
    suspects_path: str,
    trigger_path: str,
) -> str:
    """The CURATE brief with its numbers and P6's input paths filled in."""
    return _fill(
        _CURATE,
        {
            "view_bytes": view_bytes,
            "max_checks": max_checks,
            "repair_rounds": repair_rounds,
            "pytest_env": _env(pytest_env),
            "index_tokens": index_tokens,
            "overlaps_path": overlaps_path,
            "suspects_path": suspects_path,
            "trigger_path": trigger_path,
        },
    )


def write_brief_now(extra_inputs: tuple[str, ...] = ()) -> str:
    """The WRITE brief from the constants that enforce each number (imported here: they import this module's
    callers)."""
    from . import code_lint, manifest, qa, repair, sol_pass, views
    from .gate import _PYTEST_ENV

    return write_brief(
        view_bytes=views.VIEW_BYTES,
        max_checks=sol_pass.MAX_CHECKS,
        repair_rounds=repair.REPAIR_ROUNDS,
        mutation_min=qa.MUTATION_MIN,
        sample_k=qa.SAMPLE_K,
        lint_min=code_lint.LINT_MIN,
        pytest_env=_PYTEST_ENV,
        semantic_types=manifest.describe_semantic_types(),
        extra_inputs=extra_inputs,
    )


def curate_brief_now() -> str:
    """The CURATE brief from the enforcing constants and P6's input paths."""
    from . import curate, repair, sol_pass, views
    from .gate import _PYTEST_ENV
    from .library_index import INDEX_VIEW_TOKENS

    return curate_brief(
        view_bytes=views.VIEW_BYTES,
        max_checks=sol_pass.MAX_CHECKS,
        repair_rounds=repair.REPAIR_ROUNDS,
        pytest_env=_PYTEST_ENV,
        index_tokens=INDEX_VIEW_TOKENS,
        overlaps_path=curate.OVERLAPS_INPUT,
        suspects_path=curate.SUSPECTS_INPUT,
        trigger_path=curate.TRIGGER_INPUT,
    )


# --- guards (spec P10, §12.4, the 6 Oct rule) -----------------------------------------------------------------

#: Names and terms of particular benchmarks and environments (case-insensitive, whole words).
_BENCHMARK = re.compile(
    r"\b(apis|venmo|arc|grids?|office|payroll|crafter|alfworld|appworld|scienceworld|travelplanner|webshop|"
    r"minecraft|sokoban|submission)\b",
    re.IGNORECASE,
)
#: An environment's verdict text, which the harness never matches (spec §5): upper case only.
_VERDICT = re.compile(r"\bCORRECT\b")

#: Instructions to check a task's examples (the 6 Oct rule). A function docstring's ``Example:`` section is a
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
    "read_episode",
    "dismiss(",
    "batch_map.json",
    "required_parts",
    "fixture(",
)
CURATE_ONLY = ("overlap", "roll it back", "alias", "retire")
WRITER_ONLY = ("manifest", "the gate", "finish(")

#: Each rule the prompts state: (rule, spec, owners, the phrase that states it). The phrase must be in every
#: owner's text and in no other prompt (tests/memory_v2/test_prompts_v21_contradictions.py).
RULES: tuple[tuple[str, str, tuple[str, ...], str], ...] = (
    (
        "status meanings",
        "§10.1, §12.1",
        ("guide",),
        "`experimental` items are candidates",
    ),
    ("find matches structure only", "§6", ("guide",), "It cannot match plain text"),
    (
        "an input refusal means do the step yourself",
        "§4.3, §12.1",
        ("guide",),
        "Then do that step with your own code",
    ),
    (
        "coverage before finish",
        "§7.4, P4",
        ("write",),
        "`finish` is refused while any episode",
    ),
    (
        "store anything that repeats",
        "§8.2.1, P3, D33",
        ("write",),
        "Store anything that repeats",
    ),
    (
        "procedures are replayed",
        "§8.2.1",
        ("write",),
        "A procedure is verified by replaying its episode",
    ),
    (
        "start from the actor's code",
        "§8.2.2, P2, D32",
        ("write",),
        "Start from the working model's own code",
    ),
    (
        "use every signal",
        "§8.2.3, P1, D31",
        ("write",),
        "Use every signal the records hold",
    ),
    ("failure-only work is a note", "§8.2.3, §11.1", ("write",), "becomes a note"),
    (
        "WRITE never removes",
        "§9.1",
        ("write",),
        "This pass never removes or merges items",
    ),
    ("trust order", "§13.1, D39", ("write", "curate"), "in this order of trust"),
    (
        "exact trusted assertion",
        "§13.2, §13.4",
        ("write",),
        "equals an exact trusted value",
    ),
    (
        "no self-oracle",
        "§13.4",
        ("write",),
        "Never compute an expected value with the function's own expression",
    ),
    (
        "drawn inputs appended",
        "§9.1, §13.4",
        ("write",),
        "more recorded inputs of the same kind to a copy of that file",
    ),
    ("literal rule", "§9.1", ("write",), "make it a parameter"),
    (
        "deterministic tests",
        "qa determinism",
        ("write", "curate"),
        "never assert on the clock",
    ),
    ("no credentials", "§2", ("write", "curate"), "Never store credentials"),
    ("why trailer", "§8.2.6", ("write", "curate"), '"why"'),
    ("repair rounds", "§9.2, P6", ("write", "curate"), "you get a repair round"),
    ("drafts", "§8.4, D41", ("write", "curate"), "kept as a draft"),
    ("merge keeps aliases", "§10.4", ("curate",), "keeping each old name as an alias"),
    (
        "standing tests green",
        "§10.4",
        ("curate",),
        "keeps the library's existing tests green",
    ),
    (
        "retired covers stay covered",
        "§10.4, D13",
        ("curate",),
        "stays covered by a remaining function",
    ),
    ("two episodes decide", "§10.1, §11.4", ("curate",), "one episode never decides"),
)

#: v2 instructions and surfaces that v2.1 retires (spec §16, D31–D35, the 6 Oct rule): none may appear in any
#: text the actor or the writer reads under v2.1.
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

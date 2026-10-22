"""The pass manifest and the memory repo's layout policy, as the gate enforces them.

A consolidation pass writes ``.pass/manifest.json`` (never committed)::

    {"items": [{"item": "env/venmo:login", "kind": "env_function", "source_episodes": ["e1"],
                "tests": ["env/venmo/tests/test_login.py"], "covers": [["e1", 0]]}],
     "deleted": ["env/venmo:old"], "unlisted": ["env/venmo:rare"],
     "deleted_tests": ["env/venmo/tests/test_old.py"], "skeleton": ["env/venmo"],
     "support": ["unify_memory_testkit.py"], "summary": "..."}

``deleted`` and ``unlisted`` name item ids; ``deleted_tests`` names parent test files that exercise only
deleted items; ``skeleton`` names channels whose module skeleton or notes preamble changes; ``support``
names fixture helpers. Everything here is pure: parsing, and which paths the layout admits.

Manifest rules for consolidators
--------------------------------
* **Items.** Every function, note section or workflow you add, change or remove appears exactly once, in
  ``items`` (added or changed), ``deleted`` (removed) or ``unlisted`` (hidden from the index, otherwise
  untouched). The three lists are exclusive. Ids: ``env/<channel>:<function>``,
  ``env/<channel>/NOTES.md#<section-slug>``, ``workflows/<slug>.md`` (lower case). A note section's slug
  is its ``## `` heading lower-cased, every run of characters other than ``a-z`` and ``0-9`` replaced by
  one ``-``, and leading and trailing ``-`` stripped (``## Login: 2FA!`` is ``login-2fa``).
* **Environment functions** in ``items`` list ``covers``: recorded actions on their own channel,
  ``[episode_id, action_index]``, each a real recorded observation of its kind: a tool call with status
  ``ok`` and a response; a shell command with an output tail; a file read or write (status ``ok``) whose
  blob was recorded; a dialogue action (status ``ok``) with its next observation; or a recorded
  rejection (status ``error`` with its error), which lets the function refuse values of that call or
  path family, and never alone. Scope is shape, not observed values: the gate reruns each function on its
  covered inputs with unseen values of the same types and refuses it if it raises ``MemoryInputError``
  on them, unless it covers such a rejection. An item restricts a field's values only by declaring its
  type in ``field_types`` (``{"<field>": "<type>"}``, a type from the fixed list in the brief; unknown
  types are refused); the field is a CSV column (``#3`` for the third column of a file without a
  header), a JSON key path (``a.b``, ``items[]``), a YAML key path or a keyword path, and the gate checks
  that unseen in-domain values are accepted and out-of-domain values are refused. A new or changed
  environment function declares the form its first parameter takes in ``input`` (one of the brief's
  fixed list: {input_forms}) and states the same form in its
  docstring as a line ``Input: <form>``; the gate refuses an unknown or missing form and a docstring that
  disagrees, and passes each covered input to the function in that form. An action's channel is
  its key when the key has no ``:`` (``venmo``), else ``<kind>_<key>`` (``shell:uv`` is ``env/shell_uv``,
  ``worktree:workspace`` is ``env/worktree_workspace``, ``dialogue:user`` is ``env/dialogue_user``; other
  characters become ``_``). Only environment functions have covers. Tests must check the function
  against the covered observations themselves: copy the recorded outputs, file blobs or (action, next
  observation) pairs they need into data files under ``env/<channel>/tests/`` and list those files in
  ``support`` (the gate mounts nothing else). A new or changed function lists
  a new or changed test under ``env/<channel>/tests/test_*.py`` that passes on your commit and fails on
  the parent's library (or hangs there). Any other new or changed test (one that records what an unchanged
  function already does, a corrected expected value, or a repair of a parent test file that was already red)
  needs only pass on your commit; it never stands in for a changed function's red test. A changed function
  always needs a test that fails on the parent's library. So does any change
  in what a stored function returns on recorded inputs: for every channel your commit changes (any file
  under ``env/<channel>/``: a function, a private helper, a constant, an alias, an import), the gate runs
  every public function the parent's module had, under the same name in yours (a ``def``, ``A = B`` or an
  import), on both libraries over its recorded covers and the channel's recorded actions in your episodes;
  a function whose results differ there, or cannot be compared (not JSON, too slow), must be listed in
  ``items`` with a test that fails on the parent's library and passes on yours. The one exception for a
  changed function is a change that keeps behaviour in a clean-up pass (no item added; fewer function definitions or less
  code in them, and no more code anywhere in the channel modules; comments and docstrings do not count):
  a merge, an alias or a shortening needs no failing test when a parent test that passed on the parent
  calls the function (importing it is not enough) and the function returns, refuses and calls exactly
  what its parent version did on every recorded cover of it and on your episodes' recorded actions (the
  gate runs both versions on them). A repair changes behaviour, so it always needs a failing test. New tests of a clean-up pass need only pass.
* **Skeleton.** A channel's *skeleton* is its module minus its public functions and a literal
  ``__all__ = [...]``: the docstring, imports, private helpers and any other module-level statement, plus
  the preamble of its ``NOTES.md`` (the text before the first ``## `` heading). Changing it, including
  creating a new channel module that has a docstring or imports, needs ``"skeleton": ["env/<channel>"]``.
  A skeleton entry declares the channel's ``__init__.py`` and ``NOTES.md`` files, so a preamble-only edit
  needs nothing else; every function or section you add, change or remove is still listed as an item.
  A changed module skeleton also needs every public function of that module in ``items``, each with valid
  covers (unchanged functions need no new test unless their results change; their old tests must keep
  passing).
* **Tests are never lost.** Every test that passed on the parent must still pass, both in your commit's
  suite and with the parent's own tests and test kit against your library. Do not rename or drop tests.
  Your commit's suite may keep failing only tests that already failed on the parent (no new failures);
  those are reported as pre-existing notes. If a channel's suite on the parent cannot be read (say, a
  test file does not import), it protects nothing, and a commit that touches that channel must leave
  its suite fully green; a channel you do not touch may stay broken, with a note.
  ``deleted_tests`` retires a parent test file only when your commit removes it and every library name it
  imports (``from env.<channel> import ...``; ``import env.<channel>`` means the whole channel) is an item
  this manifest deletes. Every recorded input a deleted function covered (a recorded rejection aside) must
  stay covered: list it in a remaining function's ``covers``.
* **Support.** The only root helper is ``unify_memory_testkit.py``; other helpers live under
  ``env/<channel>/tests/`` and must not be named like tests, like an installed or standard module, ``env``
  or the test kit. No ``conftest.py``, pytest or packaging configuration, ``.pth`` files, executables,
  links or submodules anywhere.
* **Workflows** need YAML front matter (``---`` lines) to be unlistable: unlisting changes only the
  ``listed: false`` line.
* **No duplicate definitions**: a public function name is defined once per module.
"""

from __future__ import annotations

import importlib.machinery
import posixpath
import re
import sys
from dataclasses import dataclass, field

# Declared semantic types: the only value constraints an environment function may hold, each a
# property of a type declared on a field in its manifest item (``field_types``), never inferred from observed
# values. kind, low, high (None: unbounded). This list grows only by a reviewed code change, never by a pass.
SEMANTIC_TYPES: dict[str, tuple[str, int | None, int | None]] = {
    "month": ("int", 1, 12),
    "day_of_month": ("int", 1, 31),
    "hour": ("int", 0, 23),
    "probability": ("number", 0, 1),
    "percentage": ("number", 0, 100),
    "nonneg_count": ("int", 0, None),
    "nonneg_money": ("decimal", 0, None),
    "currency_code": (
        "code3",
        None,
        None,
    ),  # the ISO 4217 format: three uppercase letters, not a list
}
_FIELD_NAME_MAX = 200

# Declared input forms: what an environment function's first parameter takes (``input`` in its manifest item,
# and an ``Input: <form>`` line in its docstring), as the index shows it. This list grows only by a reviewed code
# change, never by a pass.
INPUT_KINDS: dict[str, str] = {
    "path": "a path to a work-tree file",
    "text": "the text of a file or of an output",
    "bytes": "the raw bytes of a file",
    "observation": "a dialogue or tool observation value, as recorded",
    "env": "the environment object, such as `apis`",
}
# The consolidator rules name the forms from the one constant (they are embedded in the writer's brief verbatim).
if __doc__:
    __doc__ = __doc__.replace(
        "{input_forms}",
        ", ".join(f"``{name}``" for name in INPUT_KINDS),
    )


def describe_semantic_types() -> str:
    """The fixed list with each type's domain, from SEMANTIC_TYPES (for the consolidator's brief)."""
    out = []
    for name, (kind, low, high) in SEMANTIC_TYPES.items():
        if kind == "code3":
            domain = "three uppercase letters, a format"
        elif high is None:
            domain = f"{'integer' if kind == 'int' else kind} ≥ {low}"
        else:
            domain = f"{'integer' if kind == 'int' else kind} {low}–{high}"
        out.append(f"{name} ({domain})")
    return ", ".join(out)


# Job functions (end-to-end code) are not expected in v0 and have no admitted location.
KINDS = ("env_function", "env_note", "workflow")
TESTKIT = "unify_memory_testkit.py"

_CH = r"[a-z][a-z0-9_]*"
ITEM_ID = {
    "env_function": re.compile(rf"^env/{_CH}:[a-z_][a-z0-9_]*\Z"),
    "env_note": re.compile(rf"^env/{_CH}/NOTES\.md#[a-z0-9-]+\Z"),
    "workflow": re.compile(r"^workflows/[a-z0-9][a-z0-9_-]*\.md\Z"),
}
SKELETON_ID = re.compile(rf"^env/{_CH}\Z")
TESTS_DIR = re.compile(r"^env/[^/]+/tests/(?P<rest>.+)\Z")
# Files that configure git, pytest or the interpreter, refused anywhere in the memory tree.
FORBIDDEN_NAMES = frozenset(
    {
        ".gitattributes",
        ".gitignore",
        ".gitmodules",
        "conftest.py",
        "pytest.ini",
        ".pytest.ini",
        "pytest.toml",
        ".pytest.toml",
        "pyproject.toml",
        "setup.cfg",
        "tox.ini",
        "sitecustomize.py",
        "usercustomize.py",
    },
)


class ManifestError(ValueError):
    pass


@dataclass
class ManifestItem:
    item: str
    kind: str
    path: str
    source_episodes: list[str]
    tests: list[str]
    covers: list[tuple[str, int]]
    field_types: dict[str, str] = field(
        default_factory=dict,
    )  # field -> SEMANTIC_TYPES key
    input: str | None = None  # an INPUT_KINDS key; None: not declared
    # library mode: the typed covers (procedures.Cover: an actor cell, a work-tree diff or an episode procedure)
    typed_covers: list = field(default_factory=list)
    # the proposals this item came from (:func:`adopt_ref`): staged files, helper output or pool entries
    adopts: list[str] = field(default_factory=list)
    #: how each adopted draft was used (ref -> one of ADOPT_HOW), the same refs as ``adopts``
    adopt_how: dict[str, str] = field(default_factory=dict)
    # library mode: the function's role (one of ITEM_ROLES; None: none), a procedure's reader, its recipe (the id of
    # the function in its module that does the whole job on a request) and its case files (``.case.json`` fixtures
    # of its package, each an input with its known output, :func:`.fixtures.make_case`)
    role: str | None = None
    reader: str | None = None
    recipe: str | None = None
    cases: list[str] = field(default_factory=list)
    # what the writer states about the item (:data:`LABEL_ITEM_FIELDS`), kept on its
    # label; empty: nothing stated
    stated: dict = field(default_factory=dict)

    @property
    def channel(self) -> str | None:
        parts = self.path.split("/")
        return parts[1] if parts[0] == "env" and len(parts) > 2 else None


@dataclass
class Manifest:
    items: list[ManifestItem] = field(default_factory=list)
    support: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    unlisted: list[str] = field(default_factory=list)
    deleted_tests: list[str] = field(default_factory=list)
    skeleton: list[str] = field(default_factory=list)


def safe_rel(path: object) -> str:
    """A normalised relative POSIX path inside the tree, or raise ManifestError."""
    if (
        not isinstance(path, str)
        or not path
        or path.startswith("/")
        or "\\" in path
        or "\0" in path
        or "\n" in path
        or posixpath.normpath(path) != path
        or path == "."
        or ".." in path.split("/")
        or path.split("/")[0] == ".git"
    ):
        raise ManifestError(f"bad path {path!r}"[:200])
    return path


def item_kind(item: str) -> str | None:
    """The kind a well-formed item id names, or None."""
    for kind, pattern in ITEM_ID.items():
        if pattern.match(item):
            return kind
    return None


def item_path(item: str) -> str:
    """The file an item id lives in: ``env/x:f`` -> ``env/x/__init__.py``; ``a.md#s`` -> ``a.md``."""
    if ":" in item:
        return item.split(":", 1)[0] + "/__init__.py"
    return item.split("#", 1)[0]


def forbidden(path: str) -> bool:
    name = path.rsplit("/", 1)[-1]
    return name in FORBIDDEN_NAMES or name.endswith(".pth")


# Code the interpreter runs without a source the gate could read: bytecode and native extensions,
# under any ABI-tagged name (``x.cpython-312.pyc``, ``x.abi3.so``, ``x.pypy310-pp73-….so``).
_COMPILED_SUFFIXES = tuple(
    sorted(
        {".pyc", ".pyo", ".so", ".pyd", ".dll", ".dylib"}
        | {s.lower() for s in importlib.machinery.BYTECODE_SUFFIXES}
        | {s.lower() for s in importlib.machinery.EXTENSION_SUFFIXES},
    ),
)
_ABI_TAGGED = re.compile(r"\.(?:cpython|pypy|graalpy)[0-9]*-|\.abi[0-9]+\.", re.I)
# Modules the interpreter imports on its own at start-up, under any suffix or as a package.
_START_HOOKS = frozenset({"sitecustomize", "usercustomize"})
# The root entries a library can hold: everything a pass can declare lives in env/<channel>/ (modules, notes,
# tests and their helpers) or workflows/ (workflow notes), and the test kit is the only root file.
ROOT_DIRS = ("env", "workflows")


def root_shadowed() -> frozenset[str]:
    """Top-level names a root package would take from the standard library or the gate's pytest."""
    return frozenset(
        set(sys.stdlib_module_names)
        | set(sys.builtin_module_names)
        | {"pytest", "_pytest", "pluggy"},
    )


def unsafe_path(path: str) -> tuple[str, str] | None:
    """``(check, reason)`` when *path* must never be extracted or run, else None.

    Refused anywhere (G6): bytecode or native code, anything under ``__pycache__``, ``.pth`` files and the
    interpreter's start-up hooks (``sitecustomize``/``usercustomize`` under any suffix or as a package).
    Refused at the root: anything but ``env/``, ``workflows/`` and the test kit (G1, layout), and a directory
    named like a standard-library or pytest module (G6). A reason names the path and the class of problem,
    never contents.
    """
    parts = path.split("/")
    name = parts[-1]
    if "__pycache__" in parts:
        return "G6", f"bytecode cache path {path}"
    if name.lower().endswith(_COMPILED_SUFFIXES) or _ABI_TAGGED.search(name):
        return "G6", f"bytecode or native code file {path}"
    if name.lower().endswith(".pth"):
        return "G6", f"path configuration file {path}"
    if any(p.split(".", 1)[0].lower() in _START_HOOKS for p in parts):
        return "G6", f"interpreter start-up hook {path}"
    if path == TESTKIT or (len(parts) > 1 and parts[0] in ROOT_DIRS):
        return None
    if len(parts) > 1 and parts[0] in root_shadowed():
        return "G6", f"root package {parts[0]}/ ({path}) shadows the {parts[0]} module"
    return (
        "G1",
        f"root entry {path} is outside the layout (the root holds only env/, workflows/ and {TESTKIT})",
    )


# an item's ``adopts`` refs: ``staging:<episode>/<file>``, ``helper:<helper name>/<file>`` or ``pool:<entry id>``, where
# <file> is a staged file (``candidates/<name>.py``, ``cases/<name>.json`` or ``notes.md``); the same files named by
# their /inputs path are accepted and stored in the ref form
#: How an item used a draft (a staged candidate, case or note, a helper's output, a pool entry): as it was, edited,
#: merged with other drafts or library items, generalised (parametrised), rewritten, or only informing new work.
ADOPT_HOW = ("as_is", "edited", "merged", "generalised", "rewritten", "informed")

ADOPT_FORMS = (
    "staging:<episode>/candidates/<name>.py, staging:<episode>/cases/<name>.json, staging:<episode>/notes.md, "
    "helper:<helper name>/<same files>, or pool:<entry id>"
)
_ADOPT_FILE = r"(?:candidates/[^/\s\\]+\.py|cases/[^/\s\\]+\.json|notes\.md)"
_ADOPT_DIR = r"(?!\.\.?/)[^/\s\\]+"
_ADOPT = re.compile(
    rf"(?:(?P<src>staging|helper):(?P<dir>{_ADOPT_DIR})/(?P<file>{_ADOPT_FILE})|pool:(?P<pool>[0-9a-f]{{16}}))",
)
_ADOPT_PATHS = (
    ("/inputs/staging/", "staging:"),
    ("/inputs/helpers/", "helper:"),
)


def adopt_ref(raw: object) -> str:
    """One ``adopts`` ref in its canonical form, or ManifestError naming the accepted forms. ``/inputs/staging/...``,
    ``/inputs/helpers/...`` and ``/inputs/pool/<id>[/...]`` paths are accepted for the same files.
    """
    if not isinstance(raw, str) or "\0" in raw:
        raise ManifestError(
            f"adopts: {raw!r} is not a string ref ({ADOPT_FORMS})"[:400],
        )
    ref = raw.strip()
    for prefix, form in _ADOPT_PATHS:
        if ref.startswith(prefix):
            ref = form + ref[len(prefix) :]
    if ref.startswith("/inputs/pool/"):
        ref = "pool:" + ref[len("/inputs/pool/") :].split("/", 1)[0]
    if not _ADOPT.fullmatch(ref):
        raise ManifestError(f"adopts: {raw!r} is not one of {ADOPT_FORMS}"[:400])
    return ref


def _adopts(raw: object, item: str) -> tuple[list[str], dict[str, str]]:
    """An item's ``adopts``: entries ``{"ref": <a draft ref>, "how": <one of ADOPT_HOW>}`` -> (refs, ref -> how)."""
    out: list[str] = []
    how: dict[str, str] = {}
    if raw is None:
        return out, how
    form = f'{{"ref": <{ADOPT_FORMS}>, "how": <one of {", ".join(ADOPT_HOW)}>}}'
    if not isinstance(raw, list):
        raise ManifestError(f"{item}: adopts must be a list of {form}"[:500])
    for r in raw:
        if not isinstance(r, dict) or set(r) != {"ref", "how"}:
            raise ManifestError(f"{item}: each adopts entry is {form}, not {r!r}"[:500])
        if r["how"] not in ADOPT_HOW:
            raise ManifestError(
                f"{item}: adopts how {r['how']!r} is not one of {', '.join(ADOPT_HOW)}"[
                    :300
                ],
            )
        ref = adopt_ref(r["ref"])
        if ref in how and how[ref] != r["how"]:
            raise ManifestError(
                f"{item}: {ref} is adopted twice with different how values"[:300],
            )
        if ref not in out:
            out.append(ref)
        how[ref] = r["how"]
    return out, how


def _str_list(value: object, what: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ManifestError(f"{what} must be a list of strings")
    return list(value)


#: The keys a library-mode manifest item takes; any other key is refused, never dropped.
LIBRARY_ITEM_FIELDS: tuple[str, ...] = (
    "item",
    "kind",
    "source_episodes",
    "tests",
    "covers",
    "field_types",
    "adopts",
    "role",
    "reader",
    "recipe",
    "cases",
)
#: What the writer may state about an item (its label keeps it): the kind of situation it is for (in general words;
#: rev 2.1: never the episodes it is meant to cover, which are its evidence), what varies (its inputs), the values it
#: holds fixed with their sources (its constants), a function's effects on the environment, and a note's form (a
#: checklist). Refused as unknown fields otherwise.
LABEL_ITEM_FIELDS: tuple[str, ...] = ("situation", "inputs", "constants", "effects", "form")
#: A function's declared effects.
DECLARED_EFFECTS: tuple[str, ...] = ("none", "reads", "writes", "unknown")
#: A note's form.
NOTE_FORMS: tuple[str, ...] = ("note", "checklist")
_STATED_TEXT_MAX = 400


def _stated(raw: dict, item: str, kind: str) -> dict:
    """The label fields of a manifest item (:data:`LABEL_ITEM_FIELDS`), checked; ``{}`` when it states none."""
    out: dict = {}
    if "situation" in raw:
        situation = raw["situation"]
        if not isinstance(situation, str) or not situation.strip() or len(situation) > _STATED_TEXT_MAX:
            raise ManifestError(
                f"{item}: situation is the kind of situation the item is for, in at most {_STATED_TEXT_MAX} "
                f"characters, given {_key_repr(situation)}",
            )
        out["situation"] = " ".join(situation.split())
    if "inputs" in raw:
        got = _str_list(raw["inputs"], f"{item}: inputs")
        if any(len(x) > _STATED_TEXT_MAX for x in got):
            raise ManifestError(f"{item}: an input is named in at most {_STATED_TEXT_MAX} characters")
        out["inputs"] = got
    if "constants" in raw:
        consts = raw["constants"]
        if not isinstance(consts, list):
            raise ManifestError(f"{item}: constants is a list of {{\"value\", \"source\"}} (or of lines)")
        lines = []
        for c in consts:
            if isinstance(c, str):
                line = c
            elif isinstance(c, dict) and isinstance(c.get("value"), str) and isinstance(c.get("source"), str):
                line = f"{c['value']} (source: {c['source']})"
            else:
                raise ManifestError(
                    f"{item}: a constant is {{\"value\": \"<what it is>\", \"source\": \"<episode>/<part>\"}}, given "
                    f"{_key_repr(c)}",
                )
            if len(line) > _STATED_TEXT_MAX:
                raise ManifestError(f"{item}: a constant is stated in at most {_STATED_TEXT_MAX} characters")
            lines.append(line)
        out["constants"] = lines
    if "effects" in raw:
        if kind != "function":
            raise ManifestError(f"{item}: only a function declares effects")
        if raw["effects"] not in DECLARED_EFFECTS:
            raise ManifestError(f"{item}: effects is one of {', '.join(DECLARED_EFFECTS)}, given {_key_repr(raw['effects'])}")
        out["effects"] = raw["effects"]
    if "form" in raw:
        if kind != "note" or raw["form"] not in NOTE_FORMS:
            raise ManifestError(f"{item}: form is one of {', '.join(NOTE_FORMS)}, on a note")
        out["form"] = raw["form"]
    return out


#: A library function's ``role``, when it has one: a whole-request procedure, or a reader of request text.
ITEM_ROLES: tuple[str, ...] = ("procedure", "reader")
#: The name ending of a case file (:data:`.fixtures.CASE_SUFFIX`).
CASE_SUFFIX = ".case.json"


def _recipe_and_cases(raw: dict, item: str, role: object) -> tuple[str | None, list[str]]:
    """A procedure's ``recipe`` (a function id in the procedure's own module) and ``cases`` (case files under its
    package's ``tests/data/``, each listed once); either is refused on any other item."""
    from . import layout

    recipe, cases = raw.get("recipe"), raw.get("cases")
    if (recipe is not None or cases is not None) and role != "procedure":
        raise ManifestError(f"{item}: recipe and cases belong on a function whose role is procedure")
    if recipe is not None:
        if not isinstance(recipe, str) or not layout.FUNCTION_ID.match(recipe):
            raise ManifestError(f"{item}: recipe names a function id (memory.<package>.<module>:<name>)"[:300])
        if recipe.split(":", 1)[0] != item.split(":", 1)[0]:
            raise ManifestError(
                f"{item}: its recipe {recipe} is in another module; a procedure's recipe lives in its own module"[:300],
            )
    out: list[str] = []
    if cases is not None:
        data_dir = f"{item.split(':', 1)[0].rsplit('.', 1)[0].replace('.', '/')}/tests/data/"
        for path in _str_list(cases, f"{item}: cases"):
            safe_rel(path)
            if not path.startswith(data_dir) or not path.endswith(CASE_SUFFIX) or len(path) <= len(data_dir):
                raise ManifestError(
                    f"{item}: case {_key_repr(path)} is not a case file of its package ({data_dir}<name>{CASE_SUFFIX}, "
                    "made with fixture(..., expected=...))",
                )
            if path in out:
                raise ManifestError(f"{item}: case {_key_repr(path)} is listed twice")
            out.append(path)
    return recipe, out
#: The top-level keys a library-mode manifest takes; any other key is refused, never dropped. ``fixtures`` is
#: always rewritten by the harness; the other keys this parser does not read are read by the gate's library checks,
#: the pool, the curate pass and the pass itself.
LIBRARY_MANIFEST_FIELDS: tuple[str, ...] = (
    "items",
    "support",
    "deleted",
    "unlisted",
    "deleted_tests",
    "skeleton",
    "fixtures",
    "tests_changed",
    "renames",
    "pool",
    "why",
    "summary",
    "kept",
    "retired",
    "aliases",
)
_KEY_REPR_MAX = 100


def _key_repr(key: object) -> str:
    """The repr of a manifest key for a refusal, bounded so a long key cannot fill the message."""
    text = repr(key)
    return text if len(text) <= _KEY_REPR_MAX else text[: _KEY_REPR_MAX - 2] + "…"


def _library_field_types(raw: object, item: str, kind: str) -> dict[str, str]:
    """``{"<fixture path>#<field>": semantic type}``: a function's declared field types, each from SEMANTIC_TYPES.
    The path is a fixture of the item's own package (``memory/<package>/tests/data/...``); the field is named as
    a JSON or YAML key path, a CSV column or ``#3``.
    """
    if raw is None:
        return {}
    if kind != "function":
        raise ManifestError(f"{item}: only functions declare field types")
    if not isinstance(raw, dict):
        raise ManifestError(f"{item}: field_types must be an object")
    package = item.split(".")[1] if item.count(".") >= 2 else ""
    data_dir = f"memory/{package}/tests/data/"
    for key, typ in raw.items():
        path, sep, name = key.partition("#") if isinstance(key, str) else ("", "", "")
        ok = (
            bool(sep)
            and bool(name)
            and len(name) <= _FIELD_NAME_MAX
            and "\n" not in name
        )
        try:
            ok = (
                ok and safe_rel(path).startswith(data_dir) and len(path) > len(data_dir)
            )
        except ManifestError:
            ok = False
        if not ok:
            raise ManifestError(
                f"{item}: field_types key {_key_repr(key)} is not <fixture path>#<field> with the path under "
                "memory/<package>/tests/data/",
            )
        if not isinstance(typ, str) or typ not in SEMANTIC_TYPES:
            raise ManifestError(
                f"{item}: unknown semantic type {_key_repr(typ)}; the fixed list is "
                f"{', '.join(SEMANTIC_TYPES)}",
            )
    return dict(raw)


def _parse_library_mode(manifest: dict, labels: bool = False) -> Manifest:
    """The library-mode manifest: ``function`` items ``memory.<package>.<module>:<name>`` and
    ``note`` items ``notes/<topic>/<slug>.md``, tests under ``memory/<package>/tests/``, test helpers as
    ``support``, and typed covers (:mod:`.procedures`: a cell, a diff or an episode). A function takes any
    arguments: its examples are the calls its listed tests make, so it lists at least one test, and it declares
    no input form. Keys outside :data:`LIBRARY_MANIFEST_FIELDS` and :data:`LIBRARY_ITEM_FIELDS` are refused.
    The ``fixtures`` and ``tests_changed`` keys are read by the gate (:mod:`.library_checks`).
    """
    from . import layout
    from .procedures import parse_cover

    patterns = {"function": layout.FUNCTION_ID, "note": layout.NOTE_ID}
    if "adopts" in manifest:
        # a manifest-wide list would credit no item; each item names its own proposals
        raise ManifestError(
            "adopts belongs on each item that came from a proposal, not on the manifest",
        )
    for key in manifest:
        if key not in LIBRARY_MANIFEST_FIELDS:
            raise ManifestError(
                f"unknown manifest field {_key_repr(key)}; the manifest takes {', '.join(LIBRARY_MANIFEST_FIELDS)}",
            )

    def item_id(entry: object) -> bool:
        return isinstance(entry, str) and any(p.match(entry) for p in patterns.values())

    def tests_of(raw: object, what: str) -> list[str]:
        out = [safe_rel(t) for t in _str_list(raw, what)]
        for t in out:
            if layout.classify(t) != "test":
                raise ManifestError(
                    f"test {t} is not memory/<package>/tests/test_<name>.py",
                )
        return out

    raw_items = manifest.get("items", [])
    if not isinstance(raw_items, list):
        raise ManifestError("items must be a list")
    out = Manifest()
    seen: set[str] = set()
    for raw in raw_items:
        if not isinstance(raw, dict) or not isinstance(raw.get("item"), str):
            raise ManifestError(f"bad item {raw!r}"[:200])
        item = raw["item"]
        if item in seen:
            raise ManifestError(f"{item} is listed twice")
        seen.add(item)
        if "input" in raw:
            raise ManifestError(
                f"{item}: input is no longer read: a function takes any arguments and its examples are the calls "
                "its tests make on fixtures; remove input",
            )
        fields = LIBRARY_ITEM_FIELDS + (LABEL_ITEM_FIELDS if labels else ())
        for key in raw:
            if key not in fields:
                raise ManifestError(
                    f"{item}: unknown field {_key_repr(key)}; an item takes {', '.join(fields)}",
                )
        kind = raw.get("kind")
        if kind not in patterns:
            raise ManifestError(f"{item}: unknown or unsupported kind {kind!r}"[:200])
        if not patterns[kind].match(item):
            raise ManifestError(f"{item!r}: not a {kind} id"[:200])
        role, reader = raw.get("role"), raw.get("reader")
        if role is not None and (kind != "function" or role not in ITEM_ROLES):
            raise ManifestError(f"{item}: role is one of {', '.join(ITEM_ROLES)}, on a function"[:300])
        if reader is not None and (role != "procedure" or not isinstance(reader, str) or not layout.FUNCTION_ID.match(reader)):
            raise ManifestError(f"{item}: reader names a function id, on a function whose role is procedure"[:300])
        tests = tests_of(raw.get("tests"), f"{item}: tests")
        if kind == "function" and not tests:
            raise ManifestError(
                f"{item}: a function lists the tests that hold its examples (tests is empty)",
            )
        raw_covers = raw.get("covers")
        if raw_covers is not None and not isinstance(raw_covers, list):
            raise ManifestError(f"{item}: covers must be a list")
        if raw_covers and kind != "function":
            raise ManifestError(f"{item}: only functions cover recorded work")
        typed: list = []
        for raw_cover in raw_covers or []:
            try:
                c = parse_cover(raw_cover)
            except ValueError as exc:
                raise ManifestError(f"{item}: {exc}"[:300]) from exc
            if isinstance(c, tuple):
                raise ManifestError(
                    f"{item}: the action cover {raw_cover!r} is no longer read: list the tests that call it on "
                    "fixtures made with fixture (covers takes only cell, diff and episode covers)",
                )
            typed.append(c)
        field_types = _library_field_types(raw.get("field_types"), item, kind)
        recipe, cases = _recipe_and_cases(raw, item, role)
        out.items.append(
            ManifestItem(
                item,
                kind,
                layout.item_path(item),
                _str_list(raw.get("source_episodes"), f"{item}: source_episodes"),
                tests,
                [],
                field_types,
                None,
                typed,
                *_adopts(raw.get("adopts"), item),
                role=role,
                reader=reader,
                recipe=recipe,
                cases=cases,
                stated=_stated(raw, item, kind) if labels else {},
            ),
        )
    out.support = [safe_rel(p) for p in _str_list(manifest.get("support"), "support")]
    for p in out.support:
        # a shared module the package's items import (private helpers, the package's MemoryInputError) is
        # declared here too; any public function in it is still an item, which the manifest must list
        if layout.classify(p) not in ("test_helper", "module"):
            raise ManifestError(
                f"support file {p} is not a test helper under memory/<package>/tests/ or a module "
                "memory/<package>/<module>.py",
            )
    out.deleted = _str_list(manifest.get("deleted"), "deleted")
    out.unlisted = _str_list(manifest.get("unlisted"), "unlisted")
    for entry in out.deleted + out.unlisted:
        if not item_id(entry):
            raise ManifestError(f"{entry!r} is not an item id"[:200])
        if entry in seen:
            raise ManifestError(f"{entry} is both in items and deleted or unlisted")
    if set(out.deleted) & set(out.unlisted):
        raise ManifestError("an item cannot be both deleted and unlisted")
    out.deleted_tests = tests_of(manifest.get("deleted_tests"), "deleted_tests")
    # the library modules whose module-level code (imports, bindings, docstring) the pass changes
    out.skeleton = _str_list(manifest.get("skeleton"), "skeleton")
    for s in out.skeleton:
        if not re.fullmatch(r"memory\.[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)?", s):
            raise ManifestError(
                f"{s!r} is not a library module (memory.<package>[.<module>])"[:200],
            )
    return out


def parse_manifest(manifest: object, *, labels: bool = False) -> Manifest:
    """Validate the manifest's shape, its ids and its paths against the layout; raise ManifestError.

    The library layout (:func:`_parse_library_mode`). *labels*: an item may state its label fields
    (:data:`LABEL_ITEM_FIELDS`).
    """
    if not isinstance(manifest, dict):
        raise ManifestError("the manifest must be an object")
    return _parse_library_mode(manifest, labels=labels)

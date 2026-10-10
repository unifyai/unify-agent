"""The pass manifest and the memory repo's layout policy, as the gate enforces them (spec §4, §7).

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
  the parent's library (or hangs there). Repairing a parent test file that was already red also counts,
  but only as a test-only repair: its parent version fails on the parent, a test it failed passes in
  your version (a file that did not even import counts as a whole), and it imports no function your pass
  changes. A changed function always needs a test that fails on the parent's library. So does any change
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
import importlib.metadata
import posixpath
import re
import sys
from dataclasses import dataclass, field
from functools import lru_cache

# Declared semantic types (spec F3a, D21): the only value constraints an environment function may hold, each a
# property of a type declared on a field in its manifest item (``field_types``), never inferred from observed
# values. kind, low, high (None: unbounded). This list grows only by the lead's decision, never by a pass.
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
# and an ``Input: <form>`` line in its docstring). The gate's held-out check passes each covered input in this
# form, and the index shows it. This list grows only by the lead's decision, never by a pass.
INPUT_KINDS: dict[str, str] = {
    "path": "a path to a work-tree file",
    "text": "the text of a file or of an output",
    "bytes": "the raw bytes of a file",
    "observation": "a dialogue or tool observation value, as recorded",
    "env": "the environment object, such as `apis`",
}
# The consolidator rules name the forms from the one constant (they are embedded in Sol's brief verbatim).
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


def describe_input_kinds() -> str:
    """The fixed list of input forms with what each passes, from INPUT_KINDS (for the consolidator's brief)."""
    return ", ".join(f"{name} ({what})" for name, what in INPUT_KINDS.items())


# Job functions (end-to-end code) are not expected in v0 (spec §4) and have no admitted location.
KINDS = ("env_function", "env_note", "workflow")
TESTKIT = "unify_memory_testkit.py"

_CH = r"[a-z][a-z0-9_]*"
ITEM_ID = {
    "env_function": re.compile(rf"^env/{_CH}:[a-z_][a-z0-9_]*\Z"),
    "env_note": re.compile(rf"^env/{_CH}/NOTES\.md#[a-z0-9-]+\Z"),
    "workflow": re.compile(r"^workflows/[a-z0-9][a-z0-9_-]*\.md\Z"),
}
SKELETON_ID = re.compile(rf"^env/{_CH}\Z")
TEST_PATH = re.compile(rf"^env/{_CH}/tests/(?:[^/]+/)*test_[^/]*\.py\Z")
TESTS_DIR = re.compile(r"^env/[^/]+/tests/(?P<rest>.+)\Z")
MODULE_PATH = re.compile(r"^env/[^/]+/__init__\.py\Z")
NOTES_PATH = re.compile(r"^env/[^/]+/NOTES\.md\Z")
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
# Names that would shadow the memory library or the test kit from inside a tests directory.
_RESERVED_STEMS = frozenset({"env", "unify_memory_testkit"})

# Import names a root entry of the export must never take: the library package and the generated helper
# (``memory.py``), in any suffix (``memory.abi3.so``, ``env.py``) or as a directory (``memory/``). The gate
# names these reserved (:func:`.catalogue.reserved`); compiled code and every other root entry are refused by
# :func:`unsafe_path`.
SHADOWED_IMPORTS = frozenset({"env", "memory"})


def shadows_import(path: str) -> bool:
    """Whether *path* is a root entry an ``import env`` or ``import memory`` in a cell could resolve to."""
    if "/" in path:
        return path.split("/", 1)[0] == "memory"
    return path.split(".", 1)[0] in SHADOWED_IMPORTS


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
    # memory v2.1: the typed covers (procedures.Cover: an actor cell, a work-tree diff or an episode procedure)
    typed_covers: list = field(default_factory=list)

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


@lru_cache(maxsize=1)
def shadowed_module_names() -> frozenset[str]:
    """Top-level names a helper on the test path must not take: the stdlib and installed distributions."""
    names = set(sys.stdlib_module_names) | set(sys.builtin_module_names)
    try:
        names |= set(importlib.metadata.packages_distributions())
    except Exception:  # a broken distribution's metadata must not open the gate
        names |= {"pytest", "_pytest", "pluggy"}
    return frozenset(names | _RESERVED_STEMS)


def forbidden(path: str) -> bool:
    name = path.rsplit("/", 1)[-1]
    return name in FORBIDDEN_NAMES or name.endswith(".pth")


# Code the interpreter runs without a source the gate could read (review I4): bytecode and native extensions,
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
    """``(check, reason)`` when *path* must never be extracted or run, else None (review I4).

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


def _stem(part: str) -> str:
    return part[:-3] if part.endswith(".py") else part


def support_allowed(path: str) -> bool:
    """The test kit at the root, or a helper under ``env/<name>/tests/`` that shadows no module."""
    if path == TESTKIT:
        return True
    m = TESTS_DIR.match(path)
    if m is None or forbidden(path):
        return False
    parts = m.group("rest").split("/")
    name = parts[-1]
    if (
        name == "__init__.py"
        or name.startswith("test_")
        or _stem(name).endswith("_test")
    ):
        return (
            False  # a test file is declared by its item, so that it runs red and green
        )
    return not any(_stem(p) in shadowed_module_names() for p in parts)


def layout_allowed(path: str) -> bool:
    """Whether the layout admits *path*: channel modules, notes, tests, helpers, workflows, the test kit.

    No submodules in v0; nothing under a tests directory takes a reserved name (``env``, the test kit).
    Compiled code, start-up hooks and root entries outside ``env/``, ``workflows/`` and the test kit are
    refused by :func:`unsafe_path`.
    """
    m = TESTS_DIR.match(path)
    if m is not None:
        if any(_stem(p) in _RESERVED_STEMS for p in m.group("rest").split("/")):
            return False
        if TEST_PATH.match(path):
            return True
        if path.endswith(".py"):
            return support_allowed(path)
        return True
    if path.endswith(".py"):
        return bool(MODULE_PATH.match(path)) or path == TESTKIT
    return True


def _str_list(value: object, what: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ManifestError(f"{what} must be a list of strings")
    return list(value)


def _covers(raw: object, item: str) -> list[tuple[str, int]]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ManifestError(f"{item}: covers must be a list")
    out: list[tuple[str, int]] = []
    for c in raw:
        if (
            not isinstance(c, (list, tuple))
            or len(c) != 2
            or not isinstance(c[0], str)
            or not isinstance(c[1], int)
            or isinstance(c[1], bool)
            or c[1] < 0
        ):
            raise ManifestError(f"{item}: bad cover {c!r}"[:200])
        out.append((c[0], c[1]))
    return out


def _field_types(raw: object, item: str, kind: str) -> dict[str, str]:
    """``{field: semantic type}``: an environment function's declared types, each from SEMANTIC_TYPES."""
    if raw is None:
        return {}
    if kind != "env_function":
        raise ManifestError(f"{item}: only environment functions declare field types")
    if not isinstance(raw, dict):
        raise ManifestError(f"{item}: field_types must be an object")
    for name, typ in raw.items():
        if (
            not isinstance(name, str)
            or not name
            or len(name) > _FIELD_NAME_MAX
            or "\n" in name
        ):
            raise ManifestError(f"{item}: bad field name in field_types")
        if not isinstance(typ, str) or typ not in SEMANTIC_TYPES:
            raise ManifestError(
                f"{item}: unknown semantic type {typ!r}; the fixed list is "
                f"{', '.join(SEMANTIC_TYPES)}"[:300],
            )
    return dict(raw)


def _input(raw: object, item: str, kind: str) -> str | None:
    """An environment function's declared input form, from INPUT_KINDS, or None when not given."""
    if raw is None:
        return None
    if kind != "env_function":
        raise ManifestError(f"{item}: only environment functions declare an input")
    if not isinstance(raw, str) or raw not in INPUT_KINDS:
        raise ManifestError(
            f"{item}: unknown input {raw!r}; the fixed list is {', '.join(INPUT_KINDS)}"[
                :300
            ],
        )
    return raw


def _parse_v21(manifest: dict) -> Manifest:
    """The memory v2.1 manifest (spec §4, §8.2): ``function`` items ``memory.<package>.<module>:<name>`` and
    ``note`` items ``notes/<topic>/<slug>.md``, tests under ``memory/<package>/tests/``, test helpers as
    ``support``, and covers that are action covers ``[episode, index]`` or typed covers (:mod:`.procedures`).
    The ``fixtures`` and ``tests_changed`` keys are read by the gate (:mod:`.gate_v21`).
    """
    from . import layout
    from .procedures import parse_cover

    patterns = {"function": layout.FUNCTION_ID, "note": layout.NOTE_ID}

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
        kind = raw.get("kind")
        if kind not in patterns:
            raise ManifestError(f"{item}: unknown or unsupported kind {kind!r}"[:200])
        if not patterns[kind].match(item):
            raise ManifestError(f"{item!r}: not a {kind} id"[:200])
        tests = tests_of(raw.get("tests"), f"{item}: tests")
        raw_covers = raw.get("covers")
        if raw_covers is not None and not isinstance(raw_covers, list):
            raise ManifestError(f"{item}: covers must be a list")
        covers: list[tuple[str, int]] = []
        typed: list = []
        for raw_cover in raw_covers or []:
            try:
                c = parse_cover(raw_cover)
            except ValueError as exc:
                raise ManifestError(f"{item}: {exc}"[:300]) from exc
            (covers if isinstance(c, tuple) else typed).append(c)
        if (covers or typed) and kind != "function":
            raise ManifestError(f"{item}: only functions cover recorded work")
        form = raw.get("input")
        if form is not None and (
            kind != "function" or not isinstance(form, str) or form not in INPUT_KINDS
        ):
            raise ManifestError(
                f"{item}: a function's input is one of {', '.join(INPUT_KINDS)}"[:300],
            )
        if raw.get("field_types") is not None:
            raise ManifestError(f"{item}: field_types is a v2 field")
        out.items.append(
            ManifestItem(
                item,
                kind,
                layout.item_path(item),
                _str_list(raw.get("source_episodes"), f"{item}: source_episodes"),
                tests,
                covers,
                {},
                form,
                typed,
            ),
        )
    out.support = [safe_rel(p) for p in _str_list(manifest.get("support"), "support")]
    for p in out.support:
        if layout.classify(p) != "test_helper":
            raise ManifestError(
                f"support file {p} is not a test helper under memory/<package>/tests/",
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
    # the library modules whose module-level code (imports, bindings, docstring) the pass changes (P6)
    out.skeleton = _str_list(manifest.get("skeleton"), "skeleton")
    for s in out.skeleton:
        if not re.fullmatch(r"memory\.[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)?", s):
            raise ManifestError(
                f"{s!r} is not a library module (memory.<package>[.<module>])"[:200],
            )
    return out


def parse_manifest(manifest: object, *, v21: bool = False) -> Manifest:
    """Validate the manifest's shape, its ids and its paths against the layout; raise ManifestError.

    *v21*: the memory v2.1 layout (:func:`_parse_v21`); off, v2's, unchanged.
    """
    if not isinstance(manifest, dict):
        raise ManifestError("the manifest must be an object")
    if v21:
        return _parse_v21(manifest)
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
        kind = raw.get("kind")
        if kind not in KINDS:
            raise ManifestError(f"{item}: unknown or unsupported kind {kind!r}"[:200])
        if not ITEM_ID[kind].match(item):
            raise ManifestError(f"{item!r}: not a {kind} id"[:200])
        tests = [safe_rel(t) for t in _str_list(raw.get("tests"), f"{item}: tests")]
        for t in tests:
            if not TEST_PATH.match(t) or not layout_allowed(t):
                raise ManifestError(f"{item}: test {t} is not under env/<name>/tests/")
        covers = _covers(raw.get("covers"), item)
        if covers and kind != "env_function":
            raise ManifestError(
                f"{item}: only environment functions cover recorded calls",
            )
        sources = _str_list(raw.get("source_episodes"), f"{item}: source_episodes")
        out.items.append(
            ManifestItem(
                item,
                kind,
                item_path(item),
                sources,
                tests,
                covers,
                _field_types(raw.get("field_types"), item, kind),
                _input(raw.get("input"), item, kind),
            ),
        )
    out.support = [safe_rel(p) for p in _str_list(manifest.get("support"), "support")]
    for p in out.support:
        if not support_allowed(p):
            raise ManifestError(
                f"support file {p} is not {TESTKIT} or an admissible helper under env/<name>/tests/",
            )
    out.deleted = _str_list(manifest.get("deleted"), "deleted")
    out.unlisted = _str_list(manifest.get("unlisted"), "unlisted")
    for entry in out.deleted + out.unlisted:
        if item_kind(entry) is None:
            raise ManifestError(f"{entry!r} is not an item id"[:200])
        if entry in seen:
            raise ManifestError(f"{entry} is both in items and deleted or unlisted")
    if set(out.deleted) & set(out.unlisted):
        raise ManifestError("an item cannot be both deleted and unlisted")
    out.deleted_tests = [
        safe_rel(t) for t in _str_list(manifest.get("deleted_tests"), "deleted_tests")
    ]
    for t in out.deleted_tests:
        if not TEST_PATH.match(t):
            raise ManifestError(f"deleted test {t} is not under env/<name>/tests/")
    out.skeleton = _str_list(manifest.get("skeleton"), "skeleton")
    for s in out.skeleton:
        if not SKELETON_ID.match(s):
            raise ManifestError(f"{s!r} is not a channel (env/<name>)"[:200])
    return out

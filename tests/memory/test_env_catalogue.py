"""The environment catalogue's calls, ``env.items`` and ``env.get``, as a cell runs them.

The library's ``env`` package carries them in its generated ``env/__init__.py`` (``unify/memory/env_helper.py`` byte
for byte), with the contract runtime beside the catalogue (``.memory/contract.py``). Here they are loaded from such a
copy exactly as a cell imports them, over a small catalogue written for the test, and checked for: the forms a
function's name may take, the errors (what was given, what was expected with an example call, the nearest names or
the right call), the old names (``memory.catalog``, ``memory.describe``) that still work and name the new calls, and
the printed text, which is the earlier catalogue's text with only the call names changed.
"""

from __future__ import annotations

import ast
import dataclasses
import importlib.util
import itertools
import json
import sys
from pathlib import Path

import pytest

from unify.common import contract_runtime
from unify.memory import catalogue, env_helper, memory_helper
from unify.memory.analysis import shapes as shapes_module

CATALOG = {
    "version": 1,
    "input_forms": {
        "path": "a path to a work-tree file",
        "observation": "a dialogue or tool observation value, as recorded",
    },
    "channels": [
        {"channel": "files", "functions": 2, "notes": 1, "summary": "Readers for the workspace's files."},
        {"channel": "dialogue", "functions": 1, "notes": 0, "summary": "", "suspect": True},
    ],
    "functions": [
        {
            "id": "env/files:read_rows",
            "channel": "files",
            "module": "env.files",
            "name": "read_rows",
            "signature": "read_rows(path: str) -> list",
            "summary": "Read the rows of a table.",
            "input": "path",
            "effect": "read",
            "tier": None,
            "doc": {"args": "path: the table (a path).", "example": ">>> read_rows('x.csv')"},
            "input_shapes": [{"kind": "value", "tree": {"a": "int"}, "lengths": {}}],
        },
        {
            "id": "env/files:parse",
            "channel": "files",
            "module": "env.files",
            "name": "parse",
            "signature": "parse(path: str) -> dict",
            "summary": "Parse a record file.",
            "input": "path",
            "effect": "read",
            "tier": None,
            "doc": {},
        },
        {
            "id": "env/dialogue:parse",
            "channel": "dialogue",
            "module": "env.dialogue",
            "name": "parse",
            "signature": "parse(obs: dict) -> int",
            "summary": "Read the score.",
            "input": "observation",
            "effect": "read",
            "tier": None,
            "doc": {},
        },
    ],
}

#: The earlier catalogue's footers (``memory_helper`` before the move), byte for byte; the new text is these with the
#: call names changed (and the README's place named from the package, not from the old helper file).
OLD_FOOTER = (
    "Import with `from env.<channel> import <function>`. `help(<function>)` or "
    '`memory.describe("env.<channel>.<function>")` shows a function\'s documentation and runnable '
    "example; `memory.find(value)` lists the functions built on inputs shaped like data you hold. Notes are "
    "in `env/<channel>/NOTES.md`; `README.md` beside this file holds the same list. Functions are "
    "candidates, not authority: check the example before relying on one. A MemoryInputError means the input "
    "differs from what the function was built from; then do the work directly. The library is a scratch "
    "copy: edits to it, or a new `proposals/<name>.md`, are discarded when the request ends but recorded for "
    "the next consolidation.\n"
)
OLD_CHANNEL_TAIL = (
    "Import with `from env.<channel> import <function>`; `help(<function>)` or "
    '`memory.describe("env.<channel>.<function>")` shows one with its example.\n'
)
FORMS = (
    "Input forms (what a function's first parameter takes): `path` a path to a work-tree file; "
    "`observation` a dialogue or tool observation value, as recorded.\n"
)


def renamed(old: str) -> str:
    """*old* catalogue text as the new calls print it."""
    return (
        old.replace("memory.catalog(", "env.items(")
        .replace("memory.describe(", "env.get(")
        .replace("`README.md` beside this file", "`README.md` in the library root")
    )


_seq = itertools.count()


def _write(root: Path, catalog: dict | None = CATALOG) -> Path:
    """A library copy holding the generated helpers byte for byte and *catalog* (None: no catalogue file)."""
    files = {
        "memory.py": Path(memory_helper.__file__).read_bytes(),
        ".memory/shapes.py": Path(shapes_module.__file__).read_bytes(),
        "env/__init__.py": Path(env_helper.__file__).read_bytes(),
        ".memory/contract.py": Path(contract_runtime.__file__).read_bytes(),
    }
    if catalog is not None:
        files[".memory/catalog.json"] = json.dumps(catalog, indent=1).encode()
    for rel, data in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(data)
    return root


def _env(root: Path):
    """``import env`` as a cell does it: the copy's ``env`` package (its ``__init__.py`` holds the calls)."""
    name = f"env_under_test_{next(_seq)}"
    spec = importlib.util.spec_from_file_location(
        name,
        root / "env" / "__init__.py",
        submodule_search_locations=[str(root / "env")],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # as an import does: its dataclasses look their module up by name
    spec.loader.exec_module(module)
    assert not module.__package__.startswith("unify")  # the copy's own branch: the runtime beside the catalogue
    return module


def _memory(root: Path):
    """``import memory`` as a cell does it in catalogue mode: the copy's top-level ``memory.py``."""
    spec = importlib.util.spec_from_file_location(f"memory_under_test_{next(_seq)}", root / "memory.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def lib(tmp_path):
    root = _write(tmp_path / "copy")
    return root, _env(root)


# --- the printed text is the earlier text --------------------------------------------------------------------------


def test_the_overview_prints_the_earlier_catalogue_with_the_new_call_names(lib):
    root, env = lib
    got = env.items()
    assert str(got) == (
        f"Memory library: 2 channels, 3 functions, at {root} (first on the import path).\n"
        "Channels:\n"
        "- `env.dialogue`: 1 function (suspect: the environment changed since these were built; verify before use)\n"
        "- `env.files`: 2 functions, 1 note. Readers for the workspace's files.\n"
        "Functions:\n"
        "- `env.dialogue.parse(obs: dict) -> int`: Read the score. (input: observation)\n"
        "- `env.files.read_rows(path: str) -> list`: Read the rows of a table. (input: path)\n"
        "- `env.files.parse(path: str) -> dict`: Parse a record file. (input: path)\n"
        + FORMS
        + renamed(OLD_FOOTER)
    )
    assert repr(got) == str(got)  # a cell's last expression shows the listing itself
    # the typed view: the whole catalogue as data, whatever the page prints
    assert type(got).__name__ == "Catalogue" and (got.channel, got.page, got.pages) == (None, 1, 1)
    assert [c.name for c in got.channels] == ["dialogue", "files"]
    assert [(c.functions, c.notes, c.suspect) for c in got.channels] == [(1, 0, True), (2, 1, False)]
    assert [f.name for f in got.functions] == ["env.dialogue.parse", "env.files.read_rows", "env.files.parse"]
    assert str(got.channels[1]) == "- `env.files`: 2 functions, 1 note. Readers for the workspace's files."
    with pytest.raises(dataclasses.FrozenInstanceError):
        got.page = 2


def test_a_channel_view_and_a_function_print_the_earlier_text(lib):
    root, env = lib
    view = env.items("files")
    assert str(view) == (
        f"Memory library, channel `env.files` (at {root}):\n"
        "- `env.files`: 2 functions, 1 note. Readers for the workspace's files.\n"
        "Functions:\n"
        "- `env.files.read_rows(path: str) -> list`: Read the rows of a table. (input: path)\n"
        "- `env.files.parse(path: str) -> dict`: Parse a record file. (input: path)\n"
        + FORMS
        + renamed(OLD_CHANNEL_TAIL)
    )
    assert view.channel == "files" and [f.name for f in view.functions] == ["env.files.read_rows", "env.files.parse"]
    fn = env.get("env.files.read_rows")
    assert str(fn) == repr(fn) == (
        "env.files.read_rows(path: str) -> list\n"
        "Read the rows of a table.\n"
        "\n"
        "Input: path (a path to a work-tree file)\n"
        "Effect: read\n"
        "Args:\n"
        "    path: the table (a path).\n"
        "Example:\n"
        "    >>> read_rows('x.csv')\n"
        "Built and checked on inputs shaped as:\n"
        "    - a value with keys a\n"
        f"Example paths are relative to the library root ({root}); check the example before relying on the "
        "function.\n"
    )
    assert (fn.name, fn.channel, fn.signature, fn.input, fn.effect, fn.suspect) == (
        "env.files.read_rows",
        "files",
        "read_rows(path: str) -> list",
        "path",
        "read",
        False,
    )
    assert "Channel env.dialogue is suspect" in str(env.get("env.dialogue.parse"))
    with pytest.raises(dataclasses.FrozenInstanceError):
        fn.name = "x"


def test_an_empty_library_says_so(tmp_path):
    env = _env(_write(tmp_path / "copy", {"version": 1, "input_forms": {}, "channels": [], "functions": []}))
    assert str(env.items()) == "Memory library: empty (no channel holds a function yet). Do the work directly.\n"
    with pytest.raises(LookupError, match=r"no function 'x': the library holds no function yet"):
        env.get("x")


# --- one function, by any name that names it -------------------------------------------------------------------


@pytest.mark.parametrize(
    "ref",
    [
        "env.files.read_rows",
        "files.read_rows",
        "env/files:read_rows",
        "files:read_rows",
        "env.files:read_rows",
        "read_rows",
        "READ_ROWS",
        "read_rows(path)",
        "`env.files.read_rows`",
        "  read_rows  ",
    ],
)
def test_every_form_of_a_functions_name_opens_it(lib, ref):
    _, env = lib
    assert env.get(ref).name == "env.files.read_rows"


def test_the_function_object_and_a_listed_function_open_it_too(lib):
    _, env = lib

    def read_rows(path):  # a stand-in for the imported function
        return path

    read_rows.__module__ = "env.files"
    assert env.get(read_rows).name == "env.files.read_rows"
    read_rows.__module__ = "somewhere_else"  # not imported from the package: by its own name
    assert env.get(read_rows).name == "env.files.read_rows"
    listed = env.items().functions[1]
    assert env.get(listed) == listed
    # the docstring's example runs: the first listed function
    assert env.get(env.items().functions[0]).name == "env.dialogue.parse"
    # a channel by any of its forms
    for form in ("files", "env.files", "env/files", "env/files/", "FILES", env.items().channels[1]):
        assert env.items(form).channel == "files", form


# --- errors that teach -----------------------------------------------------------------------------------------


def test_a_missing_function_names_the_nearest_and_how_to_browse(lib):
    _, env = lib
    with pytest.raises(LookupError) as info:
        env.get("read_rowz")
    err = info.value
    assert type(err).__name__ == "NotFound"
    first, expected, rest = str(err).split("\n")
    assert first == "env.get('read_rowz'): no function 'read_rowz' in the catalogue."
    assert expected == (
        "Expected a function's full name, <channel>.<function>, or its plain name when one channel has it, "
        "e.g. env.get('env.files.read_rows')."
    )
    assert rest.startswith("Nearest: env.files.read_rows") and rest.endswith(
        "env.items() lists every channel and function.",
    )
    assert err.nearest[0] == "env.files.read_rows" and err.call == "env.get('read_rowz')"


def test_a_name_several_channels_have_lists_each_full_name(lib):
    _, env = lib
    with pytest.raises(LookupError) as info:
        env.get("parse")
    assert type(info.value).__name__ == "Ambiguous"
    assert str(info.value) == (
        "env.get('parse'): 'parse' names 2 functions.\n"
        "Nearest: env.files.parse, env.dialogue.parse. Pass one of them, e.g. env.get('env.files.parse')."
    )
    assert env.get("dialogue.parse").name == "env.dialogue.parse"


def test_a_channel_given_to_get_and_a_function_given_to_items_name_the_right_call(lib):
    _, env = lib
    for ref in ("files", "env.files"):
        with pytest.raises(LookupError) as info:
            env.get(ref)
        assert type(info.value).__name__ == "WrongCall"
        assert str(info.value) == (
            f"env.get({ref!r}): that is a channel, not a function. List its functions with env.items('files')."
        )
    with pytest.raises(LookupError) as info:
        env.items("read_rows")
    assert type(info.value).__name__ == "WrongCall"
    assert str(info.value) == (
        "env.items('read_rows'): that is a function, not a channel. Open it with env.get('env.files.read_rows')."
    )


def test_a_missing_channel_and_a_missing_page_say_what_exists(lib):
    _, env = lib
    with pytest.raises(LookupError) as info:
        env.items("file")
    assert str(info.value) == (
        "env.items('file'): no channel 'file'.\n"
        "Expected a channel's name, e.g. env.items('files').\n"
        "Nearest: files, dialogue. env.items() lists every channel and function."
    )
    with pytest.raises(LookupError) as info:
        env.items("files", page=3)
    assert str(info.value) == (
        "env.items('files', page=3): no page 3; pages run from 1 to 1.\n"
        "Expected page: an int from 1 to 1, e.g. env.items('files', page=1)."
    )


def test_a_wrong_type_is_refused_before_anything_runs(lib):
    _, env = lib
    with pytest.raises(TypeError) as info:
        env.get(3)
    assert type(info.value).__name__ == "ArgumentError"
    text = str(info.value)
    assert text.startswith("env.get(3): ref is 3 (int).\nExpected ref: str | Function | ")
    assert text.endswith("e.g. print(env.get(env.items().functions[0])).")
    with pytest.raises(TypeError, match=r"page is '2' \(str\)\.\nExpected page: int, e\.g\. print\(env\.items\(\)\)\."):
        env.items(page="2")
    with pytest.raises(TypeError, match=r"got an unexpected keyword argument 'name'"):
        env.get("read_rows", name="x")
    with pytest.raises(TypeError, match=r"missing a required argument: 'ref'"):
        env.get()


def test_an_unreadable_catalogue_names_no_path(tmp_path):
    root = _write(tmp_path / "copy", None)
    env = _env(root)
    for call in (env.items, lambda: env.get("x")):
        with pytest.raises(LookupError) as info:
            call()
        text = str(info.value)
        assert "the library's catalogue cannot be read (FileNotFoundError)." in text
        assert str(root) not in text and str(tmp_path) not in text
        assert "help() on an imported function" in text


# --- the old names ---------------------------------------------------------------------------------------------


def test_the_old_names_work_and_print_one_line_naming_the_new_call(lib, capsys):
    root, env = lib
    memory = _memory(root)
    assert memory.catalog() == str(env.items())
    assert capsys.readouterr().out == (
        "memory.catalog() is now env.items(); import env to call it. The old name will be removed.\n"
    )
    assert memory.catalog("files", page=1) == str(env.items("files"))
    assert capsys.readouterr().out.startswith("memory.catalog('files') is now env.items('files');")
    assert memory.describe("read_rows") == str(env.get("read_rows"))
    assert capsys.readouterr().out == (
        "memory.describe('read_rows') is now env.get('read_rows'); import env to call it. "
        "The old name will be removed.\n"
    )

    def read_rows():
        pass

    read_rows.__module__ = "env.files"
    assert memory.describe(read_rows) == str(env.get("env.files.read_rows"))
    assert capsys.readouterr().out.startswith("memory.describe(read_rows) is now env.get(read_rows);")
    # the old names' results are text, as before (a cell may slice them)
    assert memory.catalog().splitlines()[0].startswith("Memory library: 2 channels")
    assert repr(memory.describe("read_rows")) == memory.describe("read_rows")
    # their errors are the new calls' own
    with pytest.raises(LookupError, match=r"^env\.get\('nope'\): no function 'nope'"):
        memory.describe("nope")
    with pytest.raises(LookupError, match=r"pages run from 1 to 1"):
        memory.catalog("files", page=0)


# --- the contract --------------------------------------------------------------------------------------------------


def test_each_call_is_checked_typed_and_documented():
    for fn in (env_helper.items, env_helper.get):
        assert getattr(fn, "__contract_checked__", False), fn.__name__
        doc = fn.__doc__ or ""
        lines = [ln.strip() for ln in doc.splitlines()]
        for header in ("Args:", "Returns:", "Raises:", "Example:"):
            assert header in lines, (fn.__name__, header)
        example = contract_runtime.example_of(fn)
        assert example and example.startswith("print(env.") and compile(example, "<example>", "eval")
        target = fn.__wrapped__
        hints = target.__annotations__
        params = [p for p in target.__code__.co_varnames[: target.__code__.co_argcount]]
        assert set(params) | {"return"} == set(hints), (fn.__name__, hints)
        for p in params:
            assert f"{p}: " in doc, (fn.__name__, p)


def test_every_error_the_calls_raise_is_a_contract_kind_with_its_parts():
    """Static: each ``raise`` in the module's calls is NotFound, Ambiguous, WrongCall or ArgumentError, with an expected
    form and an example, or the right call (``hint``); the one other raise is the import-time load of the runtime."""
    tree = ast.parse(Path(env_helper.__file__).read_text())
    kinds = {"NotFound", "Ambiguous", "WrongCall", "ArgumentError"}
    seen = 0
    for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
        for node in ast.walk(fn):
            if not isinstance(node, ast.Raise) or node.exc is None:
                continue
            call = node.exc
            assert isinstance(call, ast.Call) and isinstance(call.func, ast.Name), ast.unparse(node)
            if fn.name == "_load_contract":
                assert call.func.id == "ImportError"
                continue
            assert call.func.id in kinds, ast.unparse(node)
            keywords = {k.arg for k in call.keywords}
            assert {"expected", "example"} <= keywords or "hint" in keywords, ast.unparse(node)
            seen += 1
    assert seen >= 8


def test_the_generated_files_carry_the_calls_their_runtime_and_their_stub(tmp_path):
    tree = tmp_path / "lib"
    (tree / "env" / "x").mkdir(parents=True)
    (tree / "env" / "x" / "__init__.py").write_text('def f(a):\n    """F.\n\n    Input: path\n    """\n    return a\n')
    files = catalogue.generated(tree)
    assert files[catalogue.ENV_HELPER] == Path(env_helper.__file__).read_bytes()
    assert files[catalogue.CONTRACT] == Path(contract_runtime.__file__).read_bytes()
    stub = files[catalogue.ENV_STUB].decode()
    assert stub == env_helper.stub() and compile(stub, "<env stub>", "exec")
    assert "def items(channel: str | Channel | None = None, page: int = 1) -> Catalogue:" in stub
    assert "def get(ref: str | Function | Callable[..., Any]) -> Function:" in stub
    for cls in ("Catalogue", "Channel", "Function"):
        assert f"@dataclass(frozen=True)\nclass {cls}:" in stub, cls
    for rel in (catalogue.ENV_HELPER, catalogue.ENV_STUB, catalogue.CONTRACT):
        assert catalogue.reserved(rel), rel
    assert not catalogue.reserved("env/x/__init__.py")
    # written out, the copy's own env package serves the catalogue and still imports its channels
    catalogue.write_generated(tree)
    env = _env(tree)
    assert [f.name for f in env.items().functions] == ["env.x.f"]
    assert env.get("f").input == "path"
    channel = importlib.util.spec_from_file_location("env_channel_under_test", tree / "env" / "x" / "__init__.py")
    module = importlib.util.module_from_spec(channel)
    channel.loader.exec_module(module)
    assert module.f(1) == 1


def test_the_module_imports_only_the_standard_library_and_the_runtime():
    tree = ast.parse(Path(env_helper.__file__).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            names = [(node.module or "").split(".")[0]]
            if node.module == "unify.common.contract_runtime":  # the harness's copy of the runtime
                continue
        else:
            continue
        for name in names:
            assert name in ("__future__", "collections", "dataclasses", "importlib", "json", "os", "sys", "typing"), name

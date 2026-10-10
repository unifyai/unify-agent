"""Committed trees, read blob by blob (never through ``git archive``, whose ``.gitattributes`` can hide files).

:func:`listing` returns the exact committed file set with modes; :func:`materialise` writes exactly those
blobs and checks nothing else appeared; :func:`item_bodies` keys every memory item by id with the text that
defines it; :func:`module_skeleton`, :func:`notes_preamble` and :func:`without_listed` give the parts of a
file that belong to no item, so the gate can pin them; :func:`env_references` reads which library names a
test file imports and :func:`calls_item` whether it calls one; :func:`code_size` measures a channel
module's code and :func:`public_bindings` lists the public names it binds.
"""

from __future__ import annotations

import ast
import hashlib
import os
import stat
import subprocess
from pathlib import Path

from .gitio import _ENV, _HARD, GIT_TIMEOUT_S, GitError, Repo
from .manifest import ManifestError, safe_rel
from .memory_repo import _all_names, _front_matter, _sections, _slug


def _git_bytes(repo: Repo, args: list[str], input: bytes | None = None) -> bytes:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update(_ENV)
    try:
        proc = subprocess.run(
            ["git", *_HARD, "--git-dir", str(repo.git_dir), *args],
            input=input,
            env=env,
            capture_output=True,
            timeout=GIT_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired as exc:
        raise GitError(f"git {' '.join(args[:2])}… timed out") from exc
    if proc.returncode != 0:
        raise GitError(f"git {' '.join(args[:2])}… failed: {proc.stderr[:400]!r}")
    return proc.stdout


def listing(repo: Repo, sha: str) -> tuple[dict[str, tuple[str, str]], list[str]]:
    """``path -> (mode, blob id)`` for every plain file of *sha*, and the entries refused.

    Only mode 100644 is admitted: links, submodules and executable files are refused.
    """
    files: dict[str, tuple[str, str]] = {}
    refused: list[str] = []
    for rec in _git_bytes(repo, ["ls-tree", "-r", "-z", "--full-tree", sha]).split(
        b"\0",
    ):
        if not rec:
            continue
        meta, _, raw = rec.partition(b"\t")
        mode, kind, obj = meta.decode("ascii").split(" ")
        try:
            path = safe_rel(raw.decode("utf-8"))
        except (UnicodeDecodeError, ManifestError):
            refused.append(f"unsafe path {raw[:80]!r}")
            continue
        if kind != "blob" or mode != "100644":
            what = {
                "120000": "symlink",
                "100755": "executable file",
                "160000": "submodule",
            }
            refused.append(f"{what.get(mode, 'entry of mode ' + mode)} {path}")
            continue
        files[path] = (mode, obj)
    return files, refused


def blob_id(data: bytes, object_format: str = "sha1") -> str:
    """The git blob id of *data* (what ``git hash-object --no-filters`` prints), computed without git."""
    if object_format not in ("sha1", "sha256"):
        raise ValueError(f"unknown object format {object_format!r}")
    return hashlib.new(object_format, b"blob %d\0" % len(data) + data).hexdigest()


def tree_listing(
    tree: Path,
    object_format: str = "sha1",
) -> tuple[dict[str, tuple[str, str]], list[str]]:
    """:func:`listing` for an uncommitted directory: what committing *tree* would hold, without git.

    Walks with ``lstat`` and never follows a link. Directories are not entries (git tracks no empty
    directory); ``.git`` entries are skipped at any depth (the pass never mirrors them); a name that is
    not UTF-8 or not a safe path is refused, as :func:`listing` refuses it. A regular file with the
    owner's execute bit is an executable file (mode 100755), as git records it, and is refused like links
    and special files. Blob ids are computed in Python (:func:`blob_id`) in the repo's *object_format*.
    The commit adds ignored files too (``add --force``) and the gate refuses ``.gitignore``, so no ignore
    rule can make the two differ.
    """
    files: dict[str, tuple[str, str]] = {}
    refused: list[str] = []
    stack = [(Path(tree), "")]
    while stack:
        d, rel = stack.pop()
        with os.scandir(d) as it:
            entries = sorted(it, key=lambda e: e.name)
        for e in entries:
            if e.name == ".git":
                continue
            r = rel + e.name
            try:
                r.encode("utf-8")
                path = safe_rel(r)
            except (UnicodeEncodeError, ManifestError):
                refused.append(f"unsafe path {r[:80]!r}")
                continue
            st = os.lstat(e.path)
            if stat.S_ISDIR(st.st_mode):
                stack.append((Path(e.path), r + "/"))
            elif stat.S_ISLNK(st.st_mode):
                refused.append(f"symlink {path}")
            elif not stat.S_ISREG(st.st_mode):
                refused.append(f"special file {path}")
            elif st.st_mode & stat.S_IXUSR:
                refused.append(f"executable file {path}")
            else:
                fd = os.open(e.path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
                with os.fdopen(fd, "rb") as f:
                    files[path] = ("100644", blob_id(f.read(), object_format))
    return files, refused


def materialise(repo: Repo, files: dict[str, tuple[str, str]], dest: Path) -> Path:
    """Write exactly *files* (committed blobs) under *dest*, and check that nothing else is there."""
    dest.mkdir()
    ids = sorted({obj for _, obj in files.values()})
    blobs: dict[str, bytes] = {}
    if ids:
        data = _git_bytes(
            repo,
            ["cat-file", "--batch"],
            ("\n".join(ids) + "\n").encode(),
        )
        pos = 0
        while pos < len(data):
            nl = data.index(b"\n", pos)
            header = data[pos:nl].split(b" ")
            if len(header) != 3 or header[1] != b"blob":
                raise GitError(f"unexpected object header {data[pos:nl][:80]!r}")
            size = int(header[2])
            blobs[header[0].decode("ascii")] = data[nl + 1 : nl + 1 + size]
            pos = nl + 1 + size + 1
    for path, (_, obj) in files.items():
        target = dest / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(blobs[obj])
    written = {
        p.relative_to(dest).as_posix()
        for p in dest.rglob("*")
        if not p.is_dir() or p.is_symlink()
    }
    if written != set(files):
        raise GitError("the extracted tree differs from the committed tree")
    return dest


def item_bodies(tree: Path) -> dict[str, tuple[str, str, bool]]:
    """``item id -> (kind, body, listed)``; a function's body is its unparsed AST, a note its section text."""
    out: dict[str, tuple[str, str, bool]] = {}
    for mod in sorted(tree.glob("env/*/__init__.py")):
        try:
            module = ast.parse(mod.read_bytes())
        except (SyntaxError, ValueError):
            continue  # reported by items() under G6
        listed = _all_names(module)
        for node in module.body:
            if isinstance(
                node,
                (ast.FunctionDef, ast.AsyncFunctionDef),
            ) and not node.name.startswith("_"):
                out[f"env/{mod.parent.name}:{node.name}"] = (
                    "env_function",
                    ast.unparse(node),
                    listed is None or node.name in listed,
                )
    for notes in sorted(tree.glob("env/*/NOTES.md")):
        rel = notes.relative_to(tree).as_posix()
        for heading, body in _sections(notes.read_bytes().decode("utf-8", "replace")):
            key = f"{rel}#{_slug(heading)}"
            prev = out.get(key, ("", "", True))[1]
            out[key] = ("env_note", prev + f"## {heading}\n{body}\n", True)
    for wf in sorted(tree.glob("workflows/*.md")):
        text = wf.read_bytes().decode("utf-8", "replace")
        out[wf.relative_to(tree).as_posix()] = (
            "workflow",
            text,
            _front_matter(text).get("listed", "true") != "false",
        )
    return out


def _is_public_def(node: ast.stmt) -> bool:
    return isinstance(
        node,
        (ast.FunctionDef, ast.AsyncFunctionDef),
    ) and not node.name.startswith("_")


def _str_sequence(value: ast.expr | None) -> bool:
    return isinstance(value, (ast.List, ast.Tuple)) and all(
        isinstance(e, ast.Constant) and isinstance(e.value, str) for e in value.elts
    )


def _is_all(node: ast.stmt) -> bool:
    """Only ``__all__ = [...]`` (one target) or ``__all__: T = [...]`` with literal strings is exempt."""
    if isinstance(node, ast.Assign):
        return (
            len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "__all__"
            and _str_sequence(node.value)
        )
    if isinstance(node, ast.AnnAssign):
        return (
            isinstance(node.target, ast.Name)
            and node.target.id == "__all__"
            and _str_sequence(node.value)
        )
    return False


def duplicate_public_defs(source: bytes) -> list[str]:
    """Public top-level function names defined more than once (an earlier one is dead code)."""
    try:
        module = ast.parse(source)
    except (SyntaxError, ValueError):
        return []
    seen: set[str] = set()
    dups: set[str] = set()
    for node in module.body:
        if _is_public_def(node):
            (dups if node.name in seen else seen).add(node.name)
    return sorted(dups)


def module_skeleton(source: bytes | None) -> str:
    """A module minus its public functions and ``__all__``: docstring, imports, helpers, module-level code.

    A missing module has the empty skeleton; one that does not parse is its own bytes.
    """
    if source is None:
        return ast.dump(ast.Module(body=[], type_ignores=[]))
    try:
        module = ast.parse(source)
    except (SyntaxError, ValueError):
        return "unparsable:" + source.decode("utf-8", "replace")
    module.body = [n for n in module.body if not (_is_public_def(n) or _is_all(n))]
    return ast.dump(module)


def notes_preamble(text: str | None) -> str:
    """The text of a notes file before its first ``## `` section."""
    out: list[str] = []
    for line in (text or "").splitlines():
        if line.startswith("## "):
            break
        out.append(line)
    return "\n".join(out)


def without_listed(data: bytes) -> bytes:
    """A workflow note without the ``listed:`` line of its front matter."""
    if not data.startswith(b"---\n"):
        return data
    end = data.find(b"\n---", 4)
    if end < 0:
        return data
    front = [
        ln for ln in data[4:end].split(b"\n") if not ln.strip().startswith(b"listed:")
    ]
    return b"---\n" + b"\n".join(front) + data[end:]


def env_references(source: bytes) -> tuple[set[str], set[str], bool] | None:
    """What a test file imports from the library: item ids, whole channels, and whether all of ``env``.

    None when the file does not parse.
    """
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return None
    names: set[str] = set()
    channels: set[str] = set()
    everything = False
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            parts = node.module.split(".")
            if parts[0] != "env":
                continue
            if len(parts) == 1:
                channels.update(a.name for a in node.names if a.name != "*")
                everything = everything or any(a.name == "*" for a in node.names)
            elif len(parts) == 2 and all(a.name != "*" for a in node.names):
                names.update(f"env/{parts[1]}:{a.name}" for a in node.names)
            else:
                channels.add(parts[1])
        elif isinstance(node, ast.Import):
            for a in node.names:
                parts = a.name.split(".")
                if parts[0] == "env":
                    if len(parts) == 1:
                        everything = True
                    else:
                        channels.add(parts[1])
    return names, channels, everything


def _dotted(node: ast.expr) -> str | None:
    """``a.b.c`` for a chain of names and attributes, else None."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return ".".join(reversed(parts))


def calls_item(source: bytes, item: str) -> bool:
    """Whether a test file calls the environment function *item* (``env/<channel>:<name>``).

    A call counts only when its callee resolves to the function imported from its module: a name bound by
    ``from env.<channel> import <name>`` (under its alias, if any) or by ``from env.<channel> import *``, or
    the attribute ``<name>`` of the module bound by ``import env.<channel>`` (``env.<channel>.<name>``),
    ``import env.<channel> as m`` or ``from env import <channel> [as m]`` (``m.<name>``). Importing the
    function, or its channel, without calling it does not count; nor does an unparsable file. Rebinding a
    name after import is not tracked (ruling R16: a careless consolidator, not a malicious one).
    """
    channel, _, name = item.partition(":")
    channel = channel.split("/", 1)[1] if "/" in channel else channel
    if not channel or not name:
        return False
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return False
    module = f"env.{channel}"
    names: set[str] = set()  # local names bound to the function
    modules: set[str] = set()  # dotted local names bound to its module
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 0:
            if node.module == module:
                for a in node.names:
                    if a.name == "*":
                        names.add(name)
                    elif a.name == name:
                        names.add(a.asname or a.name)
            elif node.module == "env":
                modules.update(
                    a.asname or a.name for a in node.names if a.name == channel
                )
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name == module:
                    modules.add(a.asname or module)
                elif a.name == "env" and a.asname is None:
                    modules.add(module)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if isinstance(f, ast.Name) and f.id in names:
            return True
        if (
            isinstance(f, ast.Attribute)
            and f.attr == name
            and _dotted(f.value) in modules
        ):
            return True
    return False


def public_bindings(source: bytes | None) -> dict[str, bool] | None:
    """The public names a channel module binds at its top level: ``name -> whether it is a function definition``.

    A ``def`` (or ``async def``), an assignment to a plain name (``A = B``, ``A: T = B``) and an import
    (``import x as A``, ``from x import y as A``; ``import a.b`` binds ``a``) each bind a name; names starting
    with ``_`` are not public. A name stays a function definition only if every binding of it is a ``def``.
    A missing module binds nothing; None when it does not parse.
    """
    if source is None:
        return {}
    try:
        module = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return None
    out: dict[str, bool] = {}

    def bind(name: str, is_def: bool) -> None:
        if not name.startswith("_"):
            out[name] = out.get(name, True) and is_def

    for node in module.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            bind(node.name, True)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    bind(t.id, False)
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            if isinstance(node.target, ast.Name):
                bind(node.target.id, False)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                if a.name != "*":
                    bind(a.asname or a.name.split(".")[0], False)
    return out


def code_size(source: bytes) -> tuple[int, int, int] | None:
    """A channel module's code size: (function definitions, AST nodes in its top-level function definitions,
    AST nodes of the rest of the module), None when it does not parse.

    Comments, blank lines and formatting are not in the AST; docstrings (of the module, its classes and its
    functions) are left out, so neither changes the size.
    """
    try:
        module = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return None
    docs: set[int] = set()
    for node in ast.walk(module):
        body = getattr(node, "body", None)
        if (
            isinstance(
                node,
                (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef),
            )
            and body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            docs.add(id(body[0]))

    def nodes(root: ast.AST) -> int:
        n, stack = 0, [root]
        while stack:
            node = stack.pop()
            if id(node) in docs:
                continue
            n += 1
            stack.extend(ast.iter_child_nodes(node))
        return n

    defs = sum(
        1
        for n in ast.walk(module)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    )
    functions = other = 0
    for node in module.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions += nodes(node)
        else:
            other += nodes(node)
    return defs, functions, other

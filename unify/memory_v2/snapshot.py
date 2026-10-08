"""Committed trees, read blob by blob (never through ``git archive``, whose ``.gitattributes`` can hide files).

:func:`listing` returns the exact committed file set with modes; :func:`materialise` writes exactly those
blobs and checks nothing else appeared; :func:`item_bodies` keys every memory item by id with the text that
defines it; :func:`module_skeleton`, :func:`notes_preamble` and :func:`without_listed` give the parts of a
file that belong to no item, so the gate can pin them; :func:`env_references` reads which library names a
test file imports.
"""

from __future__ import annotations

import ast
import os
import subprocess
from pathlib import Path

from .gitio import _ENV, GitError, Repo
from .manifest import ManifestError, safe_rel
from .memory_repo import _all_names, _front_matter, _sections, _slug


def _git_bytes(repo: Repo, args: list[str], input: bytes | None = None) -> bytes:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update(_ENV)
    proc = subprocess.run(
        ["git", "--git-dir", str(repo.git_dir), *args],
        input=input,
        env=env,
        capture_output=True,
    )
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

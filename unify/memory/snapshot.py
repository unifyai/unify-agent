"""Committed trees, read blob by blob (never through ``git archive``, whose ``.gitattributes`` can hide files).

:func:`listing` returns the exact committed file set with modes; :func:`materialise` writes exactly those
blobs and checks nothing else appeared; :func:`item_bodies` keys every memory item by id with the text that
defines it; :func:`public_bindings` lists the public names a module binds.
"""

from __future__ import annotations

import ast
import hashlib
import os
import stat
import subprocess
from pathlib import Path

from .gitio import _ENV, _HARD, GIT_TIMEOUT_S, GitError, Repo, git_base_env
from .manifest import ManifestError, safe_rel


def _git_bytes(repo: Repo, args: list[str], input: bytes | None = None) -> bytes:
    env = git_base_env()
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



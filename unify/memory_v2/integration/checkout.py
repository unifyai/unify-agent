"""The request's scratch export of memory main (spec §G1, v0 D13): written freely, diffed, discarded.

The export is the committed blobs of one memory commit and nothing else: no ``.git`` (a writable ``.git``
must never steer host git, ruling R21), and no ``git archive`` (whose ``.gitattributes`` export rules can
hide or rewrite files). Diffs run with an explicit ``--git-dir`` and a temporary index, so nothing the
request wrote into the export is read as git configuration.

The harness adds its generated catalogue to the export after the commit's files (:mod:`..catalogue`);
:func:`checkout_diff` leaves out each generated file the request left byte-for-byte as written, so an
untouched export still diffs empty, and records one the request changed.
"""

from __future__ import annotations

import logging
import os
import shutil
import stat
import tempfile
from pathlib import Path

from ..blobs import BlobStore
from ..gitio import GitError, Repo
from ..manifest import compiled_artifact, shadows_import
from ..snapshot import listing, materialise
from .hardgit import git

logger = logging.getLogger(__name__)

MEMORY_DIFF_CAP = 256 * 1024
_EXCLUDE = (":(exclude,glob)**/__pycache__/**", ":(exclude,glob)**/*.pyc")


def _clear(dest: Path) -> None:
    if dest.is_symlink() or dest.is_file():
        dest.unlink()
    elif dest.exists():
        shutil.rmtree(dest)


def export_checkout(memory_dir: Path, sha: str, dest: Path) -> None:
    """Replace *dest* with the files of memory commit *sha*.

    Bytecode, compiled extensions and root entries that could shadow an import
    (:func:`..manifest.compiled_artifact`, :func:`..manifest.shadows_import`) are never exported: the gate
    refuses them now, and one an older commit holds must not be importable from the cell. A merged library
    holds none, so its export is exactly its files.
    """
    dest = Path(dest)
    _clear(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    files, refused = listing(Repo(Path(memory_dir)), sha)
    if refused:
        raise GitError(
            f"memory {sha[:12]} holds entries an export refuses: {refused[:5]}",
        )
    unsafe = _unexported(files)
    if unsafe:
        logger.warning(
            "memory v2: %d importable artefact(s) of memory %s left out of the export: %s",
            len(unsafe),
            sha[:12],
            unsafe[:5],
        )
        drop = set(unsafe)
        files = {p: v for p, v in files.items() if p not in drop}
    materialise(Repo(Path(memory_dir)), files, dest)


def _unexported(files: dict) -> list[str]:
    """The paths of a commit's listing that :func:`export_checkout` leaves out, sorted."""
    return sorted(p for p in files if compiled_artifact(p) or shadows_import(p))


def remove_checkout(dest: Path) -> None:
    _clear(Path(dest))


def _unchanged(dest: Path, rel: str, data: bytes) -> bool:
    """Whether *dest*/*rel* is a regular file (never followed) holding exactly *data*."""
    try:
        fd = os.open(dest / rel, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError:
        return False
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or st.st_size != len(data):
            return False
        with os.fdopen(os.dup(fd), "rb") as fh:
            return fh.read(len(data) + 1) == data
    finally:
        os.close(fd)


def checkout_diff(
    memory_dir: Path,
    base: str,
    dest: Path,
    blobs: BlobStore,
    cap: int = MEMORY_DIFF_CAP,
    generated: dict[str, bytes] | None = None,
) -> str:
    """What the request wrote into its export, as a binary diff against *base*; capped, the rest in a blob.

    *generated* (relative path -> the bytes the harness wrote) names the export's generated files; each one
    still holding exactly those bytes is left out of the diff.
    """
    excludes = tuple(
        f":(exclude,literal){rel}"
        for rel, data in sorted((generated or {}).items())
        if _unchanged(Path(dest), rel, data)
    )
    # what the export left out of the commit is not something the request deleted
    excludes += tuple(
        f":(exclude,literal){rel}"
        for rel in _unexported(listing(Repo(Path(memory_dir)), base)[0])
    )
    with tempfile.TemporaryDirectory(prefix="memv2-idx-") as tmp:
        env = {"GIT_INDEX_FILE": str(Path(tmp) / "index")}
        git(memory_dir, "read-tree", base, env=env, cwd=Path(tmp))
        git(
            memory_dir,
            "add",
            "-A",
            "-f",
            "--",
            ".",
            *_EXCLUDE,
            *excludes,
            work_tree=dest,
            env=env,
        )
        raw = git(
            memory_dir,
            "diff",
            "--cached",
            "--binary",
            "--no-ext-diff",
            "--no-textconv",
            base,
            work_tree=dest,
            env=env,
        )
    text = raw.decode("utf-8", "replace")
    if len(raw) <= cap:
        return text
    sha = blobs.put(raw)
    excerpt = raw[:cap].decode("utf-8", "ignore")
    return (
        excerpt
        + f"\n[memory.diff truncated at {cap} of {len(raw)} bytes; blob {sha}]\n"
    )

"""The request's scratch export of memory main (spec §G1, v0 D13): written freely, diffed, discarded.

The export is the committed blobs of one memory commit and nothing else: no ``.git`` (a writable ``.git``
must never steer host git, ruling R21), and no ``git archive`` (whose ``.gitattributes`` export rules can
hide or rewrite files). Diffs run with an explicit ``--git-dir`` and a temporary index, so nothing the
request wrote into the export is read as git configuration.

The harness adds its generated catalogue to the export after the commit's files (:mod:`..catalogue`);
:func:`checkout_diff` leaves out each generated file the request left byte-for-byte as written, so an
untouched export still diffs empty, and records one the request changed.

When the exported library's tests use the test kit (:mod:`..testkit`: they import ``memlab`` or name a recorded
blob), the export also holds the kit at ``.memlab/`` (``memlab``, the pin plugin, the blobs the tests name;
put on the worker's import path after the export, :func:`.hooks.worker_paths`), so the tests the working model
reads or runs import as they do in the gate and in Sol's box. The kit is harness-owned (the gate's layout rules
admit no ``.memlab`` in a library) and left out of the memory diff.
"""

from __future__ import annotations

import functools
import logging
import os
import shutil
import stat
import tempfile
from pathlib import Path

from .. import testkit
from ..blobs import BlobStore
from ..gitio import GitError, Repo
from ..manifest import unsafe_path
from ..snapshot import listing, materialise
from .hardgit import git

logger = logging.getLogger(__name__)

MEMORY_DIFF_CAP = 256 * 1024
_EXCLUDE = (
    ":(exclude,glob)**/__pycache__/**",
    ":(exclude,glob)**/*.pyc",
    f":(exclude,glob){testkit.EXPORT_DIR}/**",
)


def _writable_retry(func, path, exc, *, root: Path) -> None:
    """``shutil.rmtree``'s error handler: a read-only v2.1 export (:func:`make_read_only`) gets its write bits
    back where removal needs them. Any error other than a permission error is raised as before.

    Bits are restored only on real directories inside *root*, the export being removed: a path outside it, or
    one reached through a link, keeps its error."""
    if not isinstance(exc, PermissionError):
        raise exc
    parent = os.path.abspath(os.path.dirname(path) or ".")
    rel = os.path.relpath(parent, os.path.abspath(root))
    if (
        rel == ".."
        or rel.startswith(".." + os.sep)
        or os.path.realpath(
            parent,
        )
        != os.path.normpath(os.path.join(os.path.realpath(root), rel))
    ):
        raise exc
    os.chmod(parent, stat.S_IRWXU)
    if stat.S_ISDIR(os.lstat(path).st_mode):
        os.chmod(path, stat.S_IRWXU)
    func(path)


def _clear(dest: Path) -> None:
    if dest.is_symlink() or dest.is_file():
        dest.unlink()
    elif dest.exists():
        shutil.rmtree(dest, onexc=functools.partial(_writable_retry, root=dest))


def make_read_only(root: Path) -> None:
    """Drop every write bit under *root*: files 0444, directories 0555, deepest first, links never followed."""
    for dirpath, dirnames, filenames in os.walk(root, topdown=False):
        for name in filenames:
            p = os.path.join(dirpath, name)
            if not os.path.islink(p):
                os.chmod(p, 0o444)
        for name in dirnames:
            p = os.path.join(dirpath, name)
            if not os.path.islink(p):
                os.chmod(p, 0o555)
    os.chmod(root, 0o555)


def export_actor_v21(
    memory_dir: Path,
    sha: str,
    dest: Path,
    *,
    status_of=None,
    shapes=None,
    records=None,
) -> dict[str, bytes]:
    """``UNIFY_MEMORY_V21=on``: replace *dest* with the actor's read-only copy of memory commit *sha* (spec
    v2.1 §6). It holds:

    - the commit's library modules, package docstrings and notes only (:func:`..layout.exported`): never a
      tests directory, a fixture, compiled code or a path outside the layout;
    - the generated files (:func:`..library_export.generated_v21`), which replace any committed copy.

    Every write bit is then dropped, and the sandbox binds the copy read-only (:func:`.hooks.worker_readonly_mounts`).
    Returns the generated files.
    """
    from ..catalogue import write_files
    from ..layout import exported
    from ..library_export import generated_v21, item_history

    dest = Path(dest)
    _clear(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    repo = Repo(Path(memory_dir))
    files, refused = listing(repo, sha)
    if refused:
        raise GitError(
            f"memory {sha[:12]} holds entries an export refuses: {refused[:5]}",
        )
    materialise(repo, {p: v for p, v in files.items() if exported(p)}, dest)
    generated = generated_v21(
        dest,
        status_of=status_of,
        shapes=shapes,
        history=item_history(repo, sha),
        records=records,
    )
    write_files(dest, generated)
    make_read_only(dest)
    return generated


def export_checkout(
    memory_dir: Path,
    sha: str,
    dest: Path,
    blobs: BlobStore | None = None,
) -> None:
    """Replace *dest* with the files of memory commit *sha*, and the test kit at ``.memlab/`` when the
    library's tests use it (*blobs*: the recorded blobs they may name; None: no kit).

    Paths the gate refuses before extraction (:func:`..manifest.unsafe_path`: compiled code, start-up hooks,
    root entries outside the layout) are never exported: one an older commit holds must not be importable
    from the cell. A merged library holds none, so its export is exactly its files.
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
            "memory v2: %d path(s) the gate refuses in memory %s left out of the export: %s",
            len(unsafe),
            sha[:12],
            unsafe[:5],
        )
        drop = set(unsafe)
        files = {p: v for p, v in files.items() if p not in drop}
    materialise(Repo(Path(memory_dir)), files, dest)
    if blobs is not None:
        testkit.stage_for_tree(
            dest,
            dest / testkit.EXPORT_DIR,
            has_blob=blobs.has,
            read_blob=blobs.get,
            blob_size=blobs.size,
        )


def _unexported(files: dict) -> list[str]:
    """The paths of a commit's listing that :func:`export_checkout` leaves out, sorted."""
    return sorted(p for p in files if unsafe_path(p) is not None)


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

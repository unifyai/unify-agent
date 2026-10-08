"""The request's scratch export of memory main (spec §G1, v0 D13): written freely, diffed, discarded.

The export is the committed blobs of one memory commit and nothing else: no ``.git`` (a writable ``.git``
must never steer host git, ruling R21), and no ``git archive`` (whose ``.gitattributes`` export rules can
hide or rewrite files). Diffs run with an explicit ``--git-dir`` and a temporary index, so nothing the
request wrote into the export is read as git configuration.

When the exported library's tests use the test kit (:mod:`..testkit`: they import ``memlab`` or name a recorded
blob), the export also holds the kit at ``.memlab/`` (``memlab``, the pin plugin, the blobs the tests name;
put on the worker's import path after the export, :func:`.hooks.worker_paths`), so the tests the working model
reads or runs import as they do in the gate and in Sol's box. The kit is harness-owned (the gate's layout rules
admit no ``.memlab`` in a library) and left out of the memory diff.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from .. import testkit
from ..blobs import BlobStore
from ..gitio import GitError, Repo
from ..snapshot import listing, materialise
from .hardgit import git

MEMORY_DIFF_CAP = 256 * 1024
_EXCLUDE = (
    ":(exclude,glob)**/__pycache__/**",
    ":(exclude,glob)**/*.pyc",
    f":(exclude,glob){testkit.EXPORT_DIR}/**",
)


def _clear(dest: Path) -> None:
    if dest.is_symlink() or dest.is_file():
        dest.unlink()
    elif dest.exists():
        shutil.rmtree(dest)


def export_checkout(
    memory_dir: Path,
    sha: str,
    dest: Path,
    blobs: BlobStore | None = None,
) -> None:
    """Replace *dest* with exactly the files of memory commit *sha*, and the test kit at ``.memlab/`` when
    the library's tests use it (*blobs*: the recorded blobs they may name; None: no kit).
    """
    dest = Path(dest)
    _clear(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    files, refused = listing(Repo(Path(memory_dir)), sha)
    if refused:
        raise GitError(
            f"memory {sha[:12]} holds entries an export refuses: {refused[:5]}",
        )
    materialise(Repo(Path(memory_dir)), files, dest)
    if blobs is not None:
        testkit.stage_for_tree(
            dest,
            dest / testkit.EXPORT_DIR,
            has_blob=blobs.has,
            read_blob=blobs.get,
            blob_size=blobs.size,
        )


def remove_checkout(dest: Path) -> None:
    _clear(Path(dest))


def checkout_diff(
    memory_dir: Path,
    base: str,
    dest: Path,
    blobs: BlobStore,
    cap: int = MEMORY_DIFF_CAP,
) -> str:
    """What the request wrote into its export, as a binary diff against *base*; capped, the rest in a blob."""
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

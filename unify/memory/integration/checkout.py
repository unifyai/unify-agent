"""The request's read-only export of memory main (:func:`export_actor_library_mode`).

The export is the committed blobs of one memory commit that the actor may read (:func:`..layout.exported`) and
nothing else: no ``.git`` (a writable ``.git`` must never steer host git), and no ``git archive`` (whose
``.gitattributes`` export rules can hide or rewrite files). The harness adds the library's generated files after the
commit's files (:mod:`..library_export`), then drops every write bit; the sandbox binds the copy read-only.
"""

from __future__ import annotations

import functools
import os
import shutil
import stat
from pathlib import Path

from ..gitio import GitError, Repo
from ..snapshot import listing, materialise


def _writable_retry(func, path, exc, *, root: Path) -> None:
    """``shutil.rmtree``'s error handler: a read-only library-mode export (:func:`make_read_only`) gets its write bits
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


def export_actor_library_mode(
    memory_dir: Path,
    sha: str,
    dest: Path,
    *,
    status_of=None,
    shapes=None,
    records=None,
    captures=None,
    summary_of=None,
    search=None,
    serving=None,
) -> dict[str, bytes]:
    """Replace *dest* with the actor's read-only copy of memory commit *sha*. It holds:

    - the commit's library modules, package docstrings and notes only (:func:`..layout.exported`): never a
      tests directory, a fixture, compiled code or a path outside the layout;
    - the generated files (:func:`..library_export.generated_library_mode`), which replace any committed copy;
    - with *captures* (:func:`..captures.export_files`: paths under ``.memory/captures/``), a snapshot of the capture
      store as this request starts. Captures never enter ``INDEX.md`` or the prompt;
    - the librarian's catalogue (with *summary_of*'s episode summaries and the captures' cards last), the dispatch
      file and the search index (its vectors made with *search*, a :class:`..library_search.Backend`; None: a
      lexical index), and ``memory.ask`` in the package.

    Every write bit is then dropped, and the sandbox binds the copy read-only (:func:`.hooks.worker_readonly_mounts`).
    Returns the generated files.

    *serving*: called with the materialised copy before its files are generated, it gives the serving evidence and the withdrawn library items (:mod:`..serving`) the generated files
    carry (:func:`..library_export.generated_library_mode`); None: as without it.
    """
    from ..catalogue import write_files
    from ..layout import exported
    from ..library_export import generated_library_mode, item_history

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

    def committed_test(rel: str) -> bytes | None:
        """A file of the commit by path (the export holds no tests): to re-check each example call it quotes."""
        if rel not in files:
            return None
        try:
            return repo.show(sha, rel)
        except GitError:
            return None

    served = serving(dest) if serving is not None else None
    generated = generated_library_mode(
        dest,
        status_of=status_of,
        shapes=shapes,
        history=item_history(repo, sha),
        records=records,
        tests=committed_test,
        librarian=True,
        summary_of=summary_of,
        captures=captures,
        search=search,
        **({"serving": served} if served is not None else {}),
    )
    if captures:
        from ..captures import EXPORT_DIR

        generated = {
            **generated,
            **{p: v for p, v in captures.items() if p.startswith(EXPORT_DIR + "/")},
        }
    write_files(dest, generated)
    make_read_only(dest)
    return generated


def remove_checkout(dest: Path) -> None:
    _clear(Path(dest))

"""The generated catalogue of a memory export (v2.1 surfacing): README, catalog and the ``memory`` helper.

Memory is a library the working model uses like a well-kept internal Python package. With every export the
harness writes, beside the committed files and never into a commit:

* ``README.md``: the channels and, per function, its signature, one-line summary and input form;
* ``.memory/catalog.json``: per channel its function and note counts, its module's summary and, when the
  harness holds the channel suspect (drift), ``"suspect": true``; per function its module, name, signature,
  docstring sections, input form, effect, a ``tier`` field reserved for promotion (always null for now),
  and the recorded input-shape signatures of the inputs the gate admitted it on (``input_shapes``; left out
  when none were recorded). ``memory.catalog()`` prints the channels from it, so the system prompt names
  none of this;
* ``memory.py`` (importable as ``memory``) and ``.memory/shapes.py``: :mod:`.memory_helper` and
  :mod:`.analysis.shapes`, copied byte for byte, so the cell's ``memory.find`` computes shapes with the
  same functions the gate recorded them with.

Everything except ``input_shapes`` and the suspect flags (the harness's drift state, which changes only
between requests) is a pure function of the exported tree. **Where the shapes come from.**
The input-shape descriptors of each function's validated covers (:func:`.memory_helper.file_shape` of each
covered file's recorded blob, :func:`.memory_helper.value_shape` of each covered observation) are kept as a
snapshot per memory commit, written once (:mod:`.shape_rows`): a landed merge writes its commit's, and an
export of a commit without one (a commit from before snapshots, or a hide commit) derives it and freezes it.
A row is used only while its body digest (:func:`body_digest`) matches the exported function, so the same
commit gives the same bytes on every export, an old one re-exported included. Shapes derived from the
covers table for a function no merge recorded shapes for are marked ``input_shapes_backfilled``.

Sol never edits these files: the gate refuses any commit that touches a :func:`reserved` path, and Sol's
copy of the library never holds them.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import shutil
from collections.abc import Callable, Iterable
from pathlib import Path

from . import docstrings
from .manifest import INPUT_KINDS, compiled_artifact, shadows_import
from .memory_helper import channel_line
from .memory_repo import items
from .snapshot import item_bodies

README = "README.md"
HELPER = "memory.py"
CATALOG = ".memory/catalog.json"
SHAPES = ".memory/shapes.py"
GENERATED = (README, HELPER, CATALOG, SHAPES)
CATALOG_VERSION = 1
# The soft size of the catalogue (README plus the channel lines ``memory.catalog()`` prints), in estimated
# tokens. Past it the gate notes that hygiene is due (UNIFY_MEMORY_V2_SOFT_BUDGET=on); it never refuses growth.
SOFT_BUDGET_TOKENS = 4000
_MODULE_SUMMARY_CHARS = 120
_HERE = Path(__file__).resolve().parent

# (item id, body digest) -> (input shapes, whether they were backfilled from covers), or None
ShapeLookup = Callable[[str, str], "tuple[list[dict], bool] | None"]


def reserved(path: str) -> bool:
    """Whether *path* (relative, POSIX) is one the harness generates in every export, a root entry that could
    shadow an import (:func:`.manifest.shadows_import`: ``memory.*``, ``env.*``, ``memory/``, a standard or
    installed module name), or bytecode or a compiled extension anywhere (:func:`.manifest.compiled_artifact`).
    Refused in every mode (a declared safety fix: no library needs one)."""
    if path in (README, HELPER) or path == ".memory" or path.startswith(".memory/"):
        return True
    return shadows_import(path) or compiled_artifact(path)


def body_digest(body: str) -> str:
    """The key of a function's recorded input shapes: SHA-256 of its body (:func:`.snapshot.item_bodies`)."""
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / 4)


def _module_doc(path: Path) -> str:
    try:
        tree = ast.parse(path.read_bytes())
    except (SyntaxError, ValueError, OSError):
        return ""
    doc = (ast.get_docstring(tree) or "").strip()
    first = doc.splitlines()[0].strip() if doc else ""
    return first[:_MODULE_SUMMARY_CHARS]


def _docstrings(path: Path) -> dict[str, tuple[str, list[str]]]:
    """name -> (docstring, parameters) of a module's public functions."""
    try:
        tree = ast.parse(path.read_bytes())
    except (SyntaxError, ValueError, OSError):
        return {}
    return {
        n.name: (ast.get_docstring(n) or "", docstrings.params(n))
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def functions(tree: Path, shapes: ShapeLookup | None = None) -> list[dict]:
    """The listed environment functions of *tree*, by channel then module order, as the catalogue holds them."""
    tree = Path(tree)
    bodies = item_bodies(tree) if shapes is not None else {}
    docs: dict[str, dict[str, tuple[str, list[str]]]] = {}
    out: list[dict] = []
    for it in items(tree).items:
        if it.kind != "env_function" or not it.listed:
            continue
        channel = it.path.split("/")[1]
        if channel not in docs:
            docs[channel] = _docstrings(tree / it.path)
        doc, _ = docs[channel].get(it.name, ("", []))
        entry: dict = {
            "id": it.item_id,
            "channel": channel,
            "module": f"env.{channel}",
            "name": it.name,
            "signature": it.signature,
            "summary": it.doc,
            "input": it.input if it.input in INPUT_KINDS else "",
            "effect": it.effect,
            "tier": None,  # reserved: the promotion stage marks stable and experimental functions
            "doc": docstrings.as_catalog(docstrings.parse(doc)),
        }
        if shapes is not None and it.item_id in bodies:
            found = shapes(it.item_id, body_digest(bodies[it.item_id][1]))
            if found and found[0]:
                entry["input_shapes"] = found[0]
                if found[1]:
                    entry["input_shapes_backfilled"] = True
        out.append(entry)
    out.sort(key=lambda e: e["channel"])  # stable: module order within a channel
    return out


def channels(tree: Path) -> list[dict]:
    """Per channel with a listed function or note: its function and note counts and its module's summary."""
    tree = Path(tree)
    counts: dict[str, list[int]] = {}
    for it in items(tree).items:
        if it.kind == "workflow" or not it.listed:
            continue
        ch = it.path.split("/")[1]
        counts.setdefault(ch, [0, 0])[0 if it.kind == "env_function" else 1] += 1
    return [
        {
            "channel": ch,
            "functions": n_fn,
            "notes": n_notes,
            "summary": _module_doc(tree / "env" / ch / "__init__.py"),
        }
        for ch, (n_fn, n_notes) in sorted(counts.items())
    ]


def channel_lines(tree: Path, suspect: Iterable[str] = ()) -> str:
    """One line per channel, sorted; ``""`` when the library lists nothing."""
    flagged = set(suspect)
    rows = channels(tree)
    return "".join(channel_line(r, r["channel"] in flagged) + "\n" for r in rows)


def render_readme(tree: Path) -> str:
    """``README.md``: a pure function of the tree (no commit id, time or harness state)."""
    tree = Path(tree)
    rows = channels(tree)
    by_channel: dict[str, list[dict]] = {}
    for e in functions(tree):
        by_channel.setdefault(e["channel"], []).append(e)
    notes: dict[str, list[tuple[str, str]]] = {}
    for it in items(tree).items:
        if it.kind == "env_note" and it.listed:
            notes.setdefault(it.path.split("/")[1], []).append((it.name, it.doc))
    out = [
        "# Memory library\n\n",
        "Generated by the harness from this library's commit; never edit it (edits are discarded).\n",
        "Each channel is a Python module: `from env.<channel> import <function>`. `help(<function>)` or\n",
        "`memory.describe(<function>)` shows a function's documentation with a runnable example, and\n",
        "`memory.find(value)` (after `import memory`) lists the functions whose recorded inputs have the\n",
        "shape of a value you hold. Input forms (what a function's first parameter takes): "
        + "; ".join(f"`{k}` {v}" for k, v in INPUT_KINDS.items())
        + ".\n",
    ]
    for row in rows:
        ch = row["channel"]
        out.append(f"\n## env.{ch}\n\n")
        if row["summary"]:
            out.append(row["summary"] + "\n\n")
        for e in by_channel.get(ch, []):
            line = f"- `{e['signature']}`"
            if e["summary"]:
                line += f": {e['summary']}"
            if e["input"]:
                line += f" (input: {e['input']})"
            out.append(line + "\n")
        for heading, first in notes.get(ch, []):
            out.append(
                f"- note `NOTES.md` {heading}" + (f": {first}" if first else "") + "\n",
            )
    return "".join(out)


# Sol's first message carries the README while it fits this many estimated tokens (M9); past it, a compact
# view (channel lines and one line per function) cut to the same budget.
SOL_README_BUDGET_TOKENS = 2000
_COMPACT_SUMMARY_CHARS = 100


def readme_for_sol(tree: Path, budget_tokens: int = SOL_README_BUDGET_TOKENS) -> str:
    """The library part of Sol's first message under ``catalogue``: the README, or past *budget_tokens*
    the channel lines and one ``env.<channel>.<function>: <summary>`` line per function, as many as fit,
    with how to read the rest (``help()`` in the box, or the module's source). Bounded however large the
    library grows."""
    readme = render_readme(tree)
    if estimate_tokens(readme) <= budget_tokens:
        return (
            "Current library (its README, which the harness generates; never write it):\n"
            + readme
        )
    rows, fns = channels(tree), functions(tree)
    out = [
        f"Current library: {len(fns)} functions in {len(rows)} channels. Its README (generated by the "
        f"harness; never write it) is {estimate_tokens(readme)} tokens, over this message's budget of "
        f"{budget_tokens}, so this is a compact view. For a function's signature, arguments and example, "
        "run help(env.<channel>.<function>) with execute_code, or read /memory/env/<channel>/__init__.py.\n",
    ]
    used = estimate_tokens(out[0])

    def add(lines: list[str], what: str) -> bool:
        nonlocal used
        for n, line in enumerate(lines):
            if used + estimate_tokens(line) > budget_tokens - 20:
                out.append(
                    f"- ... and {len(lines) - n} more {what} (help(env.<channel>) lists them)\n",
                )
                return False
            out.append(line)
            used += estimate_tokens(line)
        return True

    def short(text: str) -> str:
        return (
            text
            if len(text) <= _COMPACT_SUMMARY_CHARS
            else text[: _COMPACT_SUMMARY_CHARS - 1] + "…"
        )

    if add(["Channels:\n"] + [channel_line(r) + "\n" for r in rows], "channels"):
        add(["Functions:\n"], "functions")
        add(
            [
                f"- {e['module']}.{e['name']}"
                + (f": {short(e['summary'])}" if e["summary"] else "")
                + "\n"
                for e in fns
            ],
            "functions",
        )
    return "".join(out)


def render_catalog(
    tree: Path,
    shapes: ShapeLookup | None = None,
    suspect: Iterable[str] = (),
) -> str:
    """``.memory/catalog.json``: sorted keys, one trailing newline. A channel in *suspect* carries
    ``"suspect": true``; no other channel has the key."""
    flagged = set(suspect)
    data = {
        "version": CATALOG_VERSION,
        "input_forms": dict(INPUT_KINDS),
        "channels": [
            dict(row, suspect=True) if row["channel"] in flagged else row
            for row in channels(tree)
        ],
        "functions": functions(tree, shapes),
    }
    return json.dumps(data, sort_keys=True, indent=1, ensure_ascii=False) + "\n"


def catalogue_tokens(tree: Path) -> int:
    """The catalogue's size for G4's soft budget: the README plus the channel lines (what
    ``memory.catalog()`` prints of the channels; the prompt carries neither)."""
    return estimate_tokens(render_readme(tree)) + estimate_tokens(channel_lines(tree))


def generated(
    tree: Path,
    shapes: ShapeLookup | None = None,
    suspect: Iterable[str] = (),
) -> dict[str, bytes]:
    """Every generated file of the export of *tree*, by relative path."""
    return {
        README: render_readme(tree).encode("utf-8"),
        CATALOG: render_catalog(tree, shapes, suspect).encode("utf-8"),
        HELPER: (_HERE / "memory_helper.py").read_bytes(),
        SHAPES: (_HERE / "analysis" / "shapes.py").read_bytes(),
    }


def write_generated(
    checkout: Path,
    shapes: ShapeLookup | None = None,
    suspect: Iterable[str] = (),
) -> dict[str, bytes]:
    """Write the generated files into the export at *checkout*, replacing whatever is there; returns them.
    The channels in *suspect* (the harness's drift state) are flagged in the catalog.

    A path the export already holds (a commit from before the paths were reserved) is replaced: the
    generated file wins. Nothing is followed: an existing link or directory at a generated path is removed.
    """
    checkout = Path(checkout)
    files = generated(checkout, shapes, suspect)
    for rel, data in files.items():
        dest = checkout / rel
        parent = dest.parent
        if parent.is_symlink() or (parent.exists() and not parent.is_dir()):
            parent.unlink()
        parent.mkdir(parents=True, exist_ok=True)
        if dest.is_symlink() or dest.is_file():
            dest.unlink()
        elif dest.is_dir():
            shutil.rmtree(dest)
        fd = os.open(dest, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
    return files

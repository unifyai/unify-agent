"""Per-item admission (stage 7): a refused candidate reduced to the items the gate did not refuse.

When every failure of a pass belongs to manifest items (:attr:`.gate._Run.item_fail`, never a pass-wide
failure), :meth:`.gate.Gate.merge` refuses those items alone. This module works out what else goes with
them and writes the reduced tree; the gate commits it as a child of the parent and runs the whole gate on
it again (one round, no search over subsets).

What goes with a refused item (:func:`with_dependents`, a closure over the manifest's items):

* an item that shares a test file with it (the file cannot be split);
* an environment function that calls it, directly or through the module's private helpers (by name, in
  the candidate's module: ``ast``, never text);
* an item whose test files import it by name (``from env.<channel> import <name>``).

What the reduction does to each refused item (:func:`build`), from the materialised candidate tree:

* an environment function the parent did not have is removed (its ``def`` and its ``__all__`` entry,
  :func:`.memory_repo.remove_function`); one the parent had gets the parent's ``def`` back, in place;
  the rest of the module is kept (a private helper only it used stays, as unused code);
* a workflow note, and every test file the item lists, is restored to the parent's version, or removed
  when the parent had none;
* a notes section is restored to the parent's sections of that slug, or dropped;
* a channel the parent did not have and that no admitted item uses is removed whole, and its ``skeleton``
  and ``support`` entries leave the manifest.

No reduction is made (the pass stays refused whole) when nothing would be admitted, or when a refused
function the parent had sits in a channel whose skeleton the pass changes (the restored function would
have to stay listed with covers the gate refused).
"""

from __future__ import annotations

import ast
import shutil
from pathlib import Path

from .manifest import Manifest, ManifestItem
from .memory_repo import _slug, remove_function
from .snapshot import env_references

DEPENDENCY = "dependency"


def _channel(it: ManifestItem) -> str:
    """``env/<channel>`` of an item (a workflow's is its own path)."""
    return "/".join(it.path.split("/")[:2]) if it.path.startswith("env/") else it.path


def _defs(module: ast.Module) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    return {
        n.name: n
        for n in module.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _called(tree: Path, it: ManifestItem) -> set[str]:
    """The module's other public functions *it* reaches by name, directly or through private helpers."""
    try:
        defs = _defs(ast.parse((tree / it.path).read_bytes()))
    except (OSError, SyntaxError, ValueError):
        return set()
    name = it.item.split(":", 1)[1]
    seen, todo, out = {name}, [name], set()
    while todo:
        node = defs.get(todo.pop())
        if node is None:
            continue
        for sub in ast.walk(node):
            ref = sub.id if isinstance(sub, ast.Name) else None
            if ref is None or ref in seen or ref not in defs:
                continue
            seen.add(ref)
            if ref.startswith("_"):
                todo.append(ref)
            else:
                out.add(ref)
    prefix = it.item.split(":", 1)[0]
    return {f"{prefix}:{n}" for n in out}


def uses(tree: Path, it: ManifestItem) -> set[str]:
    """Item ids *it* depends on in the candidate *tree*: functions it calls and names its tests import."""
    out = _called(tree, it) if it.kind == "env_function" else set()
    for t in it.tests:
        try:
            refs = env_references((tree / t).read_bytes())
        except OSError:
            continue
        if refs is not None:
            out |= refs[0]
    out.discard(it.item)
    return out


def with_dependents(
    man: Manifest,
    failed: dict[str, list[str]],
    tree: Path,
) -> tuple[dict[str, list[str]], list[str]]:
    """Every refused item with its codes (the failed ones' checks, ``dependency`` for the rest) and why."""
    refused = {i: list(c) for i, c in failed.items()}
    reasons: list[str] = []
    by_id = {it.item: it for it in man.items}
    needs = {it.item: uses(tree, it) for it in man.items}
    grew = True
    while grew:  # at most one pass per item
        grew = False
        for it in man.items:
            if it.item in refused:
                continue
            for r in sorted(refused):
                shared = set(it.tests) & set(by_id[r].tests) if r in by_id else set()
                if r in needs[it.item] or shared:
                    refused[it.item] = [DEPENDENCY]
                    how = "shares a test file with" if shared else "uses"
                    reasons.append(f"{it.item} is refused with {r}, which it {how}")
                    grew = True
                    break
    return refused, reasons


def _segment(source: str, name: str) -> tuple[int, int] | None:
    """The 0-based line span ``[start, end)`` of top-level function *name*, decorators included."""
    node = _defs(ast.parse(source)).get(name)
    if node is None:
        return None
    start = min([node.lineno] + [d.lineno for d in node.decorator_list]) - 1
    return start, node.end_lineno


def _restore_function(dest: Path, parent: Path, it: ManifestItem, had: bool) -> None:
    mod = dest / it.path
    if not mod.is_file():
        return  # nothing of it to take out; the second gate run judges the rest
    name = it.item.split(":", 1)[1]
    source = mod.read_text()
    if not had:
        try:
            mod.write_text(remove_function(source, name))
        except KeyError:
            pass
        return
    old = (parent / it.path).read_text()
    span_old = _segment(old, name)
    if span_old is None:
        return
    seg = old.splitlines(keepends=True)[span_old[0] : span_old[1]]
    if seg and not seg[-1].endswith("\n"):
        seg[-1] += "\n"
    lines = source.splitlines(keepends=True)
    span = _segment(source, name)
    if span is None:
        if lines and not lines[-1].endswith("\n"):
            lines[-1] += "\n"
        mod.write_text("".join(lines + ["\n\n"] + seg))
    else:
        mod.write_text("".join(lines[: span[0]] + seg + lines[span[1] :]))


def _restore_file(dest: Path, parent: Path, rel: str, had: bool) -> None:
    target = dest / rel
    if had:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(parent / rel, target)
    elif target.is_file():
        target.unlink()


def _split_notes(text: str) -> tuple[list[str], list[tuple[str, list[str]]]]:
    """A notes file as its preamble lines and its ``## `` sections (heading, lines with the heading)."""
    pre: list[str] = []
    secs: list[tuple[str, list[str]]] = []
    for line in text.splitlines(keepends=True):
        if line.startswith("## "):
            secs.append((line[3:].strip(), [line]))
        elif secs:
            secs[-1][1].append(line)
        else:
            pre.append(line)
    for _, body in secs:
        if body and not body[-1].endswith("\n"):
            body[-1] += "\n"
    return pre, secs


def _restore_section(dest: Path, parent: Path, it: ManifestItem, had: bool) -> None:
    target = dest / it.path
    slug = it.item.split("#", 1)[1]
    pre, secs = _split_notes(target.read_text() if target.is_file() else "")
    old = _split_notes((parent / it.path).read_text())[1] if had else []
    back = [b for h, b in old if _slug(h) == slug]
    out: list[list[str]] = []
    placed = False
    for heading, body in secs:
        if _slug(heading) == slug:
            if not placed:
                out += back
                placed = True
            continue
        out.append(body)
    if not placed:
        out += back
    if pre and out and not pre[-1].endswith("\n"):
        pre[-1] += "\n"
    text = "".join(pre) + "".join(line for body in out for line in body)
    if not text.strip() and not had:
        if target.is_file():
            target.unlink()
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)


def build(
    man: Manifest,
    raw: dict,
    refused: dict[str, list[str]],
    parent_files: set[str],
    parent_functions: set[str],
    parent_tree: Path,
    cand_tree: Path,
    dest: Path,
) -> dict | None:
    """Write the reduced tree at *dest* and return the reduced manifest, or None when there is none.

    *parent_files* are the parent's paths and *parent_functions* its environment function ids.
    """
    kept = [it for it in man.items if it.item not in refused]
    gone = [it for it in man.items if it.item in refused]
    if not kept:
        return None
    for it in gone:
        if (
            it.kind == "env_function"
            and it.item in parent_functions
            and _channel(it) in man.skeleton
        ):
            return None
    shutil.copytree(cand_tree, dest)
    for it in gone:
        if it.kind == "env_function":
            _restore_function(
                dest,
                parent_tree,
                it,
                it.item in parent_functions,
            )
        elif it.kind == "env_note":
            _restore_section(dest, parent_tree, it, it.path in parent_files)
        else:
            _restore_file(dest, parent_tree, it.path, it.path in parent_files)
        for t in it.tests:
            _restore_file(dest, parent_tree, t, t in parent_files)
    needed = {_channel(it) for it in kept} | {
        "/".join(t.split("/")[:2]) for it in kept for t in it.tests
    }
    removed = sorted(
        ch
        for ch in {_channel(it) for it in gone if it.path.startswith("env/")}
        if ch not in needed and not any(p.startswith(ch + "/") for p in parent_files)
    )
    for ch in removed:
        shutil.rmtree(dest / ch, ignore_errors=True)
    out = dict(raw)
    keep_ids = {it.item for it in kept}
    out["items"] = [r for r in raw.get("items", []) if r.get("item") in keep_ids]
    if isinstance(raw.get("skeleton"), list):
        out["skeleton"] = [s for s in raw["skeleton"] if s not in removed]
    if isinstance(raw.get("support"), list):
        out["support"] = [
            p
            for p in raw["support"]
            if not any(p.startswith(ch + "/") for ch in removed)
        ]
    return out

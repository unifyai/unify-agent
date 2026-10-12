"""The memory library's helpers: ``memory.index``, ``memory.show`` and ``memory.find``.

The harness generates this file as ``memory/__init__.py`` in every copy of the library; edits are discarded.
It imports only the standard library and reads only files beside the package: ``INDEX.md``, ``links.json``
and the harness's data under ``.memory/``. In a cell::

    import memory
    print(memory.index())                          # the whole index
    print(memory.index("<package>"))               # one package's section ("notes/<topic>" for notes)
    print(memory.show("memory.<package>.<module>:<name>"))   # source, links, status, records, history
    memory.find(value)                             # functions built on recorded inputs shaped like value

``find`` matches structured data only: a dict, a list of records, JSON text, or a structured file such as a
CSV with a header. It compares key paths and types with the shapes of the inputs each function was admitted
on, never words, so it cannot match plain text.

Imported as the package ``memory``, it also wraps each public function of every library module as that
module is imported (:class:`_CallFinder`), so that each call raises the audit event ``memory.call`` with
the item id, a short digest of the arguments (never their values) and the outcome: ``ok``, ``refused``
(a ``MemoryInputError`` left it) or ``error`` (any other exception). The wrapper returns what the function
returns and re-raises what it raises, unchanged; whatever audit hooks the process has decide what is kept.
"""

from __future__ import annotations

import ast
import functools
import hashlib
import importlib.machinery
import importlib.util
import inspect
import json
import os
import sys
from typing import Any, NamedTuple

INDEX_FILE = "INDEX.md"
LINKS_FILE = "links.json"
ITEMS_FILE = ".memory/items.json"
FIND_FILE = ".memory/find.json"
MATCHER_FILE = ".memory/matcher.py"
SHAPES_FILE = ".memory/shapes.py"
ITEM_LINE = "- `"
CALL_EVENT = "memory.call"
CALL_OUTCOMES = ("ok", "refused", "error")
REFUSAL = "MemoryInputError"
DIGEST_CHARS = 16


def split_index(text: str) -> tuple[str, list[tuple[str, str]]]:
    """An index as its head (before the first ``## `` heading) and its sections (heading, body)."""
    head: list[str] = []
    sections: list[list[str]] = []
    for line in text.splitlines(keepends=True):
        if line.startswith("## "):
            sections.append([line[3:].rstrip("\n"), ""])
        elif sections:
            sections[-1][1] += line
        else:
            head.append(line)
    return "".join(head), [(h, b) for h, b in sections]


def section_key(heading: str) -> str:
    """``memory.<package>`` or ``notes/<topic>``: a heading without its summary."""
    return heading.split(" — ", 1)[0].strip()


def wanted_key(name: object) -> str:
    """The section a name means: ``"office"`` and ``"memory/office"`` both mean ``memory.office``."""
    n = str(name).strip().strip("/")
    if n.startswith(("memory.", "notes/")):
        return n
    if n.startswith("memory/"):
        return "memory." + n[len("memory/") :].replace("/", ".")
    return "memory." + n


def find_section(text: str, name: object) -> str:
    """The full section of the index *text* that *name* means; LookupError naming the sections otherwise."""
    key = wanted_key(name)
    _, sections = split_index(text)
    for heading, body in sections:
        if section_key(heading) == key:
            return f"## {heading}\n{body}"
    keys = ", ".join(section_key(h) for h, _ in sections) or "(none)"
    raise LookupError(f"memory: no index section {key!r}; the sections are: {keys}")


def item_count(body: str) -> int:
    """The item lines of a section body."""
    return sum(1 for line in body.splitlines() if line.startswith(ITEM_LINE))


__all__ = ["Found", "find", "index", "show"]

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(
    _PKG_DIR,
)  # the copy's root: INDEX.md, links.json, memory/, notes/, .memory/
_LEVELS = {2: "exact", 1: "structure", 0: "weak"}
_MATCHER: Any = None


class _Text(str):
    """Text that shows as itself when a cell's last expression evaluates to it."""

    def __repr__(self) -> str:
        return str(self)


class Found(NamedTuple):
    """One function :func:`find` lists: its id, how to call it and how its inputs matched."""

    name: str  # memory.<package>.<module>:<name>
    signature: str
    input: str
    match: str  # exact | structure | weak (a single named key)
    summary: str
    reason: str  # the keys or columns that matched


def _read(rel: str) -> str:
    path = os.path.join(_ROOT, rel)
    try:
        with open(path, encoding="utf-8") as fh:
            return fh.read()
    except OSError as exc:
        raise RuntimeError(
            f"memory: {path} cannot be read ({type(exc).__name__})",
        ) from exc


def _json(rel: str) -> dict:
    return json.loads(_read(rel))


def index(package: str | None = None) -> str:
    """The whole index (``INDEX.md``), or the full section of one package (``"<package>"``,
    ``"memory.<package>"``) or notes topic (``"notes/<topic>"``)."""
    text = _read(INDEX_FILE)
    return _Text(text if package is None else find_section(text, package))


def _item_id(item: Any) -> str:
    if callable(item) and not isinstance(item, str):
        return f"{getattr(item, '__module__', '')}:{getattr(item, '__name__', '')}"
    s = str(item).strip()
    if s.startswith("notes/") or ":" in s:
        return s
    if s.startswith("memory.") and s.count(".") >= 3:
        module, _, name = s.rpartition(".")
        return f"{module}:{name}"
    return s


def _source(path: str, name: str) -> str:
    text = _read(path)
    for node in ast.parse(text).body:
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == name
        ):
            start = min([node.lineno] + [d.lineno for d in node.decorator_list])
            return "\n".join(text.splitlines()[start - 1 : node.end_lineno])
    raise LookupError(f"memory: {name} is not defined in {path}")


def _summary(record: Any) -> str:
    if not record:
        return "no record yet"
    return ", ".join(f"{k} {record[k]}" for k in sorted(record))


def show(item: Any) -> str:
    """One item in full: a function's signature, with an example call quoted from its tests (``Example (from
    <test id>): <call>``) when one is recorded and still in its test file, then its source with its docstring;
    or a note's text. Then its links both ways, its status with, for a function, its effect (and why, for an
    item left out of the index or given a reason), its verification and use records, and the
    subjects of the library commits that changed it, newest first."""
    wanted = _item_id(item)
    data = _json(ITEMS_FILE)
    row = data.get("items", {}).get(wanted)
    if row is None:
        raise LookupError(
            f"memory: no item {wanted!r}; memory.index() lists the items (a function is "
            "memory.<package>.<module>:<name>, a note notes/<topic>/<slug>.md)",
        )
    links = _json(LINKS_FILE).get("items", {}).get(wanted, {})
    status = row.get("status", "experimental")
    if row["kind"] == "function":
        effect = row.get("effect")
        out = [
            f"{row['signature']}  [{status}"
            + (f", effect: {effect}" if effect else "")
            + "]",
        ]
        example = row.get("example")
        if isinstance(example, dict) and example.get("call"):
            out.append(f"Example (from {example.get('test')}): {example['call']}")
    else:
        out = [
            f"{wanted}  [{status}]" + (f"  {row['title']}" if row.get("title") else ""),
        ]
    if status in ("suspect", "deprecated"):
        out.append(
            f"Not in the index: {row.get('status_reason') or 'its status is ' + status}.",
        )
    elif row.get("status_reason"):
        out.append(f"Status: {row['status_reason']}.")
    out.append("")
    if row["kind"] == "function":
        out.append(_source(row["path"], wanted.split(":", 1)[1]))
    else:
        out.append(_read(wanted).rstrip("\n"))
    out += [
        "",
        "Links to: " + (", ".join(links.get("links_to", [])) or "none"),
        "Linked from: " + (", ".join(links.get("linked_from", [])) or "none"),
        "Verification: " + _summary(row.get("verification")),
        "Use: " + _summary(row.get("use")),
    ]
    history = row.get("history") or []
    if history:
        out.append("History (newest first):")
        out += [f"  {h}" for h in history]
        if row.get("history_more"):
            out.append(f"  … and {row['history_more']} earlier change(s)")
    else:
        out.append("History: none recorded")
    if not data.get("history_complete", True):
        out.append(
            f"(history covers the newest {data.get('history_scanned')} library commits only)",
        )
    return _Text("\n".join(out) + "\n")


def _matcher() -> Any:
    """The shape matcher shipped beside the package (``.memory/matcher.py``, v2's own matching code). Its
    ``__file__`` is set at the root, so it loads its shape functions from ``.memory/shapes.py``.
    """
    global _MATCHER
    if _MATCHER is None:
        path = os.path.join(_ROOT, MATCHER_FILE)
        spec = importlib.util.spec_from_loader("_memory_matcher", loader=None)
        module = importlib.util.module_from_spec(spec)
        module.__file__ = os.path.join(_ROOT, "matcher.py")
        with open(path, "rb") as fh:
            code = compile(fh.read(), path, "exec")
        exec(code, module.__dict__)  # noqa: S102 - the harness's own generated file
        _MATCHER = module
    return _MATCHER


def _candidates(m: Any, value: Any, exts: list[str]) -> list[dict]:
    """The descriptors *value* can be seen as (v2's ``memory_helper._value_candidates``, unchanged)."""
    if isinstance(value, os.PathLike):
        value = os.fspath(value)
    if (
        isinstance(value, str)
        and value
        and len(value) <= m._MAX_PATH_CHARS
        and "\n" not in value
        and "\0" not in value
        and os.path.isfile(value)
    ):
        with open(value, "rb") as fh:
            data = fh.read(m._shapes.PARSE_LIMIT + 1)
        desc = m.file_shape(value, data)
        return [desc] if desc is not None else []
    out: list[dict] = []
    if isinstance(value, bytearray):
        value = bytes(value)
    data = value.encode("utf-8") if isinstance(value, str) else value
    if isinstance(data, bytes):
        for ext in exts:
            desc = m.file_shape("value" + ext, data[: m._shapes.PARSE_LIMIT + 1])
            if desc is not None:
                out.append(desc)
    desc = m.value_shape(value)
    if desc is not None:
        out.append(desc)
    return out


def find(value: Any) -> list[Found]:
    """The functions whose recorded inputs have the shape of *value*, best match first: v2's input-shape
    matcher and ranking, unchanged (level, then share of named keys matched, then how many recorded shapes
    match, then name; at most five). An empty list means that no listed function was built on data of this
    shape, or that the value has too little structure (plain text has none)."""
    m = _matcher()
    entries = _json(FIND_FILE).get("functions", [])
    exts = sorted(
        {
            d.get("ext", "")
            for e in entries
            for d in e.get("input_shapes", [])
            if d.get("kind") == "file"
        },
    )
    mine = _candidates(m, value, exts)
    ranked: list[tuple[int, float, int, str, Found]] = []
    for e in entries:
        best = None
        count = 0
        for d in e.get("input_shapes", []):
            hits = [x for x in (m.match(v, d) for v in mine) if x is not None]
            if not hits:
                continue
            top = max(hits, key=lambda x: (x[0], x[1]))
            if best is None or top[:2] > best[:2]:
                best, count = top, 1
            elif top[:2] == best[:2]:
                count += 1
        if best is None:
            continue
        level, share, matched = best
        shown = [p[3:] if p.startswith("[].") else p for p in matched]
        reason = (
            "matched "
            + ", ".join(shown[:8])
            + (f" and {len(shown) - 8} more" if len(shown) > 8 else "")
        )
        name = e["id"]
        found = Found(
            name,
            e.get("signature", ""),
            e.get("input", ""),
            _LEVELS[level],
            e.get("summary", ""),
            reason,
        )
        ranked.append((-level, -share, -count, name, found))
    return [f for *_, f in sorted(ranked, key=lambda r: r[:4])][: m.MAX_RESULTS]


# --- runtime call evidence ------------------------------------------------------------------------------
#
# Static call sites in cell code say what the actor wrote, not what ran. The wrapper below lets the process
# that runs the cells observe each call that actually happened and how it ended. It changes no behaviour:
# the wrapped function receives the same arguments, its return value and exceptions pass through unchanged,
# and a failing audit hook is ignored.

_DIGEST_DEPTH = 3
_DIGEST_ITEMS = 16
_DIGEST_TEXT = 64
_DIGEST_SET = 256
_ATOMS = (type(None), bool, int, float, complex)


def _shape_text(value: Any, depth: int = 0) -> str:
    """A bounded text of *value* built without running any of its own code (no ``__repr__``, ``__len__`` or
    ``__iter__`` of user types): plain values by type and content prefix, containers by type, length and
    their first items (a set's sorted, small sets only, since its order differs between processes),
    anything else by its type name only."""
    t = type(value)
    if t in _ATOMS:
        if t is int and value.bit_length() > 256:
            return f"int:{value.bit_length()}bits"
        return f"{t.__name__}:{value!r}"
    if t is str or t is bytes:
        return f"{t.__name__}[{len(value)}]:{value[:_DIGEST_TEXT]!r}"
    if t not in (list, tuple, set, frozenset, dict):
        return f"<{t.__module__}.{t.__qualname__}>"
    head = f"{t.__name__}[{len(value)}]"
    if depth >= _DIGEST_DEPTH:
        return head
    if t is dict:
        parts = [
            f"{_shape_text(k, depth + 1)}={_shape_text(v, depth + 1)}"
            for k, v in list(dict.items(value))[:_DIGEST_ITEMS]
        ]
    elif t is list or t is tuple:
        parts = [_shape_text(v, depth + 1) for v in value[:_DIGEST_ITEMS]]
    elif len(value) <= _DIGEST_SET:
        parts = sorted(_shape_text(v, depth + 1) for v in value)[:_DIGEST_ITEMS]
    else:
        return head
    return f"{head}({','.join(parts)})"


def args_digest(args: tuple, kwargs: dict) -> str:
    """A short sha256 of the call's arguments as :func:`_shape_text` sees them: equal arguments give equal
    digests, and no argument value appears in it."""
    try:
        text = (
            _shape_text(tuple(args)) + "|" + _shape_text(dict(sorted(kwargs.items())))
        )
    except Exception:  # noqa: BLE001 - a digest never fails a call
        text = "unreadable"
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()[
        :DIGEST_CHARS
    ]


def call_outcome(exc: BaseException) -> str:
    """``refused`` when *exc* is a ``MemoryInputError`` (any class of that name in its type's bases: each
    library module may define its own), else ``error``."""
    return (
        "refused" if any(c.__name__ == REFUSAL for c in type(exc).__mro__) else "error"
    )


def _emit(item: str, digest: str, outcome: str) -> None:
    try:
        sys.audit(CALL_EVENT, item, digest, outcome)
    except Exception:  # noqa: BLE001 - an audit hook never changes the call
        pass


def wrap_call(fn: Any, item: str) -> Any:
    """*fn* wrapped to raise one ``memory.call`` event per call (module docstring). A call interrupted by
    something that is not an ``Exception`` (an interrupt, an exit, a cancellation) did not end on its own,
    so it raises no event."""
    if getattr(fn, "__memory_item__", None) is not None:
        return fn
    if inspect.iscoroutinefunction(fn):

        @functools.wraps(fn)
        async def call(*args: Any, **kwargs: Any) -> Any:
            digest = args_digest(args, kwargs)
            try:
                result = await fn(*args, **kwargs)
            except Exception as exc:
                _emit(item, digest, call_outcome(exc))
                raise
            _emit(item, digest, "ok")
            return result

    else:

        @functools.wraps(fn)
        def call(*args: Any, **kwargs: Any) -> Any:
            digest = args_digest(args, kwargs)
            try:
                result = fn(*args, **kwargs)
            except Exception as exc:
                _emit(item, digest, call_outcome(exc))
                raise
            _emit(item, digest, "ok")
            return result

    call.__memory_item__ = item  # type: ignore[attr-defined]
    return call


def wrap_module(module: Any) -> None:
    """Wrap each public function *module* defines itself (not one it imports, which its own module wraps;
    not a class or other callable). A library test module is left as it is."""
    name = module.__name__
    if "tests" in name.split(".")[1:]:
        return
    for attr, value in list(vars(module).items()):
        if (
            not attr.startswith("_")
            and inspect.isfunction(value)
            and getattr(value, "__module__", None) == name
        ):
            setattr(module, attr, wrap_call(value, f"{name}:{attr}"))


class _CallLoader:
    """A library module's own loader that wraps the module's functions once it has run."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def create_module(self, spec: Any) -> Any:
        return self._inner.create_module(spec)

    def exec_module(self, module: Any) -> None:
        self._inner.exec_module(module)
        try:
            wrap_module(module)
        except Exception:  # noqa: BLE001 - an unwrapped module still works
            pass

    def __getattr__(self, attr: str) -> Any:
        return getattr(self._inner, attr)


class _CallFinder:
    """Finds the modules under this package the normal way and gives each a :class:`_CallLoader`."""

    memory_call_finder = True

    def __init__(self, package: str) -> None:
        self.prefix = package + "."

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        # a library's own tests are left to the other finders (a test runner's import hooks among them)
        if not fullname.startswith(self.prefix) or "tests" in fullname.split(".")[1:]:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        # source modules only: a namespace package or a compiled extension keeps its own loader
        if (
            spec is not None
            and isinstance(spec.origin, str)
            and spec.origin.endswith(".py")
            and hasattr(spec.loader, "exec_module")
        ):
            spec.loader = _CallLoader(spec.loader)
        return spec


def _install_call_finder(package: str) -> None:
    if not any(
        getattr(f, "memory_call_finder", False)
        and getattr(f, "prefix", None) == package + "."
        for f in sys.meta_path
    ):
        sys.meta_path.insert(0, _CallFinder(package))


# Only as the library package itself: the harness also imports this file as a module of its own.
if __name__ == "memory":
    _install_call_finder(__name__)

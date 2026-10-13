# --- asking the librarian (``memory.ask``) ------------------------------------------------------------------------
#
# This section is generated after the rest of the package only with ``UNIFY_MEMORY_LIBRARIAN=on``
# (``library_export.package_source``); it uses the names defined above it and is never imported on its own.
#
# ``ask(request)`` informs and runs nothing on the caller's behalf. First the recognisers: each distinct reader that
# the harness's ``.memory/dispatch.json`` names runs once on *request*, and then each ``applies`` whose reader parsed
# it, in a child process confined as a captured procedure's call is (an isolated interpreter, an empty environment,
# its own process group, a bounded time). The values the readers give stay in that child: only the outcomes come
# back (a reader parsed the text or did not; an ``applies`` held, did not hold or raised). Then the question and those
# outcomes go to the librarian through ``tools.ask_library``; its answer comes back as JSON text, which this section
# reads into a list of :class:`Item`, each with the librarian's reason. The call raises ``memory.helper_call`` with
# the id ``memory:ask`` (asking is never a use) and ``memory.ask_result`` with the ids the answer lists, whether it
# said nothing fits, and the outcomes (ids and labels only, never text or values).

if "wrap_call" not in globals():  # imported as a harness module of its own, not as part of the package
    from .library_helper import (  # noqa: F401
        _CAPTURE_ID,
        _GET_EXAMPLE,
        _ROOT,
        _UNREADABLE,
        _bounded_exchange,
        _stop_child,
        CAPTURE_PREFIX,
        HELPER_EVENT,
        RUN_TIMEOUT_S,
        Annotated,
        Any,
        ContractError,
        Item,
        MinLen,
        NotFound,
        __all__,
        _Catalog,
        _capture_dir,
        _emit_event,
        _first_line,
        _guard,
        _json,
        call_text,
        checked,
        json,
        os,
        sys,
        wrap_call,
    )

DISPATCH_FILE = ".memory/dispatch.json"
ASK_EVENT = "memory.ask_result"
#: A reader's outcome on the text, and an ``applies`` outcome on the value its reader gave.
READ_OUTCOMES = ("parsed", "did-not-parse")
APPLIES_OUTCOMES = ("held", "did-not-hold", "raised", "not-run", "none")
#: The tool the question goes through (the host's librarian session for this request).
ASK_TOOL = "ask_library"

__all__ = [*__all__, "ask"]

#: The child that runs one reader on the text and then the ``applies`` of each procedure that reads with it (stdlib
#: only). It prints the outcomes after a marker line, never a value.
_RECOGNISER_RUNNER = r'''
import json, sys
MARK = sys.argv[1]
out = {"read": "did-not-parse", "applies": []}
def load(spec):
    if spec["kind"] == "capture":
        ns = {"__name__": "captured_procedure", "__file__": spec["path"]}
        with open(spec["path"], encoding="utf-8") as fh:
            exec(compile(fh.read(), spec["path"], "exec"), ns)
        return ns.get(spec["name"])
    import importlib
    sys.path.insert(0, spec["root"])
    module, _, name = spec["id"].partition(":")
    return getattr(importlib.import_module(module), name)
try:
    payload = json.loads(sys.stdin.read())
    value = None
    try:
        reader = load(payload["reader"])
        if not callable(reader):
            raise LookupError("no reader")
        value = reader(payload["text"])
        out["read"] = "parsed"
    except Exception:
        out["read"] = "did-not-parse"
    for spec in payload["applies"]:
        if spec is None:
            out["applies"].append("none")
        elif out["read"] != "parsed":
            out["applies"].append("not-run")
        else:
            try:
                fn = load(spec)
                out["applies"].append("held" if bool(fn(value)) else "did-not-hold")
            except Exception:
                out["applies"].append("raised")
except Exception:
    out = {"read": "did-not-parse", "applies": []}
sys.stdout.write("\n" + MARK + json.dumps(out))
'''
_ASK_MARK = "\x1ememory-ask-facts\x1e"


def _applies_target(row: dict, applies_of: dict) -> Any:
    """The function whose ``applies`` a dispatch row's facts call: a captured procedure's own, a landed procedure's
    from ``applies_of``; None when the row has none (or it cannot be located)."""
    if row.get("applies") is not True:
        return None
    if isinstance(row.get("id"), str) and row["id"].startswith(CAPTURE_PREFIX):
        return row["id"]
    named = applies_of.get(row.get("id")) if isinstance(applies_of, dict) else None
    return named if isinstance(named, str) else None


def _spec(target: Any, name: str) -> Any:
    """How the child loads *target*: a library function by its id (from the copy's root), or a captured procedure's
    *name* function from its ``procedure.py``; None when *target* names neither."""
    if not isinstance(target, str) or not target:
        return None
    if target.startswith(CAPTURE_PREFIX):
        cid = target[len(CAPTURE_PREFIX) :]
        path = os.path.join(_capture_dir(cid), "procedure.py")
        if _CAPTURE_ID.fullmatch(cid) is None or not os.path.isfile(path):
            return None
        return {"kind": "capture", "path": path, "name": name}
    if ":" not in target or not target.startswith("memory."):
        return None
    return {"kind": "item", "id": target, "root": _ROOT}


def _recognise(reader: str, rows: list, text: str, applies_of: Any = None) -> tuple[str, list]:
    """Run *reader* once on *text*, then each row's ``applies`` on what it gave, in one confined child; the
    outcomes only (:data:`READ_OUTCOMES`, :data:`APPLIES_OUTCOMES`). A child that fails, or runs past
    :data:`RUN_TIMEOUT_S`, means the reader did not parse; nothing is retried."""
    import subprocess

    spec = _spec(reader, "read_input")
    applies = [_spec(_applies_target(r, applies_of or {}), "applies") for r in rows]
    none = ("did-not-parse", ["none" if a is None else "not-run" for a in applies])
    if spec is None:
        return none
    try:
        payload = json.dumps({"text": text, "reader": spec, "applies": applies})
    except (TypeError, ValueError):
        return none
    cwd = os.path.dirname(spec["path"]) if spec["kind"] == "capture" else _ROOT
    try:
        child = subprocess.Popen(
            [sys.executable, "-I", "-c", _RECOGNISER_RUNNER, _ASK_MARK],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env={},
            cwd=cwd,
            start_new_session=True,
        )
    except OSError:
        return none
    try:
        # bounded as memory.run's child is (library_helper: RUN_OUTPUT_MAX_BYTES); a timeout or an overrun reads as
        # "did not parse"
        raw, timed_out, over = _bounded_exchange(child, payload.encode("utf-8"), RUN_TIMEOUT_S)
        if timed_out or over:
            raw = b""
    finally:
        _stop_child(child)
    stdout = (raw or b"").decode("utf-8", "replace")
    if _ASK_MARK not in stdout:
        return none
    try:
        rep = json.loads(stdout.rsplit(_ASK_MARK, 1)[1])
    except ValueError:
        return none
    read = rep.get("read") if isinstance(rep, dict) else None
    got = rep.get("applies") if isinstance(rep, dict) else None
    if read not in READ_OUTCOMES or not isinstance(got, list) or len(got) != len(rows):
        return none
    return read, [a if a in APPLIES_OUTCOMES else "raised" for a in got]


def _facts(text: str) -> list:
    """One fact per procedure ``dispatch.json`` lists: ``{id, reader, read, applies}``, ids and outcome labels only.
    Each distinct reader runs once (:func:`_recognise`)."""
    try:
        data = _json(DISPATCH_FILE)
        rows, applies_of = data.get("procedures", []), data.get("applies_of") or {}
    except (RuntimeError, ValueError, AttributeError):
        return []
    rows = [r for r in rows if isinstance(r, dict) and isinstance(r.get("id"), str)] if isinstance(rows, list) else []
    by_reader: dict = {}
    for r in rows:
        by_reader.setdefault(r.get("reader") if isinstance(r.get("reader"), str) else None, []).append(r)
    out: dict = {}
    for reader, group in by_reader.items():
        if reader is None:
            for r in group:
                out[r["id"]] = {"id": r["id"], "reader": None, "read": "did-not-parse",
                                "applies": "none" if r.get("applies") is not True else "not-run"}
            continue
        read, applies = _recognise(reader, group, text, applies_of)
        for r, a in zip(group, applies):
            if a == "none" and r.get("applies") is True:
                a = "not-run"  # it states an applies that this copy cannot locate: not run, never guessed
            out[r["id"]] = {"id": r["id"], "reader": reader, "read": read, "applies": a}
    return [out[r["id"]] for r in rows if r["id"] in out]


class Answer(list):
    """What ``memory.ask`` returns: a list of :class:`Item`, every item the librarian judged applicable, each with
    its ``reason``. When nothing applies the list is empty and ``nearest`` holds the items the librarian named as
    nearest, each with what differs as its ``reason``. It prints as a few lines."""

    def __init__(self, fits: list, nearest: list = ()) -> None:
        super().__init__(fits)
        self.nearest = list(nearest)

    def __repr__(self) -> str:
        out: list = []
        if self:
            out.append(f"{len(self)} item(s) apply:")
            out += [f"- {i.id} ({i.kind}): {i.reason}" for i in self]
            out.append("Each is an Item: print(item) shows its card; item(...) calls a function item.")
        else:
            out.append("Nothing in the library applies.")
            out += [f"- nearest {i.id} ({i.kind}): {i.reason}" for i in self.nearest]
        return "\n".join(out) + "\n"

    __str__ = __repr__


def _answer_items(pairs: list, call: str) -> list:
    """The :class:`Item` of each ``(id, reason)`` the librarian named; an id this copy cannot build (it checks
    every id against the catalogue of the same copy, so none is expected) is kept as a bare Item with its reason."""
    try:
        cat = _Catalog(call)
    except _UNREADABLE:
        cat = None
    out = []
    for iid, reason in pairs:
        known = cat is not None and (iid in cat.rows or iid in cat.captures)
        if known:
            out.append(cat.item(iid, reason))
            continue
        out.append(
            Item(
                id=iid,
                kind="note" if iid.startswith("notes/") else "function",
                summary="",
                applies="",
                evidence="unchecked",
                status="on-trial",
                example="",
                text="",
                links=(),
                history=(),
                reason=reason,
            ),
        )
    return out


def _named(rows: Any, key: str) -> list:
    """The ``(id, text)`` pairs of an answer's ``fits`` (*key* ``why``) or ``nearest`` (*key* ``differs``)."""
    return [
        (r["id"], r[key] if isinstance(r.get(key), str) else "")
        for r in (rows if isinstance(rows, list) else [])
        if isinstance(r, dict) and isinstance(r.get("id"), str)
    ]


def _fact_rows(facts: list) -> tuple:
    return tuple((f["id"], f["read"], f["applies"]) for f in facts if isinstance(f.get("id"), str))


@checked(name="memory.ask")
def ask(request: Annotated[str, MinLen(1)]) -> list[Item]:
    """Ask the librarian which stored items apply to *request* (your request's text, or a question of your own).
    The librarian has read every item, what each was built for and how each has been used. Before the question
    goes, each stored reader runs once on *request* in a separate confined process, and the librarian is told only
    whether each parsed it and whether each stated condition held, never what they read. You may ask again, about
    the same request or another question. Nothing is run on your behalf, and nothing is ranked.

    Args:
        request: the request's text, or a question of your own (non-empty).

    Returns:
        Every item the librarian judged applicable, each an Item with ``.reason``; [] when nothing applies (the
        list's ``.nearest`` then holds the nearest items, each with what differs as its ``.reason``).

    Raises:
        ArgumentError: request is not a non-empty str.
        NotFound: the librarian cannot be reached from these cells, gave no answer, or its answer cannot be
            read.

    Example:
        memory.ask('Total the amounts in the attached table by month.')
    """
    call = call_text("memory.ask", (request,))
    tools = sys.modules.get("tools")
    if tools is None:  # the box imports it before any cell; elsewhere it may be importable but not yet imported
        try:
            import importlib

            tools = importlib.import_module("tools")
        except ImportError:
            tools = None
    send = getattr(tools, ASK_TOOL, None)
    if send is None:
        raise NotFound(
            call,
            "the librarian cannot be reached from this agent's cells.",
            hint="Browse with memory.items() and open an item with memory.get(id).",
        )
    facts = _guard(call, _facts, request)
    refused = getattr(tools, "Refused", None)
    try:
        reply = send(request, facts)
    except Exception as exc:
        if not (isinstance(refused, type) and isinstance(exc, refused)):
            raise
        raise NotFound(
            call,
            f"the librarian gave no answer ({_first_line(exc)}).",
            hint="Ask again later, or browse with memory.items() and open an item with memory.get(id).",
        ) from None
    if isinstance(reply, dict):
        data = reply
    else:
        try:
            data = json.loads(str(reply))
        except ValueError:
            raise NotFound(
                call,
                "the librarian's answer could not be read.",
                hint="Ask again, or browse with memory.items().",
            ) from None
    data = data if isinstance(data, dict) else {}
    fits, nearest = _named(data.get("fits"), "why"), _named(data.get("nearest"), "differs")
    listed = tuple(iid for iid, _ in fits)
    _emit_event(ASK_EVENT, listed, not listed, _fact_rows(facts))
    return _guard(call, lambda: Answer(_answer_items(fits, call), _answer_items(nearest, call) if not fits else []))


if __name__ == "memory":
    ask = wrap_call(ask, "memory:ask", HELPER_EVENT)

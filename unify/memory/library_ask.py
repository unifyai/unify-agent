# --- asking the librarian (``memory.ask``) ------------------------------------------------------------------------
#
# This section is generated after the rest of the package only with ``UNIFY_MEMORY_LIBRARIAN=on``
# (``library_export.package_source``); it uses the names defined above it and is never imported on its own.
#
# ``ask(text)`` informs and runs nothing on the caller's behalf. First the recognisers: each distinct reader that the
# harness's ``.memory/dispatch.json`` names runs once on *text*, and then each ``applies`` whose reader parsed it, in
# a child process confined as ``run`` confines a captured procedure (an isolated interpreter, an empty environment,
# its own process group, a bounded time). The values the readers give stay in that child: only the outcomes come
# back (a reader parsed the text or did not; an ``applies`` held, did not hold or raised). Then the question and those
# outcomes go to the librarian through ``tools.ask_library``; its answer comes back as a plain object. The call
# raises ``memory.helper_call`` with the id ``memory:ask`` (asking is never a use) and ``memory.ask_result`` with the
# ids the answer lists, whether it said nothing fits, and the outcomes (ids and labels only, never text or values).

if "wrap_call" not in globals():  # imported as a harness module of its own, not as part of the package
    from .library_helper import (  # noqa: F401
        _CAPTURE_ID,
        _ROOT,
        CAPTURE_PREFIX,
        HELPER_EVENT,
        RUN_TIMEOUT_S,
        Any,
        __all__,
        _capture_dir,
        _emit_event,
        _json,
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

__all__ = [*__all__, "Answer", "ask"]

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
    import signal
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
        raw, _ = child.communicate(payload.encode("utf-8"), timeout=RUN_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        raw = b""
    finally:
        try:
            os.killpg(child.pid, signal.SIGKILL)  # the run's whole group, whatever it left running
        except OSError:
            pass
        if child.poll() is None:
            child.kill()
        child.wait()
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


class Answer(dict):
    """The librarian's answer as a plain object: ``fits``, every item it judged applicable, each ``{id, call,
    why}``; when nothing fits, ``fits`` is empty and ``nearest`` lists ``{id, differs}``. It prints as a few lines."""

    def __repr__(self) -> str:
        fits = self.get("fits") or []
        out: list = []
        if fits:
            out.append(f"{len(fits)} item(s) fit:")
            for f in fits:
                out.append(f"- {f.get('id')}: {f.get('why')}\n  call: {f.get('call')}")
        else:
            out.append("Nothing in the library fits.")
            for n in self.get("nearest") or []:
                out.append(f"- nearest {n.get('id')}: {n.get('differs')}")
        return "\n".join(out) + "\n"

    __str__ = __repr__


def _fact_rows(facts: list) -> tuple:
    return tuple((f["id"], f["read"], f["applies"]) for f in facts if isinstance(f.get("id"), str))


def ask(text: Any) -> Answer:
    """Ask the librarian which stored items fit *text* (your request's text, or a question of your own). The
    librarian has read every item, what each was built for and how each has been used; it answers with every item
    that fits, each with a ready call and the reason, or says nothing fits and what differs from the nearest
    items. Before the question goes, each stored reader runs once on *text* in a separate confined process, and
    the librarian is told only whether each parsed it and whether each stated condition held, never what they
    read. You may ask again, about the same request or another question. Nothing is run on your behalf."""
    question = text if isinstance(text, str) else json.dumps(text, default=str)
    tools = sys.modules.get("tools")
    if tools is None:  # the box imports it before any cell; elsewhere it may be importable but not yet imported
        try:
            import importlib

            tools = importlib.import_module("tools")
        except ImportError:
            tools = None
    call = getattr(tools, ASK_TOOL, None)
    if call is None:
        raise LookupError("memory.ask: the librarian cannot be reached from this agent's cells")
    facts = _facts(question)
    reply = call(question, facts)
    if isinstance(reply, dict):
        data = reply
    else:
        try:
            data = json.loads(str(reply))
        except ValueError as exc:
            raise RuntimeError("memory.ask: the librarian's answer could not be read") from exc
    answer = Answer(data if isinstance(data, dict) else {"fits": []})
    listed = tuple(f.get("id") for f in answer.get("fits") or [] if isinstance(f, dict) and isinstance(f.get("id"), str))
    _emit_event(ASK_EVENT, listed, not listed, _fact_rows(facts))
    return answer


if __name__ == "memory":
    ask = wrap_call(ask, "memory:ask", HELPER_EVENT)

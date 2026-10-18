# --- memory.ask with each listed captured procedure run on the request (``UNIFY_MEMORY_ASK_PRERUN``) ----------------
#
# This section is generated after the ``memory.ask`` section only with ``UNIFY_MEMORY_ASK_PRERUN=on``
# (``library_export.package_source``); off, the package is byte for byte as without it. It uses the names defined
# above it and is never imported on its own.
#
# ``ask`` here is the ``ask`` above (the librarian's answer, unchanged), and then each captured procedure that answer
# lists as applying is run once on the current request's text: in a separate confined process, as calling the item
# runs it (an isolated interpreter, an empty environment, its own process group, its capture directory as its working
# directory), with no second run, bounded by :data:`PRERUN_EACH_S` per run and :data:`PRERUN_ALL_S` for all of one
# answer's. What it returned, or that it refused the request or did not return, is shown beside the item, with the
# label its record carries when the harness ran it on the input->output pairs its own episode recorded ("fits k of
# n"). It informs: the answer says what ran and what came back, and nothing else. A pre-run is not a use: it raises no
# ``memory.call``; it raises ``memory.ask_prerun`` with, per item, its id, the run's outcome and a 16-hex digest of
# what it returned (never a value).

if "wrap_call" not in globals():  # imported as a harness module of its own, not as part of the package
    from .library_ask import Answer, ask  # noqa: F401
    from .library_helper import (  # noqa: F401
        _CAPTURE_ID,
        _CAPTURE_RUNNER,
        _RUN_MARK,
        CAPTURE_PREFIX,
        Annotated,
        Any,
        Item,
        MinLen,
        _bounded_exchange,
        _capture_dir,
        _current_request,
        _emit_event,
        _json_values,
        _own_pairs_line,
        _read_json,
        _stop_child,
        checked,
        hashlib,
        json,
        os,
        sys,
    )

PRERUN_EVENT = "memory.ask_prerun"
#: One pre-run's bound, and all of one answer's, in seconds (run safety; a run past it is stopped and said so).
PRERUN_EACH_S = 30.0
PRERUN_ALL_S = 120.0
#: The characters of a returned value the answer shows (the cut is marked; calling the item gives it whole).
PRERUN_SHOWN_CHARS = 4000
#: The outcomes a pre-run reports.
PRERUN_OUTCOMES = ("returned", "refused", "error", "timeout", "not-run")

_listing_ask = ask  # the librarian's answer: the ``ask`` above, unchanged


def _prerun_one(capture_id: str, text: str, data: Any, bound: float) -> dict:
    """Run one captured procedure once on *text* (its reader reads it, with *data*), confined as calling the item
    runs it, bounded by *bound* seconds. The child's report, or ``{"outcome": "timeout" | "error", ...}``."""
    import subprocess

    d = _capture_dir(capture_id)
    path = os.path.join(d, "procedure.py")
    if _CAPTURE_ID.fullmatch(capture_id) is None or not os.path.isfile(path):
        return {"outcome": "not-run", "detail": "no captured procedure has this id"}
    payload = json.dumps({"args": [text], "kwargs": {}, "once": True, "data": data})
    try:
        child = subprocess.Popen(
            [sys.executable, "-I", "-c", _CAPTURE_RUNNER, path, _RUN_MARK],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env={},
            cwd=d,
            start_new_session=True,
        )
    except OSError as exc:
        return {"outcome": "not-run", "detail": f"the run could not start ({type(exc).__name__})"}
    try:
        raw, timed_out, over = _bounded_exchange(child, payload.encode("utf-8"), bound)
    finally:
        _stop_child(child)
    if timed_out:
        return {"outcome": "timeout", "detail": f"it ran past {bound:g} s and was stopped"}
    if over:
        return {"outcome": "error", "error": "OutputLimit", "detail": "it printed past the bound and was stopped"}
    stdout = (raw or b"").decode("utf-8", "replace")
    rep: Any = None
    if _RUN_MARK in stdout:
        try:
            rep = json.loads(stdout.rsplit(_RUN_MARK, 1)[1])
        except ValueError:
            rep = None
    if not isinstance(rep, dict) or rep.get("outcome") not in ("returned", "refused", "error"):
        return {"outcome": "error", "error": "NoReport", "detail": "the run ended without a report"}
    return rep


def _prerun_digest(value: Any) -> str:
    """The first 16 hex digits of the sha256 of *value*'s canonical JSON (the harness's ``captures.digest``)."""
    text = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()[:16]


def _prerun_text(rep: dict) -> str:
    """What one pre-run gave, as the answer shows it beside its item."""
    outcome = rep.get("outcome")
    if outcome == "returned":
        try:
            shown = repr(rep.get("result")) if not rep.get("result_is_text") else str(rep.get("result"))
        except Exception as exc:  # noqa: BLE001 - a result that cannot be shown is named by its type
            shown = f"<{type(rep.get('result')).__name__}: repr failed ({type(exc).__name__})>"
        if len(shown) > PRERUN_SHOWN_CHARS:
            shown = shown[:PRERUN_SHOWN_CHARS] + f" ... ({len(shown) - PRERUN_SHOWN_CHARS} more characters)"
        return "returned " + shown
    if outcome == "refused":
        return f"refused this request ({rep.get('error') or 'MemoryInputError'})"
    if outcome == "timeout":
        return f"did not return: {rep.get('detail')}"
    if outcome == "not-run":
        return f"not run: {rep.get('detail')}"
    return f"did not return ({rep.get('error') or 'an error'})"


class PrerunAnswer(Answer):
    """:class:`Answer` with ``runs``: by captured item id, what its pre-run on the current request gave (``outcome``
    and, when it returned, ``result``) and its own-pairs label (``label``, '' when it has none)."""

    def __init__(self, base: Any, runs: dict) -> None:
        super().__init__(
            list(base),
            base.nearest,
            base.note,
            asked=base.asked,
            already_given=base.already_given,
            calls=base.calls,
        )
        self.runs = dict(runs)

    def __repr__(self) -> str:
        if not (self.asked and len(self) and self.runs):
            return Answer.__repr__(self)
        out: list = []
        if self.already_given:
            out.append("The librarian already gave this answer in this request.")
        out.append(f"{len(self)} item(s) apply:")
        for i in self:
            out.append(f"- {i.id} ({i.kind}): {i.reason}")
            if self.calls.get(i.id):
                out.append(f"  call: {self.calls[i.id]}")
            run = self.runs.get(i.id)
            if run is not None:
                out.append("  run once on this request: " + _prerun_text(run))
                if run.get("label"):
                    out.append("  " + run["label"])
        out.append(
            "Each is an Item: print(item) shows its card; item(...) calls a function item. Each captured procedure "
            "listed was run once on this request, in a separate confined process; nothing else has run.",
        )
        if self.note:
            out.append(self.note)
        return "\n".join(out) + "\n"

    __str__ = __repr__


@checked(name="memory.ask")
def ask(question: Annotated[str, MinLen(1)] | None = None) -> list[Item]:
    """Ask the librarian which stored items apply to your current request, or to a question of your own. The
    librarian searches the library by what each item was saved for and reads what it needs. Before the question goes,
    each stored reader runs once on the current request's text in a separate confined process, and the librarian is
    told only whether each parsed it and whether each stated condition held, never what they read. Then each captured
    procedure the answer lists is run once on the current request's text, in a separate confined process as calling
    it runs it, and what it returned (or that it refused or did not return) is shown beside it, with how many of the
    input->output pairs recorded in its own episode it fits when that is known. You may ask again, with the same
    question or another; asked the same question again with the same outcomes, it returns the answer the librarian
    already gave in this request, marked as such. Nothing else is run, and nothing is ranked.

    Args:
        question: a question of your own (non-empty); None asks about the current request itself.

    Returns:
        Every item the librarian judged applicable, each an Item with ``.reason``; [] when nothing applies (the
        list's ``.nearest`` then holds the nearest items, each with what differs as its ``.reason``).
        ``.already_given`` is True when it is the answer already given in this request. ``.runs`` holds, by the id of
        each captured procedure listed, what its run on the current request gave.

    Raises:
        ArgumentError: question is not a non-empty str.
        NotFound: the librarian cannot be reached from these cells, gave no answer, or its answer cannot be
            read.

    Example:
        memory.ask()
    """
    base = _listing_ask(question)
    listed = [i.id for i in base if isinstance(getattr(i, "id", None), str) and i.id.startswith(CAPTURE_PREFIX)]
    if not listed:
        return PrerunAnswer(base, {})
    text, data = _current_request()
    if text is None:
        return PrerunAnswer(base, {})
    data = data if isinstance(data, list) else _json_values(text)
    import time

    stop = time.monotonic() + PRERUN_ALL_S
    runs: dict = {}
    rows: list = []
    for iid in dict.fromkeys(listed):
        cid = iid[len(CAPTURE_PREFIX) :]
        rec = _read_json(os.path.join(_capture_dir(cid), "record.json"))
        rec = rec if isinstance(rec, dict) else {}
        left = min(PRERUN_EACH_S, stop - time.monotonic())
        if rec.get("outcome") == "failed":
            rep: dict = {"outcome": "not-run", "detail": "this captured attempt failed; it is never run"}
        elif left <= 0:
            rep = {"outcome": "not-run", "detail": f"the pre-runs of this answer took their {PRERUN_ALL_S:g} s"}
        else:
            rep = _prerun_one(cid, text, data, left)
        rep["label"] = _own_pairs_line(rec)
        runs[iid] = rep
        returned = rep.get("outcome") == "returned"
        rows.append((iid, rep.get("outcome"), _prerun_digest(rep.get("result")) if returned else ""))
    _emit_event(PRERUN_EVENT, tuple(rows))
    return PrerunAnswer(base, runs)

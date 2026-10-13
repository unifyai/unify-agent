# --- asking the librarian (``memory.ask``) ------------------------------------------------------------------------
#
# This section is generated after the rest of the package only with ``UNIFY_MEMORY_LIBRARIAN=on``
# (``library_export.package_source``); it uses the names defined above it and is never imported on its own.
#
# ``ask(question=None)`` informs and runs nothing on the caller's behalf. First the recognisers: each distinct reader
# that the harness's ``.memory/dispatch.json`` names runs once on the current request's text (the host tool
# ``current_request``), and then each ``applies`` whose reader parsed it, all in one confined child process
# (:func:`_facts`, defined above with ``match``). The values the readers give stay in that child: only the outcomes
# come back. Then the question (the given one, else the request's text) and those outcomes go to the librarian through
# ``tools.ask_library``; its answer comes back as JSON text, which this section reads into a list of :class:`Item`,
# each with the librarian's reason. The call raises ``memory.helper_call`` with the id ``memory:ask`` (asking is never
# a use) and ``memory.ask_result`` with the ids the answer lists, whether it said nothing fits, and the outcomes (ids,
# labels and, for a reader that gave no outcome, a fixed short reason; never text or values).

if "wrap_call" not in globals():  # imported as a harness module of its own, not as part of the package
    from .library_helper import (  # noqa: F401
        _CAPTURE_ID,
        _GET_EXAMPLE,
        _ROOT,
        _UNREADABLE,
        _bounded_exchange,
        _current_request,
        _fact_rows,
        _facts,
        _no_outcome_note,
        _stop_child,
        _tools,
        APPLIES_OUTCOMES,
        CAPTURE_PREFIX,
        DISPATCH_FILE,
        HELPER_EVENT,
        READ_OUTCOMES,
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

ASK_EVENT = "memory.ask_result"
#: The tool the question goes through (the host's librarian session for this request).
ASK_TOOL = "ask_library"
#: The fact fields the question carries to the host (a reader's reason for giving no outcome goes in the audit event).
_SENT = ("id", "reader", "read", "applies")

__all__ = [*__all__, "ask"]


class Answer(list):
    """What ``memory.ask`` returns: a list of :class:`Item`, every item the librarian judged applicable, each with
    its ``reason``. When nothing applies the list is empty and ``nearest`` holds the items the librarian named as
    nearest, each with what differs as its ``reason``. ``note`` says what the answer could not use (the current
    request, a reader that gave no outcome); ``asked`` is False when nothing could be asked. It prints as a few
    lines."""

    def __init__(self, fits: list, nearest: list = (), note: str = "", asked: bool = True) -> None:
        super().__init__(fits)
        self.nearest = list(nearest)
        self.note = note
        self.asked = asked

    def __repr__(self) -> str:
        out: list = []
        if not self.asked:
            out.append(self.note)
        elif self:
            out.append(f"{len(self)} item(s) apply:")
            out += [f"- {i.id} ({i.kind}): {i.reason}" for i in self]
            out.append("Each is an Item: print(item) shows its card; item(...) calls a function item.")
        else:
            out.append("Nothing in the library applies.")
            out += [f"- nearest {i.id} ({i.kind}): {i.reason}" for i in self.nearest]
        if self.note and self.asked:
            out.append(self.note)
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


@checked(name="memory.ask")
def ask(question: Annotated[str, MinLen(1)] | None = None) -> list[Item]:
    """Ask the librarian which stored items apply to your current request, or to a question of your own. The
    librarian has read every item, what each was built for and how each has been used. Before the question goes,
    each stored reader runs once on the current request's text in a separate confined process, and the librarian is
    told only whether each parsed it and whether each stated condition held, never what they read. You may ask
    again, with the same question or another. Nothing is run on your behalf, and nothing is ranked.

    Args:
        question: a question of your own (non-empty); None asks about the current request itself.

    Returns:
        Every item the librarian judged applicable, each an Item with ``.reason``; [] when nothing applies (the
        list's ``.nearest`` then holds the nearest items, each with what differs as its ``.reason``).

    Raises:
        ArgumentError: question is not a non-empty str.
        NotFound: the librarian cannot be reached from these cells, gave no answer, or its answer cannot be
            read.

    Example:
        memory.ask()
    """
    call = call_text("memory.ask", (question,) if question is not None else ())
    tools = _tools()
    send = getattr(tools, ASK_TOOL, None)
    if send is None:
        raise NotFound(
            call,
            "the librarian cannot be reached from this agent's cells.",
            hint="Browse with memory.items() and open an item with memory.get(id).",
        )
    text, _ = _current_request()
    notes = []
    if text is None:
        if question is None:
            note = "The current request is not available here, so nothing was asked; ask a question: memory.ask('...')."
            return Answer([], [], note, asked=False)
        notes.append("The current request is not available here, so no stored reader ran; the librarian was "
                     "told only your question.")
    # the readers always read the current request; the librarian is asked the question, else the request itself
    facts = _guard(call, _facts, text) if text is not None else []
    if _no_outcome_note(facts):
        notes.append(_no_outcome_note(facts))
    # the tool's own failures (a refusal, a harness fault, a closed channel) say why in their first line
    failures = tuple(
        c
        for c in (getattr(tools, name, None) for name in ("Refused", "ToolError", "ChannelClosed"))
        if isinstance(c, type) and issubclass(c, BaseException)
    )
    try:
        reply = send(question if question is not None else text, [{k: f[k] for k in _SENT} for f in facts])
    except failures as exc:
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
    return _guard(
        call,
        lambda: Answer(_answer_items(fits, call), _answer_items(nearest, call) if not fits else [], " ".join(notes)),
    )


if __name__ == "memory":
    ask = wrap_call(ask, "memory:ask", HELPER_EVENT)

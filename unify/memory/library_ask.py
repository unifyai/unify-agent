# --- asking the librarian (``memory.ask``) ------------------------------------------------------------------------
#
# This section is generated after the rest of the package only with ``UNIFY_MEMORY_LIBRARIAN=on``
# (``library_export.package_source``); it uses the names defined above it and is never imported on its own.
#
# ``ask(question=None)`` informs and runs nothing, here or anywhere (rev 2: no stored reader or ``applies`` runs to
# recognise anything). The question (the given one, else the current request's text, from the host tool
# ``current_request``) goes to the librarian through ``tools.ask_library``; the librarian judges relevance from the
# harness's structural catalogue (``.memory/dispatch.json``) and the items' text. Its answer comes back as JSON text,
# which this section reads into a list of :class:`Item`, each with the librarian's reason (marked ``already_given``
# when the host returned the answer the librarian already gave in this request to the same question). The call raises
# ``memory.helper_call`` with the id ``memory:ask`` (asking is never a use) and ``memory.ask_result`` with the ids the
# answer lists, whether it said nothing fits, and how many catalogue rows the copy holds (never text or values).

if "wrap_call" not in globals():  # imported as a harness module of its own, not as part of the package
    from .library_helper import (  # noqa: F401
        _CAPTURE_ID,
        _GET_EXAMPLE,
        _ROOT,
        _UNREADABLE,
        _bounded_exchange,
        _current_request,
        _stop_child,
        _tools,
        CAPTURE_PREFIX,
        DISPATCH_FILE,
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

ASK_EVENT = "memory.ask_result"
#: The tool the question goes through (the host's librarian session for this request).
ASK_TOOL = "ask_library"
#: The field of a reply that is an answer the librarian already gave in this request (the host's
#: ``librarian.ALREADY_GIVEN``).
_ALREADY_GIVEN = "already_given"


def _catalogue_rows() -> int:
    """How many rows the copy's structural catalogue (:data:`DISPATCH_FILE`) holds; 0 when it cannot be read."""
    try:
        data = _json(DISPATCH_FILE)
    except (RuntimeError, ValueError):
        return 0
    rows = data.get("procedures") if isinstance(data, dict) else None
    return sum(1 for r in rows if isinstance(r, dict)) if isinstance(rows, list) else 0

__all__ = [*__all__, "ask", "AskSchemaError"]
#: The reply's fields with a schema (the host's ``librarian.SCHEMA_VALUE`` / ``SCHEMA_ERROR``).
_SCHEMA_VALUE, _SCHEMA_ERROR = "value", "schema_error"
#: How deep a schema type may nest (a bound on a self-referring dataclass).
_SCHEMA_DEPTH = 20
_SCHEMA_KINDS = (
    "a JSON Schema dict, a dataclass, a TypedDict, or a pydantic model; field types str, int, float, bool, None, "
    "Optional, list[...], dict[str, ...], Literal[...] and nested dataclasses"
)


class AskSchemaError(ContractError, ValueError):
    """The librarian's answer did not match the schema of ``memory.ask(..., schema=...)``, even after one repair:
    ``.raw`` is its answer as text, ``.errors`` the validator's ``(path, message)`` pairs and ``.schema`` the JSON
    Schema it was checked against."""

    def __init__(self, call: str, raw: str, errors: list, schema: dict) -> None:
        self.raw, self.errors, self.schema = raw, [tuple(e) for e in errors], schema
        shown_errors = "; ".join(f"{p or '/'}: {m}" for p, m in self.errors[:5])
        super().__init__(
            call,
            f"the librarian's answer did not match the schema after one repair ({shown_errors}).",
            hint="Ask again with a simpler schema, or without one: memory.ask(question).",
        )


def _type_schema(tp: Any, call: str, depth: int = 0) -> dict:
    """The JSON Schema of one field type (:data:`_SCHEMA_KINDS`); ArgumentError for a type it does not cover."""
    import dataclasses
    import types
    import typing

    if depth > _SCHEMA_DEPTH:
        raise ArgumentError(call, "the schema nests too deeply.", expected=_SCHEMA_KINDS)
    if tp is Any:
        return {}
    if tp is None or tp is type(None):
        return {"type": "null"}
    simple = {str: "string", int: "integer", float: "number", bool: "boolean"}
    if tp in simple:
        return {"type": simple[tp]}
    origin, args = typing.get_origin(tp), typing.get_args(tp)
    if origin is typing.Literal:
        return {"enum": list(args)}
    if origin in (typing.Union, getattr(types, "UnionType", typing.Union)):
        return {"anyOf": [_type_schema(a, call, depth + 1) for a in args]}
    if origin in (list, tuple, set, frozenset) and len(args) <= 1:
        return {"type": "array", "items": _type_schema(args[0], call, depth + 1) if args else {}}
    if tp in (list, tuple, set, frozenset):
        return {"type": "array"}
    if origin is dict and (not args or args[0] is str):
        return {"type": "object", "additionalProperties": _type_schema(args[1], call, depth + 1) if args else {}}
    if tp is dict:
        return {"type": "object"}
    if isinstance(tp, type) and dataclasses.is_dataclass(tp):
        hints = _hints(tp)
        fields = [f for f in dataclasses.fields(tp) if f.init]
        required = [f.name for f in fields
                    if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING]  # fmt: skip
        return {"type": "object", "properties": {f.name: _type_schema(hints.get(f.name, Any), call, depth + 1)
                                                 for f in fields},
                "required": required, "additionalProperties": False}  # fmt: skip
    if _is_typeddict(tp):
        hints = _hints(tp)
        return {"type": "object", "properties": {k: _type_schema(v, call, depth + 1) for k, v in hints.items()},
                "required": sorted(getattr(tp, "__required_keys__", hints)), "additionalProperties": False}  # fmt: skip
    raise ArgumentError(call, f"the schema type {getattr(tp, '__name__', tp)!r} is not supported.",
                        expected=_SCHEMA_KINDS)  # fmt: skip


def _hints(tp: Any) -> dict:
    """*tp*'s field types: as ``typing.get_type_hints`` resolves them, else each written annotation resolved against
    ``typing``, the builtins and its module's names (a class a cell defined), else as written."""
    import builtins
    import typing

    try:
        return typing.get_type_hints(tp)
    except Exception:  # noqa: BLE001 - names a cell defined: resolved below
        pass
    names = {**vars(builtins), **vars(typing), **vars(sys.modules.get(getattr(tp, "__module__", ""), builtins))}
    out = {}
    for k, v in (getattr(tp, "__annotations__", {}) or {}).items():
        if isinstance(v, str):
            try:
                v = eval(v, names, {})  # noqa: S307 - the caller's own annotation text, in the caller's own cell
            except Exception:  # noqa: BLE001 - left as written: reported as unsupported
                pass
        out[k] = v
    return out


def _is_typeddict(tp: Any) -> bool:
    return isinstance(tp, type) and issubclass(tp, dict) and hasattr(tp, "__required_keys__")


def _built(tp: Any, value: Any, depth: int = 0) -> Any:
    """*value* (valid JSON for :func:`_type_schema` of *tp*) built back into *tp*: dataclasses as instances, nested
    ones too; everything else as JSON gives it."""
    import dataclasses
    import typing

    if depth > _SCHEMA_DEPTH or value is None:
        return value
    origin, args = typing.get_origin(tp), typing.get_args(tp)
    if isinstance(tp, type) and dataclasses.is_dataclass(tp) and isinstance(value, dict):
        hints = _hints(tp)
        return tp(**{k: _built(hints.get(k, Any), v, depth + 1) for k, v in value.items()})
    if origin in (list, tuple, set, frozenset) and args and isinstance(value, list):
        items = [_built(args[0], v, depth + 1) for v in value]
        return origin(items) if origin is not list else items
    if origin is dict and len(args) == 2 and isinstance(value, dict):
        return {k: _built(args[1], v, depth + 1) for k, v in value.items()}
    if origin is not None and origin is not typing.Literal and args:  # a union: the member the value is
        for a in args:
            if a is type(None):
                continue
            if isinstance(a, type) and dataclasses.is_dataclass(a) and isinstance(value, dict):
                try:
                    return _built(a, value, depth + 1)
                except TypeError:
                    continue
        return value
    return value


def _schema_of(schema: Any, call: str) -> tuple[dict, Any]:
    """``(the JSON Schema, how a valid value is given back)`` for *schema* (:data:`_SCHEMA_KINDS`)."""
    import dataclasses

    if isinstance(schema, dict):
        return schema, lambda v: v
    if isinstance(schema, type) and dataclasses.is_dataclass(schema):
        return _type_schema(schema, call), lambda v: _built(schema, v)
    if _is_typeddict(schema):
        return _type_schema(schema, call), lambda v: v
    if isinstance(schema, type) and callable(getattr(schema, "model_json_schema", None)) and callable(
        getattr(schema, "model_validate", None)
    ):  # a pydantic model, where pydantic imports
        return schema.model_json_schema(), schema.model_validate
    raise ArgumentError(call, f"schema must be {_SCHEMA_KINDS}, not {type(schema).__name__}.",
                        expected=_SCHEMA_KINDS)  # fmt: skip



class Answer(list):
    """What ``memory.ask`` returns: a list of :class:`Item`, every item the librarian judged applicable, each with
    its ``reason``. When nothing applies the list is empty and ``nearest`` holds the items the librarian named as
    nearest, each with what differs as its ``reason``. ``note`` says what the answer could not use (the current
    request, a reader that gave no outcome); ``asked`` is False when nothing could be asked; ``already_given`` is True
    when this is the answer the librarian already gave in this request (the same question and outcomes); ``calls``
    holds, by item id, the call the librarian gave for each item that applies (for a procedure with a recipe, the
    call that applies the recipe to the current request). It prints as a few lines."""

    def __init__(
        self,
        fits: list,
        nearest: list = (),
        note: str = "",
        asked: bool = True,
        already_given: bool = False,
        calls: dict | None = None,
    ) -> None:
        super().__init__(fits)
        self.nearest = list(nearest)
        self.note = note
        self.asked = asked
        self.already_given = already_given
        self.calls = dict(calls or {})

    def __repr__(self) -> str:
        out: list = []
        if self.already_given:
            out.append("The librarian already gave this answer in this request.")
        if not self.asked:
            out.append(self.note)
        elif self:
            out.append(f"{len(self)} item(s) apply:")
            for i in self:
                out.append(f"- {i.id} ({i.kind}): {i.reason}")
                if self.calls.get(i.id):
                    out.append(f"  call: {self.calls[i.id]}")
            out.append("Each is an Item: print(item) shows its card; item(...) calls a function item. Nothing has run.")
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
def ask(question: Annotated[str, MinLen(1)] | None = None, schema: Any = None) -> Any:
    """Ask the librarian which stored items apply to your current request, or to a question of your own. The
    librarian searches the library by what each item was saved for, reads the harness's structural catalogue of the
    stored procedures and what else it needs. You may ask again, with the same question or another; asked the same
    question again, it returns the answer the librarian already gave in this request, marked as such. Nothing is run,
    here or on your behalf, and nothing is ranked.

    With *schema* the librarian answers in that shape instead: one JSON value, checked against the schema; a value
    that does not match goes back to it once with the checker's errors, and a second miss raises AskSchemaError.

    Args:
        question: a question of your own (non-empty); None asks about the current request itself.
        schema: None (the list of items), or the shape of the answer you want: a JSON Schema dict, a dataclass
            (its instance comes back, nested dataclasses too), a TypedDict (a dict comes back) or a pydantic model
            (validated with it). Field types: str, int, float, bool, None, Optional, list[...], dict[str, ...],
            Literal[...] and nested dataclasses.

    Returns:
        Without a schema, every item the librarian judged applicable, each an Item with ``.reason``; [] when
        nothing applies (the list's ``.nearest`` then holds the nearest items, each with what differs as its
        ``.reason``); ``.already_given`` is True when it is the answer already given in this request. With a
        schema, the answer in its shape.

    Raises:
        ArgumentError: question is not a non-empty str, or schema is not one of the kinds above.
        AskSchemaError: the answer did not match the schema after one repair (``.raw``, ``.errors``, ``.schema``).
        NotFound: the librarian cannot be reached from these cells, gave no answer, or its answer cannot be
            read.

    Example:
        memory.ask()

        @dataclass
        class Plan:
            steps: list[str]
            risky: bool

        memory.ask("what does the library say about this?", schema=Plan)
    """
    call = call_text(
        "memory.ask", (question,) if question is not None else (), {"schema": "..."} if schema is not None else None
    )
    shape, give = _schema_of(schema, call) if schema is not None else (None, None)
    tools = _tools()
    send = getattr(tools, ASK_TOOL, None)
    if send is None:
        raise NotFound(
            call,
            "the librarian cannot be reached from this agent's cells.",
            hint="Browse with memory.items() and open an item with memory.get(id).",
        )
    text = _current_request()[0] if question is None else None
    notes: list = []
    if question is None and text is None:
        note = "The current request is not available here, so nothing was asked; ask a question: memory.ask('...')."
        return Answer([], [], note, asked=False)
    # the tool's own failures (a refusal, a harness fault, a closed channel) say why in their first line
    failures = tuple(
        c
        for c in (getattr(tools, name, None) for name in ("Refused", "ToolError", "ChannelClosed"))
        if isinstance(c, type) and issubclass(c, BaseException)
    )
    try:
        if shape is None:
            reply = send(question if question is not None else text)
        else:
            reply = send(question if question is not None else text, schema=shape)
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
    if shape is not None:
        _emit_event(ASK_EVENT, (), False, _catalogue_rows())
        failed = data.get(_SCHEMA_ERROR)
        if isinstance(failed, dict):
            raise AskSchemaError(call, str(failed.get("raw") or ""), list(failed.get("errors") or []), shape)
        if _SCHEMA_VALUE not in data:
            raise NotFound(call, "the librarian's answer could not be read.", hint="Ask again.")
        try:
            return give(data[_SCHEMA_VALUE])
        except Exception as exc:  # noqa: BLE001 - a value the schema passed that the type still refuses
            raw = json.dumps(data[_SCHEMA_VALUE], ensure_ascii=False, default=str)[:20000]
            raise AskSchemaError(call, raw, [("", f"{type(exc).__name__}: {str(exc)[:200]}")], shape) from None
    fits, nearest = _named(data.get("fits"), "why"), _named(data.get("nearest"), "differs")
    calls = dict(_named(data.get("fits"), "call"))
    listed = tuple(iid for iid, _ in fits)
    _emit_event(ASK_EVENT, listed, not listed, _catalogue_rows())
    again = data.get(_ALREADY_GIVEN) is True
    return _guard(
        call,
        lambda: Answer(
            _answer_items(fits, call),
            _answer_items(nearest, call) if not fits else [],
            " ".join(notes),
            already_given=again,
            calls=calls,
        ),
    )


if __name__ == "memory":
    ask = wrap_call(ask, "memory:ask", HELPER_EVENT)

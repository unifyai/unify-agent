"""Rev 2, item 2: ``memory.ask(question, schema=...)`` answers in the shape asked for.

The cell's side turns a dataclass, a TypedDict, a pydantic model or a JSON Schema dict into a JSON Schema and builds
the answer back; the host's librarian answers one JSON value, checked against the schema, with one repair turn that
carries the checker's errors (never the request); a second miss raises ``AskSchemaError``. The answer cache keys on
the schema's digest, and the run keeps nothing of a schema but its digest. No model, network or key.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Literal, Optional, TypedDict

import pytest

from tests.memory.test_librarian_answer_once import REQUEST, Turn, _ctx, _session
from unify.actor import tool_handlers as H
from unify.memory import librarian as L
from unify.memory import library_ask as A


@dataclass
class Step:
    name: str
    cost: Optional[int] = None


@dataclass
class Plan:
    steps: list[Step]
    risky: bool
    mode: Literal["quick", "careful"] = "quick"
    notes: dict[str, str] = field(default_factory=dict)


class Total(TypedDict):
    count: int
    label: str


class Partial(TypedDict, total=False):
    count: int


PLAN = {"steps": [{"name": "read", "cost": 2}, {"name": "write"}], "risky": False, "mode": "careful"}


# --- the cell's side: the schema and the value built back ----------------------------------------------------------------


def test_a_dataclass_becomes_a_schema_and_its_answer_an_instance():
    schema, give = A._schema_of(Plan, "memory.ask(...)")
    assert schema["type"] == "object" and schema["required"] == ["steps", "risky"]
    assert schema["properties"]["mode"] == {"enum": ["quick", "careful"]}
    step = schema["properties"]["steps"]["items"]
    assert step["required"] == ["name"] and step["properties"]["cost"]["anyOf"] == [{"type": "integer"}, {"type": "null"}]
    assert schema["properties"]["notes"] == {"type": "object", "additionalProperties": {"type": "string"}}
    got = give(PLAN)
    assert got == Plan([Step("read", 2), Step("write")], False, "careful")
    assert L.schema_errors(PLAN, schema) == []


def test_typed_dicts_total_and_not_total():
    total, _ = A._schema_of(Total, "c")
    assert total["required"] == ["count", "label"] and total["properties"]["count"] == {"type": "integer"}
    partial, give = A._schema_of(Partial, "c")
    assert partial["required"] == [] and give({"count": 1}) == {"count": 1}


def test_a_json_schema_dict_is_used_as_given():
    schema = {"type": "array", "items": {"type": "string"}}
    got, give = A._schema_of(schema, "c")
    assert got is schema and give(["a"]) == ["a"]


def test_a_pydantic_model_where_pydantic_imports():
    pydantic = pytest.importorskip("pydantic")

    class Model(pydantic.BaseModel):
        steps: list[str]
        risky: bool

    schema, give = A._schema_of(Model, "c")
    assert schema["required"] == ["steps", "risky"]
    assert give({"steps": ["a"], "risky": True}) == Model(steps=["a"], risky=True)


def test_an_unsupported_type_is_an_argument_error_naming_the_kinds():
    class Plain:
        pass

    @dataclass
    class Odd:
        when: complex

    for bad in (Plain, Odd, 3, "text"):
        with pytest.raises(A.ArgumentError, match="JSON Schema dict, a dataclass, a TypedDict"):
            A._schema_of(bad, "memory.ask('q', schema=...)")


# --- the cell's side: memory.ask with a stand-in host ----------------------------------------------------------------------


def _host(monkeypatch, reply: dict) -> list:
    sent: list = []

    def ask_library(question, schema=None):
        sent.append([question, schema])
        return json.dumps(reply)

    monkeypatch.setattr(A, "_tools", lambda: SimpleNamespace(ask_library=ask_library))
    monkeypatch.setattr(A, "_current_request", lambda: (REQUEST, None))
    return sent


def test_ask_with_a_schema_sends_it_and_builds_the_answer(monkeypatch):
    sent = _host(monkeypatch, {"value": PLAN, "repairs": 1})
    got = A.ask("What does the library say about this?", schema=Plan)
    assert got == Plan([Step("read", 2), Step("write")], False, "careful")
    ((question, schema),) = sent
    assert question == "What does the library say about this?" and schema["required"] == ["steps", "risky"]


def test_a_double_miss_raises_ask_schema_error_with_the_raw_answer(monkeypatch):
    errors = [["/risky", "'no' is not of type 'boolean'"]]
    _host(monkeypatch, {"schema_error": {"raw": '{"risky": "no"}', "errors": errors}})
    with pytest.raises(A.AskSchemaError) as got:
        A.ask("q", schema=Plan)
    assert got.value.raw == '{"risky": "no"}' and got.value.errors == [tuple(errors[0])]
    assert got.value.schema["required"] == ["steps", "risky"] and "/risky" in str(got.value)


def test_without_a_schema_ask_is_as_before(monkeypatch):
    sent = _host(monkeypatch, {"fits": []})
    got = A.ask("q")
    assert list(got) == [] and sent == [["q", None]] and isinstance(got, A.Answer)


# --- the host's side: the check, the repair and the cache ---------------------------------------------------------------


GOOD = {"steps": [{"name": "read"}], "risky": False}
BAD = {"steps": "read", "risky": "no"}
SCHEMA = A._schema_of(Plan, "c")[0]


async def _ask(ctx, schema=SCHEMA, question=REQUEST):
    got = await H.ask_library(ctx, {"question": question, "schema": schema})
    return got, (json.loads(got.text) if got.status == "ok" else None)


@pytest.mark.asyncio
async def test_a_wrong_answer_is_repaired_once_with_the_checkers_errors(tmp_path):
    events: list = []
    turn = Turn([BAD, GOOD])
    session = _session(tmp_path, turn)
    session._on_event = events.append
    try:
        _, got = await _ask(_ctx(session))
    finally:
        await session.close()
    assert got == {"value": GOOD, "repairs": 1}
    # the repair carried where the value missed the schema, never the request
    repair = [m["content"] for m in session.messages if m.get("role") == "tool"][0]
    assert "does not match the schema" in repair and "/risky" in repair and REQUEST not in repair
    # the question carried the schema in place of the answer's form
    question = [m["content"] for m in session.messages if m.get("role") == "user"][0]
    assert "matches this JSON Schema" in question and '"risky"' in question
    (ev,) = [e for e in events if e.get("event") == "ask_schema"]
    assert ev["repairs"] == 1 and ev["ok"] is True and ev["schema"] == L._digest_of(SCHEMA)
    assert "properties" not in json.dumps(ev)  # the run keeps the schema's digest only


@pytest.mark.asyncio
async def test_a_second_miss_ends_the_question_and_is_never_kept(tmp_path):
    turn = Turn([BAD, BAD, GOOD])
    session = _session(tmp_path, turn)
    ctx = _ctx(session)
    try:
        _, first = await _ask(ctx)
        _, again = await _ask(ctx)
    finally:
        await session.close()
    assert set(first) == {"schema_error"} and first["schema_error"]["errors"]
    assert json.loads(first["schema_error"]["raw"]) == BAD
    assert again == {"value": GOOD, "repairs": 0} and len(turn.calls) == 3  # asked again: a miss is never kept


@pytest.mark.asyncio
async def test_the_cache_keys_on_the_schema(tmp_path):
    other = {"type": "object", "properties": {"risky": {"type": "boolean"}}, "required": ["risky"]}
    turn = Turn([GOOD, {"risky": True}])
    session = _session(tmp_path, turn)
    ctx = _ctx(session)
    try:
        _, one = await _ask(ctx)
        _, two = await _ask(ctx)
        _, three = await _ask(ctx, schema=other)
    finally:
        await session.close()
    assert one == {"value": GOOD, "repairs": 0} and two == {**one, L.ALREADY_GIVEN: True}
    assert three == {"value": {"risky": True}, "repairs": 0} and len(turn.calls) == 2


@pytest.mark.asyncio
async def test_a_schema_that_is_not_one_is_refused_before_anything_is_asked(tmp_path):
    turn = Turn([])
    session = _session(tmp_path, turn)
    try:
        for bad in ({"type": "no such type"}, ["not", "an", "object"]):
            got, _ = await _ask(_ctx(session), schema=bad)
            assert got.status == "refused" and got.form == "bad_schema"
    finally:
        await session.close()
    assert turn.calls == [] and session.questions == 0

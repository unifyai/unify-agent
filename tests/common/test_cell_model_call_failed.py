"""Symbolic: a provider error on a cell's own model call that is not the request's own fault comes back to the cell
as ``ModelCallFailed(<status>, <exception type>)``: fixed text, nothing chained, no body.

Covered: an opaque 400, a 400 that names access (a key), a 429 and a 5xx, each carrying a planted body with a
``provider_name``, a ``user_id``, a URL and a key fingerprint. None of those reaches the cell, the frame the
harness sends to the sandboxed worker, or the model's transcript, and the cell can catch the error by name.

Also: the access markers of a 400 now include ``credit``, ``quota``, ``billing``, ``payment`` and ``balance``, in
the error's type, its code, its message, or an upstream error nested in ``metadata.raw``: such a 400 is not a
``ModelRequestError`` (the request is not at fault), so a cell gets ``ModelCallFailed``.

The provider is faked at unillm's transport (``tests/scripted_model.py``, above unillm's transient-retry policy,
so a 429 or a 5xx is answered once); nothing leaves the process.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from typing import Any

import litellm
import pytest

from tests.actor.code_act.sandbox_world import needs_bwrap
from tests.common.test_cell_model_request_errors import (
    LUNA,
    SCHEMA,
    bad_schema,
    routed,
    session,  # noqa: F401 (fixture)
)
from tests.scripted_model import ScriptedModel, cell, reply, scripted
from unify.common import cell_models, reasoning

# ── ASSUMED NAMES (the integrator adapts them here, in one place) ────────────────────────────────────────────────
#: The error class a cell gets, in ``unify.common.cell_models`` and as a cell global of the same name.
CALL_FAILED_NAME = "ModelCallFailed"


def call_failed_class() -> type:
    return getattr(cell_models, CALL_FAILED_NAME)


def call_failed_fields(exc: BaseException) -> tuple:
    """The status and the provider exception's type name a ``ModelCallFailed`` carries (``status``, ``kind``)."""
    return (exc.status, exc.kind)


def call_failed_text(status: int, type_name: str) -> str:
    """The part of the fixed message that names the status and the type, which is all a cell sees."""
    return f"(status {status}, {type_name})"


# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────

PLANTED_PROVIDER = "planted-provider-name-for-the-call-failed-test"
PLANTED_USER = "planted-user-id-4242"
PLANTED_URL = "https://planted-provider.example/v1/keys?ref=planted"
PLANTED_FINGERPRINT = "planted-key-fingerprint-7f3a9c"
# what must never reach a cell, a worker frame or a transcript (the field names as well as their values)
PLANTED = (
    PLANTED_PROVIDER,
    PLANTED_USER,
    "planted-provider.example",
    PLANTED_FINGERPRINT,
    "provider_name",
    "user_id",
    "OpenrouterException",
)


def _planted_body(message: str, **error: Any) -> dict:
    """An OpenRouter-style error body carrying every planted value."""
    raw = {
        "error": {
            "message": f"{message} (see {PLANTED_URL}; key {PLANTED_FINGERPRINT})",
            **error,
        },
    }
    return {
        "error": {
            "message": f"{message} (see {PLANTED_URL})",
            "code": error.get("code", 400),
            "metadata": {
                "raw": json.dumps(raw),
                "provider_name": PLANTED_PROVIDER,
            },
        },
        "user_id": PLANTED_USER,
        "key_fingerprint": PLANTED_FINGERPRINT,
    }


def _text_of(body: dict) -> str:
    return f"OpenrouterException - {json.dumps(body)}"


def opaque_400() -> litellm.BadRequestError:
    exc = litellm.BadRequestError(
        message=_text_of(_planted_body("Provider returned error")),
        model="openrouter/openai/gpt-6-luna",
        llm_provider="openrouter",
    )
    return exc


def access_400() -> litellm.BadRequestError:
    body = _planted_body(
        "API key not valid. Please pass a valid API key.",
        type="invalid_request_error",
        code="invalid_api_key",
    )
    exc = litellm.BadRequestError(
        message=_text_of(body),
        model="openrouter/openai/gpt-6-luna",
        llm_provider="openrouter",
    )
    exc.body = body["error"]
    return exc


def rate_limited() -> litellm.RateLimitError:
    return litellm.RateLimitError(
        message=_text_of(_planted_body("Rate limit exceeded", code=429)),
        model="openrouter/openai/gpt-6-luna",
        llm_provider="openrouter",
    )


def server_error() -> litellm.InternalServerError:
    return litellm.InternalServerError(
        message=_text_of(_planted_body("Upstream provider failed", code=500)),
        model="openrouter/openai/gpt-6-luna",
        llm_provider="openrouter",
    )


CASES = [
    (opaque_400, 400, "BadRequestError"),
    (access_400, 400, "BadRequestError"),
    (rate_limited, 429, "RateLimitError"),
    (server_error, 500, "InternalServerError"),
]
IDS = ["opaque-400", "access-400", "429", "5xx"]


def _assert_clean(text: str) -> None:
    for planted in PLANTED:
        assert planted not in text, (planted, text[-2000:])
    assert "://" not in text, text[-2000:]


# ── a cell's query_llm and unillm client ───────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("make, status, type_name", CASES, ids=IDS)
@pytest.mark.asyncio
async def test_a_cells_query_llm_raises_model_call_failed_with_fixed_text(session, make, status, type_name):
    failed = call_failed_class()
    model = ScriptedModel(query_llm=[make()])
    with scripted(model):
        with pytest.raises(failed) as info:
            await cell_models.cell_query_llm("Say ok.")
    exc = info.value
    assert type(exc).__name__ == CALL_FAILED_NAME
    assert call_failed_fields(exc) == (status, type_name)
    assert not isinstance(exc, cell_models.ModelRequestError)
    # nothing chained, no body: raised apart from the provider's exception
    assert exc.__cause__ is None and exc.__context__ is None
    assert getattr(exc, "body", None) is None
    for text in (str(exc), repr(exc), json.dumps(exc.args), json.dumps(vars(exc), default=str)):
        _assert_clean(text)
    assert str(status) in str(exc) and type_name in str(exc)
    # a 400 or a 429 is sent once (no retry at this layer)
    assert model.kinds() == ["query_llm"]


@pytest.mark.asyncio
async def test_the_text_is_fixed_whatever_the_body_says(session):
    failed = call_failed_class()
    other = litellm.BadRequestError(
        message=_text_of({"error": {"message": "something else entirely", "code": 400}}),
        model="openrouter/openai/gpt-6-luna",
        llm_provider="openrouter",
    )
    model = ScriptedModel(query_llm=[opaque_400(), other])
    texts = []
    with scripted(model):
        for _ in range(2):
            with pytest.raises(failed) as info:
                await cell_models.cell_query_llm("Say ok.")
            texts.append(str(info.value))
    assert texts[0] == texts[1]
    assert "something else entirely" not in texts[1]


@pytest.mark.parametrize("make, status, type_name", CASES, ids=IDS)
@pytest.mark.asyncio
async def test_a_cells_unillm_client_raises_it_too(session, make, status, type_name):
    failed = call_failed_class()
    model = ScriptedModel(query_llm=[make()])
    with scripted(model):
        client = cell_models.cell_unillm.AsyncUnify(LUNA)
        with pytest.raises(failed) as info:
            await client.generate(user_message="Say ok.")
    assert call_failed_fields(info.value) == (status, type_name)
    assert info.value.__cause__ is None and info.value.__context__ is None
    _assert_clean(str(info.value))


@pytest.mark.asyncio
async def test_a_request_error_is_still_a_model_request_error(session):
    model = ScriptedModel(query_llm=[bad_schema()])
    with scripted(model):
        with pytest.raises(cell_models.ModelRequestError) as info:
            await cell_models.cell_query_llm("Tag this.", response_format=SCHEMA)
    assert not isinstance(info.value, call_failed_class())


@pytest.mark.asyncio
async def test_the_harness_own_calls_raise_the_provider_error_as_before(session):
    errors = [rate_limited(), server_error()]
    model = ScriptedModel(query_llm=list(errors))
    with scripted(model):
        with pytest.raises(litellm.RateLimitError) as first:
            await reasoning.query_llm("Say ok.")
        with pytest.raises(litellm.InternalServerError) as second:
            await reasoning.query_llm("Say ok.")
    assert first.value is errors[0] and second.value is errors[1]


# ── the frame the harness sends to the sandboxed worker ─────────────────────────────────────────────────────


@pytest.mark.parametrize("make, status, type_name", CASES, ids=IDS)
@pytest.mark.asyncio
async def test_the_worker_frame_carries_the_fixed_error_only(session, make, status, type_name):
    from unify.actor.execution.worker import PythonWorker

    worker = PythonWorker()
    sent: list[dict] = []

    async def capture(msg: dict) -> None:
        sent.append(json.loads(json.dumps(msg)))

    worker._send = capture  # the frame as it would be written to the worker's channel
    worker._refs[1] = cell_models.cell_query_llm
    model = ScriptedModel(query_llm=[make()])
    with scripted(model):
        await worker._serve(
            {"op": "call", "id": 7, "target": {"ref": 1}, "args": ["Say ok."], "kwargs": {}},
            {},
        )
    (frame,) = sent
    assert frame["id"] == 7 and "value" not in frame
    assert frame["error"]["type"] == CALL_FAILED_NAME
    assert frame["error"]["module"] == cell_models.__name__
    _assert_clean(json.dumps(frame))


# ── in a cell of an act ─────────────────────────────────────────────────────────────────────────────────────


@contextlib.asynccontextmanager
async def _actor():
    from unify.actor.code_act_actor import CodeActActor

    actor = CodeActActor()
    yield actor


def _tool_results(call) -> str:
    out = []
    for m in call.messages:
        if m.get("role") != "tool":
            continue
        content = m.get("content")
        if isinstance(content, list):
            content = "\n".join(str(p.get("text", "")) if isinstance(p, dict) else str(p) for p in content)
        out.append(str(content))
    return "\n".join(out)


CAUGHT = (
    "for attempt in range(4):\n"
    "    try:\n"
    "        await query_llm('Say ok.')\n"
    "        print('NO ERROR')\n"
    f"    except {CALL_FAILED_NAME} as exc:\n"
    "        print('CAUGHT', type(exc).__name__, str(exc), isinstance(exc, ModelRequestError))\n"
    "print('AFTER')\n"
)
UNCAUGHT = "await query_llm('Say ok.')\n"
NEXT = "print('NEXT CELL')\n"


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(240)
async def test_in_a_cell_it_is_caught_by_name_and_nothing_planted_reaches_the_transcript(session):
    model = ScriptedModel(
        actor=[
            reply(calls=[cell(CAUGHT)]),
            reply(calls=[cell(UNCAUGHT)]),
            reply(calls=[cell(NEXT)]),
            reply("Done."),
        ],
        query_llm=[opaque_400(), access_400(), rate_limited(), server_error(), server_error()],
    )
    async with _actor() as actor:
        with scripted(model):
            handle = await actor.act("Ask the model.")
            assert await asyncio.wait_for(handle.result(), 180) == "Done."
    after_caught = _tool_results(model.calls[5])
    for status, type_name in [(400, "BadRequestError"), (400, "BadRequestError"), (429, "RateLimitError"),
                              (500, "InternalServerError")]:
        caught = [l for l in after_caught.splitlines() if l.startswith(f"CAUGHT {CALL_FAILED_NAME} ")]
        assert any(call_failed_text(status, type_name) in l and l.endswith(" False") for l in caught), after_caught[-2000:]
    assert "AFTER" in after_caught and "NO ERROR" not in after_caught
    # uncaught, it is the cell's error by name; the next cell runs
    shown = _tool_results(model.calls[-1])
    assert CALL_FAILED_NAME in shown and "NEXT CELL" in shown, shown[-2000:]
    # nothing planted in anything the model was sent
    for call in model.of("actor"):
        _assert_clean(json.dumps(call.request))
    model.assert_used_up()


# ── the access markers ──────────────────────────────────────────────────────────────────────────────────────

ACCESS_WORDS = ("credit", "quota", "billing", "payment", "balance")


def _with_word(where: str, word: str) -> dict:
    """A 400 body that is a request-validation error in every respect but the one *word* in *where*."""
    error = {
        "message": "Invalid value for 'max_tokens'",
        "type": "invalid_request_error",
        "code": "invalid_value",
        "param": "max_tokens",
    }
    if where == "type":
        error["type"] = f"insufficient_{word}"
    elif where == "code":
        error["code"] = f"invalid_{word}"
    elif where == "message":
        error["message"] = f"Your {word.capitalize()} does not cover 'max_tokens'"
    elif where == "nested":
        error["message"] = f"Insufficient {word} for this request"
        return routed({"error": error})
    return {"error": error}


PLACES = ("type", "code", "message", "nested")


@pytest.mark.parametrize("where", PLACES)
def test_without_an_access_word_the_body_is_a_request_error(where):
    # the control: the same bodies with a neutral word are the cell's own request errors
    refused = cell_models.model_request_error(bad_schema(_with_word(where, "format")))
    assert isinstance(refused, cell_models.ModelRequestError)


@pytest.mark.parametrize("where", PLACES)
@pytest.mark.parametrize("word", ACCESS_WORDS)
def test_a_400_naming_an_access_word_is_not_a_request_error(word, where):
    assert cell_models.model_request_error(bad_schema(_with_word(where, word))) is None


@pytest.mark.parametrize("where", PLACES)
@pytest.mark.parametrize("word", ACCESS_WORDS)
@pytest.mark.asyncio
async def test_a_cell_gets_model_call_failed_for_it(session, word, where):
    model = ScriptedModel(query_llm=[bad_schema(_with_word(where, word))])
    with scripted(model):
        with cell_models.counting_model_request_errors() as count:
            with pytest.raises(call_failed_class()) as info:
                await cell_models.cell_query_llm("Say ok.")
    assert not isinstance(info.value, cell_models.ModelRequestError)
    assert call_failed_fields(info.value) == (400, "BadRequestError")
    assert count.n == 0

"""Symbolic: a provider's refusal of a cell's own model request comes back to the cell.

A cell's ``query_llm`` (or a ``unillm`` client it built) whose request the
provider refuses for the request's own content (here a ``response_format``
schema with an array that has no ``items``) raises ``ModelRequestError``, a
``ValueError`` carrying the provider's message and nothing else, and the cell
goes on. The harness's own calls, and an authentication failure from a cell,
raise exactly what they raised before. With
``UNIFY_REQUEST_METADATA_HEADERS=on`` every cell call carries
``X-Unify-Call-Kind: cell``.

The provider is faked below unillm (``litellm.acompletion``, under unillm's
retry policy, ``tests/scripted_model.py``) with the error LiteLLM raises for
the recorded 400 body; nothing leaves the process.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from typing import Any

import litellm
import pytest

from tests.scripted_model import ScriptedModel, cell, reply, scripted
from unify.common import cell_models, llm_client, reasoning
from unify.session_details import SESSION_DETAILS
from unify.settings import SETTINGS

LUNA = "openai/gpt-6-luna@openrouter"
PROVIDER_MESSAGE = (
    "Invalid schema for response_format 'answer': In context=('properties', "
    "'tags'), array schema missing items"
)
BODY = {
    "error": {
        "message": PROVIDER_MESSAGE,
        "type": "invalid_request_error",
        "param": "response_format",
        "code": "invalid_json_schema",
    },
}
# The cell's own schema: an array with no ``items``.
SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "answer",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"tags": {"type": "array"}},
            "required": ["tags"],
            "additionalProperties": False,
        },
    },
}


def bad_schema(body: dict = BODY) -> litellm.BadRequestError:
    """What LiteLLM raises for OpenRouter's 400 with *body*: the body's text in
    the message, no parsed body."""
    return litellm.BadRequestError(
        message=f"OpenrouterException - {json.dumps(body)}",
        model="openrouter/openai/gpt-6-luna",
        llm_provider="openrouter",
    )


def routed(body: dict = BODY) -> dict:
    """*body* as OpenRouter forwards an upstream provider's error: its text in
    ``metadata.raw``."""
    return {
        "error": {
            "message": "Provider returned error",
            "code": 400,
            "metadata": {"raw": json.dumps(body), "provider_name": "OpenAI"},
        },
    }


def unauthorised() -> litellm.AuthenticationError:
    return litellm.AuthenticationError(
        message='OpenrouterException - {"error":{"message":"No auth credentials found","code":401}}',
        model="openrouter/openai/gpt-6-luna",
        llm_provider="openrouter",
    )


class HeaderModel(ScriptedModel):
    """A scripted model that also keeps each request's ``extra_headers``."""

    def __init__(self, **scripts: Any) -> None:
        super().__init__(**scripts)
        self.headers: list[dict] = []

    async def __call__(self, *, shared_session=None, client=None, **kw: Any) -> Any:
        self.headers.append(dict(kw.get("extra_headers") or {}))
        return await super().__call__(
            shared_session=shared_session,
            client=client,
            **kw,
        )

    def headers_of(self, kind: str) -> list[dict]:
        return [h for c, h in zip(self.calls, self.headers) if c.kind == kind]


@pytest.fixture
def session(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(SETTINGS, "UNIFY_MODEL", LUNA)
    monkeypatch.setattr(SETTINGS, "UNIFY_CELL_LLM_MODELS", "")
    monkeypatch.setattr(SESSION_DETAILS.assistant, "default_model", "")


@pytest.fixture
def headers_on(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_REQUEST_METADATA_HEADERS", "on")


# ── which errors are the cell's own ─────────────────────────────────────────


@pytest.mark.parametrize("body", [BODY, routed(BODY)], ids=["direct", "routed"])
def test_a_request_validation_400_is_a_model_request_error(body) -> None:
    refused = cell_models.model_request_error(bad_schema(body))
    assert isinstance(refused, cell_models.ModelRequestError)
    assert isinstance(refused, ValueError)
    assert str(refused) == PROVIDER_MESSAGE
    assert refused.args == (PROVIDER_MESSAGE,)


def test_a_parsed_sdk_body_is_read_too() -> None:
    exc = bad_schema()
    exc.body = BODY["error"]
    exc.message = "litellm.BadRequestError: no body in the text"
    assert str(cell_models.model_request_error(exc)) == PROVIDER_MESSAGE


@pytest.mark.parametrize(
    "error",
    [
        # access, whatever status or shape
        unauthorised(),
        bad_schema(
            {
                "error": {
                    "message": "API key not valid. Please pass a valid API key.",
                    "type": "invalid_request_error",
                },
            },
        ),
        bad_schema({"error": {"message": "x", "code": "invalid_api_key"}}),
        # a 400 that says nothing of the request's content
        bad_schema({"error": {"message": "Provider returned error", "code": 400}}),
        litellm.BadRequestError(
            message="OpenrouterException - upstream failed",
            model="m",
            llm_provider="openrouter",
        ),
        # not a 400
        litellm.InternalServerError(
            message=f"OpenrouterException - {json.dumps(BODY)}",
            model="m",
            llm_provider="openrouter",
        ),
        ValueError(json.dumps(BODY)),
    ],
    ids=["401", "bad-key-400", "key-code", "opaque", "no-body", "500", "no-status"],
)
def test_other_errors_are_not(error) -> None:
    assert cell_models.model_request_error(error) is None


def test_the_message_carries_no_url_or_key() -> None:
    message = (
        "Invalid value for 'temperature' (see https://provider.example/docs?x=1); "
        "request from sk-planted-key-for-the-redaction-test"
    )
    refused = cell_models.model_request_error(
        bad_schema(
            {
                "error": {
                    "message": message,
                    "type": "invalid_request_error",
                    "param": "temperature",
                },
            },
        ),
    )
    text = str(refused)
    assert text.startswith("Invalid value for 'temperature' (see <url>")
    assert "provider.example" not in text and "://" not in text
    assert "sk-planted-key-for-the-redaction-test" not in text


# ── a cell's query_llm ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_the_cells_query_llm_raises_it_and_counts_it(session) -> None:
    error = bad_schema()
    model = ScriptedModel(query_llm=[error, reply('{"tags": ["a"]}')])
    with scripted(model, below_retry=True):
        with cell_models.counting_model_request_errors() as count:
            with pytest.raises(cell_models.ModelRequestError) as info:
                await cell_models.cell_query_llm("Tag this.", response_format=SCHEMA)
            # the cell goes on: a fixed request in the same block runs
            fixed = json.loads(json.dumps(SCHEMA))
            fixed["json_schema"]["schema"]["properties"]["tags"]["items"] = {
                "type": "string",
            }
            assert await cell_models.cell_query_llm(
                "Tag this.",
                response_format=fixed,
            ) == {"tags": ["a"]}
    assert str(info.value) == PROVIDER_MESSAGE
    # raised apart from the provider's exception: no cause, no context
    assert info.value.__cause__ is None and info.value.__context__ is None
    assert count.n == 1
    # sent once: a 400 is not retried
    assert model.kinds() == ["query_llm", "query_llm"]
    assert model.calls[0].request["response_format"] == SCHEMA


@pytest.mark.asyncio
async def test_an_authentication_failure_is_raised_as_before(session) -> None:
    error = unauthorised()
    model = ScriptedModel(query_llm=[error])
    with scripted(model, below_retry=True):
        with cell_models.counting_model_request_errors() as count:
            with pytest.raises(litellm.AuthenticationError) as info:
                await cell_models.cell_query_llm("Tag this.", response_format=SCHEMA)
    assert info.value is error
    assert count.n == 0


@pytest.mark.asyncio
async def test_the_harness_own_calls_raise_the_400_as_before(session) -> None:
    errors = [bad_schema(), bad_schema()]
    model = ScriptedModel(query_llm=list(errors))
    with scripted(model, below_retry=True):
        with cell_models.counting_model_request_errors() as count:
            # the harness's query_llm (not the cell's global)
            with pytest.raises(litellm.BadRequestError) as first:
                await reasoning.query_llm("Tag this.", response_format=SCHEMA)
            # and a client the harness built
            client = llm_client.new_llm_client(origin="memory_writer")
            with pytest.raises(litellm.BadRequestError) as second:
                await client.generate(
                    user_message="Tag this.",
                    response_format=SCHEMA,
                )
    assert first.value is errors[0] and second.value is errors[1]
    assert not isinstance(first.value, cell_models.ModelRequestError)
    assert count.n == 0


@pytest.mark.asyncio
async def test_a_cells_unillm_client_raises_it_too(session) -> None:
    model = ScriptedModel(query_llm=[bad_schema()])
    with scripted(model, below_retry=True):
        with cell_models.counting_model_request_errors() as count:
            client = cell_models.cell_unillm.AsyncUnify(LUNA)
            with pytest.raises(cell_models.ModelRequestError, match="missing items"):
                await client.generate(user_message="Tag this.", response_format=SCHEMA)
    assert count.n == 1


# ── the call-kind header ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_cell_calls_carry_the_cell_call_kind(session, headers_on) -> None:
    model = HeaderModel(query_llm=[bad_schema(), reply("ok"), reply("ok")])
    with scripted(model, below_retry=True):
        with pytest.raises(cell_models.ModelRequestError):
            await cell_models.cell_query_llm("Tag this.", response_format=SCHEMA)
        # an origin the cell names does not change the kind
        await cell_models.cell_query_llm("Say ok.", origin="memory_writer")
        client = cell_models.cell_unillm.AsyncUnify(LUNA)
        await client.generate(user_message="Say ok.")
    kinds = [h.get(llm_client.CALL_KIND_HEADER) for h in model.headers]
    assert kinds == [llm_client.CELL_CALL_KIND] * 3 == ["cell"] * 3
    for headers in model.headers:
        assert all(llm_client.HEADER_VALUE.match(v) for v in headers.values())


@pytest.mark.asyncio
async def test_harness_calls_keep_their_origin_as_call_kind(session, headers_on):
    model = HeaderModel(query_llm=[reply("ok"), reply("ok")])
    with scripted(model, below_retry=True):
        await reasoning.query_llm("Say ok.")
        await llm_client.new_llm_client(origin="memory_writer").generate(
            user_message="Say ok.",
        )
    kinds = [h.get(llm_client.CALL_KIND_HEADER) for h in model.headers]
    assert kinds == ["CodeActActor.query_llm", "memory_writer"]


# ── in a cell of an act ─────────────────────────────────────────────────────


@contextlib.asynccontextmanager
async def _actor():
    from unify.actor.code_act_actor import CodeActActor

    actor = CodeActActor()
    try:
        yield actor
    finally:
        await actor.close()


def _tool_results(call) -> str:
    out = []
    for m in call.messages:
        if m.get("role") != "tool":
            continue
        content = m.get("content")
        if isinstance(content, list):
            content = "\n".join(
                str(p.get("text", "")) if isinstance(p, dict) else str(p)
                for p in content
            )
        out.append(str(content))
    return "\n".join(out)


CAUGHT = (
    f"schema = {SCHEMA!r}\n"
    "try:\n"
    "    await query_llm('Tag this.', response_format=schema)\n"
    "    print('NO ERROR')\n"
    "except ModelRequestError as exc:\n"
    "    print('CAUGHT', isinstance(exc, ValueError), exc)\n"
    "print('AFTER')\n"
)
UNCAUGHT = "await query_llm('Tag this.', response_format=schema)\n"
NEXT = "print('NEXT CELL', 'schema' in dir())\n"


@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_in_a_cell_the_error_is_caught_and_the_cells_go_on(
    session,
    headers_on,
):
    model = HeaderModel(
        actor=[
            reply(calls=[cell(CAUGHT)]),
            reply(calls=[cell(UNCAUGHT)]),
            reply(calls=[cell(NEXT)]),
            reply("Done."),
        ],
        query_llm=[bad_schema(), bad_schema()],
    )
    async with _actor() as actor:
        with scripted(model, below_retry=True):
            handle = await actor.act("Tag it.", can_store=False)
            assert await asyncio.wait_for(handle.result(), 120) == "Done."
    model.assert_used_up()
    after_caught = _tool_results(model.calls[2])
    assert f"CAUGHT True {PROVIDER_MESSAGE}" in after_caught, after_caught[-2000:]
    assert "AFTER" in after_caught and "NO ERROR" not in after_caught
    # uncaught, it is the cell's error, with the provider's message; the next
    # cell runs in the same session
    shown = _tool_results(model.calls[-1])
    assert "ModelRequestError" in shown and PROVIDER_MESSAGE in shown, shown[-2000:]
    assert "NEXT CELL True" in shown, shown[-2000:]
    assert "OpenrouterException" not in shown and "litellm" not in shown
    assert [h[llm_client.CALL_KIND_HEADER] for h in model.headers_of("query_llm")] == [
        "cell",
        "cell",
    ]
    assert all(
        h[llm_client.CALL_KIND_HEADER] != "cell" for h in model.headers_of("actor")
    )

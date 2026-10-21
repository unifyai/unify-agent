"""Symbolic: a provider's refusal of a cell's own model request comes back to the cell.

A cell's ``query_llm`` (or a ``unillm`` client it built) whose request the
provider refuses for the request's own content (here a ``response_format``
schema with an array that has no ``items``) raises ``ModelRequestError``, a
``ValueError`` carrying the provider's message and nothing else, and the cell
goes on. Any other provider failure of a cell's call (an authentication
failure, an opaque 400, a rate limit, a 5xx) raises ``ModelCallFailed``, a
fixed text with the status and the error's type and nothing of the provider's
reply. The harness's own calls raise exactly what they raised before. Every
cell call carries ``X-Unify-Call-Kind: cell``.

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


BILLING_WORDS = ("credit", "quota", "billing", "payment", "balance")


def _billing_bodies(word: str) -> dict:
    """A request-validation 400 carrying *word* in its type, its code, its message, and in an upstream error
    routed in ``metadata.raw``."""
    upper = word.capitalize()
    return {
        "type": {
            "error": {
                "message": "The request was not accepted.",
                "type": f"{word}_error",
                "code": "invalid_value",
                "param": "model",
            },
        },
        "code": {
            "error": {
                "message": "The request was not accepted.",
                "type": "invalid_request_error",
                "code": f"insufficient_{word}",
            },
        },
        "message": {
            "error": {
                "message": f"{upper} too low for this request; add more and retry.",
                "type": "invalid_request_error",
                "param": "max_tokens",
                "code": "invalid_value",
            },
        },
        "routed": routed(
            {
                "error": {
                    "message": f"Your {word} does not cover this request.",
                    "type": "invalid_request_error",
                    "code": "invalid_value",
                },
            },
        ),
    }


@pytest.mark.parametrize("where", ["type", "code", "message", "routed"])
@pytest.mark.parametrize("word", BILLING_WORDS)
def test_a_400_about_billing_is_not_a_request_error(word, where) -> None:
    error = bad_schema(_billing_bodies(word)[where])
    assert cell_models.model_request_error(error) is None
    # the cell gets the fixed text instead
    assert isinstance(cell_models.model_call_failed(error), cell_models.ModelCallFailed)


def test_the_billing_bodies_are_request_errors_without_the_word() -> None:
    # the same bodies with a neutral word are the cell's own request errors: the word alone decides
    for body in _billing_bodies("budget").values():
        assert isinstance(
            cell_models.model_request_error(bad_schema(body)),
            cell_models.ModelRequestError,
        ), body


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


# ── every other provider failure: a fixed text ─────────────────────────────

# A provider reply that must never reach a cell: a provider name, a user id, a
# URL and a key fingerprint (planted, readable test strings).
PLANTED = {
    "provider": "planted-provider-name-for-the-test",
    "user": "planted-user-id-for-the-test",
    "url": "https://planted.example/for-the-test?q=1",
    "fingerprint": "planted-key-fingerprint-for-the-test",
}
PLANTED_BODY = {
    "error": {
        "message": f"Upstream refused for user {PLANTED['user']}; see {PLANTED['url']}",
        "code": 400,
        "metadata": {
            "provider_name": PLANTED["provider"],
            "user_id": PLANTED["user"],
            "key_fingerprint": PLANTED["fingerprint"],
            "raw": json.dumps(
                {"error": {"message": f"key {PLANTED['fingerprint']} has no access"}},
            ),
        },
    },
}


def planted(kind: type = litellm.BadRequestError, **extra: Any) -> Exception:
    """A LiteLLM error of *kind* whose text is the planted provider reply."""
    return kind(
        message=f"OpenrouterException - {json.dumps(PLANTED_BODY)}",
        model="openrouter/openai/gpt-6-luna",
        llm_provider="openrouter",
        **extra,
    )


def assert_nothing_planted(text: str) -> None:
    for value in (*PLANTED.values(), "provider_name", "user_id", "OpenrouterException"):
        assert value not in text, value


@pytest.mark.parametrize(
    "error, status",
    [
        (planted(), 400),
        (
            bad_schema(
                {
                    "error": {
                        "message": "API key not valid. Please pass a valid API key.",
                        "type": "invalid_request_error",
                    },
                },
            ),
            400,
        ),
        (unauthorised(), 401),
        (planted(litellm.RateLimitError), 429),
        (planted(litellm.InternalServerError), 500),
        (planted(litellm.ServiceUnavailableError), 503),
    ],
    ids=["opaque-400", "access-400", "401", "429", "500", "503"],
)
def test_an_unclassified_provider_error_is_a_fixed_text(error, status) -> None:
    assert cell_models.model_request_error(error) is None
    failed = cell_models.model_call_failed(error)
    assert isinstance(failed, cell_models.ModelCallFailed)
    assert isinstance(failed, RuntimeError)
    assert failed.status == status and failed.kind == type(error).__name__
    assert str(failed) == (
        f"the model call failed (status {status}, {type(error).__name__}); the "
        "provider's reply is not shown"
    )
    assert failed.args == (str(failed),)
    assert_nothing_planted(str(failed) + repr(failed))


def test_an_error_that_is_not_a_providers_is_left_as_it_is() -> None:
    assert cell_models.model_call_failed(ValueError(json.dumps(PLANTED_BODY))) is None
    assert cell_models.model_call_failed(TypeError("bad argument")) is None


@pytest.mark.asyncio
async def test_the_cells_calls_raise_it_with_nothing_chained(session) -> None:
    model = ScriptedModel(query_llm=[planted(), planted()])
    with scripted(model, below_retry=True):
        with cell_models.counting_model_request_errors() as count:
            with pytest.raises(cell_models.ModelCallFailed) as first:
                await cell_models.cell_query_llm("Tag this.")
            client = cell_models.cell_unillm.AsyncUnify(LUNA)
            with pytest.raises(cell_models.ModelCallFailed) as second:
                await client.generate(user_message="Tag this.")
    for info in (first, second):
        assert info.value.status == 400
        assert info.value.__cause__ is None and info.value.__context__ is None
        assert info.value.__suppress_context__ is False
        assert_nothing_planted(str(info.value))
    assert count.n == 0


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
async def test_an_authentication_failure_is_a_fixed_text(session) -> None:
    model = ScriptedModel(query_llm=[unauthorised()])
    with scripted(model, below_retry=True):
        with cell_models.counting_model_request_errors() as count:
            with pytest.raises(cell_models.ModelCallFailed) as info:
                await cell_models.cell_query_llm("Tag this.", response_format=SCHEMA)
    assert str(info.value) == (
        "the model call failed (status 401, AuthenticationError); the provider's "
        "reply is not shown"
    )
    assert info.value.__cause__ is None and info.value.__context__ is None
    # not a request error of the cell's own: not counted as one
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
async def test_cell_calls_carry_the_cell_call_kind(session) -> None:
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
async def test_harness_calls_keep_their_origin_as_call_kind(session):
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
    yield actor


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
NEXT = "print('NEXT CELL', schema['type'])\n"


@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_in_a_cell_the_error_is_caught_and_the_cells_go_on(session):
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
            handle = await actor.act("Tag it.")
            assert await asyncio.wait_for(handle.result(), 120) == "Done."
    model.assert_used_up()
    after_caught = _tool_results(model.calls[2])
    assert f"CAUGHT True {PROVIDER_MESSAGE}" in after_caught, after_caught[-2000:]
    assert "AFTER" in after_caught and "NO ERROR" not in after_caught
    # uncaught, it is the cell's error, with the provider's message; the next
    # cell runs in the same session
    shown = _tool_results(model.calls[-1])
    assert "ModelRequestError" in shown and PROVIDER_MESSAGE in shown, shown[-2000:]
    assert "NEXT CELL json_schema" in shown, shown[-2000:]
    assert "OpenrouterException" not in shown and "litellm" not in shown
    assert [h[llm_client.CALL_KIND_HEADER] for h in model.headers_of("query_llm")] == [
        "cell",
        "cell",
    ]
    assert all(
        h[llm_client.CALL_KIND_HEADER] != "cell" for h in model.headers_of("actor")
    )

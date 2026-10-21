"""An ``LLM`` event never carries the credentials of the request it reports.

unillm reports the request it sent, transport ``api_key`` and authorization
headers included, and ``unify.events.llm_event_hook`` publishes it to the
EventBus, whose subscribers may keep it. These tests publish requests holding
obviously fake credentials and check that no subscriber sees them, that the
dict unillm passed in is unchanged, and that the request actually sent still
carries them. Nothing here reaches a model or needs a key.
"""

from __future__ import annotations

import asyncio
import copy
import json
import time

import pytest
from unillm import LLMEvent

from tests.helpers import capture_events
from unify.common.redact_request import REDACTED, redact_llm_request
from unify.events.llm_event_hook import _llm_event_to_eventbus, install_llm_event_hook

FAKE_KEY = "sk-or-FAKE-0000000000000000000000000000"  # pragma: allowlist secret
FAKE_ANT = "sk-ant-FAKE-1111111111111111111111111111"  # pragma: allowlist secret
FAKE_GOOGLE = "AIzaFAKE22222222222222222222222222222222"  # pragma: allowlist secret
FAKE_PASSWORD = "FAKE-url-password-3333"  # pragma: allowlist secret
FAKE_BEARER = "FAKE-bearer-token-4444"  # pragma: allowlist secret
FAKE_COOKIE = "session=FAKE-cookie-5555"  # pragma: allowlist secret
FAKE_SECRETS = (
    FAKE_KEY,
    FAKE_ANT,
    FAKE_GOOGLE,
    FAKE_PASSWORD,
    FAKE_BEARER,
    FAKE_COOKIE,
)


def _leaked(data) -> list[str]:
    text = json.dumps(data, default=str)
    return [secret for secret in FAKE_SECRETS if secret in text]


async def _published(captured: list, count: int = 1) -> list:
    """Wait (bounded) until the hook's scheduled publish has reached the sink."""
    for _ in range(200):
        if len(captured) >= count:
            break
        await asyncio.sleep(0.01)
    return captured


@pytest.mark.asyncio
async def test_hook_publishes_no_credential_and_leaves_the_request_unchanged():
    request = {
        "model": "openai/gpt-5.6-sol@openrouter",
        "messages": [{"role": "user", "content": "Hi"}],
        "api_key": FAKE_KEY,
        "base_url": f"https://gwuser:{FAKE_PASSWORD}@gateway.invalid/v1",
        "extra_headers": {
            "Authorization": f"Bearer {FAKE_BEARER}",
            "X-Title": "unify",
        },
    }
    original = copy.deepcopy(request)

    async with capture_events("LLM") as captured:
        _llm_event_to_eventbus(LLMEvent(request=request))
        await _published(captured)

    assert len(captured) == 1
    seen = captured[0].payload["request"]
    assert _leaked(seen) == []
    assert _leaked(captured[0].payload) == []
    assert seen["api_key"] == REDACTED
    assert seen["base_url"] == f"https://{REDACTED}@gateway.invalid/v1"
    assert seen["extra_headers"] == {"Authorization": REDACTED, "X-Title": "unify"}
    assert seen["model"] == request["model"]
    assert seen["messages"] == request["messages"]
    # The dict unillm passed in is untouched.
    assert request == original


def test_nested_headers_and_key_shaped_values_are_redacted():
    request = {
        "model": "openai/gpt-5.6-sol@openrouter",
        "max_completion_tokens": 100,
        "prompt_cache_key": "session-affinity-0001",
        "OPENAI_API_KEY": FAKE_KEY,
        "x-api-key": FAKE_ANT,
        "anthropicApiKey": FAKE_ANT,
        "extra_body": {
            "provider": {
                "headers": {
                    "Proxy-Authorization": f"Basic {FAKE_BEARER}",
                    "COOKIE": FAKE_COOKIE,
                    "api-key": FAKE_KEY,
                    "Accept": "application/json",
                },
            },
            "webhook_url": f"https://user:{FAKE_PASSWORD}@hooks.invalid/x",
        },
        "default_headers": [{"authorization": f"Bearer {FAKE_BEARER}"}],
        # Key-shaped values under innocent field names, whole and inside text.
        "metadata": {"note": FAKE_GOOGLE, "tags": ["a", FAKE_ANT]},
        "messages": [
            {"role": "user", "content": f"my key is {FAKE_KEY}, keep it safe"},
        ],
        "api_key": None,
    }
    original = copy.deepcopy(request)

    out = redact_llm_request(request)

    assert _leaked(out) == []
    assert request == original
    headers = out["extra_body"]["provider"]["headers"]
    assert headers == {
        "Proxy-Authorization": REDACTED,
        "COOKIE": REDACTED,
        "api-key": REDACTED,
        "Accept": "application/json",
    }
    assert out["OPENAI_API_KEY"] == out["x-api-key"] == REDACTED
    assert out["anthropicApiKey"] == REDACTED
    assert out["default_headers"] == [{"authorization": REDACTED}]
    assert out["extra_body"]["webhook_url"] == f"https://{REDACTED}@hooks.invalid/x"
    assert out["metadata"] == {"note": REDACTED, "tags": ["a", REDACTED]}
    assert out["messages"][0]["content"] == f"my key is {REDACTED}, keep it safe"
    # What carries no credential is published as it was.
    assert out["model"] == request["model"]
    assert out["max_completion_tokens"] == 100
    assert out["prompt_cache_key"] == "session-affinity-0001"
    assert out["api_key"] is None
    # New containers all the way down: nothing is shared with the request.
    assert out["extra_body"] is not request["extra_body"]
    assert out["messages"] is not request["messages"]
    assert out["messages"][0] is not request["messages"][0]


@pytest.mark.asyncio
async def test_the_request_sent_keeps_its_credentials(monkeypatch):
    """The redaction applies to the event's copy only: the transport kwargs
    unillm sends (gateway ``api_key`` and ``api_base`` with userinfo, caller's
    ``extra_headers``) are what they would be without it, and stay unchanged
    after the event is published."""
    import unillm
    import unillm.clients.uni_llm as uni_llm

    from tests.cache_discipline_helpers import MODEL, completion

    gateway_url = f"https://gwuser:{FAKE_PASSWORD}@gateway.invalid/v1"
    monkeypatch.setenv("UNILLM_LLM_GATEWAY_URL", gateway_url)
    monkeypatch.setenv("UNILLM_LLM_GATEWAY_KEY", FAKE_KEY)
    monkeypatch.setenv("UNILLM_CACHE", "false")

    sent: list[tuple[dict, dict]] = []

    async def transport(*, shared_session=None, client=None, **kw):
        # The kwargs as received (sharing nested objects with unillm's copy)
        # and a snapshot of them at send time.
        sent.append((kw, copy.deepcopy(kw)))
        return completion("ok")

    monkeypatch.setattr(uni_llm, "_acompletion_with_transient_retry", transport)
    install_llm_event_hook()

    async with capture_events("LLM") as captured:
        client = unillm.AsyncUnify(MODEL, cache=False)
        await client.generate(
            messages=[{"role": "user", "content": "Say ok."}],
            extra_headers={"Authorization": f"Bearer {FAKE_BEARER}"},
        )
        await _published(captured)

    assert len(sent) == 1
    received, snapshot = sent[0]
    # The provider got the credentials...
    assert received["api_key"] == FAKE_KEY
    assert received["api_base"] == gateway_url
    assert received["extra_headers"]["Authorization"] == f"Bearer {FAKE_BEARER}"
    # ...and publishing the event changed nothing it was sent.
    assert received == snapshot

    assert len(captured) == 1
    seen = captured[0].payload["request"]
    assert _leaked(seen) == []
    assert seen["api_key"] == REDACTED
    assert seen["api_base"] == f"https://{REDACTED}@gateway.invalid/v1"
    assert seen["extra_headers"]["Authorization"] == REDACTED
    assert seen["messages"] == snapshot["messages"]


# Long words of the shapes a URL scheme, a key or a userinfo is made of. A
# compaction summary of this size blocked the event loop for seconds when the
# userinfo pattern started at an unbounded scheme (quadratic in the word).
LONG = 120_000
LONG_TEXTS = (
    "x" * LONG,
    "a1+.-" * (LONG // 5),
    "https://" + "u" * LONG,
    "xsk-" * (LONG // 4),
    ("word " * (LONG // 5)) + "x" * LONG,
)


@pytest.mark.parametrize("index", range(len(LONG_TEXTS)))
def test_long_text_is_redacted_in_linear_time(index):
    text = LONG_TEXTS[index]
    request = {
        "model": "openai/gpt-5.6-sol@openrouter",
        "messages": [{"role": "user", "content": text}] * 3,
    }
    started = time.perf_counter()
    out = redact_llm_request(request)
    elapsed = time.perf_counter() - started
    # Linear redaction takes milliseconds; the quadratic one took tens of
    # seconds on the first of these.
    assert elapsed < 1.0, elapsed
    assert out == request


@pytest.mark.asyncio
async def test_the_hook_keeps_a_long_request_whole_and_does_not_block_the_loop():
    """The step-cap compaction tests send summaries of 40,000 and 80,000
    characters; the hook redacts them on the loop's thread, so it must
    return at once and keep every field the event's readers use."""
    request = {
        "model": "openai/gpt-5.6-sol@openrouter",
        "messages": [{"role": "user", "content": "x" * (2 * LONG)}],
        "max_completion_tokens": 100,
        "prompt_cache_key": "session-affinity-0001",
    }
    async with capture_events("LLM") as captured:
        started = time.perf_counter()
        _llm_event_to_eventbus(LLMEvent(request=request))
        elapsed = time.perf_counter() - started
        await _published(captured)
    assert elapsed < 1.0, elapsed
    assert len(captured) == 1
    seen = captured[0].payload["request"]
    assert "redaction_failed" not in seen
    assert seen == request


@pytest.mark.parametrize(
    "text, expected",
    [
        (
            f"https://gwuser:{FAKE_PASSWORD}@gateway.invalid/v1",
            f"https://{REDACTED}@gateway.invalid/v1",
        ),
        (
            f"see git+ssh://{FAKE_BEARER}@host.invalid/repo and more",
            f"see git+ssh://{REDACTED}@host.invalid/repo and more",
        ),
        (
            f"wordglued{'x' * 100}https://u:{FAKE_PASSWORD}@h.invalid",
            f"wordglued{'x' * 100}https://{REDACTED}@h.invalid",
        ),
        ("https://host.invalid/a@b", "https://host.invalid/a@b"),
        ("mail me at someone@example.invalid", "mail me at someone@example.invalid"),
    ],
)
def test_url_userinfo_is_redacted_and_the_scheme_kept(text, expected):
    assert redact_llm_request({"content": text}) == {"content": expected}

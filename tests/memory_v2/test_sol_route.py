"""Sol's own route: ``UNIFY_MEMORY_V2_SOL_BASE_URL`` and ``UNIFY_MEMORY_V2_SOL_TOKEN``.

Unset, Sol's call is built as shipped. Both set, it goes to that base URL with that token and carries
``X-Unify-Call-Kind: memory_v2.sol``. One set, or a value in a wrong form, starts no pass. Neither name reaches a
cell's environment or Sol's box, and no repr, str, log line or error shows the token.

Credentials here are placeholders, and even those are compared by digest and checked through booleans computed
before the assert, so a failing assert never prints one. No test makes a model call: the client is a fake, and
unillm's request preparation (``_prepare_provider_request_kw``, a pure function) shows where the call would go.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import json
import logging
import os
import secrets
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import SecretStr

from unify import sandbox
from unify.memory_v2 import sol_pass
from unify.memory_v2.integration import consolidate, switch
from unify.memory_v2.sandbox_run import run_confined


@pytest.fixture(autouse=True)
def _otel_off_unless_a_test_turns_it_on(monkeypatch):
    """The route refuses while unillm records OTel spans; these tests must not depend on the host's
    ``UNILLM_OTEL`` (a host with it set made every case fail). A test that wants it on sets it itself."""
    unillm_logger = importlib.import_module("unillm.logger")
    monkeypatch.delenv("UNILLM_OTEL", raising=False)
    if hasattr(unillm_logger, "_OTEL_ENABLED"):
        monkeypatch.setattr(unillm_logger, "_OTEL_ENABLED", False)
from unify.memory_v2.sol_pass import SolRoute, unillm_turn

SOL_TOKEN = "sol-route-placeholder-token"  # pragma: allowlist secret
SOL_BASE = "http://127.0.0.1:18081/api/v1"
ACTOR_KEY = "actor-gateway-placeholder-key"  # pragma: allowlist secret
ACTOR_BASE = "http://127.0.0.1:18080/api/v1"
TRANSPORT_MODEL = "openrouter/openai/gpt-6-sol"


def _digest(value: object) -> str:
    return hashlib.sha256(str(value).encode()).hexdigest()[:16]


def _redacted(kw: dict) -> dict:
    """*kw* with its credential replaced by a digest, so a failing comparison never prints it."""
    out = dict(kw)
    if "api_key" in out:
        out["api_key"] = _digest(out["api_key"])
    return out


def _prepare(**extra) -> dict:
    """The transport kwargs unillm would send for an OpenRouter call in the current context."""
    from unillm.clients.uni_llm import _prepare_provider_request_kw

    kw = {
        "model": TRANSPORT_MODEL,
        "messages": [{"role": "user", "content": "x"}],
        **extra,
    }
    _prepare_provider_request_kw(kw=kw, provider="openrouter", stream=False)
    return kw


@pytest.fixture
def actor_gateway(monkeypatch):
    """The actor's own gateway (unillm's environment route), so a test can tell the two routes apart."""
    monkeypatch.setenv("UNILLM_LLM_GATEWAY_URL", ACTOR_BASE)
    monkeypatch.setenv("UNILLM_LLM_GATEWAY_KEY", ACTOR_KEY)


@pytest.fixture
def restore_unillm(monkeypatch):
    """Whatever this test installs over unillm's gateway lookup is undone after it."""
    from unillm.clients import uni_llm

    monkeypatch.setattr(uni_llm, "_llm_gateway", uni_llm._llm_gateway)


def _fake_client(monkeypatch, on_generate):
    import unify.common.llm_client as llm_client

    built: dict = {}

    class FakeClient:
        messages: list = []

        async def generate(self, **kw):
            built.setdefault("generate", []).append(sorted(kw))
            on_generate(kw)
            self.messages = [
                *kw["messages"],
                {"role": "assistant", "content": "ok", "tool_calls": None},
            ]

    def fake_new(model, **kw):
        built["client"] = {"model": model, **kw}
        return FakeClient()

    monkeypatch.setattr(llm_client, "new_llm_client", fake_new)
    return built


def _run(turn):
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    return asyncio.run(turn(messages, []))


# --- unset: the shipped call ------------------------------------------------------------------------------


def test_unset_route_builds_the_shipped_call(monkeypatch, actor_gateway):
    from unillm.clients import uni_llm

    seen: dict = {}

    def on_generate(kw):
        seen["route"] = sol_pass._SOL_GATEWAY.get()
        seen["transport"] = _redacted(_prepare())

    built = _fake_client(monkeypatch, on_generate)
    _run(unillm_turn("openai/gpt-6-sol", "low"))

    # the client and the generate call exactly as at 9deefbfd1: no header, no route
    assert built["client"] == {
        "model": "openai/gpt-6-sol@openrouter",
        "origin": "memory_v2.sol",
        "reasoning_effort": "low",
        "stateful": True,
    }
    assert built["generate"] == [["messages", "stateful", "tool_choice", "tools"]]
    assert seen["route"] is None
    # the transport the shipped lookup builds: the actor's gateway and its key
    original = getattr(uni_llm._llm_gateway, "__wrapped__", uni_llm._llm_gateway)
    monkeypatch.setattr(uni_llm, "_llm_gateway", original)
    assert seen["transport"] == _redacted(_prepare())
    assert seen["transport"]["api_base"] == ACTOR_BASE
    assert seen["transport"]["api_key"] == _digest(ACTOR_KEY)


def test_unset_settings_give_no_route():
    cfg = consolidate.sol_settings(SimpleNamespace())
    assert cfg.route is None
    cfg = consolidate.sol_settings(
        SimpleNamespace(
            UNIFY_MEMORY_V2_SOL_BASE_URL="  ",
            UNIFY_MEMORY_V2_SOL_TOKEN=SecretStr(""),
        ),
    )
    assert cfg.route is None


def test_settings_default_to_no_route(monkeypatch):
    from unify.settings import ProductionSettings

    monkeypatch.delenv("UNIFY_MEMORY_V2_SOL_BASE_URL", raising=False)
    monkeypatch.delenv("UNIFY_MEMORY_V2_SOL_TOKEN", raising=False)
    s = ProductionSettings()
    assert s.UNIFY_MEMORY_V2_SOL_BASE_URL == ""
    assert s.UNIFY_MEMORY_V2_SOL_TOKEN.get_secret_value() == ""
    assert consolidate.sol_settings(s).route is None


# --- both set: Sol's route --------------------------------------------------------------------------------


@pytest.mark.parametrize("cost,usd", [(0.0125, "0.0125"), (None, "unknown")])
def test_route_sends_to_its_base_url_with_its_token_and_header(
    monkeypatch,
    actor_gateway,
    restore_unillm,
    cost,
    usd,
):
    from unillm.llm_events import LLMEvent, get_llm_event_hook

    seen: dict = {}

    def on_generate(kw):
        transport = _prepare(extra_headers=kw.get("extra_headers"))
        seen["transport"] = _redacted(transport)
        event = LLMEvent(
            request=dict(transport),
            provider_cost=cost,
            origin="memory_v2.sol",
        )
        # unillm calls the scoped hook before any process-wide listener
        get_llm_event_hook()(event)
        seen["event_has_key"] = "api_key" in event.request
        seen["event_base"] = event.request.get("api_base")

    built = _fake_client(monkeypatch, on_generate)
    route = SolRoute(SOL_BASE, SecretStr(SOL_TOKEN))
    msg, got = _run(unillm_turn("openai/gpt-6-sol", "low", route=route))

    assert msg["content"] == "ok" and got == usd
    assert built["client"] == {
        "model": "openai/gpt-6-sol@openrouter",
        "origin": "memory_v2.sol",
        "reasoning_effort": "low",
        "stateful": True,
    }
    assert built["generate"] == [
        ["extra_headers", "messages", "stateful", "tool_choice", "tools"],
    ]
    t = seen["transport"]
    assert t["api_base"] == SOL_BASE
    assert t["api_key"] == _digest(SOL_TOKEN)  # Sol's token, not the actor's key
    assert t["extra_headers"] == {"X-Unify-Call-Kind": "memory_v2.sol"}
    # pricing as shipped: the request still asks the OpenRouter API for its charged cost
    assert t["extra_body"]["usage"]["include"] is True
    # the event bus never receives the token
    assert seen["event_has_key"] is False and seen["event_base"] == SOL_BASE
    # after the call, this context is back on the actor's gateway
    assert sol_pass._SOL_GATEWAY.get() is None
    after = _redacted(_prepare())
    assert after["api_base"] == ACTOR_BASE and after["api_key"] == _digest(ACTOR_KEY)


def test_a_call_that_misses_the_route_fails_the_turn(monkeypatch, restore_unillm):
    from unillm.llm_events import LLMEvent, get_llm_event_hook

    def on_generate(kw):
        get_llm_event_hook()(
            LLMEvent(
                request={"api_base": ACTOR_BASE},
                provider_cost=0.01,
                origin="memory_v2.sol",
            ),
        )

    _fake_client(monkeypatch, on_generate)
    turn = unillm_turn(
        "openai/gpt-6-sol",
        "low",
        route=SolRoute(SOL_BASE, SecretStr(SOL_TOKEN)),
    )
    with pytest.raises(RuntimeError, match="declared route"):
        _run(turn)


def test_a_forged_header_does_not_select_the_route(actor_gateway, restore_unillm):
    """The header is an extra check, never the rule: outside a Sol call, a request carrying Sol's call kind
    still goes the actor's way, with the actor's key. On the proxy side, the actor's port applies the actor's
    allow-list whatever headers arrive, so a cell that forges the header reaches nothing new either.
    """
    sol_pass._install_sol_gateway()
    t = _redacted(_prepare(extra_headers={"X-Unify-Call-Kind": "memory_v2.sol"}))
    assert t["api_base"] == ACTOR_BASE
    assert t["api_key"] == _digest(ACTOR_KEY)


def test_both_set_gives_the_route():
    cfg = consolidate.sol_settings(
        SimpleNamespace(
            UNIFY_MEMORY_V2_SOL_BASE_URL=SOL_BASE + "/",
            UNIFY_MEMORY_V2_SOL_TOKEN=SecretStr(SOL_TOKEN),
        ),
    )
    assert cfg.route is not None
    assert cfg.route.base_url == SOL_BASE
    assert _digest(cfg.route.token.get_secret_value()) == _digest(SOL_TOKEN)


def test_the_route_needs_an_openrouter_endpoint():
    with pytest.raises(ValueError, match="@openrouter"):
        consolidate.sol_settings(
            SimpleNamespace(
                UNIFY_MEMORY_V2_SOL_MODEL="anthropic/claude@anthropic",
                UNIFY_MEMORY_V2_SOL_BASE_URL=SOL_BASE,
                UNIFY_MEMORY_V2_SOL_TOKEN=SecretStr(SOL_TOKEN),
            ),
        )


# --- one set: fail closed ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "base,token,missing",
    [
        (SOL_BASE, "", "UNIFY_MEMORY_V2_SOL_TOKEN"),
        ("", SOL_TOKEN, "UNIFY_MEMORY_V2_SOL_BASE_URL"),
    ],
)
def test_one_set_starts_no_pass_and_makes_no_call(monkeypatch, base, token, missing):
    made: list = []

    def no_turn(*a, **k):  # pragma: no cover - reaching it is the failure
        made.append(1)
        raise AssertionError("a turn was built")

    monkeypatch.setattr(consolidate, "unillm_turn", no_turn)
    settings = SimpleNamespace(
        UNIFY_MEMORY_V2_SOL_BASE_URL=base,
        UNIFY_MEMORY_V2_SOL_TOKEN=SecretStr(token),
    )
    with pytest.raises(ValueError) as info:
        asyncio.run(
            consolidate.run_due_passes(
                None,  # never reached: the settings are read first
                "e1",
                "0" * 40,
                SimpleNamespace(),
                effort="low",
                settings=settings,
                emit=None,
            ),
        )
    text = str(info.value)
    leaked = bool(token) and token in text
    assert f"{missing} is empty" in text and "no consolidation pass starts" in text
    assert not leaked
    assert made == []


# --- validators --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        "ftp://proxy.internal/v1",
        "proxy.internal:8080",
        "http://",
        "http://:8080/v1",
        "http://user:pass@proxy.internal/v1",  # pragma: allowlist secret
        "http://user@proxy.internal/v1",
        "http://proxy.internal/v1?key=abc",
        "http://proxy.internal/v1#frag",
        "http://proxy.internal:99999/v1",
        "http://proxy.internal:0/v1",
        "http://proxy internal/v1",
        "http://proxy.internal/v\n1",
        "http://proxy.internal/\x00v1",
        # plain http to any host but exactly 127.0.0.1: the token would travel in clear
        "http://proxy.internal/v1",
        "http://localhost:18081/sol/v1",
        "http://127.0.0.2:18081/sol/v1",
        "http://10.0.0.1:18081/sol/v1",
        "http://192.168.4.5/sol/v1",
        "http://[::1]:18081/sol/v1",
        "http://8.8.8.8/v1",
        "http://169.254.169.254/v1",
        "http://0.0.0.0:8080/v1",
        # plain http to 127.0.0.1 always names its port, as the launcher sends it
        "http://127.0.0.1/sol/v1",
        "http://127.0.0.1:/sol/v1",
    ],
)
def test_the_base_url_validator_refuses_bad_urls_without_quoting_them(raw):
    with pytest.raises(ValueError) as info:
        switch.parse_sol_base_url(raw)
    text = str(info.value)
    quoted = raw.strip() in text or any(
        part in text for part in ("pass@", "user@", "key=abc", "#frag")
    )
    assert "UNIFY_MEMORY_V2_SOL_BASE_URL" in text
    assert not quoted


@pytest.mark.parametrize(
    "raw,want",
    [
        ("", ""),
        ("  ", ""),
        (SOL_BASE, SOL_BASE),
        (SOL_BASE + "/", SOL_BASE),
        ("https://proxy.internal", "https://proxy.internal"),
        ("HTTPS://proxy.internal:8443/api/v1", "HTTPS://proxy.internal:8443/api/v1"),
        ("http://127.0.0.1:18081/sol/v1", "http://127.0.0.1:18081/sol/v1"),
        ("http://127.0.0.1:18081/sol/v1/", "http://127.0.0.1:18081/sol/v1"),
        ("https://localhost:8443/sol/v1", "https://localhost:8443/sol/v1"),
        ("https://10.0.0.1/sol/v1", "https://10.0.0.1/sol/v1"),
    ],
)
def test_the_base_url_validator_accepts_http_urls(raw, want):
    assert switch.parse_sol_base_url(raw) == want


@pytest.mark.parametrize(
    "raw",
    [
        "tok en-placeholder",  # pragma: allowlist secret
        "tok\nen-placeholder",  # pragma: allowlist secret
        "tok\x00en-placeholder",  # pragma: allowlist secret
        "tokén-placeholder",  # pragma: allowlist secret
        "short-placehold",  # pragma: allowlist secret
        'quote"d-placeholder-value',  # pragma: allowlist secret
        "padding=in-the-placeholder",  # pragma: allowlist secret
        "padded-the-placeholder==",  # pragma: allowlist secret
        # + and / percent-encode (%2B, %2F) or JSON-escape (\/) into forms value redaction would miss
        "plus+in-the-placeholder",  # pragma: allowlist secret
        "slash/in-the-placeholder",  # pragma: allowlist secret
    ],
)
def test_the_token_validator_refuses_unsendable_tokens_without_quoting_them(raw):
    with pytest.raises(ValueError) as info:
        switch.parse_sol_token(raw)
    leaked = "placeholder" in str(info.value)
    assert not leaked


def test_the_token_validator_accepts_the_launchers_url_safe_tokens():
    """The office key proxy issues ``secrets.token_urlsafe(32)``; 16 unreserved characters is the floor."""
    tokens = [secrets.token_urlsafe(32) for _ in range(200)] + [
        "a" * 16,
        "Az09._~-" * 2,
        SOL_TOKEN,
    ]
    kept = [
        _digest(switch.parse_sol_token(t).get_secret_value()) == _digest(t)
        for t in tokens
    ]
    assert all(kept)


def test_settings_never_refuse_or_echo_the_two_values(monkeypatch):
    """A settings error would print its input, so settings only normalise; the pass start checks."""
    from unify.settings import ProductionSettings

    with_password = "http://user:pass@proxy.internal/v1"  # pragma: allowlist secret
    unsendable = "bad token-placeholder"  # pragma: allowlist secret
    monkeypatch.setenv("UNIFY_MEMORY_V2_SOL_BASE_URL", with_password)
    monkeypatch.setenv("UNIFY_MEMORY_V2_SOL_TOKEN", unsendable)
    s = ProductionSettings()  # loads
    with pytest.raises(ValueError) as info:
        consolidate.sol_settings(s)
    text = str(info.value)
    leaked = "pass@" in text or "placeholder" in text
    assert not leaked


# --- the token never shows -----------------------------------------------------------------------------


def test_the_token_never_shows_in_repr_str_dump_or_log(monkeypatch, caplog):
    from unify.settings import ProductionSettings

    monkeypatch.setenv("UNIFY_MEMORY_V2_SOL_BASE_URL", SOL_BASE)
    monkeypatch.setenv("UNIFY_MEMORY_V2_SOL_TOKEN", SOL_TOKEN)
    s = ProductionSettings()
    route = consolidate.sol_settings(s).route
    assert route is not None
    held = _digest(s.UNIFY_MEMORY_V2_SOL_TOKEN.get_secret_value()) == _digest(SOL_TOKEN)
    assert held
    log = logging.getLogger("tests.memory_v2.sol_route")
    with caplog.at_level(logging.DEBUG):
        log.info("settings %r %s", s, s)
        log.info("route %r %s token %s", route, route, s.UNIFY_MEMORY_V2_SOL_TOKEN)
    shown = [
        repr(s),
        str(s),
        repr(route),
        str(route),
        repr(s.UNIFY_MEMORY_V2_SOL_TOKEN),
        str(s.UNIFY_MEMORY_V2_SOL_TOKEN),
        s.model_dump_json(),
        str(s.model_dump()),
        caplog.text,
    ]
    leaked = [i for i, text in enumerate(shown) if SOL_TOKEN in text]
    assert leaked == []


# --- neither name reaches a cell or Sol's box --------------------------------------------------------------


def test_a_cells_environment_never_holds_the_route(monkeypatch):
    monkeypatch.setenv("UNIFY_MEMORY_V2_SOL_BASE_URL", SOL_BASE)
    monkeypatch.setenv("UNIFY_MEMORY_V2_SOL_TOKEN", SOL_TOKEN)
    policy = SimpleNamespace(network="", proxy_port=None)
    # the worker's and a bash cell's environment
    inherited = sandbox.sandbox_env(policy)
    explicit = sandbox.sandbox_env(  # a cell's own env= for a subprocess
        policy,
        {
            "UNIFY_MEMORY_V2_SOL_BASE_URL": SOL_BASE,
            "unify_memory_v2_sol_base_url": SOL_BASE,
            "UNIFY_MEMORY_V2_SOL_TOKEN": SOL_TOKEN,
            "PATH": "/usr/bin",
        },
    )
    for env in (inherited, explicit, sandbox.scrubbed_env()):
        names = {n.upper() for n in env}
        values = set(env.values())
        assert not names & sandbox.HARNESS_ONLY_ENV
        assert SOL_BASE not in values
        assert not any(SOL_TOKEN in v for v in values)
    kept = {k: v for k, v in explicit.items() if k not in sandbox.TRUSTED_ENV}
    assert kept == {"PATH": "/usr/bin", "TMPDIR": "/tmp"}


def test_sols_box_refuses_the_route_in_its_environment():
    for name in ("UNIFY_MEMORY_V2_SOL_TOKEN", "UNIFY_MEMORY_V2_SOL_BASE_URL"):
        with pytest.raises(ValueError) as info:
            run_confined(["/bin/true"], env={name: SOL_TOKEN})
        leaked = SOL_TOKEN in str(info.value)
        assert not leaked


# --- concurrency: only Sol's own call takes Sol's route ------------------------------------------------------


def _is_actor(t: dict) -> bool:
    return t.get("api_base") == ACTOR_BASE and t.get("api_key") == _digest(ACTOR_KEY)


def _is_sol(t: dict) -> bool:
    return t.get("api_base") == SOL_BASE and t.get("api_key") == _digest(SOL_TOKEN)


def test_concurrent_actor_tasks_served_calls_and_threads_keep_the_actors_route(
    monkeypatch,
    actor_gateway,
    restore_unillm,
):
    """While Sol's call is parked mid-flight: a task created before it, a task created the way the worker
    serves a cell's ``query_llm`` (``worker._serve``: from the running cell's context), a thread and an
    executor job each prepare the actor's transport; Sol's call, before and after them, prepares its own.
    """
    import unify.common.llm_client as llm_client

    seen: dict = {}
    gates: dict = {}

    class Parked:
        messages: list = []

        async def generate(self, **kw):
            seen["sol"] = _redacted(_prepare(extra_headers=kw.get("extra_headers")))
            gates["in"].set()
            await gates["out"].wait()
            seen["sol_after"] = _redacted(_prepare())
            self.messages = [{"role": "assistant", "content": "ok"}]

    monkeypatch.setattr(llm_client, "new_llm_client", lambda model, **kw: Parked())
    turn = unillm_turn(
        "openai/gpt-6-sol",
        "low",
        route=SolRoute(SOL_BASE, SecretStr(SOL_TOKEN)),
    )

    async def served() -> dict:
        return _redacted(_prepare())

    async def scenario() -> None:
        gates["in"], gates["out"], go = (
            asyncio.Event(),
            asyncio.Event(),
            asyncio.Event(),
        )

        async def actor() -> None:  # created before the Sol call
            await go.wait()
            seen["actor"] = _redacted(_prepare())
            seen["served"] = await asyncio.create_task(served())

        actor_task = asyncio.create_task(actor())
        sol_task = asyncio.create_task(turn([{"role": "user", "content": "u"}], []))
        await asyncio.wait_for(gates["in"].wait(), 10)
        go.set()
        await asyncio.wait_for(actor_task, 10)
        out: dict = {}
        thread = threading.Thread(
            target=lambda: out.__setitem__("t", _redacted(_prepare())),
        )
        thread.start()
        thread.join(10)
        seen["thread"] = out.get("t", {})
        seen["executor"] = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: _redacted(_prepare()),
        )
        seen["main"] = _redacted(_prepare())
        gates["out"].set()
        msg, _usd = await asyncio.wait_for(sol_task, 10)
        seen["msg"] = msg

    asyncio.run(scenario())

    assert _is_sol(seen["sol"]) and _is_sol(seen["sol_after"])
    assert seen["sol"]["extra_headers"] == {"X-Unify-Call-Kind": "memory_v2.sol"}
    others = {
        k: _is_actor(seen[k]) for k in ("actor", "served", "thread", "executor", "main")
    }
    assert others == dict.fromkeys(others, True)
    assert not any("extra_headers" in seen[k] for k in others)
    assert seen["msg"]["content"] == "ok"
    assert sol_pass._SOL_GATEWAY.get() is None


# --- unillm's real call path, with a fake transport (no network) ---------------------------------------------

REAL_MODEL = (
    "openai/gpt-5.6-sol"  # the model the scripted transport tests build clients for
)
_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "noop",
            "description": "Does nothing.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


def _completion(cost: float = 0.01):
    from openai.types.chat import ChatCompletion

    return ChatCompletion.model_validate(
        {
            "id": "cmpl-sol-route",
            "object": "chat.completion",
            "created": 0,
            "model": REAL_MODEL,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": "ok",
                        "tool_calls": None,
                    },
                },
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 2,
                "total_tokens": 12,
                "cost": cost,
            },
        },
    )


@pytest.fixture
def real_transport(monkeypatch, actor_gateway, restore_unillm):
    """``litellm.acompletion`` replaced by a fake that keeps each call's route (its credential by digest only)
    and can be held open; unillm's response cache off. Everything above the transport is unillm's own path.
    """
    import unillm.clients.uni_llm as uni_llm
    from unillm.settings import SETTINGS as unillm_settings

    monkeypatch.setenv("UNILLM_CACHE", "false")
    monkeypatch.setattr(unillm_settings, "UNILLM_CACHE", False)
    state = SimpleNamespace(sent=[], hold=None, started=None)

    async def fake_acompletion(*, shared_session=None, client=None, **kw):
        state.sent.append(
            {
                "api_base": kw.get("api_base"),
                "api_key": _digest(kw.get("api_key")),
                "extra_headers": dict(kw.get("extra_headers") or {}),
            },
        )
        if state.started is not None:
            state.started.set()
        if state.hold is not None:
            await state.hold.wait()
        return _completion()

    monkeypatch.setattr(uni_llm.litellm, "acompletion", fake_acompletion)
    return state


@pytest.fixture
def listener():
    """A process-wide unillm listener (as the event bus is) recording, per event, whether it held a key."""
    import unillm

    events: list[dict] = []

    def record(event) -> None:
        request = event.request if isinstance(event.request, dict) else {}
        events.append(
            {
                "has_key": "api_key" in request,
                "api_base": request.get("api_base"),
                "origin": event.origin,
            },
        )

    handle = unillm.add_llm_event_listener(record)
    try:
        yield events
    finally:
        unillm.remove_llm_event_listener(handle)


_MESSAGES = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]


def test_unillms_real_path_sends_sols_call_on_its_route_and_the_actors_on_its_own(
    real_transport,
    listener,
):
    from unify.common.llm_client import new_llm_client

    turn = unillm_turn(
        REAL_MODEL,
        "low",
        route=SolRoute(SOL_BASE, SecretStr(SOL_TOKEN)),
    )
    msg, _usd = asyncio.run(turn(_MESSAGES, _TOOLS))
    assert msg.get("content") == "ok"
    assert len(real_transport.sent) == 1
    sol = real_transport.sent[0]
    assert _is_sol(sol)
    assert sol["extra_headers"].get("X-Unify-Call-Kind") == "memory_v2.sol"
    sol_events = [e for e in listener if e["origin"] == "memory_v2.sol"]
    assert sol_events and all(
        not e["has_key"] and e["api_base"] == SOL_BASE for e in sol_events
    )

    # an actor call after it, on the same process: the actor's gateway and key, no Sol header
    actor = new_llm_client(
        f"{REAL_MODEL}@openrouter",
        cache=False,
        reasoning_effort="low",
    )

    async def actor_call() -> None:
        await actor.generate(messages=_MESSAGES, tools=_TOOLS, tool_choice="auto")

    asyncio.run(actor_call())
    assert len(real_transport.sent) == 2
    after = real_transport.sent[1]
    assert _is_actor(after)
    assert after["extra_headers"].get("X-Unify-Call-Kind") != "memory_v2.sol"


def test_a_cancelled_routed_call_strips_the_key_from_its_late_event(
    real_transport,
    listener,
):
    """Sol's caller gives up mid-call; unillm leaves the request running and bills it when it returns. Both
    the cancellation's event and the late one pass through Sol's hook, so no listener sees the token.
    """
    turn = unillm_turn(
        REAL_MODEL,
        "low",
        route=SolRoute(SOL_BASE, SecretStr(SOL_TOKEN)),
    )

    seen: dict = {"route_after": "not read"}  # replaced only inside the turn's task

    async def in_turns_task() -> None:
        try:
            await turn(_MESSAGES, _TOOLS)
        except asyncio.CancelledError:
            # the turn's own context: the route it set must have been reset on the way out
            seen["route_after"] = sol_pass._SOL_GATEWAY.get()
            raise

    async def scenario() -> dict:
        real_transport.hold = asyncio.Event()
        real_transport.started = asyncio.Event()
        task = asyncio.create_task(in_turns_task())
        await asyncio.wait_for(real_transport.started.wait(), 10)
        task.cancel()
        cancelled = False
        try:
            await task
        except asyncio.CancelledError:
            cancelled = True
        before = len(listener)
        real_transport.hold.set()
        for _ in range(500):  # the late event, bounded
            if len(listener) > before:
                break
            await asyncio.sleep(0.01)
        return {
            "cancelled": cancelled,
            "late": len(listener) > before,
            "route_after": seen["route_after"],
        }

    out = asyncio.run(scenario())
    assert out["cancelled"] and out["late"] and out["route_after"] is None
    assert len(real_transport.sent) == 1 and _is_sol(real_transport.sent[0])
    sol_events = [e for e in listener if e["origin"] == "memory_v2.sol"]
    assert len(sol_events) >= 2
    assert all(not e["has_key"] for e in sol_events)
    assert all(e["api_base"] == SOL_BASE for e in sol_events)


# --- misroutes are refused ------------------------------------------------------------------------------


def test_a_route_not_in_effect_is_refused_before_any_call(
    monkeypatch,
    actor_gateway,
    restore_unillm,
):
    """If unillm stopped reading its gateway lookup through the wrapped module global (drift), the pre-call
    check sees the actor's route and refuses: generate is never called.
    """
    from unillm.clients import uni_llm

    calls: list = []
    built = _fake_client(monkeypatch, lambda kw: calls.append(1))
    turn = unillm_turn(
        "openai/gpt-6-sol",
        "low",
        route=SolRoute(SOL_BASE, SecretStr(SOL_TOKEN)),
    )
    monkeypatch.setattr(uni_llm, "_llm_gateway", uni_llm._llm_gateway.__wrapped__)
    with pytest.raises(sol_pass.SolRouteError) as info:
        _run(turn)
    leaked = SOL_TOKEN in str(info.value)
    assert not leaked
    assert calls == [] and "generate" not in built
    assert sol_pass._SOL_GATEWAY.get() is None


# --- failed Sol calls leave no credential in notes, error rows or logs --------------------------------------


class _EchoingError(Exception):
    """A provider error whose text echoes the request's Authorization header (as an SDK puts a body in str)."""

    status_code = 401


def _echo() -> str:
    return f"Error code: 401 - {{'echo': {{'Authorization': 'Bearer {SOL_TOKEN}'}}}}"


def test_a_failed_routed_call_is_recorded_as_its_class_and_category_only(
    monkeypatch,
    restore_unillm,
):
    def on_generate(kw):
        raise _EchoingError(_echo())

    _fake_client(monkeypatch, on_generate)
    turn = unillm_turn(
        "openai/gpt-6-sol",
        "low",
        route=SolRoute(SOL_BASE, SecretStr(SOL_TOKEN)),
    )
    with pytest.raises(sol_pass.SolCallError) as info:
        _run(turn)
    text = str(info.value)
    exact = text == "Sol call failed: _EchoingError (http_401)"
    unchained = info.value.__context__ is None and info.value.__cause__ is None
    leaked = SOL_TOKEN in text
    assert exact and unchained and not leaked
    assert sol_pass._SOL_GATEWAY.get() is None


def test_a_routed_timeout_stays_a_timeout_without_its_text(monkeypatch, restore_unillm):
    def on_generate(kw):
        raise TimeoutError(_echo())

    _fake_client(monkeypatch, on_generate)
    turn = unillm_turn(
        "openai/gpt-6-sol",
        "low",
        route=SolRoute(SOL_BASE, SecretStr(SOL_TOKEN)),
    )
    with pytest.raises(TimeoutError) as info:
        _run(turn)
    leaked = SOL_TOKEN in str(info.value)
    assert not leaked and not isinstance(info.value, sol_pass.SolCallError)


def test_pass_notes_and_the_pass_row_hold_no_token(
    monkeypatch,
    tmp_path,
    restore_unillm,
):
    """The routed turn's failure in a real pass, and (defence in depth) a raw turn whose error text carries
    the registered token and a bearer header: neither credential reaches the notes or the recorded row.
    """
    from tests.memory_v2.test_sol_pass import _run as run_pass
    from tests.memory_v2.test_sol_pass import _sol
    from unify.process_secrets import register_secret

    def on_generate(kw):
        raise _EchoingError(_echo())

    _fake_client(monkeypatch, on_generate)
    routed = unillm_turn(
        "openai/gpt-6-sol",
        "low",
        route=SolRoute(SOL_BASE, SecretStr(SOL_TOKEN)),
    )
    other = "unregistered-bearer-placeholder-0123"  # pragma: allowlist secret
    register_secret("UNIFY_MEMORY_V2_SOL_TOKEN", SOL_TOKEN)

    async def raw(messages, tools):
        raise RuntimeError(f"echo {SOL_TOKEN} and Authorization: Bearer {other}")

    rows: list[str] = []
    for i, turn in enumerate((routed, raw)):
        (tmp_path / str(i)).mkdir()
        _mem, ev, sol = _sol(tmp_path / str(i), turn)
        out = run_pass(sol, f"p{i}")
        recorded = ev.db.execute(
            "SELECT reasons FROM passes WHERE pass_id=?",
            (f"p{i}",),
        ).fetchone()[0]
        rows.append(" ".join(out.reasons) + " " + str(recorded))
    named = "SolCallError: Sol call failed: _EchoingError (http_401)" in rows[0]
    leaked = [i for i, text in enumerate(rows) if SOL_TOKEN in text or other in text]
    assert named and leaked == []


def test_a_route_not_in_effect_in_a_pass_ends_it_with_its_own_reason_code(tmp_path):
    """A Sol call that finds the route not in effect ends the pass with ``route_not_in_effect``, which the end
    event carries (not the generic ``sol_error``).
    """
    from tests.memory_v2.test_sol_pass import _run as run_pass
    from tests.memory_v2.test_sol_pass import _sol

    async def misrouted(messages, tools):
        raise sol_pass.SolRouteError("a Sol call did not take Sol's declared route")

    _mem, _ev, sol = _sol(tmp_path, misrouted)
    out = run_pass(sol, "p0")
    codes = consolidate.reason_codes(out, None)
    assert not out.passed
    assert sol_pass.CODE_ROUTE_NOT_IN_EFFECT in out.codes
    assert codes[0] == "route_not_in_effect" and "sol_error" not in codes


# --- unillm's own copies of a failed call's text --------------------------------------------------------------


def test_unillms_retry_warnings_are_redacted_once_the_route_is_installed(
    monkeypatch,
    restore_unillm,
    caplog,
):
    """``unillm.retry`` logs the first 200 characters of a failed call's text as a WARNING, which reaches the
    controller's stderr. Building a routed turn puts a redacting filter on that exact logger, once.
    """
    from unify.process_secrets import register_secret

    register_secret("UNIFY_MEMORY_V2_SOL_TOKEN", SOL_TOKEN)
    _fake_client(monkeypatch, lambda kw: None)
    for _ in range(2):  # idempotent
        unillm_turn(
            "openai/gpt-6-sol",
            "low",
            route=SolRoute(SOL_BASE, SecretStr(SOL_TOKEN)),
        )
    retry = logging.getLogger(sol_pass.RETRY_LOGGER)
    ours = [f for f in retry.filters if getattr(f, "_memory_v2_sol", False)]
    other = "unregistered-bearer-placeholder-4567"  # pragma: allowlist secret
    with caplog.at_level(logging.DEBUG, logger=sol_pass.RETRY_LOGGER):
        retry.warning(
            f"not retried, treated as permanent: AuthenticationError: {_echo()}",
        )
        retry.warning("retries exhausted after %d attempts: %s", 3, _echo())
        retry.warning("echo {'Authorization': 'Bearer %s'}", other)
        try:
            raise RuntimeError(f"echo {SOL_TOKEN}")
        except RuntimeError:
            retry.warning("with a traceback", exc_info=True)
    texts = [caplog.text] + [r.getMessage() for r in caplog.records]
    leaked = [i for i, t in enumerate(texts) if SOL_TOKEN in t or other in t]
    assert len(ours) == 1
    assert len(caplog.records) == 4 and leaked == []
    assert (
        "<secret:UNIFY_MEMORY_V2_SOL_TOKEN>" in caplog.text
        and "<redacted>" in caplog.text
    )


def test_sols_per_call_log_file_is_rewritten_redacted(
    monkeypatch,
    tmp_path,
    restore_unillm,
):
    """Sol's client (only) gets unillm's ``on_log_file`` callback, which rewrites the finalised per-call log
    (``UNILLM_LOG_DIR``; it holds ``str(error)`` as is) with the token and credential structures removed.
    """
    import unify.common.llm_client as llm_client
    from unify.process_secrets import register_secret

    register_secret("UNIFY_MEMORY_V2_SOL_TOKEN", SOL_TOKEN)
    callbacks: list = []

    class LoggingClient:
        def set_on_log_file(self, cb):
            callbacks.append(cb)
            return self

    monkeypatch.setattr(
        llm_client,
        "new_llm_client",
        lambda model, **kw: LoggingClient(),
    )
    unillm_turn("openai/gpt-6-sol", "low")  # unset: the shipped client, no callback
    assert callbacks == []
    unillm_turn(
        "openai/gpt-6-sol",
        "low",
        route=SolRoute(SOL_BASE, SecretStr(SOL_TOKEN)),
    )
    assert callbacks == [sol_pass._redact_log_file]

    other = "unregistered-bearer-placeholder-8901"  # pragma: allowlist secret
    log = tmp_path / "0001.cache_miss.txt"
    log.write_text(
        json.dumps(
            {
                "request": {"model": "openai/gpt-6-sol", "api_base": SOL_BASE},
                "error": {"type": "AuthenticationError", "message": _echo()},
                "echo": f"Authorization: Bearer {other}",
            },
        ),
    )
    clean = tmp_path / "clean.txt"
    clean.write_text("nothing to redact\n")
    before = clean.stat().st_mtime_ns
    callbacks[0](log)
    callbacks[0](clean)
    text = log.read_text()
    leaked = SOL_TOKEN in text or other in text
    assert not leaked and "AuthenticationError" in text and SOL_BASE in text
    assert (
        clean.read_text() == "nothing to redact\n"
        and clean.stat().st_mtime_ns == before
    )
    left = sorted(
        p.name for p in tmp_path.iterdir()
    )  # no temporary file left beside them
    assert left == sorted([log.name, clean.name])


def test_sols_route_refuses_to_start_while_unillm_records_spans(
    monkeypatch,
    restore_unillm,
):
    """unillm's OTel spans keep a failed call's text as is (``error.message``), so with ``UNILLM_OTEL`` on the
    route fails closed: the settings refuse (no pass starts) and a routed turn cannot be built.
    """
    unillm_logger = importlib.import_module("unillm.logger")
    monkeypatch.setattr(unillm_logger, "_OTEL_ENABLED", True)
    made: list = []
    _fake_client(monkeypatch, lambda kw: made.append(1))
    routed = SimpleNamespace(
        UNIFY_MEMORY_V2_SOL_BASE_URL=SOL_BASE,
        UNIFY_MEMORY_V2_SOL_TOKEN=SecretStr(SOL_TOKEN),
    )
    with pytest.raises(switch.SolRouteRefused) as info:
        consolidate.sol_settings(routed)
    with pytest.raises(sol_pass.SolRouteError) as built:
        unillm_turn(
            "openai/gpt-6-sol",
            "low",
            route=SolRoute(SOL_BASE, SecretStr(SOL_TOKEN)),
        )
    texts = (str(info.value), str(built.value))
    leaked = any(SOL_TOKEN in t for t in texts)
    assert not leaked and all("UNILLM_OTEL" in t for t in texts)
    assert made == []
    # the shipped route is unaffected by OTel
    assert consolidate.sol_settings(SimpleNamespace()).route is None


def test_a_refused_route_is_flagged_in_the_runs_events(monkeypatch, tmp_path):
    """A refused route starts no pass, ever: each request appends one value-free ``refused`` event to the run's
    ``events.jsonl`` (and the --jsonl stream), so the cell is flagged, not read as a null result. With the route
    unset, no such event is sent.
    """
    events = tmp_path / "events.jsonl"
    errors = tmp_path / "errors.jsonl"
    stores = SimpleNamespace(paths=SimpleNamespace(events=events, errors=errors))
    emitted: list[dict] = []
    refused = SimpleNamespace(
        UNIFY_MEMORY_V2_SOL_BASE_URL=SOL_BASE,
        UNIFY_MEMORY_V2_SOL_TOKEN=SecretStr(""),
    )
    for eid in ("e1", "e2"):
        with pytest.raises(switch.SolRouteRefused):
            asyncio.run(
                consolidate.run_due_passes(
                    stores,
                    eid,
                    "0" * 40,
                    SimpleNamespace(),
                    effort="low",
                    settings=refused,
                    emit=emitted.append,
                ),
            )
    lines = [json.loads(line) for line in events.read_text().splitlines()]
    want = [
        {
            "type": "consolidation",
            "phase": "refused",
            "episode_id": eid,
            "consolidation_refused": "route_not_in_effect",
            "reason_codes": ["route_not_in_effect"],
        }
        for eid in ("e1", "e2")
    ]
    assert lines == want and emitted == want
    leaked = SOL_BASE in events.read_text()
    assert not leaked

    events.unlink()
    out = asyncio.run(
        consolidate.run_due_passes(
            stores,
            "e3",
            "0" * 40,
            SimpleNamespace(),
            effort="",  # stops right after the settings, before any store is touched
            settings=SimpleNamespace(),
            emit=emitted.append,
        ),
    )
    assert out == [] and not events.exists() and len(emitted) == 2


def test_error_rows_transcripts_and_redactors_drop_the_registered_token(tmp_path):
    from unify import transcripts
    from unify.memory_v2.integration.request import RequestRun
    from unify.memory_v2.redact import Redactor, redact_error
    from unify.process_secrets import register_secret

    register_secret("UNIFY_MEMORY_V2_SOL_TOKEN", SOL_TOKEN)
    present = any(k.upper() == "UNIFY_MEMORY_V2_SOL_TOKEN" for k in os.environ)
    errors = tmp_path / "errors.jsonl"
    consolidate._error(
        SimpleNamespace(paths=SimpleNamespace(errors=errors)),
        f"px: pass error: RuntimeError: {SOL_TOKEN}",
    )
    RequestRun._error(
        SimpleNamespace(episode_id="e1", paths=SimpleNamespace(errors=errors)),
        "passes",
        RuntimeError(f"echo {SOL_TOKEN}"),
        lambda line: None,
    )
    texts = [
        errors.read_text(),
        redact_error(f"x {SOL_TOKEN} y"),
        redact_error("headers={'Authorization': 'Bearer abcdefgh12345678'}"),
        Redactor.from_environ({}).text(f"x {SOL_TOKEN} y"),
        transcripts.scrub(f"x {SOL_TOKEN} y"),
    ]
    leaked = [
        i for i, t in enumerate(texts) if SOL_TOKEN in t or "abcdefgh12345678" in t
    ]
    assert not present and leaked == []
    assert len(errors.read_text().splitlines()) == 2


# --- the environment: any letter case, and .env --------------------------------------------------------------

_ROUTE_NAMES = ("UNIFY_MEMORY_V2_SOL_TOKEN", "UNIFY_MEMORY_V2_SOL_BASE_URL")


def test_a_token_in_any_letter_case_leaves_the_environment(monkeypatch):
    from unify.settings import ProductionSettings

    monkeypatch.setattr(switch, "_ENV_REFUSAL", None)
    monkeypatch.setenv("unify_memory_v2_sol_token", SOL_TOKEN)
    monkeypatch.setenv("UNIFY_MEMORY_V2_SOL_BASE_URL", SOL_BASE)
    s = ProductionSettings()
    held = _digest(s.UNIFY_MEMORY_V2_SOL_TOKEN.get_secret_value()) == _digest(SOL_TOKEN)
    refusal = switch.settle_sol_route_env(os.environ, s)
    left = [k for k in os.environ if k.upper() == "UNIFY_MEMORY_V2_SOL_TOKEN"]
    assert held and refusal is None and left == []
    assert consolidate.sol_settings(s).route is not None


def test_case_variants_that_disagree_refuse_every_pass(monkeypatch):
    monkeypatch.setattr(switch, "_ENV_REFUSAL", None)
    other = "a-different-placeholder-token"  # pragma: allowlist secret
    env = {"UNIFY_MEMORY_V2_SOL_TOKEN": SOL_TOKEN, "unify_memory_v2_sol_token": other}
    settings = SimpleNamespace(
        UNIFY_MEMORY_V2_SOL_BASE_URL=SOL_BASE,
        UNIFY_MEMORY_V2_SOL_TOKEN=SecretStr(SOL_TOKEN),
    )
    refusal = switch.settle_sol_route_env(env, settings)
    assert refusal is not None and env == {}
    leaked = SOL_TOKEN in refusal or other in refusal
    assert not leaked
    with pytest.raises(ValueError):
        consolidate.sol_settings(settings)


def test_sol_settings_settles_again_so_a_late_value_is_refused_off_the_cli(monkeypatch):
    """An embedder that loads ``.env`` after importing unify never runs the CLI's settle: the pass start
    settles again, so the late route is refused (not silently replaced by the actor's) and its token leaves.
    """
    monkeypatch.setattr(switch, "_ENV_REFUSAL", None)
    for name in [k for k in os.environ if k.upper() in _ROUTE_NAMES]:
        monkeypatch.delenv(name)
    settings = SimpleNamespace()  # as read before the late values arrived: neither set
    monkeypatch.setenv("UNIFY_MEMORY_V2_SOL_BASE_URL", SOL_BASE)
    monkeypatch.setenv("UNIFY_MEMORY_V2_SOL_TOKEN", SOL_TOKEN)
    with pytest.raises(switch.SolRouteRefused) as info:
        consolidate.sol_settings(settings)
    text = str(info.value)
    token_left = any(k.upper() == _ROUTE_NAMES[0] for k in os.environ)
    leaked = SOL_TOKEN in text or SOL_BASE in text
    assert not token_left and not leaked
    assert ".env" in text and "no consolidation pass starts" in text


def test_the_cli_settles_after_loading_dotenv(monkeypatch, tmp_path, capsys):
    """``cli._configure_environment`` settles after ``load_dotenv``, so a route only in ``.env`` is refused
    on the CLI path too, and the refusal it prints names settings, never values.
    """
    from unify import cli

    unify_logger = importlib.import_module("unify.logger")
    order: list[str] = []
    monkeypatch.setattr(switch, "_ENV_REFUSAL", None)
    for name in [k for k in os.environ if k.upper() in _ROUTE_NAMES]:
        monkeypatch.delenv(name)
    monkeypatch.setenv("UNIFY_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(unify_logger, "configure_log_dir", lambda d: None)

    def fake_load_dotenv(*a, **k):
        order.append("load_dotenv")
        os.environ["UNIFY_MEMORY_V2_SOL_BASE_URL"] = SOL_BASE
        os.environ["UNIFY_MEMORY_V2_SOL_TOKEN"] = SOL_TOKEN
        return True

    real_settle = switch.settle_sol_route_env

    def settle(environ, settings):
        order.append("settle")
        return real_settle(environ, settings)

    monkeypatch.setattr(cli, "load_dotenv", fake_load_dotenv)
    monkeypatch.setattr(switch, "settle_sol_route_env", settle)
    try:
        cli._configure_environment(
            SimpleNamespace(home=str(tmp_path / "home"), debug=True),
        )
        token_left = any(k.upper() == _ROUTE_NAMES[0] for k in os.environ)
        refused = switch._ENV_REFUSAL is not None
    finally:
        for name in [k for k in os.environ if k.upper() in _ROUTE_NAMES]:
            os.environ.pop(name, None)
    err = capsys.readouterr().err
    leaked = SOL_TOKEN in err or SOL_BASE in err
    assert order == ["load_dotenv", "settle"]
    assert refused and not token_left and not leaked
    assert "memory v2:" in err and ".env" in err


def test_importing_settings_takes_the_token_out_of_the_environment(tmp_path):
    """``import unify.settings`` (a fresh process) leaves no case variant of the token in ``os.environ`` and
    holds it in SETTINGS; the child prints booleans only.
    """
    import unify

    root = Path(unify.__file__).resolve().parents[1]
    env = {k: v for k, v in os.environ.items() if k.upper() not in _ROUTE_NAMES}
    env.update(
        {
            "UNIFY_MEMORY_V2_SOL_BASE_URL": SOL_BASE,
            "unify_memory_v2_sol_token": SOL_TOKEN,
            "PYTHONPATH": os.pathsep.join(
                [str(root), *filter(None, [env.get("PYTHONPATH", "")])],
            ),
        },
    )
    code = (
        "import hashlib, json, os\n"
        "import unify.settings as s\n"
        "t = s.SETTINGS.UNIFY_MEMORY_V2_SOL_TOKEN.get_secret_value()\n"
        "print(json.dumps({'left': [k.upper() for k in os.environ"
        " if k.upper() == 'UNIFY_MEMORY_V2_SOL_TOKEN'],"
        " 'held': hashlib.sha256(t.encode()).hexdigest()}))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=300,
    )
    shown = SOL_TOKEN in proc.stdout or SOL_TOKEN in proc.stderr
    assert not shown
    assert proc.returncode == 0, proc.stderr[-2000:]
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    held = out["held"] == hashlib.sha256(SOL_TOKEN.encode()).hexdigest()
    assert out["left"] == [] and held


def test_a_route_only_in_dotenv_refuses_every_pass_and_its_token_leaves(
    monkeypatch,
    tmp_path,
):
    """Settings are read before the CLI loads ``.env``: a route only there would silently leave Sol on the
    actor's route, so every pass is refused, naming the settings and never their values.
    """
    from dotenv import load_dotenv

    monkeypatch.setattr(switch, "_ENV_REFUSAL", None)
    for name in [k for k in os.environ if k.upper() in _ROUTE_NAMES]:
        monkeypatch.delenv(name)
    settings = SimpleNamespace()  # as read before .env: neither set
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        f"UNIFY_MEMORY_V2_SOL_BASE_URL={SOL_BASE}\nUNIFY_MEMORY_V2_SOL_TOKEN={SOL_TOKEN}\n",
    )
    try:
        load_dotenv(dotenv)
        refusal = switch.settle_sol_route_env(os.environ, settings)
        token_left = any(k.upper() == _ROUTE_NAMES[0] for k in os.environ)
    finally:
        for name in [k for k in os.environ if k.upper() in _ROUTE_NAMES]:
            os.environ.pop(name, None)
    assert refusal is not None and not token_left
    leaked = SOL_TOKEN in refusal or SOL_BASE in refusal
    assert not leaked
    assert ".env" in refusal and all(n in refusal for n in _ROUTE_NAMES)
    with pytest.raises(ValueError) as info:
        consolidate.sol_settings(settings)
    shown = SOL_TOKEN in str(info.value) or SOL_BASE in str(info.value)
    assert not shown and "no consolidation pass starts" in str(info.value)


# --- the token by descriptor (UNIFY_MEMORY_V2_SOL_TOKEN_FD) --------------------------------------------------

_FD = "UNIFY_MEMORY_V2_SOL_TOKEN_FD"
_FD_NAMES = (*_ROUTE_NAMES, _FD)


@pytest.fixture
def fd_route(monkeypatch):
    """A process whose descriptor token was never read. Every descriptor a test opens is closed after it,
    unless it was closed already (its number then reused by something else, which is left alone).
    """
    monkeypatch.setattr(switch, "_ENV_REFUSAL", None)
    monkeypatch.setattr(switch, "_FD_TOKEN", None)
    monkeypatch.setattr(switch, "_FD_TRIED", False)
    monkeypatch.setattr(switch, "_FD_HELD", "")
    for name in [k for k in os.environ if k.upper() in _FD_NAMES]:
        monkeypatch.delenv(name)
    opened: list[tuple[int, int, int]] = []
    yield opened
    for fd, dev, ino in opened:
        try:
            st = os.fstat(fd)
            if (st.st_dev, st.st_ino) == (dev, ino):
                os.close(fd)
        except OSError:
            pass


def _track(opened: list, *fds: int) -> None:
    for fd in fds:
        st = os.fstat(fd)
        opened.append((fd, st.st_dev, st.st_ino))


def _token_fd(opened: list, data: bytes) -> int:
    """A pipe holding *data*, its write end closed and its read end inheritable, as the launcher passes it."""
    r, w = os.pipe()
    _track(opened, r)
    try:
        os.write(w, data)
    finally:
        os.close(w)
    os.set_inheritable(r, True)
    return r


def _is_open(fd: int) -> bool:
    try:
        os.fstat(fd)
    except OSError:
        return False
    return True


def _registered(value: str) -> bool:
    from unify.process_secrets import registered_secrets

    return any(v == value for _, v in registered_secrets())


def _fd_settings(fd: object, base: str = SOL_BASE, token: str = "") -> SimpleNamespace:
    return SimpleNamespace(
        UNIFY_MEMORY_V2_SOL_BASE_URL=base,
        UNIFY_MEMORY_V2_SOL_TOKEN=SecretStr(token),
        UNIFY_MEMORY_V2_SOL_TOKEN_FD=str(fd),
    )


@pytest.mark.parametrize("kind", ["pipe", "pipe without newline", "file"])
def test_the_token_is_read_from_its_descriptor_which_is_then_closed(
    monkeypatch,
    tmp_path,
    fd_route,
    kind,
):
    """Settled once, the token comes from the inherited descriptor (one trailing newline stripped), the
    descriptor is closed, the environment holds neither the token nor the number, and the redactors hold
    the token.
    """
    from unify.settings import ProductionSettings

    token = secrets.token_urlsafe(32)
    data = token.encode() + (b"" if kind == "pipe without newline" else b"\n")
    if kind == "file":
        path = tmp_path / "token"
        path.write_bytes(data)
        fd = os.open(path, os.O_RDONLY)
        _track(fd_route, fd)
        os.set_inheritable(fd, True)
        path.unlink()
    else:
        fd = _token_fd(fd_route, data)
    monkeypatch.setenv(_FD, str(fd))
    monkeypatch.setenv("UNIFY_MEMORY_V2_SOL_BASE_URL", SOL_BASE)
    s = ProductionSettings()
    refusal = switch.settle_sol_route_env(os.environ, s)
    closed = not _is_open(fd)
    cfg = consolidate.sol_settings(s)
    in_env = any(token in v for v in os.environ.values())
    used = cfg.route is not None and _digest(
        cfg.route.token.get_secret_value(),
    ) == _digest(token)
    assert refusal is None and closed and used
    assert not in_env and not [k for k in os.environ if k.upper() == _FD]
    assert _registered(token)
    assert s.UNIFY_MEMORY_V2_SOL_TOKEN.get_secret_value() == ""


def test_a_child_spawned_after_the_read_does_not_inherit_the_descriptor(fd_route):
    """Before the read, a child spawned without ``close_fds`` would inherit the pipe (the control); after it,
    no descriptor of the child is that pipe.
    """
    fd = _token_fd(fd_route, secrets.token_urlsafe(32).encode() + b"\n")
    ino = os.fstat(fd).st_ino
    probe = (
        "import json, os, stat\n"
        "seen = False\n"
        "for name in os.listdir('/proc/self/fd'):\n"
        "    try:\n"
        "        st = os.fstat(int(name))\n"
        "    except OSError:\n"
        "        continue\n"
        f"    seen = seen or (stat.S_ISFIFO(st.st_mode) and st.st_ino == {ino})\n"
        "print(json.dumps(seen))\n"
    )

    def child_sees_it() -> bool:
        proc = subprocess.run(
            [sys.executable, "-c", probe],
            close_fds=False,  # the worst case: os.system, a fork
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert proc.returncode == 0, proc.stderr[-2000:]
        return json.loads(proc.stdout.strip().splitlines()[-1])

    before = child_sees_it()
    refusal = switch.settle_sol_route_env({}, _fd_settings(fd))
    after = child_sees_it()
    assert before and refusal is None and not after


def test_the_descriptor_name_leaves_the_environment_and_is_never_read_again(
    monkeypatch,
    fd_route,
):
    """Once settled, no letter case of the name is left in the environment; a later settle reads no
    descriptor again and does not refuse, while a different number arriving later is refused as stray.
    """
    fd = _token_fd(fd_route, secrets.token_urlsafe(32).encode() + b"\n")
    monkeypatch.setenv(_FD, str(fd))
    monkeypatch.setenv(_FD.lower(), str(fd))
    settings = _fd_settings(fd)
    first = switch.settle_sol_route_env(os.environ, settings)
    left = [k for k in os.environ if k.upper() == _FD]
    assert first is None and left == [] and not _is_open(fd)
    reads: list[int] = []
    monkeypatch.setattr(switch, "_read_token_fd", lambda n: reads.append(n) or "")
    again = switch.settle_sol_route_env(os.environ, settings)
    same_back = switch.settle_sol_route_env({_FD: str(fd)}, settings)
    assert again is None and same_back is None and reads == []
    held = switch.sol_token(settings)
    assert held is switch._FD_TOKEN and held.get_secret_value()
    late = {_FD.lower(): str(fd + 1)}
    refusal = switch.settle_sol_route_env(late, settings)
    assert refusal is not None and _FD in refusal and late == {} and reads == []


def test_a_child_spawned_after_settings_sees_neither_the_name_nor_the_descriptor(
    tmp_path,
):
    """A fresh controller given the descriptor (``pass_fds``, as the launcher passes it) imports the
    settings, which read it; a child it then spawns with its inherited environment and default
    ``close_fds`` finds no letter case of the name in its environment and the number not open. The
    processes print booleans only.
    """
    import unify

    token = secrets.token_urlsafe(32)
    r, w = os.pipe()
    try:
        os.write(w, token.encode() + b"\n")
    finally:
        os.close(w)
    ino = os.fstat(r).st_ino
    root = Path(unify.__file__).resolve().parents[1]
    env = {k: v for k, v in os.environ.items() if k.upper() not in _FD_NAMES}
    env.update(
        {
            "UNIFY_MEMORY_V2_SOL_BASE_URL": SOL_BASE,
            _FD: str(r),
            "PYTHONPATH": os.pathsep.join(
                [str(root), *filter(None, [env.get("PYTHONPATH", "")])],
            ),
        },
    )
    # the pipe is looked for by identity (a FIFO with its inode), since its number may be reused
    grandchild = (
        "import json, os, stat\n"
        "def is_pipe(n):\n"
        "    try:\n"
        "        st = os.fstat(n)\n"
        "    except OSError:\n"
        "        return False\n"
        f"    return stat.S_ISFIFO(st.st_mode) and st.st_ino == {ino}\n"
        f"fd_open = is_pipe({r})\n"
        "listed = any(is_pipe(int(n)) for n in os.listdir('/proc/self/fd'))\n"
        f"named = any(k.upper() == {_FD!r} for k in os.environ)\n"
        "print(json.dumps({'named': named, 'fd_open': fd_open, 'listed': listed}))\n"
    )
    code = (
        "import json, os, subprocess, sys\n"
        "import unify.settings\n"
        "from unify.memory_v2.integration import switch\n"
        f"named = any(k.upper() == {_FD!r} for k in os.environ)\n"
        f"proc = subprocess.run([sys.executable, '-c', {grandchild!r}], capture_output=True,"
        " text=True, timeout=120)\n"
        "child = json.loads(proc.stdout.strip().splitlines()[-1])\n"
        "print(json.dumps({'refused': switch._ENV_REFUSAL is not None, 'named': named,"
        " 'read': switch._FD_TOKEN is not None, 'child': child}))\n"
    )
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            env=env,
            cwd=tmp_path,
            pass_fds=(r,),
            capture_output=True,
            text=True,
            timeout=300,
        )
    finally:
        os.close(r)
    shown = token in proc.stdout or token in proc.stderr
    assert not shown
    assert proc.returncode == 0, proc.stderr[-2000:]
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert out == {
        "refused": False,
        "named": False,
        "read": True,
        "child": {"named": False, "fd_open": False, "listed": False},
    }


def test_both_token_settings_refuse_every_pass(fd_route):
    token = secrets.token_urlsafe(32)
    fd = _token_fd(fd_route, token.encode() + b"\n")
    settings = _fd_settings(fd, token=SOL_TOKEN)
    refusal = switch.settle_sol_route_env({}, settings)
    closed = not _is_open(fd)
    assert refusal is not None and closed
    assert "UNIFY_MEMORY_V2_SOL_TOKEN and UNIFY_MEMORY_V2_SOL_TOKEN_FD" in refusal
    with pytest.raises(switch.SolRouteRefused) as info:
        consolidate.sol_settings(settings)
    text = refusal + str(info.value)
    leaked = token in text or SOL_TOKEN in text
    assert not leaked and "no consolidation pass starts" in str(info.value)


def test_a_descriptor_without_the_base_url_refuses_every_pass(fd_route):
    token = secrets.token_urlsafe(32)
    fd = _token_fd(fd_route, token.encode() + b"\n")
    settings = _fd_settings(fd, base="")
    refusal = switch.settle_sol_route_env({}, settings)
    closed = not _is_open(fd)
    assert refusal is None and closed
    with pytest.raises(switch.SolRouteRefused) as info:
        consolidate.sol_settings(settings)
    text = str(info.value)
    leaked = token in text
    assert not leaked
    assert "UNIFY_MEMORY_V2_SOL_BASE_URL is empty" in text


_GOOD = "a-placeholder-token-from-a-pipe"  # pragma: allowlist secret


@pytest.mark.parametrize(
    "data,rule",
    [
        (b"", "is empty"),
        (b"\n", "is empty"),
        (b"A" * 4097, "more than 4096 bytes"),
        (b"A" * 4096 + b"\n", "more than 4096 bytes"),
        (b"short-token\n", "bearer token"),
        (_GOOD.encode() + b" x\n", "bearer token"),
        (_GOOD.encode() + b"\r\n", "bearer token"),
        (_GOOD.encode() + b"\n\n", "bearer token"),
        (_GOOD.encode() + b"+/=\n", "bearer token"),
        (b"\xff" * 20 + b"\n", "bearer token"),
    ],
)
def test_a_descriptor_without_a_bearer_token_refuses_every_pass(fd_route, data, rule):
    fd = _token_fd(fd_route, data)
    settings = _fd_settings(fd)
    refusal = switch.settle_sol_route_env({}, settings)
    closed = not _is_open(fd)
    assert refusal is not None and closed
    assert rule in refusal and "no consolidation pass starts" in refusal
    with pytest.raises(switch.SolRouteRefused) as info:
        consolidate.sol_settings(settings)
    body = data.rstrip(b"\n").decode("latin-1")
    text = refusal + str(info.value)
    leaked = len(body) >= 8 and body in text
    assert not leaked
    assert switch._FD_TOKEN is None and (len(body) < 8 or _registered(body))


def test_an_unusable_descriptor_refuses_every_pass(fd_route):
    """Not open, not inherited (this process's own, left open), not a pipe or a file (closed), or a bad
    number (a token set there by mistake included): every pass is refused, and no refusal quotes a value.
    """
    own_r, own_w = os.pipe()  # not inheritable (PEP 446): never the launcher's
    _track(fd_route, own_r, own_w)
    os.write(own_w, _GOOD.encode())
    device = os.open(os.devnull, os.O_RDONLY)
    _track(fd_route, device)
    os.set_inheritable(device, True)
    gone_r, gone_w = os.pipe()  # opened last and closed: a number that is not open
    os.close(gone_w)
    os.close(gone_r)
    gone = gone_r
    cases = [
        (gone, "is not open"),
        (own_r, "was not inherited by this process"),
        (device, "is not a pipe or a file"),
        ("2", "file descriptor number above 2"),
        ("abc", "file descriptor number above 2"),
        (_GOOD, "file descriptor number above 2"),
    ]
    for fd, rule in cases:
        switch._ENV_REFUSAL = None
        switch._FD_TRIED = False
        settings = _fd_settings(fd)
        refusal = switch.settle_sol_route_env({}, settings)
        with pytest.raises(switch.SolRouteRefused) as info:
            consolidate.sol_settings(settings)
        text = (refusal or "") + str(info.value)
        leaked = _GOOD in text
        assert refusal is not None and rule in refusal, rule
        assert not leaked
    assert _is_open(own_r) and not _is_open(device)
    assert _registered(_GOOD)


def test_a_writer_that_never_finishes_is_refused_in_bounded_time(monkeypatch, fd_route):
    monkeypatch.setattr(switch, "TOKEN_FD_TIMEOUT_S", 0.2)
    r, w = os.pipe()
    _track(fd_route, r, w)
    os.set_inheritable(r, True)
    os.write(w, _GOOD.encode())  # no end of file: the write end stays open
    start = time.monotonic()
    refusal = switch.settle_sol_route_env({}, _fd_settings(r))
    took = time.monotonic() - start
    assert refusal is not None and "did not reach its end" in refusal
    assert took < 5 and not _is_open(r)


def test_a_refused_descriptor_is_flagged_in_the_runs_events(fd_route, tmp_path):
    events = tmp_path / "events.jsonl"
    stores = SimpleNamespace(
        paths=SimpleNamespace(events=events, errors=tmp_path / "errors.jsonl"),
    )
    settings = _fd_settings(_token_fd(fd_route, b""))
    with pytest.raises(switch.SolRouteRefused):
        asyncio.run(
            consolidate.run_due_passes(
                stores,
                "e1",
                "0" * 40,
                SimpleNamespace(),
                effort="low",
                settings=settings,
                emit=None,
            ),
        )
    (line,) = [json.loads(x) for x in events.read_text().splitlines()]
    assert line["phase"] == "refused"
    assert line["consolidation_refused"] == "route_not_in_effect"


def test_the_descriptor_setting_is_normalised_in_settings_and_parsed_when_settled(
    monkeypatch,
    fd_route,
):
    from unify.settings import ProductionSettings

    for raw, want in [("", None), (" 3 ", 3), ("17", 17), (9, 9)]:
        assert switch.parse_sol_token_fd(raw) == want
    for raw in ["0", "1", "2", "-3", "3.0", "0x3", True, -1, _GOOD]:
        with pytest.raises(ValueError) as info:
            switch.parse_sol_token_fd(raw)
        leaked = _GOOD in str(info.value)
        assert not leaked
    monkeypatch.setenv(_FD, f" {_GOOD} ")
    s = ProductionSettings()  # loads without refusing (an error would print the value)
    held = s.UNIFY_MEMORY_V2_SOL_TOKEN_FD == _GOOD
    assert held


def test_the_descriptor_setting_never_reaches_a_cell_or_sols_box(monkeypatch):
    monkeypatch.setenv(_FD, "7")
    policy = SimpleNamespace(network="", proxy_port=None)
    explicit = sandbox.sandbox_env(
        policy,
        {_FD: "7", _FD.lower(): "7", "PATH": "/usr/bin"},
    )
    for env in (sandbox.sandbox_env(policy), explicit, sandbox.scrubbed_env()):
        assert not {n.upper() for n in env} & {_FD}
    with pytest.raises(ValueError):
        run_confined(["/bin/true"], env={_FD: "7"})


def test_importing_settings_reads_the_descriptor_and_closes_it(tmp_path):
    """In a fresh process given the descriptor (as the launcher passes it), ``import unify.settings`` reads
    the token, closes the descriptor and leaves the token out of ``os.environ``; the child prints booleans
    and a digest only.
    """
    import unify

    token = secrets.token_urlsafe(32)
    r, w = os.pipe()
    try:
        os.write(w, token.encode() + b"\n")
    finally:
        os.close(w)
    root = Path(unify.__file__).resolve().parents[1]
    env = {k: v for k, v in os.environ.items() if k.upper() not in _FD_NAMES}
    env.update(
        {
            "UNIFY_MEMORY_V2_SOL_BASE_URL": SOL_BASE,
            _FD: str(r),
            "PYTHONPATH": os.pathsep.join(
                [str(root), *filter(None, [env.get("PYTHONPATH", "")])],
            ),
        },
    )
    code = (
        "import hashlib, json, os, sys\n"
        "import unify.settings\n"
        "from unify.memory_v2.integration import switch\n"
        "from unify.process_secrets import registered_secrets\n"
        f"fd = {r}\n"
        "try:\n"
        "    os.fstat(fd); still_open = True\n"
        "except OSError:\n"
        "    still_open = False\n"
        "t = switch._FD_TOKEN.get_secret_value() if switch._FD_TOKEN else ''\n"
        "print(json.dumps({'open': still_open, 'refused': switch._ENV_REFUSAL is not None,"
        " 'in_env': bool(t) and any(t in v for v in os.environ.values()),"
        " 'registered': any(v == t for _, v in registered_secrets()),"
        " 'held': hashlib.sha256(t.encode()).hexdigest()}))\n"
    )
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            env=env,
            cwd=tmp_path,
            pass_fds=(r,),
            capture_output=True,
            text=True,
            timeout=300,
        )
    finally:
        os.close(r)
    shown = token in proc.stdout or token in proc.stderr
    assert not shown
    assert proc.returncode == 0, proc.stderr[-2000:]
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    held = out.pop("held") == hashlib.sha256(token.encode()).hexdigest()
    assert held
    assert out == {"open": False, "refused": False, "in_env": False, "registered": True}

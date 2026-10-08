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
import logging
import os
import secrets
import threading
from types import SimpleNamespace

import pytest
from pydantic import SecretStr

from unify import sandbox
from unify.memory_v2 import sol_pass
from unify.memory_v2.integration import consolidate, switch
from unify.memory_v2.sandbox_run import run_confined
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

    async def scenario() -> dict:
        real_transport.hold = asyncio.Event()
        real_transport.started = asyncio.Event()
        task = asyncio.create_task(turn(_MESSAGES, _TOOLS))
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
            "route_after": sol_pass._SOL_GATEWAY.get(),
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

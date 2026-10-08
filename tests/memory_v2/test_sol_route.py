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
    ],
)
def test_the_token_validator_refuses_unsendable_tokens_without_quoting_them(raw):
    with pytest.raises(ValueError) as info:
        switch.parse_sol_token(raw)
    leaked = "placeholder" in str(info.value)
    assert not leaked


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

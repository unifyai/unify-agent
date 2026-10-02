"""Symbolic: ``UNIFY_REVIEW_REASONING_EFFORT`` and ``UNIFY_REVIEW_MODEL``.

The storage review decides what the library keeps, so it is where a
stronger or more deliberate model could pay for itself across tasks:
ExpeL extracted insights with GPT-4 for a GPT-3.5 actor, skills SkillWeaver
synthesised with strong agents improved weaker ones by up to 54%, and
Letta configures its sleep-time agent's model apart from the main agent's
(ACE keeps one model, for a fair comparison). A forked review
(``UNIFY_REVIEW_FORK``) reads its first call from the session's cache
(0.77 of its input cached against 0.04 standalone), so the two knobs differ:
a higher effort on the same model keeps the fork's messages, tools and
cache affinity key, and only its effort changes; another model cannot
share the session's cache, so the review runs standalone and the log says
why.

Requests are captured at unillm's transport
(``tests/cache_discipline_helpers.py``); nothing leaves the process.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from tests import cache_discipline_helpers as h
from unify.actor import code_act_actor as caa
from unify.settings import SETTINGS, ProductionSettings

REVIEW_MODEL = "openai/gpt-5.6-luna@openrouter"


@pytest.fixture
def switches(monkeypatch):
    def set_(*, fork: bool, effort: str = "", model: str = "") -> None:
        monkeypatch.setattr(SETTINGS, "UNIFY_CACHE_DISCIPLINE", fork)
        monkeypatch.setattr(SETTINGS, "UNIFY_REVIEW_FORK", fork)
        monkeypatch.setattr(SETTINGS, "UNIFY_REVIEW_REASONING_EFFORT", effort)
        monkeypatch.setattr(SETTINGS, "UNIFY_REVIEW_MODEL", model)

    return set_


@pytest.fixture
def info_lines(monkeypatch):
    lines: list[str] = []
    original = caa.logger.info

    def capture(msg, *args, **kwargs):
        lines.append(str(msg))
        return original(msg, *args, **kwargs)

    monkeypatch.setattr(caa.logger, "info", capture)
    return lines


@pytest.fixture
def spies(monkeypatch):
    """The forks the review makes and the clients it builds standalone."""
    made: dict[str, list] = {"forks": [], "clients": []}
    real_fork, real_new = caa.fork_llm_client, caa.new_llm_client

    def fork_spy(parent, **kwargs):
        client = real_fork(parent, **kwargs)
        made["forks"].append({"parent": parent, "client": client})
        return client

    def new_spy(*args, **kwargs):
        client = real_new(*args, **kwargs)
        made["clients"].append({"kwargs": kwargs, "client": client})
        return client

    monkeypatch.setattr(caa, "fork_llm_client", fork_spy)
    monkeypatch.setattr(caa, "new_llm_client", new_spy)
    return made


async def _review():
    _summary, _, requests = await h.scenario_review()
    return requests


# ── effort ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_effort_changes_only_the_forks_effort(switches, spies, monkeypatch):
    switches(fork=True)
    baseline = await _review()
    switches(fork=True, effort="xhigh")
    sets = h.install_affinity_api(monkeypatch)
    spies["forks"].clear()
    requests = await _review()

    assert len(requests) == len(baseline) == 3
    # The session is untouched; the review is still the fork, byte for byte.
    assert [h.request_bytes(r) for r in requests] == [
        h.request_bytes(r) for r in baseline
    ]
    assert requests[1]["reasoning_effort"] == baseline[2]["reasoning_effort"] == "low"
    assert requests[2]["reasoning_effort"] == "xhigh"
    assert len(spies["forks"]) == 1
    fork = spies["forks"][0]
    assert fork["parent"].reasoning_effort == "low"
    # The fork keeps the session's cache affinity key; it is never re-derived.
    # (A client's construction may clear the key first: None is not a key.)
    assert fork["parent"].cache_affinity is not None
    assert fork["client"].cache_affinity == fork["parent"].cache_affinity
    assert {key for key, _n in sets if key is not None} == {
        fork["parent"].cache_affinity,
    }
    assert fork["client"].endpoint == fork["parent"].endpoint == h.MODEL


@pytest.mark.asyncio
async def test_effort_sets_the_standalone_reviews_effort(switches, spies):
    switches(fork=False, effort="medium")
    requests = await _review()
    golden = json.loads(h.GOLDEN.read_text())["review"]
    assert [h.request_bytes(r) for r in requests] == golden
    assert requests[2]["reasoning_effort"] == "medium"
    assert spies["forks"] == []


@pytest.mark.asyncio
async def test_effort_and_model_reach_the_proactive_review(switches, spies):
    from unify.actor.code_act_actor import CodeActActor

    switches(fork=False, effort="low", model=REVIEW_MODEL)
    actor = CodeActActor()
    try:
        with h.scripted([lambda: h.completion(content="Stored nothing.")]) as p:
            handle = caa._start_proactive_storage_loop(
                trajectory=[{"role": "user", "content": "task"}],
                ask_tools={},
                actor=actor,
                request="the parser",
            )
            assert await asyncio.wait_for(handle.result(), 60) == "Stored nothing."
    finally:
        await actor.close()
    assert p.requests[0]["reasoning_effort"] == "low"
    (built,) = [
        c for c in spies["clients"] if "ProactiveStorage" in c["kwargs"]["origin"]
    ]
    assert built["client"].endpoint == REVIEW_MODEL


# ── model ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_another_model_runs_the_review_standalone_and_says_why(
    switches,
    spies,
    info_lines,
):
    switches(fork=True, model=REVIEW_MODEL)
    requests = await _review()
    assert spies["forks"] == []
    skipped = [line for line in info_lines if "StorageCheck fork skipped" in line]
    assert len(skipped) == 1
    assert f"UNIFY_REVIEW_MODEL is {REVIEW_MODEL}, not the session's" in skipped[0]
    assert "running the standalone review" in skipped[0]
    # The standalone review: its own system prompt around the trajectory.
    review = requests[2]
    assert review["messages"][0]["role"] == "system"
    assert "## Completed Trajectory" in review["messages"][0]["content"]
    (built,) = [c for c in spies["clients"] if c["kwargs"]["origin"] == "StorageCheck"]
    assert built["client"].endpoint == REVIEW_MODEL
    # No effort set: the effort a client named for a model gets.
    assert review["reasoning_effort"] == "high"


@pytest.mark.asyncio
async def test_the_sessions_own_model_still_forks(switches, spies, info_lines):
    switches(fork=True, model=h.MODEL, effort="high")
    requests = await _review()
    assert len(spies["forks"]) == 1
    assert not any("fork skipped" in line for line in info_lines)
    assert requests[2]["reasoning_effort"] == "high"
    assert "## Completed Trajectory" not in json.dumps(requests[2]["messages"])


def test_the_fork_source_refuses_another_model(switches):
    from types import SimpleNamespace

    from unify.common._async_tool import cache_discipline as cd

    switches(fork=True, model=REVIEW_MODEL)
    client = h.new_client("You are a scripted actor.")
    client._messages.append({"role": "user", "content": "task"})
    cd.record_sent_request(
        client,
        list(client.messages),
        {"tools": [{"type": "function", "function": {"name": "t"}}]},
    )
    inner = SimpleNamespace(_client=client, _compression=SimpleNamespace(count=0))
    source, why = caa._review_fork_source(inner, SimpleNamespace(_preprocess_msgs=None))
    assert source is None
    assert why.startswith(f"UNIFY_REVIEW_MODEL is {REVIEW_MODEL}")
    assert h.MODEL in why


# ── off ──────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_off_the_review_is_upstreams(switches, spies, info_lines):
    switches(fork=False)
    requests = await _review()
    golden = json.loads(h.GOLDEN.read_text())["review"]
    assert [h.request_bytes(r) for r in requests] == golden
    (built,) = [c for c in spies["clients"] if c["kwargs"]["origin"] == "StorageCheck"]
    assert built["client"].reasoning_effort == requests[2]["reasoning_effort"]
    assert not any("UNIFY_REVIEW_MODEL" in line for line in info_lines)


def test_off_the_standalone_client_is_the_actors(switches, monkeypatch):
    switches(fork=False)
    calls: list = []
    monkeypatch.setattr(
        caa,
        "new_llm_client",
        lambda *a, **k: calls.append((a, k)) or h.new_client(),
    )
    actor = type("A", (), {"_model": "m@p"})()
    client = caa._storage_review_client(actor, origin="StorageCheck")
    assert calls == [(("m@p",), {"purpose": "planning", "origin": "StorageCheck"})]
    assert client.reasoning_effort == "low"  # the built client's, untouched


# ── settings ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "value",
    ["", "none", "low", "Medium", " high ", "xhigh", "max"],
)
def test_the_effort_accepts_unillms_efforts(value, monkeypatch):
    monkeypatch.setenv("UNIFY_REVIEW_REASONING_EFFORT", value)
    assert ProductionSettings().UNIFY_REVIEW_REASONING_EFFORT == value.strip().lower()


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("UNIFY_REVIEW_REASONING_EFFORT", "extreme"),
        ("UNIFY_REVIEW_MODEL", "gpt-5"),
    ],
)
def test_bad_values_are_refused(name, value, monkeypatch):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=name):
        ProductionSettings()


def test_the_model_is_an_endpoint(monkeypatch):
    monkeypatch.setenv("UNIFY_REVIEW_MODEL", f" {REVIEW_MODEL} ")
    assert ProductionSettings().UNIFY_REVIEW_MODEL == REVIEW_MODEL

"""The comparison profile ``UNIFY_PROFILE=paper``: one setting for every reported row.

Under it the memory writer runs on the actor's model, at the actor's effort, through its own accounting lane (the
writer route, required, so its calls are told apart from the actor's), and no checker line is taken. A run that also
sets one of those settings to another value is refused before anything runs, naming the settings and never a value;
without the lane no pass starts. The profile is recorded in the
episode's meta, the journal's header and each pass's start event; a pass's end event has its output tokens beside its
wall time. Without the profile everything is as before.

No model is called: the writer is a fake turn factory over ``consolidate.unillm_turn``. Planted values are
placeholders, checked through booleans so a failing assert never prints one.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
from pydantic import SecretStr

from tests.memory.integration import test_consolidate as tc
from tests.memory.integration.test_episode_journal import (  # noqa: F401 (the home fixture)
    REQUEST,
    TURNS,
    _lines,
    _publish,
    _recorded,
    home,
)
from unify.memory.integration import consolidate, request as request_mod, switch
# the writer's lane refuses while unillm records OTel spans, which the test runner turns on for every session
from tests.memory.test_writer_route import _otel_off_unless_a_test_turns_it_on  # noqa: F401 - an autouse fixture
from unify.memory.integration.state import State
from unify.memory.writer_pass import PassOutcome, WriterPass
from unify.settings import SETTINGS, ProductionSettings

ACTOR_MODEL = "vendor/actor-model@openrouter"
PLANTED = {
    switch.WRITER_MODEL: "vendor/planted-writer-model",
    switch.WRITER_EFFORT: "low",
    switch.CHECKER_VISIBLE: "on",
}


class Turns:
    """A turn factory that records every turn it builds, with its keyword arguments; each turn ends the pass."""

    def __init__(self, usage: dict | None = None) -> None:
        self.made: list[tuple[str, str, dict]] = []
        self.usage = usage

    def __call__(self, model, effort, **kw):
        self.made.append((model, effort, dict(kw)))

        async def turn(messages, tools):
            msg = tc._finish()
            if self.usage is not None:
                msg = {**msg, "usage": dict(self.usage)}
            return msg, "0.01"

        return turn


#: The writer's own accounting lane (a placeholder route and token).
LANE = {
    switch.WRITER_BASE_URL: "http://127.0.0.1:18099/writer/v1",
    switch.WRITER_TOKEN: SecretStr("lane-writer-token-0123456789"),  # pragma: allowlist secret
}


def _paper(**extra) -> SimpleNamespace:
    s = tc._settings()
    s.UNIFY_PROFILE = "paper"
    s.UNIFY_MODEL = ACTOR_MODEL
    for k, v in LANE.items():
        setattr(s, k, v)
    for k, v in extra.items():
        setattr(s, k, v)
    return s


def _pass(tmp_path, settings, effort: str, fake) -> list[dict]:
    stores = tc._stores(tmp_path)
    sha, _ = tc._record(stores, "e1")
    got: list[dict] = []
    asyncio.run(
        consolidate.run_due_passes(
            stores, "e1", sha, State(stores.paths.state), effort=effort, settings=settings, emit=got.append
        ),
    )
    return got


# --- the setting ----------------------------------------------------------------------------------------------------


def test_the_setting_takes_empty_or_paper_and_defaults_to_empty():
    assert ProductionSettings.model_fields["UNIFY_PROFILE"].default == ""
    assert switch.parse_profile("") == "" and switch.parse_profile(" Paper ") == "paper"
    assert ProductionSettings(UNIFY_PROFILE="paper").UNIFY_PROFILE == "paper"
    with pytest.raises(ValueError, match="UNIFY_PROFILE"):
        switch.parse_profile("arm-b")
    assert switch.profile(SimpleNamespace()) == "" and not switch.paper_profile(SimpleNamespace())
    assert switch.profile_conflicts(tc._settings()) == [] and switch.profile_refusal(tc._settings()) is None


# --- what the profile sets --------------------------------------------------------------------------------------


def test_paper_runs_the_writer_on_the_actors_model_and_effort_on_its_own_lane(tmp_path, monkeypatch):
    monkeypatch.setattr(request_mod, "actor_model", lambda: ACTOR_MODEL)
    cfg = consolidate.writer_settings(_paper())
    assert (cfg.model, cfg.profile) == (ACTOR_MODEL, "paper") and cfg.route is not None
    fake = Turns()
    monkeypatch.setattr(consolidate, "unillm_turn", fake)
    got = _pass(tmp_path, _paper(), "high", fake)
    # one turn, on the actor's model at the cell's effort, on the writer's own lane
    ((model, effort, kw),) = fake.made
    assert (model, effort) == (ACTOR_MODEL, "high")
    assert set(kw) == {"route", "session"} and kw["route"] is not None and kw["session"] == "writer.e1.p0"
    start = got[0]
    assert start["phase"] == "start" and start["profile"] == "paper"
    assert (start["writer_model"], start["writer_effort"]) == (ACTOR_MODEL, "high")
    # the budget is the cell's own: the allowance cap at the effort's scale and calls, as without the profile
    assert (start["effort_scale"], start["max_calls"]) == ("5", 80)


def test_paper_keeps_the_writers_effort_the_actors_and_the_checker_line_off(monkeypatch):
    from unify import cli

    monkeypatch.setattr(SETTINGS, "UNIFY_PROFILE", "paper")
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY_WRITER_EFFORT", "low")
    assert request_mod.writer_effort("medium") == "medium"
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY_LIBRARY", "on")
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY_LIBRARY_CHECKER_VISIBLE", "on")
    assert request_mod.checker_visible_on() is False and cli._checker_lines_on() is False
    assert switch.checker_visible(SETTINGS) is False
    # without the profile, as before
    monkeypatch.setattr(SETTINGS, "UNIFY_PROFILE", "")
    assert request_mod.writer_effort("medium") == "low"
    assert request_mod.checker_visible_on() is True and switch.checker_visible(SETTINGS) is True


def test_without_the_profile_the_writer_is_as_before(tmp_path, monkeypatch):
    fake = Turns()
    monkeypatch.setattr(consolidate, "unillm_turn", fake)
    got = _pass(tmp_path, tc._settings(), "low", fake)
    assert fake.made == [("openai/gpt-6-sol", "low", {})]
    assert "profile" not in got[0] and consolidate.writer_settings(tc._settings()).profile == ""


@pytest.mark.parametrize("unset", list(LANE))
def test_paper_without_the_writers_lane_starts_no_pass(tmp_path, monkeypatch, unset):
    """On the actor's own route the writer's calls could not be told from the actor's, nor bounded by its caps."""
    monkeypatch.setattr(request_mod, "actor_model", lambda: ACTOR_MODEL)
    settings = _paper(**{switch.WRITER_BASE_URL: "", switch.WRITER_TOKEN: SecretStr("")})
    if unset == switch.WRITER_TOKEN:  # a route half set is refused as for any run
        settings = _paper(**{switch.WRITER_TOKEN: SecretStr("")})
    made: list = []
    monkeypatch.setattr(consolidate, "unillm_turn", lambda *a, **k: made.append(1))
    with pytest.raises(switch.WriterRouteRefused) as refused:
        consolidate.writer_settings(settings)
    assert switch.WRITER_BASE_URL in str(refused.value)
    stores = tc._stores(tmp_path)
    sha, _ = tc._record(stores, "e1")
    with pytest.raises(switch.WriterRouteRefused):
        asyncio.run(
            consolidate.run_due_passes(
                stores, "e1", sha, State(stores.paths.state), effort="low", settings=settings, emit=None
            ),
        )
    assert made == []


# --- a conflicting setting: refused by name, before anything runs ----------------------------------------------------


@pytest.mark.parametrize("name", list(PLANTED))
def test_a_setting_the_profile_fixes_is_refused_by_name_and_nothing_runs(tmp_path, monkeypatch, name):
    settings = _paper(**{name: PLANTED[name]})
    planted = PLANTED[name]
    raw = planted.get_secret_value() if isinstance(planted, SecretStr) else planted
    assert switch.profile_conflicts(settings) == [name]
    refusal = switch.profile_refusal(settings)
    leaked = len(raw) > 2 and raw in refusal
    assert name in refusal and "UNIFY_PROFILE=paper" in refusal and not leaked
    # the request does not start
    with pytest.raises(request_mod.MemoryUnavailable) as info:
        request_mod.check_profile(settings)
    assert name in str(info.value)
    # no pass starts and no turn is built
    made: list = []
    monkeypatch.setattr(consolidate, "unillm_turn", lambda *a, **k: made.append(1))
    with pytest.raises(switch.WriterRouteRefused) as refused:
        consolidate.writer_settings(settings)
    leaked = len(raw) > 2 and raw in str(refused.value)
    assert name in str(refused.value) and not leaked
    stores = tc._stores(tmp_path)
    sha, _ = tc._record(stores, "e1")
    with pytest.raises(switch.WriterRouteRefused):
        asyncio.run(
            consolidate.run_due_passes(
                stores, "e1", sha, State(stores.paths.state), effort="low", settings=settings, emit=None
            ),
        )
    assert made == []


def test_the_writer_model_set_to_the_actors_own_is_no_conflict_and_a_given_default_is_one():
    assert switch.profile_conflicts(_paper(UNIFY_MEMORY_WRITER_MODEL="vendor/actor-model")) == []
    # a value given at all counts, even the default's (the settings record which were given)
    given = ProductionSettings(
        UNIFY_PROFILE="paper",
        UNIFY_MODEL=ACTOR_MODEL,
        UNIFY_MEMORY_WRITER_MODEL=switch.WRITER_MODEL_DEFAULT,
    )
    assert switch.WRITER_MODEL in switch.profile_conflicts(given)
    same = ProductionSettings(UNIFY_PROFILE="paper", UNIFY_MODEL=ACTOR_MODEL, UNIFY_MEMORY_WRITER_MODEL=ACTOR_MODEL)
    assert switch.WRITER_MODEL not in switch.profile_conflicts(same)


def test_the_cli_refuses_a_conflicting_run_before_it_starts(monkeypatch, capsys):
    from unify import cli

    started: list = []
    monkeypatch.setattr(cli, "_configure_environment", lambda args: None)
    monkeypatch.setattr(cli, "_run_act", lambda args: started.append(1))
    monkeypatch.setattr(SETTINGS, "UNIFY_PROFILE", "paper")
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY_WRITER_EFFORT", "high")
    assert cli.main(["act", "hello"]) == 2 and started == []
    err = capsys.readouterr().err
    assert "UNIFY_MEMORY_WRITER_EFFORT" in err and "UNIFY_PROFILE=paper" in err


# --- what is recorded ---------------------------------------------------------------------------------------------


def _paper_settings(monkeypatch) -> None:
    """The profile on the process's settings, with every setting it fixes at the value it gives it."""
    monkeypatch.setattr(SETTINGS, "UNIFY_PROFILE", "paper")
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY_WRITER_MODEL", SETTINGS.UNIFY_MODEL)
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY_WRITER_EFFORT", "actor")
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY_LIBRARY_CHECKER_VISIBLE", "off")


@pytest.mark.parametrize("paper", [True, False])
def test_the_episode_meta_and_the_journal_header_record_the_profile(home, monkeypatch, paper):  # noqa: F811
    if paper:
        _paper_settings(monkeypatch)
    run = home.ctx.run(request_mod.RequestRun.begin, REQUEST)
    try:
        path = home.paths.journal / f"{run.episode_id}.jsonl"
        (header,) = _lines(path)
        assert header.get("profile") == ("paper" if paper else None)
        assert ("profile" in header) is paper  # a default run's header is as before
        _publish(home, TURNS)
        with asyncio.Runner() as runner:
            runner.run(run.finish(None, progress=lambda _t: None, consolidate=False), context=home.ctx)
    finally:
        home.ctx.run(run.abort)
    repo, rel, ep = _recorded(home.paths, run.episode_id)
    meta = json.loads(repo.show("main", f"{rel}/meta.json"))
    assert ep.profile == ("paper" if paper else "")
    assert meta.get("profile") == ("paper" if paper else None) and ("profile" in meta) is paper


def test_a_pass_reports_its_output_tokens_beside_its_wall_time(monkeypatch):
    counted = SimpleNamespace(
        _cache_tokens=[0, 0, 0, 0, 0],
        _output_tokens=[0, 0],
        _first_prompt=(False, None),
        _max_prompt=None,
    )
    for usage in ({"input": 100, "output": 7}, {"input": 50}, {"output": 5, "cache_read": 20}):
        WriterPass._count_cache(counted, usage)
    assert counted._output_tokens == [12, 2] and counted._cache_tokens == [150, 20, 2, 1, 3]
    monkeypatch.setattr(consolidate, "_library_after", lambda stores: (None, None))
    out = PassOutcome("p1", False, None, "0.01", 3, coverage={}, output_tokens=12, output_tokens_calls=2)
    end = consolidate._end_event(None, "p1", out, [], 4.25, None)
    assert (end["output_tokens"], end["output_tokens_calls"], end["seconds"]) == (12, 2, 4.25)
    none = consolidate._end_event(None, "p1", PassOutcome("p1", False, None, "0", 0, coverage={}), [], 1.0, None)
    assert (none["output_tokens"], none["output_tokens_calls"]) == (None, 0)

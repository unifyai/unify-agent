"""``UNIFY_MEMORY_FROZEN`` (held-out evaluation): memory is served as the home holds
it and nothing changes it. Each part is checked here on its own; the conformance suite runs it end to end through the
CLI (``tests/memory/conformance/test_surfaces.py::test_frozen_memory_serves_what_it_holds_and_changes_nothing``)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from unify.agents import run_record as rr
from unify.agents import run_record_tools as T
from unify.agents.options import Options
from unify.memory.integration import switch

FAST = Options(max_live=16, max_total=64, max_entries=None, max_record_bytes=rr.MAX_RECORD_BYTES, min_post_interval_s=0.0)
_LIBRARY = dict(
    UNIFY_MEMORY="on",
    UNIFY_MEMORY_LIBRARY="on",
    UNIFY_MEMORY_LIBRARY_FORK="on",
    UNIFY_MEMORY_LIBRARY_POOL="on",
    UNIFY_MEMORY_LIBRARIAN="on",
    UNIFY_MEMORY_WRITER_SERVICE_CAP_USD="10",
)


@pytest.fixture
def frozen(monkeypatch):
    from unify.settings import SETTINGS

    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY_FROZEN", "on")


def test_the_setting_is_on_or_off_and_parsed_by_the_settings():
    for v, want in (("", "off"), ("off", "off"), ("on", "on"), (" ON ", "on")):
        assert switch.parse_memory_frozen(v) == want
    for v in ("yes", "1", "frozen", "true"):
        with pytest.raises(ValueError, match="UNIFY_MEMORY_FROZEN must be empty, 'off' or 'on'"):
            switch.parse_memory_frozen(v)
    assert switch.PARSERS[switch.FROZEN] is switch.parse_memory_frozen
    assert switch.memory_frozen(SimpleNamespace(UNIFY_MEMORY_FROZEN="on"))
    assert not switch.memory_frozen(SimpleNamespace(UNIFY_MEMORY_FROZEN="")) and not switch.memory_frozen(SimpleNamespace())


def test_the_settings_field_defaults_off_and_takes_on(monkeypatch):
    from unify.settings import SETTINGS

    monkeypatch.delenv("UNIFY_MEMORY_FROZEN", raising=False)
    assert type(SETTINGS)().UNIFY_MEMORY_FROZEN == "off"
    monkeypatch.setenv("UNIFY_MEMORY_FROZEN", "on")
    assert type(SETTINGS)().UNIFY_MEMORY_FROZEN == "on"
    monkeypatch.setenv("UNIFY_MEMORY_FROZEN", "sometimes")
    with pytest.raises(Exception, match="UNIFY_MEMORY_FROZEN must be empty, 'off' or 'on'"):
        type(SETTINGS)()


def test_it_is_allowed_beside_the_profile_and_every_arm_and_the_profile_never_sets_it():
    kit = {**_LIBRARY, **switch.REDESIGN_COMPANIONS, "UNIFY_HOST_LEASE_END": "2026-10-21T06:00:00Z", switch.FROZEN: "on"}
    for arm, values in switch.REDESIGN_ARMS.items():
        settings = SimpleNamespace(**({**kit, **values} if arm != "no_memory" else {**switch.REDESIGN_COMPANIONS,
                                                                                     **values, switch.FROZEN: "on"}))
        assert switch.profile_refusal(settings) is None, (arm, switch.profile_refusal(settings))
    assert switch.FROZEN not in switch.REDESIGN_SWITCHES and switch.FROZEN not in switch.REDESIGN_CONFLICTS
    assert not any(switch.FROZEN in v for v in switch.REDESIGN_ARMS.values())


def test_the_writer_service_is_refused_and_never_spawned(tmp_path):
    from unify.memory import writer_service as ws

    spawned: list = []
    settings = SimpleNamespace(
        **{**_LIBRARY, "UNIFY_MEMORY_PROFILE": "redesign", **switch.REDESIGN_COMPANIONS,
           "UNIFY_HOST_LEASE_END": "2099-01-01T00:00:00Z", switch.FROZEN: "on"},
    )
    got = ws.ensure_service(tmp_path / "home", settings=settings, popen=lambda *a, **k: spawned.append(a), wait_s=0.0)
    assert (got.status, got.detail) == ("refused", "memory is frozen") and spawned == []
    assert not (tmp_path / "home").exists() or not ws.read_state(tmp_path / "home")


def test_no_pass_or_capture_worker_starts(tmp_path):
    from unify.memory.integration import async_pass as ap
    from unify.memory.integration import consolidate
    from unify.memory.integration.paths import Paths

    stores = consolidate.open_stores(Paths.under(tmp_path))
    calls: list = []
    settings = SimpleNamespace(**{**_LIBRARY, switch.FROZEN: "on"})
    ev = ap.maybe_spawn(stores, "e1", "1" * 40, effort="low", settings=settings, emit=calls.append,
                        popen=lambda *a, **k: calls.append(a))
    assert ev == {"type": "consolidation", "phase": "frozen", "episode_id": "e1"} and calls == []
    assert ap.read_inflight(stores.paths) is None and not consolidate.events_path(stores.paths).exists()


def _record(home: Path, *, with_writer: bool) -> rr.RunRecord:
    rec = rr.join_run_record(home, options=FAST)
    rec.join("actor-3f9a", "actor", episode="ep-1")
    rec.join("helper-1", "helper", spawner="actor-3f9a")
    if with_writer:
        rec.join("writer", "writer", state="idle")
    return rec


@pytest.mark.parametrize("with_writer", [True, False], ids=["the writer on the roster", "no writer row"])
def test_a_message_to_the_writer_is_answered_not_available_and_not_posted(tmp_path, frozen, with_writer):
    rec = _record(tmp_path / "home", with_writer=with_writer)
    view = rr.RunRecordView(rec, "actor-3f9a", "actor")
    before = len(rr.read_entries(tmp_path / "home"))
    text, form = T.send_text(view, "@writer the served note was wrong here")
    assert form is not None and "@writer: not available: memory is frozen" in text, (text, form)
    assert len(rr.read_entries(tmp_path / "home")) == before
    # to the writer and someone else: posted to the other only, and the sender told why
    text, form = T.send_text(view, "@writer @helper-1 the served note was wrong here")
    assert form is None and "sent to helper-1" in text and "@writer: not available: memory is frozen" in text, text
    (row,) = rr.read_entries(tmp_path / "home")[before:]
    assert row["mentions"] == ["helper-1"]


def test_off_a_message_to_the_writer_is_posted_as_before(tmp_path, monkeypatch):
    from unify.settings import SETTINGS

    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY_FROZEN", "off")
    rec = _record(tmp_path / "home", with_writer=True)
    text, form = T.send_text(rr.RunRecordView(rec, "actor-3f9a", "actor"), "@writer the served note was wrong here")
    assert form is None and "sent to writer" in text and "frozen" not in text


def test_serving_counts_an_actors_report_only_from_a_recorded_episode_when_frozen(tmp_path, monkeypatch):
    from unify.memory import serving as sv
    from unify.memory.integration import consolidate
    from unify.memory.integration.paths import Paths

    stores = consolidate.open_stores(Paths.under(tmp_path / "home"))
    with stores.evidence.db:
        stores.evidence.db.execute(
            "INSERT INTO episodes(episode_id, commit_sha, started_at, regime, memory_main, request) VALUES(?,?,?,?,?,?)",
            ("ep-recorded", "1" * 40, "2026-10-20T00:00:00Z", "dense", "a" * 40, ""),
        )
    reports = [{"item": "notes/a.md", "episode": "ep-recorded", "seq": 3},
               {"item": "notes/a.md", "episode": "ep-frozen-request", "seq": 9}]
    monkeypatch.setattr(sv, "actor_reports", lambda home: reports)
    live = sv.item_evidence(stores, checker_visible=True, home=tmp_path, labels=lambda i: None)
    held = sv.item_evidence(stores, checker_visible=True, home=tmp_path, labels=lambda i: None, recorded_only=True)
    assert [(r["ref"], r["pin"]) for r in live["notes/a.md"]["failure_record"]] == [("entry:3", "a" * 40), ("entry:9", "")]
    assert [r["ref"] for r in held["notes/a.md"]["failure_record"]] == ["entry:3"]


def test_the_checkpoint_tool_says_memory_is_frozen(monkeypatch):
    from unify.memory.integration import in_episode, request

    monkeypatch.setattr(request, "current", lambda: SimpleNamespace(in_episode=None, frozen=True))
    out = in_episode.checkpoint(SimpleNamespace(), {"note": "x"})
    assert "memory is frozen in this run" in str(out)
    monkeypatch.setattr(request, "current", lambda: SimpleNamespace(in_episode=None, frozen=False))
    assert "memory during an episode is off in this run" in str(in_episode.checkpoint(SimpleNamespace(), {}))

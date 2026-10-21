"""Memory during an episode (``UNIFY_MEMORY_IN_EPISODE=on``): checkpoints, segment forks and a re-pin at the next cell.

Each checkpoint source starts one segment fork for the cells since the last pause, with its own staging dir, and a
settled segment becomes an attention entry; a commit the writer lands is served from the next cell boundary, never
mid-cell, with one line naming exactly what changed; uses carry the pin served when they were made. The re-pin tests
run cells in the real sandboxed worker (bubblewrap); no model is called and no fork process starts.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tests.actor.code_act.sandbox_world import needs_bwrap, world  # noqa: F401
from tests.memory.test_layout import DATES, LIB
from tests.memory.test_library_helper import _commit
from unify.common import tool_registry as reg
from unify.memory import memory_writer, segments
from unify.memory.attention import HomeSources, Ledger
from unify.memory.episodes import Cell, Episode
from unify.memory.gitio import Repo
from unify.memory.integration import fork as fork_mod
from unify.memory.integration import hooks
from unify.memory.integration import in_episode as ie_mod
from unify.memory.integration import request as request_mod
from unify.memory.integration.checkout import export_actor_library_mode
from unify.memory.integration.in_episode import HELD_LINE, InEpisode, diff, notice
from unify.memory.integration.paths import Paths
from unify.memory.integration.request import RequestRun, with_pins
from unify.memory.redact import Redactor
from unify.settings import SETTINGS

EID = "a1b2c3d4e5f60718"


# --- a request run as the in-episode code reads it, with no model and no fork process ----------------------------------


class _Run:
    """The parts of a :class:`RequestRun` :class:`InEpisode` reads: its paths and episode id, the episode as it stands
    (one cell per code text so far), and the session handle a segment fork continues."""

    def __init__(self, home: Path) -> None:
        self.paths = Paths.under(home)
        self.episode_id = EID
        self.code: list[str] = []
        self.handle: Any = object()
        self.repinned_trees: list[Path] = []

    def partial_episode(self) -> tuple[Episode, Redactor]:
        cells = [Cell(i, c, f"out {i}\n") for i, c in enumerate(self.code)]
        ep = Episode(
            episode_id=self.episode_id,
            started_at="2026-10-20T00:00:00.000Z",
            ended_at="2026-10-20T00:00:30.000Z",
            build="b",
            model="m",
            effort="low",
            regime="none",
            memory_main="0" * 40,
            worktree_before=None,
            worktree_after=None,
            request=["Total the rows."],
            transcript=[],
            cells=cells,
            actions=[],
        )
        return ep, Redactor()

    def repinned(self, tree: Path) -> None:
        self.repinned_trees.append(tree)


class _Forks:
    """``fork.start`` without a process: each call is kept, and its fork settles (a summary and ``fork.json``) at once
    or when the test says so."""

    def __init__(self, settle_now: bool = True) -> None:
        self.settle_now = settle_now
        self.calls: list[dict] = []
        self.pending: list = []

    def start(self, paths, handle, eid, settings, *, reservation=None, parts=None, instruction=None, **_kw):
        self.calls.append({"eid": eid, "handle": handle, "parts": parts, "instruction": instruction,
                           "staging": reservation.staging})  # fmt: skip

        def settle() -> None:
            (reservation.staging / "summary.md").write_text("## Asked\nTotal the rows.\n## Approach\n- summed\n")
            (reservation.staging / "fork.json").write_text(
                json.dumps({"episode": eid, "status": "ok", "files": ["summary.md"]}),
            )
            reservation.close()

        if self.settle_now:
            settle()
        else:
            self.pending.append(settle)
        writer = threading.Thread(target=lambda: None)
        writer.start()
        return SimpleNamespace(writer=writer, proc=None, staging=reservation.staging, eid=eid)


class _ForksOnly(HomeSources):
    """The home's settled forks, and nothing else the ledger reads (no evidence store in these tests)."""

    def capture_rows(self):
        return {}

    def use_failures(self):
        return []

    def late_signals(self):
        return []

    def episode_calls(self):
        return []

    def episode_items(self, episode):
        return ()

    def gate_refusals(self):
        return []

    def record_entries(self):
        return []


@pytest.fixture
def fork_on(monkeypatch):
    services: list = []
    monkeypatch.setattr(ie_mod, "_ensure_service", lambda home: services.append(Path(home)))
    return services


def _tree(files: dict[str, str], items: dict | None = None, withdrawn: dict | None = None) -> dict[str, str]:
    out = dict(files)
    out[".memory/items.json"] = json.dumps({"version": 1, "items": items or {}})
    if withdrawn is not None:
        out[".memory/captures/withdrawn.json"] = json.dumps({"captures": withdrawn})
    return out


def _exporter(trees: dict[str, dict[str, str]], seen: list | None = None):
    def export(sha: str, dest: Path) -> None:
        if seen is not None:
            seen.append(sha)
        for rel, text in trees[sha].items():
            (dest / rel).parent.mkdir(parents=True, exist_ok=True)
            (dest / rel).write_text(text)

    return export


def _ie(tmp_path, trees, first="c1" * 20, **kw) -> tuple[_Run, InEpisode]:
    run = _Run(tmp_path / "home")
    run.paths.state_dir.mkdir(parents=True)
    ie = InEpisode(run, _exporter(trees, kw.get("seen")))
    ie.open(first)
    return run, ie


def _cells(run: _Run, ie: InEpisode, n: int) -> None:
    for _ in range(n):
        run.code.append(f"x = {len(run.code)}")
        ie.cell_done()


C1, C2, C3 = "c1" * 20, "c2" * 20, "c3" * 20
NEW_DATES = DATES.replace('return "-".join(tokens(s.replace("-", " ")))', 'return "v2:" + s')
EXTRA = '"""More text."""\n\n\ndef shout(s):\n    """Upper case."""\n    return s.upper()\n'


# --- 1. each checkpoint source makes a segment fork and an entry -------------------------------------------------------


@pytest.mark.parametrize("source", ["actor_call", "compaction", "new_message", "signal"])
def test_each_checkpoint_source_starts_a_segment_fork_and_makes_an_entry(tmp_path, monkeypatch, fork_on, source):
    forks = _Forks()
    monkeypatch.setattr(fork_mod, "start", forks.start)
    run, ie = _ie(tmp_path, {C1: _tree(LIB)})
    _cells(run, ie, 3)
    handle = object()
    if source in ("actor_call", "signal"):
        ie.mark(source, "first part done" if source == "actor_call" else "")
        event = ie.at_boundary(handle)
    else:
        event = ie.at_boundary(handle, compacting=source == "compaction", new_message=source == "new_message")
    assert event["sources"] == [source] and event["segment"] == 1 and event["cells"] == [0, 2], event
    assert event["fork"] == "started"
    [call] = forks.calls
    name = f"{EID}.s1"
    assert call["eid"] == name and call["handle"] is handle
    # its own staging dir, its own record of the episode as it stood, and the short instruction for its cells
    assert call["staging"] == run.paths.state_dir / "memory-staging" / name
    assert segments.load_snapshot(run.paths.state_dir, name).cells[-1].code == "x = 2"
    assert "`cell:0` to `cell:2`" in call["instruction"] and call["instruction"] != fork_mod.fork_instruction()
    assert ("first part done" in call["parts"]) is (source == "actor_call")
    assert fork_on == [run.paths.home]  # the writer service is called at the checkpoint
    # the settled segment is a new_experience entry naming its segment
    [entry] = Ledger(run.paths.home).sync(_ForksOnly(run.paths.home))
    assert (entry.kind, entry.refs.episodes, entry.refs.segment) == ("new_experience", (EID,), 1)
    # the checkpoint is kept with the episode's id
    row = json.loads((run.paths.state_dir / "memory-segments" / "checkpoints.jsonl").read_text().splitlines()[-1])
    assert row["episode"] == EID and row["sources"] == [source] and row["segment"] == 1


def test_segments_cover_the_cells_since_the_last_pause_and_a_pause_with_no_cell_starts_nothing(
    tmp_path, monkeypatch, fork_on,
):
    forks = _Forks()
    monkeypatch.setattr(fork_mod, "start", forks.start)
    run, ie = _ie(tmp_path, {C1: _tree(LIB)})
    assert ie.at_boundary() is None  # nothing marked: nothing runs
    _cells(run, ie, 2)
    ie.mark("actor_call")
    ie.mark("signal")
    first = ie.at_boundary()
    assert first["sources"] == ["actor_call", "signal"] and first["cells"] == [0, 1]  # two marks, one segment
    skipped = ie.at_boundary(new_message=True)
    assert skipped["skipped"] == "no cell ran since the last pause" and skipped["cells"] is None
    _cells(run, ie, 1)
    second = ie.at_boundary(compacting=True)
    assert second["segment"] == 2 and second["cells"] == [2, 2]
    assert [c["eid"] for c in forks.calls] == [f"{EID}.s1", f"{EID}.s2"]
    assert "`cell:2`" in forks.calls[1]["instruction"]


def test_a_pause_the_actor_marks_at_the_end_is_left_to_the_episode_fork(tmp_path, monkeypatch, fork_on):
    monkeypatch.setattr(fork_mod, "start", _Forks().start)
    run, ie = _ie(tmp_path, {C1: _tree(LIB)})
    _cells(run, ie, 1)
    ie.mark("actor_call")
    ie.finish(SETTINGS)
    assert ie.events[-1]["skipped"].startswith("the episode ended first") and not ie.pending


def test_the_actor_tool_and_the_hooks(tmp_path, monkeypatch, fork_on):
    monkeypatch.setattr(fork_mod, "start", _Forks().start)
    run, ie = _ie(tmp_path, {C1: _tree(LIB)})
    run.in_episode = ie
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY", "on")
    monkeypatch.setattr(request_mod, "_CURRENT", run)
    # tools.checkpoint marks a pause
    reply = ie_mod.checkpoint(None, {"note": "read the input"})
    assert reply.status == "ok" and ie.pending == [("actor_call", "read the input")]
    assert ie_mod.checkpoint(None, {"note": 3}).status == "refused"
    # a code cell's result counts one cell
    run.code.append("x = 0")
    hooks.tool_result("execute_code", "c0", {"error": None})
    hooks.tool_result("send_message", "m0", {"error": None})
    assert ie.cells == 1
    # only the request's own actor's loop takes its turn boundary
    client = object()
    run.handle = SimpleNamespace(_inner=SimpleNamespace(_client=client))
    hooks.turn_boundary(object())
    assert ie.pending
    hooks.turn_boundary(client)
    assert not ie.pending and ie.events[-1]["segment"] == 1
    assert "checkpoint" in reg.allow_list("actor", frozenset({"checkpoint"}))
    # with no run, the tool says why
    monkeypatch.setattr(request_mod, "_CURRENT", None)
    refused = ie_mod.checkpoint(None, {})
    assert refused.status == "refused" and "memory during an episode is off" in refused.text


def test_the_checkpoint_tool_exists_only_with_its_flag():
    assert "checkpoint" not in reg.allow_list("actor", frozenset())
    assert "checkpoint" in reg.allow_list("actor", frozenset({"checkpoint"}))
    assert "checkpoint" not in reg.allow_list("writer", frozenset({"checkpoint"}))
    assert reg.help_text("checkpoint", "actor")


@pytest.mark.parametrize("flagged", [True, False])
def test_the_actor_gets_checkpoint_only_with_in_episode(monkeypatch, flagged):
    from unify.actor import tool_handlers as th

    monkeypatch.setattr(th, "_in_episode", lambda settings: flagged)
    flags = th.actor_flags("actor", can_store=False, librarian=False)
    assert ("checkpoint" in flags) is flagged
    assert "checkpoint" not in th.actor_flags("teammate", can_store=False, librarian=False)


def test_an_in_episode_verdict_marks_a_signal_pause(tmp_path, monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY", "on")
    run = RequestRun("r", Paths.under(tmp_path / "home"))
    marks: list = []
    run.in_episode = SimpleNamespace(mark=lambda source, note="": marks.append(source))
    monkeypatch.setattr(run, "_observations_now", lambda: 0)
    assert run.take_checker({"label": "fail"})["accepted"] is True
    assert marks == ["signal"]


# --- 2./5./7. a commit is served from the next cell, never mid-cell (real worker) ---------------------------------------


def _library_run(monkeypatch, paths: Paths, mem: Repo, first: str) -> RequestRun:
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY", "on")
    run = RequestRun("r", paths)
    run.library_mode = True
    run.pin = first
    run.stores = SimpleNamespace(memory=mem)
    run.in_episode = InEpisode(run, lambda sha, dest: export_actor_library_mode(mem.git_dir, sha, dest))
    run.in_episode.open(first)
    monkeypatch.setattr(request_mod, "_CURRENT", run)
    return run


async def _cell(ex: Any, code: str, session: int = 0) -> Any:
    res = await asyncio.wait_for(ex.execute(code=code, state_mode="stateful", session_id=session), timeout=120)
    hooks.tool_result("execute_code", f"c{session}-{time.monotonic_ns()}", res)
    return res


def _text(res: Any) -> str:
    from unify.actor.execution.types import parts_to_text

    return parts_to_text(res["stdout"])


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_a_commit_landed_mid_cell_is_served_from_the_next_cell_with_one_notice(
    world,  # noqa: F811
    monkeypatch,
):
    from tests.memory.cells import Cells

    paths = Paths.under(world["state"])
    mem = Repo.init_bare(paths.memory)
    first = _commit(mem, LIB, "seed the library")
    head = {"sha": first}
    monkeypatch.setattr(memory_writer, "served_head", lambda repo: head["sha"])
    run = _library_run(monkeypatch, paths, mem, first)
    started, go = world["workspace"] / "started", world["workspace"] / "go"
    ex = Cells()
    try:
        # cell 0, session 1: another worker loads the library at the first pin
        res = await _cell(ex, "import memory.text.dates as d1\nd1.parse_date('2026-10-09')", session=1)
        assert res["error"] is None and res["result"] == "2026-10-09", res["error"]
        # cell 1: two commits land while it runs; it reads the first pin throughout, and no notice ends it
        code = (
            "import os, time\n"
            "from memory.text.dates import parse_date as held\n"
            "bound = memory.index().splitlines()[0]\n"
            "open('/workspace/started', 'w').close()\n"
            "while not os.path.exists('/workspace/go'):\n    time.sleep(0.05)\n"
            "import memory.text.dates as again\n"
            "[held('2026-10-09'), again.parse_date('2026-10-09'),"
            " open('/memory-library/current/memory/text/dates.py').read().count('v2:'),"
            " os.path.exists('/memory-library/current/memory/text/extra.py')]"
        )
        task = asyncio.create_task(_cell(ex, code))
        for _ in range(1200):
            if started.exists():
                break
            await asyncio.sleep(0.05)
        assert started.exists()
        second = _commit(mem, {"memory/text/dates.py": NEW_DATES}, "change parse_date")
        head["sha"] = second
        third = _commit(mem, {"memory/text/extra.py": EXTRA}, "add shout")
        head["sha"] = third
        go.write_text("")
        res = await task
        assert res["error"] is None, res["error"]
        assert res["result"] == ["2026-10-09", "2026-10-09", 0, False]
        assert "memory changed" not in _text(res)
        assert [p["commit"] for p in run.in_episode.pin_rows()] == [first]
        # cell 2: one re-pin, to the latest commit, and one line naming exactly both changes
        res = await _cell(
            ex,
            "import memory.text.extra as x\nimport memory.text.dates\n"
            "[held('2026-10-09'), memory.text.dates.parse_date('2026-10-09'), x.shout('a')]",
        )
        assert res["error"] is None, res["error"]
        assert res["result"] == ["2026-10-09", "v2:2026-10-09", "A"]  # a held name keeps its old code
        lines = [ln for ln in _text(res).splitlines() if ln.startswith("memory changed")]
        assert lines == [
            f"memory changed (commit {third[:12]}): +fn:memory.text.extra:shout ~fn:memory.text.dates:parse_date; "
            f"call memory.items() to see them. {HELD_LINE}",
        ]
        assert run.in_episode.pin_rows() == [
            {"from_cell": 0, "commit": first},
            {"from_cell": 2, "commit": third, "landed": ["fn:memory.text.dates:parse_date",
                                                         "fn:memory.text.extra:shout"], "retired": []},
        ]  # fmt: skip
        # cell 3: no new commit, no re-pin and no notice
        res = await _cell(ex, "memory.text.dates.parse_date('x')")
        assert res["result"] == "v2:x" and "memory changed" not in _text(res)
        # cell 4, session 1: the other worker drops what it loaded at the old pin, silently
        res = await _cell(ex, "import memory.text.dates as d2\n[d2.parse_date('y'), d1 is d2]", session=1)
        assert res["result"] == ["v2:y", False] and "memory changed" not in _text(res)
    finally:
        await ex.close()
    # 4. uses before and after the re-pin carry their pins: the held name's calls are counted under the pin its code
    # came from (cells 0, 1 and 2), the new code's under the new pin (cells 2, 3 and 4)
    item = "memory.text.dates:parse_date"
    assert run.pin_counts[first][item]["ok"] == 4
    assert run.pin_counts[third][item]["ok"] == 3
    assert run.pin_counts[third]["memory.text.extra:shout"]["ok"] == 1
    record = with_pins({"items": {}, "run_rows": []}, run.pin_counts, run.cell_pins)
    assert set(record["items"][item]["pins"]) == {first, third}
    assert record["pins_served"] == sorted({first, third})
    assert run.cell_pins[:3] == [first, first, third]


# --- 3. the notice names exactly the difference between the pins ---------------------------------------------------------


def test_the_notice_names_exactly_what_changed_between_two_pins(tmp_path):
    cap = "0123456789abcdef"
    note = "notes/text/dates.md"
    old = _tree(
        {**LIB, f".memory/captures/{cap}/procedure.py": "def run(x):\n    return x\n"},
        items={"memory.web.fetch:get": {"status": "on-trial"}},
    )
    new_files = {k: v for k, v in LIB.items() if k != "memory/web/fetch.py"}
    new_files["memory/text/dates.py"] = NEW_DATES
    new_files[note] = LIB[note] + "More.\n"
    new_files["memory/text/extra.py"] = EXTRA
    new = _tree(
        new_files,
        items={"memory.web.fetch:get": {"status": "retired"}},
        withdrawn={cap: {"status": "superseded", "by": "memory.text.extra:shout"}},
    )
    trees = {C1: old, C2: new}
    run, ie = _ie(tmp_path, trees)
    _cells(run, ie, 4)
    step = ie.before_cell(C2)
    ie.after_cell()
    added, changed, gone = diff(ie.pins[0].path, ie.pins[1].path)
    assert added == ["fn:memory.text.extra:shout"]
    assert changed == ["fn:memory.text.dates:parse_date", note]
    assert gone == [(f"capture:{cap}", "superseded"), ("fn:memory.web.fetch:get", "retired")]
    assert step["notice"] == notice(C2, added, changed, gone, first=True)
    assert step["notice"].startswith(
        f"memory changed (commit {C2[:12]}): +fn:memory.text.extra:shout ~fn:memory.text.dates:parse_date "
        f"~{note} −capture:{cap} (superseded) −fn:memory.web.fetch:get (retired); call memory.items() to see them.",
    )
    assert ie.pin_rows()[1]["from_cell"] == 4 and run.repinned_trees == [ie.pins[1].path]
    # the link swap is atomic, to a fresh directory, and the old pin stays whole for a reader that holds it
    assert (ie.copy_root).resolve() == ie.pins[1].path.resolve()
    assert (ie.pins[0].path / "memory/web/fetch.py").is_file()


def test_no_re_pin_while_another_cell_runs_and_at_most_one_per_boundary(tmp_path):
    seen: list = []
    trees = {C1: _tree(LIB), C2: _tree({**LIB, "memory/text/dates.py": NEW_DATES}),
             C3: _tree({**LIB, "memory/text/dates.py": NEW_DATES, "memory/text/extra.py": EXTRA})}  # fmt: skip
    run, ie = _ie(tmp_path, trees, seen=seen)
    running = ie.before_cell(None)  # a cell is running in one worker
    assert running == {"generation": 0, "notice": None}
    other = ie.before_cell(C2)  # another worker's boundary while it runs: no swap under the running cell
    assert other == {"generation": 0, "notice": None} and ie.current == C1
    ie.after_cell()
    ie.after_cell()
    # two commits landed: one re-pin, to the latest; a failed export leaves the pin as it was
    step = ie.before_cell(C3)
    ie.after_cell()
    assert step["generation"] == 1 and ie.current == C3 and seen == [C1, C3]
    assert "extra:shout" in step["notice"] and "parse_date" in step["notice"] and HELD_LINE in step["notice"]
    again = ie.before_cell(C3)
    ie.after_cell()
    assert again == {"generation": 1, "notice": None}

    def broken(sha, dest):
        raise OSError("disk")

    ie.export = broken
    failed = ie.before_cell("c4" * 20)
    ie.after_cell()
    assert failed == {"generation": 1, "notice": None} and ie.current == C3
    assert ie.events[-1]["error"] == "OSError"
    # the second notice of the episode no longer repeats the warning about held names
    ie.export = _exporter({**trees, "c5" * 20: _tree(LIB)})
    later = ie.before_cell("c5" * 20)
    ie.after_cell()
    assert later["notice"] and HELD_LINE not in later["notice"]


# --- 4. uses are attributed to the version served when they were made ---------------------------------------------------


def test_a_use_counts_for_the_version_it_was_served_under():
    from unify.memory.lifecycle import EpisodeUse, ItemUse, served, version_use

    item = "memory.text.dates:parse_date"
    u = ItemUse(used=True, uses=2, refused=False, errored=True, unknown=False, basis="runtime", calls=(1, 0, 1))
    f = EpisodeUse(EID, C1, "dense", False, False, {item: u})
    # the old version raised before the re-pin; the new version returned after it
    pins = {(EID, item): [(C1, 0, 0, 1), (C3, 1, 0, 0)]}
    assert served(f, item, {C3}, pins) and served(f, item, {C1}, pins) and not served(f, item, {C2}, pins)
    new = version_use(f, item, {C3}, pins).items[item]
    old = version_use(f, item, {C1}, pins).items[item]
    assert (new.used, new.errored, new.calls) == (True, False, (1, 0, 0))
    assert (old.used, old.errored, old.calls) == (False, True, (0, 0, 1))
    # without pins, by the episode's first pin, as built
    assert served(f, item, {C1}, {}) and not served(f, item, {C3}, {}) and version_use(f, item, {C3}, {}) is f


def test_the_use_record_and_the_evidence_store_carry_the_pins(tmp_path):
    from unify.memory.evidence import EvidenceStore, _pin_rows
    from unify.memory.lifecycle import use_pins

    item = "memory.text.dates:parse_date"
    record = with_pins(
        {"items": {item: {"calls": 2}}, "run_rows": [{"cell": 0, "target": item}, {"cell": 1, "target": item}]},
        {C1: {item: {"ok": 0, "refused": 0, "error": 1}}, C3: {item: {"ok": 1, "refused": 0, "error": 0}}},
        [C1, C3],
    )
    assert record["items"][item]["pins"] == {C1: {"ok": 0, "refused": 0, "error": 1},
                                             C3: {"ok": 1, "refused": 0, "error": 0}}  # fmt: skip
    assert [r["pin"] for r in record["run_rows"]] == [C1, C3]
    rows = _pin_rows(EID, record)
    assert rows == [(item, EID, C1, 0, 0, 1), (item, EID, C3, 1, 0, 0)]
    ev = EvidenceStore(tmp_path / "evidence.sqlite")
    from unify.memory.evidence import _PINS_SCHEMA

    ev.db.execute(_PINS_SCHEMA)
    ev.db.executemany("INSERT INTO item_use_pins VALUES(?,?,?,?,?,?)", rows)
    assert use_pins(ev) == {(EID, item): [(C1, 0, 0, 1), (C3, 1, 0, 0)]}
    # a use record that is not one is left alone
    assert with_pins({"error": "x"}, {}, []) == {"error": "x"}


# --- 6. no checkpoints, or the switch off: as today ------------------------------------------------------------------------


def test_off_every_hook_is_inert_and_the_fork_text_is_as_built(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY", "on")
    run = SimpleNamespace(in_episode=None, handle=None)
    monkeypatch.setattr(request_mod, "_CURRENT", run)
    assert hooks.in_episode_active() is False and hooks.before_cell() is None and hooks.repin_roots() == []
    hooks.turn_boundary(object(), compacting=True)
    hooks.after_cell()
    assert fork_mod.fork_message("parts") == f"{fork_mod.fork_instruction()}\n\nparts"
    assert "checkpoint" not in reg.allow_list("actor", frozenset())
    monkeypatch.setattr(request_mod, "_CURRENT", None)
    assert hooks.in_episode_active() is False and hooks.before_cell() is None


def test_on_with_no_pause_and_no_commit_nothing_changes(tmp_path, monkeypatch, fork_on):
    forks = _Forks()
    monkeypatch.setattr(fork_mod, "start", forks.start)
    run, ie = _ie(tmp_path, {C1: _tree(LIB)})
    for _ in range(3):
        assert ie.before_cell(C1) == {"generation": 0, "notice": None}
        ie.after_cell()
        _cells(run, ie, 1)
        assert ie.at_boundary() is None
    ie.finish(SETTINGS)
    assert forks.calls == [] and ie.events == [] and ie.pin_rows() == [{"from_cell": 0, "commit": C1}]
    assert not (run.paths.state_dir / "memory-staging").exists()
    assert sorted(p.name for p in (ie.checkout / ".pins").iterdir()) == [f"0-{C1[:12]}"]


# --- 8. an episode that ends while its segment fork runs ----------------------------------------------------------------


def test_a_segment_fork_still_running_at_the_end_settles_and_its_entry_comes_after(tmp_path, monkeypatch, fork_on):
    forks = _Forks(settle_now=False)
    monkeypatch.setattr(fork_mod, "start", forks.start)
    run, ie = _ie(tmp_path, {C1: _tree(LIB)})
    _cells(run, ie, 2)
    ie.mark("actor_call")
    assert ie.at_boundary()["fork"] == "started"
    # the episode ends: its own fork reserves its own staging dir beside the segment's
    ended = datetime.now(timezone.utc)
    own = fork_mod.reserve(run.paths, EID, SETTINGS)
    assert own is not None and own.staging != forks.calls[0]["staging"]
    ie.finish(SETTINGS)
    clock = {"now": ended + timedelta(seconds=1)}
    ledger = Ledger(run.paths.home, clock=lambda: clock["now"])
    assert ledger.sync(_ForksOnly(run.paths.home)) == []  # not settled yet: no entry
    for settle in forks.pending:
        settle()
    clock["now"] = ended + timedelta(seconds=5)
    [entry] = ledger.sync(_ForksOnly(run.paths.home))
    assert entry.refs.episodes == (EID,) and entry.refs.segment == 1
    assert entry.created_at > ended.isoformat(timespec="milliseconds").replace("+00:00", "Z")
    own.close()


# --- the conformance findings: a segment is a citable source; the forks after a compaction run --------------------------


def test_after_a_compaction_the_forks_continue_the_compacted_session_only_in_episode(monkeypatch):
    from tests import cache_discipline_helpers as h
    from unify.actor import code_act_actor as caa
    from unify.common._async_tool import cache_discipline as cd

    client = h.new_client("You are a scripted actor.")
    client._messages.extend([{"role": "user", "content": "(the compacted summary)"}, {"role": "user", "content": "go"}])
    cd.record_sent_request(client, list(client.messages), {"tools": [{"type": "function", "function": {"name": "t"}}],
                                                           "tool_choice": "auto"})  # fmt: skip
    inner = SimpleNamespace(_client=client, _compression=SimpleNamespace(count=1))
    actor = SimpleNamespace(_preprocess_msgs=None)
    assert caa._session_fork_source(inner, actor) == (None, "the session's history was compressed")
    source, why = caa._session_fork_source(inner, actor, allow_compressed=True)
    assert why is None and source["sent_messages"][-1]["content"] == "go"
    # the fork asks for it only under the switch (off: the call as built, which test doubles of two arguments take)
    seen: list = []

    def fake(i, a, **kw):
        seen.append(kw)
        return None, "stop"

    monkeypatch.setattr(caa, "_session_fork_source", fake)
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY", "on")
    for on in ("off", "on"):
        monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY_IN_EPISODE", on)
        fork_mod.request_doc(SimpleNamespace(_inner=inner, _actor=actor), EID, "http://127.0.0.1:1", "t")
    assert seen == [{}, {"allow_compressed": True}]


# --- the replay's recorded pauses -----------------------------------------------------------------------------------------



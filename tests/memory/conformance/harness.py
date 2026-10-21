"""The conformance suite's run: the redesign's profile on a fresh memory home, the CLI in process, every memory lane
answered by :mod:`.fake_lanes`, and the run's end as a kit ends it (the drain, verified, before the lanes stop).

Nothing here names a world. The actor's own calls are scripted in process (:mod:`tests.scripted_model`); the fork, the
writer service and its helpers reach their models through loopback routes to :mod:`.fake_lanes`, the form the build
accepts, so the service's own process (a detached child) is driven exactly as on a host. No key exists: the routes'
tokens are fixed test strings, never a credential.

The checks every surface runs at its end (:func:`check_run`):

* **no world vocabulary** in anything a memory agent was sent (the world-vocabulary guard's pattern and allow-list, over every
  memory lane's request: the fork's, the writer's, the helpers');
* **the coverage guarantee** (the attention ledger): every attention entry is closed, unreadable or not reached, with a reason;
* **metering sums** (A3): the entries' shares plus the handed-but-unread zeros equal the passes' known USD, exactly,
  in Decimal; the items' shares do not exceed the passes'; the lanes' booked USD equals what the fake lanes charged;
* **verified termination** (A2): the drain's report says terminated, and no process of the service's group is left.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Iterable

from tests.memory.conformance import fake_lanes as FL

#: Route tokens: test strings only (the build checks their form, never their value).
TOKENS = {"writer": "writer-token-0123456789abcdef", "fork": "fork-token-0123456789abcdef",
          "helper": "helper-token-0123456789abcdef"}
#: The models the lanes are asked for (any string the build accepts; the fake answers every model).
WRITER_MODEL = "openai/gpt-6-luna"
#: The writer service's own cap in these runs (a decimal string, as kits set it).
SERVICE_CAP_USD = "100"  # no writer journal here: every pass settles at its reservation (worst case)
#: How long a surface waits for the service to have nothing owed before it drains.
SETTLE_S = 180.0
#: The in-episode arm's setting (rev 2's main profile runs without in-episode memory): the surfaces that check segment
#: forks and serving while an episode runs set it on top of :func:`redesign_env`.
IN_EPISODE_ON = {"UNIFY_MEMORY_IN_EPISODE": "on"}

GUARD = re.compile(
    r"\b(grid|grids|puzzle|demo|demos|task|tasks|airline|booking|crafter|arc|poker|hand|game|colleague|office|stencil"
    r"|colour|color)\b",
    re.IGNORECASE,
)
ALLOWLIST = Path(__file__).resolve().parents[1] / "no_world_vocabulary_allowlist.tsv"


def lease_end(hours: float = 6.0) -> str:
    return (datetime.now(timezone.utc) + timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%SZ")


def redesign_env(port: int) -> dict[str, str]:
    """The settings a kit gives a redesign cell, with this run's loopback routes."""
    return {
        "UNIFY_MEMORY": "on",
        "UNIFY_MEMORY_PROFILE": "redesign",
        "UNIFY_MEMORY_LIBRARY": "on",
        "UNIFY_MEMORY_LIBRARY_FORK": "on",
        "UNIFY_MEMORY_LIBRARY_POOL": "on",
        "UNIFY_MEMORY_LIBRARY_WAIT_SLOT": "on",
        "UNIFY_MEMORY_LIBRARIAN": "on",
        "UNIFY_MEMORY_LIBRARY_CHECKER_VISIBLE": "on",
        "UNIFY_TOOLS_IN_CODE": "on",
        "UNIFY_REQUEST_METADATA_HEADERS": "on",
        "UNIFY_MEMORY_WRITER_SERVICE_CAP_USD": SERVICE_CAP_USD,
        "UNIFY_HOST_LEASE_END": lease_end(),
        "UNIFY_MEMORY_WRITER_MODEL": WRITER_MODEL,
        "UNIFY_MEMORY_WRITER_BASE_URL": FL.route(port, "writer"),
        "UNIFY_MEMORY_WRITER_TOKEN": TOKENS["writer"],
        "UNIFY_LOCAL_EMBEDDINGS": "false",
        "UNIFY_EMBED_URL": f"http://127.0.0.1:{port}/embed/v1/embeddings",
    }


def apply_settings(monkeypatch: Any, env: dict[str, str]) -> None:
    """Set *env* for this process's settings object and for every child it starts (the fork worker, the service):
    the environment first, then each field copied from a settings object built from it (so types are the model's)."""
    from unify.settings import SETTINGS

    for name, value in env.items():
        monkeypatch.setenv(name, value)
    fresh = type(SETTINGS)()
    for name in env:
        if name in type(SETTINGS).model_fields:
            monkeypatch.setattr(SETTINGS, name, getattr(fresh, name))


def _pipe(text: str) -> int:
    r, w = os.pipe()
    os.write(w, (text + "\n").encode())
    os.close(w)
    os.set_inheritable(r, True)
    return r


def settle_lanes(monkeypatch: Any, port: int) -> None:
    """The fork's and the helpers' routes as a controller gets them: a base URL and an inherited pipe with the token,
    each settled once in this process (so both names leave the environment), as the key proxy's handover does."""
    from types import SimpleNamespace

    from unify.memory.integration import switch as sw

    for name, value in (("_FORK_TOKEN", None), ("_FORK_BASE", ""), ("_FORK_TRIED", False), ("_FORK_REFUSAL", None)):
        if hasattr(sw, name):
            monkeypatch.setattr(sw, name, value)
    for name, value in (("_HELPER_TOKEN", None), ("_HELPER_BASE", ""), ("_HELPER_TRIED", False),
                        ("_HELPER_REFUSAL", None)):
        if hasattr(sw, name):
            monkeypatch.setattr(sw, name, value)
    fork = {sw.LIBRARY_FORK_BASE_URL: FL.route(port, "fork"), sw.LIBRARY_FORK_TOKEN_FD: str(_pipe(TOKENS["fork"]))}
    assert sw.settle_fork_route_env(fork, SimpleNamespace(**fork)) is None
    helper = {sw.LIBRARY_HELPER_BASE_URL: FL.route(port, "helper"),
              sw.LIBRARY_HELPER_TOKEN_FD: str(_pipe(TOKENS["helper"]))}
    assert sw.settle_helper_route_env(helper, SimpleNamespace(**helper)) is None


@dataclass
class Session:
    """One CLI session's result: its exit code and every line it emitted."""

    code: int
    lines: list[dict]
    sent: list[dict] = field(default_factory=list)

    def of(self, kind: str) -> list[dict]:
        return [x for x in self.lines if x.get("type") == kind]

    @property
    def episode_id(self) -> str | None:
        ids = [x.get("episode_id") for x in self.of("episode")]
        return ids[-1] if ids else None


async def cli_session(world: dict, monkeypatch: Any, model: Any, first: str,
                      host: Callable[[dict, list[dict]], list[dict] | None], *, timeout_s: float = 300) -> Session:
    """One ``unify act --persist --jsonl --no-clarify --quiet <first>`` session in process (as the end-to-end tests
    run it). After every emitted line *host* (the line, all lines so far) returns the stdin lines to send next (dicts),
    or None to send nothing; a ``{"quit": true}`` it returns closes the input."""
    from tests.actor.code_act.core_world import new_actor
    from tests.scripted_model import scripted
    from unify.cli import Act, _parse_args

    read_fd, write_fd = os.pipe()
    monkeypatch.setattr(sys, "stdin", os.fdopen(read_fd, "r"))
    session = Act(_parse_args(["act", "--persist", "--jsonl", "--no-clarify", "--quiet", first]))
    lines: list[dict] = []
    sent: list[dict] = []
    closed = False

    def emit(**payload: Any) -> None:
        nonlocal closed
        row = json.loads(json.dumps(payload, default=str))
        lines.append(row)
        if closed:
            return
        for item in host(row, lines) or []:
            sent.append(item)
            os.write(write_fd, (json.dumps(item) + "\n").encode())
            if item == {"quit": True}:
                closed = True
                os.close(write_fd)
                return

    async def start() -> None:
        os.chdir(world["workspace"])
        session._actor = new_actor()

    monkeypatch.setattr(session, "_emit", emit)
    monkeypatch.setattr(session, "start", start)
    try:
        with scripted(model):
            code = await asyncio.wait_for(session.run(first), timeout_s)
    finally:
        await session.close()
        if not closed:
            os.close(write_fd)
    return Session(code, lines, sent)


def replies(*steps: list[dict] | None) -> Callable[[dict, list[dict]], list[dict] | None]:
    """A host that answers the n-th ``response`` line with ``steps[n]`` (then quits after the last)."""
    queue = list(steps)

    def host(row: dict, _lines: list[dict]) -> list[dict] | None:
        if row.get("type") != "response":
            return None
        if queue:
            return queue.pop(0)
        return [{"quit": True}]

    return host


# --- the service -------------------------------------------------------------------------------------------------


def ledger(home: Path) -> Any:
    from unify.memory.attention import Ledger

    return Ledger(Path(home))


def owed(home: Path) -> list[Any]:
    """The attention entries still owed something (open, handed or read)."""
    return [e for e in ledger(home).entries().values() if e.active]


def wait_until_settled(home: Path, timeout_s: float = SETTLE_S, poll_s: float = 0.25, quiet_s: float = 8.0) -> None:
    """Wait until the service has nothing owed, is in no pass, and has stayed so for *quiet_s* (several of its
    polls: the entries a just-settled fork makes appear only at its next look, after the capture step)."""
    from unify.memory import writer_service as ws

    end = time.monotonic() + timeout_s
    quiet_since = None
    while time.monotonic() < end:
        doc = ws.read_state(home) or {}
        calm = not owed(home) and not doc.get("pass_id") and doc.get("state") in (None, "idle", "stopped")
        now = time.monotonic()
        if calm:
            quiet_since = quiet_since if quiet_since is not None else now
            if now - quiet_since >= quiet_s:
                return
        else:
            quiet_since = None
        time.sleep(poll_s)
    raise AssertionError(f"the service did not settle in {timeout_s:g}s: owed {[e.kind for e in owed(home)]}")


def drain(home: Path) -> dict:
    """The kit's teardown step: ``python -m unify.memory.integration.async_pass drain --home H`` in a child with this
    process's environment (exit 2 means not verified). Returns its report."""
    r = subprocess.run([sys.executable, "-m", "unify.memory.integration.async_pass", "drain", "--home", str(home)],
                       capture_output=True, text=True, timeout=600, env=dict(os.environ))
    assert r.returncode in (0, 2), (r.returncode, r.stderr[-2000:])
    report = json.loads(r.stdout.strip().splitlines()[-1])
    report["_exit"] = r.returncode
    return report


# --- the checks -------------------------------------------------------------------------------------------------


def _allowlist() -> list[tuple[str, str]]:
    rows = []
    for line in ALLOWLIST.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            parts = line.split("\t")
            rows.append((parts[0], parts[1]))
    return rows


def world_words(texts: Iterable[str], *, own: Iterable[str] = ()) -> list[str]:
    """Every guard hit in *texts* that no allow-list phrase covers and that is not in the run's own request texts
    (*own*: what the surface itself typed, which a model is free to echo). Returns the hits' context snippets."""
    phrases = [p for _scope, p in _allowlist()]
    own_text = "\n".join(own)
    out = []
    for text in texts:
        for m in GUARD.finditer(text):
            lo, hi = max(0, m.start() - 60), min(len(text), m.end() + 60)
            window = text[lo:hi]
            if any(p in text[max(0, m.start() - 200):m.end() + 200] for p in phrases):
                continue
            if m.group(0).lower() in own_text.lower():
                continue
            out.append(window)
    return out


def model_inputs(lanes: FL.Lanes, actor_model: Any = None) -> list[str]:
    """What the harness sent each model: each lane call's messages (not the model's own turns) and tools, and the
    actor's requests."""
    # the harness's texts only: a scripted model's own turns (role assistant) are the test's, not the build's
    texts = [json.dumps([m for m in c.body.get("messages") or [] if not (isinstance(m, dict) and m.get("role") ==
                         "assistant")], ensure_ascii=False) + json.dumps(c.body.get("tools") or [])
             for c in lanes.calls]
    if actor_model is not None:
        for call in getattr(actor_model, "calls", []) or []:
            texts.append(json.dumps(getattr(call, "messages", None), ensure_ascii=False, default=str))
    return texts


def check_coverage(home: Path) -> Any:
    report = ledger(home).report()
    assert report.guaranteed, report.lines()
    for kind, k in report.kinds.items():
        assert k.created == k.closed + k.unreadable + k.not_reached, (kind, k)
    return report


def check_metering(home: Path, lanes: FL.Lanes) -> dict:
    """A3: entries' shares + handed-not-read zeros == the passes' known USD (Decimal, exact); items <= passes."""
    from unify.memory import metering as m

    passes = m.metered_passes(home)
    entries = m.entries_view(passes)
    items = m.items_view(home, passes, entries)
    known = sum((Decimal(str((p.get("usd") or {}).get("known") or "0")) for p in passes), Decimal(0))
    shares = sum((Decimal(str(e.get("share_usd") or "0")) for e in entries), Decimal(0))
    assert shares == known, (shares, known)
    # an entry handed and not read gets 0; a pass that read nothing books its USD on one row without an entry, the
    # waste (A3 rule 4: counted apart)
    assert all(Decimal(str(e["share_usd"])) == 0 for e in entries
               if e.get("basis") == "handed_not_read" and e.get("entry") is not None), entries
    assert all(e.get("basis") == "handed_not_read" for e in entries if e.get("entry") is None), entries
    from_entries = sum((Decimal(str(i.get("from_entries_usd") or "0")) for i in items), Decimal(0))
    assert from_entries <= known, (from_entries, known)
    booked = sum((Decimal(str((p.get("usd") or {}).get("known") or "0")) for p in passes), Decimal(0))
    charged = lanes.booked_usd("writer") + lanes.booked_usd("helper")
    assert booked <= charged, (booked, charged)
    return {"passes": len(passes), "known": str(known), "charged_writer_helper": str(charged)}


def check_terminated(report: dict, home: Path) -> None:
    from unify.memory import writer_service as ws

    assert report["_exit"] == 0 and report.get("terminated") is not False, report
    service = report.get("service")
    assert service is None or service.get("verified") is True, report
    doc = ws.read_state(home) or {}
    pgid = doc.get("pgid")
    if isinstance(pgid, int):
        assert not ws.group_pids(pgid, since=doc.get("proc_start")), pgid


def check_run(home: Path, lanes: FL.Lanes, *, actor_model: Any = None, own_texts: Iterable[str] = ()) -> dict:
    """Every surface's end: drain (verified), then the checks of the module text. Returns the facts the surface
    test asserts more about."""
    report = drain(home)
    check_terminated(report, home)
    # the memory agents' inputs (fork, writer, helpers): the actor's own harness texts are outside the guard's scope, and
    # what memory adds to the actor's prompt is in the guard's test (tests/memory/test_no_world_vocabulary.py)
    words = world_words(model_inputs(lanes), own=own_texts)
    assert not words, words[:10]
    coverage = check_coverage(home)
    metering = check_metering(home, lanes)
    assert not lanes.unexpected, lanes.unexpected
    return {"drain": report, "coverage": coverage, "metering": metering}


def run_facts(home: Path, lanes: FL.Lanes, facts: dict) -> dict:
    """What the run did, as ids, kinds and counts (for the surface's own asserts and its log line)."""
    from unify.memory import writer_service as ws

    entries = ledger(home).entries().values()
    return {
        "lane_calls": {lane: len(lanes.of(lane)) for lane in sorted({c.lane for c in lanes.calls})},
        "embed_calls": lanes.embed_calls,
        "entries": sorted((e.kind, e.state, getattr(e, "decision", None) or getattr(e, "why", None)) for e in entries),
        "service_events": [r.get("event") for r in ws.events(home)],
        "drain": {k: facts["drain"].get(k) for k in ("terminated", "ended", "service")},
        "metering": facts["metering"],
        "passes": [{k: r.get(k) for k in ("ended", "items", "entries", "reason_codes", "gate_passed")}
                   for r in _jsonl(Path(home) / "memory-state" / "pass-results.jsonl")],
        "pass_ends": [{k: v for k, v in e.items() if k not in ("type", "phase")}
                      for e in _jsonl(Path(home) / "memory-state" / "events.jsonl")
                      if e.get("type") == "consolidation" and e.get("phase") == "end"],
    }


def _jsonl(path: Path) -> list[dict]:
    try:
        return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
    except OSError:
        return []

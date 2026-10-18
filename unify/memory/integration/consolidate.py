"""The consolidation driver (integration Task 24, v1): checker signal, size trigger, writer pass, gate.

Under ``UNIFY_MEMORY_LIBRARY=on``, after a WRITE pass, a CURATE pass follows when the library's state warrants it
(:mod:`..curate`): overlap candidates, items retired by their outcomes or an index over its view, each shown to
CURATE once.

After each request's episode is committed and indexed, :func:`run_due_passes` asks the batched size
trigger (:class:`..trigger.Trigger`) whether the experience recorded since the last pass has
reached E tokens. A due pass covers every channel with new evidence (there is no channel filter: the
gate's G2 dispatches covers per action kind, so tool, shell, work-tree and dialogue channels all take
part). The pass runs the writer (:class:`..writer_pass.WriterPass`) behind the gate, blocking the next request.

Bounds per pass: ``PassConfig(model, effort, max_calls=M[effort], deadline_s=900, max_usd=E x a_tok x S[effort])``,
with S and M from ``UNIFY_MEMORY_WRITER_EFFORT_SCALE`` (``low:1,medium:2,high:5``) and
``UNIFY_MEMORY_WRITER_MAX_CALLS`` (``low:40,medium:80,high:80``): one rule for every bed, by the pass's writer effort
(an effort with no entry starts no pass and is logged; the request stays due).
The writer's reasoning effort is the actor's for the run unless ``UNIFY_MEMORY_WRITER_EFFORT`` fixes another (a declared
mismatch ablation), passed in by the caller.
The run guard (``UNIFY_MEMORY_WRITER_RUN_GUARD_USD``, empty for none) starts no further pass once the writer
USD this home has committed plus the next pass's cap would exceed it; the request then stays due. The
commitment is kept in a ledger in the state dir (:func:`committed_writer_usd`): a pass reserves its whole cap
before it starts and settles at its end to its known USD plus a per-call reserve for each unpriced call,
so a pass that never ends (a killed process) keeps its whole cap committed. No reason text is read. The
last call of a pass can overshoot the pass's cap (the cap is checked before each call), and with it the
guard, by at most one call.

Each pass sends two events through ``emit`` and appends them to the harness-only state dir's
``events.jsonl`` (:func:`events_path`):

* ``{"type": "consolidation", "phase": "start", "pass_id", "trigger_tokens", "episodes", "writer_model",
  "writer_effort", "cap_usd", "max_calls", "effort_scale"}`` (the ledger's reserve line also keeps the last two);
* ``{"type": "consolidation", "phase": "end", "pass_id", "usd", "unknown_cost_calls", "calls", "checks",
  "seconds", "gate_passed", "items", "index_tokens", "reason_codes", "items_merged", "items_refused"}``
  (``calls`` counts model calls and the writer's ``check`` calls; ``checks`` is the latter alone;
  ``items_merged`` lists the manifest items that landed and ``items_refused`` maps each refused item to its
  value-free codes, ``G1``..``G6``, ``dependency`` or ``pass``: per-item admission, :meth:`..gate.Gate.merge`).

``reason_codes`` are codes only, never free text (deduplicated, at most 10): ``G1``..``G6`` for the gate
checks that refused, ``no_manifest``, ``manifest_invalid``, ``over_quota``, ``deadline``, ``pass_cap``,
``run_guard``, ``writer_error``, ``route_not_in_effect`` (a writer call found the writer's declared route not in effect),
``no_progress`` (the no-progress rule ended the writer's or curator's round; a gate check that then refused the
final merge is listed beside it, and only those checks are in ``refused_by``), and ``ok`` for a passed pass. They are structural: :attr:`PassOutcome.codes`
(set where each cause arises in the pass and the gate), the type of an exception that ended a pass, or the
run guard; reason text is never read. A failed pass with no listed cause (memory ``main`` moved during the
merge) is ``writer_error``. A pass the run guard holds back sends one end event (``calls`` 0, ``usd`` "0",
``reason_codes`` ``["run_guard"]``) and no start event, since no pass started. The full reasons stay
harness-side in the evidence store's ``passes`` row.

When the writer's route is set but refused (:class:`.switch.WriterRouteRefused`: malformed, half set, not what the
settings hold, or ``UNILLM_OTEL`` on), no pass ever starts, so each request instead sends one value-free
event, ``{"type": "consolidation", "phase": "refused", "episode_id", "consolidation_refused":
"route_not_in_effect", "reason_codes": ["route_not_in_effect"]}``, before the refusal is raised (and recorded
in ``errors.jsonl``): a run with any such event consolidated nothing because of its route, which is not a
null result of consolidation.

The writer's transcript (its messages, tool calls and tool results, redacted by the environment's credentials
and every registered secret, the writer's route token among them, then by :func:`..redact.redact_error`:
:func:`..writer_pass.transcript_lines`) is kept whole, never cut: as chunk files under the pass record root
(:mod:`..transcripts`), and on the :data:`..writer_pass.TRANSCRIPT_REF` notes ref of the same episode commit as one
JSON line per message with its ``pass_id`` while they fit, else as one row naming the chunks and their sha256.

Money is a plain decimal string, never an exponent; a measurement that could not be taken is ``None``.
The writer's per-turn cost rows go as note lines on ``refs/notes/costs`` of the request's episode commit: a pass
can only start after that commit (the trigger and the export read it), and one episode is one append-only
commit. :func:`episode_costs` merges the commit's ``cost.jsonl`` with those notes.

The checker contributes ``pass``/``fail`` only (:func:`post_checker`); the writer sees the episodes
through :func:`..writer_pass.export_for_writer`, which carries no signal.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import tempfile
import time
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

from .. import curate as _curate
from .. import library_checks
from ..blobs import BlobStore
from ..episodes import Action, CostRow, Episode, episode_dir, load_episode
from ..evidence import EvidenceStore
from ..gate import CHECKS as GATE_CHECKS, Gate
from ..gitio import Repo
from ..index import build_index, estimate_tokens
from ..memory_repo import items as memory_items
from ..qa import QAConfig
from ..reconcile import unavailable as unavailable_reconcile
from ..redact import Redactor, redact_error
from ..signals import Signal, SignalMasked, post_signal
from .. import similar as _similar
from ..snapshot import listing, materialise
from ..writer_pass import (
    CODE_NO_PROGRESS,
    CODE_ROUTE_NOT_IN_EFFECT,
    CODE_STEP_GUARD,
    STEP_GUARD,
    OTEL_REFUSAL,
    PassConfig,
    PassOutcome,
    WriterPass,
    WriterRoute,
    TRANSCRIPT_REF,
    otel_on,
    unillm_turn,
)
from ..trigger import EXPERIENCE_BUDGET, USD_PER_TOKEN, PassRequest, Trigger
from .cost import UNKNOWN, money, recording_turn
from .paths import Paths
from .switch import (
    WRITER_BASE_URL,
    WRITER_EFFORT_SCALE,
    WRITER_MAX_CALLS,
    WriterRouteRefused,
    settle_writer_route_env,
    writer_effort_scale_map,
    writer_max_calls_map,
    writer_prices,
    writer_route,
    writer_token,
    checker_visible,
    surfacing_options,
    library_enabled,
    library_pool,
    library_edit_tools,
    library_compact,
    library_fork,
    notebook_enabled,
    library_coverage,
    library_max_reads,
)

__all__ = [
    "DEADLINE_S",
    "MAX_CALLS",
    "WRITER_MODEL",
    "EpisodeLookup",
    "WriterSettings",
    "Stores",
    "committed_writer_usd",
    "episode_costs",
    "events_path",
    "open_stores",
    "post_checker",
    "post_visible_checkers",
    "run_due_passes",
    "writer_settings",
]

WRITER_MODEL = "openai/gpt-6-sol"
MAX_CALLS = 40  # the low-effort default of UNIFY_MEMORY_WRITER_MAX_CALLS
DEADLINE_S = 900.0
# A pass ends itself at its deadline; this outer bound only catches a pass that overruns it (a cell or a
# model call in flight at the deadline), after which the pass is cancelled and recorded failed.
OVERRUN_S = 180.0
COSTS_REF = "costs"
MAX_REASON_CODES = 10
REASON_CODES = frozenset(
    {f"G{n}" for n in range(1, 7)}
    | {
        "no_manifest",
        "manifest_invalid",
        "over_quota",
        "deadline",
        "pass_cap",
        CODE_STEP_GUARD,
        "run_guard",
        "writer_error",
        CODE_ROUTE_NOT_IN_EFFECT,
        CODE_NO_PROGRESS,
        "ok",
    },
)
_EPISODE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_PLAIN_DECIMAL = re.compile(r"^[0-9]+(\.[0-9]+)?\Z")
_NO_INDEX_BUDGET = 10**9


# --- stores ----------------------------------------------------------------------------------------------


@dataclass
class Stores:
    paths: Paths
    memory: Repo
    episodes: Repo
    blobs: BlobStore
    evidence: EvidenceStore


def _bare(path: Path) -> Repo:
    return Repo(path) if (path / "HEAD").is_file() else Repo.init_bare(path)


def open_stores(paths: Paths, *, busy_timeout_s: float | None = None) -> Stores:
    """The harness-side stores under ``paths.home``; the two bare repos are created on first use. *busy_timeout_s*
    (the library mode's pass worker) is the evidence store's SQLite busy timeout; None keeps v2's.
    """
    paths.home.mkdir(parents=True, exist_ok=True)
    return Stores(
        paths,
        _bare(paths.memory),
        _bare(paths.episodes),
        BlobStore(paths.blobs),
        EvidenceStore(paths.evidence, timeout_s=busy_timeout_s),
    )


def events_path(paths: Paths) -> Path:
    """``paths.events`` (the harness-only state dir's ``events.jsonl``); the same file without that property."""
    events = getattr(paths, "events", None)
    return Path(events) if events is not None else paths.state_dir / "events.jsonl"


def ledger_path(paths: Paths) -> Path:
    """The run guard's ledger: one ``reserve`` line before each pass starts, one ``settle`` line after it."""
    return paths.state_dir / "writer-spend.jsonl"


class EpisodeLookup:
    """Recorded episodes and their actions, read from the episodes repo through the evidence index."""

    def __init__(self, stores: Stores, cache_size: int = 64) -> None:
        self.stores = stores
        self._cache: OrderedDict[str, Episode] = OrderedDict()
        self._size = cache_size

    def episode(self, eid: str) -> Episode:
        """The episode *eid* (``KeyError`` when it was never indexed)."""
        hit = self._cache.get(eid)
        if hit is not None:
            self._cache.move_to_end(eid)
            return hit
        sha, started_at = self.stores.evidence.episode_ref(eid)
        rel = episode_dir(SimpleNamespace(episode_id=eid, started_at=started_at))
        ep = load_episode(self.stores.episodes, sha, rel, self.stores.blobs)
        self._cache[eid] = ep
        if len(self._cache) > self._size:
            self._cache.popitem(last=False)
        return ep

    def action(self, eid: Any, idx: Any) -> Action | None:
        """Recorded action *idx* of episode *eid*, of any kind (G2 dispatches per kind); else None.

        The ids come from a model-written manifest, so nothing here raises: an unsafe or unknown
        episode id, a non-integer or out-of-range index, or an unreadable episode give None.
        """
        if not isinstance(eid, str) or not _EPISODE_ID.match(eid):
            return None
        if isinstance(idx, bool) or not isinstance(idx, int) or idx < 0:
            return None
        try:
            actions = self.episode(eid).actions
        except Exception:  # noqa: BLE001 - unknown or unreadable: not a recorded action
            return None
        return deepcopy(actions[idx]) if idx < len(actions) else None


def episode_costs(stores: Stores, eid: str) -> list[CostRow]:
    """The episode's ``cost.jsonl`` rows, then the late rows noted on its commit (the writer's)."""
    sha, _ = stores.evidence.episode_ref(eid)
    rows = list(EpisodeLookup(stores).episode(eid).costs)
    fields = ("purpose", "model", "prompt_tokens", "completion_tokens", "usd")
    for line in stores.episodes.notes(sha, ref=COSTS_REF):
        try:
            row = json.loads(line)
            rows.append(CostRow(**{k: row[k] for k in fields}))
        except (ValueError, KeyError, TypeError):
            continue
    return rows


# --- the checker -----------------------------------------------------------------------------------------


def post_checker(
    stores: Stores,
    eid: str,
    sha: str,
    solved: bool | None,
    ts: str,
) -> bool:
    """Post the checker's verdict on episode *eid* as ``pass`` or ``fail``; nothing else of it is kept.

    ``None`` (no checker, or the outcome left it open) posts nothing. A regime in which checker signals
    are not observable posts nothing. Returns whether a signal was posted.
    """
    if not isinstance(solved, bool):
        return False
    sig = Signal(
        f"{eid}.checker",
        eid,
        "checker",
        "pass" if solved else "fail",
        ts,
        refers_to=eid,
        regime=stores.evidence.regime_of(eid),
    )
    try:
        post_signal(sig, stores.episodes, sha, stores.evidence)
    except SignalMasked:
        return False
    return True


def post_visible_checkers(
    stores: Stores,
    ep: Episode,
    sha: str,
    entries: list[dict],
    lines: list[dict],
    counterpart: str,
) -> int:
    """Post the verdicts the bed showed the actor on episode *ep* as agent-visible checker
    signals, one per entry, in order: ``<eid>.checker.<n>``, ``pass`` or ``fail``.

    Each entry is ``{"label", "obs", "at"}`` (:meth:`.request.RequestRun.take_checker`): ``obs`` is how many
    observations the transcript held when the line came, so the verdict's own message is observation number
    ``obs``. The signal refers to the dialogue action that message answers (``<eid>/actions/<index>``), found
    with the dialogue adapter's own pairing (:func:`.adapters.dialogue.answered_observations`). Without the
    dialogue setting, without an answered action, or when the episode's dialogue actions and the pairing
    disagree in number, or when the entries' counts do not strictly increase, it refers to the episode as a
    whole, never to a guessed action. An entry whose verdict message never reached the actor (its
    count unreadable, or at or past the transcript's final observation count) is dropped and logged.
    A regime in which checker signals are not observable posts nothing. Returns how many were posted.
    """
    from .adapters import dialogue

    eid = ep.episode_id
    # A verdict counts only if its message reached the actor in this request: its observation number
    # is below the transcript's final observation count. A request that ended first (cancel, quit, deadline), or
    # an entry whose count could not be read, drops it, logged and never posted.
    total = dialogue.observation_count(lines)
    delivered: list[tuple[int, dict]] = []
    for n, entry in enumerate(entries, 1):
        obs = entry.get("obs")
        if isinstance(obs, int) and not isinstance(obs, bool) and 0 <= obs < total:
            delivered.append((n, entry))
        else:
            why = (
                "count unreadable"
                if not isinstance(obs, int)
                else f"observation {obs} of {total} never shown"
            )
            _error(
                stores,
                f"{eid}: checker line {n} dropped: the actor never saw its verdict ({why})",
            )
    answered: list[int] = []
    positions: list[int] = []
    if counterpart:
        answered = dialogue.answered_observations(lines)
        positions = [i for i, a in enumerate(ep.actions) if a.kind == "dialogue"]
        obs_seen = [e["obs"] for _, e in delivered]
        increasing = all(a < b for a, b in zip(obs_seen, obs_seen[1:]))
        if len(answered) != len(positions) or not increasing:
            # the pairing is not certain for every entry: no entry names an action
            answered, positions = [], []
    posted = 0
    for n, entry in delivered:
        refers_to = eid
        obs = entry["obs"]
        if obs in answered:
            refers_to = f"{eid}/actions/{positions[answered.index(obs)]}"
        sig = Signal(
            f"{eid}.checker.{n}",
            eid,
            "checker",
            entry["label"],
            entry["at"],
            refers_to=refers_to,
            regime=stores.evidence.regime_of(eid),
            visible_to_actor=True,
        )
        try:
            post_signal(sig, stores.episodes, sha, stores.evidence)
        except SignalMasked:
            continue
        posted += 1
    return posted


# --- settings --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class WriterSettings:
    model: str
    experience_budget: int
    usd_per_token: Decimal
    run_guard_usd: Decimal | None
    # The writer's own route (UNIFY_MEMORY_WRITER_BASE_URL / _WRITER_TOKEN or _WRITER_TOKEN_FD); None: as shipped
    route: WriterRoute | None = None
    show_usage: bool = (
        False  # UNIFY_MEMORY_WRITER_USAGE: the use table in each pass's first message
    )
    # UNIFY_MEMORY_WRITER_EFFORT_SCALE / _WRITER_MAX_CALLS, by writer effort (low, medium, high)
    effort_scale: dict[str, Decimal] = field(
        default_factory=lambda: writer_effort_scale_map(""),
    )
    max_calls: dict[str, int] = field(default_factory=lambda: writer_max_calls_map(""))
    library_mode: bool = (
        False  # UNIFY_MEMORY_LIBRARY: the writer's batch map, views and coverage
    )
    # UNIFY_PROFILE (empty: today's behaviour); under ``paper`` the model is the actor's and there is no route
    profile: str = ""

    @property
    def cap_usd(self) -> Decimal:
        return Decimal(self.experience_budget) * self.usd_per_token


def _decimal(name: str, value: Any) -> Decimal:
    text = str(value).strip()
    if not _PLAIN_DECIMAL.match(text):
        raise ValueError(
            f"{name} must be a plain decimal string (no exponent), not {value!r}"[:200],
        )
    try:
        return Decimal(text)
    except (
        InvalidOperation
    ) as exc:  # pragma: no cover - the pattern admits only valid decimals
        raise ValueError(f"{name} is not a decimal: {value!r}"[:200]) from exc


def writer_settings(settings: Any) -> WriterSettings:
    """The writer model, E, the USD allowance per token, the run guard, the writer's route and whether the writer is shown
    the use table (``UNIFY_MEMORY_WRITER_USAGE``) from *settings*.

    Defaults if unset. :func:`run_due_passes` reads them before anything else, so a refused value (the writer's
    route with one of its two settings empty, say) starts no pass and makes no call. The writer's route is settled
    against the process environment again first (:func:`.switch.settle_writer_route_env`), so a value that
    reached the environment late (``.env`` loaded by an embedder after unify was imported) is refused here
    too, not only on the CLI's path. Every refusal of the route is a :class:`.switch.WriterRouteRefused`. Under
    ``UNIFY_PROFILE=paper`` the model is the actor's (:func:`.request.actor_model`), and a run that sets a setting the
    profile fixes is refused before anything else is read (:func:`.switch.profile_refusal`).
    """
    from .switch import (
        library_enabled,
        library_experience_budget,
        library_writer_usd_per_token,
        profile,
        profile_refusal,
    )

    # a run that sets a setting its profile fixes starts no pass and makes no call (the settings named, never a
    # value); checked first, so no route setting is read for it
    refused = profile_refusal(settings)
    if refused is not None:
        raise WriterRouteRefused(refused)
    settle_writer_route_env(os.environ, settings)
    model = str(getattr(settings, "UNIFY_MEMORY_WRITER_MODEL", "") or "").strip()
    run_profile = profile(settings)
    if run_profile:
        # the paper profile: the writer runs on the actor's model, through its own accounting lane (checked below)
        from .request import actor_model

        model = actor_model()

    if library_enabled(settings):
        e = library_experience_budget(
            settings,
        )  # E = 100k recorded tokens in the library mode
    else:
        raw_e = getattr(settings, "UNIFY_MEMORY_E", "") or EXPERIENCE_BUDGET
        if isinstance(raw_e, bool):
            raise ValueError(
                f"UNIFY_MEMORY_E must be a positive integer, not {raw_e!r}",
            )
        try:
            e = int(str(raw_e).strip())
        except ValueError:
            e = 0
        if e <= 0:
            raise ValueError(
                f"UNIFY_MEMORY_E must be a positive integer, not {raw_e!r}"[:200],
            )
    if library_enabled(settings):
        # the library-mode writer's rate is its own setting, so the offline replay can size it
        a_tok = library_writer_usd_per_token(settings)
    else:
        a_tok = _decimal(
            "UNIFY_MEMORY_WRITER_ALLOWANCE_USD_PER_TOKENS",
            getattr(settings, "UNIFY_MEMORY_WRITER_ALLOWANCE_USD_PER_TOKENS", "")
            or format(USD_PER_TOKEN, "f"),
        )
    if (
        a_tok <= 0
    ):  # a zero cap would consume experience with passes that can make no call
        raise ValueError(
            "UNIFY_MEMORY_WRITER_ALLOWANCE_USD_PER_TOKENS must be greater than zero",
        )
    guard_raw = getattr(settings, "UNIFY_MEMORY_WRITER_RUN_GUARD_USD", "") or ""
    guard = (
        _decimal("UNIFY_MEMORY_WRITER_RUN_GUARD_USD", guard_raw)
        if str(guard_raw).strip()
        else None
    )
    # both or neither, checked here (before any pass starts); no error quotes a value
    pair = writer_route(getattr(settings, WRITER_BASE_URL, ""), writer_token(settings))
    route = WriterRoute(*pair) if pair is not None else None
    if run_profile and route is None:
        from .switch import PAPER_LANE_REFUSAL

        raise WriterRouteRefused(PAPER_LANE_REFUSAL)
    model = model or WRITER_MODEL
    if route is not None and otel_on():
        raise WriterRouteRefused(OTEL_REFUSAL)
    if route is not None and not (
        model if "@" in model else f"{model}@openrouter"
    ).endswith("@openrouter"):
        raise WriterRouteRefused(
            f"{WRITER_BASE_URL} replaces the OpenRouter transport; UNIFY_MEMORY_WRITER_MODEL must be an "
            "@openrouter endpoint (or a bare model id)",
        )
    from .switch import LIBRARY_MODE, parse_memory_library_mode, parse_writer_usage

    usage = parse_writer_usage(
        getattr(settings, "UNIFY_MEMORY_WRITER_USAGE", "") or "",
    )
    library_mode = (
        parse_memory_library_mode(getattr(settings, LIBRARY_MODE, "") or "") == "on"
    )
    scale = writer_effort_scale_map(getattr(settings, WRITER_EFFORT_SCALE, "") or "")
    calls = writer_max_calls_map(getattr(settings, WRITER_MAX_CALLS, "") or "")
    return WriterSettings(
        model,
        e,
        a_tok,
        guard,
        route,
        usage == "on",
        scale,
        calls,
        library_mode,
        run_profile,
    )


# --- money -----------------------------------------------------------------------------------------------


def _usd(value: Decimal) -> str:
    return format(abs(value) if value.is_zero() else value, "f")


def _ledger(stores: Stores, row: dict) -> None:
    path = ledger_path(stores.paths)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")
        fh.flush()


def _reserve(
    stores: Stores,
    pass_id: str,
    cap: Decimal,
    per_call: Decimal,
    max_calls: int,
    scale: Decimal,
) -> None:
    _ledger(
        stores,
        {
            "pass_id": pass_id,
            "phase": "reserve",
            "cap_usd": _usd(cap),
            "per_call_usd": _usd(per_call),
            "max_calls": int(max_calls),
            "effort_scale": _usd(scale),
        },
    )


def _settle(
    stores: Stores,
    pass_id: str,
    usd: str,
    unknown_cost_calls: int,
    unknown_usd_each: Decimal | None = None,
) -> None:
    row = {
        "pass_id": pass_id,
        "phase": "settle",
        "usd": usd,
        "unknown_cost_calls": int(unknown_cost_calls),
    }
    if unknown_usd_each is not None:  # library mode: unpriced calls at the worst case
        row["unknown_usd_each"] = format(unknown_usd_each, "f")
    _ledger(stores, row)


def committed_writer_usd(stores: Stores) -> Decimal:
    """The writer USD committed in this home, from the ledger's structured fields only.

    A pass with a ``settle`` line counts its known USD plus ``unknown_cost_calls`` times the per-call
    reserve it held, or its line's ``unknown_usd_each`` (library mode: the worst case of one writer call) when it
    has one (an unpriced call's cost is unknown, never zero). A pass with only its ``reserve``
    line (it started and never settled) counts its whole cap. An unreadable line is skipped.
    """
    total = Decimal(0)
    open_: dict[str, tuple[Decimal, Decimal]] = (
        {}
    )  # reservations not yet settled, per pass id
    try:
        text = ledger_path(stores.paths).read_text(encoding="utf-8")
    except FileNotFoundError:
        return total
    for line in text.splitlines():
        try:
            row = json.loads(line)
            pid, phase = str(row["pass_id"]), row["phase"]
            if phase == "reserve":
                held = (
                    _decimal("cap", row["cap_usd"]),
                    _decimal("per call", row["per_call_usd"]),
                )
                if pid in open_:  # an earlier start of this id never settled
                    total += open_[pid][0]
                open_[pid] = held
            elif phase == "settle":
                known, n = money(row["usd"]), row["unknown_cost_calls"]
                if (
                    known == UNKNOWN
                    or isinstance(n, bool)
                    or not isinstance(n, int)
                    or n < 0
                ):
                    continue  # unreadable: the reservation (if any) stays at its whole cap
                per_call = open_.pop(pid, (Decimal(0), Decimal(0)))[1]
                if "unknown_usd_each" in row:
                    per_call = _decimal("unknown each", row["unknown_usd_each"])
                total += Decimal(known) + n * per_call
        except (ValueError, KeyError, TypeError):
            continue
    return total + sum((cap for cap, _ in open_.values()), Decimal(0))


# --- events and notes ------------------------------------------------------------------------------------


def _error(stores: Stores, text: str) -> None:
    try:
        stores.paths.errors.parent.mkdir(parents=True, exist_ok=True)
        with open(stores.paths.errors, "a", encoding="utf-8") as fh:
            fh.write(
                json.dumps({"where": "consolidate", "error": redact_error(text)[:500]})
                + "\n",
            )
    except OSError:
        pass


#: What a running pass reports between its start and its end: each writer model call (``turn``) and each gate verdict
#: (``gate``: a check, or the merge that lands), each with ``seconds`` since the pass started. Real events only, as
#: they happen, so a host that watches the session's output for work sees the pass working (the waiting CLI relays
#: them with the slot's start and end: :data:`.async_pass._RELAYED`).
PROGRESS_PHASES = ("turn", "gate")


def _reporting_turn(turn: Any, report: Callable[..., None]) -> Any:
    """*turn*, reporting each call as it returns or fails (its number, whether it returned); the return value and
    the turn's attributes pass through unchanged."""
    count = [0]

    async def reported(messages: list[dict], tools: list[dict]) -> Any:
        count[0] += 1
        n = count[0]
        try:
            out = await turn(messages, tools)
        except BaseException:
            report("turn", turn=n, returned=False)
            raise
        report("turn", turn=n, returned=True)
        return out

    reported.__dict__.update(getattr(turn, "__dict__", None) or {})
    return reported


class _ReportingGate:
    """A pass's gate, reporting each verdict as it is given: a check (no merge) and the merge, whether it passed,
    the refusing checks by name and, for the merge, whether ``main`` moved. Everything else is the gate's own."""

    def __init__(self, gate: Any, report: Callable[..., None]) -> None:
        self._gate, self._report = gate, report

    def check(self, *args: Any, **kwargs: Any) -> Any:
        res = self._gate.check(*args, **kwargs)
        self._report("gate", step="check", passed=bool(res.passed), refused=list(getattr(res, "refused", []) or []))
        return res

    def merge(self, *args: Any, **kwargs: Any) -> Any:
        res = self._gate.merge(*args, **kwargs)
        self._report(
            "gate",
            step="merge",
            passed=bool(res.passed),
            refused=list(getattr(res, "refused", []) or []),
            landed=getattr(res, "merged", None) is not None,
        )
        return res

    def __getattr__(self, name: str) -> Any:
        return getattr(self._gate, name)


def _result(outcome: PassOutcome | None) -> str:
    """How the pass ended, for its end event: ``landed`` (items reached ``main``), ``nothing_to_store`` (the writer
    stored nothing, by its own word) or ``abandoned`` (refused, failed or cut)."""
    if outcome is not None and outcome.passed and list(outcome.items_merged or []):
        return "landed"
    if outcome is not None and getattr(outcome, "nothing_to_store", False):
        return "nothing_to_store"
    return "abandoned"


def _deliver(stores: Stores, emit: Callable[[dict], None] | None, row: dict) -> None:
    line = json.dumps(row, sort_keys=True)
    try:
        path = events_path(stores.paths)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except OSError as exc:
        _error(stores, f"events file: {type(exc).__name__}: {exc}")
    if emit is not None:
        try:
            emit(dict(row))
        except Exception as exc:  # noqa: BLE001 - reporting never stops a pass
            _error(stores, f"emit: {type(exc).__name__}: {exc}")


def _note_costs(
    stores: Stores,
    sha: str,
    eid: str,
    pass_id: str,
    rows: list[CostRow],
    effort: str,
) -> None:
    for r in rows:
        note = {
            "purpose": r.purpose,
            "model": r.model,
            "prompt_tokens": r.prompt_tokens,
            "completion_tokens": r.completion_tokens,
            "usd": r.usd,
            "episode": eid,
            "pass_id": pass_id,
            "writer_effort": effort,
        }
        try:
            stores.episodes.add_note(
                sha,
                json.dumps(note, sort_keys=True),
                ref=COSTS_REF,
            )
        except Exception as exc:  # noqa: BLE001
            _error(stores, f"cost note: {type(exc).__name__}: {exc}")


def _note_transcript(
    stores: Stores,
    sha: str,
    pass_id: str,
    writer: WriterPass,
) -> None:
    """The writer's redacted, whole transcript on :data:`..writer_pass.TRANSCRIPT_REF` of *sha*: its lines, or a
    row pointing to its chunk files under the pass record root (:meth:`..writer_pass.WriterPass.transcript_note`).
    """
    try:
        stores.episodes.append_note_lines(
            sha,
            writer.transcript_note(pass_id),
            TRANSCRIPT_REF,
        )
    except Exception as exc:  # noqa: BLE001 - a record, never a failure of the pass
        _error(stores, f"{pass_id}: transcript: {type(exc).__name__}")


def _count_items(tree: Path) -> int:
    """The items the library at *tree* holds: its functions and notes in the library layout, else the earlier
    layout's listed items."""
    from ..layout import discover

    lib = discover(tree)
    if lib.functions or lib.notes or (tree / "memory").is_dir():
        return len(lib.functions) + len(lib.notes)
    return sum(
        1 for it in memory_items(tree).items if it.kind != "workflow" and it.listed
    )


def library_items_at(stores: Stores, commit: str | None) -> int | None:
    """The number of items the library holds at memory commit *commit*; None when it cannot be read."""
    if not commit:
        return None
    tmp = Path(tempfile.mkdtemp(prefix="memory-landed-"))
    try:
        files, _ = listing(stores.memory, commit)
        return _count_items(materialise(stores.memory, files, tmp / "tree"))
    except Exception:  # noqa: BLE001 - a measurement, never a failure
        return None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _library_after(stores: Stores) -> tuple[int | None, int | None]:
    """(items, index tokens) of memory ``main`` when the pass's end is recorded; None for what could not be
    measured. In the library mode that is before the pass's commit lands, and the landed event counts the library at
    the landed commit; the earlier layout's gate has merged by then.
    """
    tmp = Path(tempfile.mkdtemp(prefix="memory-after-"))
    try:
        files, _ = listing(stores.memory, stores.memory.head())
        tree = materialise(stores.memory, files, tmp / "main")
        listed = _count_items(tree)
        try:
            index_tokens: int | None = estimate_tokens(
                build_index(tree, budget_tokens=_NO_INDEX_BUDGET),
            )
        except ValueError:
            index_tokens = None
        return listed, index_tokens
    except Exception:  # noqa: BLE001 - a measurement, never a failure of the pass
        return None, None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def reason_codes(outcome: PassOutcome | None, failure_code: str | None) -> list[str]:
    """The end event's codes: the outcome's structured codes (known ones, deduplicated, at most 10)."""
    raw = (
        list(outcome.codes) if outcome is not None else [failure_code or "writer_error"]
    )
    out: list[str] = []
    for code in raw:
        if code in REASON_CODES and code not in out:
            out.append(code)
    if not out or (outcome is not None and not outcome.passed and out == ["ok"]):
        out = ["ok"] if outcome is not None and outcome.passed else ["writer_error"]
    return out[:MAX_REASON_CODES]


#: The call kind (origin) of helper model calls, as the key proxy and the cost records see it.
from .cost import HELPER_ORIGIN  # noqa: E402


def _helper_turns(settings: Any, effort: str, library_mode: bool) -> dict:
    """``WriterPass`` keywords for writer-launched helpers: a factory of fresh host-side turns on the key proxy's
    helper lane (its own token and ledger; the helper model at the pass's effort), or ``helper_turn=None`` with why.

    Fail closed: with no helper lane configured, or a refused one, ``delegate`` is not offered, and there is never a
    fallback to the writer's or the actor's route. The token stays in this process; helper boxes never see it.
    """
    if not library_mode:
        return {}
    from .switch import helper_lane_route, library_helper_model, settle_helper_route_env

    settle_helper_route_env(os.environ, settings)
    pair, why = helper_lane_route()
    model = library_helper_model(settings)
    if pair is None or not model:
        return {"helper_turn": None, "helper_refusal": why or "no helper model"}
    route = WriterRoute(*pair)
    endpoint = model if "@" in model else f"{model}@openrouter"

    def factory(session: str | None = None):
        return unillm_turn(
            endpoint,
            effort,
            origin=HELPER_ORIGIN,
            route=route,
            session=session,
        )

    return {"helper_turn": factory, "helper_refusal": None}


def _by_category(usd: str, helpers: list[dict]) -> dict:
    """The pass's known USD split by lane: the writer's own calls and its helpers' (one row per delegate call), never
    by model, since a helper's model may equal another lane's. Decimal strings; unknown stays unknown: the helpers'
    share is unknown when any helper call was unpriced."""
    try:
        spent = sum(
            (Decimal(str(r.get("usd_known", r.get("usd")) or 0)) for r in helpers),
            Decimal(0),
        )
        total = Decimal(usd)
    except (InvalidOperation, ValueError):
        return {"writer": UNKNOWN, "helpers": UNKNOWN}
    unpriced = any(int(r.get("unknown_cost_calls") or 0) for r in helpers)
    return {
        "writer": _usd(total - spent),
        "helpers": UNKNOWN if unpriced else _usd(spent),
    }


def _reader_category(rows: list[dict] | None) -> dict:
    """``{"reader": USD}``: the batch episodes' readers' own spend (each fork's ``fork.json`` ``usd``, which the proxy
    journals under ``fork.<episode>``, outside the pass's own spend), summed; unknown when any reader recorded none.
    Empty without staging."""
    if rows is None:
        return {}
    total = Decimal(0)
    for r in rows:
        usd = r.get("reader_usd")
        if usd is None:
            continue
        if usd == UNKNOWN:
            return {"reader": UNKNOWN}
        try:
            total += Decimal(str(usd))
        except (InvalidOperation, ValueError):
            return {"reader": UNKNOWN}
    return {"reader": _usd(total)}


def _spend(outcome: PassOutcome | None, rows: list[CostRow]) -> tuple[str, int, int]:
    """(known USD, unpriced calls, calls): the pass's own figures, else the cost rows'."""
    if outcome is not None and money(outcome.usd) != UNKNOWN:
        return money(outcome.usd), int(outcome.unknown_cost_calls), int(outcome.calls)
    known = sum((Decimal(r.usd) for r in rows if r.usd != UNKNOWN), Decimal(0))
    return _usd(known), sum(1 for r in rows if r.usd == UNKNOWN), len(rows)


def _start_event(
    req: PassRequest,
    pass_id: str,
    model: str,
    effort: str,
    cap: Decimal,
    max_calls: int,
    scale: Decimal,
) -> dict:
    return {
        "type": "consolidation",
        "phase": "start",
        "pass_id": pass_id,
        "trigger_tokens": req.experience_tokens,
        "episodes": list(req.episodes),
        "held": list(getattr(req, "held", []) or []),
        "writer_model": model,
        "writer_effort": effort,
        "cap_usd": _usd(cap),
        "max_calls": int(max_calls),
        "effort_scale": _usd(scale),
    }


def _end_event(
    stores: Stores,
    pass_id: str,
    outcome: PassOutcome | None,
    rows: list[CostRow],
    seconds: float,
    failure_code: str | None,
) -> dict:
    usd, unknown, calls = _spend(outcome, rows)
    listed, index_tokens = _library_after(stores)
    return {
        "type": "consolidation",
        "phase": "end",
        "pass_id": pass_id,
        "usd": usd,
        "unknown_cost_calls": unknown,
        "calls": calls,  # model calls plus check calls (WriterPass counts both against max_calls)
        "checks": int(outcome.checks) if outcome is not None else 0,
        "seconds": round(max(0.0, float(seconds)), 3),
        "gate_passed": bool(outcome is not None and outcome.passed),
        # main before this pass's commit lands; the landed event counts the library at the landed commit
        "library_items_before_landing": listed,
        "index_tokens": index_tokens,
        "reason_codes": reason_codes(outcome, failure_code),
        "items_merged": list(outcome.items_merged) if outcome is not None else [],
        "items_refused": (
            {i: list(c) for i, c in sorted(outcome.items_refused.items())}
            if outcome is not None
            else {}
        ),
        # library mode only (absent with UNIFY_MEMORY_LIBRARY off): the writer's coverage, its reader calls and the
        # blob bytes its /inputs held (uncapped in the library mode, so a disk-full stage leaves its size here)
        **(
            {
                "coverage": outcome.coverage,
                # UNIFY_MEMORY_LIBRARY_COVERAGE=optional only (absent otherwise): measures, read by no gate
                **(
                    {"coverage_measures": outcome.coverage_measures}
                    if getattr(outcome, "coverage_measures", None) is not None
                    else {}
                ),
                "reads": int(outcome.reads),
                "exported_bytes": outcome.exported_bytes,
                # the gate checks that refused the pass and the first reason line; the writer's last
                # finish accepted, and its parsed manifest listing nothing (the writer's own "nothing to store")
                "refused_by": sorted({c for c in outcome.codes if c in GATE_CHECKS}),
                "first_reason": next(
                    (
                        r.splitlines()[0][:300]
                        for r in outcome.reasons
                        if r and r != "ok"
                    ),
                    None,
                ),
                "finished": bool(outcome.finished),
                "nothing_to_store": bool(outcome.nothing_to_store),
                # spend by lane (the fork's is in the proxy journal, session fork.<episode>), one row per delegate
                # call, the batch's staging by source
                "usd_by_category": {
                    **_by_category(usd, outcome.helpers),
                    **_reader_category(getattr(outcome, "staging_episodes", None)),
                },
                "helpers": list(outcome.helpers),
                "staging": outcome.staging,
                # with staging on, per batch episode: its staging's source, status, files, candidates and the files
                # withheld from an unfinished reader, its summary.md's size, how many of the summary's pointers do
                # not resolve in the recorded episode, and whether the writer read the summary (a measure only)
                **(
                    {"staging_episodes": outcome.staging_episodes}
                    if getattr(outcome, "staging_episodes", None) is not None
                    else {}
                ),
                **({"pool": outcome.pool} if outcome.pool is not None else {}),
                # UNIFY_MEMORY_LIBRARY_COMPACT: the writer's compactions, the messages they took out and those
                # messages' prompt characters (all 0 with it off); its calls' prompt and cache-read tokens
                "compactions": int(getattr(outcome, "compactions", 0)),
                "chars_moved": int(getattr(outcome, "chars_moved", 0)),
                "messages_moved": int(getattr(outcome, "messages_moved", 0)),
                "input_tokens": getattr(outcome, "input_tokens", None),
                "cache_read_tokens": getattr(outcome, "cache_read_tokens", None),
                # how many of the writer's calls reported each sum, out of all its calls, and the first call's
                # prompt (the brief and the first message, before any reading)
                "input_tokens_calls": int(getattr(outcome, "input_tokens_calls", 0)),
                "cache_read_calls": int(getattr(outcome, "cache_read_calls", 0)),
                # the writer's output tokens (its reasoning included, as providers count it) and how many calls
                # reported them (None: no call did): the pass's own tokens beside its wall time (seconds)
                "output_tokens": getattr(outcome, "output_tokens", None),
                "output_tokens_calls": int(getattr(outcome, "output_tokens_calls", 0)),
                "writer_calls": int(getattr(outcome, "writer_calls", 0)),
                "writer_first_prompt_tokens": getattr(outcome, "writer_first_prompt_tokens", None),
                # the largest prompt one call reported, and the model's window as the compaction trigger reads it
                # (None: unknown, so no compaction)
                "max_prompt_tokens": getattr(outcome, "max_prompt_tokens", None),
                "context_window": getattr(outcome, "context_window", None),
                # an uncapped pass's refusals the gate repeated for the same reason in consecutive rounds, so no
                # further repair round was offered for them, and whether that ended repair (None: none)
                "repair_stuck": getattr(outcome, "repair_stuck", None),
                # the writer's (or curator's) no-progress notices and whether 2 * k no-progress model calls in a row
                # ended its round (loop_stop.AgentNoProgress.report; None: no notice was given)
                "no_progress": getattr(outcome, "no_progress", None),
                # the landed functions' evidence levels (counts per level) and the evidence findings that set them
                # (counts by reason code); None when the pass landed no function with a level
                "levels": getattr(outcome, "levels", None),
                "capped_reasons": getattr(outcome, "capped_reasons", None),
            }
            if outcome is not None and outcome.coverage is not None
            else {}
        ),
        # library layout, store passes: staged candidates at each hand-off, and each landed item's adopted proposals
        **(
            {"chain": outcome.chain, "provenance": list(outcome.provenance)}
            if outcome is not None and getattr(outcome, "chain", None) is not None
            else {}
        ),
    }


# --- the driver ------------------------------------------------------------------------------------------


def after_passes(
    stores: Stores,
    settings: Any,
    recorded: list[str],
    emit: Callable[[dict], None] | None,
    stop: Callable[[], bool] | None = None,
) -> None:
    """Library mode: once a consolidation's passes are recorded, the item records of
    ``main``'s head, the status changes and the bisects (:func:`..lifecycle.consolidate_records`). Nothing runs
    with ``UNIFY_MEMORY_LIBRARY`` off, or when no pass was recorded (the run guard held them all back). Never raises:
    a failure is logged, and the next consolidation computes everything again from the same evidence.
    """
    if not recorded or not library_enabled(settings):
        return
    from .. import lifecycle
    from ..captures import store_dir as captures_store_dir
    from .switch import checker_visible

    try:
        with tempfile.TemporaryDirectory(prefix="memlib-records-") as work:
            summary = lifecycle.consolidate_records(
                stores,
                checker_visible=checker_visible(settings),
                work=Path(work),
                action_lookup=EpisodeLookup(stores).action,
                stop=stop,  # the worker's SIGTERM, checked between bisect runs
                # the capture store: a landed procedure's record takes its capture's signature and applies
                captures=captures_store_dir(stores.paths.state_dir),
            )
    except Exception as exc:  # noqa: BLE001 - the records never stop the request
        _error(stores, redact_error(f"item records: {type(exc).__name__}: {exc}"))
        return
    _deliver(
        stores,
        emit,
        {
            "type": "consolidation",
            "phase": "records",
            "passes": list(recorded),
            "noted": summary["noted"],
            "status_changes": summary["changes"],
            "bisected": summary["bisected"],
        },
    )


# --- library-mode CURATE --------------------------------------------------------------------------------------


def _lifecycle(
    stores: Stores,
    sha: str,
) -> tuple[Callable[[str], str] | None, dict[str, dict], dict[str, dict]]:
    """The item records at library commit *sha*: each item's status, the items retired by their outcomes (with
    their reasons, episodes, bisect result and rollback target; not the curator's retirements) and the use
    records by item. The one place CURATE reads the
    item records (``item_records.records_at`` and ``status_of``; the record fields)."""
    from .. import item_records

    recs, _changed_at = item_records.records_at(stores.memory, sha)
    status_of = item_records.status_of(recs)
    suspects: dict[str, dict] = {}
    use: dict[str, dict] = {}
    for item, r in sorted(recs.items()):
        if not isinstance(r, dict):
            continue
        if item_records.is_outcome_retirement(r):
            # the item record fields: status_reason (str) and status_evidence (episode ids)
            reason = r.get("status_reason")
            suspects[item] = {
                "reasons": [reason] if isinstance(reason, str) and reason else [],
                "episodes": [
                    e for e in (r.get("status_evidence") or []) if isinstance(e, str)
                ],
                "bisect": r.get("bisect"),
                "rollback": r.get("rollback"),
            }
        if isinstance(r.get("use"), dict):
            use[item] = dict(r["use"])
    return status_of, suspects, use


def _rollback_files(
    stores: Stores,
    suspects: dict[str, dict],
) -> dict[str, dict[str, bytes]]:
    """For each item retired by its outcomes with a rollback target: its module (or note) and its package's test
    files there."""
    from .. import layout

    out: dict[str, dict[str, bytes]] = {}
    for item in sorted(suspects):
        target = (suspects[item] or {}).get("rollback")
        if not isinstance(target, str) or not target:
            continue
        files, _ = listing(stores.memory, target)
        path = layout.item_path(item)
        tests = path.rsplit("/", 1)[0] + "/tests/"
        keep = [
            p
            for p in sorted(files)
            if p == path or (p.startswith(tests) and layout.classify(p) == "test")
        ]
        out[item] = {
            p: stores.memory.run("show", f"{target}:{p}").encode("utf-8") for p in keep
        }
    return out


def _curate_state(
    stores: Stores,
    lookup: "EpisodeLookup",
) -> _curate.CurateState | None:
    """CURATE's trigger on memory ``main`` now, after WRITE's gate: the state, with the reasons no
    CURATE pass has been shown, or None when CURATE is not due. It reads library state only: the commit, its
    records (with each function's input fixtures) and shapes, and the fingerprints earlier CURATE passes were
    shown, never the stream.
    """
    from .. import layout
    from ..library_export import item_history
    from ..item_records import effect_of, episodes_of, level_of, records_at
    from ..example_view import example_of, tree_reader
    from ..library_index import build_links, render_index
    from ..overlap import fixture_signatures, overlap_candidates
    from ..shape_rows import shapes_at

    sha = stores.memory.head()
    tmp = Path(tempfile.mkdtemp(prefix="memory-curate-"))
    try:
        files, _ = listing(stores.memory, sha)
        tree = materialise(stores.memory, files, tmp / "main")
        status_of, suspects, use = _lifecycle(stores, sha)
        rows = shapes_at(
            stores.memory,
            stores.evidence,
            sha,
            tree,
            lookup=lookup.action,
            blobs=stores.blobs,
            library_mode=True,
        )
        lib = layout.discover(tree)
        records = records_at(stores.memory, sha)[0]
        state = _curate.CurateState(
            commit=sha,
            overlap=overlap_candidates(
                tree,
                shapes=rows,
                fixtures=fixture_signatures(records),
            ),
            suspects=suspects,
            # the index as the actor's copy renders it, with each function's recorded effect
            index_tokens=estimate_tokens(
                render_index(
                    lib,
                    build_links(lib),
                    status_of,
                    effect_of(records),
                    example_of=example_of(records, tree_reader(tree)),
                    episodes_of=episodes_of(records),
                    level_of=level_of(records),
                ),
            ),
            bodies=layout.function_bodies(tree),
            use=use,
            history=item_history(stores.memory, sha)[0],
            aliases=stores.evidence.aliases(),
            rollback_files=_rollback_files(stores, suspects),
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    state.fired = _curate.due(state, stores.evidence.curate_seen())
    return state if state.fired else None


def _curate_settled(
    stores: Stores,
    pass_id: str,
    state: _curate.CurateState,
    outcome: Any,
    writer: Any,
) -> list[str]:
    """The fingerprints of the reasons CURATE pass *pass_id* dealt with (:func:`..curate.settled`): from what the
    gate landed (its items and curation rows) and the ``kept`` block of the writer's accepted manifest
    (``WriterPass.final_manifest``; None when the pass never finished, so nothing counts as kept).
    """
    manifest = getattr(writer, "final_manifest", None)
    landed = bool(
        outcome is not None and outcome.passed and getattr(outcome, "commit", None),
    )
    changed = _curate.changed_items(
        pass_id,
        landed,
        list(getattr(outcome, "items_merged", None) or []),
        stores.evidence.curations(),
        manifest,
    )
    return _curate.settled(state, changed, _curate.kept(manifest))


def capture_settled_forks(stores: Stores, settings: Any) -> list[dict]:
    """Capture each settled fork's attempt that has no capture yet (:func:`..captures.capture_settled`): the
    harness writes the capture store, confined runs only; never raises, so it never stops a pass."""
    from .. import captures

    try:
        state = Path(stores.paths.state_dir)
        root = state / "memory-staging"
        eids = sorted(os.listdir(root)) if root.is_dir() else []
        lookup = EpisodeLookup(stores)
        return captures.capture_settled(
            state,
            eids,
            episode=lookup.episode,
            signals=stores.evidence.signals_for,
            checker_visible=checker_visible(settings),
        )
    except Exception as exc:  # noqa: BLE001 - capture never stops a pass; the error is recorded
        _error(stores, f"capture step: {type(exc).__name__}")
        return []


def verdict_check(paths: Any, settings: Any) -> Callable[[str], bool] | None:
    """In the library mode, whether an episode's capture is a verdict (:func:`..captures.verdict`), read from the
    capture store now; None with the library mode off (no verdict passes)."""
    from .. import captures

    if not library_enabled(settings):
        return None
    found = captures.verdict_episodes(captures.store_dir(Path(paths.state_dir)))
    return found.__contains__


def verdict_candidates(stores: Stores, settings: Any) -> list[str]:
    """The settled forks not captured yet, so that capturing one may make a verdict (:func:`..captures.verdict`): a
    worker starts to capture them and to run the pass a verdict makes due. Every such fork is a candidate, with or
    without a signal (an environment may give none); a positive one comes first, then one with no signal, then a
    contradicted one (it is captured as failed and makes no pass). Files and the evidence store only, nothing is run;
    [] with the fork off. Never raises."""
    from .. import captures, trust
    from ..staging import STATUS_FILES

    if not (library_enabled(settings) and library_fork(settings)):
        return []
    try:
        state = Path(stores.paths.state_dir)
        root = state / "memory-staging"
        done = captures.episode_rows(captures.store_dir(state))
        ranked: list[tuple[int, str]] = []
        for eid in sorted(os.listdir(root)) if root.is_dir() else []:
            d = root / eid
            if eid in done or d.is_symlink() or not d.is_dir():
                continue
            if not any(os.path.lexists(d / name) for name in STATUS_FILES):
                continue  # not settled yet
            if not stores.evidence.episode_exists(eid):
                continue
            regime = stores.evidence.regime_of(eid)
            got = trust.signal_outcome(
                list(stores.evidence.signals_for(eid)),
                regime if regime in trust.REGIMES else "none",
                checker_visible=checker_visible(settings),
            )
            contrary = got["verdict"] == "rejected" or got["end_outcome"] == "contrary"
            positive = got["verdict"] == "accepted" or got["end_outcome"] == "positive"
            ranked.append((2 if contrary else 0 if positive else 1, eid))
        return [eid for _, eid in sorted(ranked)]
    except Exception as exc:  # noqa: BLE001 - a worker then starts only when a pass is due
        _error(stores, f"verdict candidates: {type(exc).__name__}")
        return []


async def run_due_passes(
    stores: Stores,
    eid: str,
    sha: str,
    state: Any,
    *,
    effort: str,
    settings: Any,
    emit: Callable[[dict], None] | None,
    clock: Callable[[], float] = time.monotonic,
    supervise: Any = None,
    first: int = 0,
    new_verdicts_only: set[str] | None = None,
    ran: list[PassRequest] | None = None,
) -> list[PassOutcome]:
    """Run the passes due once episode *eid* (commit *sha*) is indexed, in order; blocking.

    *state* is the harness state (:class:`.state.State`): its drift channels ride with the pass and are
    cleared once it is recorded; a passed pass clears them from ``suspect``. A pass recorded passed or
    failed advances the trigger's cursor; a pass the run guard holds back stays due. *effort* is the
    actor's effort unless ``UNIFY_MEMORY_WRITER_EFFORT`` fixes another (``request.writer_effort``); it picks the
    pass's scale of the allowance cap and its call limit.

    With *supervise* (the library mode's worker, :class:`.async_pass.Supervisor`), the writer's deadline is the supervisor's
    ``pass_deadline_s``, the pass runs under its wall bound, its writer-lane calls are reconciled from the proxy
    journal's window before it is settled (unpriced calls are booked at the worst case of one
    writer call; with no readable journal the pass is not settled and its whole cap stays committed), the end event
    adds ``ended`` and ``reconciled``, and only the first due pass runs.

    In the library mode a capture that is a verdict makes a pass due at once (kind ``"verdict"``,
    :class:`..trigger.Trigger`'s ``verdict``). *first* numbers the passes from ``<eid>.p<first>`` (a worker's later
    slots); with *new_verdicts_only* only a due pass that covers a verdict episode not in it runs; *ran* gets each
    WRITE pass's request as it starts.

    Under ``UNIFY_MEMORY_PASSES=off`` nothing runs: no settings are read, no pass starts and no call is made.
    """
    from .switch import memory_passes_on

    if not memory_passes_on(settings):
        return []
    try:
        cfg = writer_settings(settings)
    except WriterRouteRefused:
        # visible in the run's own events, value-free; the request stays due and refuses again next time
        if stores is not None:
            _deliver(
                stores,
                emit,
                {
                    "type": "consolidation",
                    "phase": "refused",
                    "episode_id": eid,
                    "consolidation_refused": CODE_ROUTE_NOT_IN_EFFECT,
                    "reason_codes": [CODE_ROUTE_NOT_IN_EFFECT],
                },
            )
        raise
    if not isinstance(effort, str) or not effort.strip():
        _error(stores, f"no pass for {eid}: the writer's effort is empty")
        return []  # nothing recorded; the request stays due
    level = effort.strip().lower()
    if level not in cfg.effort_scale or level not in cfg.max_calls:
        _error(
            stores,
            f"no pass for {eid}: the writer's effort has no pass limits (low, medium or high)",
        )
        return []  # nothing recorded; the request stays due
    scale, max_calls = cfg.effort_scale[level], cfg.max_calls[level]
    from . import fork as _fork

    if library_enabled(settings) and library_fork(settings):
        # every settled fork's attempt is captured at once, before any pass and whether or not one is due
        capture_settled_forks(stores, settings)

    trig = Trigger(
        stores.evidence,
        mode="batched",
        experience_budget=cfg.experience_budget,
        usd_per_token=cfg.usd_per_token,
        # with the fork on, an episode joins a batch only once its fork has settled; the rest roll to the next pass
        admission=_fork.admission(stores.paths, settings),
        # the library mode: a capture that is a verdict makes a pass due at once
        verdict=verdict_check(stores.paths, settings),
    )
    drift = sorted(getattr(state, "drift", set()) or set())
    for ch in drift:
        trig.queue_drift(ch)
    due = trig.after_episode(eid)
    if new_verdicts_only is not None:
        due = [r for r in due if set(r.verdicts) - new_verdicts_only]
    if not due:
        return []
    cap = trig.pass_budget_usd() * scale
    reserve = cap / max_calls
    # a library-mode pass is uncapped only when the writer's calls go through a proxy route,
    # whose ceilings bound it in flight; with no route it keeps the allowance cap and the call limit
    uncapped = bool(cfg.library_mode and cfg.route is not None)
    lookup = EpisodeLookup(stores)
    common = dict(
        action_lookup=lookup.action,
        # the library-mode switches (each default is v2's); the writer's brief follows the gate (cadence_replay does the same)
        **surfacing_options(settings).gate_kwargs(),
        qa=QAConfig.from_settings(settings),  # stage-5 test checks; all off by default
    )
    gate = Gate(
        stores.memory,
        stores.evidence,
        stores.blobs,
        **common,
        # library mode: the library layout and checks for WRITE; v2 when the switch is off
        library_mode=(
            library_checks.config_for(
                stores,
                lookup,
                checker_visible=checker_visible(settings),
            )
            if library_enabled(settings)
            else None
        ),
    )
    curate_gate: Gate | None = None  # built when a CURATE pass is first queued
    config = PassConfig(
        model=cfg.model,
        effort=effort,
        # library mode: no USD cap and no count limits on a pass; the step guard and the deadline are
        # operational. The allowance (cap) stays the run guard's booking for the pass (_reserve), never a pass limit.
        max_calls=STEP_GUARD if uncapped else max_calls,
        deadline_s=supervise.pass_deadline_s if supervise is not None else DEADLINE_S,
        max_usd=None if uncapped else cap,
        # staging lives beside the fork's (fork.staging_dir); helpers write beside it
        **(
            {
                # staging exists only with the episode-end fork (or its offline replay, which runs with the fork
                # setting on): with the fork off the writer is told about no staging and its batch map has none
                "staging_root": (
                    str(Path(stores.paths.state_dir) / "memory-staging")
                    if library_fork(settings)
                    else None
                ),
                # each pass's tool events, box opens and refusal codes, kept after the pass (one folder per pass)
                "pass_record_root": str(Path(stores.paths.state_dir) / "passes"),
                # UNIFY_MEMORY_NOTEBOOK=on: store passes read and extend the run's notebook (notebook.py)
                "notebook_state": (
                    str(stores.paths.state_dir) if notebook_enabled(settings) else None
                ),
                "pool": library_pool(settings),
                # UNIFY_TOOLS_IN_CODE=on: the writer, the curator and their helpers carry one JSON tool, execute_code,
                # and call every other tool from their cells
                "tools_in_code": getattr(settings, "UNIFY_TOOLS_IN_CODE", "off") == "on",
                "edit_tools": library_edit_tools(settings),
                # the writer's brief and rows name staging only when the episode-end fork writes it
                "fork": library_fork(settings),
                "compact": library_compact(settings),
                "coverage_rule": library_coverage(settings),
                "trees_cache": str(
                    Path(stores.paths.state_dir) / "memory-workspace-trees",
                ),
                "similar_index": str(_similar.index_dir(stores.paths.state_dir)),
                "worktree_git": (
                    str(stores.paths.worktree_git)
                    if Path(stores.paths.worktree_git).is_dir()
                    else None
                ),
            }
            if cfg.library_mode
            else {}
        ),
        show_usage=cfg.show_usage
        or cfg.library_mode,  # library-mode writers always see measured library use
        library_mode=cfg.library_mode,
        # library mode: the writer's reader calls (UNIFY_MEMORY_LIBRARY_MAX_READS); off, PassConfig's default as before
        **({"max_reads": library_max_reads(settings)} if cfg.library_mode else {}),
    )
    outcomes: list[PassOutcome] = []
    queue = list(
        due,
    )  # in the library mode a CURATE pass joins after a WRITE pass when the library state warrants it
    curate_state: _curate.CurateState | None = None
    stopped = (
        supervise.stop.is_set if supervise is not None else None
    )  # bounds the item-record bisects on SIGTERM
    recorded: list[str] = (
        []
    )  # the passes recorded since the last item records (library mode, after_passes)
    i = 0
    while i < len(queue):
        req, pass_id = queue[i], f"{eid}.p{first + i}"
        i += 1
        curating = req.kind == "curate"
        if curating and curate_gate is None:
            curate_gate = Gate(
                stores.memory,
                stores.evidence,
                stores.blobs,
                **common,
                library_mode=library_checks.config_for(
                    stores,
                    lookup,
                    role="curate",
                    checker_visible=checker_visible(settings),
                ),
            )
        if (
            cfg.run_guard_usd is not None
            and committed_writer_usd(stores) + cap > cfg.run_guard_usd
        ):
            # the run guard: no further pass starts; the request stays due
            _deliver(
                stores,
                emit,
                _end_event(stores, pass_id, None, [], 0.0, "run_guard"),
            )
            break
        rows: list[CostRow] = []
        # the pass this spend belongs to, as the fork's calls carry fork.<episode>, on the writer's route (always set
        # under the paper profile); a default run's call without a route is as shipped
        session = f"{'curator' if curating else 'writer'}.{pass_id}"
        if cfg.route is not None:
            turn = unillm_turn(cfg.model, effort, route=cfg.route, session=session)
        else:
            turn = unillm_turn(cfg.model, effort)
        # the pass's progress, as it runs: seconds from its start (set when the start event is delivered)
        began: list[float | None] = [None]

        def report(phase: str, _pass: str = pass_id, _began: list = began, **fields: Any) -> None:
            seconds = round(max(0.0, clock() - _began[0]), 3) if _began[0] is not None else 0.0
            _deliver(
                stores,
                emit,
                {"type": "consolidation", "phase": phase, "pass_id": _pass, "seconds": seconds, **fields},
            )

        writer = WriterPass(
            stores.memory,
            _ReportingGate(curate_gate if curating else gate, report),
            stores.evidence,
            lookup.episode,
            recording_turn(_reporting_turn(turn, report), rows, cfg.model),
            config,
            redactor=Redactor.from_environ(os.environ),
            curate=curate_state if curating else None,
            **({} if curating else _helper_turns(settings, effort, cfg.library_mode)),
        )
        try:
            _reserve(stores, pass_id, cap, reserve, max_calls, scale)
        except OSError as exc:  # an unrecorded commitment: start nothing
            _error(stores, f"{pass_id}: ledger: {type(exc).__name__}: {exc}")
            break
        start = _start_event(req, pass_id, cfg.model, effort, cap, max_calls, scale)
        if cfg.profile:
            start["profile"] = cfg.profile  # UNIFY_PROFILE, only when set
        if req.verdicts:
            # due at once on these episodes' captures (or a batch that covers them)
            start["pass_kind"] = req.kind
            start["verdict_episodes"] = list(req.verdicts)
        if ran is not None and not curating:
            ran.append(req)
        if cfg.library_mode:
            # the E and the read cap this pass ran with (v2's start event is unchanged)
            start["experience_budget"] = int(cfg.experience_budget)
            start["max_reads"] = int(
                config.max_reads,
            )  # recorded; not enforced on an uncapped pass
            # no pass cap or count limits; the operational step guard; the allowance only as the run guard's
            # booking for this pass
            if uncapped:
                start["cap_usd"] = None
                start["max_calls"] = None
                start["operational_step_guard"] = STEP_GUARD
                start["run_guard_booking_usd"] = _usd(cap)
            else:
                start["capped"] = (
                    "no writer route: the allowance cap and call limit apply"
                )
        if curating:
            # why it runs; codes and ids only
            start["curate"] = sorted(curate_state.fired.values())
        _deliver(stores, emit, start)
        started = clock()
        began[0] = started
        outcome: PassOutcome | None = None
        failure: str | None = None
        failure_code: str | None = None
        try:
            if supervise is None:
                outcome = await asyncio.wait_for(
                    writer.run(req, pass_id),
                    timeout=DEADLINE_S + OVERRUN_S,
                )
            else:  # library mode: the worker's wall bound; SIGTERM cancels through the abort path
                supervised = await supervise(lambda: writer.run(req, pass_id))
                outcome = supervised.outcome
                if outcome is None:
                    failure_code = (
                        "deadline" if supervised.ended == "deadline" else "writer_error"
                    )
        except (
            Exception
        ) as exc:  # noqa: BLE001 - WriterPass recorded the pass as failed
            # the outer bound (a pass that overran its deadline) is a deadline; anything else an error
            failure_code = (
                "deadline" if isinstance(exc, TimeoutError) else "writer_error"
            )
            failure = redact_error(f"pass error: {type(exc).__name__}: {exc}")
            _error(stores, f"{pass_id}: {failure}")
        except BaseException:
            failure_code = "writer_error"  # cancelled or interrupted
            raise
        finally:
            recon = None
            if supervise is not None:
                try:
                    recon = await supervise.reconcile(prices=writer_prices(settings))
                except Exception as exc:  # noqa: BLE001
                    # unreconciled: no settle line, so the whole cap stays committed
                    _error(stores, f"{pass_id}: reconcile: {type(exc).__name__}")
                    recon = unavailable_reconcile()
            _note_costs(stores, sha, eid, pass_id, rows, effort)
            _note_transcript(stores, sha, pass_id, writer)
            try:
                if recon is None:
                    _settle(stores, pass_id, *_spend(outcome, rows)[:2])
                elif recon["journal"] == "read":
                    _settle(
                        stores,
                        pass_id,
                        recon["usd"],
                        recon["unknown"],
                        Decimal(recon["worst_case_usd"]),
                    )
                # else no readable journal: no settle line, so the whole cap stays committed
            except OSError as exc:  # unsettled: the whole cap stays committed
                _error(stores, f"{pass_id}: ledger: {type(exc).__name__}: {exc}")
            event = _end_event(
                stores,
                pass_id,
                outcome,
                rows,
                clock() - started,
                failure_code,
            )
            event["episodes"] = list(
                req.episodes,
            )  # ties the fork's and the helpers' spend to this pass's batch
            event["result"] = _result(outcome)
            if library_enabled(settings):
                # Amendment C: the lag when the pass ended, before publishing
                event["served_lag_commits"] = _served_lag(stores)
            if supervise is not None:
                read = recon["journal"] == "read"
                event["usd"] = recon["usd"] if read else UNKNOWN
                event["unknown_cost_calls"] = recon["unknown"] if read else None
                event["ended"] = supervise.last.ended if supervise.last else None
                event["reconciled"] = recon
            _deliver(stores, emit, event)
            if stores.evidence.pass_exists(pass_id):
                recorded.append(pass_id)
                if curating:
                    # settled, not merely shown (:mod:`..curate`): a reason the pass neither changed an item of
                    # nor kept with a reason fires again next time
                    stores.evidence.record_curate_seen(
                        pass_id,
                        _curate_settled(stores, pass_id, curate_state, outcome, writer),
                    )
                else:
                    trig.mark_done(req)
                    if hasattr(state, "drift"):
                        state.drift.difference_update(drift)
                    if (
                        outcome is not None
                        and outcome.passed
                        and hasattr(state, "suspect")
                    ):
                        state.suspect.difference_update(drift)
        if outcome is None:
            break
        outcomes.append(outcome)
        if _CALL_MAY_BE_IN_FLIGHT & set(getattr(outcome, "codes", None) or ()):
            # a model call ended by the deadline or an error may still be running at the writer's proxy, which serves
            # one writer call at a time: start no further pass in this session (the requests stay due)
            break
        if cfg.library_mode and not curating:
            # WRITE's item records and statuses first, so CURATE's trigger reads fresh
            # suspects and use; then CURATE if the library's state warrants it
            _records_step(stores, settings, recorded, emit, stopped)
            recorded = []
            try:
                curate_state = _curate_state(stores, lookup)
            except (
                Exception
            ) as exc:  # noqa: BLE001 - a measurement never stops consolidation; nothing is due
                _error(stores, f"{pass_id}: curate state: {type(exc).__name__}")
                curate_state = None
            if curate_state is not None and not (
                supervise is not None and supervise.stop.is_set()
            ):
                episodes = sorted(
                    {
                        e
                        for row in curate_state.suspects.values()
                        for e in (row or {}).get("episodes", [])
                        if isinstance(e, str) and stores.evidence.episode_exists(e)
                    },
                )
                curate_req = PassRequest("curate", None, episodes, False, None)
                if supervise is not None:
                    queue[i:] = [
                        curate_req,
                    ]  # the worker's slot: this WRITE, then the CURATE it made due
                else:
                    queue.append(curate_req)
        if supervise is not None and not (i < len(queue) and queue[i].kind == "curate"):
            break  # one pass per worker, and the CURATE it made due in the same slot
    _records_step(
        stores,
        settings,
        recorded,
        emit,
        stopped,
    )  # library mode only (CURATE's records, or every pass's without library-mode CURATE); at once with the switch off
    if library_enabled(settings):
        # the pin-able head moves last, after the item records are written
        from ..memory_writer import publish

        try:
            served = publish(stores.memory)
        except Exception as exc:  # noqa: BLE001
            # served stays where it was: requests keep a recorded pin
            _error(stores, f"publish: {type(exc).__name__}")
            served = None
        lag = _served_lag(stores)
        if served is not None:
            _deliver(
                stores,
                emit,
                {
                    "type": "consolidation",
                    "phase": "published",
                    "served": served,
                    "served_lag_commits": lag,
                },
            )
        if lag:
            # Amendment C: commits without their item records, which no request can pin yet
            _deliver(
                stores,
                emit,
                {
                    "type": "consolidation",
                    "phase": "served_lag",
                    "served_lag_commits": lag,
                },
            )
    return outcomes


#: How a supervised pass ended when the worker may start the next slot: the writer is done and no call of it can
#: still be in flight at its proxy.
_SLOT_ENDS = ("landed", "refused")


async def run_slots(
    stores: Stores,
    eid: str,
    sha: str,
    state: Any,
    *,
    effort: str,
    settings: Any,
    supervisor: Callable[[int], Any],
    on_slot: Callable[[int, str], None] | None = None,
    on_slot_end: Callable[[int, Any, list[PassOutcome]], None] | None = None,
) -> list[tuple[Any, list[PassOutcome]]]:
    """The library mode's worker, one writer at a time: its first slot runs the due pass (and the CURATE it makes
    due) as :func:`run_due_passes` does under a supervisor; then, slot after slot, each pass a verdict made due
    meanwhile (a capture taken at the slot's start, :func:`capture_settled_forks`, that is a verdict), each starting
    only after the slot before it ended. A verdict is never dropped: one that is due while a writer runs waits for it,
    and one this worker cannot reach stays due in the capture store and the cursor, for the next worker. Nothing caps
    the slots; the loop ends when no pass that covers a new verdict is due, when the stop is set, or when a pass did not
    end as landed or refused (a writer call may still be in flight). *supervisor(k)* makes slot *k*'s supervisor;
    *on_slot(k, pass_id)* is called before each slot (the worker's in-flight record and failsafe), and
    *on_slot_end(k, supervisor, outcomes)* after each slot that ran (its result row). The slots that ran, each with its
    supervisor and outcomes."""
    slots: list[tuple[Any, list[PassOutcome]]] = []
    tried: set[str] = set()
    k = 0
    while True:
        sup = supervisor(k)
        first = 2 * k  # a slot's WRITE pass, then the CURATE it made due
        if on_slot is not None:
            on_slot(k, f"{eid}.p{first}")
        ran: list[PassRequest] = []
        outcomes = await run_due_passes(
            stores,
            eid,
            sha,
            state,
            effort=effort,
            settings=settings,
            emit=None,
            supervise=sup,
            first=first,
            new_verdicts_only=tried if k else None,
            ran=ran,
        )
        if sup.last is None:
            break  # nothing was due, or the run guard held it back
        slots.append((sup, outcomes))
        if on_slot_end is not None:
            on_slot_end(k, sup, outcomes)
        for req in ran:
            tried.update(req.verdicts)
        if sup.stop.is_set() or sup.runs[0][0].ended not in _SLOT_ENDS:
            break
        if any(_CALL_MAY_BE_IN_FLIGHT & set(getattr(o, "codes", None) or ()) for o in outcomes):
            break
        k += 1
    return slots


def _records_step(
    stores: Stores,
    settings: Any,
    recorded: list[str],
    emit: Callable[[dict], None] | None,
    stop: Callable[[], bool] | None,
) -> None:
    """:func:`after_passes`, never stopping the slot: a failure is logged, and served is still published (its lag
    then shows the commits without records)."""
    try:
        after_passes(stores, settings, recorded, emit, stop=stop)
    except Exception as exc:  # noqa: BLE001 - the records never stop consolidation
        _error(stores, redact_error(f"item records: {type(exc).__name__}: {exc}"))


def _served_lag(stores: Stores) -> int | None:
    """``served``'s first-parent lag behind memory ``main``, or None when it cannot be read."""
    from ..memory_writer import served_lag

    try:
        return served_lag(stores.memory)
    except Exception:  # noqa: BLE001 - an unreadable lag is unknown, never 0
        return None


#: Pass end codes after which a model call may still be in flight at the writer's proxy.
_CALL_MAY_BE_IN_FLIGHT = frozenset({"deadline", "writer_error", "route_not_in_effect"})

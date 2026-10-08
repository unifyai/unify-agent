"""The consolidation driver (integration Task 24, v1): checker signal, size trigger, Sol pass, gate.

After each request's episode is committed and indexed, :func:`run_due_passes` asks the batched size
trigger (:class:`..trigger.Trigger`, spec §F1) whether the experience recorded since the last pass has
reached E tokens. A due pass covers every channel with new evidence (there is no channel filter: the
gate's G2 dispatches covers per action kind, so tool, shell, work-tree and dialogue channels all take
part). The pass runs Sol (:class:`..sol_pass.SolPass`) behind the gate, blocking the next request.

Bounds per pass: ``PassConfig(model, effort, max_calls=M[effort], deadline_s=900, max_usd=E x a_tok x S[effort])``,
with S and M from ``UNIFY_MEMORY_V2_SOL_EFFORT_SCALE`` (``low:1,medium:2,high:5``) and
``UNIFY_MEMORY_V2_SOL_MAX_CALLS`` (``low:40,medium:80,high:80``): one rule for every bed, by the pass's Sol effort
(an effort with no entry starts no pass and is logged; the request stays due).
Sol's reasoning effort is the actor's for the run unless ``UNIFY_MEMORY_V2_SOL_EFFORT`` fixes another (a declared
mismatch ablation; the lead, 8 Oct), passed in by the caller.
The run guard (``UNIFY_MEMORY_V2_SOL_RUN_GUARD_USD``, empty for none) starts no further pass once the Sol
USD this home has committed plus the next pass's cap would exceed it; the request then stays due. The
commitment is kept in a ledger in the state dir (:func:`committed_sol_usd`): a pass reserves its whole cap
before it starts and settles at its end to its known USD plus a per-call reserve for each unpriced call,
so a pass that never ends (a killed process) keeps its whole cap committed. No reason text is read. The
last call of a pass can overshoot the pass's cap (the cap is checked before each call), and with it the
guard, by at most one call.

Each pass sends two events through ``emit`` and appends them to the harness-only state dir's
``events.jsonl`` (:func:`events_path`):

* ``{"type": "consolidation", "phase": "start", "pass_id", "trigger_tokens", "episodes", "sol_model",
  "sol_effort", "cap_usd", "max_calls", "effort_scale"}`` (the ledger's reserve line also keeps the last two);
* ``{"type": "consolidation", "phase": "end", "pass_id", "usd", "unknown_cost_calls", "calls", "checks",
  "seconds", "gate_passed", "items", "index_tokens", "reason_codes", "items_merged", "items_refused"}``
  (``calls`` counts model calls and Sol's ``check`` calls; ``checks`` is the latter alone;
  ``items_merged`` lists the manifest items that landed and ``items_refused`` maps each refused item to its
  value-free codes, ``G1``..``G6``, ``dependency`` or ``pass``: per-item admission, :meth:`..gate.Gate.merge`).

``reason_codes`` are codes only, never free text (deduplicated, at most 10): ``G1``..``G6`` for the gate
checks that refused, ``no_manifest``, ``manifest_invalid``, ``over_quota``, ``deadline``, ``pass_cap``,
``run_guard``, ``sol_error``, ``route_not_in_effect`` (a Sol call found Sol's declared route not in effect),
and ``ok`` for a passed pass. They are structural: :attr:`PassOutcome.codes`
(set where each cause arises in the pass and the gate), the type of an exception that ended a pass, or the
run guard; reason text is never read. A failed pass with no listed cause (memory ``main`` moved during the
merge) is ``sol_error``. A pass the run guard holds back sends one end event (``calls`` 0, ``usd`` "0",
``reason_codes`` ``["run_guard"]``) and no start event, since no pass started. The full reasons stay
harness-side in the evidence store's ``passes`` row.

When Sol's route is set but refused (:class:`.switch.SolRouteRefused`: malformed, half set, not what the
settings hold, or ``UNILLM_OTEL`` on), no pass ever starts, so each request instead sends one value-free
event, ``{"type": "consolidation", "phase": "refused", "episode_id", "consolidation_refused":
"route_not_in_effect", "reason_codes": ["route_not_in_effect"]}``, before the refusal is raised (and recorded
in ``errors.jsonl``): a run with any such event consolidated nothing because of its route, which is not a
null result of consolidation.

Sol's transcript (its messages, tool calls and tool results, redacted by the environment's credentials
and every registered secret, Sol's route token among them, then by :func:`..redact.redact_error`, then
bounded: :func:`..sol_pass.transcript_lines`) goes as note lines on ``refs/notes/sol-transcripts`` of the
same episode commit, one JSON line per message with its ``pass_id``.

Money is a plain decimal string, never an exponent; a measurement that could not be taken is ``None``.
Sol's per-turn cost rows go as note lines on ``refs/notes/costs`` of the request's episode commit: a pass
can only start after that commit (the trigger and the export read it), and one episode is one append-only
commit. :func:`episode_costs` merges the commit's ``cost.jsonl`` with those notes.

Ruling R10: the checker contributes ``pass``/``fail`` only (:func:`post_checker`); Sol sees the episodes
through :func:`..sol_pass.export_for_sol`, which carries no signal.
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

from ..blobs import BlobStore
from ..episodes import Action, CostRow, Episode, episode_dir, load_episode
from ..evidence import EvidenceStore
from ..gate import Gate
from ..gitio import Repo
from ..index import build_index, estimate_tokens
from ..memory_repo import items as memory_items
from ..qa import QAConfig
from ..redact import Redactor, redact_error
from ..signals import Signal, SignalMasked, post_signal
from ..snapshot import listing, materialise
from ..sol_pass import (
    CODE_ROUTE_NOT_IN_EFFECT,
    OTEL_REFUSAL,
    PassConfig,
    PassOutcome,
    SolPass,
    SolRoute,
    TRANSCRIPT_REF,
    otel_on,
    unillm_turn,
)
from ..trigger import EXPERIENCE_BUDGET, USD_PER_TOKEN, PassRequest, Trigger
from .cost import UNKNOWN, money, recording_turn
from .paths import Paths
from .switch import (
    SOL_BASE_URL,
    SOL_EFFORT_SCALE,
    SOL_MAX_CALLS,
    SolRouteRefused,
    settle_sol_route_env,
    sol_effort_scale_map,
    sol_max_calls_map,
    sol_route,
    sol_token,
    surfacing_options,
)

__all__ = [
    "DEADLINE_S",
    "MAX_CALLS",
    "SOL_MODEL",
    "EpisodeLookup",
    "SolSettings",
    "Stores",
    "committed_sol_usd",
    "episode_costs",
    "events_path",
    "open_stores",
    "post_checker",
    "run_due_passes",
    "sol_settings",
]

SOL_MODEL = "openai/gpt-6-sol"
MAX_CALLS = 40  # the low-effort default of UNIFY_MEMORY_V2_SOL_MAX_CALLS
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
        "run_guard",
        "sol_error",
        CODE_ROUTE_NOT_IN_EFFECT,
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


def open_stores(paths: Paths) -> Stores:
    """The harness-side stores under ``paths.home``; the two bare repos are created on first use."""
    paths.home.mkdir(parents=True, exist_ok=True)
    return Stores(
        paths,
        _bare(paths.memory),
        _bare(paths.episodes),
        BlobStore(paths.blobs),
        EvidenceStore(paths.evidence),
    )


def events_path(paths: Paths) -> Path:
    """``paths.events`` (the harness-only state dir's ``events.jsonl``); the same file without that property."""
    events = getattr(paths, "events", None)
    return Path(events) if events is not None else paths.state_dir / "events.jsonl"


def ledger_path(paths: Paths) -> Path:
    """The run guard's ledger: one ``reserve`` line before each pass starts, one ``settle`` line after it."""
    return paths.state_dir / "sol-spend.jsonl"


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
    """The episode's ``cost.jsonl`` rows, then the late rows noted on its commit (Sol's)."""
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


# --- settings --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SolSettings:
    model: str
    experience_budget: int
    usd_per_token: Decimal
    run_guard_usd: Decimal | None
    # Sol's own route (UNIFY_MEMORY_V2_SOL_BASE_URL / _SOL_TOKEN or _SOL_TOKEN_FD); None: as shipped
    route: SolRoute | None = None
    show_usage: bool = (
        False  # UNIFY_MEMORY_V2_SOL_USAGE: the use table in each pass's first message
    )
    # UNIFY_MEMORY_V2_SOL_EFFORT_SCALE / _SOL_MAX_CALLS, by Sol effort (low, medium, high)
    effort_scale: dict[str, Decimal] = field(
        default_factory=lambda: sol_effort_scale_map(""),
    )
    max_calls: dict[str, int] = field(default_factory=lambda: sol_max_calls_map(""))

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


def sol_settings(settings: Any) -> SolSettings:
    """The Sol model, E, the USD allowance per token, the run guard, Sol's route and whether Sol is shown
    the use table (``UNIFY_MEMORY_V2_SOL_USAGE``) from *settings*.

    Defaults if unset. :func:`run_due_passes` reads them before anything else, so a refused value (Sol's
    route with one of its two settings empty, say) starts no pass and makes no call. Sol's route is settled
    against the process environment again first (:func:`.switch.settle_sol_route_env`), so a value that
    reached the environment late (``.env`` loaded by an embedder after unify was imported) is refused here
    too, not only on the CLI's path. Every refusal of the route is a :class:`.switch.SolRouteRefused`.
    """
    settle_sol_route_env(os.environ, settings)
    model = str(getattr(settings, "UNIFY_MEMORY_V2_SOL_MODEL", "") or "").strip()
    raw_e = getattr(settings, "UNIFY_MEMORY_V2_E", "") or EXPERIENCE_BUDGET
    if isinstance(raw_e, bool):
        raise ValueError(f"UNIFY_MEMORY_V2_E must be a positive integer, not {raw_e!r}")
    try:
        e = int(str(raw_e).strip())
    except ValueError:
        e = 0
    if e <= 0:
        raise ValueError(
            f"UNIFY_MEMORY_V2_E must be a positive integer, not {raw_e!r}"[:200],
        )
    a_tok = _decimal(
        "UNIFY_MEMORY_V2_SOL_ALLOWANCE_USD_PER_TOKENS",
        getattr(settings, "UNIFY_MEMORY_V2_SOL_ALLOWANCE_USD_PER_TOKENS", "")
        or format(USD_PER_TOKEN, "f"),
    )
    if (
        a_tok <= 0
    ):  # a zero cap would consume experience with passes that can make no call
        raise ValueError(
            "UNIFY_MEMORY_V2_SOL_ALLOWANCE_USD_PER_TOKENS must be greater than zero",
        )
    guard_raw = getattr(settings, "UNIFY_MEMORY_V2_SOL_RUN_GUARD_USD", "") or ""
    guard = (
        _decimal("UNIFY_MEMORY_V2_SOL_RUN_GUARD_USD", guard_raw)
        if str(guard_raw).strip()
        else None
    )
    # both or neither, checked here (before any pass starts); no error quotes a value
    pair = sol_route(getattr(settings, SOL_BASE_URL, ""), sol_token(settings))
    route = SolRoute(*pair) if pair is not None else None
    model = model or SOL_MODEL
    if route is not None and otel_on():
        raise SolRouteRefused(OTEL_REFUSAL)
    if route is not None and not (
        model if "@" in model else f"{model}@openrouter"
    ).endswith("@openrouter"):
        raise SolRouteRefused(
            f"{SOL_BASE_URL} replaces the OpenRouter transport; UNIFY_MEMORY_V2_SOL_MODEL must be an "
            "@openrouter endpoint (or a bare model id)",
        )
    from .switch import parse_sol_usage

    usage = parse_sol_usage(getattr(settings, "UNIFY_MEMORY_V2_SOL_USAGE", "") or "")
    scale = sol_effort_scale_map(getattr(settings, SOL_EFFORT_SCALE, "") or "")
    calls = sol_max_calls_map(getattr(settings, SOL_MAX_CALLS, "") or "")
    return SolSettings(model, e, a_tok, guard, route, usage == "on", scale, calls)


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


def _settle(stores: Stores, pass_id: str, usd: str, unknown_cost_calls: int) -> None:
    _ledger(
        stores,
        {
            "pass_id": pass_id,
            "phase": "settle",
            "usd": usd,
            "unknown_cost_calls": int(unknown_cost_calls),
        },
    )


def committed_sol_usd(stores: Stores) -> Decimal:
    """The Sol USD committed in this home, from the ledger's structured fields only.

    A pass with a ``settle`` line counts its known USD plus ``unknown_cost_calls`` times the per-call
    reserve it held (an unpriced call's cost is unknown, never zero). A pass with only its ``reserve``
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
            "sol_effort": effort,
        }
        try:
            stores.episodes.add_note(
                sha,
                json.dumps(note, sort_keys=True),
                ref=COSTS_REF,
            )
        except Exception as exc:  # noqa: BLE001
            _error(stores, f"cost note: {type(exc).__name__}: {exc}")


def _note_transcript(stores: Stores, sha: str, pass_id: str, sol: SolPass) -> None:
    """Sol's redacted, bounded transcript as note lines on :data:`..sol_pass.TRANSCRIPT_REF` of *sha*."""
    try:
        stores.episodes.append_note_lines(sha, sol.transcript(pass_id), TRANSCRIPT_REF)
    except Exception as exc:  # noqa: BLE001 - a record, never a failure of the pass
        _error(stores, f"{pass_id}: transcript: {type(exc).__name__}")


def _library_after(stores: Stores) -> tuple[int | None, int | None]:
    """(listed items, index tokens) of memory ``main`` now; None for what could not be measured."""
    tmp = Path(tempfile.mkdtemp(prefix="memv2-after-"))
    try:
        files, _ = listing(stores.memory, stores.memory.head())
        tree = materialise(stores.memory, files, tmp / "main")
        listed = sum(
            1 for it in memory_items(tree).items if it.kind != "workflow" and it.listed
        )
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
    raw = list(outcome.codes) if outcome is not None else [failure_code or "sol_error"]
    out: list[str] = []
    for code in raw:
        if code in REASON_CODES and code not in out:
            out.append(code)
    if not out or (outcome is not None and not outcome.passed and out == ["ok"]):
        out = ["ok"] if outcome is not None and outcome.passed else ["sol_error"]
    return out[:MAX_REASON_CODES]


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
        "sol_model": model,
        "sol_effort": effort,
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
        "calls": calls,  # model calls plus check calls (SolPass counts both against max_calls)
        "checks": int(outcome.checks) if outcome is not None else 0,
        "seconds": round(max(0.0, float(seconds)), 3),
        "gate_passed": bool(outcome is not None and outcome.passed),
        "items": listed,
        "index_tokens": index_tokens,
        "reason_codes": reason_codes(outcome, failure_code),
        "items_merged": list(outcome.items_merged) if outcome is not None else [],
        "items_refused": (
            {i: list(c) for i, c in sorted(outcome.items_refused.items())}
            if outcome is not None
            else {}
        ),
    }


# --- the driver ------------------------------------------------------------------------------------------


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
) -> list[PassOutcome]:
    """Run the passes due once episode *eid* (commit *sha*) is indexed, in order; blocking.

    *state* is the harness state (:class:`.state.State`): its drift channels ride with the pass and are
    cleared once it is recorded; a passed pass clears them from ``suspect``. A pass recorded passed or
    failed advances the trigger's cursor; a pass the run guard holds back stays due. *effort* is the
    actor's effort unless ``UNIFY_MEMORY_V2_SOL_EFFORT`` fixes another (``request.sol_effort``); it picks the
    pass's scale of the allowance cap and its call limit.
    """
    try:
        cfg = sol_settings(settings)
    except SolRouteRefused:
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
        _error(stores, f"no pass for {eid}: Sol's effort is empty")
        return []  # nothing recorded; the request stays due
    level = effort.strip().lower()
    if level not in cfg.effort_scale or level not in cfg.max_calls:
        _error(
            stores,
            f"no pass for {eid}: Sol's effort has no pass limits (low, medium or high)",
        )
        return []  # nothing recorded; the request stays due
    scale, max_calls = cfg.effort_scale[level], cfg.max_calls[level]
    trig = Trigger(
        stores.evidence,
        mode="batched",
        experience_budget=cfg.experience_budget,
        usd_per_token=cfg.usd_per_token,
    )
    drift = sorted(getattr(state, "drift", set()) or set())
    for ch in drift:
        trig.queue_drift(ch)
    due = trig.after_episode(eid)
    if not due:
        return []
    cap = trig.pass_budget_usd() * scale
    reserve = cap / max_calls
    lookup = EpisodeLookup(stores)
    gate = Gate(
        stores.memory,
        stores.evidence,
        stores.blobs,
        action_lookup=lookup.action,
        # the v2.1 switches (each default is v2's); Sol's brief follows the gate (cadence_replay does the same)
        **surfacing_options(settings).gate_kwargs(),
        qa=QAConfig.from_settings(settings),  # stage-5 test checks; all off by default
    )
    config = PassConfig(
        model=cfg.model,
        effort=effort,
        max_calls=max_calls,
        deadline_s=DEADLINE_S,
        max_usd=cap,
        show_usage=cfg.show_usage,
    )
    outcomes: list[PassOutcome] = []
    for i, req in enumerate(due):
        pass_id = f"{eid}.p{i}"
        if (
            cfg.run_guard_usd is not None
            and committed_sol_usd(stores) + cap > cfg.run_guard_usd
        ):
            # the run guard: no further pass starts; the request stays due
            _deliver(
                stores,
                emit,
                _end_event(stores, pass_id, None, [], 0.0, "run_guard"),
            )
            break
        rows: list[CostRow] = []
        sol = SolPass(
            stores.memory,
            gate,
            stores.evidence,
            lookup.episode,
            recording_turn(
                (
                    unillm_turn(cfg.model, effort)
                    if cfg.route is None
                    else unillm_turn(cfg.model, effort, route=cfg.route)
                ),
                rows,
                cfg.model,
            ),
            config,
            redactor=Redactor.from_environ(os.environ),
        )
        try:
            _reserve(stores, pass_id, cap, reserve, max_calls, scale)
        except OSError as exc:  # an unrecorded commitment: start nothing
            _error(stores, f"{pass_id}: ledger: {type(exc).__name__}: {exc}")
            break
        _deliver(
            stores,
            emit,
            _start_event(req, pass_id, cfg.model, effort, cap, max_calls, scale),
        )
        started = clock()
        outcome: PassOutcome | None = None
        failure: str | None = None
        failure_code: str | None = None
        try:
            outcome = await asyncio.wait_for(
                sol.run(req, pass_id),
                timeout=DEADLINE_S + OVERRUN_S,
            )
        except Exception as exc:  # noqa: BLE001 - SolPass recorded the pass as failed
            # the outer bound (a pass that overran its deadline) is a deadline; anything else an error
            failure_code = "deadline" if isinstance(exc, TimeoutError) else "sol_error"
            failure = redact_error(f"pass error: {type(exc).__name__}: {exc}")
            _error(stores, f"{pass_id}: {failure}")
        except BaseException:
            failure_code = "sol_error"  # cancelled or interrupted
            raise
        finally:
            _note_costs(stores, sha, eid, pass_id, rows, effort)
            _note_transcript(stores, sha, pass_id, sol)
            try:
                _settle(stores, pass_id, *_spend(outcome, rows)[:2])
            except OSError as exc:  # unsettled: the whole cap stays committed
                _error(stores, f"{pass_id}: ledger: {type(exc).__name__}: {exc}")
            _deliver(
                stores,
                emit,
                _end_event(
                    stores,
                    pass_id,
                    outcome,
                    rows,
                    clock() - started,
                    failure_code,
                ),
            )
            if stores.evidence.pass_exists(pass_id):
                trig.mark_done(req)
                if hasattr(state, "drift"):
                    state.drift.difference_update(drift)
                if outcome is not None and outcome.passed and hasattr(state, "suspect"):
                    state.suspect.difference_update(drift)
        if outcome is None:
            break
        outcomes.append(outcome)
        if _CALL_MAY_BE_IN_FLIGHT & set(getattr(outcome, "codes", None) or ()):
            # a model call ended by the deadline or an error may still be running at Sol's proxy, which serves
            # one Sol call at a time: start no further pass in this session (the requests stay due)
            break
    return outcomes


#: Pass end codes after which a model call may still be in flight at Sol's proxy.
_CALL_MAY_BE_IN_FLIGHT = frozenset({"deadline", "sol_error", "route_not_in_effect"})

"""The writer's pass over the attention ledger's entries, run by the writer service, and the stores it works on.

The writer service (:mod:`..writer_service`) hands each pass up to :data:`..attention.K` attention entries
(:mod:`..attention`); :func:`run_attention_pass` runs the writer (:class:`..writer_pass.WriterPass`) over them behind
the gate (:class:`..gate.Gate`), merges what the gate accepts into memory ``main`` with one compare-and-swap, writes
the item records (:func:`after_passes`) and publishes ``served``.

Bounds per pass: ``PassConfig(model, effort, max_calls=M[effort], max_usd=E x a_tok x S[effort])``, with E, a_tok, S
and M fixed (:data:`.switch.PASS_EXPERIENCE`, :data:`.switch.WRITER_USD_PER_TOKEN`, :data:`.switch.WRITER_EFFORT_SCALE`
and :data:`.switch.WRITER_MAX_CALLS`), by the pass's writer effort (the actor's for the run). On a writer lane the
lane's caps bound the pass instead. Every pass also stays within the room memory's spend cap leaves
(:func:`..writer_service.pass_room`).

Each pass sends its events through ``emit`` and appends them to the harness-only state dir's ``events.jsonl``
(:func:`events_path`):

* ``{"type": "consolidation", "phase": "start", "pass_id", "trigger_tokens", "episodes", "writer_model",
  "writer_effort", "cap_usd", "max_calls", "effort_scale"}``;
* the progress rows ``turn`` (each writer model call) and ``gate`` (each gate verdict), with ``seconds``;
* ``{"type": "consolidation", "phase": "end", "pass_id", "usd", "unknown_cost_calls", "calls", "checks",
  "seconds", "gate_passed", "items", "index_tokens", "reason_codes", "items_merged", "items_refused"}``.

``reason_codes`` are codes only, never free text (deduplicated, at most 10): ``G1``..``G6`` for the gate checks that
refused, ``no_manifest``, ``manifest_invalid``, ``over_quota``, ``deadline``, ``pass_cap``, ``writer_error``,
``route_not_in_effect`` (a writer call found the writer's lane not in effect), ``no_progress`` and ``ok`` for a passed
pass. They are structural (:attr:`PassOutcome.codes`); reason text is never read. The full reasons stay harness-side
in the evidence store's ``passes`` row.

The writer's transcript (redacted by the environment's credentials and every registered secret, the writer lane's
token among them, then by :func:`..redact.redact_error`: :func:`..writer_pass.transcript_lines`) is kept whole, never
cut: as chunk files under the pass record root (:mod:`..transcripts`), and on the :data:`..writer_pass.TRANSCRIPT_REF`
notes ref of the episodes repo's head as one JSON line per message with its ``pass_id`` while they fit, else as one row
naming the chunks and their sha256.

Money is a plain decimal string, never an exponent; a measurement that could not be taken is ``None``.
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

from .. import library_checks
from ..blobs import BlobStore
from ..episodes import Action, CostRow, Episode, episode_dir, load_episode
from ..evidence import EvidenceStore
from ..gate import CHECKS as GATE_CHECKS, Gate
from ..gitio import Repo
from ..index import build_index, estimate_tokens
from ..memory_repo import items as memory_items
from ..reconcile import unavailable as unavailable_reconcile
from ..redact import Redactor, redact_error
from ..signals import Signal
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
from ..trigger import PassRequest
from .cost import UNKNOWN, money, recording_turn
from .paths import Paths
from .switch import (
    PASS_EXPERIENCE,
    PASS_MAX_READS,
    PASS_WALL_S,
    WRITER_BASE_URL,
    WRITER_EFFORT_SCALE,
    WRITER_MAX_CALLS,
    WRITER_USD_PER_TOKEN,
    WriterRouteRefused,
    in_episode,
    settle_writer_route_env,
    writer_journal,
    writer_model,
    writer_prices,
    writer_route,
    writer_token,
)

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
        "writer_error",
        CODE_ROUTE_NOT_IN_EFFECT,
        CODE_NO_PROGRESS,
        "ok",
    },
)
_EPISODE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
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
    """The harness-side stores under ``paths.home``; the two bare repos are created on first use. *busy_timeout_s* is
    the evidence store's SQLite busy timeout (the writer service and a request share it); None keeps the store's own.
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


class EpisodeLookup:
    """Recorded episodes and their actions, read from the episodes repo through the evidence index."""

    def __init__(self, stores: Stores, cache_size: int = 64) -> None:
        self.stores = stores
        self._cache: OrderedDict[str, Episode] = OrderedDict()
        self._size = cache_size

    def episode(self, eid: str) -> Episode:
        """The episode *eid* (``KeyError`` when it was never indexed). A segment of an episode (``<episode>.s<n>``,
        ``UNIFY_MEMORY=in_episode``) is read from its own record (:mod:`..segments`) when it has one."""
        hit = self._cache.get(eid)
        if hit is not None:
            self._cache.move_to_end(eid)
            return hit
        from .. import segments

        if segments.is_segment_name(eid):
            seg = segments.load_snapshot(self.stores.paths.state_dir, eid)
            if seg is not None:
                return seg
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


# --- the checker -----------------------------------------------------------------------------------------


def checker_signal(stores: Stores, eid: str, solved: bool | None, ts: str) -> Signal | None:
    """The checker's verdict on episode *eid* as the signal :func:`post_checker` posts (None for no verdict),
    in the episode's regime; not posted."""
    if not isinstance(solved, bool):
        return None
    return Signal(
        f"{eid}.checker",
        eid,
        "checker",
        "pass" if solved else "fail",
        ts,
        refers_to=eid,
        regime=stores.evidence.regime_of(eid),
    )


def visible_checker_signals(
    stores: Stores,
    ep: Episode,
    entries: list[dict],
    lines: list[dict],
    counterpart: str,
    regime: str,
) -> list[Signal]:
    """The agent-visible checker signals :func:`post_visible_checkers` posts for *entries*, in *regime*, each
    referring to the action it judged where the pairing is certain; not posted. An entry whose verdict never
    reached the actor is dropped and logged (once, here)."""
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
    out: list[Signal] = []
    for n, entry in delivered:
        refers_to = eid
        obs = entry["obs"]
        if obs in answered:
            refers_to = f"{eid}/actions/{positions[answered.index(obs)]}"
        out.append(
            Signal(
                f"{eid}.checker.{n}",
                eid,
                "checker",
                entry["label"],
                entry["at"],
                refers_to=refers_to,
                regime=regime,
                visible_to_actor=True,
            ),
        )
    return out


# --- settings --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class WriterSettings:
    model: str
    # The writer's own lane (UNIFY_MEMORY_WRITER_BASE_URL and _WRITER_TOKEN_FD); None: the controller's own route
    route: WriterRoute | None = None

    @property
    def cap_usd(self) -> Decimal:
        """One pass's USD cap at effort scale 1: E recorded tokens at the writer's rate per token."""
        return Decimal(PASS_EXPERIENCE) * WRITER_USD_PER_TOKEN


def writer_settings(settings: Any) -> WriterSettings:
    """The writer's model (:func:`.switch.writer_model`) and its lane, from *settings*. The lane is settled against
    the process environment again first (:func:`.switch.settle_writer_route_env`), so a value that reached the
    environment late (``.env`` loaded by an embedder after unify was imported) is refused here too. Every refusal of
    the lane is a :class:`.switch.WriterRouteRefused`; no pass starts and no call is made."""
    settle_writer_route_env(os.environ, settings)
    model = writer_model(settings)
    # both or neither, checked here (before any pass starts); no error quotes a value
    pair = writer_route(getattr(settings, WRITER_BASE_URL, ""), writer_token(settings))
    route = WriterRoute(*pair) if pair is not None else None
    if route is not None and otel_on():
        raise WriterRouteRefused(OTEL_REFUSAL)
    if route is not None and not (model if "@" in model else f"{model}@openrouter").endswith("@openrouter"):
        raise WriterRouteRefused(
            f"{WRITER_BASE_URL} replaces the OpenRouter transport; the writer's model must be an @openrouter "
            "endpoint (or a bare model id)",
        )
    return WriterSettings(model, route)


# --- money -----------------------------------------------------------------------------------------------


def _usd(value: Decimal) -> str:
    return format(abs(value) if value.is_zero() else value, "f")


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
    tmp: Path | None = None
    try:
        tmp = Path(tempfile.mkdtemp(prefix="memory-after-"))
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
        if tmp is not None:
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


def _helper_turns(settings: Any, effort: str) -> dict:
    """``WriterPass`` keywords for writer-launched helpers: a factory of fresh host-side turns on the key proxy's
    helper lane (its own token and ledger; the helper model at the pass's effort), or ``helper_turn=None`` with why.

    Fail closed: with no helper lane configured, or a refused one, ``delegate`` is not offered, and there is never a
    fallback to the writer's or the actor's route. The token stays in this process; helper boxes never see it.
    """
    from .switch import helper_lane_route, settle_helper_route_env

    settle_helper_route_env(os.environ, settings)
    pair, why = helper_lane_route()
    model = writer_model(settings)
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
        # library mode only (absent off): the writer's coverage, its reader calls and the
        # blob bytes its /inputs held (uncapped in the library mode, so a disk-full stage leaves its size here)
        **(
            {
                "coverage": outcome.coverage,
                # measures, read by no gate (absent unless recorded)
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
                # the writer's compactions, the messages they took out and those
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
    recorded: list[str],
    emit: Callable[[dict], None] | None,
) -> None:
    """Once a pass is recorded: the item records of ``main``'s head and the status changes
    (:func:`..lifecycle.consolidate_records`). Nothing runs when no pass was recorded. Never raises: a failure is
    logged, and the next pass computes everything again from the same evidence.
    """
    if not recorded:
        return
    from .. import lifecycle
    from ..captures import store_dir as captures_store_dir

    try:
        summary = lifecycle.consolidate_records(
            stores,
            action_lookup=EpisodeLookup(stores).action,
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
        },
    )


# --- library-mode CURATE --------------------------------------------------------------------------------------


def capture_settled_forks(stores: Stores) -> list[dict]:
    """Capture each settled fork's attempt that has no capture yet (:func:`..captures.capture_settled`): what each
    episode (or segment) did, as the harness records it; nothing is run. Never raises, so it never stops a pass."""
    from .. import captures, segments

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
            segment_episode=segments.loader(state),
        )
    except Exception as exc:  # noqa: BLE001 - capture never stops a pass; the error is recorded
        _error(stores, f"capture step: {type(exc).__name__}")
        return []


def _records_step(
    stores: Stores,
    recorded: list[str],
    emit: Callable[[dict], None] | None,
) -> None:
    """:func:`after_passes`, never stopping the slot: a failure is logged, and served is still published (its lag
    then shows the commits without records)."""
    try:
        after_passes(stores, recorded, emit)
    except Exception as exc:  # noqa: BLE001 - the records never stop consolidation
        _error(stores, redact_error(f"item records: {type(exc).__name__}: {exc}"))


def _served_lag(stores: Stores) -> int | None:
    """``served``'s first-parent lag behind memory ``main``, or None when it cannot be read."""
    from ..memory_writer import served_lag

    try:
        return served_lag(stores.memory)
    except Exception:  # noqa: BLE001 - an unreadable lag is unknown, never 0
        return None


# --- the writer service's pass over the attention ledger's entries -------------------------------------------------------


def _model_worst_case(model: str) -> Decimal | None:
    """One request's worst case for *model* (the pinned catalogue's bound, :mod:`..call_bounds`), or None when the
    catalogue does not bound it."""
    from ..call_bounds import NoBound, call_worst_case_usd

    try:
        return call_worst_case_usd(model)
    except NoBound:
        return None


def _helper_spend(writer: Any, helper_turns: list) -> tuple[Decimal, int]:
    """(known USD, unpriced calls) of this run of a pass's helpers: each helper's own record of its priced and
    unpriced calls (a killed run's, its lost calls included, come from its turn files: turns_cost)."""
    known, unknown = Decimal(0), 0
    for row in getattr(writer, "_helper_rows", None) or ():
        try:
            known += Decimal(str(row.get("usd_known") or "0"))
        except (InvalidOperation, ValueError):
            unknown += 1
        unknown += int(row.get("unknown_cost_calls") or 0)
    return known, unknown


@dataclass
class AttentionPassReport:
    """What one writer-service pass did, for the service (its settle, read checks and returns) and the metering row
    (:func:`..metering.write_pass_row`). Money is a decimal string or ``unknown``."""

    pass_id: str
    #: the entries handed to the pass (ids), and the PassRequest the writer got
    entries: list[str]
    ended: str
    outcome: PassOutcome | None = None
    reconciled: dict | None = None
    #: the pass's known USD (the writer lane's journal window when it was read, else the pass's own turns)
    usd_known: str = UNKNOWN
    #: calls with no price (booked at the worst case of one writer call), and that booking
    unknown_calls: int = 0
    worst_case_booked: str = "0"
    #: model turns lost to a killed service and made again (each once, by turn id)
    killed_turns: list[str] = field(default_factory=list)
    #: what the service settles against its cap (None: nothing readable, so the reservation stays committed): the
    #: writer lane's charge (its unpriced calls at the worst case) plus the helpers'
    settle_usd: str | None = None
    #: the helpers' share of the above (their lane is not in the writer journal): priced USD, and unpriced calls
    helper_usd_known: str = "0"
    helper_unknown_calls: int = 0
    reads: Any = None
    delivered: list[int] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    turns: int = 0
    replayed_turns: int = 0
    seconds: float = 0.0
    #: set by the service: the entries verified read, and those that went back
    read: list[str] = field(default_factory=list)
    returned: list[str] = field(default_factory=list)
    #: with ``ended == "error"``: why, as a class name and where (``file:line:function``, or the step that refused);
    #: never a message (rev 2: every failed pass row carries them)
    error_class: str | None = None
    error_where: str | None = None


def _pass_reads(stores: Stores, pass_id: str, writer: Any) -> Any:
    """The pass's read evidence: ``read-log.json`` beside its records when the pass wrote it, else gathered now from
    its agents' read logs, its tool rows and its boxes' open rows, and kept there."""
    from .. import attention

    folder = Path(stores.paths.state_dir) / "passes" / pass_id
    path = folder / attention.READ_LOG_FILE
    if path.is_file():
        return attention.pass_reads(stores.paths.state_dir, pass_id)

    def rows(name: str) -> list[dict]:
        out = []
        try:
            for line in (folder / name).read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if isinstance(row, dict):
                    out.append(row)
        except OSError:
            pass
        return out

    reads = attention.PassReads.collect(
        read_logs=dict(getattr(writer, "_read_logs", {}) or {}),
        tool_events=rows("tool-events.jsonl"),
        box_opens=rows("box-opens.jsonl"),
    )
    try:
        folder.mkdir(parents=True, exist_ok=True)
        attention.write_pass_reads(path, reads)
    except OSError:
        pass
    return reads


async def run_attention_pass(
    home: Path | str,
    pass_id: str,
    handed: list,
    *,
    resume: bool,
    folder: Path,
    journal_start: int | None,
    effort: str,
    settings: Any,
    stop: asyncio.Event | None = None,
    emit: Callable[[dict], None] | None = None,
    turn_factory: Callable[..., Any] | None = None,
    helper_factory: Callable[..., Any] | None = None,
) -> AttentionPassReport:
    """One writer pass over the attention entries *handed* (:class:`..attention.AttentionEntry`), run by the writer
    service: the pass's agent and its helpers on the run record, the writer's brief given the entries, the gate,
    the compare-and-swap merge and the item records as any library-mode pass; its model turns checkpointed under
    *folder* (:class:`..writer_service.ResumableTurn`; with *resume*, the turns already done are answered from there),
    its calls priced from the writer lane's journal window from the pass's first start (*journal_start*).
    *turn_factory* and *helper_factory* replace the writer's and the helpers'
    model turns (tests: scripted turns). Never raises for a pass that failed: the report says how it ended."""
    from unify.agents.run_record import clean_text

    from .. import attention
    from ..writer_service import ResumableTurn
    from .async_pass import Supervisor

    home = Path(home)
    paths = Paths.under(home)
    stores = open_stores(paths, busy_timeout_s=60.0)
    cfg = writer_settings(settings)
    level = effort.strip().lower()
    if level not in WRITER_EFFORT_SCALE or level not in WRITER_MAX_CALLS:
        return AttentionPassReport(
            pass_id,
            [e.entry for e in handed],
            "error",
            notes=["no pass limits for the effort"],
            reads=attention.PassReads(),
            error_class="NoPassLimits",
            error_where="run_attention_pass:limits",
        )
    scale = WRITER_EFFORT_SCALE[level]
    cap = cfg.cap_usd * scale
    max_calls = WRITER_MAX_CALLS[level]
    report = AttentionPassReport(pass_id, [e.entry for e in handed], "nothing")
    capture_settled_forks(stores)
    from ..call_bounds import NoBound
    from ..writer_service import pass_call_worst_case, pass_room, turns_cost

    try:
        # one call's worst case for this pass's models (the pinned catalogue's bound): every unpriced, lost or
        # in-flight call counts at it
        worst = pass_call_worst_case(settings)
    except NoBound as exc:
        return AttentionPassReport(
            pass_id, [e.entry for e in handed], "error", notes=[str(exc)[:300]], reads=attention.PassReads(),
            error_class=type(exc).__name__, error_where="run_attention_pass:worst_case",
        )
    # what a killed run of this pass already spent, per lane, as its turn files stand before this run writes any (a
    # run that diverges from it overwrites them): its priced calls, and its unpriced and lost ones at the worst case.
    # It comes off the room, and is settled with this run's figures; its done calls replay at no cost
    # (ResumableTurn), so nothing is counted twice
    prior = turns_cost(folder) if resume else {"writer": (Decimal(0), 0), "helpers": (Decimal(0), 0)}
    prior_usd = sum((k + Decimal(n) * worst for k, n in prior.values()), Decimal(0))
    room = pass_room(home, pass_id, settings)
    if room is not None:
        room = max(Decimal(0), room - prior_usd)
    supervise = Supervisor(
        float(PASS_WALL_S),
        stop if stop is not None else asyncio.Event(),
        journal=writer_journal(settings),
        models=(cfg.model.split("@", 1)[0],),
    )
    lookup = EpisodeLookup(stores)
    gate = Gate(
        stores.memory,
        stores.evidence,
        stores.blobs,
        action_lookup=lookup.action,
        library_mode=library_checks.config_for(stores, lookup),
        # UNIFY_MEMORY=in_episode: an item may cite a recorded segment of an open episode as its source
        **({"segments_dir": str(paths.state_dir)} if in_episode(settings) else {}),
    )
    # on the run record, held by the service, over the attention ledger's entries (the brief, the first message and
    # the writer's ledger tools, attention_tools.py)
    config = PassConfig(
        model=cfg.model,
        effort=effort,
        max_calls=STEP_GUARD if cfg.route is not None else max_calls,
        deadline_s=supervise.pass_deadline_s,
        max_usd=None if cfg.route is not None else cap,
        # the service's room for this pass (its cap less what its other passes committed, less a resumed pass's lost
        # calls): no writer or helper call starts past it, whatever the pass's own limits
        usd_limit=None if room is None else _usd(room),
        usd_call_worst=_usd(worst),
        staging_root=str(Path(paths.state_dir) / "memory-staging"),
        pass_record_root=str(Path(paths.state_dir) / "passes"),
        trees_cache=str(Path(paths.state_dir) / "memory-workspace-trees"),
        similar_index=str(_similar.index_dir(paths.state_dir)),
        worktree_git=str(paths.worktree_git) if Path(paths.worktree_git).is_dir() else None,
        max_reads=PASS_MAX_READS,
        # the pass's agents join the run record, the writer's row held by the service
        run_home=str(home),
        attention_home=str(home),
    )
    session = f"writer.{pass_id}"
    if turn_factory is not None:
        base_turn = turn_factory()
    elif cfg.route is not None:
        base_turn = unillm_turn(cfg.model, effort, route=cfg.route, session=session)
    else:
        base_turn = unillm_turn(cfg.model, effort)
    inputs = {"pass_id": pass_id, "entries": report.entries}
    turn = ResumableTurn(base_turn, folder, inputs=inputs, scrub=clean_text)
    helper_turns: list[ResumableTurn] = []
    helpers = _helper_turns(settings, effort) if helper_factory is None else {
        "helper_turn": helper_factory,
        "helper_refusal": None,
    }
    if helpers.get("helper_turn") is not None:
        made = helpers["helper_turn"]
        import inspect

        takes_session = bool(inspect.signature(made).parameters)

        def resumable_helper(session_label: str | None = None) -> Any:
            # the k-th helper of the pass checkpoints under h<k>: a resumed pass starts its helpers in the same order
            inner = made(session_label) if takes_session else made()
            rt = ResumableTurn(inner, folder / f"h{len(helper_turns)}", inputs=inputs, scrub=clean_text)
            helper_turns.append(rt)
            return rt

        helpers = {"helper_turn": resumable_helper, "helper_refusal": None}
    rows: list[CostRow] = []
    began: list[float | None] = [None]

    def progress(phase: str, **fields: Any) -> None:
        seconds = round(max(0.0, time.monotonic() - began[0]), 3) if began[0] is not None else 0.0
        _deliver(stores, emit, {"type": "consolidation", "phase": phase, "pass_id": pass_id, "seconds": seconds, **fields})

    writer = WriterPass(
        stores.memory,
        _ReportingGate(gate, progress),
        stores.evidence,
        lookup.episode,
        recording_turn(_reporting_turn(turn, progress), rows, cfg.model),
        config,
        redactor=Redactor.from_environ(os.environ),
        **helpers,
    )
    # each episode by the name it can be read by now (a running one by its latest segment record); one that cannot be
    # read yet is not asked for (the ledger hands no entry naming one; a resumed pass's may)
    resolve = attention.episode_resolver(paths.state_dir, attention.indexed_by(stores.evidence))
    req = PassRequest(
        "attention", None, attention.pass_episodes(handed, resolve=resolve), False, None, attention=list(handed)
    )
    start = _start_event(req, pass_id, cfg.model, effort, cap, max_calls, scale)
    start.update(pass_kind="attention", attention_entries=list(report.entries), resumed=bool(resume))
    _deliver(stores, emit, start)
    started = time.monotonic()
    began[0] = started
    outcome: PassOutcome | None = None
    failure_code: str | None = None
    recon: dict | None = None
    try:
        supervised = await supervise(lambda: writer.run(req, pass_id))
        outcome = supervised.outcome
        report.ended = supervised.ended
        if supervised.error is not None:
            report.error_class, report.error_where = supervised.error["class"], supervised.error["where"]
        if outcome is None:
            failure_code = "deadline" if supervised.ended == "deadline" else "writer_error"
    except Exception as exc:  # noqa: BLE001 - the pass recorded itself
        report.ended = "error"
        from .async_pass import error_of

        raised = error_of(exc)
        report.error_class, report.error_where = raised["class"], raised["where"]
        failure_code = "writer_error"
        report.notes.append(redact_error(f"pass error: {type(exc).__name__}: {exc}")[:300])
        _error(stores, f"{pass_id}: pass error: {type(exc).__name__}")
    finally:
        start_at, end_at = supervise.window
        if resume and journal_start is not None:
            # the window from the pass's first start: the killed turn's call is in it once
            supervise.window = (journal_start, end_at)
        # the writer lane's journal books its unpriced calls at the writer's model's own bound (bounded: checked above)
        writer_worst = _model_worst_case(cfg.model) or worst
        try:
            recon = await supervise.reconcile(prices=writer_prices(settings), worst_case_usd=writer_worst)
        except Exception as exc:  # noqa: BLE001
            _error(stores, f"{pass_id}: reconcile: {type(exc).__name__}")
            recon = unavailable_reconcile(writer_worst)
        # the pass's cost rows and transcript go on the episodes repo's head (an attention pass has no one episode
        # that made it due); its episodes are in its end event
        try:
            head = stores.episodes.head()
        except Exception:  # noqa: BLE001 - no episode recorded yet: nothing to attach the notes to
            head = ""
        if head:
            _note_costs(stores, head, (req.episodes or [""])[0], pass_id, rows, effort)
            _note_transcript(stores, head, pass_id, writer)
        killed = list(turn.killed) + [k for h in helper_turns for k in h.killed]
        report.killed_turns = killed
        report.turns = turn.n + sum(h.n for h in helper_turns)
        report.replayed_turns = turn.replayed + sum(h.replayed for h in helper_turns)
        known, unknown, _calls = _spend(outcome, rows)
        # the helpers' lane: its calls are not in the writer lane's journal; each helper's priced calls, its unpriced
        # ones, and its turns a killed service lost (billed, never priced here), at one call's worst case
        helper_known, helper_unknown = _helper_spend(writer, helper_turns)
        helper_known += prior["helpers"][0]
        helper_unknown += prior["helpers"][1]
        report.helper_usd_known, report.helper_unknown_calls = _usd(helper_known), helper_unknown
        if recon is not None and recon.get("journal") == "read":
            # the journal books the writer's unpriced calls at its own worst case; the helpers' at this pass's. A
            # killed run's writer calls are in the window only when the journal could be read at its start (the
            # window then opens there); else they are counted from its turn files, as the helpers' are
            journal_worst = Decimal(recon["worst_case_usd"])
            w_known, w_unpriced = prior["writer"] if (resume and journal_start is None) else (Decimal(0), 0)
            report.usd_known = _usd(Decimal(recon["usd"]) + w_known + helper_known)
            report.unknown_calls = int(recon["unknown"]) + w_unpriced + helper_unknown
            report.worst_case_booked = _usd(
                Decimal(int(recon["unknown"])) * journal_worst + Decimal(w_unpriced + helper_unknown) * worst
            )
            report.settle_usd = _usd(
                Decimal(recon["booked_usd"]) + w_known + helper_known + Decimal(w_unpriced + helper_unknown) * worst
            )
        else:
            # no readable journal: the pass's own figures (this run's, writer and helpers), with what a killed run of it
            # spent (its lost calls among the unpriced), each unpriced call at the worst case
            report.usd_known = _usd(Decimal(known) + prior["writer"][0] + prior["helpers"][0]) if known != UNKNOWN else known
            report.unknown_calls = int(unknown) + prior["writer"][1] + prior["helpers"][1]
            report.worst_case_booked = _usd(Decimal(report.unknown_calls) * worst)
            # the pass's own priced calls (the writer's and its helpers') and every unpriced or lost call at the worst
            # case: what the service settles when no journal could be read
            report.settle_usd = (
                _usd(Decimal(report.usd_known) + Decimal(report.unknown_calls) * worst)
                if known != UNKNOWN
                else None
            )
        report.reconciled = recon
        report.outcome = outcome
        report.seconds = round(time.monotonic() - started, 3)
        event = _end_event(stores, pass_id, outcome, rows, time.monotonic() - started, failure_code)
        event.update(
            episodes=list(req.episodes),
            result=_result(outcome),
            pass_kind="attention",
            ended=report.ended,
            reconciled=recon,
            killed_turns=len(killed),
        )
        if recon is not None and recon.get("journal") == "read":
            event["usd"], event["unknown_cost_calls"] = recon["usd"], recon["unknown"]
        _deliver(stores, emit, event)
        report.reads = _pass_reads(stores, pass_id, writer)
        report.delivered = sorted(attention.delivered_seqs(home, pass_id))
    if outcome is not None:
        report.notes.extend(str(r) for r in (outcome.reasons or ()))
    _records_step(stores, [pass_id] if stores.evidence.pass_exists(pass_id) else [], emit)
    from ..memory_writer import publish

    try:
        served = publish(stores.memory)
        _deliver(stores, emit, {"type": "consolidation", "phase": "published", "served": served, "served_lag_commits": _served_lag(stores)})
    except Exception as exc:  # noqa: BLE001 - served stays where it was
        _error(stores, f"publish: {type(exc).__name__}")
    return report

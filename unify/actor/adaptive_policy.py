"""``UNIFY_ADAPTIVE_POLICY``: the harness chooses its own drafts, hand-back decision and consequence settings per
session, learned from the deployment's own records (docs/design/adaptive-per-world-policy-20261015.md).

Empty (off) by default and ``on`` otherwise, with ``UNIFY_TOOLS_IN_CODE=on``; while it is on, ``UNIFY_DRAFTS``,
``UNIFY_HANDBACK_DECISION`` and ``UNIFY_CONSEQUENCES`` must be unset, so a run has one source of truth (refused at
start-up otherwise). Off, nothing here runs and the actor's prompt and tools are unchanged.

**What it chooses.** One of five fixed configurations (:data:`ARMS`) at each session's start, held for the session:
off; drafts; drafts + the decision when something was built; that + the consequence mirror; drafts + the decision at
every hand-back with undecided definitions + the mirror. Every text an arm shows is the switches' own, fixed in
advance; the policy never writes prompt text. While it is on, the session's surface is the same under every arm
(``tools.decide`` exists, the workspace is on the import path); only the arm's behaviour differs.

**How it learns.** Discounted Gaussian Thompson sampling (:class:`Learner`) with a flat prior, no context, never a
task id or a position in a stream; each update discounts every arm (``GAMMA``), so the choice follows a changing
world. The rewards:
- at a session's end, its own: progress minus cost over the deployment's running median cost (``COST_WINDOW``
  sessions), cost being the session's model calls; progress the posted outcome's verdict (``solved``; else a ``score``
  in [0, 1]; else the passed share of its checks), 0 when none was posted;
- later, delayed, to the arm that caused it: when a session imports a draft an earlier session wrote, ``REUSE`` to the
  writer session's arm (once per importing session and draft); when a session defines again a definition (the same
  normalised source) that an earlier session made and did not keep, ``-REWORK`` to that earlier session's arm (once per
  later session and definition).

**Where it lives.** ``policy-state.json`` in the deployment's harness home (beside the draft record), which no cell
sees: the arms' discounted statistics, the cost window, each session's arm, and the hashes of the definitions each
session made (kept or not). Every choice (with its propensity: the share of ``DRAWS`` posterior draws its arm
wins), every delayed credit and every session's outcome is a row of the request's ``tool-events.jsonl``.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import random
import statistics
import tempfile
from pathlib import Path
from typing import Any, Optional

SETTING = "UNIFY_ADAPTIVE_POLICY"
STATE = "policy-state.json"
#: The five configurations: drafts on or off, the hand-back decision (never / when built / always), the mirror.
ARMS: tuple[dict, ...] = (
    {"drafts": False, "decision": "never", "mirror": False},
    {"drafts": True, "decision": "never", "mirror": False},
    {"drafts": True, "decision": "built", "mirror": False},
    {"drafts": True, "decision": "built", "mirror": True},
    {"drafts": True, "decision": "always", "mirror": True},
)
GAMMA = 0.97
DRAWS = 256
COST_WINDOW = 50
REUSE = 0.5
REWORK = 0.5
#: The flat prior: every arm starts with this pseudo-weight at mean 0 and this variance.
PRIOR_WEIGHT = 1.0
PRIOR_VARIANCE = 1.0
#: The most definitions' hashes the state keeps (the oldest are dropped first).
SHADOW_MAX = 5000


def _on(settings: Any, name: str) -> bool:
    from unify.actor import drafts

    try:
        return drafts.parse(getattr(settings, name, "") or "") == drafts.ON
    except ValueError:
        return False


def _settings(settings: Any) -> Any:
    if settings is None:
        from unify.settings import SETTINGS as settings
    return settings


def enabled(settings: Any = None) -> bool:
    """Whether ``UNIFY_ADAPTIVE_POLICY=on``."""
    return _on(_settings(settings), SETTING)


def refusal(settings: Any = None) -> str | None:
    """Why this run may not start under the policy (settings named, never a value), or None."""
    settings = _settings(settings)
    if not enabled(settings):
        return None
    if str(getattr(settings, "UNIFY_TOOLS_IN_CODE", "") or "").strip().lower() != "on":
        return f"{SETTING}=on needs UNIFY_TOOLS_IN_CODE=on; nothing runs"
    if str(getattr(settings, "UNIFY_MEMORY", "") or "").strip().lower() == "on":
        return f"{SETTING}=on chooses among the drafts arms, so UNIFY_MEMORY must be off; nothing runs"
    from unify.actor import drafts

    # UNIFY_DRAFTS and UNIFY_HANDBACK_DECISION count as set in any of their modes
    set_by_hand = {
        "UNIFY_DRAFTS": drafts.mode(settings) != "",
        "UNIFY_HANDBACK_DECISION": drafts.decision_mode(settings) != "",
        "UNIFY_CONSEQUENCES": _on(settings, "UNIFY_CONSEQUENCES"),
    }
    explicit = [n for n, on in set_by_hand.items() if on]
    if explicit:
        return f"{SETTING}=on chooses {', '.join(explicit)} itself; unset them, so nothing runs"
    return None


def source_hash(source: str) -> str:
    """A definition's normalised source (its syntax tree, so layout and comments do not count), hashed."""
    try:
        text = ast.dump(ast.parse(source))
    except (SyntaxError, ValueError):
        text = " ".join(source.split())
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


def progress_of(outcome: Optional[dict]) -> Optional[float]:
    """A posted outcome's progress in [0, 1] from its typed fields only: ``solved``; else a ``score`` in [0, 1]; else
    the passed share of its checks. None when it carries none of them."""
    if not isinstance(outcome, dict):
        return None
    solved = outcome.get("solved")
    if isinstance(solved, bool):
        return 1.0 if solved else 0.0
    score = outcome.get("score")
    if isinstance(score, (int, float)) and not isinstance(score, bool) and 0.0 <= float(score) <= 1.0:
        return float(score)
    total, passed = outcome.get("checks_total"), outcome.get("checks_passed")
    if isinstance(total, int) and isinstance(passed, int) and total > 0 and 0 <= passed <= total:
        return passed / total
    checks = outcome.get("checks")
    if isinstance(checks, list):
        judged = [c.get("passed") for c in checks if isinstance(c, dict) and isinstance(c.get("passed"), bool)]
        if judged:
            return sum(judged) / len(judged)
    return None


class Learner:
    """Discounted Gaussian Thompson sampling over :data:`ARMS`: per arm the discounted weight, sum and sum of squares
    of its rewards; every update discounts every arm by :data:`GAMMA`."""

    def __init__(self, stats: Optional[list] = None) -> None:
        ok = isinstance(stats, list) and len(stats) == len(ARMS) and all(isinstance(s, dict) for s in stats)
        self.stats = [dict(s) for s in stats] if ok else [{"w": 0.0, "s": 0.0, "q": 0.0} for _ in ARMS]

    def update(self, arm: int, reward: float) -> None:
        for s in self.stats:
            s["w"], s["s"], s["q"] = GAMMA * s["w"], GAMMA * s["s"], GAMMA * s["q"]
        s = self.stats[arm]
        s["w"] += 1.0
        s["s"] += reward
        s["q"] += reward * reward

    def posterior(self, arm: int) -> tuple[float, float]:
        """(mean, variance of the mean) under the flat prior."""
        s = self.stats[arm]
        w = s["w"] + PRIOR_WEIGHT
        mean = s["s"] / w
        spread = (s["q"] + PRIOR_WEIGHT * PRIOR_VARIANCE) / w - mean * mean
        return mean, max(spread, 1e-6) / w

    def sample(self, rng: random.Random) -> list[float]:
        out = []
        for arm in range(len(ARMS)):
            mean, var = self.posterior(arm)
            out.append(rng.gauss(mean, var**0.5))
        return out

    def choose(self, rng: random.Random) -> tuple[int, float]:
        """An arm by one posterior draw, and its propensity: the share of :data:`DRAWS` draws it wins."""
        draws = self.sample(rng)
        arm = max(range(len(ARMS)), key=lambda i: draws[i])
        wins = 0
        for _ in range(DRAWS):
            more = self.sample(rng)
            wins += max(range(len(ARMS)), key=lambda i: more[i]) == arm
        return arm, wins / DRAWS


class PolicyState:
    """The deployment's ``policy-state.json``: read whole, replaced whole (written new, then renamed)."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def load(self) -> dict:
        try:
            data = json.loads(self.path.read_text())
        except (OSError, ValueError):
            data = {}
        if not isinstance(data, dict):
            data = {}
        data.setdefault("version", 1)
        data.setdefault("stats", None)
        data.setdefault("costs", [])
        data.setdefault("sessions", {})
        data.setdefault("shadow", {})
        return data

    def save(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".policy-state-", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w") as fh:
                json.dump(data, fh, sort_keys=True)
            os.replace(tmp, self.path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise


class SessionPolicy:
    """One session's arm, chosen at its start, and its rewards (immediate at its end, delayed to earlier sessions)."""

    def __init__(self, home: Path, session_id: str, events: Any, rng: Optional[random.Random] = None) -> None:
        self.state = PolicyState(Path(home) / STATE)
        self.session_id = session_id
        self.events = events
        self.rng = rng or random.Random()
        self.outcome: Optional[dict] = None
        self._credited: set = set()
        self._penalised: set = set()
        self._hashes: dict[str, str] = {}  # this session's definitions: hash -> name
        data = self.state.load()
        learner = Learner(data["stats"])
        self.arm, propensity = learner.choose(self.rng)
        data["stats"] = learner.stats
        data["sessions"][session_id] = {"arm": self.arm}
        self._save(data)
        self._event(
            {
                "type": "policy_choice",
                "session": session_id,
                "arm": self.arm,
                "knobs": dict(ARMS[self.arm]),
                "propensity": propensity,
                "means": [round(learner.posterior(i)[0], 6) for i in range(len(ARMS))],
                "weights": [round(s["w"], 6) for s in learner.stats],
            },
        )

    @property
    def knobs(self) -> dict:
        return dict(ARMS[self.arm])

    def _save(self, data: dict) -> None:
        try:
            self.state.save(data)
        except Exception:  # noqa: BLE001 - the policy never ends a request
            pass

    def _event(self, row: dict) -> None:
        try:
            self.events.append(row)
        except Exception:  # noqa: BLE001
            pass

    def _credit(self, data: dict, arm: int, reward: float, why: str, **extra: Any) -> None:
        learner = Learner(data["stats"])
        learner.update(arm, reward)
        data["stats"] = learner.stats
        self._event({"type": "policy_credit", "session": self.session_id, "arm": arm, "reward": reward, "why": why,
                     **extra})  # fmt: skip

    def take_outcome(self, outcome: dict) -> None:
        """The session's posted outcome (the latest wins)."""
        self.outcome = outcome

    def imported(self, module: str, writer: Optional[dict]) -> None:
        """A cell imported the draft *module*, which *writer* (``{"session", "arm"}`` from the draft record) wrote:
        the writer's arm gets the delayed reuse credit, once per draft in this session, never for its own."""
        if not isinstance(writer, dict) or writer.get("session") in (None, self.session_id) or module in self._credited:
            return
        arm = writer.get("arm")
        if not isinstance(arm, int) or not 0 <= arm < len(ARMS):
            return
        self._credited.add(module)
        data = self.state.load()
        self._credit(data, arm, REUSE, "reuse", draft=module, writer=writer.get("session"))
        self._save(data)

    def defined(self, defs: dict) -> None:
        """At a hand-back, the session's definitions (name -> source): one an earlier session made and did not keep
        is rework, debited to that session's arm (once per definition in this session); each is then this session's."""
        hashes = {source_hash(src): name for name, src in defs.items() if isinstance(src, str)}
        if not hashes:
            return
        data = self.state.load()
        shadow = data["shadow"]
        for h, name in hashes.items():
            seen = shadow.get(h)
            if (
                isinstance(seen, dict)
                and seen.get("session") != self.session_id
                and not seen.get("kept")
                and h not in self._penalised
                and isinstance(seen.get("arm"), int)
                and 0 <= seen["arm"] < len(ARMS)
            ):
                self._penalised.add(h)
                self._credit(data, seen["arm"], -REWORK, "rework", name=name, maker=seen.get("session"))
        self._hashes.update(hashes)
        self._save(data)

    def close(self, calls: Optional[int], kept: set) -> None:
        """The session's end: its own reward from its progress and cost, and its definitions into the shadow record
        (*kept*: the names its draft still holds; the others vanish with the interpreter)."""
        data = self.state.load()
        costs = [c for c in data["costs"] if isinstance(c, (int, float)) and c >= 0]
        progress = progress_of(self.outcome)
        reward = progress or 0.0
        if calls is not None:
            ref = statistics.median(costs) if costs else float(max(calls, 1))
            reward -= calls / max(ref, 1.0)
            costs.append(calls)
        data["costs"] = costs[-COST_WINDOW:]
        learner = Learner(data["stats"])
        learner.update(self.arm, reward)
        data["stats"] = learner.stats
        shadow = data["shadow"]
        for h, name in self._hashes.items():
            shadow.pop(h, None)
            shadow[h] = {"session": self.session_id, "arm": self.arm, "kept": name in kept}
        while len(shadow) > SHADOW_MAX:
            shadow.pop(next(iter(shadow)))
        self._save(data)
        self._event(
            {
                "type": "policy_outcome",
                "session": self.session_id,
                "arm": self.arm,
                "calls": calls,
                "progress": progress,
                "progress_from": "outcome" if progress is not None else "none",
                "reward": reward,
                "means": [round(learner.posterior(i)[0], 6) for i in range(len(ARMS))],
            },
        )


_CURRENT: dict[str, SessionPolicy] = {}


def register(policy: SessionPolicy) -> None:
    """The process's live actor session policy (a posted outcome reaches it through :func:`take_outcome`)."""
    _CURRENT["actor"] = policy


def unregister(policy: SessionPolicy) -> None:
    if _CURRENT.get("actor") is policy:
        _CURRENT.pop("actor", None)


def take_outcome(outcome: dict) -> bool:
    """Give the live session's policy a posted (normalised) outcome; False when there is none."""
    policy = _CURRENT.get("actor")
    if policy is None:
        return False
    policy.take_outcome(outcome)
    return True

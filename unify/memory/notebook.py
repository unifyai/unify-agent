"""The world notebook (``UNIFY_MEMORY_NOTEBOOK=on``; off by default, and then nothing here runs).

Memory for worlds whose rules hold across episodes: what it keeps is a *lesson about the world* ("in this
situation, this action was followed by that consequence"), and what makes a lesson worth showing is a count of
outcomes across episodes, never one episode's account of itself. Three parts, each world-general (nothing here
knows a world, a task or a word):

* **Moments** (:class:`MomentDetector`). After each episode the harness marks the steps whose consequences it can
  observe, from the record alone: the episode's end; a bounded quantity the observations show (a ``name: a/b``
  line) falling faster than its own drift, or falling at all once the run's episodes have ended with it far below
  its usual level; an action that changed none of the quantities its observation shows although the same action
  usually changes one; and an action whose call failed. Drift, end levels and each action's usual effect are
  learned from this run's own earlier steps (:data:`MIN_HISTORY` steps before drift is trusted). The moments only
  point the episode-end fork at steps to explain (:func:`fork_moments_text`); they are never shown as rules.
* **Lessons as rates** (:class:`Notebook`). The writer records lessons ``{situation, action, consequence,
  alternative}`` and, per episode it reads, how often the situation and action occurred and how often the
  consequence followed, and the same for the alternative action (``/memory/.pass/notebook.json``,
  :func:`absorb`). The harness keeps the arithmetic: per lesson the summed rate ``followed / occurrences``, the
  alternative's rate, and per episode a verdict by majority (supported when the consequence followed in more than
  half of that episode's occurrences, contradicted otherwise). A stochastic consequence stays a rate ("in 27 of 30")
  rather than a true/false rule.
* **Trials.** A lesson the writer marks ``trial`` is a remedy proposed for a recurring failure and not yet tried;
  it is shown labelled as such until its own counts establish or drop it.

Delivery (:func:`render_section`): established lessons (support ≥ :data:`ESTABLISHED_SUPPORT` episodes and more
support than contradiction) and open trials are appended to the actor's memory section at every request start;
weak lessons are not pushed (the writer's notes in the library remain the pull path).

The store is ``<state>/notebook/``: ``lessons.json`` (lessons and judgements), ``detector.json`` (the moment
detector's learned state) and ``moments/<episode>.json``. Every write is atomic (a temporary file renamed in place).
"""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

NOTEBOOK = "UNIFY_MEMORY_NOTEBOOK"
DIR = "notebook"
LESSONS_FILE = "lessons.json"
DETECTOR_FILE = "detector.json"
MOMENTS_DIR = "moments"
#: what the writer writes in its box (beside the manifest; never committed)
WRITER_FILE = ".pass/notebook.json"
#: what the writer reads (the pass's /inputs)
INPUT_FILE = "notebook.json"

#: an action is "usually effective" once it changed a shown quantity in this share of at least NOCHANGE_MIN_USES uses
NOCHANGE_SHARE = 0.3
NOCHANGE_MIN_USES = 10
#: steps of a quantity's history before its drift is trusted (until then every fall of a bounded quantity counts)
MIN_HISTORY = 50
#: the recent changes kept per quantity for its usual step size
RECENT_CHANGES = 64
#: a bounded quantity is end-linked when episodes end, on average, with its last shown value at least this share of
#: its range below its own mean over that episode
END_DEFICIT = 1 / 3
#: a lesson is established at this many supporting episodes and more support than contradiction
ESTABLISHED_SUPPORT = 2
#: a lesson (or trial) is dropped from view at this many contradicting episodes and more contradiction than support
DROPPED_CONTRADICT = 2

_QTY_LINE = re.compile(
    r"^[ \t]*(?:[-*][ \t]*)?([A-Za-z_][A-Za-z0-9_ ]{0,63}?)[ \t]*:[ \t]*(-?\d+(?:\.\d+)?)"
    r"(?:[ \t]*/[ \t]*(\d+(?:\.\d+)?))?[ \t]*$",
    re.M,
)
_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_TEXT_MAX = 600
_COUNT_MAX = 1_000_000
_JSON_DEPTH = 6
_JSON_LEAVES = 256


def notebook_on(settings: Any = None) -> bool:
    """``UNIFY_MEMORY_NOTEBOOK`` is ``on`` (off when it cannot be read)."""
    if settings is None:
        try:
            from unify.settings import SETTINGS as settings
        except Exception:  # noqa: BLE001 - no settings: off
            return False
    return str(getattr(settings, NOTEBOOK, "off") or "off").strip().lower() == "on"


def root(state_dir: Path | str) -> Path:
    return Path(state_dir) / DIR


# --- quantities ---------------------------------------------------------------------------------------------------


def _num(text: str) -> float:
    v = float(text)
    return int(v) if v.is_integer() else v


def _json_leaves(value: Any, prefix: str, out: dict, depth: int) -> None:
    if len(out) >= _JSON_LEAVES or depth > _JSON_DEPTH:
        return
    if isinstance(value, bool) or value is None:
        return
    if isinstance(value, (int, float)):
        if math.isfinite(value):
            out[prefix or "value"] = (value, None)
        return
    if isinstance(value, dict):
        if "__blob__" in value:
            return
        for k, v in value.items():
            _json_leaves(v, f"{prefix}.{k}" if prefix else str(k), out, depth + 1)
    elif isinstance(value, list):
        for i, v in enumerate(value):
            _json_leaves(v, f"{prefix}[{i}]", out, depth + 1)


def quantities(observation: Any) -> dict[str, tuple[float, float | None]]:
    """The numbers an observation shows, by name: ``{name: (value, bound)}``. In text, each line of the form
    ``name: n`` (bound None) or ``name: a/b`` (bound b), optionally after a list marker; in a JSON observation (a
    dict or a list, or text that is one JSON object), each numeric leaf by its key path (bound None). Structure
    only: no name is looked for."""
    if isinstance(observation, (dict, list)):
        out: dict = {}
        _json_leaves(observation, "", out, 0)
        return out
    if not isinstance(observation, str):
        return {}
    text = observation.strip()
    if text[:1] in "{[":
        try:
            parsed = json.loads(text)
        except ValueError:
            parsed = None
        if isinstance(parsed, (dict, list)):
            return quantities(parsed)
    out = {}
    for m in _QTY_LINE.finditer(observation):
        name = " ".join(m.group(1).split())
        out[name] = (_num(m.group(2)), _num(m.group(3)) if m.group(3) else None)
    return out


# --- steps of an episode ------------------------------------------------------------------------------------------


@dataclass
class Step:
    """One action of an episode as the detector reads it: its part name, a key that is the same for the same action
    (channel, method and arguments), the channel whose observations it is compared within, the observation it got
    back, and whether its call failed."""

    part: str
    key: str
    channel: str
    observation: Any
    failed: bool = False
    error: str = ""


def action_key(channel: str, method: str, args: Any = (), kwargs: Any = None) -> str:
    try:
        arg_text = json.dumps(
            [list(args or []), kwargs or {}],
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        )
    except (TypeError, ValueError):
        arg_text = repr((args, kwargs))
    return f"{channel}.{method}{arg_text[:200]}"


def steps_of(ep: Any) -> tuple[list[Step], dict[str, Any]]:
    """(the episode's actions as :class:`Step` rows, the observation each channel starts from). A dialogue
    counterpart's first message is the request itself, so a dialogue channel starts from ``ep.request[0]``; any
    other channel starts with nothing to compare."""
    steps: list[Step] = []
    start: dict[str, Any] = {}
    request = list(getattr(ep, "request", []) or [])
    for i, a in enumerate(getattr(ep, "actions", []) or []):
        channel = f"{a.kind}:{a.channel}"
        if a.kind == "dialogue" and channel not in start and request:
            start[channel] = request[0]
        status = str(getattr(a, "status", "") or "")
        err = getattr(a, "error", None)
        steps.append(
            Step(
                part=f"action:{i}",
                key=action_key(a.channel, a.method, a.args, a.kwargs),
                channel=channel,
                observation=a.response,
                failed=status == "error" or bool(err),
                error=str(err or "")[:200],
            ),
        )
    return steps, start


# --- the moment detector ------------------------------------------------------------------------------------------


def _median(values: list[float]) -> float:
    s = sorted(values)
    if not s:
        return 0.0
    mid = len(s) // 2
    return s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2


class MomentDetector:
    """The consequence-moment detector and its state learned from the run's earlier steps (JSON, :meth:`state`).

    Kinds of moment, each a dict with ``part`` and ``kind``:

    * ``end``: the episode's last action;
    * ``fall``: a bounded quantity fell (``field``, ``before``, ``after``, ``bound``, ``why``). It counts when the
      run has not yet trusted the quantity's drift (``why`` ``"early"``), when the run's episodes have ended with the
      quantity far below its usual level (``"ended-low"``: on average at least :data:`END_DEFICIT` of its range below
      its mean over the episode), or when the fall is larger than its drift predicts over the steps
      since it last changed plus its usual step (``"faster-than-drift"``);
    * ``no-change``: the action changed no quantity of its channel's observation although it changed one in at
      least :data:`NOCHANGE_SHARE` of its :data:`NOCHANGE_MIN_USES` or more earlier uses (``uses``, ``changed``);
    * ``failed``: the action's call failed (``error``).
    """

    def __init__(self, state: Mapping | None = None) -> None:
        state = dict(state or {})
        self.fields: dict[str, dict] = {
            k: dict(v) for k, v in (state.get("fields") or {}).items()
        }
        self.actions: dict[str, list[int]] = {
            k: [int(v[0]), int(v[1])] for k, v in (state.get("actions") or {}).items()
        }
        self.episodes = int(state.get("episodes") or 0)

    def state(self) -> dict:
        return {
            "version": 1,
            "episodes": self.episodes,
            "fields": self.fields,
            "actions": self.actions,
        }

    def _field(self, name: str) -> dict:
        return self.fields.setdefault(
            name,
            {
                "steps": 0,
                "mean_change": 0.0,
                "recent": [],
                "ended": 0,
                "end_deficit": 0.0,
            },
        )

    @staticmethod
    def end_linked(f: Mapping) -> bool:
        ended = int(f.get("ended") or 0)
        return ended > 0 and float(f.get("end_deficit") or 0.0) / ended >= END_DEFICIT

    def _fall_reason(self, f: dict, fall: float, gap: int, bounded: bool) -> str | None:
        if self.end_linked(f):
            return "ended-low"
        if f["steps"] < MIN_HISTORY:
            return "early" if bounded else None
        if not bounded:
            return None
        drift = max(0.0, -float(f["mean_change"])) * max(1, gap)
        usual = _median([abs(x) for x in f["recent"] if x]) if f["recent"] else 0.0
        return "faster-than-drift" if fall > drift + usual else None

    def episode(
        self, steps: Iterable[Step], start: Mapping[str, Any] | None = None
    ) -> list[dict]:
        """The moments of one episode, in step order; learns from it afterwards (each step only from the steps
        before it)."""
        steps = list(steps)
        prev: dict[str, dict] = {
            ch: quantities(obs) for ch, obs in (start or {}).items()
        }
        last_change: dict[tuple[str, str], int] = {}
        last_seen: dict[str, tuple[float, float | None]] = {}
        level: dict[str, list[float]] = (
            {}
        )  # per quantity this episode: sum and count of its shown values
        moments: list[dict] = []
        for t, s in enumerate(steps):
            q = quantities(s.observation)
            p = prev.get(s.channel)
            if s.failed:
                moments.append({"part": s.part, "kind": "failed", "error": s.error})
            if q and p:
                changed = any(q.get(n) != p.get(n) for n in set(q) | set(p))
                for name in sorted(set(q) & set(p)):
                    (v, bound), (pv, pbound) = q[name], p[name]
                    bound = bound if bound is not None else pbound
                    delta = v - pv
                    f = self._field(name)
                    gap = t - last_change.get((s.channel, name), -1)
                    if delta < 0:
                        why = self._fall_reason(f, -delta, gap, bound is not None)
                        if why:
                            moments.append(
                                {
                                    "part": s.part,
                                    "kind": "fall",
                                    "field": name,
                                    "before": pv,
                                    "after": v,
                                    "bound": bound,
                                    "why": why,
                                },
                            )
                    if delta:
                        last_change[(s.channel, name)] = t
                        f["recent"] = (f["recent"] + [delta])[-RECENT_CHANGES:]
                    f["steps"] += 1
                    f["mean_change"] += (delta - f["mean_change"]) / f["steps"]
                uses = self.actions.setdefault(s.key, [0, 0])
                if (
                    not changed
                    and uses[0] >= NOCHANGE_MIN_USES
                    and uses[1] >= NOCHANGE_SHARE * uses[0]
                ):
                    moments.append(
                        {
                            "part": s.part,
                            "kind": "no-change",
                            "uses": uses[0],
                            "changed": uses[1],
                        }
                    )
                uses[0] += 1
                uses[1] += int(changed)
            if q:
                prev[s.channel] = q
                for name, vb in q.items():
                    last_seen[name] = vb
                    acc = level.setdefault(name, [0.0, 0])
                    acc[0] += vb[0]
                    acc[1] += 1
        if steps:
            moments.append({"part": steps[-1].part, "kind": "end"})
        # learn the level each bounded quantity ended at: its last shown value against its mean over the episode
        for name, (v, bound) in last_seen.items():
            if bound is None or bound <= 0:
                continue
            f = self._field(name)
            total, count = level[name]
            f["ended"] = int(f.get("ended") or 0) + 1
            f["end_deficit"] = float(f.get("end_deficit") or 0.0) + max(
                0.0, (total / count - v) / bound
            )
        self.episodes += 1
        return moments


def _fmt(v: Any) -> str:
    return str(int(v)) if isinstance(v, float) and v.is_integer() else str(v)


def moment_line(m: Mapping) -> str:
    """One moment as the fork reads it (facts only: what changed, never what it means)."""
    kind = m.get("kind")
    if kind == "fall":
        bound = m.get("bound")
        tail = f"/{_fmt(bound)}" if bound is not None else ""
        why = {
            "early": "",
            "ended-low": "; earlier episodes ended with it far below its usual level",
            "faster-than-drift": "; faster than its usual drift",
        }.get(m.get("why"), "")
        return f"`{m['part']}`: {m['field']} fell from {_fmt(m['before'])}{tail} to {_fmt(m['after'])}{tail}{why}"
    if kind == "no-change":
        return (
            f"`{m['part']}`: the action changed none of the numbers shown, though the same action changed one in "
            f"{m['changed']} of its {m['uses']} earlier uses"
        )
    if kind == "failed":
        err = f" ({m['error']})" if m.get("error") else ""
        return f"`{m['part']}`: the call failed{err}"
    return f"`{m['part']}`: the episode's last action; what followed is the end of the record"


#: The paragraph before the moment list in the fork's message (``UNIFY_MEMORY_NOTEBOOK=on`` only).
MOMENTS_HEAD = (
    "Moments the harness marked in this record, from changes it could observe (a shown quantity falling, an action "
    "that changed nothing it usually changes, a failed call, the end). They are pointers, not conclusions. In "
    "`notes.md`, for each moment whose consequence matters for later episodes, write one lesson: the situation as "
    "the record showed it just before (only what was visible), the action taken, and the consequence, with the part "
    "names. Where the same action in a similar situation had a different consequence, or another action was taken "
    "there, say so with its parts too. If a failure recurs and you can think of a remedy that was not tried here, "
    "write it as a proposed remedy, saying it is untried."
)


def fork_moments_text(moments: list[dict]) -> str:
    """The moment list the fork's message ends with: :data:`MOMENTS_HEAD` and one line per moment, runs of the same
    quantity falling on consecutive actions folded into one line (first and last value).
    """
    lines: list[str] = []
    i = 0
    while i < len(moments):
        m = moments[i]
        j = i
        if m.get("kind") == "fall":
            while (
                j + 1 < len(moments)
                and moments[j + 1].get("kind") == "fall"
                and moments[j + 1].get("field") == m.get("field")
                and _index(moments[j + 1]["part"]) == _index(moments[j]["part"]) + 1
            ):
                j += 1
        if j > i:
            last = moments[j]
            tail = f"/{_fmt(m['bound'])}" if m.get("bound") is not None else ""
            lines.append(
                f"- `{m['part']}` to `{last['part']}`: {m['field']} fell on each action, from {_fmt(m['before'])}"
                f"{tail} to {_fmt(last['after'])}{tail}",
            )
        else:
            lines.append("- " + moment_line(m))
        i = j + 1
    return MOMENTS_HEAD + "\n" + "\n".join(lines) if lines else ""


def _index(part: str) -> int:
    try:
        return int(str(part).rsplit(":", 1)[1])
    except (IndexError, ValueError):
        return -10


# --- atomic files -------------------------------------------------------------------------------------------------


def _write_json(path: Path, doc: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, sort_keys=True, ensure_ascii=False)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _read_json(path: Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def episode_moments(state_dir: Path | str, ep: Any) -> list[dict]:
    """The moments of recorded episode *ep*, with the run's detector state read from and written back to
    ``<state>/notebook/detector.json``; the moments are kept as ``moments/<episode>.json`` for the writer.
    """
    base = root(state_dir)
    det = MomentDetector(_read_json(base / DETECTOR_FILE))
    steps, start = steps_of(ep)
    moments = det.episode(steps, start)
    _write_json(base / DETECTOR_FILE, det.state())
    eid = str(getattr(ep, "episode_id", "") or "")
    if eid and "/" not in eid and eid not in (".", ".."):
        _write_json(base / MOMENTS_DIR / f"{eid}.json", moments)
    return moments


def read_moments(state_dir: Path | str, eid: str) -> list[dict] | None:
    if not eid or "/" in eid or eid in (".", ".."):
        return None
    doc = _read_json(root(state_dir) / MOMENTS_DIR / f"{eid}.json")
    return doc if isinstance(doc, list) else None


# --- lessons and their counts -------------------------------------------------------------------------------------

LESSON_TEXT_FIELDS = ("situation", "action", "consequence", "alternative")


@dataclass
class Tally:
    """A lesson's counts: episodes that supported or contradicted it (majority within the episode), occurrences
    and how often the consequence followed, and the same for the alternative action."""

    support: int = 0
    contradict: int = 0
    occurrences: int = 0
    followed: int = 0
    alt_occurrences: int = 0
    alt_followed: int = 0
    episodes: list = field(default_factory=list)

    @property
    def status(self) -> str:
        if self.contradict >= DROPPED_CONTRADICT and self.contradict > self.support:
            return "dropped"
        if self.support >= ESTABLISHED_SUPPORT and self.support > self.contradict:
            return "established"
        return "weak"


class Notebook:
    """The lessons and the writer's per-episode judgements (``lessons.json``); the counts are arithmetic over the
    judgements (:meth:`tally`), never stored, so a re-judged episode replaces its earlier judgement.
    """

    def __init__(self, doc: Mapping | None = None) -> None:
        doc = dict(doc or {})
        self.lessons: dict[str, dict] = {
            k: dict(v) for k, v in (doc.get("lessons") or {}).items()
        }
        self.judgements: dict[str, dict[str, dict]] = {
            k: {e: dict(j) for e, j in v.items()}
            for k, v in (doc.get("judgements") or {}).items()
        }

    def doc(self) -> dict:
        return {"version": 1, "lessons": self.lessons, "judgements": self.judgements}

    def add_lesson(
        self,
        lesson_id: str,
        *,
        situation: str,
        action: str,
        consequence: str,
        alternative: str = "",
        trial: bool = False,
        remedy_for: str = "",
        origin: str = "",
    ) -> None:
        old = self.lessons.get(lesson_id, {})
        self.lessons[lesson_id] = {
            "situation": situation,
            "action": action,
            "consequence": consequence,
            "alternative": alternative,
            "trial": bool(trial),
            "remedy_for": remedy_for,
            "origin": old.get("origin") or origin,
            "retired": old.get("retired"),
        }

    def judge(
        self,
        lesson_id: str,
        episode: str,
        *,
        occurrences: int,
        followed: int,
        alt_occurrences: int = 0,
        alt_followed: int = 0,
        parts: Iterable[str] = (),
        origin: str = "",
    ) -> None:
        self.judgements.setdefault(lesson_id, {})[episode] = {
            "occurrences": int(occurrences),
            "followed": int(followed),
            "alt_occurrences": int(alt_occurrences),
            "alt_followed": int(alt_followed),
            "parts": [str(p)[:40] for p in list(parts)[:20]],
            "origin": origin,
        }

    def retire(self, lesson_id: str, reason: str) -> None:
        if lesson_id in self.lessons:
            self.lessons[lesson_id]["retired"] = reason[:_TEXT_MAX] or "retired"

    def tally(self, lesson_id: str) -> Tally:
        t = Tally()
        for eid, j in sorted(self.judgements.get(lesson_id, {}).items()):
            n, k = int(j.get("occurrences", 0)), int(j.get("followed", 0))
            t.alt_occurrences += int(j.get("alt_occurrences", 0))
            t.alt_followed += int(j.get("alt_followed", 0))
            if n <= 0:
                continue
            t.occurrences += n
            t.followed += k
            t.episodes.append(eid)
            if 2 * k > n:
                t.support += 1
            else:
                t.contradict += 1
        return t

    def rows(self) -> list[dict]:
        """Every lesson with its counts and status (``established``, ``weak``, ``dropped`` or ``retired``)."""
        out = []
        for lid, lesson in sorted(self.lessons.items()):
            t = self.tally(lid)
            status = "retired" if lesson.get("retired") else t.status
            out.append(
                {
                    "id": lid,
                    **{k: lesson.get(k, "") for k in LESSON_TEXT_FIELDS},
                    "trial": bool(lesson.get("trial")),
                    "remedy_for": lesson.get("remedy_for", ""),
                    "status": status,
                    "support": t.support,
                    "contradict": t.contradict,
                    "occurrences": t.occurrences,
                    "followed": t.followed,
                    "alt_occurrences": t.alt_occurrences,
                    "alt_followed": t.alt_followed,
                    "episodes": len(t.episodes),
                },
            )
        return out


def load(state_dir: Path | str) -> Notebook:
    doc = _read_json(root(state_dir) / LESSONS_FILE)
    return Notebook(doc if isinstance(doc, dict) else None)


def save(state_dir: Path | str, nb: Notebook) -> None:
    _write_json(root(state_dir) / LESSONS_FILE, nb.doc())


# --- what the actor is shown --------------------------------------------------------------------------------------

#: The heading of the pushed lessons (``UNIFY_MEMORY_NOTEBOOK=on`` only).
SECTION_HEAD = (
    "Notebook: what earlier episodes showed, counted by the harness from their records. Each line is a count, not a "
    "rule: where it says k of n, the other n - k times went differently."
)
TRIALS_HEAD = (
    "Proposed remedies for failures that recur, not yet established (trials). Their outcomes are counted when they "
    "are tried:"
)


def _times(k: int, n: int) -> str:
    return f"{k} of {n} time{'' if n == 1 else 's'}"


def lesson_line(row: Mapping) -> str:
    text = (
        f"- When {row['situation']}, {row['action']} was followed by {row['consequence']} in "
        f"{_times(row['followed'], row['occurrences'])} ({row['support']} episode"
        f"{'' if row['support'] == 1 else 's'} mostly so, {row['contradict']} mostly not)"
    )
    if row.get("alternative"):
        if row["alt_occurrences"]:
            text += f"; {row['alternative']} instead was followed by it in {_times(row['alt_followed'], row['alt_occurrences'])}"
        else:
            text += f"; {row['alternative']} instead: not yet seen"
    return text + "."


def trial_line(row: Mapping) -> str:
    text = (
        f"- Trial: when {row['situation']}, {row['action']}, for {row['consequence']}: "
    )
    if row["occurrences"]:
        text += f"tried {row['occurrences']} time{'' if row['occurrences'] == 1 else 's'}, {row['followed']} with that result"
    else:
        text += "not tried yet"
    if row.get("alternative") and row["alt_occurrences"]:
        text += f"; {row['alternative']} gave it in {_times(row['alt_followed'], row['alt_occurrences'])}"
    return text + "."


def render_section(nb: Notebook) -> str:
    """The pushed part of the notebook: established lessons, most supported first, then open trials (a trial that
    has become established is listed with the lessons). ``""`` when there is nothing to push.
    """
    rows = nb.rows()
    est = sorted(
        (r for r in rows if r["status"] == "established"),
        key=lambda r: (-r["support"], -r["occurrences"], r["id"]),
    )
    trials = [r for r in rows if r["trial"] and r["status"] == "weak"]
    if not est and not trials:
        return ""
    parts = [SECTION_HEAD]
    parts += [lesson_line(r) for r in est]
    if trials:
        parts.append(TRIALS_HEAD)
        parts += [trial_line(r) for r in trials]
    return "\n".join(parts) + "\n"


def render_for(state_dir: Path | str) -> str:
    """:func:`render_section` of the run's notebook; ``""`` when it cannot be read."""
    try:
        return render_section(load(state_dir))
    except Exception:  # noqa: BLE001 - the notebook never stops a request
        return ""


# --- the writer's side --------------------------------------------------------------------------------------------

#: The writer's brief section (``UNIFY_MEMORY_NOTEBOOK=on`` only; constant for the run).
WRITER_BRIEF = (
    "\n# Notebook\n"
    "Besides the library, you keep a notebook of lessons about the world the episodes take place in: in this "
    "situation, this action was followed by that consequence. The harness counts; you judge.\n"
    "- /inputs/notebook.json: `lessons`, every lesson so far with its counts (`support` and `contradict` are "
    "episodes where the consequence mostly followed or mostly did not; `occurrences` and `followed` are summed over "
    "episodes; `alt_occurrences` and `alt_followed` are the same for the alternative action; `status` is "
    "`established`, `weak`, `dropped` or `retired`); and `moments`, per batch episode, the steps the harness marked "
    "from changes it could observe (a shown quantity falling, an action that changed nothing it usually changes, a "
    "failed call, the end). Established lessons and open trials are shown to the working model at the start of "
    "every request, with their counts.\n"
    "- For each batch episode, judge each lesson its record tests: how many times the situation and the action "
    "occurred in it, how many of those times the consequence followed, and the same two numbers for the "
    "alternative action in the same situation. Count from the record (`read_episode`), not from a summary. An "
    "episode that never had the situation does not test the lesson: leave it out.\n"
    "- Add a lesson when a moment, a staged note or the record shows a consequence that matters, and judge it on "
    "the episodes that show it. State the situation by what was visible before the action, never by an episode, a "
    "request or an id. When a consequence follows only some of the time, keep the lesson and let the counts say "
    "how often; do not turn it into a rule. Revise a lesson's wording by writing it again under its id; retire "
    "one only when another lesson says the same (say which).\n"
    "- When a failure recurs and no lesson's action avoids it, you may propose a remedy that the records do not "
    "show was tried: a lesson with `trial` true, `remedy_for` the failing lesson's id, the remedy as its action "
    "and the outcome it aims for as its consequence. It is shown as a trial and counted like any lesson.\n"
    "- Write `/memory/.pass/notebook.json` (never committed; the harness reads it when the pass ends, whatever the "
    "gate decides): "
    '`{"lessons": [{"id": "<lowercase letters, digits, - or _>", "situation": "...", "action": "...", '
    '"consequence": "...", "alternative": "<another action, or empty>", "trial": false, "remedy_for": ""}], '
    '"judgements": [{"lesson": "<id>", "episode": "<batch episode id>", "occurrences": n, "followed": k, '
    '"alternative_occurrences": n2, "alternative_followed": k2, "parts": ["action:12"]}], '
    '"retire": {"<id>": "<the lesson that says the same>"}}`. '
    "List only lessons you add or revise; judgements only for this batch's episodes.\n"
)


def writer_input(state_dir: Path | str, episodes: Iterable[str]) -> dict:
    """``/inputs/notebook.json``: every lesson with its counts, and the batch episodes' moments."""
    nb = load(state_dir)
    return {
        "lessons": nb.rows(),
        "moments": {e: read_moments(state_dir, e) or [] for e in episodes},
    }


def _text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    value = " ".join(value.split())
    return value[:_TEXT_MAX] if value else None


def _count(value: Any) -> int | None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value <= _COUNT_MAX
    ):
        return None
    return value


def absorb(
    state_dir: Path | str, doc: Any, batch: Iterable[str], origin: str = ""
) -> dict:
    """Merge the writer's ``notebook.json`` *doc* into the run's notebook: new or revised lessons, judgements of
    *batch* episodes only (an episode re-judged replaces its earlier judgement), and retirements. Each malformed
    row is skipped and reported; the report is ``{"lessons", "judgements", "retired", "skipped": [reasons]}``.
    """
    report: dict = {"lessons": 0, "judgements": 0, "retired": 0, "skipped": []}
    if not isinstance(doc, dict):
        report["skipped"].append("notebook.json is not a JSON object")
        return report
    batch = set(batch)
    nb = load(state_dir)
    for row in doc.get("lessons") or []:
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("id"), str)
            or not _ID.match(row["id"])
        ):
            report["skipped"].append(f"lesson without a valid id: {str(row)[:80]}")
            continue
        texts = {k: _text(row.get(k)) for k in ("situation", "action", "consequence")}
        if None in texts.values():
            report["skipped"].append(
                f"lesson {row['id']}: situation, action and consequence are needed as text"
            )
            continue
        remedy_for = row.get("remedy_for") or ""
        if remedy_for and (
            not isinstance(remedy_for, str) or not _ID.match(remedy_for)
        ):
            remedy_for = ""
        nb.add_lesson(
            row["id"],
            **texts,
            alternative=_text(row.get("alternative")) or "",
            trial=row.get("trial") is True,
            remedy_for=remedy_for,
            origin=origin,
        )
        report["lessons"] += 1
    for row in doc.get("judgements") or []:
        if not isinstance(row, dict):
            report["skipped"].append("a judgement that is not an object")
            continue
        lid, eid = row.get("lesson"), row.get("episode")
        if lid not in nb.lessons:
            report["skipped"].append(f"judgement of an unknown lesson {str(lid)[:64]}")
            continue
        if eid not in batch:
            report["skipped"].append(
                f"judgement of {str(eid)[:64]}, not an episode of this batch"
            )
            continue
        n, k = _count(row.get("occurrences")), _count(row.get("followed"))
        n2 = _count(row.get("alternative_occurrences", 0))
        k2 = _count(row.get("alternative_followed", 0))
        if None in (n, k, n2, k2) or k > n or k2 > n2:
            report["skipped"].append(
                f"judgement {lid}/{eid}: counts must be whole numbers with followed <= occurrences"
            )
            continue
        parts = [p for p in row.get("parts") or [] if isinstance(p, str)]
        nb.judge(
            lid,
            eid,
            occurrences=n,
            followed=k,
            alt_occurrences=n2,
            alt_followed=k2,
            parts=parts,
            origin=origin,
        )
        report["judgements"] += 1
    retire = doc.get("retire") or {}
    if isinstance(retire, dict):
        for lid, reason in retire.items():
            if lid in nb.lessons and isinstance(reason, str):
                nb.retire(lid, reason)
                report["retired"] += 1
    save(state_dir, nb)
    report["skipped"] = report["skipped"][:50]
    return report

"""The writer's repair rounds: the round budget and the gate's full result as a file.

After the writer's accepted ``finish``, :class:`.writer_pass.WriterPass` runs :meth:`.gate.Gate.check` (the whole
gate, no merge) on the round's candidate. A refusal is written in full to ``/inputs/gate/result-<n>.md`` and the
writer gets another round while the pass's time and step guard hold one more (:func:`may_repair`); there is no round
count. Only the final candidate goes to :meth:`.gate.Gate.merge`, so per-item admission decides what lands. Nothing
here calls a model, git or the jail.

Stuck refusals: when the gate refuses the same item under the same check for the same reason in :data:`STUCK_REPEATS` consecutive rounds (:class:`StuckRefusals`), no further round is
offered for it; once every refusal of a round is stuck, no round opens and the pass goes to its final merge, whose
per-item admission lands the items that pass. A round whose candidate (its tree and manifest) did not change since
the last check repeats every refusal by construction.

Time (stated once, here): round 0 runs from the pass's start to the writer's first accepted ``finish``; round k >= 1
from the repair message that opens it to the next accepted ``finish``. A round's seconds are its wall time plus the
gate check that ends it. The time a next round needs is the median of the completed rounds' seconds plus the last
gate check's seconds, because the final :meth:`.gate.Gate.merge` runs the whole gate once more.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from .gate import GateResult
from .redact import redact_error

#: A refusal repeated in this many consecutive rounds is stuck: the first refusal, then two repair rounds that
#: ended with the same item refused for the same reason. A further round for it would most likely spend time and
#: money on the same result, so the pass stops offering one and the final merge keeps the item as a draft.
STUCK_REPEATS = 3
#: The item of a refusal no manifest item owns (it refuses the whole pass).
PASS_WIDE = "(pass-wide)"
def may_repair(
    done: int,
    remaining_s: float,
    calls_left: int,
    round_s: list[float],
    last_check_s: float = 0.0,
) -> str | None:
    """None when repair round ``done + 1`` may start; else why not, as one note line.

    The time it needs is the median of the completed rounds' seconds plus *last_check_s*, the last gate
    check's seconds, which the final merge spends again.
    """
    nxt = done + 1
    if calls_left < 1:
        return f"repair: no round {nxt}: the pass's call cap is reached"
    seconds = (float(statistics.median(round_s)) if round_s else 0.0) + max(
        0.0,
        float(last_check_s),
    )
    if remaining_s < seconds:
        return (
            f"repair: no round {nxt}: {remaining_s:.1f} s left, "
            f"below the round time reserve of {seconds:.1f} s (a round and the final merge's gate run)"
        )
    return None


def render_result(
    pass_id: str,
    n: int,
    res: GateResult,
    *,
    final: bool,
    candidate: str | None = None,
    pass_notes: Iterable[str] = (),
) -> str:
    """The gate's full result for round *n* as markdown: every check line and note, whole.

    Lines starting ``note: `` are the gate's notes; every other reason is a check line (``item refused:`` and
    ``reduced:`` lines of a reduced merge included). *pass_notes* are the pass's own notes (deadline, unpriced
    calls, repair decisions), given for a final result. Every string passes :func:`.redact.redact_error`.
    """
    reasons = [redact_error(r) for r in res.reasons if not r.startswith("note: ")]
    notes = [redact_error(r) for r in res.reasons if r.startswith("note: ")]
    own = [redact_error(r) for r in pass_notes]
    kind = "final (merge)" if final else "check (not merged)"
    lines = [
        f"# Gate result: pass {pass_id}, round {n}, {kind}",
        "",
        f"- passed: {'yes' if res.passed else 'no'}",
        f"- candidate: {candidate or 'none'}",
        f"- refused checks: {', '.join(res.refused) or 'none'}",
    ]
    if final:
        lines.append(f"- items merged: {', '.join(res.items_merged) or 'none'}")
    if res.items_refused:
        lines.append("- items refused:")
        lines += [
            f"  - {i}: {', '.join(c or [])}"
            for i, c in sorted(res.items_refused.items())
        ]
    else:
        lines.append("- items refused: none")
    for title, body in (("Reasons", reasons), ("Notes", notes), ("Pass notes", own)):
        lines += ["", f"## {title} ({len(body)})"]
        for r in body:
            lines += ["", r]
    levels = dict(getattr(res, "levels", None) or {})
    if levels:
        found = getattr(res, "level_reasons", None) or {}
        lines += [
            "",
            f"## Evidence level ({len(levels)})",
            "",
            "Each function's evidence level. These are not refusals: an item lands at its level, and a later pass "
            "can raise it by adding the evidence. The findings that set a level to `unchecked` are the notes that "
            "start `evidence level:`.",
            "",
        ]
        lines += [
            f"- {item}: {level} ({len(found.get(item, []))} findings)"
            for item, level in sorted(levels.items())
        ]
    return "\n".join(lines) + "\n"


def repair_message(
    n: int,
    path: str,
    res: GateResult,
    stuck: Sequence[tuple[str, str, str]] = (),
    repeats: int = STUCK_REPEATS,
) -> str:
    """The writer's message opening repair round *n* (>= 1) after round ``n - 1``'s refused check. No round count is
    stated: only the deadline and the step guard end the pass. *stuck* lists the refusals (check, item, reason) no further
    round is offered for (:class:`StuckRefusals`); the message names them and says why."""
    items = (
        ", ".join(
            f"{i} ({', '.join(c or [])})" for i, c in sorted(res.items_refused.items())
        )
        or "none named; the refusal is pass-wide"
    )

    # with tools called from code, the two calls are named as the writer makes them
    read, finish = (
        ("tools.read(path)", "tools.finish(summary)")
    )
    return (
        f"The gate refused round {n - 1}'s candidate (checks: {', '.join(res.refused) or 'none'}; items: "
        f"{items}). Its full result, with every check line, every failing test's output and every note, is "
        f"{path}; read it with {read}. Fix what it refused in /memory, keep /memory/.pass/manifest.json up to "
        f"date, then call {finish} again. "
        + f"This is repair round {n}. When the pass ends, the items that pass land and the rest are kept as a draft "
        "for the next pass."
        + (
            f" No further repair round is offered for {_pairs(stuck)}: the gate refused "
            f"{'it' if len(stuck) == 1 else 'each'} for the same reason in {repeats} rounds in a row ("
            + "; ".join(redact_error(r)[:200] for _, _, r in stuck)
            + "). The pass opens no round for these alone: the final merge keeps what it refuses as a draft with "
            "its failure, so this round is for the other refusals."
            if stuck
            else ""
        )
    )


# --- stuck refusals -----------------------------------------------------------------------------------------

_TMP_PATH = re.compile(r"/tmp/\S*")
_HEX_ID = re.compile(r"\b[0-9a-f]{12,}\b")
_SECONDS = re.compile(r"\b\d+(?:\.\d+)?\s?s\b")
_CHECK = re.compile(r"(G[1-6]): ")


def refusal_reason(line: str) -> str:
    """A refusal line's comparable text: its first line, with temporary paths, commit or blob ids and second counts
    replaced, so the timings and paths that vary from round to round never hide a repeat."""
    text = line.split("\n", 1)[0]
    text = _TMP_PATH.sub("<tmp>", text)
    text = _HEX_ID.sub("<id>", text)
    text = _SECONDS.sub("<s>", text)
    return " ".join(text.split())


def _split(line: str) -> tuple[str, str]:
    """(check, the rest) of a reason line; a line naming no check is the gate's own (``gate``)."""
    m = _CHECK.match(line)
    return (m.group(1), line[m.end() :]) if m else ("gate", line)


def refusals(res: GateResult) -> dict[tuple[str, str], str]:
    """Every refusal of a refused check result: (check, item) -> its comparable reason (:func:`refusal_reason`;
    several lines of one pair are joined in sorted order). The item is the manifest item a failure is attributed
    to (:attr:`.gate.GateResult.item_reasons`, else every line of a check the item is refused under), or
    :data:`PASS_WIDE`. Notes are not refusals."""
    lines = [r for r in res.reasons if not r.startswith("note: ")]
    found: dict[tuple[str, str], set[str]] = {}
    attributed: set[str] = set()
    for item, own in (getattr(res, "item_reasons", None) or {}).items():
        for line in own:
            check, rest = _split(line)
            found.setdefault((check, item), set()).add(refusal_reason(rest))
            attributed.add(line)
    for item, codes in (res.items_refused or {}).items():
        for check in codes or []:
            if (check, item) in found:
                continue
            same = [ln for ln in lines if _split(ln)[0] == check]
            found[(check, item)] = {refusal_reason(_split(ln)[1]) for ln in same} or {check}
            attributed.update(same)
    for line in lines:
        if line not in attributed:
            check, rest = _split(line)
            found.setdefault((check, PASS_WIDE), set()).add(refusal_reason(rest))
    return {key: "; ".join(sorted(texts)) for key, texts in found.items()}


@dataclass
class StuckRefusals:
    """The repair rounds' refusals, round to round: how many consecutive rounds refused each (check, item) for
    the same reason, which became stuck, and whether the pass stopped repairing because every refusal was.

    Nothing here limits a round: a refusal that changes resets its count, and one absent from a round is gone.
    """

    repeats: int = STUCK_REPEATS
    # (check, item) -> (comparable reason, consecutive rounds) for the last checked round's refusals
    counts: dict[tuple[str, str], tuple[str, int]] = field(default_factory=dict)
    # (check, item) -> the evidence row of each refusal that became stuck in this pass, in order
    rows: dict[tuple[str, str], dict] = field(default_factory=dict)
    stopped: bool = False
    _last: object = None

    def update(self, res: GateResult, candidate: object = None) -> list[tuple[str, str, str]]:
        """Count one checked round's refusals; *candidate* identifies what was checked (None: unknown). A round
        whose candidate equals the last one's repeats every refusal by construction, whatever its text.
        Returns the round's stuck refusals as (check, item, reason), sorted."""
        unchanged = candidate is not None and candidate == self._last
        self._last = candidate
        counts: dict[tuple[str, str], tuple[str, int]] = {}
        for key, reason in refusals(res).items():
            prev = self.counts.get(key)
            same = prev is not None and (prev[0] == reason or unchanged)
            counts[key] = (reason, prev[1] + 1 if same else 1)
        self.counts = counts
        stuck = [(c, i, r) for (c, i), (r, n) in sorted(counts.items()) if n >= self.repeats]
        for c, i, r in stuck:
            self.rows[(c, i)] = {
                "check": c,
                "item": i,
                "reason": redact_error(r)[:300],
                "rounds": counts[(c, i)][1],
            }
        return stuck

    def all_stuck(self) -> bool:
        """Every refusal of the last checked round is stuck (False when it refused nothing)."""
        return bool(self.counts) and all(n >= self.repeats for _, n in self.counts.values())

    def report(self) -> dict | None:
        """The pass's evidence: None when no refusal became stuck."""
        if not self.rows:
            return None
        return {
            "repeats": self.repeats,
            "stopped": self.stopped,
            "refusals": list(self.rows.values()),
        }


def _pairs(stuck: Sequence[tuple[str, str, str]]) -> str:
    return ", ".join(f"{i} ({c})" for c, i, _ in stuck)


def stuck_note(stuck: Sequence[tuple[str, str, str]], repeats: int = STUCK_REPEATS) -> str:
    """The pass note for refusals that became stuck this round."""
    return redact_error(
        f"repair: stuck: the gate refused {_pairs(stuck)} for the same reason in {repeats} rounds in a row "
        f"({'; '.join(r[:200] for _, _, r in stuck)}); no further repair round is offered for them, and the final "
        "merge keeps what it refuses as a draft",
    )


def stopped_note(nxt: int, stuck: Sequence[tuple[str, str, str]], repeats: int = STUCK_REPEATS) -> str:
    """The pass note when every refusal of a round is stuck, so repair round *nxt* does not open."""
    return redact_error(
        f"repair: no round {nxt}: every refusal is stuck (the same item, check and reason in {repeats} rounds in a "
        f"row: {_pairs(stuck)}); the final merge lands the items that pass and keeps the rest as a draft",
    )

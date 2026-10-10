"""Memory v2.1 repair rounds (spec §9.2-9.3, D37): the round budget and the gate's full result as a file.

After Sol's accepted ``finish``, :class:`.sol_pass.SolPass` runs :meth:`.gate.Gate.check` (the whole gate, no
merge) on the round's candidate. A refusal is written in full to ``/inputs/gate/result-<n>.md`` and Sol gets
another round, at most :data:`REPAIR_ROUNDS`, while the pass's budget holds one more round
(:func:`may_repair`). Only the final candidate goes to :meth:`.gate.Gate.merge`, so per-item admission decides
what lands. Nothing here calls a model, git or the jail.

Budget (stated once, here): round 0 runs from the pass's start to Sol's first accepted ``finish``; round k >= 1
from the repair message that opens it to the next accepted ``finish``. A round's USD is the change in the
pass's charge (known spend plus ``cap / max_calls`` per unpriced call); its seconds are its wall time plus the
gate check that ends it. ``ROUND_RESERVE`` is the calibrated constant when given, else the median USD of the
pass's completed rounds. The time reserve is the median of their seconds plus the last gate check's seconds,
because the final :meth:`.gate.Gate.merge` runs the whole gate once more (Amendment A).
"""

from __future__ import annotations

import re
import statistics
from decimal import Decimal
from typing import Iterable

from .gate import GateResult
from .redact import redact_error

#: Spec §15 ``R``: repair rounds after the first finish, at most.
REPAIR_ROUNDS = 2
#: The sandbox keeps each of a run's stdout and stderr up to this many bytes, from the end, and marks any
#: bytes it drops (sandbox_run.CAPTURE_MAX_BYTES); the result says so.
OUTPUT_CAPTURE_BYTES = 1024**2


def _money(value: Decimal) -> str:
    return format(value, "f")


def round_reserve(round_usd: list[Decimal], fixed: Decimal | None = None) -> Decimal:
    """``ROUND_RESERVE``: *fixed* when calibrated (spec §15), else the median USD of the completed rounds."""
    if fixed is not None:
        return Decimal(fixed)
    if not round_usd:
        raise ValueError("no completed round to take a reserve from")
    return Decimal(statistics.median(round_usd))


def may_repair(
    done: int,
    remaining_usd: Decimal | None,
    remaining_s: float,
    calls_left: int,
    round_usd: list[Decimal],
    round_s: list[float],
    fixed_usd: Decimal | None = None,
    last_check_s: float = 0.0,
) -> str | None:
    """None when repair round ``done + 1`` may start; else why not, as one note line.

    The time it needs is the median of the completed rounds' seconds plus *last_check_s*, the last gate
    check's seconds, which the final merge spends again.
    """
    nxt = done + 1
    uncapped = (
        remaining_usd is None
    )  # memory v2.1 r5 (S6): no round count and no USD reserve; time still applies
    if not uncapped and done >= REPAIR_ROUNDS:
        return f"repair: no round {nxt}: at most {REPAIR_ROUNDS} repair rounds per pass"
    if calls_left < 1:
        return f"repair: no round {nxt}: the pass's call cap is reached"
    need = None if uncapped else round_reserve(round_usd, fixed_usd)
    if need is not None and remaining_usd < need:
        return (
            f"repair: no round {nxt}: {_money(remaining_usd)} USD left, "
            f"below the round reserve of {_money(need)} USD"
        )
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


def _fence(text: str) -> str:
    """A code fence longer than any backtick run in *text*, so the output can never close it early."""
    longest = max((len(m) for m in re.findall(r"`+", text)), default=0)
    return "`" * max(3, longest + 1)


def render_result(
    pass_id: str,
    n: int,
    res: GateResult,
    *,
    final: bool,
    candidate: str | None = None,
    pass_notes: Iterable[str] = (),
) -> str:
    """The gate's full result for round *n* as markdown: every check line, note and kept test output, whole.

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
    outputs = sorted(getattr(res, "outputs", {}).items())
    lines += [
        "",
        f"## Test output ({len(outputs)})",
        "",
        "Each failing run's whole output, stdout then stderr. The sandbox keeps each stream's last "
        f'{OUTPUT_CAPTURE_BYTES:,} bytes and marks any it drops ("[N earlier bytes dropped]").',
    ]
    for label, raw in outputs:
        text = redact_error(raw)
        fence = _fence(text)
        lines += ["", f"### {label}", "", fence + "text", text, fence]
    return "\n".join(lines) + "\n"


def repair_message(n: int, path: str, res: GateResult) -> str:
    """Sol's message opening repair round *n* (>= 1) after round ``n - 1``'s refused check."""
    items = (
        ", ".join(
            f"{i} ({', '.join(c or [])})" for i, c in sorted(res.items_refused.items())
        )
        or "none named; the refusal is pass-wide"
    )
    return (
        f"The gate refused round {n - 1}'s candidate (checks: {', '.join(res.refused) or 'none'}; items: "
        f"{items}). Its full result, with every check line, every failing test's output and every note, is "
        f"{path}; read it with read. Fix what it refused in /memory, keep /memory/.pass/manifest.json up to "
        f"date, then call finish again. This is repair round {n} of {REPAIR_ROUNDS}. After the last round the "
        "items that pass land and the rest are kept as a draft for the next pass."
    )

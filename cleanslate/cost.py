"""Cost record: one line per model request, money as decimal strings.

Rules:
  * USD is a decimal string, never a float. Sums use Decimal.
  * A request whose price cannot be established is recorded with usd = null and counted as
    unpriced; totals then report the priced subtotal and the unpriced count, and the total
    itself is unknown (null). Unknown is never written as zero.
  * Identities: run_id (the cell's run), attempt_id (one launch of the cell), solve_id (one
    instance served), request_id (one model request), plus the provider's generation id.
  * Nothing secret is recorded: no headers, no keys, no message bodies unless the caller
    explicitly passes capture_content=True (then only the reply text).

Two files:
  * the cost record (`path`): this module's own lines (attempt, solve, request);
  * the lab journal (`journal`): rows in the request-journal format that
    journal_accounting.read_journal and cost_ledger.py read (`runs/<run>/attempts/<attempt>/costs.jsonl`):
    a request_started row before each call, then a response row with the provider-reported
    account_charge, or a transport_error row. Scripted-fake calls are local_replay rows with
    a zero charge, which the ledger counts as replays, not spend.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from decimal import Decimal

MILLION = Decimal(1_000_000)


def _dec(x) -> Decimal:
    if isinstance(x, float):  # floats only arrive from providers' JSON; go through their text form
        x = repr(x)
    return Decimal(str(x))


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class LabJournal:
    """Writer for the lab's request-journal format (one owner: run_id, attempt_id)."""

    def __init__(self, path: str, run_id: str, attempt_id: str, provider: str = "openrouter"):
        self.path, self.run_id, self.attempt_id, self.provider = path, run_id, attempt_id, provider
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def _row(self, rid: str, origin: str, status: str | None, **kw) -> dict:
        row = {"source_id": f"{rid}:{origin}", "provider": self.provider, "run_id": self.run_id,
               "attempt_id": self.attempt_id, "currency": "USD", "unit": "openrouter_credits",
               "request_attempt_id": rid, "call_id": rid, "origin": origin, "status": status,
               "generation_id": None, "account_charge": None, "upstream_inference_cost": None,
               "tokens": {"prompt": None, "completion": None, "cached": None}, "time": time.time()}
        row.update(kw)
        with open(self.path, "a") as f:
            f.write(json.dumps(row) + "\n")
        return row

    def started(self, rid: str):
        self._row(rid, "request_started", "started")

    def completed(self, rid: str, usage: dict):
        tokens = {"prompt": usage.get("prompt_tokens"), "completion": usage.get("completion_tokens"),
                  "cached": usage.get("cached_tokens")}
        charge = usage.get("provider_cost")
        upstream = usage.get("upstream_inference_cost")
        self._row(rid, "response", "completed", generation_id=usage.get("id"), tokens=tokens,
                  account_charge=None if charge is None else str(_dec(charge)),
                  upstream_inference_cost=None if upstream is None else str(_dec(upstream)))

    def failed(self, rid: str, error: str):
        self._row(rid, "transport_error", "failed", error=error[:200])

    def local_replay(self, rid: str):
        self._row(rid, "local_replay", None, account_charge="0", upstream_inference_cost="0")


class CostLedger:
    def __init__(self, run_id: str | None = None, path: str | None = None, prices: dict | None = None,
                 capture_content: bool = False, attempt_id: str | None = None, journal_path: str | None = None):
        """prices: {model: {"input": "<USD per 1M prompt tokens>", "output": "<USD per 1M completion tokens>"}}
        as decimal strings. A model missing from the table has no price."""
        self.run_id = run_id or f"run-{uuid.uuid4().hex[:12]}"
        self.attempt_id = attempt_id or f"att-{uuid.uuid4().hex[:12]}"
        self.path, self.capture = path, capture_content
        self.prices = {m: {k: _dec(v) for k, v in p.items() if v is not None} for m, p in (prices or {}).items()}
        self.journal = LabJournal(journal_path, self.run_id, self.attempt_id) if journal_path else None
        self.lines: list[dict] = []

    def new_solve(self, label: str = "") -> str:
        solve = f"solve-{uuid.uuid4().hex[:12]}"
        self._write({"type": "solve", "run_id": self.run_id, "attempt_id": self.attempt_id, "solve_id": solve,
                     "label": label, "at": _now(), "time": time.time()})
        return solve

    def estimate(self, usage: dict) -> Decimal | None:
        """Price-table estimate (for caps and the guard only; never written as a charge)."""
        p = self.prices.get(usage.get("model"))
        tin, tout = usage.get("prompt_tokens"), usage.get("completion_tokens")
        if p and tin is not None and tout is not None and "input" in p and "output" in p:
            return p["input"] * _dec(tin) / MILLION + p["output"] * _dec(tout) / MILLION
        return None

    def price(self, usage: dict) -> tuple[str | None, str]:
        """(usd as a decimal string or None, where the number came from)."""
        if usage.get("fake"):
            return "0", "fake model: no provider request was made"
        if usage.get("provider_cost") is not None:
            return str(_dec(usage["provider_cost"])), "provider-reported cost"
        return None, "unpriced: the provider reported no cost"

    def begin(self, solve_id: str, usage_hint: dict | None = None) -> str:
        rid = f"req-{uuid.uuid4().hex[:16]}"
        if self.journal and not (usage_hint or {}).get("fake"):
            self.journal.started(rid)
        return rid

    def record(self, solve_id: str, usage: dict, reply: str | None = None, error: str | None = None,
               request_id: str | None = None) -> dict:
        rid = request_id or f"req-{uuid.uuid4().hex[:16]}"
        usd, source = self.price(usage) if error is None else (None, "failed request: charge unknown")
        est = self.estimate(usage)
        line = {"type": "request", "run_id": self.run_id, "attempt_id": self.attempt_id, "solve_id": solve_id,
                "request_id": rid, "provider_request_id": usage.get("id"), "model": usage.get("model"),
                "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
                "usd": usd, "usd_source": source, "usd_estimate": None if est is None else str(est),
                "error": error, "at": _now(), "time": time.time()}
        if self.capture and reply is not None:
            line["reply"] = reply
        self._write(line)
        if self.journal:
            if usage.get("fake"):
                self.journal.local_replay(rid)
            elif error is not None:
                self.journal.failed(rid, error)
            else:
                self.journal.completed(rid, usage)
        return line

    def summary(self, solve_id: str | None = None) -> dict:
        reqs = [ln for ln in self.lines if ln["type"] == "request" and (solve_id is None or ln["solve_id"] == solve_id)]
        priced = [ln for ln in reqs if ln["usd"] is not None]
        subtotal = sum((Decimal(ln["usd"]) for ln in priced), Decimal(0))
        unpriced = len(reqs) - len(priced)
        return {"run_id": self.run_id, "attempt_id": self.attempt_id, "solve_id": solve_id, "calls": len(reqs),
                "priced_calls": len(priced), "unpriced_calls": unpriced, "priced_usd": str(subtotal),
                "usd": str(subtotal) if unpriced == 0 else None}

    def spend_bound(self, solve_id: str | None = None, since: float | None = None,
                    unknown_call_usd: Decimal = Decimal("0.05")) -> Decimal:
        """An upper-leaning figure for caps and the runaway guard: the charge where known, else
        the price-table estimate, else `unknown_call_usd` per call. Never written as a charge."""
        total = Decimal(0)
        for ln in self.lines:
            if ln["type"] != "request" or (solve_id and ln["solve_id"] != solve_id) or (since and ln["time"] < since):
                continue
            total += Decimal(ln["usd"] if ln["usd"] is not None else ln["usd_estimate"] or unknown_call_usd)
        return total

    def _write(self, line: dict):
        self.lines.append(line)
        if self.path:
            with open(self.path, "a") as f:
                f.write(json.dumps(line) + "\n")


class RunawayStop(RuntimeError):
    """Spend is rising faster than the guard allows; the whole cell stops."""


class RunawayGuard:
    """Stops the cell when spend in the last `window_minutes` extrapolates above `max_usd_per_hour`.
    Reads this process's cost record (known charges, else estimates, else a per-call ceiling)."""

    def __init__(self, ledger: CostLedger, max_usd_per_hour: str = "40", window_minutes: float = 10.0,
                 unknown_call_usd: str = "0.05"):
        self.ledger, self.max_rate = ledger, Decimal(max_usd_per_hour)
        self.window, self.unknown = window_minutes, Decimal(unknown_call_usd)

    def rate(self) -> Decimal:
        spent = self.ledger.spend_bound(since=time.time() - self.window * 60, unknown_call_usd=self.unknown)
        return spent * Decimal(60) / Decimal(str(self.window))

    def check(self):
        r = self.rate()
        if r > self.max_rate:
            raise RunawayStop(f"spend rate {r:.2f} USD/h over the last {self.window:g} min exceeds {self.max_rate} USD/h")

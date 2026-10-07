"""The agent loop: one tool (a persistent Python workspace), answers delivered from code,
harness-side bookkeeping (provenance, smell gate, procedure memory, compaction)."""
from __future__ import annotations

import ast
import os
import re
import time
from dataclasses import dataclass, field
from decimal import Decimal

from .analysis import (backward_slice, file_smells, preview, shape, strip_displays, touches_files,
                       ungrounded_literals, value_smells)
from .cost import CostLedger, RunawayGuard
from .memory import ProcedureStore, canon
from .sandbox import Sandbox
from .world import Gateway, credential_params, no_secrets_in

SYSTEM = """You work in a Python workspace whose current directory holds the user's files.
Each turn, reply with exactly one ```python code block (a short note before it is fine).
Variables persist between turns. After each cell the harness shows what it printed, the value of
its last expression, and which variables changed, so you do not need to print everything.
Deliver the final answer by calling deliver(value) in code; if the work produced files, deliver
their paths, e.g. deliver("report.csv"). The answer must come from code that ran here; the harness
keeps that code and may offer it again for similar requests."""


# ---------------------------------------------------------------- models


class ScriptedModel:
    """Offline stand-in for an LLM: returns scripted replies and records what it was shown.
    A step may be a string or a function of the messages (to react to what it saw)."""
    is_fake = True

    def __init__(self, steps):
        self.steps, self.seen = list(steps), []
        self.last_usage: dict = {}

    def complete(self, messages: list[dict]) -> str:
        self.seen.append([dict(m) for m in messages])
        self.last_usage = {"model": "scripted-fake", "fake": True}
        step = self.steps.pop(0) if self.steps else "(script exhausted)\n```python\ndeliver(None)\n```"
        return step(messages) if callable(step) else step


@dataclass(frozen=True)
class Caps:
    """Per-instance caps. Defaults match the everyday office runner's (15 min, 200 calls, USD 0.50 per task)."""
    max_usd: Decimal = Decimal("0.50")
    max_calls: int = 200
    max_wall_s: float = 900.0


# ---------------------------------------------------------------- context and compaction


class Context:
    """History with deterministic compaction. Computed state lives in the workspace, so old
    turns can be folded to one line each without a model call: the state card lists the live
    variables, the pinned findings and a digest of every folded turn."""

    def __init__(self, request_block: str, budget_chars: int = 24000, keep_last: int = 2):
        self.head = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": request_block}]
        self.turns: list[tuple[str, str, str]] = []  # (assistant, observation, one-line digest)
        self.pinned: list[str] = []
        self.folded = 0
        self.budget, self.keep_last = budget_chars, keep_last

    def add(self, assistant: str, observation: str, digest: str):
        self.turns.append((assistant, observation, digest))

    def pin(self, fact: str):
        if fact not in self.pinned:
            self.pinned.append(fact)

    def _card(self, inventory: dict) -> str:
        """Half the budget at most: pinned findings first, then variables, then as many
        one-line digests of folded cells as fit (newest first)."""
        room = self.budget // 2
        lines = ["STATE CARD (the harness folded older turns; the request above is verbatim)."]
        if self.pinned:
            lines += ["Pinned findings:"] + [f"  - {p}" for p in self.pinned]
        lines.append("Variables in the workspace now:")
        names = list(inventory)
        for i, k in enumerate(names):
            t, r = inventory[k]
            line = f"  {k}: {t} = {r[:60]}"
            if sum(map(len, lines)) + len(line) > room * 0.6:
                lines.append(f"  ... and {len(names) - i} more: {', '.join(names[i:])[:300]}")
                break
            lines.append(line)
        digests = [t[2] for t in self.turns[:self.folded]]
        shown = []
        for d in reversed(digests):
            if sum(map(len, lines)) + sum(map(len, shown)) + len(d) > room:
                break
            shown.insert(0, f"  {d}")
        lines.append(f"Earlier cells ({len(digests) - len(shown)} not listed; their results are in the variables):")
        return "\n".join(lines + shown)

    def messages(self, inventory_fn) -> list[dict]:
        def render():
            msgs = list(self.head)
            if self.folded:
                msgs.append({"role": "user", "content": self._card(inventory_fn())})
            for a, o, _ in self.turns[self.folded:]:
                msgs += [{"role": "assistant", "content": a}, {"role": "user", "content": o}]
            return msgs
        msgs = render()
        while sum(len(m["content"]) for m in msgs) > self.budget and len(self.turns) - self.folded > self.keep_last:
            self.folded = max(self.folded + 1, len(self.turns) - max(self.keep_last, (len(self.turns) - self.folded) // 2))
            msgs = render()
        return msgs


# ---------------------------------------------------------------- the loop


@dataclass
class Result:
    answer: object = None
    delivered: bool = False
    steps: int = 0
    cells: list = field(default_factory=list)
    events: list = field(default_factory=list)
    noop_cells: int = 0
    stored: str | None = None
    used_offer: str | None = None
    files: dict | None = None  # for file deliveries: {path: {sha256, bytes, head}}
    run_id: str | None = None
    attempt_id: str | None = None
    solve_id: str | None = None
    cap: str | None = None  # which per-instance cap ended the task, if any
    offers: list = field(default_factory=list)  # procedures offered as a ready value or a verified cell
    references: list = field(default_factory=list)  # procedures shown as code only
    holds: list = field(default_factory=list)  # one entry per held delivery: the smells
    transcript: list | None = None  # the prompt and every (reply, observation), full text
    world_calls: int | None = None  # API calls made through the gateway
    cost: dict | None = None


def _code_block(text: str) -> str | None:
    m = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
    return m[-1] if m else None


def _text_files(workdir: str, limit: int = 2_000_000) -> dict[str, str]:
    files, total = {}, 0
    for root, _, names in os.walk(workdir):
        for n in sorted(names):
            path = os.path.join(root, n)
            if os.path.islink(path):  # never follow links out of the task directory
                continue
            try:
                with open(path, encoding="utf-8") as f:
                    text = f.read()
            except (UnicodeDecodeError, OSError):
                continue
            total += len(text)
            if total > limit:
                return files
            files[os.path.relpath(path, workdir)] = text
    return files


class Agent:
    def __init__(self, model, store: ProcedureStore | None = None, max_steps: int = 40,
                 budget_chars: int = 24000, cell_timeout: float = 10.0, ledger: CostLedger | None = None,
                 caps: Caps | None = None, guard: RunawayGuard | None = None, hold_hook=None, offers: bool = True,
                 world_upstream: str | None = None):
        self.model, self.store, self.max_steps = model, store, max_steps
        self.budget_chars, self.timeout = budget_chars, cell_timeout
        self.ledger = ledger or CostLedger()
        self.caps, self.guard = caps or Caps(), guard
        self.hold_hook = hold_hook  # called with (workdir, hold number) before a held delivery is answered
        # offers=False: stored procedures appear only as reference code; nothing is bound to the new request,
        # pre-run or placed in the workspace (no ready values, no ready cells)
        self.offers = offers
        # an API world (for example AppWorld's relay socket): the workspace reaches it only through a gateway
        # that records every call; the work finishes with the world's completion call
        self.world_upstream = world_upstream
        self._gw = None

    def _ask(self, messages, solve_id) -> str:
        fake = getattr(self.model, "is_fake", False)  # declared by the model: no provider request is made
        rid = self.ledger.begin(solve_id, {"fake": fake})
        remaining = self.caps.max_wall_s - (time.monotonic() - self._t0)
        try:
            reply = self.model.complete(messages, timeout=max(1.0, remaining)) if getattr(self.model, "takes_timeout", False) \
                else self.model.complete(messages)
        except Exception as exc:  # a failed request may still have been charged: record it, price unknown
            self.ledger.record(solve_id, getattr(self.model, "last_usage", {}) or {}, error=type(exc).__name__,
                               request_id=rid)
            raise
        self.ledger.record(solve_id, getattr(self.model, "last_usage", {}) or {}, reply=reply, request_id=rid)
        return reply

    def _cap_hit(self, solve_id) -> str | None:
        """Checked before every model call. The guard raises RunawayStop (ends the cell)."""
        if self.guard:
            self.guard.check()
        calls = sum(1 for ln in self.ledger.lines if ln["type"] == "request" and ln["solve_id"] == solve_id)
        if calls >= self.caps.max_calls:
            return f"calls: {calls} of {self.caps.max_calls}"
        spent = self.ledger.spend_bound(solve_id)
        if spent >= self.caps.max_usd:
            return f"usd: {spent} of {self.caps.max_usd} (charges where known, estimates otherwise)"
        if time.monotonic() - self._t0 >= self.caps.max_wall_s:
            return f"wall: {self.caps.max_wall_s:g} s"
        return None

    def _offers(self, request, start_files, sb, res, ctx) -> tuple[str, dict]:
        """Retrieve -> verify recorded cases -> bind -> run on today's files -> smell. Only
        results that survive all four steps are put into the workspace as offer_N."""
        self._referenced, self._tried, self._file_offers = [], [], []
        if not self.store:
            return "", {}
        lines, refs, cells_text, offers = [], [], [], {}
        for p in self.store.candidates(request):
            if not self.store.verify(p, timeout=self.timeout):
                res.events.append(f"procedure {p['id']} failed replay of its recorded cases; marked stale")
                continue
            if not self.offers:  # references only: nothing is bound, pre-run or placed in the workspace
                self._referenced.append(p["id"])
                past = p["cases"][-1]
                refs.append(f"Procedure {p['id']} (reference only).\n  Recorded case: {past['request']!r} -> "
                            f"{past['answer']!r} with {past['params']}.\n  Code:\n" + _indent(p["code"]))
                res.events.append(f"procedure {p['id']} shown as reference only (offers are off)")
                continue
            got = self.store.try_on(p, request, start_files, timeout=self.timeout, upstream=self.world_upstream)
            if not got["ok"]:
                res.events.append(f"procedure {p['id']} not offered: {got['why']}")
                if not got.get("reference"):  # same request shape, but it failed on today's files
                    self._tried.append(p["id"])
                self._referenced.append(p["id"])  # shown as code to read, never as a ready answer
                past = p["cases"][-1]
                refs.append(f"Procedure {p['id']} (no ready answer: {got['why']}).\n  Recorded case: "
                            f"{past['request']!r} -> {past['answer']!r} with {past['params']}.\n  Code:\n"
                            + _indent(p["code"]))
                continue
            if got["smells"]:
                self._tried.append(p["id"])
                ctx.pin(f"stored procedure {p['id']} gave {got['value']!r} here but: {'; '.join(got['smells'])}")
                res.events.append(f"procedure {p['id']} not offered: smells {got['smells']}")
                continue
            past = p["cases"][-1]
            if got["outputs"] is not None or got.get("cell"):  # file or world work: offer the verified cell itself
                produced = got.get("preview") or "produced: " + "; ".join(
                    f"{k} ({v['bytes']} bytes): {v['head'][:120]!r}" for k, v in got["outputs"].items())
                cells_text.append(
                    f"Procedure {p['id']} ({p['status']}, {len(p['cases'])} recorded cases, replay-verified just now) "
                    f"with parameters {got['params']}: {produced}\n"
                    f"  Last case: {past['request']!r}.\n  To do the same here, run this cell:\n"
                    f"```python\n{got['code']}\n```")
                self._file_offers.append(p["id"])
                continue
            name = f"offer_{len(offers) + 1}"
            offers[name] = (p["id"], got)
            sb.set(**{name: got["value"]})
            lines.append(f"{name} = {got['value']!r}\n  from procedure {p['id']} ({p['status']}, {len(p['cases'])} "
                         f"recorded cases, replay-verified just now), parameters {got['params']}.\n  Last case: "
                         f"{past['request']!r} -> {past['answer']!r}.\n  Code:\n" + _indent(p["code"]))
        text = ""
        if lines:
            text += ("\n\nThe harness ran stored procedures on this request. Their results are already in the "
                     "workspace. If one answers this request, `deliver(offer_N)`; otherwise compute your own.\n"
                     + "\n".join(lines))
        if cells_text:
            text += ("\n\nStored procedures that produce files or act on the world, checked by replaying their "
                     "recorded cases. If one does what this request asks, run its cell as it is; otherwise do your "
                     "own work.\n"
                     + "\n".join(cells_text))
        if refs:
            text += ("\n\nSimilar past work, for reference only (its parameters are p1, p2, ...; set them "
                     "before running its code):\n" + "\n".join(refs))
        return text, offers

    def solve(self, request: str, workdir: str, hidden: list[str] = (), label: str = "") -> Result:
        """`hidden`: host paths that must never be visible to the workspace (checkers, expected
        answers, generators). `label` goes only into the cost record, never to the model."""
        res = Result(run_id=self.ledger.run_id, attempt_id=self.ledger.attempt_id)
        res.solve_id = solve = self.ledger.new_solve(label)
        self._t0 = time.monotonic()
        self._ctx = None
        try:
            return self._solve(request, workdir, hidden, res, solve)
        finally:
            res.cost = self.ledger.summary(solve)
            if self._ctx is not None:
                res.transcript = [dict(m) for m in self._ctx.head] + [
                    {"role": role, "content": text} for a, o, _ in self._ctx.turns
                    for role, text in (("assistant", a), ("user", o))]

    def _solve(self, request, workdir, hidden, res, solve) -> Result:
        run_id = solve
        start_files = _text_files(workdir)
        cells: list[str] = []  # successful cells only, in order (the provenance log)
        reads: list[set] = []
        writes: list[set] = []
        truncs: list[set] = []
        held: set[str] = set()
        self._world_state = {"held": held, "cells": cells, "res": res, "code": "", "workdir": workdir}
        self._gw = Gateway("live", upstream=self.world_upstream, on_complete=self._world_gate) \
            if self.world_upstream else None
        try:
            return self._loop(request, workdir, hidden, res, solve, run_id, start_files, cells, reads, writes,
                              truncs, held)
        finally:
            if self._gw:
                res.world_calls = len(self._gw.calls)
                self._gw.close()

    def _loop(self, request, workdir, hidden, res, solve, run_id, start_files, cells, reads, writes, truncs, held):
        with Sandbox(workdir, timeout=self.timeout, hidden=hidden, world=self._gw) as sb:
            ctx = self._ctx = Context("", self.budget_chars)
            offer_text, offers = self._offers(request, start_files, sb, res, ctx)
            res.offers = [pid for pid, _ in offers.values()] + list(self._file_offers)
            res.references = [p for p in self._referenced if p not in res.offers]
            listing = ", ".join(sorted(start_files)) or "(none)"
            ctx.head[1]["content"] = f"REQUEST:\n{request}\n\nFiles: {listing}{offer_text}"
            stuck = 0
            for step in range(self.max_steps):
                cap = self._cap_hit(solve)
                if cap:
                    res.cap = cap.split(":")[0]
                    res.events.append(f"cap: {cap}")
                    return res
                reply = self._ask(ctx.messages(sb.inventory), solve)
                res.steps = step + 1
                plan = re.search(r"^plan:\s*(.+)$", reply, re.I | re.M)
                if plan:
                    ctx.pin(f"latest plan: {plan.group(1)[:300]}")
                code = _code_block(reply)
                if code is None:  # prose answer: treated as delivering a typed-in literal
                    code = f"deliver({reply.strip()!r})"
                    reply = f"```python\n{code}\n```"
                if self._gw:
                    self._gw.cell, self._world_state["code"] = len(cells), code
                out = sb.run(code)
                completed = self._gw is not None and self._gw.completion is not None
                if self._gw and "delivered" in out and not completed:
                    out.pop("delivered")
                    out["stdout"] = (out.get("stdout") or "") + ("\n[harness] This request finishes with the "
                                                                 "completion call it names, not with deliver().")
                if completed:  # the completion call passed the gate in the gateway and reached the world
                    out["delivered"] = {"value": self._gw.effects(), "files": None, "note": "completion"}
                res.cells.append({"code": code, "ok": out.get("ok"), "changed": out.get("changed")})
                if out.get("ok"):
                    cells.append(code)
                    reads.append(set(out.get("reads", [])))
                    writes.append(set(out.get("writes", [])))
                    truncs.append(set(out.get("truncates", [])))
                obs, effect = _observation(len(res.cells), out)
                if not effect:
                    res.noop_cells += 1
                stuck = stuck + 1 if (not effect or not out.get("ok")) else 0
                if stuck == 3:
                    obs += ("\nNOTE: the last three cells failed or had no effect. Look at the data or the "
                            "error directly before trying the same thing again.")
                if out.get("limit"):  # a resource limit ended the task; the folder is scored as it is
                    res.events.append(f"limit: {out['error']}")
                    ctx.add(reply, obs, _digest(len(res.cells), code, obs))
                    return res
                if out.get("reset"):
                    ctx.pin("the workspace was restarted after a timeout or crash; earlier variables are gone")
                if "delivered" in out:
                    log = (cells, reads, writes, truncs)
                    verdict = None if completed else self._gate(out, log, start_files, offers, held, ctx)
                    if verdict is None:
                        ctx.add(reply, obs + "\nDelivery accepted.", _digest(len(res.cells), code, obs))
                        res.answer, res.delivered = out["delivered"]["value"], True
                        res.files = out["delivered"].get("files")
                        self._learn(request, log, start_files, offers, res, run_id)
                        return res
                    obs += "\n" + verdict
                    res.events.append("delivery held: " + verdict.splitlines()[1])
                    res.holds.append(verdict.splitlines()[1])
                    if self.hold_hook:
                        self.hold_hook(workdir, len(res.holds))
                ctx.add(reply, obs, _digest(len(res.cells), code, obs))
            res.events.append("step budget exhausted without an accepted delivery")
            return res

    def _slice(self, out, log, offers):
        """Cells that produced the delivery: name and file dependencies; for file deliveries, every
        cell that changed files is a seed (moves and deletes are not visible through open())."""
        cells, reads, writes, truncs = log
        seeds = set(self._gw.changing_cells()) if self._gw else set()
        if out["delivered"].get("files") is not None:
            for i, c in enumerate(cells):
                # a write that a later cell overwrites from scratch is not part of the result
                live = {w for w in writes[i] if not any(w in truncs[j] for j in range(i + 1, len(cells)))}
                if live or touches_files(c):
                    seeds.add(i)
        return backward_slice(cells, len(cells) - 1, set(offers), reads, writes, seeds,
                              given={"apis", "_cred"} if self._gw else set())

    def _gate(self, out, log, start_files, offers, held, ctx) -> str | None:
        """Smell gate. Returns None to accept, or a message that holds the delivery once."""
        cells, reads = log[0], log[1]
        value, delivered_files = out["delivered"]["value"], out["delivered"].get("files")
        smells = file_smells(delivered_files) if delivered_files is not None else value_smells(value)
        if out.get("ok"):
            idx, ext = self._slice(out, log, offers)
            code = "\n".join(cells[i] for i in idx)
            files = {f: start_files.get(f, "") for i in idx for f in reads[i] if f in start_files}
            smells += ungrounded_literals(code, files)
            lone = delivered_files is None or all(_lone_value(f) for f in delivered_files.values())
            if start_files and lone and not files and not ext and len(idx) == 1 and \
                    (delivered_files is not None or _literal_delivery(cells[idx[0]])):
                smells.append("the answer was typed in rather than computed from the files")
        key = canon(value) + "|" + "|".join(sorted(smells))
        if not smells or key in held:
            if smells:
                ctx.pin(f"answer {value!r} delivered again after the check ({'; '.join(smells)})")
            return None
        held.add(key)
        for s in smells:
            ctx.pin(s)
        evidence = preview({f: start_files[f] for f in sorted({f for r in reads for f in r}) if f in start_files})
        return ("DELIVERY HELD for one check:\n" + "; ".join(smells) +
                ("\nFiles the session read:\n  " + "\n  ".join(evidence) if evidence else "") +
                "\nLook at the data that produced this value (for example the distinct values you filtered on). "
                "If the answer is right, deliver the same value again.")

    def _world_gate(self, kwargs: dict) -> str | None:
        """Runs in the gateway thread when the workspace makes the completion call, before it reaches
        the world. Returns a message to hold it once, or None to let it through."""
        st = self._world_state
        answer = kwargs.get("answer")
        smells = value_smells(answer) if answer is not None else []
        cells = st["cells"] + [st["code"]]
        seeds = set(self._gw.changing_cells())
        try:
            idx, _ = backward_slice(cells, len(cells) - 1, seeds=seeds, given={"apis", "_cred"})
            smells += ungrounded_literals("\n".join(cells[i] for i in idx), {"API responses": self._gw.response_text()})
        except SyntaxError:
            pass
        key = canon(answer) + "|" + "|".join(sorted(smells))
        if not smells or key in st["held"]:
            return None
        st["held"].add(key)
        st["res"].holds.append("; ".join(smells))
        st["res"].events.append("completion held: " + "; ".join(smells))
        if self.hold_hook:
            self.hold_hook(st["workdir"], len(st["res"].holds))
        return ("HELD BY THE HARNESS for one check (nothing was sent to the world): " + "; ".join(smells) +
                ". Look at the data that produced this answer. If it is right, make the same call again.")

    def _learn(self, request, log, start_files, offers, res, run_id):
        if not self.store:
            return
        cells, reads = log[0], log[1]
        idx, ext = self._slice({"delivered": {"files": res.files}}, log, offers)
        if ext:  # the answer came from a harness offer
            name = sorted(ext)[0]
            pid, got = offers[name]
            if len(idx) == 1 and re.fullmatch(rf"\s*deliver\(\s*{name}\s*\)\s*", cells[idx[0]]):
                self.store.accepted(pid, request, got["params"], got["inputs"], res.answer, run_id)
                res.used_offer = pid
                res.events.append(f"answer delivered from {pid}; case recorded")
            else:
                res.events.append(f"answer mixes {name} with new code; nothing stored")
            return
        recording = None
        if self._gw is not None:
            if self._gw.completion is None or self._gw.completion["cell"] != len(cells) - 1:
                res.events.append("memory: the completion cell did not finish cleanly; nothing stored")
                return
            try:
                code, recipes = credential_params("\n".join(filter(None, (strip_displays(cells[i]) for i in idx))),
                                                  self._gw.book)
            except ValueError as exc:
                res.events.append(f"memory: {exc}")
                return
            if not no_secrets_in(code, self._gw.book):
                res.events.append("memory: a secret is still in the code; nothing stored")
                return
            if recipes:
                res.events.append(f"memory: {len(recipes)} credential(s) became parameters fetched at call time")
            pid, why = self.store.record(request, code, {}, res.answer, run_id, timeout=self.timeout,
                                         offered=[p for p, _ in offers.values()] + self._file_offers + self._tried,
                                         recording=self._gw.record())
            self._after_record(pid, why, request, offers, res)
            return
        read_any = any(reads[i] for i in idx)
        if len(idx) == 1 and _literal_delivery(cells[idx[0]]) and not read_any and res.files is None:
            res.events.append("memory: answer was typed in, not computed; nothing stored")
            return
        code = "\n".join(filter(None, (strip_displays(cells[i]) for i in idx)))
        if res.files is not None:  # file work may touch files it never open()s: replay needs the whole start tree
            files = dict(start_files)
        else:
            files = {f: start_files[f] for i in idx for f in reads[i] if f in start_files}
        pid, why = self.store.record(request, code, files, res.answer, run_id,
                                     offered=[pid for pid, _ in offers.values()] + self._file_offers + self._tried,
                                     timeout=self.timeout)
        self._after_record(pid, why, request, offers, res)

    def _after_record(self, pid, why, request, offers, res):
        res.stored = pid
        res.events.append(f"memory: {why}")
        if pid in self._file_offers:
            res.used_offer = pid
        new_shape = shape(self.store.get(pid)["code"]) if pid else None
        for shown in [p for p, _ in offers.values()] + self._referenced + self._file_offers:
            if shown != pid and shape(self.store.get(shown)["code"]) == new_shape:
                res.events.append(f"memory: {shown} has the same shape as the delivered work; not a negative")
            elif shown != pid:  # it was shown, and the work delivered was a different procedure
                self.store.rejected(shown, request)
                res.events.append(f"memory: {shown} recorded as not applicable to this request")


def _lone_value(f: dict) -> bool:
    lines = [ln for ln in f.get("head", "").splitlines() if ln.strip()]
    return len(lines) == 1 and len(lines[0].split()) == 1


def _literal_delivery(code: str) -> bool:
    calls = [n for n in ast.walk(ast.parse(code)) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "deliver"]
    return bool(calls) and isinstance(calls[-1].args[0] if calls[-1].args else None, ast.Constant)


def _observation(n: int, out: dict) -> tuple[str, bool]:
    parts = [f"[cell {n}] {'ok' if out.get('ok') else 'ERROR'}"]
    if out.get("stdout"):
        parts.append("stdout:\n" + out["stdout"].rstrip())
    if out.get("last") is not None:
        parts.append("value: " + out["last"])
    if out.get("changed"):
        parts.append("changed: " + "; ".join(f"{k} ({t}) = {r}" for k, (t, r) in out["changed"].items()))
    if out.get("writes"):
        parts.append("wrote: " + ", ".join(out["writes"]))
    if out.get("error"):
        parts.append("error:\n" + out["error"].strip())
    if "delivered" in out:
        parts.append(f"delivered: {out['delivered']['value']!r}")
    effect = len(parts) > 1
    if out.get("ok") and not effect:
        parts.append("This cell had no visible effect: it printed nothing and changed no variable. "
                     "End a cell with an expression to see its value.")
    return "\n".join(parts), effect


def _digest(n: int, code: str, obs: str) -> str:
    first = next((ln.strip() for ln in code.splitlines() if ln.strip() and not ln.strip().startswith("#")), "")
    body = " ".join(obs.splitlines()[1:])
    return f"cell {n}: {first[:80]} -> {body[:160]}"


def _indent(code: str) -> str:
    return "\n".join("    " + ln for ln in code.splitlines())

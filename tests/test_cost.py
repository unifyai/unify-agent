"""Cost record and lab journal: decimal strings, unknown never zero, ids kept, the lab's own
journal parser accepts the journal; per-instance caps; the runaway guard."""
import json
import os
import sys
import tempfile
import time
import unittest
from decimal import Decimal
from pathlib import Path

from helpers import EXPENSES, py, workdir

from cleanslate import Agent, Caps, CostLedger, RunawayGuard, RunawayStop, ScriptedModel

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
import journal_accounting  # noqa: E402  (the lab ledger's journal parser, unchanged)

PRICES = {"m-priced": {"input": "5", "output": "30"}}


class PricedFake:
    """A model that reports a provider charge per call, like OpenRouter with usage accounting."""
    is_fake = False  # it stands in for a paid provider: its charges are recorded as charges

    def __init__(self, steps, cost="0.2", sleep=0.0):
        self.steps, self.cost, self.sleep, self.n, self.last_usage = list(steps), cost, sleep, 0, {}

    def complete(self, messages):
        self.n += 1
        time.sleep(self.sleep)
        self.last_usage = {"id": f"gen-{int(time.time())}-{self.n}", "model": "m-priced", "prompt_tokens": 100,
                           "completion_tokens": 5, "provider_cost": self.cost}
        return self.steps.pop(0) if self.steps else py(f"x{self.n} = {self.n}")


class FlakyModel:
    def __init__(self):
        self.n, self.last_usage = 0, {}

    def complete(self, messages):
        self.n += 1
        if self.n == 1:
            self.last_usage = {"id": "gen-1-a", "model": "m-unpriced", "prompt_tokens": 10, "completion_tokens": 2}
            return py("x = 1")
        self.last_usage = {"model": "m-unpriced"}
        raise ConnectionError("dropped")


class CostRecordTest(unittest.TestCase):
    def test_provider_charge_is_the_charge_and_estimates_stay_estimates(self):
        led = CostLedger(prices=PRICES)
        s = led.new_solve()
        line = led.record(s, {"model": "m-priced", "provider_cost": 0.001234, "prompt_tokens": 100, "completion_tokens": 5})
        self.assertEqual((line["usd"], line["usd_source"], line["usd_estimate"]),
                         ("0.001234", "provider-reported cost", "0.00065"))
        line = led.record(s, {"model": "m-priced", "prompt_tokens": 100, "completion_tokens": 5})
        self.assertIsNone(line["usd"])  # a price table is an estimate, never the charge
        self.assertEqual(line["usd_estimate"], "0.00065")
        sm = led.summary(s)
        self.assertEqual((sm["usd"], sm["priced_usd"], sm["unpriced_calls"]), (None, "0.001234", 1))
        self.assertEqual(led.spend_bound(s), Decimal("0.001884"))

    def test_unknown_is_never_zero(self):
        led = CostLedger(prices=PRICES)
        s = led.new_solve()
        for usage in ({"model": "m-unpriced", "prompt_tokens": 100, "completion_tokens": 5}, {}):
            self.assertIsNone(led.record(s, usage)["usd"])
        self.assertIsNone(led.summary(s)["usd"])
        self.assertEqual(led.spend_bound(s, unknown_call_usd=Decimal("0.05")), Decimal("0.10"))

    def test_ids_and_files(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "cost-record.jsonl")
            led = CostLedger(run_id="run-test", attempt_id="att-x", path=path, prices=PRICES)
            s1, s2 = led.new_solve("first"), led.new_solve("second")
            r1 = led.record(s1, {"model": "m-priced", "provider_cost": "0.1"}, reply="reply text")
            r2 = led.record(s2, {"model": "m-priced", "provider_cost": "0.1"})
            lines = [json.loads(x) for x in open(path)]
        self.assertEqual([ln["type"] for ln in lines], ["solve", "solve", "request", "request"])
        self.assertTrue(all((ln["run_id"], ln["attempt_id"]) == ("run-test", "att-x") for ln in lines))
        self.assertNotEqual(s1, s2)
        self.assertNotEqual(r1["request_id"], r2["request_id"])
        self.assertNotIn("reply", lines[2])  # content is only kept when capture is explicit
        self.assertTrue(all(isinstance(ln.get("usd"), (str, type(None))) for ln in lines))

    def test_lab_journal_is_read_by_the_lab_parser(self):
        with tempfile.TemporaryDirectory() as d:
            journal = Path(d, "runs", "run-j", "attempts", "att-j", "costs.jsonl")
            led = CostLedger(run_id="run-j", attempt_id="att-j", journal_path=str(journal))
            s = led.new_solve()
            for cost in ("0.0012", "0.0003"):
                rid = led.begin(s)
                led.record(s, {"id": f"gen-{int(time.time())}-{cost}", "model": "openai/gpt-6-luna",
                               "prompt_tokens": 900, "completion_tokens": 40, "provider_cost": cost}, request_id=rid)
            rid = led.begin(s)
            led.record(s, {"model": "openai/gpt-6-luna"}, error="ModelError", request_id=rid)
            rid = led.begin(s)
            led.record(s, {"id": "gen-x", "model": "openai/gpt-6-luna"}, request_id=rid)  # completed, no cost
            fake = led.begin(s, {"fake": True})
            led.record(s, {"model": "scripted-fake", "fake": True}, request_id=fake)
            got = journal_accounting.read_journal(journal)
        self.assertTrue(got["coverage_complete"])
        self.assertEqual(got["usd"], Decimal("0.0015"))
        self.assertEqual((got["completed"], got["failed"], got["local_replay"]), (3, 1, 1))
        self.assertEqual((got["unpriced_completed"], got["unpriced_terminal"], got["unpriced_all"]), (1, 2, 2))

    def test_agent_records_every_call_and_failed_calls_as_unknown(self):
        ws = workdir({"expenses.csv": EXPENSES})
        self.addCleanup(ws.cleanup)
        led = CostLedger(prices=PRICES)
        res = Agent(ScriptedModel([py("x = 2"), py("deliver(x * 21)")]), ledger=led).solve("q", ws.name)
        self.assertEqual((res.cost["calls"], res.cost["usd"]), (2, "0"))  # the scripted fake makes no request
        with self.assertRaises(ConnectionError):
            Agent(FlakyModel(), ledger=led).solve("q", ws.name)
        failed = [ln for ln in led.lines if ln.get("error")]
        self.assertEqual(len(failed), 1)
        self.assertIsNone(failed[0]["usd"])
        self.assertIsNone(led.summary()["usd"])


class CapsAndGuardTest(unittest.TestCase):
    def setUp(self):
        self.ws = workdir({"expenses.csv": EXPENSES})
        self.addCleanup(self.ws.cleanup)

    def test_call_cap(self):
        res = Agent(PricedFake([], cost="0"), caps=Caps(max_calls=3)).solve("q", self.ws.name)
        self.assertEqual((res.cap, res.steps, res.delivered), ("calls", 3, False))

    def test_money_cap(self):
        res = Agent(PricedFake([], cost="0.2"), caps=Caps(max_usd=Decimal("0.50"))).solve("q", self.ws.name)
        self.assertEqual((res.cap, res.cost["calls"], res.cost["usd"]), ("usd", 3, "0.6"))

    def test_money_cap_counts_unpriced_calls_by_estimate(self):
        class Unpriced(PricedFake):
            def complete(self, messages):
                out = super().complete(messages)
                self.last_usage.pop("provider_cost")
                return out
        led = CostLedger(prices={"m-priced": {"input": "1000000", "output": "0"}})  # 100 tokens -> 100 USD estimate
        res = Agent(Unpriced([]), ledger=led, caps=Caps(max_usd=Decimal("150"))).solve("q", self.ws.name)
        self.assertEqual((res.cap, res.cost["calls"], res.cost["usd"]), ("usd", 2, None))

    def test_wall_cap(self):
        res = Agent(PricedFake([], cost="0", sleep=0.4), caps=Caps(max_wall_s=1.0)).solve("q", self.ws.name)
        self.assertEqual(res.cap, "wall")
        self.assertLessEqual(res.steps, 3)

    def test_runaway_guard_stops_the_cell(self):
        led = CostLedger()
        guard = RunawayGuard(led, max_usd_per_hour="40", window_minutes=10)
        agent = Agent(PricedFake([], cost="1.0"), ledger=led, caps=Caps(max_usd=Decimal("100")), guard=guard)
        with self.assertRaises(RunawayStop):
            agent.solve("q", self.ws.name)
        calls = led.summary()["calls"]
        self.assertEqual(calls, 7)  # 7 USD in 10 minutes = 42 USD/h > 40: stopped before the 8th call


if __name__ == "__main__":
    unittest.main()

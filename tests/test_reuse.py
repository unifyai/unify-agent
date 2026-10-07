"""Verify before reuse: stored procedures are replayed on their recorded cases, re-bound to the
new request, run on today's files and smell-tested before the model ever sees them."""
import os
import tempfile
import unittest

from helpers import EXPENSES, py, workdir
from test_capture import REQUEST, SESSION

from cleanslate import Agent, ProcedureStore, ScriptedModel

SECOND = "What was the total spend on travel in 2026-04 according to expenses.csv?"


class ReuseTest(unittest.TestCase):
    def setUp(self):
        self.dir = workdir({"expenses.csv": EXPENSES})
        self.mem = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.addCleanup(self.mem.cleanup)
        self.store = ProcedureStore(os.path.join(self.mem.name, "procedures.json"))
        Agent(ScriptedModel(SESSION), self.store).solve(REQUEST, self.dir.name)

    def test_repeat_visit_is_one_turn(self):
        model = ScriptedModel([py("deliver(offer_1)")])
        res = Agent(model, self.store).solve(SECOND, self.dir.name)
        first_prompt = model.seen[0][1]["content"]
        self.assertIn("offer_1 = 99.1", first_prompt)
        self.assertIn("'p2': 'travel'", first_prompt)
        self.assertEqual((res.answer, res.steps, res.used_offer), (99.1, 1, "L1v1"))
        proc = self.store.get("L1v1")
        self.assertEqual(len(proc["cases"]), 2)
        self.assertEqual(proc["status"], "trusted")  # two distinct requests, both verified

    def test_drifted_procedure_is_not_offered(self):
        proc = self.store.get("L1v1")
        proc["code"] = proc["code"].replace("round(", "1 + round(", 1)  # no longer reproduces its case
        self.store.save()
        model = ScriptedModel([py("deliver(0)")] * 3)
        res = Agent(model, self.store).solve(SECOND, self.dir.name)
        self.assertNotIn("offer_1", model.seen[0][1]["content"])
        self.assertEqual(self.store.get("L1v1")["status"], "stale")
        self.assertTrue(any("failed replay" in e for e in res.events))

    def test_look_alike_request_gets_code_reference_not_a_ready_answer(self):
        model = ScriptedModel([py("import csv\nn = sum(1 for r in csv.DictReader(open('expenses.csv'))\n"
                                  "        if r['category'] == 'meals' and r['date'].startswith('2026-03'))\ndeliver(n)")])
        res = Agent(model, self.store).solve(
            "What was the number of meals rows in 2026-03 according to expenses.csv?", self.dir.name)
        prompt = model.seen[0][1]["content"]
        self.assertNotIn("offer_1", prompt)
        self.assertIn("for reference only", prompt)
        self.assertIn("total spend on -> number of", prompt)
        self.assertTrue(any("beyond the parameters" in e for e in res.events))
        self.assertEqual(res.answer, 2)
        self.assertEqual(res.stored, "L2v1")  # a different job: a new lineage, not a new version
        self.assertIn("What was the number of meals rows", self.store.get("L1v1")["negative"][0])
        # the next time this look-alike comes, L1v1 is no longer retrieved for it
        again = "What was the number of meals rows in 2026-04 according to expenses.csv?"
        self.assertEqual([p["id"] for p in self.store.candidates(again)], ["L2v1"])

    def test_smelly_offer_is_withheld(self):
        model = ScriptedModel([py("deliver(1)")])
        res = Agent(model, self.store).solve(
            "What was the total spend on lodging in 2026-04 according to expenses.csv?", self.dir.name)
        self.assertNotIn("offer_1 =", model.seen[0][1]["content"])
        self.assertTrue(any("smells" in e for e in res.events))

    def test_changed_data_makes_a_new_version(self):
        renamed = EXPENSES.replace("date,category,amount", "date,category,amount_usd")
        new_dir = workdir({"expenses.csv": renamed})
        self.addCleanup(new_dir.cleanup)
        model = ScriptedModel([py("import csv\nrows = list(csv.DictReader(open('expenses.csv')))\n"
                                  "t = round(sum(float(r['amount_usd']) for r in rows if r['category'] == 'travel'"
                                  " and r['date'].startswith('2026-04')), 2)\ndeliver(t)")])
        res = Agent(model, self.store).solve(SECOND, new_dir.name)
        self.assertIn("KeyError", model.seen[0][1]["content"])  # the failure is shown, not hidden
        self.assertEqual((res.answer, res.stored), (99.1, "L1v2"))
        self.assertEqual(self.store.get("L1v2")["supersedes"], "L1v1")
        # next visit on the new data: v1 fails on today's files, v2 is offered
        model = ScriptedModel([py("deliver(offer_1)")])
        res = Agent(model, self.store).solve(
            "What was the total spend on meals in 2026-03 according to expenses.csv?", new_dir.name)
        self.assertEqual((res.answer, res.used_offer), (42.75, "L1v2"))


if __name__ == "__main__":
    unittest.main()

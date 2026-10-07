"""Capture: the harness, not the model, extracts the code that produced the delivered answer,
lifts the request's values into parameters and stores it only if a fresh replay reproduces it."""
import os
import tempfile
import unittest

from helpers import EXPENSES, py, workdir

from cleanslate import Agent, ProcedureStore, ScriptedModel

REQUEST = "What was the total spend on meals in 2026-03 according to expenses.csv?"
SESSION = [
    py("import csv\nrows = list(csv.DictReader(open('expenses.csv')))\nlen(rows)"),
    py("for r in rows[:0]:\n    print(r)"),  # does nothing: must be flagged and kept out of the procedure
    py("rows[0]"),  # inspection only: not part of the procedure either
    py("total = round(sum(float(row['amount']) for row in rows\n"
       "                  if row['category'] == 'meals' and row['date'].startswith('2026-03')), 2)\ntotal"),
    py("deliver(total)"),
]


class CaptureTest(unittest.TestCase):
    def setUp(self):
        self.dir = workdir({"expenses.csv": EXPENSES})
        self.mem = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.addCleanup(self.mem.cleanup)
        self.store = ProcedureStore(os.path.join(self.mem.name, "procedures.json"))

    def test_slice_is_stored_with_parameters_and_a_verified_case(self):
        res = Agent(ScriptedModel(SESSION), self.store).solve(REQUEST, self.dir.name)
        self.assertTrue(res.delivered)
        self.assertEqual(res.answer, 42.75)
        self.assertEqual(res.noop_cells, 1)
        self.assertEqual(res.stored, "L1v1")
        proc = ProcedureStore(self.store.path).get("L1v1")  # persisted
        self.assertNotIn("rows[:0]", proc["code"])
        self.assertNotIn("rows[0]\n", proc["code"] + "\n")
        self.assertEqual(sorted(proc["cases"][0]["params"].values()), ["2026-03", "expenses.csv", "meals"])
        self.assertEqual(proc["cases"][0]["answer"], 42.75)
        self.assertEqual(set(proc["cases"][0]["files"]), {"expenses.csv"})
        self.assertEqual(proc["status"], "candidate")

    def test_unreproducible_answer_is_not_stored(self):
        session = [py("import random\nv = random.random()"), py("deliver(v)")]
        res = Agent(ScriptedModel(session), self.store).solve("give me a random number", self.dir.name)
        self.assertTrue(res.delivered)
        self.assertIsNone(res.stored)
        self.assertTrue(any("not stored" in e for e in res.events))
        self.assertEqual(self.store.items, [])


if __name__ == "__main__":
    unittest.main()

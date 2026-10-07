"""Smell gate: a confident zero from filtering on a value the data does not contain ('meal'
versus 'meals') is held once, with evidence, before it becomes the answer."""
import os
import tempfile
import unittest

from helpers import EXPENSES, py, workdir

from cleanslate import Agent, ProcedureStore, ScriptedModel

REQUEST = "Total spend on meal expenses in March 2026 from expenses.csv"
LOAD = py("import csv\nrows = list(csv.DictReader(open('expenses.csv')))")
WRONG = py("total = sum(float(r['amount']) for r in rows if r['category'] == 'meal' and r['date'][:7] == '2026-03')"
           "\ntotal")
FIXED = py("total = sum(float(r['amount']) for r in rows if r['category'] == 'meals' and r['date'][:7] == '2026-03')"
           "\ntotal")


def fix_if_told(messages):
    last = messages[-1]["content"]
    assert "DELIVERY HELD" in last and "'meals'" in last, last
    return FIXED


class SmellTest(unittest.TestCase):
    def setUp(self):
        self.dir = workdir({"expenses.csv": EXPENSES})
        self.mem = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.addCleanup(self.mem.cleanup)
        self.store = ProcedureStore(os.path.join(self.mem.name, "procedures.json"))

    def test_meal_vs_meals_zero_is_caught_and_fixed(self):
        model = ScriptedModel([LOAD, WRONG, py("deliver(total)"), fix_if_told, py("deliver(total)")])
        res = Agent(model, self.store).solve(REQUEST, self.dir.name)
        self.assertEqual(res.answer, 42.75)
        self.assertTrue(any("delivery held" in e for e in res.events))
        proc = self.store.get(res.stored)
        self.assertIn("'meals'", proc["code"])
        self.assertNotIn("'meal'", proc["code"])  # the wrong cell is not in the procedure

    def test_model_may_insist_once_with_a_record(self):
        model = ScriptedModel([LOAD, WRONG, py("deliver(total)"), py("deliver(total)")])
        res = Agent(model, self.store).solve(REQUEST, self.dir.name)
        self.assertEqual((res.delivered, res.answer, res.steps), (True, 0, 4))

    def test_prose_answer_without_computing_is_held(self):
        model = ScriptedModel(["The total is 42.75.", LOAD, FIXED, py("deliver(total)")])
        res = Agent(model, self.store).solve(REQUEST, self.dir.name)
        self.assertIn("typed in", model.seen[1][-1]["content"])
        self.assertEqual(res.answer, 42.75)


if __name__ == "__main__":
    unittest.main()

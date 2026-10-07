"""Words that set a boundary, compare, quantify, negate or combine change what is asked: a request that differs
from a recorded one in such a word must never get the recorded procedure's ready answer. Requests that differ
only in values still bind."""
import os
import tempfile
import unittest

from helpers import py, workdir

from cleanslate import Agent, ProcedureStore, ScriptedModel
from cleanslate.memory import bind, similarity, tokens

SIBLINGS = [  # (recorded request, new request): each pair differs only in meaning words
    ("Count orders placed at or before 2021 in orders.csv", "Count orders placed before 2021 in orders.csv"),
    ("Count orders placed after 2021 in orders.csv", "Count orders placed on or after 2021 in orders.csv"),
    ("List customers with more than 5 orders in orders.csv", "List customers with at least 5 orders in orders.csv"),
    ("Total the orders with a discount in orders.csv", "Total the orders without a discount in orders.csv"),
    ("Count orders since 2021 in orders.csv", "Count orders until 2021 in orders.csv"),
    ("Count orders that were shipped in orders.csv", "Count orders that weren't shipped in orders.csv"),
    ("Sum amounts over 100 in orders.csv", "Sum amounts under 100 in orders.csv"),
    ("Count orders in 2021 or 2022 in orders.csv", "Count orders in 2021 and 2022 in orders.csv"),
    ("Count all orders between 2020 and 2022 in orders.csv", "Count all orders except 2020 and 2022 in orders.csv"),
]
VALUE_ONLY = [
    ("Count orders placed at or before 2021 in orders.csv", {"p1": "2021", "p2": "orders.csv"},
     "Count orders placed at or before 2023 in orders.csv", {"p1": "2023", "p2": "orders.csv"}),
    ("What was the total spend on meals in 2026-03 according to expenses.csv?",
     {"p1": "meals", "p2": "2026-03", "p3": "expenses.csv"},
     "What was the total spend on travel in 2026-04 according to expenses.csv?",
     {"p1": "travel", "p2": "2026-04", "p3": "expenses.csv"}),
    ("Customers with more than 5 orders in orders.csv", {"p1": "orders.csv"},
     "Customers with more than 5 orders in sales.csv", {"p1": "sales.csv"}),
]


def params_for(request: str) -> dict:
    """The values a capture would lift from these requests: numbers and file names."""
    words = [w.strip("?.,") for w in request.split()]
    vals = [w for w in words if any(c.isdigit() for c in w) or w.endswith(".csv")]
    return {f"p{i}": v for i, v in enumerate(dict.fromkeys(vals), 1)}


class BindingTest(unittest.TestCase):
    def test_sibling_requests_never_bind_to_each_others_ready_offers(self):
        for a, b in SIBLINGS:
            for old, new in ((a, b), (b, a)):
                params, extra = bind(old, params_for(old), new)
                self.assertTrue(params is None or extra, f"{old!r} -> {new!r} would get a ready offer")
                self.assertNotEqual(tokens(old), tokens(new), (old, new))  # retrieval sees the difference too

    def test_value_only_returns_still_bind(self):
        for old, old_params, new, want in VALUE_ONLY:
            params, extra = bind(old, old_params, new)
            self.assertEqual((params, extra), (want, []), (old, new))

    def test_similar_requests_are_still_retrieved_as_candidates(self):
        a, b = SIBLINGS[0]
        self.assertGreaterEqual(similarity(a, b), 0.3)  # found, then shown only as a reference


ORDERS = "id,year,amount\n1,2019,10\n2,2020,20\n3,2021,30\n4,2022,40\n"


class EndToEndTest(unittest.TestCase):
    def test_strictly_before_never_gets_the_at_or_before_answer(self):
        ws = workdir({"orders.csv": ORDERS})
        self.addCleanup(ws.cleanup)
        with tempfile.TemporaryDirectory() as mem:
            store = ProcedureStore(os.path.join(mem, "procedures.json"))
            first = py("import csv\nrows = list(csv.DictReader(open('orders.csv')))\n"
                       "n = sum(1 for r in rows if int(r['year']) <= 2021)\ndeliver(n)")
            res = Agent(ScriptedModel([first]), store).solve("Count orders placed at or before 2021 in orders.csv", ws.name)
            self.assertEqual((res.answer, res.stored), (3, "L1v1"))
            model = ScriptedModel([py("import csv\nrows = list(csv.DictReader(open('orders.csv')))\n"
                                      "deliver(sum(1 for r in rows if int(r['year']) < 2021))")])
            res = Agent(model, store).solve("Count orders placed before 2021 in orders.csv", ws.name)
            prompt = model.seen[0][1]["content"]
            self.assertNotIn("offer_1 =", prompt)  # no ready answer for a different boundary
            self.assertIn("for reference only", prompt)
            self.assertEqual(res.answer, 2)
            # and a value-only return does get the ready answer
            model = ScriptedModel([py("deliver(offer_1)")])
            res = Agent(model, store).solve("Count orders placed at or before 2020 in orders.csv", ws.name)
            self.assertEqual((res.answer, res.used_offer, res.steps), (2, "L1v1", 1))


if __name__ == "__main__":
    unittest.main()

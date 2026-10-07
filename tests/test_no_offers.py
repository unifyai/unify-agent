"""offers=False (--no-offers): stored procedures appear only as reference code. Nothing is bound to the new
request, pre-run, or placed in the workspace; the model's own work is still captured and stored."""
import json
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from helpers import EXPENSES, py, workdir
from test_capture import REQUEST, SESSION
from test_files import total_cell

from cleanslate import Agent, ProcedureStore, ScriptedModel
from cleanslate import memory

SECOND = "What was the total spend on travel in 2026-04 according to expenses.csv?"
LOOK = py("globals().get('offer_1', 'absent')")
PROBE = py("import csv\nrows = list(csv.DictReader(open('expenses.csv')))\n"
           "deliver(round(sum(float(r['amount']) for r in rows if r['category'] == 'travel' "
           "and r['date'].startswith('2026-04')), 2))")


def refuse(*a, **k):
    raise AssertionError("binding or a pre-run was attempted with offers off")


class NoOffersTest(unittest.TestCase):
    def setUp(self):
        self.dir = workdir({"expenses.csv": EXPENSES})
        self.mem = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.addCleanup(self.mem.cleanup)
        self.store = ProcedureStore(os.path.join(self.mem.name, "procedures.json"))
        Agent(ScriptedModel(SESSION), self.store).solve(REQUEST, self.dir.name)

    def test_value_procedure_is_reference_only_and_never_bound(self):
        model = ScriptedModel([LOOK, PROBE])
        with mock.patch.object(memory, "bind", refuse), mock.patch.object(ProcedureStore, "try_on", refuse):
            res = Agent(model, self.store, offers=False).solve(SECOND, self.dir.name)
        self.assertIn("value: 'absent'", model.seen[1][-1]["content"])  # nothing was placed in the workspace
        prompt = model.seen[0][1]["content"]
        self.assertNotIn("offer_1", prompt)
        self.assertIn("Procedure L1v1 (reference only).", prompt)
        self.assertIn("for reference only", prompt)
        self.assertEqual(res.answer, 99.1)
        self.assertEqual((res.offers, res.references, res.used_offer), ([], ["L1v1"], None))
        self.assertTrue(any("offers are off" in e for e in res.events))
        self.assertEqual(self.store.get("L1v1")["stats"]["offered"], 0)
        # the model's own work is still captured and stored (here a new procedure: its code has another shape)
        self.assertEqual(res.stored, "L2v1")

    def test_file_procedure_gets_no_ready_cell(self):
        Agent(ScriptedModel([total_cell("meals", "2026-03")]), self.store).solve(
            "Write the total spend on meals in 2026-03 from expenses.csv to answer.txt", self.dir.name)
        Path(self.dir.name, "answer.txt").unlink()
        model = ScriptedModel([total_cell("travel", "2026-04")])
        with mock.patch.object(memory, "bind", refuse), mock.patch.object(ProcedureStore, "try_on", refuse):
            Agent(model, self.store, offers=False).solve(
                "Write the total spend on travel in 2026-04 from expenses.csv to answer.txt", self.dir.name)
        prompt = model.seen[0][1]["content"]
        self.assertNotIn("run this cell", prompt)
        self.assertNotIn("produced:", prompt)
        self.assertRegex(prompt, r"Procedure L\dv\d \(reference only\)")

    def test_offers_on_still_offers(self):  # the default is unchanged
        model = ScriptedModel([py("deliver(offer_1)")])
        res = Agent(model, self.store).solve(SECOND, self.dir.name)
        self.assertEqual((res.answer, res.used_offer), (99.1, "L1v1"))


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "adapters"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
import office  # noqa: E402
import office_analysis  # noqa: E402
from office_fake import OfficeFake  # noqa: E402


@unittest.skipUnless((office.OFFICE_DIR / "stream.json").is_file(), "office-v1 data not installed")
class NoOffersCellTest(unittest.TestCase):
    def test_office_run_and_analysis_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            cell = Path(tmp, "cell")
            with mock.patch.object(memory, "bind", refuse), mock.patch.object(ProcedureStore, "try_on", refuse):
                office.run(office.load_entries(["ofc-d01", "ofc-r01"]), cell, OfficeFake, offers=False)
            run = json.loads((cell / "run.json").read_text())
            recs = [json.loads(x) for x in (cell / "records.jsonl").read_text().splitlines()]
            out = Path(tmp, "r.json")
            office_analysis.main(["--cell", str(cell), "--out", str(out)])
            arms = json.loads(out.read_text())["arms"]
        self.assertFalse(run["offers"])
        self.assertTrue(all(r["offers"] == [] for r in recs))
        self.assertEqual(list(arms), ["cleanslate-full-no-offers"])
        self.assertEqual(recs[1]["references"], ["L1v1"])


if __name__ == "__main__":
    unittest.main()

"""The preregistered analysis runs on a fake cell (office-v1 data needed): metrics, the false-hold
measurement, the fit rate and the criteria evaluation all produce values from real records."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import helpers  # noqa: F401

PROTO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROTO / "analysis"), str(PROTO / "adapters")]
import office  # noqa: E402
import office_analysis  # noqa: E402
from office_fake import OfficeFake  # noqa: E402


@unittest.skipUnless((office.OFFICE_DIR / "stream.json").is_file(), "office-v1 data not installed")
class AnalysisTest(unittest.TestCase):
    def test_report_on_a_fake_cell(self):
        with tempfile.TemporaryDirectory(prefix="cs-analysis-") as tmp:
            cell = Path(tmp, "cell")
            office.run(office.load_entries(["ofc-d01", "ofc-r01"]), cell, OfficeFake)
            # a stand-in baseline cell in the Unify runner's record schema
            base = Path(tmp, "a1")
            base.mkdir()
            rows = [{"task_id": "ofc-d01", "visit": "first", "passed": True, "usd": "0.004", "usd_priced": "0.004",
                     "unpriced_calls": 0, "calls": 9},
                    {"task_id": "ofc-r01", "visit": "return", "passed": False, "usd": None, "usd_priced": "0.003",
                     "unpriced_calls": 1, "calls": 7}]
            (base / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            out = Path(tmp, "report.json")
            office_analysis.main(["--cell", str(cell), "--baseline", f"A1={base}", "--fit", "--out", str(out)])
            report = json.loads(out.read_text())
        full = report["arms"]["cleanslate-full"]
        self.assertEqual((full["solved_per_run"], full["runs"][0]["holds"], full["runs"][0]["false_holds"]), ([2], 1, 0))
        fit = full["fit"]
        self.assertEqual((fit["returns"], fit["with_procedure"], fit["bindable"], fit["passed"]), (1, 1, 0, 0))
        a1 = report["arms"]["A1"]["runs"][0]
        self.assertEqual((a1["usd"], a1["usd_priced"], a1["unpriced_calls"]), (None, "0.007", 1))  # unknown stays unknown
        crit = report["criteria"]
        self.assertIs(crit["S4_false_hold_rate"]["pass"], True)
        self.assertEqual(crit["S2_solved_margin_vs_lean_all"]["value"], "1")
        self.assertIsNone(crit["S3_offer_precision"]["pass"])  # no offers accepted: not evaluated, not passed


READOUT = PROTO.parents[1] / "artifacts/everyday-v1/readout-confirmation-v1/office-rows.jsonl"


@unittest.skipUnless(READOUT.is_file(), "EVAL's office readout not on this machine")
class BaselineReadoutTest(unittest.TestCase):
    def test_a0_a1_rows_load_as_in_evals_readout(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp, "r.json")
            office_analysis.main(["--baseline-readout", str(READOUT), "--out", str(out)])
            arms = json.loads(out.read_text())["arms"]
        self.assertEqual(sorted(arms["A0"]["solved_per_run"]), [22, 22, 22])
        self.assertEqual(sorted(arms["A1"]["solved_per_run"]), [21, 22, 23])
        self.assertEqual(sorted(m["usd_priced"] for m in arms["A1"]["runs"]),
                         ["0.092471285", "0.092791620", "0.093580275"])


if __name__ == "__main__":
    unittest.main()

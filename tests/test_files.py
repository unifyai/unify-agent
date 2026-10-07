"""File deliverables: deliver(path) records content hashes; empty, header-only, zero-row and
lone-zero files are held once; file-producing procedures are captured, replayed and offered as
a verified cell to run."""
import hashlib
import os
import re
import tempfile
import unittest
from pathlib import Path

from helpers import EXPENSES, py, workdir

from cleanslate import Agent, ProcedureStore, ScriptedModel
from cleanslate.analysis import file_smells

LOAD = "import csv\nrows = list(csv.DictReader(open('expenses.csv')))"


def total_cell(category, month):
    return py(LOAD + f"\nt = sum(float(r['amount']) for r in rows if r['category'] == '{category}' "
                     f"and r['date'].startswith('{month}'))\nopen('answer.txt', 'w').write(f'{{t:.2f}}')\n"
                     "deliver('answer.txt')")


class FileDeliveryTest(unittest.TestCase):
    def setUp(self):
        self.dir = workdir({"expenses.csv": EXPENSES})
        self.mem = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.addCleanup(self.mem.cleanup)
        self.store = ProcedureStore(os.path.join(self.mem.name, "procedures.json"))

    def test_deliver_path_records_hash(self):
        res = Agent(ScriptedModel([total_cell("meals", "2026-03")]), self.store).solve(
            "Write the total spend on meals in 2026-03 from expenses.csv to answer.txt", self.dir.name)
        digest = hashlib.sha256(b"42.75").hexdigest()
        self.assertEqual(res.answer, {"__files__": {"answer.txt": digest}})
        self.assertEqual(res.files["answer.txt"]["bytes"], 5)
        self.assertEqual(Path(self.dir.name, "answer.txt").read_text(), "42.75")
        self.assertIsNotNone(res.stored)  # one computing cell is enough to store a procedure

    def test_folder_delivery_and_outside_paths(self):
        model = ScriptedModel([py("import os\nos.makedirs('out', exist_ok=True)\n"
                                  "open('out/a.csv','w').write('x,y\\n1,2\\n')\nopen('out/b.txt','w').write('hi')\n"
                                  "deliver('out')")])
        res = Agent(model).solve("make the out folder", self.dir.name)
        self.assertEqual(sorted(res.files), ["out/a.csv", "out/b.txt"])
        # a path outside the working folder is not a file delivery, just a string value
        res = Agent(ScriptedModel([py("open('x.csv','w').write('a\\n1\\n')\ndeliver('/etc/passwd')"),
                                   py("deliver('/etc/passwd')")])).solve("q", self.dir.name)
        self.assertEqual(res.answer, "/etc/passwd")
        self.assertIsNone(res.files)

    def test_file_smells(self):
        def f(name, text):
            return {name: {"bytes": len(text.encode()), "head": text, "sha256": "x"}}
        self.assertEqual(file_smells(f("a.txt", "")), ["a.txt is empty"])
        self.assertEqual(file_smells(f("r.csv", "id,name\n")), ["r.csv has a header and no rows"])
        self.assertEqual(file_smells(f("r.json", "[]")), ["r.json holds an empty JSON value ([])"])
        self.assertEqual(file_smells(f("answer.txt", "0.00\n")), ["answer.txt: the answer '0.00' looks empty or zero"])
        self.assertEqual(file_smells(f("r.csv", "id,name\n1,a\n")), [])
        self.assertEqual(file_smells(f("answer.txt", "42.75")), [])

    def test_zero_file_is_held_with_evidence_then_fixed(self):
        def fix(messages):
            last = messages[-1]["content"]
            assert "DELIVERY HELD" in last and "answer.txt" in last and "'meals'" in last, last
            return total_cell("meals", "2026-03")
        model = ScriptedModel([total_cell("meal", "2026-03"), fix])
        res = Agent(model, self.store).solve("Write the total spend on meal expenses in 2026-03 to answer.txt",
                                             self.dir.name)
        self.assertEqual(Path(self.dir.name, "answer.txt").read_text(), "42.75")
        self.assertEqual(res.steps, 2)

    def test_header_only_csv_is_held(self):
        cell = py(LOAD + "\nimport csv as c\nw = c.writer(open('big.csv', 'w', newline=''))\n"
                         "w.writerow(['date', 'amount'])\nw.writerows([r['date'], r['amount']] for r in rows "
                         "if float(r['amount']) > 1000)\ndel w\ndeliver('big.csv')")
        model = ScriptedModel([cell, py("deliver('big.csv')")])
        res = Agent(model).solve("list expenses over 1000 into big.csv", self.dir.name)
        self.assertIn("big.csv has a header and no rows", model.seen[1][-1]["content"])
        self.assertTrue(res.delivered)  # accepted on the second, unchanged delivery

    def test_file_procedure_is_replayed_and_offered_as_a_cell(self):
        Agent(ScriptedModel([total_cell("meals", "2026-03")]), self.store).solve(
            "Write the total spend on meals in 2026-03 from expenses.csv to answer.txt", self.dir.name)
        Path(self.dir.name, "answer.txt").unlink()

        def run_offered_cell(messages):
            prompt = messages[1]["content"]
            assert "produced: answer.txt (5 bytes): '99.10'" in prompt, prompt
            return "```python\n" + re.findall(r"```python\n(.*?)```", prompt, re.S)[0] + "```"
        res = Agent(ScriptedModel([run_offered_cell]), self.store).solve(
            "Write the total spend on travel in 2026-04 from expenses.csv to answer.txt", self.dir.name)
        self.assertEqual(Path(self.dir.name, "answer.txt").read_text(), "99.10")
        self.assertEqual((res.steps, res.used_offer), (1, "L1v1"))
        self.assertEqual(self.store.get("L1v1")["status"], "trusted")


if __name__ == "__main__":
    unittest.main()

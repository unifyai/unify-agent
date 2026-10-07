"""Office adapter dry run with the scripted fake: two real office-v1 instances, confined,
scored by the benchmark's own scorer, cleaned up and verified. Skipped if the office data
is not installed on this machine."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import helpers  # noqa: F401  (puts the prototype on sys.path)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "adapters"))
import office  # noqa: E402
from office_fake import OfficeFake  # noqa: E402

from cleanslate import sandbox  # noqa: E402

TASKS = ["ofc-d01", "ofc-r01"]


class RecordingFake(OfficeFake):
    seen: list = []

    def complete(self, messages):
        RecordingFake.seen.append([m["content"] for m in messages])
        return super().complete(messages)


@unittest.skipUnless((office.OFFICE_DIR / "stream.json").is_file(), "office-v1 data not installed")
class OfficeDryRunTest(unittest.TestCase):
    def test_dry_run_two_instances(self):
        entries = office.load_entries(TASKS)
        self.assertEqual([e["task_id"] for e in entries], TASKS)  # stream order
        with tempfile.TemporaryDirectory(prefix="cs-office-") as tmp:
            out = Path(tmp) / "run"
            RecordingFake.seen = []
            summary = office.run(entries, out, RecordingFake)
            records = [json.loads(x) for x in (out / "records.jsonl").read_text().splitlines()]
            costs = [json.loads(x) for x in (out / "cost-record.jsonl").read_text().splitlines()]
            self.assertTrue(summary["cleanup"]["clean"], summary["cleanup"])
            self.assertEqual(list(out.glob("w-*")), [])
        self.assertEqual(sandbox.survivors(), [])
        self.assertGreaterEqual(summary["cleanup"]["workspaces_started"], 2)
        # the model got the stream's prompt byte for byte, and never a task id or job name
        firsts = [s[1] for s in RecordingFake.seen if len(s) == 2]
        for e, first in zip(entries, firsts):
            self.assertTrue(first.startswith(f"REQUEST:\n{e['prompt']}\n\n"), first[:200])
        everything = "\n".join(c for s in RecordingFake.seen for c in s)
        for e in entries:
            self.assertNotIn(e["task_id"], everything)
            self.assertNotIn(e["job"], everything)
        self.assertNotIn(str(office.OFFICE_DIR), everything)
        # every instance was scored by the benchmark's scorer (a known grade), with ids kept
        self.assertEqual(len(records), 2)
        for r in records:
            self.assertIn(r["passed"], (True, False), r)
            self.assertTrue(r["workspace_tree_ok"] and r["task_dir_removed"])
            self.assertEqual(r["delivered_files"], ["answer.txt"])
            self.assertEqual(r["cost"]["calls"], r["steps"])
        self.assertEqual({c["run_id"] for c in costs}, {summary["run_id"]})
        self.assertEqual(len({c["request_id"] for c in costs if c["type"] == "request"}),
                         sum(r["steps"] for r in records))
        # the second instance says 'meal' where the data says 'meals': the gate must have held it
        self.assertTrue(any("delivery held" in ev and "'meals'" in ev for ev in records[1]["events"]), records[1])
        # the held state was scored too: it would have failed, so this hold was not a false hold
        self.assertEqual(records[1]["held_would_have_passed"], [False])

    def test_out_dir_inside_repo_or_data_is_refused(self):
        with self.assertRaises(SystemExit):
            office.run([], office.REPO / "cs-should-not-exist", OfficeFake)
        self.assertFalse((office.REPO / "cs-should-not-exist").exists())


if __name__ == "__main__":
    unittest.main()

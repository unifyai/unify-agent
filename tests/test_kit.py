"""Worker run kit, in fake mode on this laptop (local limits profile): venv from the system
interpreter, one cell with verified cleanup, results and the cost journal where the lab's ledger
reads them. Paid mode is only exercised up to its refusals: no provider request is made."""
import contextlib
import hashlib
import http.server
import io
import json
import os
import subprocess
import threading
import time
import uuid
import sys
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path

import helpers  # noqa: F401
from test_cost import REPO, journal_accounting

KIT = Path(__file__).resolve().parents[1] / "kit"
sys.path.insert(0, str(KIT))
import run_cell  # noqa: E402
import office  # noqa: E402  (put on sys.path by run_cell)
from office_fake import OfficeFake  # noqa: E402

# cost_ledger.py needs Python 3.12 (the lab runs it with 3.12); call its reader in that interpreter
LEDGER_PY = next((str(p) for p in [*sorted(Path.home().glob(".local/share/uv/python/cpython-3.12*/bin/python3.12")),
                                   Path("/usr/bin/python3.12")] if Path(p).exists()), None)


def ledger_rows(root):
    code = ("import sys, json; sys.path.insert(0, sys.argv[1]); import cost_ledger; "
            "rows = cost_ledger.runs_under(__import__('pathlib').Path(sys.argv[2]), 'appworld-r0'); "
            "print(json.dumps(rows, default=str))")
    done = subprocess.run([LEDGER_PY, "-I", "-B", "-c", code, str(REPO), str(root)], capture_output=True, text=True,
                          timeout=120, env={"PATH": "/usr/bin:/bin", "HOME": os.environ["HOME"]})
    assert done.returncode == 0, done.stderr[-2000:]
    return json.loads(done.stdout)

HAVE_DATA = (office.OFFICE_DIR / "stream.json").is_file()
from test_limits import CPU_DELEGATED  # noqa: E402


@unittest.skipUnless(HAVE_DATA, "office-v1 data not installed")
class KitFakeCellTest(unittest.TestCase):
    def test_fake_cell_end_to_end(self):
        with tempfile.TemporaryDirectory(prefix="cs-kit-") as tmp:
            wb, venv = Path(tmp, "workbench"), Path(tmp, "venv")
            env = {"PATH": "/usr/bin:/bin", "HOME": os.environ["HOME"], "CLEANSLATE_VENV": str(venv),
                   "XDG_RUNTIME_DIR": os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")}
            done = subprocess.run([str(KIT / "run_cell.sh"), "--arm", "full", "--run-index", "1",
                                   "--tasks", "ofc-d01,ofc-r01", "--order", "frozen", "--fake",
                                   "--limits-profile", "local", "--workbench", str(wb)],
                                  env=env, capture_output=True, text=True, timeout=300)
            self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
            out = json.loads(done.stdout)
            self.assertTrue(out["clean"])
            self.assertTrue(out["cell"].startswith("fake-everyday-office-cleanslate-full-r1-"))
            self.assertTrue((venv / "bin" / "python").exists())
            cell = Path(out["cell_dir"])
            pointer = json.loads((cell / "attempt.json").read_text())
            detail = journal_accounting.read_journal(Path(pointer["attempt_dir"]) / "costs.jsonl")
            self.assertTrue(detail["coverage_complete"])
            self.assertEqual((detail["usd"], detail["local_replay"], detail["unpriced_all"]),
                             (Decimal("0"), out["cost"]["calls"], 0))
            if LEDGER_PY:
                rows = ledger_rows(wb / "appworld-r0")
                self.assertEqual([(r["run"], r["usd"], r["instances_done"], r["accounting_coverage_complete"])
                                  for r in rows], [(out["cell"], "0", 2, True)])
            meta = json.loads((cell / "cell.json").read_text())
            self.assertEqual((meta["mode"], meta["limits_profile"]), ("fake", "local"))
            self.assertIn(meta["limits"]["mode"], ("scope", "rlimit-only"))
            self.assertEqual(list(wb.glob("preflight-*")), [])
            records = [json.loads(x) for x in (cell / "records.jsonl").read_text().splitlines()]
            self.assertEqual([r["task_id"] for r in records], ["ofc-d01", "ofc-r01"])
            self.assertTrue((cell / "procedures-before-01.json").exists())

    def test_fake_cell_with_no_offers(self):
        with tempfile.TemporaryDirectory(prefix="cs-kit-no-offers-") as tmp:
            buf = io.StringIO()
            from cleanslate import limits as L
            self.addCleanup(L.set_default, L.default())
            with contextlib.redirect_stdout(buf):
                code = run_cell.main(["--arm", "full", "--run-index", "1", "--tasks", "ofc-d01,ofc-r01", "--fake",
                                      "--no-offers", "--limits-profile", "local", "--workbench", f"{tmp}/wb"])
            out = json.loads(buf.getvalue())
            self.assertEqual(code, 0)
            self.assertTrue(out["cell"].startswith("fake-everyday-office-cleanslate-full-no-offers-r1-"))
            cell = Path(out["cell_dir"])
            meta, run = (json.loads((cell / n).read_text()) for n in ("cell.json", "run.json"))
            self.assertEqual((meta["offers"], run["offers"]), (False, False))
            recs = [json.loads(x) for x in (cell / "records.jsonl").read_text().splitlines()]
            self.assertEqual([r["offers"] for r in recs], [[], []])


class StandIn(http.server.BaseHTTPRequestHandler):
    """A local stand-in for OpenRouter that answers with the office fake and charges 0.0004 per call."""
    fakes: dict = {}
    calls = 0

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        assert self.headers["Authorization"] == "Bearer " + os.environ["OPENROUTER_API_KEY"]
        msgs = body["messages"]
        key = msgs[1]["content"]
        if len(msgs) == 2 or key not in StandIn.fakes:
            StandIn.fakes[key] = OfficeFake()
        StandIn.calls += 1
        out = json.dumps({"id": f"gen-{int(time.time())}-{StandIn.calls}", "model": body["model"],
                          "choices": [{"message": {"content": StandIn.fakes[key].complete(msgs)}}],
                          "usage": {"prompt_tokens": 2000, "completion_tokens": 300, "cost": 0.0004}}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a):
        pass


@unittest.skipUnless(HAVE_DATA, "office-v1 data not installed")
class KitRehearsalTest(unittest.TestCase):
    def test_paid_path_against_a_local_stand_in(self):
        from cleanslate import limits as L
        old = L.default()
        self.addCleanup(L.set_default, old)
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), StandIn)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        key = f"sk-or-v1-TESTONLY-{uuid.uuid4().hex}"
        os.environ["OPENROUTER_API_KEY"] = key
        self.addCleanup(os.environ.pop, "OPENROUTER_API_KEY", None)
        with tempfile.TemporaryDirectory(prefix="cs-kit-rh-") as tmp:
            wb = Path(tmp, "wb")
            code = run_cell.main(["--arm", "no-memory", "--run-index", "2", "--tasks", "ofc-d01,ofc-r01",
                                  "--confirm-paid", "--limits-profile", "local", "--workbench", str(wb),
                                  "--base-url", f"http://127.0.0.1:{server.server_address[1]}/api/v1"])
            self.assertEqual(code, 0)
            cell = next((wb / "appworld-r0").iterdir())
            self.assertTrue(cell.name.startswith("rehearse-everyday-office-cleanslate-no-memory-r2-"))
            pointer = json.loads((cell / "attempt.json").read_text())
            detail = journal_accounting.read_journal(Path(pointer["attempt_dir"]) / "costs.jsonl")
            summary = json.loads((cell / "summary.json").read_text())
            records = [json.loads(x) for x in (cell / "records.jsonl").read_text().splitlines()]
            run = json.loads((cell / "run.json").read_text())
            transcripts = sorted(p.name for p in (cell / "transcripts").glob("*.json"))
            self.assertEqual(transcripts, ["00-ofc-d01.json", "01-ofc-r01.json"])  # content capture is on
            for f in Path(tmp).rglob("*"):
                if f.is_file():
                    self.assertNotIn(key, f.read_text(errors="replace"), f)
        self.assertTrue(detail["coverage_complete"])
        self.assertEqual((detail["completed"], detail["unpriced_all"]), (StandIn.calls, 0))
        self.assertEqual(detail["usd"], Decimal("0.0004") * StandIn.calls)
        self.assertEqual(summary["cost"]["usd"], str(Decimal("0.0004") * StandIn.calls))
        self.assertEqual([r["passed"] for r in records], [True, True])
        self.assertEqual((run["memory"], run["mode"], run["model"]), (False, "rehearsal", "openai/gpt-6-luna"))
        self.assertTrue(all(r["offers"] == [] and r["stored"] is None for r in records))  # no memory arm


class KitPaidRefusalTest(unittest.TestCase):
    def setUp(self):
        self.saved = os.environ.pop("OPENROUTER_API_KEY", None)
        self.addCleanup(lambda: self.saved is not None and os.environ.__setitem__("OPENROUTER_API_KEY", self.saved))
        self.tmp = tempfile.TemporaryDirectory(prefix="cs-kit-refuse-")
        self.addCleanup(self.tmp.cleanup)
        self.base = ["--arm", "full", "--run-index", "1", "--confirm-paid", "--workbench", f"{self.tmp.name}/wb"]

    def test_refusals_before_any_request(self):
        self.assertEqual(run_cell.main(self.base), run_cell.REFUSED)  # no key in the environment
        os.environ["OPENROUTER_API_KEY"] = "sk-or-v1-TESTONLY-not-a-key"
        self.addCleanup(os.environ.pop, "OPENROUTER_API_KEY", None)
        self.assertEqual(run_cell.main(self.base), run_cell.REFUSED)  # no prereg
        draft = Path(self.tmp.name, "prereg.md")
        draft.write_text("# PREREG (DRAFT)\n\nStatus: DRAFT\n")
        sha = hashlib.sha256(draft.read_bytes()).hexdigest()
        self.assertEqual(run_cell.main(self.base + ["--prereg", str(draft), "--prereg-sha256", "0" * 64]),
                         run_cell.REFUSED)  # hash mismatch
        self.assertEqual(run_cell.main(self.base + ["--prereg", str(draft), "--prereg-sha256", sha]),
                         run_cell.REFUSED)  # not frozen
        self.assertEqual(run_cell.main(self.base + ["--limits-profile", "local", "--prereg", str(draft),
                                                    "--prereg-sha256", sha]), run_cell.REFUSED)
        # the real draft names the frozen status line in its instructions; it must still count as not frozen
        real = Path(__file__).resolve().parents[1] / "prereg" / "PREREG-office-v1-DRAFT.md"
        real_sha = hashlib.sha256(real.read_bytes()).hexdigest()
        self.assertEqual(run_cell.main(self.base + ["--prereg", str(real), "--prereg-sha256", real_sha]),
                         run_cell.REFUSED)
        self.assertFalse(Path(self.tmp.name, "wb").exists())  # nothing was started

    def test_frozen_prereg_passes_its_gate_and_the_worker_preflight_decides(self):
        if CPU_DELEGATED:  # a worker-like host: the paid path would proceed to a real request; never here
            self.skipTest("cpu is delegated on this host; the worker preflight would pass")
        os.environ["OPENROUTER_API_KEY"] = "sk-or-v1-TESTONLY-not-a-key"
        self.addCleanup(os.environ.pop, "OPENROUTER_API_KEY", None)
        from cleanslate import limits as L
        self.addCleanup(L.set_default, L.default())
        frozen = Path(self.tmp.name, "frozen.md")
        frozen.write_text("# PREREG: test\n\nStatus: FROZEN 2026-10-07T00:00Z by MAIN\n")
        sha = hashlib.sha256(frozen.read_bytes()).hexdigest()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = run_cell.main(self.base + ["--prereg", str(frozen), "--prereg-sha256", sha, "--tasks", "ofc-s01"])
        reply = json.loads(buf.getvalue().splitlines()[0]) if buf.getvalue().startswith("{") else {}
        self.assertEqual(code, run_cell.REFUSED)
        self.assertIn("confinement preflight failed", reply.get("refused", ""))
        self.assertIn("cpu", reply["refused"])
        self.assertEqual(list(Path(self.tmp.name, "wb").glob("*")), [])  # the preflight folder was removed


if __name__ == "__main__":
    unittest.main()

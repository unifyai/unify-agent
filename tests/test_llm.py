"""The real-model client, against a local stand-in for OpenRouter (127.0.0.1; no paid call).
Proves the request shape, the charge accounting, and that the key never reaches the workspace,
any process's argv, or any file the run writes."""
import http.server
import json
import os
import tempfile
import threading
import time
import unittest
import uuid
from decimal import Decimal
from pathlib import Path

from helpers import EXPENSES, py, workdir
from test_cost import journal_accounting

from cleanslate import Agent, ChatClient, CostLedger
from cleanslate.llm import ModelError

SEARCH = ("import os\nkey_hits = []\nfor p in os.listdir('/proc'):\n    if p.isdigit():\n"
          "        for f in ('environ', 'cmdline'):\n            try:\n"
          "                data = open(f'/proc/{p}/{f}', 'rb').read()\n            except OSError:\n"
          "                continue\n            key_hits.append(MARK in data)\n"
          "found = any(key_hits) or any(MARK.decode() in v for v in os.environ.values())\n"
          "deliver({'found': found, 'looked': len(key_hits)})")


class FakeOpenRouter(http.server.BaseHTTPRequestHandler):
    replies: list = []
    seen: list = []
    status = 200

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeOpenRouter.seen.append({"auth": self.headers.get("Authorization"), "body": body, "path": self.path})
        if FakeOpenRouter.status != 200:
            out = json.dumps({"error": {"message": f"bad key {self.headers.get('Authorization')}"}}).encode()
            self.send_response(FakeOpenRouter.status)
        else:
            n = len(FakeOpenRouter.seen)
            content = FakeOpenRouter.replies.pop(0) if FakeOpenRouter.replies else py("deliver(1)")
            out = json.dumps({"id": f"gen-{int(time.time())}-{n}", "model": body["model"],
                              "choices": [{"message": {"role": "assistant", "content": content}}],
                              "usage": {"prompt_tokens": 1200, "completion_tokens": 80, "cost": 0.00016,
                                        "prompt_tokens_details": {"cached_tokens": 0},
                                        "cost_details": {"upstream_inference_cost": 0.00016}}}).encode()
            self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a):
        pass


class ClientTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeOpenRouter)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.base = f"http://127.0.0.1:{cls.server.server_address[1]}/api/v1"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        FakeOpenRouter.replies, FakeOpenRouter.seen, FakeOpenRouter.status = [], [], 200
        self.key = f"sk-or-v1-TESTONLY-{uuid.uuid4().hex}"
        os.environ["OPENROUTER_API_KEY"] = self.key
        self.addCleanup(os.environ.pop, "OPENROUTER_API_KEY", None)
        self.ws = workdir({"expenses.csv": EXPENSES})
        self.addCleanup(self.ws.cleanup)

    def test_request_shape_and_charge(self):
        client = ChatClient(base_url=self.base)
        FakeOpenRouter.replies = ["hello", "hello again"]
        self.assertEqual(client.complete([{"role": "user", "content": "hi"}]), "hello")
        sent = FakeOpenRouter.seen[0]
        self.assertEqual(sent["path"], "/api/v1/chat/completions")
        self.assertEqual(sent["auth"], f"Bearer {self.key}")
        self.assertEqual((sent["body"]["model"], sent["body"]["reasoning"], sent["body"]["usage"]),
                         ("openai/gpt-6-luna", {"effort": "low"}, {"include": True}))
        self.assertEqual(client.last_usage["provider_cost"], 0.00016)
        self.assertNotIn(self.key, json.dumps(client.last_usage) + repr(vars(client)))
        runner_env = ChatClient(base_url=self.base, environ={"OPENROUTER_API_KEY": self.key})  # a runner's mapping
        self.assertEqual(runner_env.complete([{"role": "user", "content": "hi"}]), "hello again")
        self.assertNotIn(self.key, repr(vars(runner_env)))

    def test_missing_key_and_redacted_errors(self):
        os.environ.pop("OPENROUTER_API_KEY")
        with self.assertRaises(ModelError):
            ChatClient(base_url=self.base).complete([{"role": "user", "content": "hi"}])
        self.assertEqual(FakeOpenRouter.seen, [])  # nothing was sent
        os.environ["OPENROUTER_API_KEY"] = self.key
        FakeOpenRouter.status = 401
        with self.assertRaises(ModelError) as err:
            ChatClient(base_url=self.base).complete([{"role": "user", "content": "hi"}])
        self.assertNotIn(self.key, str(err.exception))
        self.assertIn("[key]", str(err.exception))

    def test_workspace_cannot_see_the_key_and_no_file_holds_it(self):
        with tempfile.TemporaryDirectory() as run:
            journal = Path(run, "runs", "run-k", "attempts", "att-k", "costs.jsonl")
            led = CostLedger(run_id="run-k", attempt_id="att-k", path=f"{run}/cost-record.jsonl",
                             journal_path=str(journal))
            mark = f"MARK = {self.key.encode()!r}"
            FakeOpenRouter.replies = [py(mark + "\n" + SEARCH)]
            res = Agent(ChatClient(base_url=self.base), ledger=led).solve("Is there a key anywhere?", self.ws.name)
            self.assertTrue(res.delivered)
            self.assertEqual(res.answer["found"], False)
            self.assertGreater(res.answer["looked"], 0)
            # no host process carries the key in its arguments
            for p in os.listdir("/proc"):
                if p.isdigit():
                    try:
                        self.assertNotIn(self.key.encode(), Path(f"/proc/{p}/cmdline").read_bytes())
                    except OSError:
                        pass
            # nothing the run wrote holds the key; the charge was recorded and the lab parser counts it
            for f in Path(run).rglob("*"):
                if f.is_file():
                    self.assertNotIn(self.key, f.read_text(), f)
            got = journal_accounting.read_journal(journal)
        self.assertEqual((got["usd"], got["completed"]), (Decimal("0.00016"), 1))
        self.assertEqual(res.cost["usd"], "0.00016")


if __name__ == "__main__":
    unittest.main()

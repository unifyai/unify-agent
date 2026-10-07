"""The AppWorld and ARC systems (runner-facing learners): the request text reaches the model byte
for byte, nothing about examples or checking is added, ARC deliveries become the protocol's JSON
action line, and AppWorld runs through the recording gateway to the runner's relay."""
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

import helpers  # noqa: F401
from fake_world import FakeRelay, FakeWorld
from helpers import py

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "adapters"))
import cleanslate_systems as S  # noqa: E402

from cleanslate import ScriptedModel  # noqa: E402
from cleanslate.agent import SYSTEM  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
ARC_LOG = REPO / "artifacts/research-regression-diagnosis-20261001/lean-actor-v1/first-turns-all.jsonl"
AW_LOG = REPO / ("artifacts/research-regression-diagnosis-20261001/cancellation-ownership-canary-v3/"
                 "paid-readiness-v1/worker1-native-logs-v1/turns.jsonl")
ADDED_WORDS = re.compile(r"example|demonstrat|verify|check your|test (your|against)", re.I)


def logged_arc_message() -> str | None:
    if not ARC_LOG.exists():
        return None
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
    from binding_probe import runner_message
    for line in ARC_LOG.read_text().splitlines():
        row = json.loads(line)
        if row.get("bench") == "arc" and (msg := runner_message(row["body"])):
            return msg
    return None


def logged_appworld_message() -> str | None:
    if not AW_LOG.exists():
        return None
    return json.loads(AW_LOG.read_text().splitlines()[0])["message"]


class Recording(ScriptedModel):
    pass


def harness_text(prompt_messages, request) -> str:
    """Everything the model sees that the harness added (system text and framing), without the request."""
    return "\n".join(m["content"] for m in prompt_messages).replace(request, "")


class ARCSystemTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def system(self, steps):
        model = Recording(steps)
        sysm = S.CleanslateARC(model_factory=lambda: model, limits_profile="local")
        sysm.attach(Path(self.tmp.name), {})
        return sysm, model

    def test_logged_request_reaches_the_model_byte_for_byte_and_grid_becomes_submit(self):
        message = logged_arc_message() or "New instance. Task id: task-0\nTest input (1x2):\n3 4\n"
        sysm, model = self.system([py("grid = [[4, 3]]\ndeliver(grid)")])
        reply = sysm.ask(message)
        first = model.seen[0]
        self.assertTrue(first[1]["content"].startswith(f"REQUEST:\n{message}\n\nFiles: (none)"))
        self.assertIsNone(ADDED_WORDS.search(harness_text(first, message)))  # no example checking is prompted
        self.assertIsNone(ADDED_WORDS.search(SYSTEM))
        self.assertEqual(json.loads(reply.splitlines()[-1]), {"action": "submit", "grid": [[4, 3]]})

    def test_feedback_continues_the_instance_and_actions_pass_through(self):
        sysm, model = self.system([py("deliver([[1, 1]])"), py("deliver({'action': 'request_demos'})")])
        sysm.ask("New instance. Task id: task-9\nTest input (1x2):\n0 0\n")
        reply = sysm.ask("Incorrect. Wrong attempts used: 1/8.")
        second = model.seen[1][1]["content"]
        self.assertIn('[your reply]\n{"action": "submit", "grid": [[1, 1]]}', second)
        self.assertIn("Incorrect. Wrong attempts used: 1/8.", second)
        self.assertEqual(json.loads(reply), {"action": "request_demos"})
        sysm.new_session()
        self.assertEqual(sysm.messages, [])


class AppWorldSystemTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.world = FakeWorld()
        self.relay = FakeRelay(self.world)
        self.addCleanup(self.relay.close)

    def test_relay_route_through_the_gateway_with_the_logged_message(self):
        message = logged_appworld_message() or "New task.\nTask: Like every rock song in Road Trip."
        model = Recording([py("pw = [a for a in apis.supervisor.show_account_passwords() "
                              "if a['account_name'] == 'spotify'][0]['password']\n"
                              "tok = apis.spotify.login(username='ada@example.com', password=pw)['access_token']\n"
                              "apis.supervisor.complete_task(answer=None)")])
        sysm = S.CleanslateAppWorld(model_factory=lambda: model, limits_profile="local")
        sysm.attach(Path(self.tmp.name), {"APPWORLD_RELAY_SOCKET": self.relay.path})
        reply = sysm.ask(message)
        self.assertTrue(model.seen[0][1]["content"].startswith(f"REQUEST:\n{message}\n\nFiles: (none)"))
        self.assertIn("Done", reply)
        self.assertEqual((self.world.completed, self.world.calls), (None, 3))
        self.assertEqual(sysm.last_result.world_calls, 3)
        stored = (Path(self.tmp.name) / "cleanslate" / "procedures.json").read_text()
        self.assertNotIn(self.world.password, stored)
        sysm.wipe_state()
        self.assertFalse((Path(self.tmp.name) / "cleanslate").exists())

    def test_no_memory_arm_stores_nothing(self):
        model = Recording([py("apis.supervisor.complete_task(answer=None)")])
        sysm = S.CleanslateAppWorld(model_factory=lambda: model, memory=False, limits_profile="local")
        sysm.attach(Path(self.tmp.name), {"APPWORLD_RELAY_SOCKET": self.relay.path})
        sysm.ask("New task.\nTask: finish.")
        self.assertFalse((Path(self.tmp.name) / "cleanslate" / "procedures.json").exists())


class StartTest(unittest.TestCase):
    def test_worker_profile_is_proved_at_start_or_refused(self):
        from cleanslate import limits as L
        from test_limits import CPU_DELEGATED
        self.addCleanup(L.set_default, L.default())
        with tempfile.TemporaryDirectory() as tmp:
            sysm = S.CleanslateAppWorld(model_factory=lambda: None)  # the default profile is the worker's
            sysm.attach(Path(tmp), {})
            if CPU_DELEGATED:
                sysm.start()
                self.assertIn("cpu", sysm.limits_report["cgroup"]["verified"])
            else:
                with self.assertRaises(S.Unavailable) as err:
                    sysm.start()
                self.assertIn("cpu", str(err.exception))
            self.assertFalse((Path(tmp) / "cleanslate" / "preflight").exists())


if __name__ == "__main__":
    unittest.main()

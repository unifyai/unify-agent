"""API worlds (the AppWorld route): the workspace reaches the world only through a recording
gateway; stored procedures hold no secrets and fetch credentials at call time; replay answers
from the recording and can never reach the live world; state-changing procedures are offered
as cells, never pre-run; the completion call is gated before it reaches the world."""
import json
import os
import re
import tempfile
import unittest
from pathlib import Path

from fake_world import FakeRelay, FakeWorld
from helpers import py, workdir

from cleanslate import Agent, ProcedureStore, ScriptedModel
from cleanslate.memory import replay
from cleanslate.sandbox import Sandbox
from cleanslate.world import Gateway

LIKE = "Like every rock song in my Spotify playlist Road Trip."
LIKE_JAZZ = "Like every jazz song in my Spotify playlist Road Trip."
LOOK = py("accounts = apis.supervisor.show_account_passwords()\naccounts")


def hardcoded_login(messages):
    """Copies the password and, later, the token out of the observations, as models often do."""
    pw = re.search(r"'account_name': 'spotify', 'password': '([^']+)'", messages[-1]["content"]).group(1)
    return py(f"tok = apis.spotify.login(username='ada@example.com', password='{pw}')['access_token']\ntok")


def hardcoded_token_likes(messages):
    tok = re.search(r"'(tok-[0-9a-f]+)'", messages[-1]["content"]).group(1)
    return py(f"songs = apis.spotify.show_playlist(access_token='{tok}', playlist='Road Trip')\n"
              f"done = [apis.spotify.like_song(access_token='{tok}', song_id=s['id']) for s in songs if s['genre'] == 'rock']\n"
              "apis.supervisor.complete_task(answer=None)")


class WorldTest(unittest.TestCase):
    def setUp(self):
        self.world = FakeWorld()
        self.relay = FakeRelay(self.world)
        self.addCleanup(self.relay.close)
        self.ws = workdir({})
        self.addCleanup(self.ws.cleanup)
        self.mem = tempfile.TemporaryDirectory()
        self.addCleanup(self.mem.cleanup)
        self.store = ProcedureStore(os.path.join(self.mem.name, "procedures.json"))

    def first_visit(self):
        model = ScriptedModel([LOOK, hardcoded_login, hardcoded_token_likes])
        return Agent(model, self.store, world_upstream=self.relay.path).solve(LIKE, self.ws.name)

    def test_capture_redacts_secrets_and_credentials_become_parameters(self):
        res = self.first_visit()
        self.assertTrue(res.delivered)
        self.assertEqual(self.world.liked, [11, 13])
        self.assertEqual(res.stored, "L1v1", res.events)
        self.assertIn("memory: 2 credential(s) became parameters fetched at call time", res.events)
        stored = Path(self.store.path).read_text()
        token = next(iter(self.world.tokens))
        self.assertNotIn(self.world.password, stored)  # neither the code nor the recording holds a secret
        self.assertNotIn(token, stored)
        proc = self.store.get("L1v1")
        self.assertIn("cred1 = _cred(apis.supervisor.show_account_passwords(), "
                      "[{'match': {'account_name': p1}}, 'password'])", proc["code"])  # 'spotify' is in the request
        self.assertIn("cred2 = _cred(apis.spotify.login(username='ada@example.com', password=cred1), "
                      "['access_token'])", proc["code"])
        self.assertTrue(proc["world"] and proc["changes_state"])
        calls = proc["cases"][0]["recording"]["calls"]
        self.assertEqual([c["changes_state"] for c in calls if c["api"] == "like_song"], [True, True])
        self.assertEqual(sorted(proc["cases"][0]["params"].values()), ["Road Trip", "rock", "spotify"])
        self.assertNotIn("show_account_passwords()\naccounts", proc["code"])  # inspection cells are not in it
        # replay verification before storing made no call to the live world: it saw only the session's calls
        self.assertEqual(self.world.calls, res.world_calls)

    def test_replay_cannot_reach_the_live_world(self):
        self.first_visit()
        before = self.world.calls
        proc = self.store.get("L1v1")
        self.assertTrue(self.store.verify(proc))  # replays from the recording
        with self.assertRaises(AssertionError):  # a replay gateway cannot even be given an upstream
            Gateway("replay", upstream=self.relay.path, recording=proc["cases"][0]["recording"])
        case = proc["cases"][0]
        off_script = replay(proc["code"], {**case["params"], "p2": "jazz"}, {}, recording=case["recording"])
        self.assertNotIn("delivered", off_script)  # the jazz likes are not in the recording: refused, not sent
        self.assertEqual(self.world.calls, before)
        self.assertEqual(self.world.liked, [11, 13])
        gw = Gateway("replay", recording=case["recording"])
        self.addCleanup(gw.close)
        with Sandbox(self.ws.name, world=gw) as sb:
            seen = sb.run(f"import os\n(os.path.exists({self.relay.path!r}), os.listdir('/run/world'))")
            self.assertEqual(seen["last"], "(False, ['gw.sock'])")  # the live relay's socket is not in the sandbox
            r = sb.run("apis.spotify.like_song(access_token='x', song_id=99)")
            self.assertIn("not in the recording", r["error"])
        self.assertEqual(self.world.calls, before)

    def test_return_visit_runs_the_offered_cell_with_credentials_bound_live(self):
        self.first_visit()
        world2 = FakeWorld(password="pw-new-task-7f3a")  # a new task's world: other credentials, other state
        relay2 = FakeRelay(world2)
        self.addCleanup(relay2.close)

        def run_offer(messages):
            prompt = messages[1]["content"]
            assert "not pre-run: it changes the world" in prompt, prompt
            assert world2.calls == 0  # nothing touched the new world before the model chose to act
            return "```python\n" + re.findall(r"```python\n(.*?)```", prompt, re.S)[0] + "```"
        res = Agent(ScriptedModel([run_offer]), self.store, world_upstream=relay2.path).solve(LIKE_JAZZ, self.ws.name)
        self.assertEqual((res.delivered, res.steps, res.used_offer), (True, 1, "L1v1"))
        self.assertEqual(world2.liked, [12])  # jazz, with the new task's password fetched at call time
        self.assertEqual(self.store.get("L1v1")["status"], "trusted")
        self.assertNotIn("pw-new-task-7f3a", Path(self.store.path).read_text())

    def test_completion_is_gated_before_it_reaches_the_world(self):
        count = py("tok = apis.spotify.login(username='ada@example.com', "
                   "password=[a for a in apis.supervisor.show_account_passwords() if a['account_name'] == 'spotify'][0]['password'])['access_token']\n"
                   "n = len([s for s in apis.spotify.show_playlist(access_token=tok, playlist='Road Trip') if s['genre'] == 'Rock'])\n"
                   "apis.supervisor.complete_task(answer=n)")

        def fix(messages):
            last = messages[-1]["content"]
            assert "HELD BY THE HARNESS" in last and "'rock'" in last, last
            assert self.world.completed is None  # the held call was never sent
            return py("n = len([s for s in apis.spotify.show_playlist(access_token=tok, playlist='Road Trip') "
                      "if s['genre'] == 'rock'])\napis.supervisor.complete_task(answer=n)")
        res = Agent(ScriptedModel([count, fix]), self.store, world_upstream=self.relay.path).solve(
            "How many rock songs are in my Spotify playlist Road Trip?", self.ws.name)
        self.assertEqual((res.delivered, self.world.completed, len(res.holds)), (True, 2, 1))
        self.assertEqual(res.answer["__answer__"], 2)

    def test_deliver_is_not_the_completion_in_a_world(self):
        model = ScriptedModel([py("deliver(3)"), py("apis.supervisor.complete_task(answer=3)")])
        res = Agent(model, world_upstream=self.relay.path).solve("How many?", self.ws.name)
        self.assertIn("finishes with the completion call", model.seen[1][-1]["content"])
        self.assertEqual((res.steps, self.world.completed), (2, 3))


if __name__ == "__main__":
    unittest.main()

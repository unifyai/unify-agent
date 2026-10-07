"""Resource limits, proved by exceeding them from inside the workspace. The scope tests need a
user systemd that can create scopes (true here and on the bench workers); the cpu-quota check is
only enforceable where the cpu controller is delegated (not on this laptop: see design.md)."""
import os
import pwd
import unittest
from pathlib import Path

from helpers import py, workdir

from cleanslate import Agent, Limits, ScriptedModel
from cleanslate.limits import CGROUP_ROOT, scope_available
from cleanslate.sandbox import ConfinementError, Sandbox

SCOPE = scope_available()
USER_CG = CGROUP_ROOT / f"user.slice/user-{os.getuid()}.slice/user@{os.getuid()}.service"
CPU_DELEGATED = "cpu" in (USER_CG / "cgroup.controllers").read_text().split() if USER_CG.exists() else False


@unittest.skipUnless(SCOPE, "no user systemd scope on this host")
class ScopeTest(unittest.TestCase):
    def setUp(self):
        self.dir = workdir({"a.txt": "x"})
        self.addCleanup(self.dir.cleanup)

    def sandbox(self, **kw):
        sb = Sandbox(self.dir.name, timeout=20, limits=Limits(mode="scope", **kw))
        self.addCleanup(sb.close)
        return sb

    def test_limits_are_read_back_from_the_cgroup(self):
        sb = self.sandbox(memory_max_mb=256, tasks_max=16)
        cg = sb.limits_report["cgroup"]
        self.assertEqual(sb.mode, "scope")
        self.assertEqual((cg["memory.max"], cg["memory.swap.max"], cg["pids.max"]), (str(256 << 20), "0", "16"))
        self.assertIn("memory", cg["verified"])
        cgroup = cg["cgroup"]
        self.assertTrue(Path(cgroup).exists())
        sb.close()
        self.assertFalse(Path(cgroup).exists())  # the scope is gone after termination

    def test_memory_limit_kills_the_workspace(self):
        sb = self.sandbox(memory_max_mb=128)
        r = sb.run("block = b'x' * (400 * 1024 * 1024)")
        self.assertTrue(r.get("reset"), r)
        self.assertTrue(sb.run("1 + 1")["ok"])  # a fresh workspace replaced it

    def test_private_tmp_counts_against_memory(self):
        sb = self.sandbox(memory_max_mb=128)
        r = sb.run("for i in range(8):\n    open(f'/tmp/f{i}', 'wb').write(b'x' * (40 * 1024 * 1024))")
        self.assertTrue(r.get("reset"), r)

    def test_process_count_limit(self):
        sb = self.sandbox(tasks_max=8)
        r = sb.run("import subprocess\nkids = []\ntry:\n    for i in range(30):\n"
                   "        kids.append(subprocess.Popen(['sleep', '30']))\nexcept OSError as e:\n"
                   "    err = type(e).__name__\n(len(kids), err)")
        n, err = eval(r["last"])
        self.assertLess(n, 8)
        self.assertEqual(err, "BlockingIOError")

    def test_cpu_quota_required_but_not_delegated_is_refused(self):
        if CPU_DELEGATED:
            sb = self.sandbox(require=("memory", "pids", "cpu"), cpu_quota_pct=50)
            self.assertEqual(sb.limits_report["cgroup"]["cpu.max"], "50000 100000")
        else:
            with self.assertRaises(ConfinementError) as err:
                Sandbox(self.dir.name, limits=Limits(mode="scope", require=("memory", "pids", "cpu")))
            self.assertIn("cpu", str(err.exception))


class ProcessLimitTest(unittest.TestCase):
    def setUp(self):
        self.dir = workdir({"a.txt": "x"})
        self.addCleanup(self.dir.cleanup)

    def test_cpu_seconds_limit(self):
        sb = Sandbox(self.dir.name, timeout=20, limits=Limits(mode="rlimit-only", cpu_seconds=2))
        self.addCleanup(sb.close)
        r = sb.run("while True:\n    pass")
        self.assertTrue(r.get("reset"), r)
        self.assertIn("exited", r["error"])  # killed by SIGXCPU/SIGKILL well before the 20 s timeout

    def test_task_folder_size_limit_ends_the_task(self):
        model = ScriptedModel([py("for i in range(4):\n    open(f'big{i}.bin', 'wb').write(b'x' * (1 << 20))"),
                               py("deliver(1)")])
        from cleanslate import limits as L  # the limit reaches the agent through the sandbox default
        old = L.default()
        L.set_default(old.with_(workdir_max_mb=2))
        self.addCleanup(L.set_default, old)
        res = Agent(model).solve("write files", self.dir.name)
        self.assertFalse(res.delivered)
        self.assertTrue(any("over its 2 MB limit" in e for e in res.events), res.events)

    def test_worker_profile_refuses_rlimit_only_and_wrong_user(self):
        with self.assertRaises(ConfinementError):
            Sandbox(self.dir.name, limits=Limits.worker().with_(mode="rlimit-only"))
        me = pwd.getpwuid(os.getuid()).pw_name
        with self.assertRaises(ConfinementError) as err:
            Sandbox(self.dir.name, limits=Limits.worker(dedicated_user="cleanslate-run").with_(mode="prlimit-user"))
        self.assertIn(me, str(err.exception))
        with self.assertRaises(ConfinementError):  # no dedicated user named: prlimit mode is refused too
            Sandbox(self.dir.name, limits=Limits.worker().with_(mode="prlimit-user"))


if __name__ == "__main__":
    unittest.main()

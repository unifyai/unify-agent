"""Confinement, proved from inside the workspace (bubblewrap). Each check runs code in the
workspace and inspects what that code could see or do."""
import os
import tempfile
import unittest
import uuid
from pathlib import Path

from helpers import workdir

from cleanslate.sandbox import ConfinementError, Sandbox, check_task_dir, marker_alive, ns_members

REPO = Path(__file__).resolve().parents[3]  # the research repository
HOME = Path.home()


def inside(sb, expr):
    r = sb.run(expr)
    assert r["ok"], r.get("error")
    return eval(r["last"])  # the worker returns repr() of plain values


class ConfinementTest(unittest.TestCase):
    def setUp(self):
        self.dir = workdir({"data.csv": "a,b\n1,2\n"})
        self.addCleanup(self.dir.cleanup)
        self.sb = Sandbox(self.dir.name)
        self.addCleanup(self.sb.close)

    def test_a_no_parent_environment(self):
        os.environ["CS_PARENT_MARKER"] = "visible-only-in-parent"
        self.addCleanup(os.environ.pop, "CS_PARENT_MARKER")
        sb = Sandbox(self.dir.name)  # started after the marker was set
        self.addCleanup(sb.close)
        env = inside(sb, "import os\ndict(os.environ)")
        self.assertEqual(set(env) - {"PWD"}, {"PATH", "HOME", "LANG", "PYTHONDONTWRITEBYTECODE"})
        self.assertNotIn("visible-only-in-parent", repr(env))
        # nor through /proc: every process the workspace can see has a clean environment
        environs = inside(sb, "import os\n[open(f'/proc/{p}/environ','rb').read() for p in os.listdir('/proc') "
                              "if p.isdigit()]")
        self.assertTrue(environs)
        self.assertFalse(any(b"visible-only-in-parent" in e for e in environs))

    def test_a_host_processes_invisible(self):
        pids = inside(self.sb, "import os\nsorted(int(p) for p in os.listdir('/proc') if p.isdigit())")
        self.assertLessEqual(len(pids), 3)  # bwrap's init, the worker, maybe a reaper
        self.assertEqual(inside(self.sb, f"import os\ntry:\n    os.kill({os.getpid()}, 0); r = 'reachable'\n"
                                          "except ProcessLookupError:\n    r = 'no such process'\nr"),
                         "no such process")

    def test_b_home_and_repo_not_readable(self):
        for path in (HOME, REPO, REPO / "AGENTS.md", HOME / ".local/share/continual-harness-research"):
            self.assertFalse(inside(self.sb, f"import os\nos.path.exists({str(path)!r})"), path)
        err = self.sb.run(f"open({str(REPO / 'AGENTS.md')!r}).read()")["error"]
        self.assertIn("FileNotFoundError", err)
        self.assertEqual(inside(self.sb, "import os\nos.environ['HOME'], os.listdir(os.environ['HOME'])"),
                         ("/tmp/home", []))
        top = set(inside(self.sb, "import os\nos.listdir('/')"))
        self.assertTrue(top <= {"usr", "bin", "lib", "lib64", "proc", "dev", "tmp", "work"}, top)

    def test_c_network_fails(self):
        r = self.sb.run("import socket\nsocket.create_connection(('1.1.1.1', 53), timeout=3)")
        self.assertIn("OSError", r["error"].replace("Network is unreachable", "OSError"))
        r = self.sb.run("import socket\nsocket.getaddrinfo('example.com', 80)")
        self.assertIn("gaierror", r["error"])
        self.assertEqual(inside(self.sb, "import socket\n[n for _, n in socket.if_nameindex()]"), ["lo"])

    def test_d_writes_outside_task_dir_fail(self):
        for target in ("/usr/cs-probe", "/cs-probe", "/etc/cs-probe", "/bin/cs-probe", "/lib/cs-probe",
                       f"{HOME}/cs-probe", f"{REPO}/cs-probe"):
            r = self.sb.run(f"open({target!r}, 'w').write('x')")
            self.assertFalse(r["ok"], target)
            self.assertRegex(r["error"], "Read-only file system|No such file or directory|Permission denied")
            self.assertFalse(os.path.exists(target), target)
        # the private /tmp is writable but disappears: nothing reaches the host's /tmp
        name = f"/tmp/cs-probe-{uuid.uuid4().hex}"
        self.assertTrue(self.sb.run(f"open({name!r}, 'w').write('x')")["ok"])
        self.assertFalse(os.path.exists(name))
        # the task directory is the one writable host path
        self.assertTrue(self.sb.run("open('out.txt', 'w').write('ok')")["ok"])
        self.assertEqual(Path(self.dir.name, "out.txt").read_text(), "ok")

    def test_e_checker_path_given_to_adapter_is_not_visible(self):
        with tempfile.TemporaryDirectory(prefix="cs-checker-") as checker:
            Path(checker, "expected.json").write_text('{"answer": 42}')
            Path(checker, "check.py").write_text("print('checker')")
            sb = Sandbox(self.dir.name, hidden=[checker])
            self.addCleanup(sb.close)
            self.assertFalse(inside(sb, f"import os\nos.path.exists({checker!r})"))
            found = inside(sb, "import os\n[os.path.join(r, n) for top in ('/tmp', '/work', '/dev') "
                               "for r, _, ns in os.walk(top) for n in ns if n in ('expected.json', 'check.py')]")
            self.assertEqual(found, [])
            # and the harness refuses task directories that would expose or sit inside a hidden path
            with self.assertRaises(ConfinementError):
                check_task_dir(str(Path(checker).parent), hidden=[checker])
            inner = Path(checker, "ws")
            inner.mkdir()
            with self.assertRaises(ConfinementError):
                check_task_dir(str(inner), hidden=[checker])
            with self.assertRaises(ConfinementError):
                check_task_dir(str(HOME))

    def test_termination_is_verified(self):
        sb = Sandbox(self.dir.name, timeout=1.0)
        marker, namespaces = sb.marker, set(sb.namespaces)
        self.assertTrue(marker_alive(marker))
        sb.run("import subprocess\nchild = subprocess.Popen(['sleep', '60'])")
        self.assertGreaterEqual(len(ns_members(namespaces)), 3)  # init, worker, the background child
        r = sb.run("while True:\n    pass")
        self.assertTrue(r["reset"])
        # every process of the old namespace is gone, including the child the code left behind
        self.assertEqual(marker_alive(marker), [])
        self.assertEqual(ns_members(namespaces), [])
        new_marker, new_ns = sb.marker, set(sb.namespaces)
        self.assertNotEqual(new_ns, namespaces)
        sb.close()
        self.assertEqual(marker_alive(new_marker) + ns_members(new_ns), [])


if __name__ == "__main__":
    unittest.main()

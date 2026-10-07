import os
import unittest

from helpers import workdir

from cleanslate.sandbox import Sandbox


class SandboxTest(unittest.TestCase):
    def setUp(self):
        self.dir = workdir({"a.txt": "hello"})
        self.addCleanup(self.dir.cleanup)

    def test_state_persists_and_changes_are_reported(self):
        with Sandbox(self.dir.name) as sb:
            r1 = sb.run("x = 41")
            self.assertEqual(r1["changed"]["x"][1], "41")
            r2 = sb.run("x + 1")
            self.assertEqual(r2["last"], "42")
            self.assertEqual(r2["changed"], {})

    def test_reads_cwd_and_no_inherited_environment(self):
        os.environ["CS_TEST_MARKER"] = "parent-only"
        self.addCleanup(os.environ.pop, "CS_TEST_MARKER")
        with Sandbox(self.dir.name) as sb:
            r = sb.run("import os\n(open('a.txt').read(), os.environ.get('CS_TEST_MARKER'), os.getcwd())")
            self.assertIn("'hello', None", r["last"])
            self.assertEqual(r["reads"], ["a.txt"])

    def test_timeout_kills_and_restarts(self):
        with Sandbox(self.dir.name, timeout=1.0) as sb:
            sb.run("y = 5")
            r = sb.run("while True:\n    pass")
            self.assertFalse(r["ok"])
            self.assertTrue(r["reset"])
            self.assertIn("timeout", r["error"])
            self.assertIn("NameError", sb.run("y")["error"])  # state lost, and said so
            self.assertTrue(sb.run("1")["ok"])  # the new process works

    def test_deliver_is_reported(self):
        with Sandbox(self.dir.name) as sb:
            r = sb.run("deliver({'a': 1.5})")
            self.assertEqual(r["delivered"]["value"], {"a": 1.5})


if __name__ == "__main__":
    unittest.main()

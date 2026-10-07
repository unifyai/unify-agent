"""Compaction: long sessions stay under a context budget without a model call, and the facts
that matter (the request, live variables, pinned findings, the plan) survive."""
import unittest

from helpers import py, workdir

from cleanslate import Agent, ScriptedModel

REQUEST = "Find the account id in ledger.txt and report it with the number of ledger lines."
BUDGET = 4000


class CompactionTest(unittest.TestCase):
    def setUp(self):
        lines = ["header"] + [f"line {i}" for i in range(50)] + ["account: AC-7731"]
        self.dir = workdir({"ledger.txt": "\n".join(lines) + "\n"})
        self.addCleanup(self.dir.cleanup)

    def test_long_session_keeps_key_facts_under_budget(self):
        seen_sizes = []

        def check_and_deliver(messages):
            seen_sizes.append(sum(len(m["content"]) for m in messages))
            card = next(m["content"] for m in messages if m["content"].startswith("STATE CARD"))
            assert "account_id: str = 'AC-7731'" in card, card
            assert "latest plan: find the account line, then count lines" in card, card
            assert REQUEST in messages[1]["content"]
            return py("deliver(f'{account_id}, {n_lines} lines')")

        def filler(i):
            def step(messages):
                seen_sizes.append(sum(len(m["content"]) for m in messages))
                return py(f"chunk_{i} = 'x' * 400\nprint(chunk_{i})")
            return step

        steps = [py("text = open('ledger.txt').read()\nn_lines = len(text.splitlines())",
                    "Plan: find the account line, then count lines."),
                 py("account_id = text.split('account: ')[1].strip()")]
        steps += [filler(i) for i in range(25)] + [check_and_deliver]
        res = Agent(ScriptedModel(steps), budget_chars=BUDGET).solve(REQUEST, self.dir.name)
        self.assertEqual(res.answer, "AC-7731, 52 lines")
        self.assertLessEqual(max(seen_sizes), BUDGET + 1500)  # the card itself may exceed slightly
        self.assertEqual(res.steps, 28)


if __name__ == "__main__":
    unittest.main()

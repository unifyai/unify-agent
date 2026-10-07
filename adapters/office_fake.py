"""A scripted stand-in for the model, for the office dry run. It makes no provider request.

It reads only what the harness shows it (the messages). It handles one shape of request,
"how much did a team spend on a category in a quarter, write it to a file", well enough to
exercise the plumbing: the copy-in, the confined workspace, file delivery, the smell gate,
the procedure store, scoring and cleanup. It is a test fixture, not a solver: it knows
nothing about task ids, expected answers or checkers.

On purpose it uses the request's own wording for the category (e.g. 'meal'), so a request
that says 'meal' while the data says 'meals' shows whether the smell gate catches it.
"""
from __future__ import annotations

import re

QUARTERS = {"1": "01 02 03", "2": "04 05 06", "3": "07 08 09", "4": "10 11 12"}


def _cell(code: str, note: str = "") -> str:
    return f"{note}\n```python\n{code}\n```"


class OfficeFake:
    is_fake = True  # no provider request: the cost journal records local replays at zero charge

    def __init__(self):
        self.last_usage: dict = {}
        self.category: str | None = None

    def complete(self, messages: list[dict]) -> str:
        self.last_usage = {"model": "office-fake", "fake": True}
        request = messages[1]["content"].split("\n\n")[0]
        last = messages[-1]["content"] if len(messages) > 2 else ""
        if "run this cell" in messages[1]["content"] and len(messages) == 2:
            code = re.findall(r"```python\n(.*?)```", messages[1]["content"], re.S)[0]
            return _cell(code, "Plan: reuse the verified procedure the harness offered.")
        if len(messages) == 2:
            return _cell("import csv\nrows = list(csv.DictReader(open('expenses.csv')))\n"
                         "sorted(rows[0]), len(rows)", "Plan: look at the export, then filter and sum.")
        if "DELIVERY HELD" in last:
            near = re.findall(r"the data has '([^']+)'", last)
            if near and self.category:
                self.category = near[0]
                return self._compute(request)
            return _cell("deliver('answer.txt')", "The value is right; delivering it again.")
        if "Delivery accepted" in last:
            return _cell("deliver('answer.txt')")
        return self._compute(request)

    def _compute(self, request: str) -> str:
        team = re.search(r"(\w+) team", request)
        quarter = re.search(r"Q([1-4]) (\d{4})", request)
        if self.category is None:
            cat = re.search(r"\b(?:on|approved) (\w+)", request)
            self.category = cat.group(1) if cat else ""
        if not (team and quarter):
            return _cell("deliver(None)", "I cannot parse this request.")
        months = [f"{quarter.group(2)}-{m}" for m in QUARTERS[quarter.group(1)].split()]
        return _cell(
            "total = sum(float(r['amount_usd']) for r in rows\n"
            f"            if r['department'] == {team.group(1)!r} and r['category'] == {self.category!r}\n"
            f"            and r['status'] == 'approved' and r['date'][:7] in {months!r})\n"
            "open('answer.txt', 'w').write(f'{total:.2f}')\n"
            "deliver('answer.txt')")

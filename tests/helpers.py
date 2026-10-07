import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

EXPENSES = """date,category,amount
2026-03-02,meals,12.50
2026-03-05,travel,140.00
2026-03-09,meals,30.25
2026-04-01,meals,8.00
2026-04-03,travel,99.10
2026-03-20,office,45.00
"""


def workdir(files: dict[str, str]) -> tempfile.TemporaryDirectory:
    d = tempfile.TemporaryDirectory(prefix="cs-test-")
    for name, text in files.items():
        with open(os.path.join(d.name, name), "w") as f:
            f.write(text)
    return d


def py(code: str, note: str = "") -> str:
    return f"{note}\n```python\n{code}\n```"

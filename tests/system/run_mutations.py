"""Prove the system tests are not vacuous: apply each mutation in ``mutations.json`` to the harness, run its tests,
record whether they fail and where, then restore the file exactly. Also runs every listed test once unmutated (the
baseline), so a test that already fails is compared by where it fails.

Run from the repo root of a keyless trial tree (never a live one): ``python tests/system/run_mutations.py OUT.json``
(``--only M1,M5`` for a subset). Stdlib only; it calls pytest through the tree's own interpreter.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = Path(__file__).with_name("mutations.json")
FAIL_AT = re.compile(r"^(?:/\S+/)?(tests/system/\S+\.py):(\d+): (\w+)", re.M)
ERR = re.compile(r"^E\s+(\S.*)$", re.M)


def run(tests: list[str]) -> dict:
    p = subprocess.run([sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-q", "-rA", "-n", "4", *tests],
                       cwd=ROOT, capture_output=True, text=True, timeout=3600)
    out = p.stdout + p.stderr
    results = {t: ("PASSED" if re.search(r"PASSED \S* ?" + re.escape(t), out) else
                   "FAILED" if re.search(r"(FAILED|ERROR) \S* ?" + re.escape(t), out) else "UNKNOWN") for t in tests}
    where = [f"{m.group(1)}:{m.group(2)}" for m in FAIL_AT.finditer(out) if m.group(3) in ("AssertionError", "Failed")]
    first = ERR.search(out)
    errors = [e[:400] for e in ERR.findall(out)][:40]
    return {"results": results, "fail_at": sorted(set(where)), "first_error": first.group(1)[:300] if first else "",
            "errors": errors}


def main() -> int:
    out_path = Path(sys.argv[1])
    only = None
    if "--only" in sys.argv:
        only = set(sys.argv[sys.argv.index("--only") + 1].split(","))
    spec = [m for m in json.loads(SPEC.read_text()) if only is None or m["id"] in only]
    report = {"baseline": {}, "mutations": []}
    tests = sorted({t for m in spec for t in m["tests"]})
    report["baseline"] = run(tests)
    out_path.write_text(json.dumps(report, indent=1))
    for m in spec:
        f = ROOT / m["file"]
        original = f.read_bytes()
        text = original.decode()
        row = {k: m[k] for k in ("id", "what", "file", "tests")}
        if text.count(m["old"]) != 1:
            row["applied"] = False
            row["why"] = f"the mutated text occurs {text.count(m['old'])} times"
        else:
            f.write_text(text.replace(m["old"], m["new"]))
            try:
                row["applied"] = True
                row.update(run(m["tests"]))
            finally:
                f.write_bytes(original)
            assert f.read_bytes() == original, f"{f} was not restored"
        report["mutations"].append(row)
        out_path.write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

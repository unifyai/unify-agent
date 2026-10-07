"""Can lifting request values into parameters ever give a ready offer on a stream's returns?

    python3 analysis/binding_probe.py --arc-first-turns FILE [--out report.json]

A ready offer needs the new request to differ from a recorded one only at parameter positions
(memory.bind's shape check). Parameters are literals in the captured code that also appear in
the request as whole words of at least two characters (analysis.lift_parameters). So, for each
pair of requests for the same task (the earlier visit and the next one), this probe asks two
questions, with no code needed:

  * possible: is every differing word liftable (at least two characters and not 0 or 1)? If any
    differing word is not, then no code whatsoever can make the pair shape-equal, so no ready
    offer is possible.
  * generous: lift every liftable word that differs, bind them, and check the shape. This is the
    best case for any captured code.

It reads the logged first-turn requests of Unify's ARC runs (lean-actor-v1/first-turns-all.jsonl:
request bodies whose last user message is Unify's library line, then a '---' separator, then
the runner's message). Only the runner's message is used.
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cleanslate.memory import _words, bind  # noqa: E402

SEP = "\n\n---\n\n"


def runner_message(body: dict) -> str | None:
    """The runner's message among the request's user messages (Unify prefixes its library line)."""
    for msg in body.get("messages", []):
        content = msg.get("content")
        if not isinstance(content, str):
            content = "".join(p.get("text", "") for p in content or [] if isinstance(p, dict))
        if msg.get("role") == "user" and "Task id: task-" in content:
            return content.split(SEP, 1)[1] if SEP in content else content
    return None


def liftable(word: str) -> bool:
    return len(word) >= 2 and word not in ("0", "1")


def probe_pair(a: str, b: str) -> dict:
    wa, wb = _words(a), _words(b)
    ops = difflib.SequenceMatcher(None, [w.lower() for w in wa], [w.lower() for w in wb], autojunk=False).get_opcodes()
    differing = [w for tag, i1, i2, _, _ in ops if tag != "equal" for w in wa[i1:i2]]
    unliftable = [w for w in differing if not liftable(w)]
    params = {f"p{i}": w for i, w in enumerate(dict.fromkeys(w for w in differing if liftable(w)), 1)}
    bound, extra = bind(a, params, b)
    return {"differing_words": len(differing), "unliftable_differing": len(unliftable),
            "possible": not unliftable, "generous_bindable": bound is not None,
            "generous_shape_equal": bound is not None and not extra,
            "examples_unliftable": unliftable[:8]}


def arc_pairs(path: Path) -> list[tuple[str, str, str]]:
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    seen: dict[tuple, str] = {}
    pairs = []
    for r in rows:
        if r.get("bench") != "arc":
            continue
        msg = runner_message(r["body"])
        if msg is None:
            continue
        m = re.search(r"Task id: (task-[0-9a-f]+)", msg)
        if not m:
            continue
        key = (r.get("run_id"), m.group(1))
        if key in seen:
            pairs.append((m.group(1), seen[key], msg))
        seen[key] = msg
    return pairs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arc-first-turns", required=True)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    pairs = arc_pairs(Path(a.arc_first_turns))
    items = [{"task": t, **probe_pair(x, y)} for t, x, y in pairs]
    report = {"source": a.arc_first_turns, "pairs": len(items),
              "possible": sum(i["possible"] for i in items),
              "generous_bindable": sum(i["generous_bindable"] for i in items),
              "generous_shape_equal": sum(i["generous_shape_equal"] for i in items),
              "median_differing_words": sorted(i["differing_words"] for i in items)[len(items) // 2] if items else None,
              "items": items}
    text = json.dumps(report, indent=1)
    if a.out:
        Path(a.out).write_text(text + "\n")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())

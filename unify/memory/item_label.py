"""An item's evidence label: what is known about it, never a filter.

Every library item's record carries a label (``label`` in its item record, :mod:`.item_records`), built by the harness
from its own records each time the item records are written (:func:`build`):

* ``form``: ``function``, ``note`` or ``checklist`` (a note the writer stated is one);
* ``effects``: what the writer declared its calls do to the environment (``none``, ``reads``, ``writes`` or
  ``unknown``; ``declared`` says whether it declared one);
* ``covers``: the episodes it was built from, its evidence (``stated_by``: ``harness``; the writer never
  states episodes as an item's scope);
* ``situation``: the kind of situation it is for, as the writer stated it (None when it stated none);
* ``inputs_vs_constants``: what the writer said varies and what it holds fixed, each constant with its source;
* ``tests``: the writer's stated tests of its current version (:mod:`.writer_tests`), by outcome;
* ``uses``: its uses since its current version, clean, failed or neutral (:func:`.trust.judge`), and by regime;
* ``failure_record``: each failed use: the episode, the version served when it was called (``pin``), why
  (``error``, ``holds_false``, ``contrary_signal``, ``late_contrary`` or ``actor_reported``) and a ref, as serving
  gives it (:func:`.serving.item_evidence`), so the label and the card agree;
* ``findings``: the quality findings the gate recorded on it instead of refusing it;
* ``status`` and ``status_basis``: ``verdicts``, ``end_outcomes``, ``implicit`` or ``no_outcome_signals`` (no outcome
  signal has arrived for any use of it).

Nothing here groups, gates or refuses an item; the label is evidence for the writer, the librarian and the reports.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

VERSION = 1
DECLARED = ("none", "reads", "writes", "unknown")
TEST_OUTCOMES = ("as_stated", "not_as_stated", "error", "timeout", "not_run")
STATUS_BASES = ("verdicts", "end_outcomes", "implicit", "no_outcome_signals")


def tests_by_item(state_dir: Path | str | None) -> dict[str, list[dict]]:
    """Every finished stated test of every pass (``<state>/passes/<pass id>/tests.jsonl``), by item: each row the
    result with its ``pass_id``, ``mode`` and ``effects`` (the reads and writes its replay saw)."""
    from .writer_tests import TESTS_FILE

    out: dict[str, list[dict]] = {}
    if state_dir is None:
        return out
    root = Path(state_dir) / "passes"
    try:
        passes = sorted(p for p in root.iterdir() if p.is_dir() and not p.is_symlink())
    except OSError:
        return out
    for d in passes:
        stated: dict[str, dict] = {}
        try:
            lines = (d / TESTS_FILE).read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if not isinstance(row, dict):
                continue
            if row.get("phase") == "stated" and isinstance(row.get("test_id"), str):
                stated[row["test_id"]] = row
            elif row.get("phase") == "result" and row.get("test_id") in stated:
                s = stated[row["test_id"]]
                item = s.get("item")
                if isinstance(item, str):
                    out.setdefault(item, []).append(
                        {
                            "pass_id": s.get("pass_id") or d.name,
                            "outcome": row.get("outcome"),
                        },
                    )
    return out


def _tests(rows: Iterable[Mapping[str, Any]]) -> dict:
    out: dict = {k: 0 for k in TEST_OUTCOMES}
    for r in rows:
        if r.get("outcome") in out:
            out[r["outcome"]] += 1
    return out


def status_basis(track: Mapping[str, Any] | None, uses: Mapping[str, Any]) -> str:
    """What the item's status rests on: verdicts, end outcomes, implicit signals only, or no outcome signal at all."""
    t = track if isinstance(track, Mapping) else {}
    verdicts = t.get("verdicts") if isinstance(t.get("verdicts"), Mapping) else {}
    ends = t.get("end_outcomes") if isinstance(t.get("end_outcomes"), Mapping) else {}
    if sum(int(v or 0) for v in verdicts.values()):
        return "verdicts"
    if sum(int(v or 0) for v in ends.values()):
        return "end_outcomes"
    if int(uses.get("failed") or 0) or int(uses.get("clean") or 0):
        return "implicit"
    return "no_outcome_signals"


def build(
    *,
    kind: str,
    record: Mapping[str, Any],
    verification: Mapping[str, Any] | None,
    tests: Iterable[Mapping[str, Any]] = (),
    uses: Iterable[Any] = (),
    failures: Iterable[Mapping[str, Any]] = (),
    basis: str | None = None,
) -> dict:
    """The label of one item (module docstring). *verification* is its gate record (``stated``, ``label_findings``);
    *tests* its stated tests' results since its current version; *uses* its :class:`.trust.Use` values since then;
    *failures* its failed uses (``episode``, ``pin``, ``why``, ``ref``); *basis* what its status rests on as serving
    states it (:func:`.serving.item_evidence`), else derived here (:func:`status_basis`)."""
    from . import trust

    v = verification if isinstance(verification, Mapping) else {}
    stated = v.get("stated") if isinstance(v.get("stated"), Mapping) else {}
    tests = [dict(t) for t in tests]
    declared = stated.get("effects") if stated.get("effects") in DECLARED else None
    if kind == "function":
        form = "function"
    else:
        form = "checklist" if stated.get("form") == "checklist" else "note"
    counts = {"clean": 0, "failed": 0, "neutral": 0, "by_regime": {}}
    for u in uses:
        verdict, _why, _ = trust.judge(u)
        counts[verdict] = counts.get(verdict, 0) + 1
        counts["by_regime"][u.regime] = counts["by_regime"].get(u.regime, 0) + 1
    prov = record.get("provenance") if isinstance(record.get("provenance"), Mapping) else {}
    failure_rows = [dict(f) for f in failures]
    return {
        "v": VERSION,
        "form": form,
        "effects": {
            "declares": declared or ("none" if kind != "function" else "unknown"),
            "declared": declared is not None,
        },
        # the episodes it comes from are its evidence (the harness's provenance), never a scope the writer states;
        # the writer states the kind of situation it is for
        "covers": {"episodes": list(prov.get("episodes") or []), "stated_by": "harness"},
        "situation": stated.get("situation") if isinstance(stated.get("situation"), str) else None,
        "inputs_vs_constants": {
            "inputs": list(stated.get("inputs") or []),
            "constants": list(stated.get("constants") or []),
        },
        "tests": _tests(tests),
        "uses": counts,
        "failure_record": failure_rows,
        "findings": list(v.get("label_findings") or []),
        "status": record.get("status"),
        "status_basis": basis if basis in STATUS_BASES else status_basis(record.get("track"), counts),
    }

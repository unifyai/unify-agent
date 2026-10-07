# scratch: the clean-slate harness (snapshot, 7 Oct 2026)

This branch holds the clean-slate harness that MAIN built on 7 Oct 2026. It is a snapshot of `prototypes/cleanslate-v1/` in the research repo (`continual-harness-research`), copied unchanged. It is an orphan branch: it shares no history with `main` or `harness-learning`, and it does not import `unify`.

**Status.** Clean-slate is stopped (MAIN, 7 Oct 09:35Z) pending a redesign with no lexical mechanism. This branch preserves v1 as it stood; it is not a working baseline.

**Left out:**
- `frozen/`: the frozen office task snapshots;
- caches.

The benchmark content stays in the research repo, outside Git.

**Dependencies.**
- The code uses the standard library only; the tests need `pytest`. Bubblewrap is needed for the confinement tests.
- Six tests reach into the research repo by a path relative to this folder (`REPO = Path(__file__).resolve().parents[3]`):
  - the lab ledger's `journal_accounting.py`;
  - the office benchmark's `benchmarks/everyday/MANIFEST.json`.

**Tests, measured 7 Oct:**

| Where | Command | Result |
|---|---|---|
| In place, in the research repo | `python -m pytest tests` | 76 passed |
| This branch alone, research repo on `PYTHONPATH` | `PYTHONPATH=<research repo> python -m pytest tests` | 69 passed, 6 failed, 1 skipped |

The 6 failures on the branch alone are those relative-path lookups. Run the full suite from the research repo's copy.

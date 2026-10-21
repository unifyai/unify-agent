"""
Fixture test that records the environment parallel_run.sh gave its session.

It drops a JSON snapshot of the variables the runner is responsible for in
the run's log directory, as ``env-reports/<n>.json``. The runner's own tests
read the snapshots to check what each session received.
"""

import json
import os
import uuid
from pathlib import Path

REPORTED_VARS = (
    "UNILLM_CACHE",
    "UNIFY_TEST_SOCKET",
    "UNIFY_LOG_SUBDIR",
    "UNIFY_TMUX_SESSION_ID",
)


def test_report_env():
    # The results directory is private to the run and lives in its log directory.
    reports = Path(os.environ["UNIFY_TEST_RESULTS_DIR"]).parent / "env-reports"
    reports.mkdir(parents=True, exist_ok=True)
    snapshot = {name: os.environ.get(name) for name in REPORTED_VARS}
    (reports / f"{uuid.uuid4().hex}.json").write_text(json.dumps(snapshot), encoding="utf-8")

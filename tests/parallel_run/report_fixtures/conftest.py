"""
Minimal conftest for the result-reporting fixtures.

SKIP_UNIFY_TEST_INIT keeps the shared harness from activating a project.

With REPORT_FIXTURE_WROTE and REPORT_FIXTURE_RELEASE naming files, a session
holds its pytest process open once the shared harness has written the
session's result files: it creates the first file, then waits for the second.
The runner reads those results only after pytest exits, so a runner test can
finish another run's session inside that window.
"""

import os
import time
from pathlib import Path


def pytest_configure(config):
    os.environ["SKIP_UNIFY_TEST_INIT"] = "1"


def pytest_unconfigure(config):
    # Runs after pytest_sessionfinish and pytest_terminal_summary, which write
    # the result files. The runner's --collect-only pass writes no results
    # the runner reads, so it must not trip the hold.
    wrote = os.environ.get("REPORT_FIXTURE_WROTE")
    release = os.environ.get("REPORT_FIXTURE_RELEASE")
    if config.option.collectonly or not (wrote and release):
        return
    Path(wrote).touch()
    deadline = time.monotonic() + 300
    while not Path(release).exists() and time.monotonic() < deadline:
        time.sleep(0.05)

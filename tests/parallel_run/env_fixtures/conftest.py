"""
Minimal conftest for the environment-report fixture.

SKIP_UNIFY_TEST_INIT keeps the shared harness from initialising the runtime,
so the run stays fast enough to spawn many sessions at once.
"""

import os


def pytest_configure(config):
    os.environ["SKIP_UNIFY_TEST_INIT"] = "1"

"""Per-test time limits that hold even when a test never yields control.

``@pytest.mark.timeout(N)`` was registered but nothing enforced it: the
pytest-timeout plugin is not installed, so the marker was a label. On 2 Oct a
live test marked ``timeout(300)`` ran for over twenty minutes while
model-written code walked the filesystem inside the actor's event loop, where
no asyncio timeout can fire.

Each phase of a test (setup, call, teardown) now gets the test's limit: its
``timeout`` marker, else ``UNIFY_TEST_TIMEOUT`` seconds (default 900; 0
leaves unmarked tests unlimited). Two mechanisms enforce it:

* at the limit, ``SIGALRM`` fails the test with a timeout message. Its
  handler runs in the main thread between bytecodes, so it interrupts Python
  code, including a blocked event loop; it cannot interrupt one long C call,
  and an exception raised inside another asyncio task is kept by that task;
* ``UNIFY_TEST_TIMEOUT_GRACE`` seconds later (default 60), faulthandler's
  watchdog thread, which needs no GIL, writes every thread's traceback to
  stderr and ends the process with status 1. Under the test sandbox
  (tests/_test_sandbox.py) every process the test started ends with it.

The end of the session is bounded too: a test that timed out may leave
non-daemon threads (an executor running model code) that would keep the
interpreter from exiting, so after the session finishes the process has the
grace period to exit before the watchdog ends it.

``UNIFY_TEST_TIMEOUTS=off`` disables all of this, and so does ``--pdb``.
"""

from __future__ import annotations

import faulthandler
import os
import signal
import sys
import threading
from typing import Optional

import pytest

DEFAULT_TIMEOUT_S = 900.0
DEFAULT_GRACE_S = 60.0


def _number(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    return float(raw) if raw else default


def _enabled(config: pytest.Config) -> bool:
    if os.environ.get("UNIFY_TEST_TIMEOUTS", "on").strip().lower() in (
        "off",
        "0",
        "false",
        "no",
    ):
        return False
    return not (config.getoption("usepdb", False) or config.getoption("trace", False))


def limit_for(item: pytest.Item) -> Optional[float]:
    """Seconds each phase of *item* may take, or None for no limit."""
    if not _enabled(item.config):
        return None
    marker = item.get_closest_marker("timeout")
    if marker is not None and (marker.args or "seconds" in marker.kwargs):
        seconds = float(marker.args[0] if marker.args else marker.kwargs["seconds"])
    else:
        seconds = _number("UNIFY_TEST_TIMEOUT", DEFAULT_TIMEOUT_S)
    return seconds if seconds > 0 else None


# A descriptor for the terminal's stderr, taken while pytest's capture is
# suspended: during a test, descriptor 2 may point at a capture file that is
# lost when the watchdog ends the process.
_STDERR_FD: Optional[int] = None


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    global _STDERR_FD
    if _STDERR_FD is None:
        try:
            _STDERR_FD = os.dup(2)
        except OSError:
            _STDERR_FD = None


def _stderr() -> int:
    return _STDERR_FD if _STDERR_FD is not None else sys.__stderr__.fileno()


def _guarded(item: pytest.Item, phase: str):
    limit = limit_for(item)
    if (
        limit is None
        or not hasattr(signal, "SIGALRM")
        or threading.current_thread() is not threading.main_thread()
    ):
        return (yield)
    grace = _number("UNIFY_TEST_TIMEOUT_GRACE", DEFAULT_GRACE_S)

    def _expired(signum, frame):
        pytest.fail(
            f"Timeout: the {phase} of {item.nodeid} took more than {limit:g}s; "
            f"the process is killed if it is still running {grace:g}s later",
            pytrace=False,
        )

    previous = signal.signal(signal.SIGALRM, _expired)
    signal.setitimer(signal.ITIMER_REAL, limit)
    faulthandler.dump_traceback_later(limit + grace, exit=True, file=_stderr())
    try:
        return (yield)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        faulthandler.cancel_dump_traceback_later()


@pytest.hookimpl(wrapper=True)
def pytest_runtest_setup(item):
    return (yield from _guarded(item, "setup"))


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item):
    return (yield from _guarded(item, "call"))


@pytest.hookimpl(wrapper=True)
def pytest_runtest_teardown(item, nextitem):
    return (yield from _guarded(item, "teardown"))


@pytest.hookimpl(trylast=True)
def pytest_unconfigure(config):
    if _enabled(config) and threading.current_thread() is threading.main_thread():
        grace = _number("UNIFY_TEST_TIMEOUT_GRACE", DEFAULT_GRACE_S)
        faulthandler.dump_traceback_later(grace, exit=True, file=_stderr())

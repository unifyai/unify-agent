"""Pinned clock and randomness for the gate's confined runs (memory v2.1 stage 5, determinism).

Shipped into the gate's box as ``/inputs/_memv2_pin.py`` and loaded by pytest as a plugin (``-p _memv2_pin`` in
``PYTEST_ADDOPTS``) when ``UNIFY_MEMORY_V2_QA_DETERMINISM`` is on, and by the gate's probe whenever a stage-5
check runs. The box also gets ``PYTHONHASHSEED=0`` and ``TZ=UTC`` from the gate. Importing it installs:

* a clock that starts at :data:`EPOCH_NS` (2001-09-09T01:46:40Z) and advances :data:`STEP_NS` per read, so a
  loop on the clock still ends and two runs read the same sequence: ``time.time``, ``time.time_ns``, and
  ``datetime.datetime.now``/``utcnow``/``today`` (``datetime.datetime`` becomes a subclass whose ``now``
  reads the pinned clock; ``datetime.date.today`` already reads ``time.time``);
* ``random.seed(0)``.

It installs only under its box name ``_memv2_pin`` (or through :func:`install`): imported on the host as
``unify.memory_v2.pin`` it changes nothing.

Before each test (``pytest_runtest_setup``) the clock and the seed are reset, so a test's outcome does not
depend on the tests before it. Not pinned (documented limits): ``time.monotonic``/``perf_counter`` (pytest
times itself with them), ``os.urandom``, ``uuid.uuid4``, ``secrets``, a ``random.Random()`` seeded from the
system, and a module that bound ``datetime.datetime`` before the plugin loaded. Running each test file twice
catches what these leave nondeterministic. Standard library only.
"""

from __future__ import annotations

import datetime as _dt
import random
import time

EPOCH_NS = 1_000_000_000 * 10**9
STEP_NS = 1_000_000  # one millisecond per read
_state = {"reads": 0, "installed": False}


def _now_ns() -> int:
    _state["reads"] += 1
    return EPOCH_NS + STEP_NS * _state["reads"]


def _time() -> float:
    return _now_ns() / 1e9


class PinnedDateTime(_dt.datetime):
    """``datetime.datetime`` reading the pinned clock."""

    @classmethod
    def now(cls, tz=None):  # type: ignore[override]
        return cls.fromtimestamp(_time(), tz)

    @classmethod
    def utcnow(cls):  # type: ignore[override]
        return cls.fromtimestamp(_time(), _dt.timezone.utc).replace(tzinfo=None)

    @classmethod
    def today(cls):  # type: ignore[override]
        return cls.fromtimestamp(_time())


def reset() -> None:
    """The clock back to its start and the global random generator reseeded."""
    _state["reads"] = 0
    random.seed(0)


def install() -> None:
    """Pin the clock and randomness in this process (idempotent)."""
    if _state["installed"]:
        return
    _state["installed"] = True
    time.time = _time
    time.time_ns = _now_ns
    _dt.datetime = PinnedDateTime
    reset()


BOX_NAME = "_memv2_pin"
if __name__ == BOX_NAME:
    install()


def pytest_runtest_setup(item) -> None:  # noqa: ARG001 - pytest's hook signature
    if _state["installed"]:
        reset()

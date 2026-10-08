"""Pinned clock and randomness for the gate's confined runs (memory v2.1 stage 5, determinism).

Shipped into the gate's box as ``/inputs/_memv2_pin.py`` and loaded by pytest as a plugin (``-p _memv2_pin`` in
``PYTEST_ADDOPTS``) when ``UNIFY_MEMORY_V2_QA_DETERMINISM`` is on, and by the gate's probe whenever a stage-5
check runs. The box also gets ``PYTHONHASHSEED=0`` and ``TZ=UTC`` from the gate. Importing it installs:

* a clock that starts at the variant's epoch (:data:`EPOCHS_NS`) and advances :data:`STEP_NS` per read, so a
  loop on the clock still ends and two runs of one variant read the same sequence: ``time.time``,
  ``time.time_ns``, and ``datetime.datetime.now``/``utcnow``/``today`` (``datetime.datetime`` becomes a
  subclass whose ``now`` reads the pinned clock; ``datetime.date.today`` already reads ``time.time``);
* ``random.seed`` with the variant's seed (:data:`SEEDS`).

The variant is ``MEMV2_PIN`` in the environment (``0`` by default, or ``1``): 2001-09-09T01:46:40Z with seed 0,
or 2033-05-18T03:33:20Z with seed 1. The gate runs a new test file's first run under variant 0 and its second
under variant 1 (with ``PYTHONHASHSEED`` 0 and 1), so the pins make a run reproducible but never make a test
pass: a test that asserts a pinned value (the year, a random draw, a hash order) gives different outcomes on the
two runs and is refused. The pins therefore never become something a stored library depends on.

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
import os
import random
import time

EPOCHS_NS = (1_000_000_000 * 10**9, 2_000_000_000 * 10**9)  # per variant
SEEDS = (0, 1)
EPOCH_NS = EPOCHS_NS[0]
STEP_NS = 1_000_000  # one millisecond per read
VARIANT_ENV = "MEMV2_PIN"
_state = {"reads": 0, "installed": False, "variant": 0}


def variant() -> int:
    """The pin variant this process runs under (``MEMV2_PIN``: 1, else 0)."""
    return 1 if os.environ.get(VARIANT_ENV, "0").strip() == "1" else 0


def _now_ns() -> int:
    _state["reads"] += 1
    return EPOCHS_NS[_state["variant"]] + STEP_NS * _state["reads"]


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
    random.seed(SEEDS[_state["variant"]])


def install() -> None:
    """Pin the clock and randomness in this process (idempotent)."""
    if _state["installed"]:
        return
    _state["installed"] = True
    _state["variant"] = variant()
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

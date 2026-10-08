"""The memory v2 suite runs only where no real provider key is loadable (see conftest.py)."""

from __future__ import annotations

import os

from tests.memory_v2.conftest import PLACEHOLDER, PROVIDER_KEYS, REAL_KEY_SEEN


def test_no_real_provider_key_is_loadable():
    # names only, never values
    assert not REAL_KEY_SEEN, (
        "a real provider key was loadable on this host; run tests/memory_v2 only on keyless hosts: "
        + ", ".join(REAL_KEY_SEEN)
    )


def test_every_provider_key_is_the_placeholder_while_tests_run():
    assert all(os.environ.get(n) == PLACEHOLDER for n in PROVIDER_KEYS)

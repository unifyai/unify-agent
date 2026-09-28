"""
Fixture test that always skips.

Its session exits 0 having run nothing, so only the outcome counts the shared
harness records let the runner report it as skipped rather than passed.
"""

import pytest


def test_skip():
    pytest.skip("runs nothing by design")

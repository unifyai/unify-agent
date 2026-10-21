"""Assistant identity env overrides for CM integration tests."""

from __future__ import annotations

import os


def ensure_test_assistant_identity_env() -> None:
    """Pin the assistant identity CM tests boot with, ignoring machine state.

    ``ASSISTANT_ID`` is forced blank unconditionally: a developer's ``.env`` may
    pin an assistant id, and a CM boot inheriting it would bind its context
    root to that assistant instead of the default local identity the tests
    seed. ``populate_from_env`` skips blank values, so ``agent_id`` stays
    ``None`` and the boot binds the default local identity.
    """
    os.environ["ASSISTANT_ID"] = ""

"""The legacy conversation manager's fixtures, for its evals (model_evals/README.md): the test suite's own, imported
from its conftest. The suite's guard against provider calls (tests/conftest.py) is not imported."""

from tests.legacy.conversation_manager.conftest import *  # noqa: F401,F403 - the fixtures the evals take

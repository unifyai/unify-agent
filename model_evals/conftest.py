"""The fixtures the evals use (model_evals/README.md). This directory is not the test suite: nothing here is collected
by a plain ``pytest``, and the suite's guard against provider calls (tests/conftest.py) does not apply."""

from __future__ import annotations

import pytest

from unify.settings import SETTINGS

_DEFAULT_MODEL = SETTINGS.UNIFY_MODEL


@pytest.fixture(params=[pytest.param({"model": _DEFAULT_MODEL}, id=_DEFAULT_MODEL.replace("@", "-"))])
def llm_config(request) -> dict[str, str]:
    """The production-default model's client configuration."""
    return request.param

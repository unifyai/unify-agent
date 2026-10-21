# tests/async_tool_loop/test_hidden_parameters.py
from __future__ import annotations

import asyncio
import pytest

import unify.common.llm_helpers as llmh


# --------------------------------------------------------------------------- #
# Shared assertion helper                                                     #
# --------------------------------------------------------------------------- #
def _assert_internal_queues_hidden(fn) -> None:
    """
    Verify that *_internal_in*, *_internal_out*, and *_internal_note* never show up
    in the generated schema nor in the docstring description that is forwarded
    to the LLM.
    """
    schema = llmh.method_to_schema(fn, fn.__name__)
    props = schema["function"]["parameters"]["properties"]
    required = schema["function"]["parameters"]["required"]
    desc = schema["function"]["description"]

    for name in ("_internal_in", "_internal_out", "_internal_note"):
        # ––– must be absent from JSON schema –––
        assert name not in props
        assert name not in required
        # ––– and also stripped from the docstring forwarded to the model –––
        assert name not in desc


# --------------------------------------------------------------------------- #
# 1. Original “Google style” example (colon after each name)                  #
# --------------------------------------------------------------------------- #
def tool_google_style(
    a: int,
    _internal_in: asyncio.Queue[str],
    _internal_out: asyncio.Queue[str],
    _internal_note: asyncio.Queue | None = None,
) -> int:
    """
    Google-style.

    Args:
        a: Some value.
        _internal_in: internal.
        _internal_out: internal.
        _internal_note: internal.
    """
    return a


# --------------------------------------------------------------------------- #
# 2. New *slash-separated* NumPy variant: “name_a / name_b : type”            #
# --------------------------------------------------------------------------- #
def tool_numpy_slash(
    a: int,
    _internal_in: asyncio.Queue[str] | None = None,
    _internal_out: asyncio.Queue[str] | None = None,
    _internal_note: asyncio.Queue | None = None,
) -> int:
    """
    NumPy style — slash-separated synonyms.

    Parameters
    ----------
    a : int
        Some value.
    _internal_in / _internal_out / _internal_note : asyncio.Queue | None
        Internal queues.
    """
    return a


# --------------------------------------------------------------------------- #
# 3. New *comma-separated* variant: “name_a, name_b, …” (no type, no colon)   #
# --------------------------------------------------------------------------- #
def tool_numpy_commas(
    a: int,
    _internal_in: asyncio.Queue[str] | None = None,
    _internal_out: asyncio.Queue[str] | None = None,
    _internal_note: asyncio.Queue | None = None,
) -> int:
    """
    NumPy style — comma-separated, no type / no colon.

    Parameters
    ----------
    a : int
        Some value.

    _internal_in, _internal_out, _internal_note
        Internal queues.
    """
    return a


# --------------------------------------------------------------------------- #
# Parametrised test that covers all three doc-string flavours                 #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "fn",
    [tool_google_style, tool_numpy_slash, tool_numpy_commas],
    ids=["google", "numpy-slash", "numpy-commas"],
)
def test_hidden_parameters_are_removed_from_schema_and_docs(fn) -> None:
    """
    Any internal queues must be invisible to the API schema and the LLM
    description, regardless of how they are documented in the original
    docstring (Google, NumPy-slash, or NumPy-commas).
    """
    _assert_internal_queues_hidden(fn)

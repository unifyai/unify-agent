"""The globals of the actor's cells: safe builtins, common standard library modules and typing names, pydantic,
the model helpers (``query_llm``, ``list_llms``, ``unillm``), ``run_coro_sync``, and the globals a registered
environment binds."""

import asyncio
import collections
import datetime
import functools
import json
import re
import typing
from typing import Any, Dict

try:
    import pydantic
    from pydantic import BaseModel, Field

    HAS_PYDANTIC = True
except ImportError:
    pydantic = None
    BaseModel = None
    Field = None
    HAS_PYDANTIC = False


def create_base_globals() -> Dict[str, Any]:
    """
    Creates a dictionary of safe global functions for code execution.

    This provides a controlled environment with:
    - Safe built-in functions (excluding dangerous ones like eval, exec)
    - Common standard library modules (asyncio, re, json, datetime, collections)
    - Typing module and common type hints
    - Pydantic support (if available)

    Returns:
        A dictionary of globals allowed within the execution environment.
    """
    # Safe builtins - excluding dangerous functions
    safe_builtins = {
        k: __builtins__.get(k)
        for k in [
            "repr",
            "print",
            "len",
            "str",
            "int",
            "float",
            "bool",
            "list",
            "dict",
            "set",
            "tuple",
            "range",
            "type",
            "object",
            "bytes",
            "frozenset",
            "isinstance",
            "hasattr",
            "getattr",
            "setattr",
            "callable",
            "issubclass",
            "dir",
            # The prompt's introspection bridge teaches `help(...)` for
            # live docs in-sandbox; site injects it into builtins.
            "help",
            "vars",
            "iter",
            "next",
            "filter",
            "map",
            "reversed",
            "enumerate",
            "zip",
            "any",
            "all",
            "sum",
            "min",
            "max",
            "abs",
            "round",
            "pow",
            "divmod",
            "sorted",
            "format",
            "chr",
            "ord",
            "id",
            "hash",
            "hex",
            "oct",
            "bin",
            "bytearray",
            "memoryview",
            "complex",
            "slice",
            # Exception hierarchy
            "Exception",
            "BaseException",
            "ArithmeticError",
            "LookupError",
            "NotImplementedError",
            "ValueError",
            "TypeError",
            "KeyError",
            "IndexError",
            "AttributeError",
            "RuntimeError",
            "StopIteration",
            "StopAsyncIteration",
            "AssertionError",
            "NameError",
            "ImportError",
            "ModuleNotFoundError",
            "OSError",
            "IOError",
            "FileNotFoundError",
            "FileExistsError",
            "PermissionError",
            "IsADirectoryError",
            "NotADirectoryError",
            "EOFError",
            "UnicodeError",
            "UnicodeDecodeError",
            "UnicodeEncodeError",
            "ZeroDivisionError",
            "OverflowError",
            "TimeoutError",
            "ConnectionError",
            "BrokenPipeError",
            # File I/O
            "open",
            # Class-related
            "super",
            "property",
            "classmethod",
            "staticmethod",
            "__build_class__",
            "__name__",
            "__import__",
        ]
        if __builtins__.get(k) is not None
    }

    globals_dict = {
        "__builtins__": safe_builtins,
        # Standard library modules
        "asyncio": asyncio,
        "re": re,
        "json": json,
        "datetime": datetime,
        "collections": collections,
        # Additional useful modules
        "statistics": __import__("statistics"),
        # Typing module and common types
        "typing": typing,
        "Any": typing.Any,
        "Callable": typing.Callable,
        "Dict": typing.Dict,
        "List": typing.List,
        "Optional": typing.Optional,
        "Tuple": typing.Tuple,
        "Set": typing.Set,
        "Union": typing.Union,
        "Literal": typing.Literal,
        "functools": functools,
    }

    if HAS_PYDANTIC:
        globals_dict.update(
            {
                "pydantic": pydantic,
                "BaseModel": BaseModel,
                "Field": Field,
            },
        )

    return globals_dict


def create_execution_globals() -> Dict[str, Any]:
    """
    Creates the globals of the actor's cells.

    Extends create_base_globals() with:
    - `unillm` for advanced direct LLM usage (client constructors for the
      endpoints UNIFY_CELL_LLM_MODELS allows)
    - The `query_llm` helper for focused one-shot LLM queries (and
      `list_llms`), held to the same endpoints
    - The `run_coro_sync` helper for sync façades that must drive async work
      under an already-running event loop

    Returns:
        A dictionary of globals for the cells.
    """
    globals_dict = create_base_globals()

    from unify.common.asyncio_compat import run_coro_sync
    from unify.common.cell_models import (
        ModelCallFailed,
        ModelRequestError,
        cell_list_llms,
        cell_query_llm,
        cell_unillm,
    )

    # Cell-scoped: each refuses an endpoint UNIFY_CELL_LLM_MODELS does not
    # allow (unify/common/cell_models.py); the harness's own query_llm,
    # list_llms and unillm are unchanged.
    globals_dict["query_llm"] = cell_query_llm
    globals_dict["list_llms"] = cell_list_llms
    globals_dict["run_coro_sync"] = run_coro_sync
    globals_dict["unillm"] = cell_unillm
    # What a cell's model call raises when the provider refuses the request's
    # own content, so the cell can catch it by name.
    globals_dict["ModelRequestError"] = ModelRequestError
    # What it raises for any other failure of the call: a fixed text with the
    # status and the error's type, never the provider's reply.
    globals_dict["ModelCallFailed"] = ModelCallFailed

    # Globals a registered environment binds beside its namespaces; never in
    # place of one of the above.
    from unify.function_manager.primitives.environment import environment_globals

    for name, value in environment_globals().items():
        globals_dict.setdefault(name, value)

    return globals_dict

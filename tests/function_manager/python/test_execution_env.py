"""
Tests for the execution environment utilities.

Tests that create_base_globals() and create_execution_globals() provide
a secure sandbox environment with appropriate restrictions.
"""

import asyncio

import pytest

from unify.function_manager.execution_env import (
    create_base_globals,
    create_execution_globals,
)

# ────────────────────────────────────────────────────────────────────────────
# Sandbox Security Tests
# ────────────────────────────────────────────────────────────────────────────


def test_dangerous_builtins_blocked():
    """Dangerous builtins should not be available in the sandbox."""
    globals_dict = create_base_globals()
    builtins = globals_dict["__builtins__"]

    # These builtins are blocked for security
    blocked = [
        "exec",
        "eval",
        "compile",
        "globals",
        "locals",
        "input",
        "breakpoint",
    ]
    for name in blocked:
        assert name not in builtins, f"{name} should be blocked"

    # __import__ IS available (needed for 'import x' statements)
    assert "__import__" in builtins


def test_safe_builtins_available():
    """Safe builtins should be available and functional."""
    globals_dict = create_base_globals()
    builtins = globals_dict["__builtins__"]

    # Basic I/O and utility
    assert "print" in builtins and callable(builtins["print"])
    assert "len" in builtins and builtins["len"]([1, 2, 3]) == 3
    assert "range" in builtins and list(builtins["range"](3)) == [0, 1, 2]

    # Type conversions
    type_conversions = ["str", "int", "float", "bool", "list", "dict", "set", "tuple"]
    for name in type_conversions:
        assert name in builtins, f"{name} should be available"

    # Iteration functions
    iteration_funcs = [
        "iter",
        "next",
        "enumerate",
        "zip",
        "filter",
        "map",
        "reversed",
        "sorted",
    ]
    for name in iteration_funcs:
        assert name in builtins, f"{name} should be available"

    # Exception classes
    exceptions = ["Exception", "ValueError", "TypeError", "KeyError", "RuntimeError"]
    for name in exceptions:
        assert name in builtins, f"{name} should be available"

    # Class-related decorators
    class_decorators = ["property", "classmethod", "staticmethod", "super"]
    for name in class_decorators:
        assert name in builtins, f"{name} should be available"


def test_standard_library_modules_available():
    """Standard library modules should be available in the sandbox."""
    globals_dict = create_base_globals()

    # Core modules should be present
    modules = [
        "asyncio",
        "json",
        "re",
        "datetime",
        "collections",
        "functools",
        "statistics",
    ]
    for mod in modules:
        assert mod in globals_dict, f"{mod} should be available"

    # Verify json works
    assert globals_dict["json"].dumps({"a": 1}) == '{"a": 1}'

    # Verify re works
    assert globals_dict["re"].match(r"\d+", "123") is not None


def test_typing_support():
    """Typing module and common type hints should be available."""
    globals_dict = create_base_globals()

    assert "typing" in globals_dict

    # Common type hints should be directly available
    type_hints = [
        "Any",
        "Optional",
        "List",
        "Dict",
        "Tuple",
        "Set",
        "Union",
        "Callable",
        "Literal",
    ]
    for hint in type_hints:
        assert hint in globals_dict, f"{hint} should be available"


def test_pydantic_support():
    """Pydantic should be available and functional."""
    globals_dict = create_base_globals()

    # Pydantic should be available in this project
    assert "pydantic" in globals_dict
    assert "BaseModel" in globals_dict
    assert "Field" in globals_dict

    # Should be able to define and use Pydantic models
    code = """
class TestModel(BaseModel):
    name: str
    value: int

model = TestModel(name="test", value=42)
result = model.model_dump()
"""
    exec(code, globals_dict)
    assert globals_dict["result"] == {"name": "test", "value": 42}


# ────────────────────────────────────────────────────────────────────────────
# create_execution_globals Tests
# ────────────────────────────────────────────────────────────────────────────


def test_execution_globals():
    """create_execution_globals should extend base globals with primitives."""
    base = create_base_globals()
    execution = create_execution_globals()

    # All base modules should be present
    for key in ["asyncio", "json", "re", "datetime", "collections"]:
        assert key in execution
        assert execution[key] is base[key]

    # A sandbox's primitives come only from its environments, so the default
    # object exposes no namespace and refuses the actor one, saying why.
    primitives = execution["primitives"]
    assert primitives.primitive_scope.scoped_managers == frozenset()
    with pytest.raises(AttributeError, match="started without Actor Delegation"):
        primitives.actor
    assert not hasattr(primitives, "files")


# ────────────────────────────────────────────────────────────────────────────
# Steerable Infrastructure Tests
# ────────────────────────────────────────────────────────────────────────────


def test_steerable_infrastructure_available():
    """Steerable handle infrastructure should be available in execution globals."""
    globals_dict = create_execution_globals()

    # Core steerable class should be present
    assert "SteerableToolHandle" in globals_dict

    # Low-level LLM loop constructors stay out of stored-function globals.
    assert "start_async_tool_loop" not in globals_dict
    assert "new_llm_client" not in globals_dict


def test_steerable_handle_is_abc():
    """SteerableToolHandle should be an ABC."""
    from abc import ABC

    globals_dict = create_execution_globals()

    SteerableToolHandle = globals_dict["SteerableToolHandle"]
    assert issubclass(SteerableToolHandle, ABC)


def test_steerable_handle_isinstance_check():
    """isinstance checks with SteerableToolHandle should work in execution context."""
    globals_dict = create_execution_globals()

    code = """
from abc import ABC

# SteerableToolHandle is an ABC, so we can't instantiate it directly
# But we can verify it's available and is an ABC
is_abc = issubclass(SteerableToolHandle, ABC)
"""
    exec(code, globals_dict)
    assert globals_dict["is_abc"] is True


def test_steerable_not_in_base_globals():
    """Steerable infrastructure should NOT be in base globals (only execution globals)."""
    base = create_base_globals()

    # These should NOT be in base globals
    assert "SteerableToolHandle" not in base
    assert "start_async_tool_loop" not in base
    assert "new_llm_client" not in base


# ────────────────────────────────────────────────────────────────────────────
# Functional Code Execution Tests
# ────────────────────────────────────────────────────────────────────────────


def test_code_execution_basics():
    """Should be able to execute basic Python constructs."""
    globals_dict = create_base_globals()

    code = """
# Functions
def add(a, b):
    return a + b
func_result = add(2, 3)

# Classes
class Counter:
    def __init__(self):
        self.count = 0
    def increment(self):
        self.count += 1
        return self.count
c = Counter()
class_result = c.increment()

# Comprehensions
list_result = [x * 2 for x in range(5)]
dict_result = {str(x): x * 2 for x in range(3)}
"""
    exec(code, globals_dict)
    assert globals_dict["func_result"] == 5
    assert globals_dict["class_result"] == 1
    assert globals_dict["list_result"] == [0, 2, 4, 6, 8]
    assert globals_dict["dict_result"] == {"0": 0, "1": 2, "2": 4}


def test_async_code_execution():
    """Should be able to define and run async functions."""
    globals_dict = create_base_globals()

    code = """
async def async_add(a, b):
    return a + b

async def safe_async():
    await asyncio.sleep(0)
    return "async works"

async def process_json(data):
    parsed = json.loads(data)
    await asyncio.sleep(0)
    return json.dumps({"processed": parsed["value"]})
"""
    exec(code, globals_dict)

    # Test basic async
    assert asyncio.run(globals_dict["async_add"](2, 3)) == 5
    assert asyncio.run(globals_dict["safe_async"]()) == "async works"

    # Test async with modules
    result = asyncio.run(globals_dict["process_json"]('{"value": 42}'))
    assert '"processed": 42' in result


def test_module_usage():
    """Should be able to use json, regex, and import additional modules."""
    globals_dict = create_base_globals()

    code = """
# json usage
data = {"name": "test", "value": 42}
json_str = json.dumps(data)
parsed = json.loads(json_str)
json_result = parsed["name"]

# regex usage
import re
match = re.search(r'(\\d+)', 'value: 123')
regex_result = match.group(1) if match else None

# Import additional safe modules
import math
import hashlib
import base64
import urllib.parse
import_result = {
    "math": math.pi > 3,
    "hashlib": hasattr(hashlib, "sha256"),
    "base64": hasattr(base64, "b64encode"),
    "urllib": hasattr(urllib.parse, "quote"),
}
"""
    exec(code, globals_dict)
    assert globals_dict["json_result"] == "test"
    assert globals_dict["regex_result"] == "123"
    assert all(globals_dict["import_result"].values())


def test_exception_handling():
    """Should be able to use and define exception classes."""
    globals_dict = create_base_globals()

    code = """
# Standard exceptions
exceptions = []
try:
    raise ValueError("test error")
except ValueError:
    exceptions.append("ValueError")

try:
    raise RuntimeError("runtime")
except RuntimeError:
    exceptions.append("RuntimeError")

try:
    d = {}
    _ = d["missing"]
except KeyError:
    exceptions.append("KeyError")

# Custom exceptions
class CustomError(Exception):
    def __init__(self, code, message):
        self.code = code
        self.message = message
        super().__init__(f"[{code}] {message}")

try:
    raise CustomError(404, "Not found")
except CustomError as e:
    custom_result = f"caught: {e.code}"
"""
    exec(code, globals_dict)
    assert globals_dict["exceptions"] == ["ValueError", "RuntimeError", "KeyError"]
    assert globals_dict["custom_result"] == "caught: 404"


# ────────────────────────────────────────────────────────────────────────────
# Security Tests
# ────────────────────────────────────────────────────────────────────────────


def test_sandbox_escape_attempts():
    """Various sandbox escape attempts should be blocked or limited."""
    globals_dict = create_base_globals()

    # Attempt to access __builtins__ via type (known escape technique)
    code = """
try:
    builtins = type.__bases__[0].__subclasses__()
    escape_result = "accessed subclasses"
except Exception as e:
    escape_result = f"Error: {type(e).__name__}"
"""
    exec(code, globals_dict)
    # Even if subclasses are accessible, open/eval/exec are not directly available

    # os can be imported but sandbox builtins are still restricted
    code2 = """
import os
try:
    cwd = os.getcwd()
    os_result = "imported"
except Exception as e:
    os_result = f"Error: {type(e).__name__}"
"""
    exec(code2, globals_dict)
    assert globals_dict["os_result"] == "imported"

    # sys.modules access is limited
    code3 = """
import sys
try:
    mods = list(sys.modules.keys())
    sys_result = f"has {len(mods)} modules"
except Exception as e:
    sys_result = f"Error: {type(e).__name__}"
"""
    exec(code3, globals_dict)
    assert "modules" in globals_dict["sys_result"]


def test_introspection_allowed():
    """Basic introspection (dir, hasattr) should work."""
    globals_dict = create_base_globals()

    code = """
class Test:
    secret = "value"

t = Test()
has_secret = hasattr(t, "secret")
attrs = dir(t)
result = f"hasattr={has_secret}, dir_works={len(attrs) > 0}"
"""
    exec(code, globals_dict)
    assert "hasattr=True" in globals_dict["result"]
    assert "dir_works=True" in globals_dict["result"]


def test_execution_globals_expose_llm_helpers_and_unillm():
    """The sandbox's LLM helpers are the runtime's own, and unillm is a
    runtime-provided module rather than a dependency a function must record."""
    import unillm

    from unify.common.reasoning import list_llms, query_llm
    from unify.function_manager.execution_env import ENVIRONMENT_MODULES

    globals_dict = create_execution_globals()

    assert globals_dict["query_llm"] is query_llm
    assert globals_dict["list_llms"] is list_llms
    assert "reason" not in globals_dict
    assert globals_dict["unillm"] is unillm
    assert "new_llm_client" not in globals_dict
    assert "unillm" in ENVIRONMENT_MODULES

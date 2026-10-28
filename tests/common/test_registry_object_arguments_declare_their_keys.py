"""Every object a tool takes declares the keys it takes.

The argument checker refuses any name an object does not declare (``tool_channel.first_problem``), so an object
argument with no ``properties`` refuses every call that gives it a key. The writer's ``tools.test`` had such an
``expect`` and every call of it was refused ("there is no argument 'kind' in expect"), though its own text shows
``{'kind': 'returns', ...}``; ``land_capture``'s cases had the same shape.
"""

from __future__ import annotations

from unify.common.tool_channel import first_problem
from unify.common.tool_registry import registry


def _objects_without_keys(schema, at="arguments"):
    if not isinstance(schema, dict):
        return []
    out = []
    if schema.get("type") == "object" and "properties" not in schema:
        out.append(at)
    for name, sub in (schema.get("properties") or {}).items():
        out += _objects_without_keys(sub, f"{at}.{name}")
    if isinstance(schema.get("items"), dict):
        out += _objects_without_keys(schema["items"], f"{at}[]")
    for i, sub in enumerate(schema.get("anyOf") or []):
        out += _objects_without_keys(sub, f"{at}|{i}")
    return out


def test_no_object_argument_refuses_every_key():
    found = {name: bad for name, spec in registry().items() if (bad := _objects_without_keys(spec.parameters))}
    assert not found, found


def test_the_writers_stated_test_and_extra_cases_pass_the_checker():
    reg = registry()
    for expect in ({"kind": "returns", "value": [6, 9]}, {"kind": "raises", "exception": "KeyError"}, {"kind": "refuses"}):
        args = {"item": "memory.a.b:c", "on": "e1/cell:2", "expect": expect}
        assert first_problem(reg["test"].parameters, args) is None, expect
    assert first_problem(reg["test"].parameters, {"item": "x", "on": "e1", "expect": {"kind": "maybe"}}) is not None
    cases = [{"input": "e1/observation:0", "expected": {"source": "e1/action:1", "slice": "total"}}]
    args = {"capture_id": "a" * 16, "package": "p", "name": "n", "cases": cases}
    assert first_problem(reg["land_capture"].parameters, args) is None

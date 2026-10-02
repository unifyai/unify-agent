"""Public storage obtains metadata without executing proposed definitions.

Loadability checks and verification are explicitly off in these tests; their
separate execution authority is not changed by metadata inspection.
"""

import hashlib

import pytest

from tests.helpers import _handle_project
from unify.function_manager.function_manager import FunctionManager
from unify.settings import SETTINGS


@pytest.fixture
def metadata_only(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_STORE_CHECK", "")
    monkeypatch.setattr(SETTINGS, "UNIFY_STORE_VERIFY", "")


@pytest.mark.parametrize(
    "position",
    ["default", "annotation", "return", "decorator", "body"],
)
@_handle_project
def test_add_functions_does_not_execute_for_metadata(metadata_only, capsys, position):
    marker = "PROPOSED_DEFINITION_EXECUTED"
    expression = f"print({marker!r})"
    sources = {
        "default": f"def metadata_probe(value={expression}):\n    pass",
        "annotation": f"def metadata_probe(value: {expression}):\n    pass",
        "return": f"def metadata_probe() -> {expression}:\n    pass",
        "decorator": f"@{expression}\ndef metadata_probe():\n    pass",
        "body": f"def metadata_probe():\n    {expression}",
    }
    source = sources[position]
    manager = FunctionManager()
    assert manager.add_functions(implementations=source) == {"metadata_probe": "added"}
    stored = manager.list_functions(include_implementations=True)["metadata_probe"]
    assert stored["argspec"] == ("()" if position == "body" else "(...)")
    assert stored["docstring"] == ""
    assert stored["function_id"] is not None
    assert stored["implementation"] == source
    assert (
        hashlib.sha256(stored["implementation"].encode()).digest()
        == hashlib.sha256(source.encode()).digest()
    )
    assert marker not in capsys.readouterr().out


@_handle_project
def test_add_functions_rejects_module_statements_without_execution(
    metadata_only,
    capsys,
):
    marker = "PROPOSED_MODULE_EXECUTED"
    source = f"print({marker!r})\ndef metadata_probe():\n    pass"
    manager = FunctionManager()
    with pytest.raises(ValueError, match="exactly one top-level function"):
        manager.add_functions(implementations=source)
    assert "metadata_probe" not in manager.list_functions(include_implementations=True)
    assert marker not in capsys.readouterr().out


@pytest.mark.parametrize("prefix", ["def", "async def"], ids=["sync", "async"])
@_handle_project
def test_stored_metadata_preserves_signature_and_clean_docstring(metadata_only, prefix):
    source = (
        f"{prefix} metadata_probe(a: int, /, b=1, *items, enabled: bool = True, **options) -> str:\n"
        '    """Summary.\n\n'
        "        Details are indented.\n"
        '    """\n'
        "    return str(a)\n"
    )
    manager = FunctionManager()
    assert manager.add_functions(implementations=source) == {"metadata_probe": "added"}
    stored = manager.list_functions(include_implementations=True)["metadata_probe"]
    assert stored["argspec"] == (
        "(a: int, /, b=1, *items, enabled: bool = True, **options) -> str"
    )
    assert stored["docstring"] == "Summary.\n\nDetails are indented."
    assert stored["implementation"] == source
    assert stored["function_id"] is not None

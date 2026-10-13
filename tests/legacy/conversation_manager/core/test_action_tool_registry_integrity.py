"""Every tool ``as_tools`` can register must actually exist on the class.

``as_tools`` binds tools by attribute (``tools["x"] = self.x``), so a method that
is renamed or deleted while its registration survives raises AttributeError --
but only once the slow brain builds its tool dict mid-conversation. Nothing
imports or type-checks its way into that path, so the break reaches a live
assistant and surfaces as a crashed LLM turn.

This reads the registrations statically rather than calling ``as_tools``, which
needs a whole ConversationManager to construct.
"""

import ast
import pathlib

import unify.legacy.conversation_manager.domains.brain_action_tools as module
from unify.legacy.conversation_manager.domains.brain_action_tools import (
    ConversationManagerBrainActionTools,
)

SOURCE = pathlib.Path(module.__file__)


def _registered_tool_attributes() -> dict[str, str]:
    """Map ``tool name -> attribute name`` for every ``tools[...] = self.x``."""
    tree = ast.parse(SOURCE.read_text())
    found: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not (
            isinstance(target, ast.Subscript)
            and isinstance(target.value, ast.Name)
            and target.value.id == "tools"
            and isinstance(target.slice, ast.Constant)
            and isinstance(target.slice.value, str)
        ):
            continue
        # Bare ``self.x``; wrapped forms such as ``self._with_doc_suffix(...)``
        # are skipped, since the attribute they wrap is not this node.
        value = node.value
        if isinstance(value, ast.Attribute) and isinstance(value.value, ast.Name):
            if value.value.id == "self":
                found[target.slice.value] = value.attr
    return found


def test_registrations_were_found() -> None:
    """Guard the guard: a parser that finds nothing would pass vacuously.

    The tool registered once the managers are initialized is pinned by
    name, so a parse that silently found nothing cannot pass.
    """
    registered = _registered_tool_attributes()
    assert {"act"} <= set(registered)


def test_every_registered_tool_exists() -> None:
    missing = {
        name: attr
        for name, attr in _registered_tool_attributes().items()
        if not hasattr(ConversationManagerBrainActionTools, attr)
    }
    assert not missing, f"registered but not defined: {missing}"

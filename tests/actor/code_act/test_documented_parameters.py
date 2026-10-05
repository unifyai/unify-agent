"""Symbolic: a tool documents only the parameters it takes.

``search_functions``'s contract offers ``include_dormant``. The function
manager takes it, but the actor's ``FunctionManager_search_functions`` tool
and the core surface's ``functions.search`` do not: on the JSON surface a
call passing it is refused as an unknown argument, and in a core-surface
cell it raises ``TypeError``. Each copy of the contract leaves it out where
the parameter is not taken; the function manager's own contract keeps it.
"""

from __future__ import annotations

import inspect
import re

from unify.actor import core_surface
from unify.common.llm_helpers import method_to_schema
from unify.function_manager.base import (
    BaseFunctionManager,
    search_doc_without_dormant,
)
from unify.function_manager.function_manager import FunctionManager


def _documented(doc: str) -> set:
    """The names under a numpydoc ``Parameters`` heading of *doc*."""
    doc = inspect.cleandoc(doc or "")
    if "Parameters\n----------" not in doc:
        return set()
    section = doc.split("Parameters\n----------", 1)[1]
    section = re.split(r"\n\n[A-Z][a-z]+\n-+", section)[0]
    return set(re.findall(r"^(\w+) :", section, re.M))


def _actor_tools():
    from unify.actor.code_act_actor import CodeActActor

    return CodeActActor(environments={}).get_tools("act")


def test_every_documented_parameter_of_an_actor_tool_is_in_its_schema():
    missing = {}
    for name, tool in _actor_tools().items():
        schema = method_to_schema(getattr(tool, "fn", tool), name)["function"]
        extra = _documented(schema["description"]) - set(
            schema["parameters"]["properties"],
        )
        if extra:
            missing[name] = sorted(extra)
    assert missing == {}


def test_the_search_tool_does_not_offer_include_dormant():
    tool = _actor_tools()["FunctionManager_search_functions"]
    schema = method_to_schema(
        getattr(tool, "fn", tool),
        "FunctionManager_search_functions",
    )["function"]
    assert "include_dormant" not in schema["parameters"]["properties"]
    assert "include_dormant" not in schema["description"]
    # The rest of the contract is kept.
    assert "always see the whole store). Freshly stored" in " ".join(
        schema["description"].split(),
    )
    assert "include_implementations" in schema["description"]


def test_every_documented_parameter_of_a_core_library_method_is_taken():
    core_surface._document()
    missing = {}
    for library in (core_surface.FunctionLibrary, core_surface.GuidanceLibrary):
        for attr in dir(library):
            if attr.startswith("_"):
                continue
            method = getattr(library, attr)
            if not callable(method):
                continue
            extra = _documented(inspect.getdoc(method) or "") - set(
                inspect.signature(method).parameters,
            )
            if extra:
                missing[f"{library.__name__}.{attr}"] = sorted(extra)
    assert missing == {}
    assert "include_dormant" not in core_surface.help_text(
        core_surface.FunctionLibrary.search,
        "functions.search",
    )


def test_the_function_manager_keeps_documenting_what_it_takes():
    assert (
        "include_dormant"
        in inspect.signature(
            FunctionManager.search_functions,
        ).parameters
    )
    assert "include_dormant" in _documented(
        BaseFunctionManager.search_functions.__doc__,
    )
    assert "include_dormant" in (FunctionManager.search_functions.__doc__ or "")


def test_only_the_dormant_text_is_removed():
    doc = BaseFunctionManager.search_functions.__doc__

    def flat(text: str) -> str:
        return " ".join(text.split())

    expected = flat(doc).replace(
        ", and ``include_dormant=True`` brings them back here)",
        ")",
    )
    expected = re.sub(
        r" include_dormant : bool, default ``False`` .*? recently\.",
        "",
        expected,
    )
    assert flat(search_doc_without_dormant(doc)) == expected != flat(doc)

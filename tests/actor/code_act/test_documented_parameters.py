"""Symbolic: a tool, and ``help()`` in a core-surface cell, documents only the parameters it takes.

``search_functions``'s contract offers ``include_dormant``. The function
manager takes it, but the core surface's ``functions.search`` does not: in a
core-surface cell it raises ``TypeError``. Its copy of the contract leaves it
out; the function manager's own contract keeps it.

``help(functions.search)`` in a core-surface cell also described what the
harness-only ``_return_callable``, ``_namespace`` and ``_also_return_metadata``
select (callables instead of rows, a dict of both, and the errors for passing
them wrongly), though a cell cannot pass them. That text is left out too.
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


PRIVATE = ("_return_callable", "_namespace", "_also_return_metadata")


def _core_helps():
    core_surface._document()
    out = {}
    for label, cls in (
        ("functions", core_surface.FunctionLibrary),
        ("guidance", core_surface.GuidanceLibrary),
    ):
        obj = cls.__new__(cls)
        out[label] = core_surface.help_text(obj, label)
        for attr in dir(obj):
            if not attr.startswith("_") and callable(getattr(obj, attr)):
                out[f"{label}.{attr}"] = core_surface.help_text(
                    getattr(obj, attr),
                    f"{label}.{attr}",
                )
    return out


def test_core_help_names_no_parameter_a_cell_cannot_pass():
    helps = _core_helps()
    found = {
        label: [w for w in (*PRIVATE, "include_dormant") if w in text]
        for label, text in helps.items()
    }
    assert {k: v for k, v in found.items() if v} == {}


def test_core_help_keeps_what_the_default_mode_returns():
    text = _core_helps()["functions.search"]
    flat = " ".join(text.split())
    assert "- Up to ``n`` results, best match first." in flat
    assert "list[Callable" in flat  # the declared return type is as written
    assert "Raises" not in text  # its only entries were about private modes
    # The blank line before Returns, which the skipped entries took, is kept.
    assert "payload size.\n\n    Returns" in text
    assert len(text) < 2700


def test_public_doc_keeps_a_contract_without_private_parameters():
    doc = """Do a thing.

    Parameters
    ----------
    x : int
        The input.

    Returns
    -------
    int
        - When ``flag``: twice ``x``.

    Raises
    ------
    ValueError
        If ``x`` is negative.
    """
    assert core_surface.public_doc(doc) == inspect.cleandoc(doc)

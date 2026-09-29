"""Symbolic: with ``UNIFY_SEARCH_SKIP_UNLOADABLE`` one broken row no longer fails a search.

A search that loads its results (as the actor's search tool always does)
executes every returned function into the sandbox. Shipped, one row that
cannot load (a syntax error, a dependency that cannot be installed) raises out
of the whole search, so every later search in the library fails the same
way. With the switch on, that row is left out, the rest load and come back,
and a trailing warning names the row and its error. List and filter behave
the same. Vectors come from a deterministic bag-of-words embedder, so no
model is called.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np
import pytest

from tests.helpers import _handle_project
from unify.common import embeddings
from unify.common.embeddings import Embedder
from unify.function_manager.function_manager import FunctionManager
from unify.settings import SETTINGS

GOOD = (
    "def forecast_summary(city: str) -> str:\n"
    '    """Summarise the forecast for a city."""\n'
    "    return f'{city}: sunny'\n"
)


def _bag_of_words(texts: list[str]) -> np.ndarray:
    vectors = np.zeros((len(texts), 256), dtype=np.float32)
    for row, text in enumerate(texts):
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            vectors[row, int(hashlib.sha256(word.encode()).hexdigest(), 16) % 256] += 1
        vectors[row, 0] += 1e-3  # never a zero vector
    return vectors


@pytest.fixture(autouse=True)
def local_vectors(monkeypatch):
    monkeypatch.setattr(
        embeddings,
        "embedder",
        lambda: Embedder("tests-bag-of-words/256", _bag_of_words),
    )


def _library() -> FunctionManager:
    fm = FunctionManager()
    assert fm.add_functions(implementations=[GOOD]) == {"forecast_summary": "added"}
    # A row an earlier version stored without checking it: it cannot load.
    fm._insert_function(
        {
            "name": "forecast_summary_broken",
            "argspec": "(city: str)",
            "docstring": "Summarise the forecast for a city, broken.",
            "implementation": "def forecast_summary_broken(city:\n    return city\n",
            "depends_on": [],
            "third_party_imports": [],
            "dependencies": [],
            "precondition": None,
            "stale_reasons": [],
        },
    )
    return fm


@_handle_project
def test_shipped_one_broken_row_fails_the_whole_search():
    assert SETTINGS.UNIFY_SEARCH_SKIP_UNLOADABLE is False
    fm = _library()
    with pytest.raises(SyntaxError):
        fm.search_functions(
            query="summarise the forecast for a city",
            _return_callable=True,
            _namespace={},
            _also_return_metadata=True,
        )


@_handle_project
def test_search_leaves_the_broken_row_out_and_names_it(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_SEARCH_SKIP_UNLOADABLE", True)
    fm = _library()
    namespace: dict = {}
    result = fm.search_functions(
        query="summarise the forecast for a city",
        _return_callable=True,
        _namespace=namespace,
        _also_return_metadata=True,
    )
    names = [row.get("name") for row in result["metadata"]]
    assert "forecast_summary" in names and "forecast_summary_broken" not in names
    warning = result["metadata"][-1]["warning"]
    assert "forecast_summary_broken: SyntaxError" in warning
    assert len(result["callables"]) == len(result["metadata"]) - 1
    assert namespace["forecast_summary"]("Oslo") == "Oslo: sunny"
    assert "forecast_summary_broken" not in namespace


@_handle_project
def test_list_and_filter_skip_the_same_way(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_SEARCH_SKIP_UNLOADABLE", True)
    fm = _library()
    listed = fm.list_functions(
        _return_callable=True,
        _namespace={},
        _also_return_metadata=True,
    )
    assert "forecast_summary" in listed["callables"]
    assert "forecast_summary_broken" not in listed["metadata"]
    assert "forecast_summary_broken" in listed["metadata"]["(unloadable functions)"]

    filtered = fm.filter_functions(
        filter="is_primitive = 0",
        _return_callable=True,
        _namespace={},
        _also_return_metadata=True,
    )
    assert [row.get("name") for row in filtered["metadata"][:-1]] == [
        "forecast_summary",
    ]
    assert "forecast_summary_broken" in filtered["metadata"][-1]["warning"]


@_handle_project
def test_a_search_with_nothing_broken_is_unchanged(monkeypatch):
    fm = FunctionManager()
    fm.add_functions(implementations=[GOOD])
    kwargs = dict(query="forecast", _return_callable=True, _also_return_metadata=True)
    off = fm.search_functions(_namespace={}, **kwargs)["metadata"]
    monkeypatch.setattr(SETTINGS, "UNIFY_SEARCH_SKIP_UNLOADABLE", True)
    on = fm.search_functions(_namespace={}, **kwargs)["metadata"]
    strip = lambda rows: [
        {k: v for k, v in r.items() if not k.startswith("usage")} for r in rows
    ]
    assert strip(on) == strip(off)

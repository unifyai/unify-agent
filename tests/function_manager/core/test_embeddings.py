"""Stored functions are embedded when written and found by meaning.

Symbolic: every test embeds through the deterministic ``fake_embeddings``
provider or runs without a key, so nothing reaches the network.
"""

from __future__ import annotations

import numpy as np

from tests.helpers import _handle_project
from unify.common import embeddings
from unify.common.embeddings import EmbeddingsUnavailable
from unify.common.text_search import text_match
from unify.function_manager.function_manager import (
    SEARCHED_FUNCTION_FIELDS,
    FunctionManager,
    function_search_text,
)

PAY_UTILITY_BILL = '''
def pay_utility_bill(vendor: str, amount: float) -> str:
    """Pay a utility bill to a named vendor."""
    return f"Paid {amount} to {vendor}."
'''

LOOKUP_COMPANY_RECORD = '''
def lookup_company_record(company: str) -> dict:
    """Look up the record held for a company."""
    return {"company": company}
'''

SEND_PLAIN_EMAIL = '''
def send_plain_email(recipient: str, subject: str, body: str) -> str:
    """Send a short plain-text email."""
    return f"Sent {subject!r} to {recipient}."
'''

# Shares no word with pay_utility_bill, and only "company" with the lookup.
PARAPHRASE = "settle what I owe the electricity company"


def _library() -> FunctionManager:
    fm = FunctionManager(include_primitives=False)
    fm.add_functions(
        implementations=[PAY_UTILITY_BILL, LOOKUP_COMPANY_RECORD, SEND_PLAIN_EMAIL],
    )
    return fm


def _search_text(fm: FunctionManager, name: str) -> str:
    return function_search_text(fm._get_function_data_by_name(name=name))


@_handle_project
def test_embedding_and_similarity_search(fake_embeddings):
    fm = FunctionManager()
    fm.add_functions(
        implementations=[
            '''
def pay_utility_bill_via_console(vendor: str, amount: float):
    """Pays a utility bill through the command line interface."""
    return f"Paying {amount} to {vendor} via console."
''',
            '''
def search_internal_documents(query: str):
    """Searches the internal knowledge base for a given query."""
    return f"Searching for {query} in internal documents."
''',
            SEND_PLAIN_EMAIL,
        ],
    )

    # No stored docstring says "gas" ...
    assert not fm.filter_functions(filter="docstring LIKE '%gas%' AND is_primitive = 0")

    # ... while a search by meaning finds the bill payer.
    query = "pay gas bill online"
    [best] = fm.search_functions(query=query, n=1)
    assert best["name"] == "pay_utility_bill_via_console"

    top_two = fm.search_functions(query=query, n=2)
    assert len(top_two) == 2
    assert top_two[0]["name"] == "pay_utility_bill_via_console"


@_handle_project
def test_embedding_populated_on_insert(fake_embeddings):
    fm = _library()

    text = _search_text(fm, "pay_utility_bill")
    assert text in fake_embeddings.embedded
    batches_after_write = len(fake_embeddings.calls)

    fm.search_functions(query=PARAPHRASE, n=3)

    # The search embeds its query and nothing already written.
    assert fake_embeddings.calls[batches_after_write:] == [[PARAPHRASE]]


@_handle_project
def test_embedding_refreshed_on_overwrite(fake_embeddings):
    fm = FunctionManager(include_primitives=False)
    fm.add_functions(
        implementations='''
def overwrite_target_fn() -> str:
    """Bake sourdough bread loaves for the bakery."""
    return "v1"
''',
    )
    original = embeddings.embed(_search_text(fm, "overwrite_target_fn"))

    fm.add_functions(
        implementations='''
def overwrite_target_fn() -> str:
    """Reconcile quarterly expenses against budget forecasts."""
    return "v2"
''',
        overwrite=True,
    )
    updated_text = _search_text(fm, "overwrite_target_fn")

    assert "Reconcile quarterly expenses" in updated_text
    assert updated_text in fake_embeddings.embedded
    assert not np.array_equal(embeddings.embed(updated_text), original)
    [best] = fm.search_functions(query="forecast the budget", n=1)
    assert best["name"] == "overwrite_target_fn"


@_handle_project
def test_search_finds_a_function_by_meaning_alone(fake_embeddings):
    fm = _library()
    target = fm._get_function_data_by_name(name="pay_utility_bill")
    assert text_match(
        target,
        {field: PARAPHRASE for field in SEARCHED_FUNCTION_FIELDS},
    ) == (0, 0)

    hits = fm.search_functions(query=PARAPHRASE, n=3)

    assert [hit["name"] for hit in hits][0] == "pay_utility_bill"
    assert hits[0]["_similarity"] > hits[1]["_similarity"]


@_handle_project
def test_search_falls_back_to_word_match_without_a_key(
    no_embedding_key,
    semantic_search_warnings,
):
    fm = _library()

    hits = fm.search_functions(query=PARAPHRASE, n=3)

    # Word match ranks the function sharing "company" first and cannot
    # place the bill payer above the rest.
    assert hits[0]["name"] == "lookup_company_record"
    assert {hit["name"] for hit in hits} == {
        "pay_utility_bill",
        "lookup_company_record",
        "send_plain_email",
    }
    [warning] = semantic_search_warnings()
    assert "OPENROUTER_API_KEY" in warning


@_handle_project
def test_functions_stored_without_embeddings_are_embedded_by_search(
    fake_embeddings,
    monkeypatch,
):
    def unavailable(model):
        def provider(texts):
            raise EmbeddingsUnavailable("endpoint unreachable")

        return provider

    with monkeypatch.context() as offline:
        offline.setattr(embeddings, "_provider", unavailable)
        fm = _library()
    assert fake_embeddings.calls == []

    hits = fm.search_functions(query=PARAPHRASE, n=3)

    assert hits[0]["name"] == "pay_utility_bill"
    assert _search_text(fm, "pay_utility_bill") in fake_embeddings.embedded

"""Guidance is embedded when written and found by meaning.

Symbolic: every test embeds through the deterministic ``fake_embeddings``
provider or runs without a key, so nothing reaches the network. The builtin
library is part of every search, so the assistant's own entries compete with
it here as they do in use.
"""

from __future__ import annotations

from tests.helpers import _handle_project
from unify.common.text_search import text_match
from unify.guidance_manager.guidance_manager import GuidanceManager

BILLS = {
    "title": "Paying utility bills",
    "content": "Pay each utility bill to its vendor before its due date.",
}
DIRECTORY = {
    "title": "Company directory",
    "content": "Look up a company record in the directory.",
}
# Shares no word with the bills entry, and only "company" with the directory.
PARAPHRASE = "settle what we owe the electricity company every month"


def _library() -> tuple[GuidanceManager, int]:
    gm = GuidanceManager()
    bills_id = gm.add_guidance(**BILLS)["details"]["guidance_id"]
    gm.add_guidance(**DIRECTORY)
    return gm, bills_id


@_handle_project
def test_search_finds_guidance_by_meaning(fake_embeddings):
    gm, bills_id = _library()
    assert text_match(BILLS, {"title": PARAPHRASE, "content": PARAPHRASE}) == (0, 0)

    results = gm.search(references={"content": PARAPHRASE}, k=3)

    assert results[0].guidance_id == bills_id
    assert results[0].is_builtin is False


@_handle_project
def test_add_and_update_embed_the_written_text(fake_embeddings):
    gm, bills_id = _library()
    assert {BILLS["title"], BILLS["content"]} <= set(fake_embeddings.embedded)

    content = "Reconcile the ledger and balance the accounts at month end."
    gm.update_guidance(guidance_id=bills_id, content=content)
    assert fake_embeddings.embedded[-1] == content

    results = gm.search(references={"content": "close the books"}, k=1)
    assert results[0].guidance_id == bills_id


@_handle_project
def test_search_without_references_embeds_nothing(fake_embeddings):
    gm, _ = _library()
    batches_after_write = len(fake_embeddings.calls)

    results = gm.search(k=2)

    assert len(results) == 2
    assert len(fake_embeddings.calls) == batches_after_write


@_handle_project
def test_search_falls_back_to_word_match_without_a_key(
    no_embedding_key,
    semantic_search_warnings,
):
    gm, bills_id = _library()

    results = gm.search(references={"content": PARAPHRASE}, k=3)

    # Word match still answers, but the paraphrase shares no word with the
    # bills entry, so something else ranks first.
    assert len(results) == 3
    assert results[0].guidance_id != bills_id
    [warning] = semantic_search_warnings()
    assert "OPENROUTER_API_KEY" in warning

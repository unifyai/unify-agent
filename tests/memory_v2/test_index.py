# tests/memory_v2/test_index.py
import pytest
from unify.memory_v2.index import IndexOverBudget, build_index
from tests.memory_v2.test_memory_repo import _checkout


def test_index_lists_signatures_and_frames_as_candidates(tmp_path):
    text = build_index(_checkout(tmp_path))
    assert "candidates to check, not authority" in text
    assert (
        "login(apis, username: str, password: str) -> str" in text
        and "_token" not in text
    )
    assert "Pagination" in text and "workflows" not in text


def test_index_suspect_flag_and_budget(tmp_path):
    root = _checkout(tmp_path)
    assert "suspect" in build_index(root, suspect={"venmo"})
    with pytest.raises(IndexOverBudget):
        build_index(root, budget_tokens=10)
    assert "venmo" not in build_index(root, budget_tokens=10_000, channels=[])

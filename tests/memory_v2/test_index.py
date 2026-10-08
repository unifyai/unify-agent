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


def test_index_line_names_the_declared_input_and_is_byte_stable(tmp_path):
    root = _checkout(tmp_path)
    text = build_index(root)
    lines = {
        ln.split("`")[1].split("(")[0]: ln
        for ln in text.splitlines()
        if ln.startswith("- `")
    }
    assert lines["login"].endswith(
        "— Log in once and return the access token. (input: env)",
    ), lines["login"]
    assert "(input:" not in lines["list_friends"]  # no Input: line, no suffix
    assert build_index(root) == text  # the same tree gives the same bytes
    # an Input: line naming no known form is not shown
    mod = root / "env" / "venmo" / "__init__.py"
    mod.write_text(mod.read_text().replace("Input: env", "Input: file"))
    assert "(input:" not in build_index(root)

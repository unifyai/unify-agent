"""No single-world words anywhere in the harness: code, comments, docstrings, prompts and data files under ``unify/``.

The harness is general; one benchmark's vocabulary (its name, its objects, its protocol's words) must not appear in
what agents read or in the code that shapes it. This scans every text file under ``unify/`` with one pattern; the only
exception is listed with its reason. Tests and documentation outside ``unify/`` are not agent-readable and are not
scanned.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

#: One benchmark's vocabulary, whole words, any case.
WORDS = re.compile(
    r"(?<![A-Za-z0-9_])(arc|arcprize|arc-agi|grids?|demos?|demonstrations?|puzzles?|colou?rs?|abstraction and reasoning)"
    r"(?![A-Za-z0-9_])",
    re.IGNORECASE,
)

#: (path relative to the repo, the exact matched text) -> why it is not a single-world word.
ALLOWED = {
    ("unify/actor/execution/file_tools.py", "color"): "ripgrep's `--color never` flag",
}


def _hits(root: Path) -> list[tuple[str, int, str]]:
    out = []
    for path in sorted((root / "unify").rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        rel = path.relative_to(root).as_posix()
        for n, line in enumerate(text.splitlines(), 1):
            for m in WORDS.finditer(line):
                if (rel, m.group(0).lower()) not in ALLOWED:
                    out.append((rel, n, line.strip()[:160]))
    return out


def test_no_single_world_word_under_unify():
    hits = _hits(ROOT)
    assert hits == [], "single-world words in the harness:\n" + "\n".join(f"{p}:{n}: {line}" for p, n, line in hits)


def test_the_scan_finds_a_planted_word(tmp_path):
    (tmp_path / "unify").mkdir()
    (tmp_path / "unify" / "m.py").write_text("x = 1  # a grid of values\n", encoding="utf-8")
    (tmp_path / "unify" / "n.py").write_text('"""Request another demo."""\n', encoding="utf-8")
    assert [(p, n) for p, n, _ in _hits(tmp_path)] == [("unify/m.py", 1), ("unify/n.py", 1)]


def test_the_allowed_flag_is_still_where_it_is_allowed():
    text = (ROOT / "unify/actor/execution/file_tools.py").read_text(encoding="utf-8")
    assert '"--color",' in text


def test_query_llm_help_in_a_cell_names_no_single_world_word():
    from unify.common import cell_models, reasoning

    for doc in (reasoning.query_llm.__doc__ or "", cell_models._cell_query_llm().__doc__ or ""):
        assert doc.strip()
        assert WORDS.search(doc) is None, WORDS.search(doc)

"""No single-world words and no research tags anywhere in the harness: code, comments, docstrings, prompts and data
files under ``unify/``.

The harness is general; one benchmark's vocabulary (its name, its objects, its protocol's words) must not appear in
what agents read or in the code that shapes it. This scans every text file under ``unify/`` with one pattern; the only
exception is listed with its reason. Tests and documentation outside ``unify/`` are not agent-readable and are not
scanned.

The same scan refuses the tags of the research that built the harness (revision labels, planning item numbers, design
section marks, retired role names, who decided what). They mean nothing to a reader of the code, and an agent that reads
them reads noise.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

#: One benchmark's vocabulary, whole words, any case.
WORDS = re.compile(
    r"(?<![A-Za-z0-9_])(arc|arcprize|arc-agi|grids?|demos?|demonstrations?|puzzles?|colou?rs?|abstraction and reasoning)"
    r"(?![A-Za-z0-9_])",
    re.IGNORECASE,
)

#: The research's own tags: revision labels ("rev 2", "rev2.1", "revision 3"), planning documents ("PLAN-REV2"),
#: memory's old version label ("memory v2"), the retired "curator" role, design-section marks ("§2.3"), session and
#: commit tags, and who decided what ("MAIN", "the lead's").
TAGS = re.compile(
    r"(?<![A-Za-z0-9_])(?:rev(?:ision)? ?[0-9]|plan-rev|memory[ _-]v[0-9]|curators?(?![A-Za-z0-9_])|kc3h)"
    r"|§"
    r"|(?-i:(?<![A-Za-z0-9_])(?:MAIN|LEAD)(?![A-Za-z0-9_]))"
    r"|(?<![A-Za-z0-9_])(?:the lead's|\(lead[,)])",
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
            if TAGS.search(line):
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


@pytest.mark.parametrize(
    "line",
    [
        "x = 1  # (rev 2.2) keep the order",
        "# rev2: the old path",
        '"""Revision 3: the writer."""',
        "#: Rev 2 (PLAN-REV2 item 1): the roles",
        "# memory v2's store",
        "# the curator decides",
        "# see spec §2.3",
        "# (MAIN, 13 Oct) approved",
        "# by the lead's design",
        "# kc3hf68se",
    ],
)
def test_the_scan_finds_a_planted_research_tag(tmp_path, line):
    (tmp_path / "unify").mkdir()
    (tmp_path / "unify" / "m.py").write_text(line + "\n", encoding="utf-8")
    assert [(p, n) for p, n, _ in _hits(tmp_path)] == [("unify/m.py", 1)]


@pytest.mark.parametrize(
    "line",
    [
        "def main() -> int:",
        "# the main agent's reply",
        "def _address_hint(lead: str) -> str:",
        "# a previous revision of the note",
        "# the plan's next step",
        "x = 'remain'",
    ],
)
def test_the_tag_scan_passes_plain_words(tmp_path, line):
    (tmp_path / "unify").mkdir()
    (tmp_path / "unify" / "m.py").write_text(line + "\n", encoding="utf-8")
    assert _hits(tmp_path) == []


def test_the_allowed_flag_is_still_where_it_is_allowed():
    text = (ROOT / "unify/actor/execution/file_tools.py").read_text(encoding="utf-8")
    assert '"--color",' in text


def test_query_llm_help_in_a_cell_names_no_single_world_word():
    from unify.common import cell_models, reasoning

    for doc in (reasoning.query_llm.__doc__ or "", cell_models._cell_query_llm().__doc__ or ""):
        assert doc.strip()
        assert WORDS.search(doc) is None, WORDS.search(doc)

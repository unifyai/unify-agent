"""The memory harness's text carries no internal jargon.

Comments, docstrings, string literals, identifiers and file names in the memory harness are read by people who never
saw the project's planning documents, so they name things by role and give plain technical reasons. This test scans
the memory harness tree and fails on internal shorthand: version tags and codenames, plan-step, decision and ruling
tags, section references into internal documents, agent or session names, preregistration and rehearsal ids, odd
label prefixes, and model names used as role names (models are configuration values, roles are writer, actor,
helper and fork).

The scan is standard library only (``tokenize`` and ``ast``) and never imports the harness. Tags that look like
ordinary short names (``r1``, ``a1``, ``P2`` …) are checked only in comments and docstrings, where they can only be
references; the rest are checked everywhere. Model-id string literals (``"provider/model-name"``) are configuration
values and are skipped.
"""

from __future__ import annotations

import ast
import io
import re
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Directories scanned whole, single files scanned, and the extra rule: any file whose name mentions memory.
SCAN_DIRS = ("unify/memory_v2", "tests/memory_v2")
SCAN_FILES = ("unify/settings.py", "unify/setting_aliases.py")
NAME_ROOTS = ("unify", "tests")
SOURCE_SUFFIXES = (".py", ".pysrc")

# Files exempt by design, with the reason.
ALLOWED_FILES = {
    # The map from former setting names to current ones: it must spell the former names.
    "unify/setting_aliases.py",
    # Reads a recorded commit's file by the path it had at that commit.
    "tests/memory_v2/golden/extract_v2_screen.py",
    # This file spells the patterns it bans.
    "tests/memory_v2/test_no_internal_jargon.py",
}

# Scopes a token can be found in.
COMMENT, DOCSTRING, STRING, NAME, PATH = "comment", "docstring", "string", "name", "path"
PROSE = frozenset({COMMENT, DOCSTRING})
TEXT = frozenset({COMMENT, DOCSTRING, STRING})
EVERYWHERE = frozenset({COMMENT, DOCSTRING, STRING, NAME, PATH})

# Word edges that treat ``_`` as a separator, so ``sol_reply`` and ``memory_v21`` are caught and ``consolidate`` is not.
_L = r"(?<![A-Za-z0-9])"
_R = r"(?![A-Za-z0-9])"

RULES: tuple[tuple[str, re.Pattern[str], frozenset[str]], ...] = (
    ("version tag", re.compile(_L + r"[vV](?:2\.1|21)" + _R), EVERYWHERE),
    ("milestone code", re.compile(_L + r"[mM][0-9][a-z]" + _R), EVERYWHERE),
    ("codename", re.compile(_L + r"[kK][cC][0-9]+" + _R), EVERYWHERE),
    ("miner codename", re.compile(_L + r"S0" + _R), EVERYWHERE),
    ("miner codename", re.compile(_L + r"s0" + _R), PROSE),
    ("plan step tag", re.compile(_L + r"P[0-9]{1,2}[a-z]?" + _R), PROSE),
    ("decision tag", re.compile(_L + r"D[0-9]{1,2}" + _R), PROSE),
    ("ruling tag", re.compile(_L + r"R[0-9]{1,2}" + _R), PROSE),
    ("workstream tag", re.compile(_L + r"W[0-9]{1,2}" + _R), PROSE),
    ("design revision tag", re.compile(_L + r"r[0-9]{1,2}" + _R), PROSE),
    ("review item tag", re.compile(_L + r"(?:[BCFIJLMNST][1-9][0-9]?[a-z]?|I-Q[0-9]+)" + _R), PROSE),
    ("document section reference", re.compile(r"§"), EVERYWHERE),
    (
        "document item reference",
        re.compile(
            r"\b(?:[Ss]pec|[Rr]eview|re-review|[Rr]uling|[Rr]ulings|[Pp]lan|[Dd]ecision|[Ss]tep)\s+"
            r"[A-Z]{1,2}[0-9]+[a-z]?" + _R
        ),
        PROSE,
    ),
    ("agent name", re.compile(_L + r"(?:MAIN|RUNTIME|CRITIC|COMMS)" + _R), EVERYWHERE),
    ("agent name", re.compile(_L + r"(?:OPS|a1|b1|0f)" + _R), PROSE),
    ("person title", re.compile(r"\b[Tt]he lead\b"), TEXT),
    ("preregistration id", re.compile(r"\bprereg-[0-9a-f]{4,}"), EVERYWHERE),
    ("results id", re.compile(r"\bresults-[0-9a-f]{6,}"), EVERYWHERE),
    ("rehearsal id", re.compile(_L + r"dr[0-9]+-"), EVERYWHERE),
    ("label prefix", re.compile(_L + r"b2p?:"), EVERYWHERE),
    ("model name as a role", re.compile(_L + r"(?:[sS]ol|SOL|[lL]una|LUNA)" + _R), EVERYWHERE),
)

# Plan-step tags in test names and file names (``test_library_p7_off``, ``p1_t7_inputs.py``); elsewhere ``p1`` is an
# ordinary variable.
TEST_NAME_RULE = ("plan step tag", re.compile(_L + r"[pP][0-9]{1,2}" + _R))

# A string literal that is a whole model id, such as ``"openai/gpt-5"``: a configuration value.
MODEL_ID = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.:@+-]+")


def scanned_files(root: Path = ROOT) -> list[Path]:
    """Every source file in the memory harness tree, sorted, with the allow-listed files left out."""
    found: set[Path] = set()
    for d in SCAN_DIRS:
        found.update(p for p in (root / d).rglob("*") if p.is_file())
    for f in SCAN_FILES:
        if (root / f).is_file():
            found.add(root / f)
    for d in NAME_ROOTS:
        found.update(p for p in (root / d).rglob("*") if p.is_file() and "memory" in p.name.lower())
    return sorted(
        p
        for p in found
        if "__pycache__" not in p.parts and p.relative_to(root).as_posix() not in ALLOWED_FILES
    )


def _docstring_lines(tree: ast.AST) -> set[int]:
    """Start lines of string statements: docstrings and bare string expressions used as notes."""
    return {
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    }


def _string_body(tok: str) -> str:
    """A string token's text without its prefix and quotes (good enough to tell a bare model id)."""
    s = tok.lstrip("rRbBuUfF")
    for q in ('"""', "'''", '"', "'"):
        if s.startswith(q) and s.endswith(q) and len(s) >= 2 * len(q):
            return s[len(q) : -len(q)]
    return s


def _pieces(text: str) -> list[tuple[int, str, str]]:
    """(line, scope, text) pieces of a Python source: comments, docstrings, other strings and identifiers."""
    try:
        doc_lines = _docstring_lines(ast.parse(text))
    except SyntaxError:
        return [(n, COMMENT, line) for n, line in enumerate(text.splitlines(), 1)]
    out: list[tuple[int, str, str]] = []
    fstring_middle = getattr(tokenize, "FSTRING_MIDDLE", None)
    for tok in tokenize.generate_tokens(io.StringIO(text).readline):
        if tok.type == tokenize.COMMENT:
            out.append((tok.start[0], COMMENT, tok.string))
        elif tok.type == tokenize.STRING or (fstring_middle is not None and tok.type == fstring_middle):
            if MODEL_ID.fullmatch(_string_body(tok.string)):
                continue
            scope = DOCSTRING if tok.type == tokenize.STRING and tok.start[0] in doc_lines else STRING
            for k, line in enumerate(tok.string.split("\n")):
                out.append((tok.start[0] + k, scope, line))
        elif tok.type == tokenize.NAME:
            out.append((tok.start[0], NAME, tok.string))
    return out


def violations_in(path: Path, root: Path = ROOT) -> list[str]:
    """``file:line: token (rule)`` for every banned token in one file, its path included (as line 0)."""
    rel = path.relative_to(root).as_posix()
    found: list[str] = []
    for rule, rx, scopes in RULES + (TEST_NAME_RULE + (frozenset({PATH}),),):
        if PATH in scopes:
            for m in rx.finditer(rel):
                found.append(f"{rel}:0: {m.group(0)!r} ({rule}, in the file name)")
    if path.suffix not in SOURCE_SUFFIXES:
        return found
    text = path.read_text(encoding="utf-8")
    for line, scope, piece in _pieces(text):
        rules = RULES
        if scope == NAME and (piece.startswith("test_") or piece.startswith("Test")):
            rules = RULES + (TEST_NAME_RULE + (frozenset({NAME}),),)
        for rule, rx, scopes in rules:
            if scope not in scopes:
                continue
            for m in rx.finditer(piece):
                found.append(f"{rel}:{line}: {m.group(0)!r} ({rule}, in a {scope})")
    return found


def all_violations(root: Path = ROOT) -> list[str]:
    return [v for p in scanned_files(root) for v in violations_in(p, root)]


def test_the_scan_covers_the_memory_harness_tree():
    rels = {p.relative_to(ROOT).as_posix() for p in scanned_files()}
    assert "unify/settings.py" in rels
    assert "unify/memory_v2/writer_pass.py" in rels
    assert any(r.startswith("tests/memory_v2/") for r in rels)
    assert not rels & ALLOWED_FILES


def test_the_rules_catch_tags_and_leave_ordinary_words(tmp_path):
    src = tmp_path / "unify" / "memory_v2" / "sample.py"
    src.parent.mkdir(parents=True)
    src.write_text(
        '"""Consolidate the isolation report (r5 §4, MAIN 11 Oct)."""\n'
        "r1, r2 = 1, 2  # P7: the writer reads both\n"
        "sol_reply = consolidate(r1)\n"
        'MODEL = "openai/gpt-5"\n'
        "def test_library_p7_off():\n"
        "    pass\n",
        encoding="utf-8",
    )
    got = violations_in(src, tmp_path)
    assert "unify/memory_v2/sample.py:1: 'r5' (design revision tag, in a docstring)" in got
    assert any(v.startswith("unify/memory_v2/sample.py:1: '§'") for v in got)
    assert any(v.startswith("unify/memory_v2/sample.py:1: 'MAIN'") for v in got)
    assert any(v.startswith("unify/memory_v2/sample.py:2: 'P7'") for v in got)
    assert any(v.startswith("unify/memory_v2/sample.py:3: 'sol'") for v in got)
    assert any(v.startswith("unify/memory_v2/sample.py:5: 'p7'") for v in got)
    # ordinary names: r1/r2 as variables, "consolidate", "isolation", a model id literal
    assert not any(":2: 'r" in v or ":4:" in v for v in got)
    assert not any("consolidate" in v or "isolation" in v for v in got)


def test_the_memory_harness_has_no_internal_jargon():
    found = all_violations()
    assert not found, f"{len(found)} internal-jargon tokens:\n" + "\n".join(found)


if __name__ == "__main__":  # a standalone run prints the findings, one per line
    for v in all_violations():
        print(v)

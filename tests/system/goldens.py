"""The rendered goldens (guard J2): every static text of :mod:`.render`, per role and mode, as it must read.

A golden is today's render with each known-false span replaced by design-true text (:data:`CORRECTIONS`, from the
harness-truth audit's findings and their fixes; ``⟨reword …⟩`` marks a span the design rules out without fixing its
words). So a golden fails today exactly where the text is false. After a fix, re-record with
``python -m tests.system.record_goldens`` on a keyless tree, review the diff against the claims registry
(``claims.tsv``) and drop the corrections that no longer apply.

Pure Python: nothing here imports the harness.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

GOLDENS = "goldens/goldens.json"


@dataclass(frozen=True)
class Correction:
    id: str  # the audit finding
    keys: str  # a regex over render keys (<role>/<mode>/<what>)
    span: str  # the false text, exactly as rendered
    true: str  # what it must read instead ("" drops it)


def _rw(fid: str, why: str) -> str:
    return f"⟨reword {fid}: {why}⟩"


CORRECTIONS: tuple[Correction, ...] = (
    Correction("A1/HL1", r"^(helper|writer)/on/brief$", "tested functions", "functions"),
    Correction("A21", r"^memory_help/", "every item it judges applicable", "the best fit and at most one fallback"),
    Correction("A21", r"^memory_help/", "every item the librarian judged applicable",
               "the best fit and at most one fallback"),
    Correction("A5", r"^actor/", "may already end with", "may already begin with"),
    Correction("A6", r"^actor/", "browses every function and note", "browses every function, note and recorded procedure"),
    Correction("A14", r"^actor/.*/library_line$", "memory.ask() shows the ones that apply", "memory.ask() names the best fit"),
    Correction("A16", r"^(actor|teammate)/off/refusal/(read|grep)/not_found$",
               "; /inputs/attention/index.json lists your entries' material, and memory.items() the library's items", ""),
    Correction("A16", r"^(actor|teammate)/on/refusal/(read|grep)/not_found$",
               "/inputs/attention/index.json lists your entries' material, and memory.items() the library's items",
               "memory.items() lists the library's items"),
    Correction("A16", r"^librarian/on/refusal/(read|grep)/not_found$",
               "/inputs/attention/index.json lists your entries' material, and memory.items() the library's items",
               "tools.read('/memory') lists the library"),
    Correction("A19", r"^cards/", "ns = {}; exec(memory.get('capture:<id>').text, ns); ns['run'](state, tools)",
               _rw("A19", "a recipe a cell can run as written")),
    Correction("A24", r"^memory_help/", "a captured procedure, or a procedure with a recipe)",
               "a captured procedure of the value or state form, or a procedure with a recipe)"),
    Correction("A23", r"^memory_help/", "or a note's text", "or a note's Rule section"),
    Correction("L1", r"^librarian/on/(tools_list|help/finish)$", "end the pass; the gate judges the manifest and files",
               "hand your answer back to the agent that asked"),
    Correction("L1", r"^(librarian|writer)/on/help/finish$", "a writer or curator runs check first",
               "a writer runs check first"),
    Correction("L1", r"^((actor|teammate)/.*/help/ask_library|librarian/on/help/finish)$", "every item that fits",
               "the best fit and at most one fallback"),
    Correction("L2", r"^(librarian|memory_help)/", "its evidence level", "its verification and use records"),
    Correction("L6", r"^librarian/", "with its own facts", "with its own question"),
    Correction("W1", r"^writer/on/brief$", "The harness has already run each procedure on its recorded request, confined; ",
               _rw("W1", "nothing runs a capture") + " "),
    Correction("W2", r"^writer/on/brief$", "A capture is not in the library: it becomes library items only as you write them.",
               "A capture is in the library, served for reuse, from the moment it exists."),
    Correction("W4", r"^writer/on/brief$", " A failed capture is knowledge too: a note of what broke, or a test marked as a known failure.", ""),
    Correction("W8", r"^writer/on/brief$", "later passes revise or remove it", "later passes revise or retire it"),
    Correction("W9", r"^writer/on/brief$", "reads from it every value the job needs", _rw("W9", "steering")),
    Correction("W18", r"^writer/on/help/fixture$", "The gate runs a procedure's recipe on its cases",
               _rw("W18", "the gate runs no code")),
    Correction("W19", r"^writer/on/help/fixture$",
               "The gate runs the function on the recorded bytes again and refuses a fixture that differs.",
               _rw("W19", "the gate runs no code")),
    Correction("W21", r"^writer/", "A capture whose row says it reproduces its recorded result can become",
               "A capture can become"),
    Correction("W23", r"^writer/", "the gate's reasons with the output of any failing test", "the gate's reasons and labels"),
    Correction("W20", r"^writer/", "When the capture's same_as names a capture already landed, this request becomes a new "
               "case of that procedure instead.", ""),
    Correction("W28", r"^writer/on/brief$", "Where code looks reusable, ", ""),
    Correction("W28", r"^(helper|writer)/on/help/similar$", "you are about to generalise", _rw("W28", "steering")),
    Correction("HL2", r"^helper/on/(help/read|tools_list)$", "/inputs, /memory or /outputs", "/inputs, /task, /memory or /work"),
    Correction("J1", r"^judge/", "Each was written from one piece of finished work",
               "Each was written from one recorded episode's work"),
    Correction("F2", r"^fork/", " and could be reused", ""),
    Correction("A17", r"^(helper|librarian|writer)/", "A call returns text", "Most calls return text"),
)


def golden_of(render: dict[str, str]) -> tuple[dict[str, str], dict[str, int]]:
    """The golden of *render* (each correction applied to every key it names), and how often each correction
    applied (by index into :data:`CORRECTIONS`)."""
    out = dict(render)
    applied: dict[int, int] = {}
    for i, c in enumerate(CORRECTIONS):
        rx = re.compile(c.keys)
        for key, text in out.items():
            if rx.search(key) and c.span in text:
                out[key] = text.replace(c.span, c.true)
                applied[i] = applied.get(i, 0) + 1
    return out, applied


FIRST_REQUESTS = "goldens/first_requests-{mode}.json"

#: What varies between two runs of the same scripted requests (ids, names, clocks, places), and its stand-in.
_VARYING = (
    (r"\d{8}T\d{6}-[0-9a-f]{8}", "<episode>"),
    (r"\b[0-9a-f]{40}\b", "<sha>"),
    (r"\b[0-9a-f]{16}\b", "<id16>"),
    (r"\b(actor|librarian|writer|fork|helper|h\d+)-[0-9a-f]{4,12}\b", r"\1-<n>"),
    (r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(:\d{2}(\.\d+)?)?(Z|[+-]\d{2}:?\d{2})?", "<time>"),
    (r"\b\d{2}:\d{2}:\d{2}\b", "<clock>"),
    (r"127\.0\.0\.1:\d+", "127.0.0.1:<port>"),
    (r"/[\w./-]*/logs/sandbox_worlds/[\w.-]+", "<world>"),
    (r'"duration_ms": \d+', '"duration_ms": <ms>'),
    (r"\b\d+(\.\d+)? ?(s|ms) (left|elapsed|ago)\b", r"<n> \2 \3"),
)


def normalised(text: str) -> str:
    for rx, to in _VARYING:
        text = re.sub(rx, to, text)
    return text


def _roles_of(c: Correction, static_keys: list[str]) -> set[str]:
    """The roles a correction is about: those of the static render keys (``<role>/<mode>/<what>``) its regex names."""
    rx = re.compile(c.keys)
    return {k.split("/", 1)[0] for k in static_keys if rx.search(k)}


def design_true(texts: dict[str, str]) -> dict[str, str]:
    """*texts* (``<role>/<mode>/<what>`` keys) with every correction for that role applied wherever its span occurs:
    the design-true text a dynamic golden holds."""
    import json
    from pathlib import Path

    static_keys = list(json.loads((Path(__file__).parent / GOLDENS).read_text()))
    out = dict(texts)
    for c in CORRECTIONS:
        roles = _roles_of(c, static_keys)
        for key, text in out.items():
            if key.split("/", 1)[0] in roles and c.span in text:
                out[key] = text.replace(c.span, c.true)
    return out

"""Kinship: a judge of behaviour names the earlier captures that do the same kind of work as a new one.

The harness's own relations (:mod:`.capture_groups`: a shared call sequence or run of calls, a shared used item, the
same input shape below the envelope) relate two captures only when their records show it. Two pieces of work of the
same kind often touch the world exactly as two of different kinds do (the same calls on the same envelope), so many
captures find no related capture that way. For a capture with none, the writer service asks one model call: given the
new capture and the earlier ones, each described by its behaviour only (what its code does, the input it reads, the
calls it makes and the outcome its episode had), which earlier ones do the same kind of work? A named capture becomes a
``judged`` relation (the ledger's ``kin`` rows, :meth:`.attention.Ledger.record_kin`).

What the judge is shown is behaviour, never the request a capture came from: the procedure's code and its docstring,
its entry form, the key paths of its input, its ordered calls and its outcome. The captures are labelled ``[1]``,
``[2]``, ... in the question; the labels are mapped back here, so the question carries no stored identifier. Nothing is
run, and no word of a capture is matched by the harness: the judgement is the model's.

The earlier captures shown are bounded (:data:`CANDIDATES`, the most recently stored first), so the question stays
small whatever the size of the library; a capture outside that window can still be related to a later one, which
sees it from the other side, or by the harness's own relations.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

#: The most earlier captures one question shows, the most recently stored first.
CANDIDATES = 40
#: The most characters of the new capture's code the question shows; an earlier capture shows at most :data:`BRIEF_CODE`.
NEW_CODE = 4000
BRIEF_CODE = 600
BRIEF = (
    "You compare recorded procedures by what they do. Each was written from one piece of finished work: its code, "
    "the shape of the input it reads, the calls it makes through its environment, in order, and the outcome that "
    "work had.\n"
    "You are shown one new procedure and earlier ones, numbered [1], [2], and so on. Name every earlier procedure "
    "that does the same kind of work as the new one: work that one general procedure could do for both, with what "
    "differs between them as its inputs or parameters. Judge from what the code does with its input. Two procedures "
    "that make the same calls on differently computed values are not the same kind of work; two that compute the "
    "same thing in different words are.\n"
    "Answer in your reply, without running any code, with one JSON object and nothing after it: "
    '{"kin": [{"number": <the number of an earlier procedure>, "why": "<one sentence on the work they share>"}]}, '
    'or {"kin": []} when none does the same kind of work.'
)


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _saved_docstring(capture_dir: Path) -> str:
    try:
        saved = json.loads(_read(capture_dir / "saved.json") or "{}")
    except ValueError:
        return ""
    doc = saved.get("docstring") if isinstance(saved, dict) else None
    return " ".join(str(doc).split())[:600] if isinstance(doc, str) else ""


def card(profile: Any, capture_dir: Path, *, code_chars: int) -> str:
    """One capture as the judge sees it: its behaviour only (module docstring)."""
    code = _read(capture_dir / "procedure.py")
    if len(code) > code_chars:
        code = code[:code_chars] + "\n# ... (the rest of the code is not shown)"
    lines = [
        "calls in order: " + (" > ".join(getattr(profile, "sequence", ()) or ()) or "none"),
        "input key paths: " + (", ".join(sorted(getattr(profile, "input", ()) or ())[:20]) or "none recorded"),
        f"outcome of its work: {getattr(profile, 'outcome', None) or 'not shown'}",
    ]
    doc = _saved_docstring(capture_dir)
    if doc:
        lines.append(f"what it does, as its author described it: {doc}")
    lines.append("code:\n" + code.rstrip("\n"))
    return "\n".join(lines)


def question(new: tuple[Any, Path], earlier: Iterable[tuple[str, Any, Path]]) -> tuple[list[dict], dict[str, str]]:
    """The judge's messages for the *new* capture ``(profile, dir)`` against *earlier* ``(entry, profile, dir)``
    captures (at most :data:`CANDIDATES`, in the order given), and the label -> entry map."""
    labels: dict[str, str] = {}
    parts = ["The new procedure:\n" + card(new[0], new[1], code_chars=NEW_CODE), "", "The earlier procedures:"]
    for i, (entry, profile, d) in enumerate(list(earlier)[:CANDIDATES], start=1):
        label = f"[{i}]"
        labels[label] = entry
        parts += ["", f"{label}:", card(profile, d, code_chars=BRIEF_CODE)]
    return [{"role": "system", "content": BRIEF}, {"role": "user", "content": "\n".join(parts)}], labels


def answer(message: Mapping[str, Any], labels: Mapping[str, str]) -> tuple[list[str], str]:
    """The entries the judge named (numbers it was shown, in its order, each once) and its reasons, from the last
    JSON object of its reply *message*; ``([], why)`` when it named none or the reply cannot be read."""
    content = message.get("content")
    text = content if isinstance(content, str) else ""
    args = _last_json_object(text)
    if not isinstance(args, dict) or not isinstance(args.get("kin"), list):
        return [], "the answer could not be read"
    named: list[str] = []
    whys: list[str] = []
    for row in args["kin"]:
        if not isinstance(row, Mapping):
            continue
        number = str(row.get("number") if row.get("number") is not None else row.get("label") or "").strip("[] ")
        entry = labels.get(f"[{number}]")
        if entry is not None and entry not in named:
            named.append(entry)
            whys.append(" ".join(str(row.get("why") or "").split())[:300])
    return named, "; ".join(w for w in whys if w) or ("none of them" if not named else "")


def _last_json_object(text: str) -> Any:
    """The last complete JSON object in *text* (a fenced block's or bare), or None."""
    decoder = json.JSONDecoder()
    found = None
    i = text.find("{")
    while i != -1:
        try:
            value, end = decoder.raw_decode(text, i)
        except ValueError:
            i = text.find("{", i + 1)
            continue
        if isinstance(value, dict):
            found = value
        i = text.find("{", end)
    return found

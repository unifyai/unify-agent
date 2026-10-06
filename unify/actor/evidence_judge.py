"""``UNIFY_EVIDENCE_LIST_MATCHER=judge``: a model decides whether a stored entry does this request's job.

The evidence list's possibly related tier compares the whole request with
each entry's statement and lists what clears one cosine floor. The memory-a
offline replay (research artifact memory-a-v1, 6 Oct) found no floor serves
every stream: at the 0.45 default it listed an entry on 45% of Continual-ARC
starts where nothing fitted and on 13% of reworded AppWorld repeats; at 0.35
the reworded repeats came back (56%) and the empty lists went (86% listed).
The matching bake-off (research artifact matching-bakeoff-v1) separated
the two jobs a floor tries to do at once:

* **Shortlist by embedding, no floor.** The request and each entry's card
  (name, signature, description and the request it was stored for) are
  embedded; the 5 closest cards are candidates. On real-work data the right
  entry is among them for 90-100% of requests.
* **Decide with one small model call that may answer "none".** The model is
  shown the request and the candidates' cards and picks the one that does
  the same job -- one parametric procedure would serve both -- or none. On
  held-out real-work data balanced accuracy rose from 70 (floor) to 88; on
  a fresh, preregistered set it was 0.98 [0.94, 1.0] against 0.63 for the
  floor, which listed 80% of near-misses where the model listed 3%. About
  USD 0.0001 per request.

The prompt is the bake-off's ``JUDGE_B`` verbatim, including its boundary: a
changed condition or direction (most vs least, sent vs received) is a
changed input, since a parametric function takes it as an argument; a
different action, object or output, or added or dropped steps, is another
job. Nothing here names a benchmark, and the model is never asked to check
an answer against examples.

The pick is listed as seen before, saying a model judged it; an exact
repeat of a recorded request is still listed without asking (that tier was
right on all 66 of its firings in the bake-off). With the judge on, no
cosine floor lists anything: the model's "none" is the silence. A failed
call lists nothing beyond the exact repeats.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)

logger = logging.getLogger(__name__)

K = 5
#: A card's description, as the bake-off cut it.
DOC_CHARS = 300
#: The original request is shown only when it is at most this long.
ORIGIN_CHARS = 1500
#: The embedded card: description and request, cut as the bake-off cut them.
EMBED_DOC_CHARS = 600
EMBED_ORIGIN_CHARS = 4000
MAX_COMPLETION_TOKENS = 1500
#: Entries recorded under a request sharing a rare whole identifier with
#: this one join the candidates ahead of the closest cards, and each card
#: names the identifiers: identifier evidence goes to the judge,
#: never around it. In the bake-off an identifier key that skipped the judge
#: listed 8 of 30 near-misses sharing a file name; on Continual-ARC the task
#: id was the only thing that found the entry.
POOL_KEYED = True
NAME_SHARED = True

# ``UNIFY_EVIDENCE_LIST_MATCHER=judge2``: two pools (the MEMORY judge replay's
# v5 policy, artifacts memory-a-v1/judge-replay/scripts/jr_v5.py). The entries
# sharing an evidence identifier with the request are judged alone, so a
# generic note cannot out-compete them, and their pick is listed from
# ``C_K``; the closest other cards are judged apart and their pick is listed
# only from ``C_R``. On seen ARC runs this kept the same-puzzle hit (96%) and
# cut listings where nothing fits from 71% to 9%.
C_K = 50.0
C_R = 90.0
#: An identifier is an environment token when it is in more than this share of
#: the earlier logged requests, once at least ENV_MIN_EARLIER are logged.
ENV_SHARE = 0.5
ENV_MIN_EARLIER = 4
_SPAN = re.compile(
    r"https?://\S+|[^\s'\"`<>()\[\]{},;]*[/\\][^\s'\"`<>()\[\]{},;]*",
)
_LETTERS = re.compile(r"[A-Za-z]+")
#: Letter runs that make a token a number with a unit, a date or a time.
UNITS = frozenset(
    {
        # time
        *"ms s sec secs second seconds min mins minute minutes h hr hrs hour hours".split(),
        *"d day days wk wks week weeks mo mos month months y yr yrs year years".split(),
        *"am pm night nights q t w h1 fy".split(),
        # months
        *"jan feb mar apr may jun jul aug sep sept oct nov dec january february".split(),
        *"march april june july august september october november december".split(),
        # ordinals
        *"st nd rd th".split(),
        # size, distance, mass, volume, data, screen, rate, money multipliers
        *"mm cm m km mi mile miles ft in inch inches mg g kg lb lbs oz ml l".split(),
        *"kb mb gb tb kbps mbps gbps px pt em dpi x hz khz mhz ghz k bn".split(),
        *"percent pct person people page pages seat seats item items".split(),
    },
)


def unit_token(word: str) -> bool:
    """True for a number with a unit, a date or a time (``30-minute``, ``3rd``, ``Q3``): no evidence of a job."""
    runs = [r.lower() for r in _LETTERS.findall(word)]
    return bool(runs) and bool(re.search(r"\d", word)) and all(r in UNITS for r in runs)


def environment_tokens(request: str, earlier: Sequence[str]) -> Set[str]:
    """Identifiers (lower case) of *request* that describe its environment, not its job.

    Those inside a file path or URL in the request (a username in a home
    directory), and those in more than ``ENV_SHARE`` of the earlier logged
    requests once ``ENV_MIN_EARLIER`` are logged (a standing line every
    request carries).
    """
    from unify.function_manager import task_origin

    out: Set[str] = set()
    for span in _SPAN.findall(request):
        out.update(task_origin.identifiers(span).keys())
    earlier = list(earlier)
    if len(earlier) >= ENV_MIN_EARLIER:
        sets = [set(task_origin.identifiers(text)) for text in earlier]
        for low in task_origin.identifiers(request):
            if sum(1 for found in sets if low in found) > ENV_SHARE * len(sets):
                out.add(low)
    return out


# Verbatim from the matching bake-off (scripts/mb_prompts.py, JUDGE_B).
PROMPT = """A new request has arrived for an AI assistant. Below it are up to 5 entries (stored functions or notes)
saved from earlier work, each with what it is for.
Pick the entry that does the SAME JOB as the new request, meaning one reusable procedure would serve both if its
inputs were parameters. Inputs that may differ: names, people or groups, numbers, amounts, dates and periods, files,
and the setting of a condition or direction (for example most vs least, liked vs not liked, sent vs received, earlier
vs later, before vs after). Answer "none" if no entry does the same job: when the action differs (for example create
vs cancel, like vs rate, count vs list), the kind of object or output differs, steps would be added or dropped, or an
entry is only about the same topic or app.
New request:
----------
{text}
----------
Entries:
{cards}
Return JSON {{"choice": "<entry id or none>", "confidence": <0-100>}}."""

Key = Tuple[str, str]
#: ``generate(prompt) -> reply text``; raises on failure.
Generate = Callable[[str], Awaitable[Any]]


def _clean(text: Any) -> str:
    return " ".join(str(text or "").split())


def _name(kind: str, row: Dict[str, Any]) -> str:
    if kind == "guidance":
        return _clean(row.get("title")) or f"guidance {row.get('guidance_id')}"
    return str(row.get("name") or "")


def _doc(kind: str, row: Dict[str, Any]) -> str:
    return str(
        (row.get("content") if kind == "guidance" else row.get("docstring")) or "",
    )


def _signature(kind: str, row: Dict[str, Any]) -> str:
    if kind == "guidance":
        return ""
    argspec = str(row.get("argspec") or "").strip()
    return argspec if argspec.startswith("(") else f"({argspec})"


def origin_text(row: Dict[str, Any]) -> str:
    """The first request the entry was recorded under, or ``""``."""
    from unify.function_manager import task_origin

    texts = [t for t in task_origin._origins(row)[1] if t and t.strip()]
    return texts[0] if texts else ""


def card_text(kind: str, row: Dict[str, Any]) -> str:
    """The text embedded for an entry: name, signature, description and the request it was stored for."""
    text = f"{_name(kind, row)}\n{_signature(kind, row)}\n{_doc(kind, row)[:EMBED_DOC_CHARS]}"
    origin = origin_text(row)
    if origin:
        text += f"\nWritten for: {origin[:EMBED_ORIGIN_CHARS]}"
    return text.strip()


def judge_card(
    label: str,
    kind: str,
    row: Dict[str, Any],
    shared: Sequence[str] = (),
) -> str:
    """One entry as the judge reads it (the bake-off's card), plus any identifiers it shares with the request."""
    from unify.actor import related_shortlist

    statement, written = related_shortlist.statement_of(kind, row)
    doc = _doc(kind, row)[:DOC_CHARS].strip() or "(no description)"
    card = (
        f"{label}: {_name(kind, row)}\n   what it does: {doc}\n"
        f"   written for (job): {statement if written else '(unknown)'}"
    )
    origin = origin_text(row)
    if origin and len(origin) <= ORIGIN_CHARS:
        card += f"\n   original request: {origin}"
    if shared:
        card += "\n   its request also named: " + ", ".join(shared)
    return card


def prompt(request: str, cards: Sequence[str]) -> str:
    return PROMPT.format(text=request, cards="\n".join(cards))


_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def parse_choice(raw: Any, n: int) -> Tuple[Optional[int], Optional[float]]:
    """``(index, confidence)`` of the picked card (0-based), ``(None, ...)`` for "none" or an unreadable reply."""
    match = _JSON_OBJECT.search(str(raw or ""))
    if match is None:
        return None, None
    try:
        data = json.loads(match.group(0))
    except ValueError:
        return None, None
    if not isinstance(data, dict):
        return None, None
    try:
        confidence = float(data.get("confidence"))
    except (TypeError, ValueError):
        confidence = None
    choice = str(data.get("choice", "none")).strip()
    if choice[:1].upper() == "E" and choice[1:].isdigit():
        i = int(choice[1:])
        if 1 <= i <= n:
            return i - 1, confidence
    return None, confidence


def pool(
    keyed: Sequence[Key],
    scores: Dict[Key, float],
    *,
    k: int = K,
    exclude: Sequence[Key] = (),
    include_keyed: bool = True,
) -> List[Key]:
    """At most *k* candidates: entries the keys matched (best first, when *include_keyed*), then the closest cards."""
    skip = set(exclude)
    out: List[Key] = []
    if include_keyed:
        for key in keyed:
            if key not in skip and key not in out:
                out.append(key)
    for key, _ in sorted(scores.items(), key=lambda item: -item[1]):
        if len(out) >= k:
            break
        if key not in skip and key not in out:
            out.append(key)
    return out[:k]


@dataclass
class Verdict:
    """What the judge decided for one request (for the listing, analysis and tests)."""

    candidates: List[Key]
    choice: Optional[Key]
    confidence: Optional[float]
    failed: bool = False


async def decide(
    request: str,
    entries: Dict[Key, Dict[str, Any]],
    candidates: Sequence[Key],
    *,
    generate: Generate,
    shared: Optional[Dict[Key, List[str]]] = None,
) -> Verdict:
    """Ask the judge which of *candidates* does *request*'s job; a failed call picks nothing."""
    candidates = [key for key in candidates if key in entries]
    if not candidates:
        return Verdict([], None, None)
    shared = shared or {}
    cards = [
        judge_card(f"E{i}", key[0], entries[key], shared.get(key, ()))
        for i, key in enumerate(candidates, 1)
    ]
    try:
        raw = await generate(prompt(request, cards))
    except Exception as exc:  # an aid; the task starts without it
        logger.warning(
            f"evidence list judge failed ({type(exc).__name__}: {exc}); no pick",
        )
        return Verdict(list(candidates), None, None, failed=True)
    index, confidence = parse_choice(raw, len(candidates))
    return Verdict(
        list(candidates),
        candidates[index] if index is not None else None,
        confidence,
    )


def client_generate(model: Optional[str]) -> Generate:
    """``generate`` through a fresh client of *model* (the session's default when ``None``), at low effort."""
    from unify.common.llm_client import new_llm_client

    async def generate(text: str) -> Any:
        client = new_llm_client(
            model,
            origin="EvidenceList.judge",
            reasoning_effort="low",
            max_completion_tokens=MAX_COMPLETION_TOKENS,
        )
        return await client.generate(user_message=text)

    return generate


__all__ = [
    "K",
    "PROMPT",
    "Verdict",
    "card_text",
    "client_generate",
    "decide",
    "judge_card",
    "origin_text",
    "parse_choice",
    "pool",
    "prompt",
]

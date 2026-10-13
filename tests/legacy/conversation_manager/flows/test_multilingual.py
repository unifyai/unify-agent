"""
tests/conversation_manager/flows/test_multilingual.py
=========================================================

Tests for multilingual conversation handling.

Verifies that the ConversationManager:

1. Carries the user's language both ways: the model is shown the user's
   message as written, and its reply in that language reaches the chat
   unchanged (Spanish, Japanese, Arabic script)
2. Keeps the queries delegated to ``act`` in English: the ``act`` tool tells
   the model to write them in English, and the English query reaches the
   action unchanged

The slow brain's model is scripted (``tests/scripted_model.py``) with the
replies and queries a model makes in each language; which language a live
model chooses is not tested here.
"""

import json
import re

import pytest

from tests.helpers import _handle_project
from tests.legacy.conversation_manager.cm_helpers import (
    filter_events_by_type,
    get_exactly_one,
)
from tests.legacy.conversation_manager.scripted_brain import (
    WAIT,
    act,
    brain_requests,
    brain_turn,
    conversation,
    say,
)
from tests.scripted_model import ScriptedModel, scripted
from unify.legacy.conversation_manager.events import (
    UnifyMessageReceived,
    UnifyMessageSent,
    ActorHandleStarted,
)

# ---------------------------------------------------------------------------
#  Language detection helpers
# ---------------------------------------------------------------------------

# Spanish markers — distinctive words unlikely to appear in English text.
_SPANISH_MARKERS = [
    # Greetings / closings
    "hola",
    "gracias",
    "buenos",
    "por favor",
    "disculpe",
    "encantado",
    "entendido",
    "perfecto",
    # Common conversational words
    "cómo",
    "también",
    "mucho",
    "claro",
    "ahora",
    "aquí",
    "muy",
    "algo",
    "todo",
    "todos",
    "después",
    # Verbs / verb forms
    "necesito",
    "puedo",
    "puede",
    "tengo",
    "quiero",
    "estoy",
    "creo",
    "hacer",
    "hablar",
    "revisar",
    "enviar",
    "enviamos",
    "vamos",
    "déjame",
    "dime",
    # Nouns / domain words
    "número",
    "reunión",
    "información",
    "mañana",
    "semana",
    "momento",
    "novedades",
    "propuesta",
    "proyecto",
    "resultado",
    "prioridades",
    "actualización",
    # Phrases
    "cómo estás",
    "vamos a ver",
    "buen",
    "bien",
    "contigo",
    "gusto",
    "confirmar",
]

# French markers — distinctive words and contractions unlikely to appear
# in English text.  Includes both formal and conversational vocabulary so
# that natural LLM responses ("De rien, c'était avec plaisir") are caught.
_FRENCH_MARKERS = [
    # Greetings / closings
    "bonjour",
    "bonsoir",
    "salut",
    "merci",
    "enchanté",
    "bienvenue",
    "bonne journée",
    "cordialement",
    # Common conversational words
    "très",
    "aussi",
    "alors",
    "donc",
    "voilà",
    "toujours",
    "vraiment",
    "maintenant",
    "peut-être",
    "quelque",
    "beaucoup",
    "seulement",
    # Verbs / verb forms
    "aider",
    "faire",
    "confirmer",
    "vérifier",
    # Contractions (matched after apostrophe normalisation)
    "j'ai",
    "c'est",
    "c'était",
    "d'accord",
    "l'appel",
    "n'est",
    "qu'on",
    "s'il vous plaît",
    "s'il te plaît",
    "aujourd'hui",
    # Distinctive short phrases
    "je suis",
    "je vais",
    "je vous",
    "il y a",
    "en prie",
    "mise à jour",
    "bien sûr",
    "de rien",
    "avec plaisir",
    "que cela",
    # Nouns / domain words
    "réunion",
    "monsieur",
    "madame",
    "journée",
    "besoin",
    "côté",
    "cela",
    # Accented / conjugated forms (high signal — accents rare in English)
    "terminé",
    "résumé",
    "prêt",
    "priorité",
    "également",
    "absolument",
    "certainement",
    "exactement",
    "actuellement",
    "malheureusement",
    "rapidement",
    "aidé",
    "ravie",
    "ravi",
    # Filler / discourse markers
    "heureux",
    "heureuse",
    "bonne",
    "plaisir",
    "ça",
]

# French diacritics — used with ≥1 marker hit as an escape hatch for short
# courtesy replies that under-hit the two-marker threshold (e.g. "…aidé").
# Diacritics alone are not enough (English loanwords like "café").
_FRENCH_DIACRITIC_RE = re.compile(
    r"[àâäæçéèêëïîôœùûüÿÀÂÄÆÇÉÈÊËÏÎÔŒÙÛÜŸ]",
)

_CJK_RE = re.compile(r"[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]")
_ARABIC_RE = re.compile(r"[\u0600-\u06FF]")
_NON_LATIN_RE = re.compile(
    r"[\u0600-\u06FF\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\u0400-\u04FF\uAC00-\uD7AF]",
)


def _normalize_apostrophes(text: str) -> str:
    """Collapse typographic quotes to ASCII apostrophe for reliable matching."""
    return text.replace("\u2019", "'").replace("\u2018", "'")


def _has_spanish(text: str) -> bool:
    """True if *text* contains clear Spanish language indicators."""
    low = _normalize_apostrophes(text.lower())
    hits = sum(1 for w in _SPANISH_MARKERS if w in low)
    return hits >= 2 or "¿" in text or "¡" in text


def _has_french(text: str) -> bool:
    """True if *text* contains clear French language indicators."""
    low = _normalize_apostrophes(text.lower())
    hits = sum(1 for w in _FRENCH_MARKERS if w in low)
    # Diacritics alone can appear in English loanwords (café); require a
    # marker hit alongside them — parallel to Spanish ¿/¡ for short replies.
    return hits >= 2 or (hits >= 1 and bool(_FRENCH_DIACRITIC_RE.search(text)))


def _has_japanese(text: str) -> bool:
    """True if *text* contains hiragana, katakana, or kanji."""
    return bool(_CJK_RE.search(text))


def _has_arabic(text: str) -> bool:
    """True if *text* contains Arabic script characters."""
    return bool(_ARABIC_RE.search(text))


_QUOTED_RE = re.compile(r'"[^"]*"|\'[^\']*\'|「[^」]*」|“[^”]*”')


def _instruction_text(query: str) -> str:
    """The act query minus its payload.

    An act query is written in English, but it carries the user's own words
    where they are the deliverable: a quoted message, a filename, a term the
    user chose. Those are quoted spans and non-Latin tokens; what is left is
    the instruction the actor reads.
    """
    without_quotes = _QUOTED_RE.sub(" ", query)
    tokens = [t for t in without_quotes.split() if not _NON_LATIN_RE.search(t)]
    return " ".join(tokens)


def _is_english(text: str) -> bool:
    """True if *text* appears to be written in English.

    Checks that there are no significant non-English markers: non-Latin
    script characters, or multiple distinctive Spanish / French words.
    """
    if _NON_LATIN_RE.search(text):
        return False
    if _has_spanish(text):
        return False
    if _has_french(text):
        return False
    return True


def _act_tool(call) -> str:
    """The ``act`` tool a brain request offers, as JSON."""
    (tool,) = [
        t
        for t in call.request["tools"]
        if (t.get("function") or {}).get("name") == "act"
    ]
    return json.dumps(tool, ensure_ascii=False)


async def _reply_round_trip(cm, message: str, answer: str) -> str:
    """Step *message* with the brain scripted to reply *answer*; check the
    model was shown *message* as written and return the reply that reached
    the chat."""
    model = ScriptedModel(actor=[brain_turn(say(answer), WAIT)])
    with scripted(model):
        result = await cm.step_until_wait(UnifyMessageReceived(content=message))

    (request,) = brain_requests(model)
    assert conversation(request) == [message], "the model sees the words as written"
    model.assert_used_up()
    return get_exactly_one(result.output_events, UnifyMessageSent).content


async def _act_round_trip(cm, message: str, query: str, answer: str):
    """Step *message* with the brain scripted to act on *query*; check the
    ``act`` tool asked for English and return the started actions."""
    model = ScriptedModel(actor=[brain_turn(act(query), say(answer), WAIT)])
    with scripted(model):
        result = await cm.step_until_wait(UnifyMessageReceived(content=message))

    (request,) = brain_requests(model)
    assert conversation(request) == [message]
    assert "written in English" in _act_tool(request)
    model.assert_used_up()
    return filter_events_by_type(result.output_events, ActorHandleStarted)


# =====================================================================
#  Group 1 — Response matches the sender's language
#
#  Messages are deliberately simple greetings and thank-yous that
#  require nothing more than a polite reply — no calendar lookups, no
#  information retrieval, no action delegation.
# =====================================================================


@pytest.mark.asyncio
@_handle_project
async def test_spanish_message_reply_in_spanish(initialized_cm):
    """The user writes in Spanish -> the Spanish reply reaches the user."""
    answer = "¡De nada! Me alegra mucho que te haya sido útil. ¡Buen día también!"
    sent = await _reply_round_trip(
        initialized_cm,
        (
            "¡Hola! Muchas gracias por tu ayuda ayer, fue muy útil. "
            "Espero que tengas un excelente día."
        ),
        answer,
    )
    assert sent == answer
    assert _has_spanish(
        sent,
    ), f"Expected Spanish reply to a Spanish-speaking user, got: {sent}"


@pytest.mark.asyncio
@_handle_project
async def test_japanese_unify_message_reply_in_japanese(initialized_cm):
    """The user writes in Japanese -> the Japanese reply reaches the user."""
    answer = "どういたしまして！お役に立ててうれしいです。良い一日をお過ごしください。"
    sent = await _reply_round_trip(
        initialized_cm,
        (
            "こんにちは！先日はお手伝いいただきありがとうございました。"
            "おかげさまでとても助かりました。良い一日をお過ごしください。"
        ),
        answer,
    )
    assert sent == answer
    assert _has_japanese(
        sent,
    ), f"Expected Japanese reply to a Japanese-speaking user, got: {sent}"


@pytest.mark.asyncio
@_handle_project
async def test_arabic_message_reply_in_arabic(initialized_cm):
    """The user writes in Arabic -> the Arabic reply reaches the user."""
    answer = "على الرحب والسعة! يسعدني أن ذلك كان مفيداً. أتمنى لك يوماً سعيداً أيضاً."
    sent = await _reply_round_trip(
        initialized_cm,
        (
            "مرحبا! شكراً جزيلاً على مساعدتك بالأمس. "
            "كان ذلك مفيداً جداً. أتمنى لك يوماً سعيداً."
        ),
        answer,
    )
    assert sent == answer
    assert _has_arabic(
        sent,
    ), f"Expected Arabic reply to an Arabic-speaking user, got: {sent}"


# =====================================================================
#  Group 2 — Act queries stay in English
#
#  The user asks for a file to be written into the workspace, which is
#  what forces the ``act`` delegation.
# =====================================================================


@pytest.mark.asyncio
@_handle_project
async def test_act_query_english_when_user_speaks_spanish(initialized_cm):
    """
    The user gives instructions in Spanish -> the act query must still be English.

    Writing a file into the workspace needs the actor, so the request is
    dispatched as an act whose query is an internal interface and must stay
    English even though the user-facing conversation is Spanish.
    """
    cm = initialized_cm
    query = (
        "Create a text file with a checklist for preparing a meeting "
        "and save it in the workspace."
    )

    # The user gives an instruction in Spanish -> needs the actor (writes a file)
    actor_events = await _act_round_trip(
        cm,
        (
            "Crea un archivo de texto con una lista de verificación para "
            "preparar una reunión y guárdalo en mi espacio de trabajo"
        ),
        query,
        "¡Claro! Estoy preparando la lista de verificación en tu espacio de trabajo.",
    )

    assert actor_events, (
        "Expected act to be called (ActorHandleStarted), "
        f"got tools={cm.all_tool_calls}"
    )

    for event in actor_events:
        assert event.query == query, "the English query reaches the action"
        instruction = _instruction_text(event.query)
        assert (
            len(instruction.split()) >= 5
        ), f"Act query has no instruction: {event.query}"
        assert _is_english(
            instruction,
        ), f"Internal act query should be in English, got: {event.query}"
    assert cm.cm.in_flight_actions[0]["query"] == query


@pytest.mark.asyncio
@_handle_project
async def test_act_query_english_when_user_speaks_japanese(initialized_cm):
    """
    The user gives instructions in Japanese -> act query must still be English.

    Writing a file into the workspace needs the actor. Even though the entire
    conversation is in Japanese, the act query must be in English with no CJK
    character leakage.
    """
    cm = initialized_cm
    query = (
        "Write a meeting preparation checklist to a text file "
        "and save it in the workspace."
    )

    # The user gives an instruction in Japanese -> needs the actor (writes a file)
    actor_events = await _act_round_trip(
        cm,
        "会議の準備チェックリストをテキストファイルにまとめて、ワークスペースに保存してください",
        query,
        "承知しました。チェックリストを作成してワークスペースに保存します。",
    )

    assert len(actor_events) >= 1, "Expected act to be called (ActorHandleStarted)"

    query_sent = actor_events[0].query
    assert query_sent == query, "the English query reaches the action"
    instruction = _instruction_text(query_sent)
    assert len(instruction.split()) >= 5, f"Act query has no instruction: {query_sent}"
    assert not _has_japanese(
        instruction,
    ), f"Act query instruction must be English, got: {query_sent}"
    assert _is_english(
        instruction,
    ), f"Act query should be in English, got: {query_sent}"
    assert cm.cm.in_flight_actions[0]["query"] == query


# =====================================================================
#  Group 3 — Language sticks across turns
# =====================================================================


@pytest.mark.asyncio
@_handle_project
async def test_spanish_multi_turn_stays_spanish(initialized_cm):
    """Two-turn Spanish conversation: both replies reach the user in Spanish,
    and the second turn is shown the first exchange as written."""
    cm = initialized_cm
    first = "¡Hola! Muchas gracias por todo tu trabajo. Me ha sido de gran ayuda."
    first_answer = "¡Muchas gracias a ti! Me alegra mucho poder ayudarte."
    second = (
        "¡Qué amable! También quería decirte que todo salió "
        "muy bien con el proyecto. ¡Buen trabajo!"
    )
    second_answer = "¡Qué buena noticia! Me alegra que el proyecto saliera tan bien."
    model = ScriptedModel(
        actor=[
            brain_turn(say(first_answer), WAIT),
            brain_turn(say(second_answer), WAIT),
        ],
    )
    with scripted(model):
        # Turn 1: simple greeting and thanks
        result1 = await cm.step_until_wait(UnifyMessageReceived(content=first))
        msg1 = get_exactly_one(result1.output_events, UnifyMessageSent)

        # Turn 2: follow-up thanks and well-wishing
        result2 = await cm.step_until_wait(UnifyMessageReceived(content=second))
        msg2 = get_exactly_one(result2.output_events, UnifyMessageSent)

    assert msg1.content == first_answer
    assert _has_spanish(
        msg1.content,
    ), f"Turn-1 reply should be in Spanish, got: {msg1.content}"
    assert msg2.content == second_answer
    assert _has_spanish(
        msg2.content,
    ), f"Turn-2 reply should be in Spanish, got: {msg2.content}"

    _, later = brain_requests(model)
    assert conversation(later) == [first, first_answer, second]
    model.assert_used_up()

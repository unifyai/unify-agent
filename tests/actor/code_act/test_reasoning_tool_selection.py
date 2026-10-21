"""A cell's ``query_llm`` calls inside a deterministic Python loop.

The actor's model is scripted (``tests/scripted_model.py``) to do what the
request asks: loop over the messages in Python and judge each one's intent
with ``query_llm``. ``query_llm`` itself is faked in the harness
(``reasoning.query_llm``). The test checks the path between them: each call
from the cell reaches the harness's ``query_llm`` with its prompt and
response format, its structured result comes back into the loop, the loop's
output reaches the model's next turn, and the answer is the result. Whether
a live model chooses ``query_llm`` over substring rules is the model's
behaviour.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from tests.actor.code_act.helpers import (
    extract_code_act_execute_code_snippets,
    get_code_act_tool_calls,
)
from tests.scripted_model import ScriptedModel, cell, reply, scripted
from unify.actor.code_act_actor import CodeActActor
from unify.common import reasoning


def _coerce_query_llm_result(response_format: Any, result: dict[str, Any]) -> Any:
    if isinstance(response_format, type) and issubclass(response_format, BaseModel):
        fields = response_format.model_fields
        return response_format(**{k: v for k, v in result.items() if k in fields})
    if isinstance(response_format, dict):
        return result
    return (
        f"category={result['category']}; "
        f"needs_reply={result['needs_reply']}; "
        f"confidence={result['confidence']}"
    )


@pytest.mark.asyncio
@pytest.mark.timeout(240)
async def test_code_act_sprinkles_query_llm_into_semantic_python_loop(
    monkeypatch: pytest.MonkeyPatch,
):
    query_llm_calls: list[str] = []

    async def fake_query_llm(
        prompt: str,
        *,
        response_format: Any = None,
        **_kwargs: Any,
    ) -> Any:
        query_llm_calls.append(prompt)
        text = prompt.lower()
        needs_reply = (
            "thursday fits" in text
            or "renewal paperwork" in text
            or "signature" in text
        )
        category = "scheduling" if "move" in text else "follow_up"
        if "newsletter" in text:
            category = "newsletter"
        result = {
            "category": category,
            "needs_reply": needs_reply,
            "confidence": 0.96,
            "rationale": "Semantic intent requires or does not require a reply.",
        }
        return _coerce_query_llm_result(response_format, result)

    monkeypatch.setattr(reasoning, "query_llm", fake_query_llm)

    messages = [
        {
            "id": "m1",
            "subject": "Calendar question",
            "body": "Thursday fits my week better than Tuesday, if that still works on your side.",
        },
        {
            "id": "m2",
            "subject": "Product newsletter",
            "body": "Here are this month's feature highlights and release notes.",
        },
        {
            "id": "m3",
            "subject": "Follow-up needed",
            "body": "The renewal paperwork is waiting on your signature before the account can continue.",
        },
    ]

    request = (
        "Classify these messages and return the ids that need a human reply. "
        "Write Python that loops over the messages and keeps exact iteration, "
        "list construction, and output formatting deterministic. The hard part "
        "is interpreting message intent, so do not use substring rules as the "
        f"whole classifier. Messages: {messages}. "
        "Do not ask clarifying questions. Return only the final list of ids."
    )

    # The loop the model writes: exact iteration and list building in
    # Python, the intent judgement from query_llm with a JSON schema.
    loop = (
        f"messages = {messages!r}\n"
        "schema = {'type': 'json_schema', 'json_schema': {'name': 'triage', "
        "'schema': {'type': 'object', 'properties': {"
        "'category': {'type': 'string'}, 'needs_reply': {'type': 'boolean'}, "
        "'confidence': {'type': 'number'}}, "
        "'required': ['category', 'needs_reply', 'confidence']}}}\n"
        "needs_reply = []\n"
        "for m in messages:\n"
        "    verdict = await query_llm(\n"
        "        'Does this message need a human reply? '\n"
        "        f\"Subject: {m['subject']}. Body: {m['body']}\",\n"
        "        response_format=schema,\n"
        "    )\n"
        "    if verdict['needs_reply']:\n"
        "        needs_reply.append(m['id'])\n"
        "print(needs_reply)\n"
    )
    model = ScriptedModel(
        actor=[reply(calls=[cell(loop)]), reply("['m1', 'm3']")],
    )

    actor = CodeActActor(environments=[], timeout=200)
    with scripted(model):
        handle = await actor.act(request, clarification_enabled=False)
        result = await handle.result()

    assert "execute_code" in set(get_code_act_tool_calls(handle))
    snippets = extract_code_act_execute_code_snippets(handle)
    assert snippets == [loop]

    # One query_llm call per message, in order, each with that message.
    assert len(query_llm_calls) == 3, query_llm_calls
    for prompt, message in zip(query_llm_calls, messages):
        assert message["body"] in prompt

    # The loop's output reached the model's next turn.
    _first, second = model.of("actor")
    assert any(
        m.get("role") == "tool" and "['m1', 'm3']" in str(m.get("content"))
        for m in second.messages
    ), second.messages

    result_text = str(result)
    assert "m1" in result_text
    assert "m3" in result_text
    assert "m2" not in result_text
    model.assert_used_up()

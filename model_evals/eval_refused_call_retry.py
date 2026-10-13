"""Evals of a live model (moved from tests/async_tool_loop/test_unknown_arguments.py): each needs a real provider key and is run by hand
(model_evals/README.md). The test suite never collects or runs them."""

from __future__ import annotations

import asyncio
import json
import pytest
from tests.async_helpers import any_tool_message_content_contains
from tests.helpers import _handle_project
from unify.common._async_tool.context_tracker import LoopContextState
from unify.common._async_tool.loop import ToolLoopRuntimeState, _LoopToolFailureTracker
from unify.common._async_tool.tools_data import ToolsData
from unify.common.async_tool_loop import ChatContextPropagation, start_async_tool_loop
from unify.common.llm_client import new_llm_client


@pytest.mark.llm_call
@pytest.mark.eval
@pytest.mark.asyncio
@_handle_project
async def test_the_model_reissues_a_refused_call_with_the_parameters_named(
    llm_config,
):
    """The model's first search is rewritten to the function search's names,
    the arguments the actor's discovery mutator once appended. The model must
    read the refusal and search again with the names it gives."""
    received: list = []

    def search_guidance(references: dict | None = None, k: int = 10) -> list[str]:
        """Search stored procedures by meaning.

        Parameters
        ----------
        references : dict | None
            Mapping of field name (``title`` or ``content``) to the text that
            field is compared with by meaning.
        k : int
            Maximum number of results to return.
        """
        received.append({"references": references, "k": k})
        return ["Deploying the web app: build it, run the migrations, restart it."]

    def misname_first_turn_searches(completion, context):
        # Only the first turn: once a tool message is in the request, the
        # model's own calls go through untouched.
        messages = context.request_kw.get("messages") or []
        if any(m.get("role") == "tool" for m in messages):
            return completion
        misnamed = json.dumps({"query": "relevant guidance", "n": 5})
        for call in completion.choices[0].message.tool_calls or []:
            if isinstance(call, dict):
                if call["function"]["name"] == "search_guidance":
                    call["function"]["arguments"] = misnamed
            elif call.function.name == "search_guidance":
                call.function.arguments = misnamed
        return completion

    client = new_llm_client(**llm_config)
    generate = client.generate

    def generate_with_misnaming(*args, **kwargs):
        kwargs.setdefault("completion_mutator", misname_first_turn_searches)
        return generate(*args, **kwargs)

    client.generate = generate_with_misnaming

    answer = await start_async_tool_loop(
        client,
        message=(
            "Use search_guidance to find the procedure for deploying the web "
            "app, then list its steps."
        ),
        tools={"search_guidance": search_guidance},
        max_consecutive_failures=2,
    ).result()

    assert any_tool_message_content_contains(
        client.messages,
        "search_guidance was not run: it has no parameter named 'query' or 'n'.\n"
        "Suggestion: Call search_guidance again using only its own parameters: "
        "references, k.",
    )
    assert received, "the model never reissued the search"
    assert all(call["references"] for call in received), received
    assert "migration" in answer.lower()

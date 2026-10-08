"""Token-budget ratchet over the fixed per-LLM-call prompt payload.

Every LLM call pays its system prompt and serialized tool schemas before a
single trajectory token. These budgets pin the currently rendered sizes so
that growth is a deliberate decision — raise the constant in the same change
that grows the payload, and say why — rather than an accident, and so that
every landed cut is banked by tightening the constant to the new size.
"""

from __future__ import annotations

from unify.actor.core_surface import PromptSurface
import json

import pytest

from unify.common.token_utils import count_tokens
from unify.manager_registry import ManagerRegistry

pytestmark = pytest.mark.no_unify_context

# Budgets sit just above the measured rendered size at the last tightening.
ACTOR_SYSTEM_PROMPT_BUDGET = 4_000
ACTOR_ACT_TOOL_SCHEMAS_BUDGET = 14_500
STORAGE_REVIEW_DOCTRINE_BUDGET = 4_700


@pytest.fixture(autouse=True)
def _clean_registry():
    ManagerRegistry.clear()
    yield
    ManagerRegistry.clear()


def _simulated_actor():
    from unify.actor.code_act_actor import CodeActActor
    from unify.function_manager.simulated import SimulatedFunctionManager
    from unify.guidance_manager.simulated import SimulatedGuidanceManager

    return CodeActActor(
        function_manager=SimulatedFunctionManager(description="token budget"),
        guidance_manager=SimulatedGuidanceManager(description="token budget"),
    )


def _actor_system_prompt() -> str:
    from unify.actor.environments import ActorEnvironment
    from unify.actor.prompt_builders import build_code_act_prompt

    return build_code_act_prompt(
        environments={"primitives": ActorEnvironment()},
        can_store=True,
        core=PromptSurface(),
    )


def _actor_act_tool_schemas() -> str:
    from unify.common.llm_helpers import method_to_schema

    return json.dumps(
        [
            method_to_schema(getattr(tool, "fn", tool), name)
            for name, tool in _simulated_actor().get_tools("act").items()
        ],
    )


def _storage_review_doctrine() -> str:
    # The static prefix every skill-librarian loop pays, in prompt order: the
    # minimal rulebook (baked in at the code freeze) and the instructions.
    from unify.actor.code_act_actor import (
        _storage_base_instructions,
        _storage_doctrine_sections,
    )

    return _storage_doctrine_sections() + _storage_base_instructions()


_CASES = {
    "actor_system_prompt": (_actor_system_prompt, ACTOR_SYSTEM_PROMPT_BUDGET),
    "actor_act_tool_schemas": (
        _actor_act_tool_schemas,
        ACTOR_ACT_TOOL_SCHEMAS_BUDGET,
    ),
    "storage_review_doctrine": (
        _storage_review_doctrine,
        STORAGE_REVIEW_DOCTRINE_BUDGET,
    ),
}


@pytest.mark.parametrize("case", sorted(_CASES))
def test_fixed_prompt_payload_stays_within_budget(case):
    render, budget = _CASES[case]
    tokens = count_tokens(render())
    assert tokens < budget, (
        f"{case} renders {tokens:,} tokens against a budget of {budget:,}. "
        "If the growth is deliberate, raise the budget constant in this file "
        "in the same change and say why; otherwise trim the payload."
    )

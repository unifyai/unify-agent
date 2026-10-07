"""Token-budget ratchet over the legacy conversation manager's system prompt
(moved from tests/test_prompt_token_budgets.py, whose docstring explains the
ratchet).
"""

from __future__ import annotations

import pytest

from unify.common.token_utils import count_tokens

pytestmark = pytest.mark.no_unify_context

# Sits just above the measured rendered size at the last tightening.
CM_SYSTEM_PROMPT_BUDGET = 11_100


def test_cm_system_prompt_stays_within_budget():
    from unify.legacy.conversation_manager.prompt_builders import build_system_prompt

    tokens = count_tokens(
        build_system_prompt(
            bio="A helpful assistant.",
            first_name="Alice",
            surname="Smith",
        ).flatten(),
    )
    assert tokens < CM_SYSTEM_PROMPT_BUDGET, (
        f"cm_system_prompt renders {tokens:,} tokens against a budget of "
        f"{CM_SYSTEM_PROMPT_BUDGET:,}. If the growth is deliberate, raise the "
        "budget constant in this file in the same change and say why; "
        "otherwise trim the payload."
    )

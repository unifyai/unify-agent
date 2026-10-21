from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

from unify.common.prompt_helpers import PromptParts
from unify.legacy.conversation_manager.prompt_builders import build_system_prompt
from unify.session_details import (
    PLACEHOLDER_USER_FIRST_NAME,
    PLACEHOLDER_USER_SURNAME,
    SESSION_DETAILS,
)

if TYPE_CHECKING:
    from unify.legacy.conversation_manager.conversation_manager import (
        ConversationManager,
    )
    from unify.legacy.conversation_manager.domains.renderer import SnapshotState


@dataclass(frozen=True)
class BrainSpec:
    """
    Fully materialized inputs needed for a single Main CM Brain run.

    This is intentionally a small, explicit data structure so we can refactor
    how the brain is executed (plain generate vs async tool loop) without
    tangling prompt/model construction with execution and side effects.
    """

    system_prompt: PromptParts
    state_prompt: str

    def state_message(self) -> dict:
        # Mark this as a state snapshot so the async tool loop can treat it as
        # transient state (e.g., keep only the latest snapshot when generating).
        return {
            "role": "user",
            "content": self.state_prompt,
            "_cm_state_snapshot": True,
        }


def build_brain_spec(
    cm: "ConversationManager",
    snapshot_state: "SnapshotState",
) -> BrainSpec:
    """
    Build the prompt inputs for a single Main CM Brain run.

    The returned spec is *pure* (no side effects). The user's and the
    assistant's identities come from ``SESSION_DETAILS``.

    Parameters
    ----------
    cm : ConversationManager
        The conversation manager instance.
    snapshot_state : SnapshotState
        Pre-rendered conversation state (caller computes this once and reuses it
        for both the BrainSpec and incremental-diff tracking).
    """
    prompt = snapshot_state.full_render

    assistant = SESSION_DETAILS.assistant
    user = SESSION_DETAILS.user
    bio_parts: list[str] = []
    job_title = assistant.job_title.strip()
    if job_title:
        bio_parts.append(f"Role / specialization: {job_title}.")
    if assistant.about:
        bio_parts.append(assistant.about)

    system_prompt = build_system_prompt(
        bio="\n".join(bio_parts),
        first_name=user.first_name or PLACEHOLDER_USER_FIRST_NAME,
        surname=user.surname or PLACEHOLDER_USER_SURNAME,
    )

    # Validate we can JSON-encode state prompt early (helps catch accidental objects)
    json.dumps({"state_prompt": prompt})

    return BrainSpec(system_prompt=system_prompt, state_prompt=prompt)

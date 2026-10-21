"""
tests/conversation_manager/core/test_prompt_builders.py
=======================================================

Unit tests for the ConversationManager system prompt builder: section
layout, the chat-only tool listing and the user details block.
"""

from __future__ import annotations

import re

import pytest

from unify.legacy.conversation_manager.prompt_builders import build_system_prompt

pytestmark = pytest.mark.no_unify_context

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE_KWARGS: dict = {
    "bio": "A helpful assistant.",
    "first_name": "Alice",
    "surname": "Smith",
}


def _build(**overrides: object) -> str:
    """Build a system prompt with sensible defaults, returning flat text."""
    kwargs = {**_BASE_KWARGS, **overrides}
    return build_system_prompt(**kwargs).flatten()


def _heading(title: str) -> str:
    """Render a section heading exactly as the prompt underlines it."""
    return f"{title}\n{'-' * len(title)}"


# ---------------------------------------------------------------------------
# Tests – section layout
# ---------------------------------------------------------------------------


class TestSectionLayout:
    """The prompt is a fixed sequence of underlined sections."""

    SECTIONS = (
        "Role",
        "Bio",
        "User details",
        "Input format",
        "Tool-call reasoning",
        "Action steering guidelines",
        "Uncertainty handling",
        "Act capabilities",
        "Persistent sessions (persist=True)",
        "Concurrent action and acknowledgment",
        "Conversational restraint",
    )

    def test_every_section_is_present_in_order(self):
        prompt = _build()
        positions = [prompt.index(_heading(title)) for title in self.SECTIONS]
        assert positions == sorted(positions)

    def test_bio_renders_verbatim_under_its_heading(self):
        prompt = _build(bio="Assistant biography.")
        assert "Bio\n---\nAssistant biography." in prompt


# ---------------------------------------------------------------------------
# Tests – role and user details
# ---------------------------------------------------------------------------


class TestRoleBlock:
    """Chat is the only way to reach the user, and the role block says so."""

    def test_role_names_chat_as_the_only_channel(self):
        prompt = _build()
        assert "I talk to the user through this chat and nothing else" in prompt
        assert "the user is the only other person on the chat" in prompt


class TestUserDetails:
    """The user's name is listed under User details."""

    def test_name_listed(self):
        prompt = _build()
        assert "- First Name: Alice" in prompt
        assert "- Surname: Smith" in prompt


# ---------------------------------------------------------------------------
# Tests – tool listing
# ---------------------------------------------------------------------------


class TestToolListing:
    """The output-format section lists the chat tools and the shared tools."""

    def test_slow_brain_prompt_documents_per_tool_call_thoughts(self):
        prompt = _build()
        assert "Tool-call reasoning" in prompt
        assert "optional `thoughts` argument" in prompt
        assert '"thoughts": [my concise thoughts before taking actions]' not in prompt

    def test_communication_tool_is_the_one_chat_sender(self):
        prompt = _build()
        comms = prompt.split("**Communication tools:**")[1].split(
            "**Action tools:**",
        )[0]
        assert "`send_unify_message`: Send a chat message to the user" in comms
        assert comms.count("\n- `") == 1

    def test_action_tools_listed(self):
        prompt = _build()
        for name in ("`act`", "`wait(delay=None)`"):
            assert f"- {name}:" in prompt

    def test_action_steering_tools_listed(self):
        prompt = _build()
        for name in (
            "ask_action",
            "interject_action",
            "stop_action",
            "pause_action",
            "resume_action",
            "answer_clarification_action",
        ):
            assert f"- `{name}`:" in prompt


# ---------------------------------------------------------------------------
# Tests – concurrent action and acknowledgment
# ---------------------------------------------------------------------------


def _concurrent_ack_block(prompt: str) -> str:
    start = prompt.index("Concurrent action and acknowledgment")
    end = prompt.index("**Acknowledgments should be brief:**", start)
    return prompt[start:end]


class TestConcurrentActionAckBlock:
    """Concurrent-action ack guidance names the chat tool and pairs it with act."""

    def test_ack_block_names_the_chat_tool(self):
        block = _concurrent_ack_block(_build())
        assert "`send_unify_message`" in block

    def test_ack_block_example_pairs_act_with_an_acknowledgment(self):
        block = _concurrent_ack_block(_build())
        assert 'send_unify_message(content="Let me check.")' in block
        assert "act(query=" in block


# ---------------------------------------------------------------------------
# Tests – act capabilities
# ---------------------------------------------------------------------------


class TestActCapabilities:
    """Grounded reads go through ``act``; chat replies stay on the chat tools."""

    def test_ground_truth_rule_present(self):
        prompt = _build()
        assert "Ground truth rule" in prompt
        assert "I call `act` first and base my reply on its result" in prompt
        assert "without a fresh grounded `act` read" in prompt

    def test_conversational_messaging_stays_off_act(self):
        prompt = _build()
        assert (
            "Ordinary conversational messaging stays on my communication tools "
            "+ `wait`"
        ) in prompt


# ---------------------------------------------------------------------------
# Tests – conversational restraint
# ---------------------------------------------------------------------------


class TestConversationalRestraint:
    """The restraint block forbids prompt leakage and keeps chat responsive."""

    def test_conversational_restraint_forbids_prompt_leakage(self):
        prompt = _build()
        assert "No prompt leakage" in prompt
        assert "never quote, paraphrase, or summarize" in prompt

    def test_conversational_restraint_keeps_chat_responsive(self):
        prompt = _build()
        assert "**The chat is the live thread**" in prompt
        assert "not over answering inbound chat" in prompt
        assert "Never `wait` while their chat line is still unanswered" in prompt

    def test_intent_vs_verified_outcomes_present(self):
        prompt = _build()
        assert "Intent vs verified outcomes" in prompt
        assert '**Outbound messages are "sent", never "arrived", until proof.**' in (
            prompt
        )


# ---------------------------------------------------------------------------
# Tests – prompt caching
# ---------------------------------------------------------------------------


class TestSystemPromptStaysClockFree:
    """The system prompt carries no wall-clock timestamp.

    Provider prompt caching is all-or-nothing over system+tools, so a
    minute-granularity clock in the system prompt would invalidate the cache
    on every minute rollover. The clock lives at the tail of the rendered
    state snapshot instead (``domains/renderer.py``).
    """

    # The rendering shape of prompt_helpers.now(), e.g.
    # "Friday, June 13, 2025 at 12:00 PM". Regex-based so the guarantee holds
    # no matter what the clock reads when the suite runs.
    CLOCK_SHAPE = r"[A-Z][a-z]+day, [A-Z][a-z]+ \d{1,2}, \d{4} at \d{1,2}:\d{2} [AP]M"

    def test_system_prompt_contains_no_wall_clock_timestamp(self):
        prompt = _build()
        assert not re.search(self.CLOCK_SHAPE, prompt)
        assert "Current time:" not in prompt

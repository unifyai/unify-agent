"""Symbolic: under the default settings, the actor's first request is the recorded one.

The code freeze made the chosen configuration the default (lean-all, Python
tool mode, the shared agent record). This pins what a model receives from it
on a fresh actor's first call, byte for byte: the system prompt, the tool
list (``execute_code`` alone), the tool choice and the first user message.
Nothing is pinned: every setting is at its default. Requests are captured at
unillm's transport (``tests/cache_discipline_helpers.py``), so nothing leaves
the process. Since step 4 of the freeze this golden replaces the
switches-off equivalence test: the baked values are the only code path.

To record it again after a deliberate change, run this file with
``UNIFY_RECORD_GOLDEN=1``: the recording is written to
``logs/actor_baked_prompt_golden.json`` (the checkout is read-only inside the
test sandbox), to be copied over ``tests/actor_baked_prompt_golden.json``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tests import cache_discipline_helpers as h
from tests.baked_defaults import BAKED_DEFAULTS

BAKED_GOLDEN = Path(__file__).resolve().parents[2] / "actor_baked_prompt_golden.json"
RECORDED = Path(__file__).resolve().parents[3] / "logs" / BAKED_GOLDEN.name


def baked_recording(requests: list[dict]) -> dict:
    """The first request of the actor's session, its parts as compared.

    ``$UNIFY_HOME`` stands for the test's own home, which the system prompt
    names in the workspace path.
    """
    home = os.environ.get("UNIFY_HOME") or ""

    def _norm(text: str) -> str:
        return text.replace(home, "$UNIFY_HOME") if home else text

    first = h.session_requests(requests)[0]
    messages = first["messages"]
    return {
        "system_prompt": _norm(messages[0]["content"]),
        "tools": _norm(json.dumps(first["tools"], indent=1, default=str)),
        "tool_choice": first["tool_choice"],
        "first_user_message": _norm(
            next(m["content"] for m in messages if m["role"] == "user"),
        ),
    }


async def record_first_request() -> dict:
    """``CodeActActor.act`` on a fresh actor under the defaults; its first request."""
    _result, _, requests = await h.scenario_actor(
        [lambda: h.completion(content="done")] * 8,
    )
    return baked_recording(requests)


def test_the_defaults_are_the_baked_values():
    from unify.settings import ProductionSettings

    # A switch the strip deleted has no field: its baked value is the code.
    fields = ProductionSettings.model_fields
    assert {
        name: fields[name].default
        for name in BAKED_DEFAULTS
        if name in fields and fields[name].default != BAKED_DEFAULTS[name]
    } == {}


@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_the_default_first_request_is_the_recorded_one():
    recorded = await record_first_request()
    if os.environ.get("UNIFY_RECORD_GOLDEN") == "1":
        RECORDED.parent.mkdir(parents=True, exist_ok=True)
        RECORDED.write_text(json.dumps(recorded, indent=1) + "\n")
        pytest.skip(f"recorded to {RECORDED}")
    golden = json.loads(BAKED_GOLDEN.read_text())
    assert [t["function"]["name"] for t in json.loads(recorded["tools"])] == [
        "execute_code",
    ]
    assert recorded["tools"] == golden["tools"]
    assert recorded["system_prompt"] == golden["system_prompt"]
    assert recorded["first_user_message"] == golden["first_user_message"]
    assert recorded == golden

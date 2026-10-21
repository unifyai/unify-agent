"""The memory agents' models have a context window. unillm registers one for the writer's model and the helpers'
model on OpenRouter (``WRITER_MODEL``, ``HELPER_MODEL``), so the compaction trigger
(:func:`.writer_pass.compaction_due`, which reads the window) sees it for the writer, the curator and each
helper. The window reaches them on the model turn: :func:`.writer_pass.unillm_turn` sets ``turn.max_input_tokens`` from
``unillm.get_max_input_tokens``. The values and their source are recorded in docs/design/model-windows-20261012.md.
No test makes a model call.
"""

from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace

import pytest
import unillm

from unify.memory import writer_pass
# the helper's lane refuses while unillm records OTel spans, which the test runner turns on for every session
from tests.memory.test_writer_route import _otel_off_unless_a_test_turns_it_on  # noqa: F401 - an autouse fixture
from unify.memory.integration import consolidate, switch
from unify.memory.integration.cost import HELPER_ORIGIN
from tests.memory.integration.test_consolidate import (
    FakeWriter,
)
from tests.memory.test_writer_compaction import (
    Compacting,
    _events as _pass_events,
    _reads,
    _run,
    _summaries,
)

WRITER_MODEL = "openai/gpt-6-sol"
HELPER_MODEL = "openai/gpt-6-luna"
#: OpenRouter's ``context_length`` for each model, as unillm registers it
WINDOWS = {WRITER_MODEL: 1_050_000, HELPER_MODEL: 1_050_000}


def _window(model: str) -> int:
    return unillm.get_max_input_tokens(f"{model}@openrouter")


@pytest.mark.parametrize("model", [WRITER_MODEL, HELPER_MODEL])
def test_unillm_knows_the_window(model):
    assert _window(model) == WINDOWS[model]


@pytest.mark.parametrize(
    ("model", "origin"),
    [(WRITER_MODEL, "memory_writer"), (HELPER_MODEL, HELPER_ORIGIN)],
)
def test_the_real_turn_carries_the_window(model, origin):
    """The shipped turn (no route; the real client is built, no call is made) holds the model's window."""
    turn = writer_pass.unillm_turn(model, "low", origin=origin)
    assert turn.max_input_tokens == _window(model) == WINDOWS[model]


class RealWindowWriter(FakeWriter):
    """A fake writer whose turns carry the window unillm registers for the model consolidate asks for."""

    def __call__(self, model, effort, **kw):
        turn = super().__call__(model, effort, **kw)
        turn.max_input_tokens = _window(model)
        return turn


def test_the_helper_turn_consolidate_builds_carries_the_registered_window(
    monkeypatch,
):
    """``_helper_turns`` builds each helper's turn with the real :func:`.writer_pass.unillm_turn` on the helper lane;
    the turn holds the helper model's window. The lane is a placeholder and the client a stand-in: no call is made.
    """
    import unify.common.llm_client as llm_client
    from unillm.clients import uni_llm

    monkeypatch.setattr(uni_llm, "_llm_gateway", uni_llm._llm_gateway)
    monkeypatch.setattr(
        llm_client,
        "new_llm_client",
        lambda model, **kw: SimpleNamespace(messages=[]),
    )
    for name, value in (
        ("_HELPER_TOKEN", "helper-lane-placeholder-0123456789"),  # pragma: allowlist secret
        ("_HELPER_BASE", "http://127.0.0.1:18081/helper/v1"),
        ("_HELPER_TRIED", True),
        ("_HELPER_REFUSAL", None),
    ):
        monkeypatch.setattr(switch, name, value)
    s = SimpleNamespace(
        UNIFY_MEMORY_LIBRARY_HELPER_BASE_URL="",
        UNIFY_MEMORY_LIBRARY_HELPER_TOKEN_FD="",
        UNIFY_MEMORY_LIBRARY_HELPER_MODEL=HELPER_MODEL,
        UNIFY_MODEL="vendor/actor-model@openrouter",
    )
    got = consolidate._helper_turns(s, "low", True)
    assert got["helper_refusal"] is None
    turn = got["helper_turn"]("s1")
    assert turn.max_input_tokens == _window(HELPER_MODEL) == WINDOWS[HELPER_MODEL]


class Stepped(Compacting):
    """A turn with the writer model's registered window whose calls report *prompts* prompt tokens in turn, then
    *after* for every later call (the summary call included)."""

    def __init__(self, turns, prompts, after):
        super().__init__(turns, window=_window(WRITER_MODEL))
        self.prompts, self.after = list(prompts), after

    def _usage(self):
        prompt = self.prompts.pop(0) if self.prompts else self.after
        return {"prompt_tokens": prompt, "prompt_tokens_details": {"cached_tokens": 0}}


LOW = 2_000  # a prompt well under any trigger


def _threshold(window: int) -> int:
    """The fewest prompt tokens for which :func:`.writer_pass.compaction_due` fires on *window*, found by bisection
    of the trigger itself, so the tests hold whatever share or fixed size the rule uses."""
    lo, hi = 1, window
    assert writer_pass.compaction_due(hi, window)
    assert not writer_pass.compaction_due(LOW, window)
    while lo < hi:
        mid = (lo + hi) // 2
        if writer_pass.compaction_due(mid, window):
            hi = mid
        else:
            lo = mid + 1
    assert writer_pass.compaction_due(lo, window)
    assert not writer_pass.compaction_due(lo - 1, window)
    return lo


def test_a_prompt_crossing_the_threshold_compacts_once_and_the_summary_is_booked(
    tmp_path,
):
    window = _window(WRITER_MODEL)
    crossing = _threshold(window)
    model = Stepped(_reads(3), prompts=[LOW, crossing], after=LOW)
    out, _ = _run(tmp_path, model, compact=True)
    assert out.context_window == window
    assert out.compactions == 1 and len(_summaries(model)) == 1
    # the summary is the third request: right after the call that reached the trigger
    assert model.requests.index(_summaries(model)[0]) == 2
    rows = [r for r in _pass_events(tmp_path) if r.get("event") == "compaction"]
    assert len(rows) == 1 and rows[0]["usage"]["usd"] == "0.002"
    normal = len(model.requests) - 1
    assert out.calls == len(model.requests)
    assert Decimal(out.usd) == Decimal("0.001") * normal + Decimal("0.002")


def test_prompts_below_the_threshold_never_compact(tmp_path):
    window = _window(WRITER_MODEL)
    model = Stepped(_reads(3), prompts=[], after=_threshold(window) - 1)
    out, _ = _run(tmp_path, model, compact=True)
    assert out.context_window == window
    assert out.compactions == 0 and not _summaries(model)

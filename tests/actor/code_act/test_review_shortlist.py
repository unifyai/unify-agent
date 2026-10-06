"""Symbolic: ``UNIFY_REVIEW_SHORTLIST``, the storage review is shown the library entries closest to its session.

The stored functions the session called come first, then the functions and
guidance entries whose cards are closest to its request, each in full. The
review keeps its tools. The embedder and the review model are stand-ins:
nothing leaves the process.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from tests.actor.code_act import test_outcome_channel as toc
from tests.actor.code_act.test_outcome_channel import switches  # noqa: F401 (fixture)
from unify.actor import review_shortlist as rs
from unify.settings import ProductionSettings, SETTINGS


def _fn(fid, name, doc, source, origin=""):
    from unify.function_manager import task_origin

    metadata = {task_origin.REQUESTS_FIELD: [origin]} if origin else {}
    return {
        "function_id": fid,
        "name": name,
        "argspec": "(x)",
        "docstring": doc,
        "implementation": source,
        "metadata": metadata,
    }


FUNCTIONS = [
    _fn(
        1,
        "total_payments",
        "Total card payments.",
        "def total_payments(x):\n    return sum(x)\n",
        "Total my card payments.",
    ),
    _fn(
        2,
        "refund_payment",
        "Refund one payment.",
        "def refund_payment(x):\n    return x\n",
        "Refund a payment.",
    ),
    _fn(
        3,
        "plan_trip",
        "Plan a trip.",
        "def plan_trip(x):\n    return x\n",
        "Plan my trip to Rome.",
    ),
]
NOTES = [
    {
        "guidance_id": 7,
        "title": "Card payments",
        "content": "Amounts are negative.",
        "function_ids": [1],
        "metadata": {},
    },
]


class _Manager:
    def __init__(self, rows):
        self._rows_ = rows

    def _evidence_rows(self):
        return list(self._rows_)


def _embed(texts):
    out = []
    for t in texts:
        low = t.lower()
        v = np.array(
            [0.1, "payment" in low, "total" in low, "trip" in low],
            dtype=np.float32,
        )
        out.append(v / np.linalg.norm(v))
    return np.stack(out)


@pytest.fixture
def on(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_REVIEW_SHORTLIST", True)


def _trajectory(request, code=""):
    traj = [{"role": "user", "content": request}]
    if code:
        traj.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "execute_code", "arguments": code}},
                ],
            },
        )
    return traj


def test_the_switch_is_off_by_default():
    assert ProductionSettings.model_fields["UNIFY_REVIEW_SHORTLIST"].default is False


def test_off_no_section_and_no_embedding(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_REVIEW_SHORTLIST", False)

    def boom(texts):
        raise AssertionError("embedded while off")

    assert (
        rs.section(
            _Manager(FUNCTIONS),
            _Manager(NOTES),
            _trajectory("Total my payments."),
            embed=boom,
        )
        == ""
    )


def test_used_functions_come_first_then_the_closest_cards_in_full(on):
    text = rs.section(
        _Manager(FUNCTIONS),
        _Manager(NOTES),
        _trajectory("Total my card payments for May.", code="print(plan_trip(3))"),
        embed=_embed,
    )
    assert text.startswith(rs.HEADER)
    first = text.index("### function `plan_trip(x)`")
    assert first < text.index("### function `total_payments(x)`")
    assert "def total_payments(x):\n    return sum(x)" in text
    assert "Stored for: Total my card payments." in text
    assert "### guidance #7: Card payments\nAmounts are negative." in text


def test_an_embedding_failure_keeps_the_used_entries(on):
    def broken(texts):
        raise RuntimeError("provider down")

    text = rs.section(
        _Manager(FUNCTIONS),
        _Manager(NOTES),
        _trajectory("Anything.", code="refund_payment(2)"),
        embed=broken,
    )
    assert "### function `refund_payment(x)`" in text
    assert "total_payments" not in text


def test_an_empty_library_adds_nothing(on):
    assert (
        rs.section(_Manager([]), _Manager([]), _trajectory("Hi."), embed=_embed) == ""
    )


@pytest.mark.asyncio
async def test_the_section_reaches_the_reviews_prompt(monkeypatch, switches):
    monkeypatch.setattr(SETTINGS, "UNIFY_REVIEW_SHORTLIST", True)
    marker = rs.HEADER + "### function `stand_in(x)`\nshown in full\n\n"
    with patch.object(rs, "section", lambda *a, **k: marker):
        _note, requests, _handle, _ = await toc._persistent_review()
    carrying = [
        r for r in requests if "### function `stand_in(x)`" in toc._review_text(r)
    ]
    assert len(carrying) >= 1

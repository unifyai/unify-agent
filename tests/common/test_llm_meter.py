"""Symbolic: the run meter sums money exactly and keeps unknown costs unknown."""

from __future__ import annotations

import json
import threading
from decimal import Decimal
from types import SimpleNamespace

import pytest

from unify.common import llm_meter
from unify.common.llm_meter import RunMeter, decimal_string, to_decimal_usd


def _add(meter: RunMeter, cost, purpose: str = "planning") -> None:
    meter.add(purpose, prompt_tokens=10, completion_tokens=2, cost=cost)


@pytest.mark.timeout(30)
def test_reported_costs_sum_without_float_error():
    meter = RunMeter()
    for _ in range(10):
        _add(meter, 0.1)
    # Accumulated as floats with +=, as the meter did, ten 0.1s make
    # 0.9999999999999999 (Python 3.12's sum() compensates; += does not).
    as_float = 0.0
    for _ in range(10):
        as_float += 0.1
    assert as_float != 1.0
    assert meter.cost_usd("planning") == Decimal("1.0")
    assert meter.snapshot()["cost"] == {"planning": "1.0"}

    meter = RunMeter()
    _add(meter, 0.1)
    _add(meter, 0.2)
    assert meter.cost_usd("planning") == Decimal("0.3")


@pytest.mark.timeout(30)
def test_small_and_mixed_inputs_keep_every_digit():
    meter = RunMeter()
    _add(meter, 1e-7)
    _add(meter, "0.000412")
    _add(meter, Decimal("0.0000003"))
    _add(meter, 3)
    assert meter.cost_usd("planning") == Decimal("3.0004124")
    # Never an exponent, so the string is a plain decimal anywhere it goes.
    assert meter.snapshot()["cost"]["planning"] == "3.0004124"
    assert decimal_string(Decimal("1E-7")) == "0.0000001"


@pytest.mark.timeout(30)
def test_a_zero_cost_is_known():
    meter = RunMeter()
    _add(meter, 0.0)
    _add(meter, 0)
    assert meter.cost_usd("planning") == Decimal(0)
    assert meter.unknown_cost_calls("planning") == 0


@pytest.mark.timeout(30)
@pytest.mark.parametrize(
    "cost",
    [None, float("nan"), float("inf"), "n/a", True, Decimal("NaN"), object()],
)
def test_an_unreported_cost_is_unknown_never_zero(cost):
    meter = RunMeter()
    _add(meter, 0.25)
    _add(meter, cost)
    assert meter.calls["planning"] == 2
    assert meter.cost_usd("planning") is None
    assert meter.total_cost_usd() is None
    assert meter.known_cost_usd("planning") == Decimal("0.25")
    assert meter.unknown_cost_calls("planning") == 1
    snapshot = meter.snapshot()
    assert snapshot["cost"] == {"planning": None}
    assert snapshot["cost_known"] == {"planning": "0.25"}
    assert snapshot["cost_unknown_calls"] == {"planning": 1}
    json.dumps(snapshot)  # the snapshot is plain data


@pytest.mark.timeout(30)
def test_a_fresh_meter_costs_zero():
    meter = RunMeter()
    assert meter.cost_usd("planning") == Decimal(0)
    assert meter.total_cost_usd() == Decimal(0)
    assert meter.snapshot()["cost"] == {"planning": "0"}


@pytest.mark.timeout(30)
def test_the_float_view_still_works_and_is_deprecated():
    meter = RunMeter()
    _add(meter, 0.1)
    _add(meter, 0.2)
    _add(meter, None)
    with pytest.warns(DeprecationWarning, match="cost_usd"):
        view = meter.cost
    assert view == {"planning": 0.3}
    assert isinstance(view["planning"], float)


@pytest.mark.timeout(30)
def test_an_unknown_purpose_is_counted_as_planning():
    meter = RunMeter()
    _add(meter, "0.5", purpose="nonsense")
    assert meter.cost_usd("planning") == Decimal("0.5")


@pytest.mark.timeout(30)
def test_concurrent_adds_are_exact():
    meter = RunMeter()

    def worker():
        for _ in range(1000):
            _add(meter, 0.001)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert meter.calls["planning"] == 8000
    assert meter.cost_usd("planning") == Decimal("8.000")


@pytest.mark.timeout(30)
def test_the_listener_counts_a_cache_hit_as_unknown():
    """unillm reports no provider cost for cache hits, streaming and errors."""
    meter = RunMeter()
    token = llm_meter.current_run_meter.set(meter)
    try:
        usage = {"usage": {"prompt_tokens": 7, "completion_tokens": 3}}
        llm_meter._on_llm_event(SimpleNamespace(response=usage, provider_cost=0.002))
        llm_meter._on_llm_event(SimpleNamespace(response=usage, provider_cost=None))
    finally:
        llm_meter.current_run_meter.reset(token)
    assert meter.tokens["planning"] == {"prompt": 14, "completion": 6}
    assert meter.known_cost_usd("planning") == Decimal("0.002")
    assert meter.cost_usd("planning") is None


@pytest.mark.timeout(30)
def test_to_decimal_usd_reads_floats_by_their_shortest_repr():
    assert to_decimal_usd(0.1) == Decimal("0.1")
    assert to_decimal_usd(1.1e-05) == Decimal("0.000011")
    assert to_decimal_usd("12.50") == Decimal("12.50")
    assert to_decimal_usd(None) is None
    assert to_decimal_usd(False) is None

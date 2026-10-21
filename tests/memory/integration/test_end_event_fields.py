"""The end event's spend split by lane, helpers, staging and similarity-miner fields."""

from decimal import Decimal

from unify.memory.integration import consolidate


def test_spend_by_category_splits_writer_and_helpers_by_lane():
    got = consolidate._by_category(
        "0.50",
        [{"usd": "0.10", "usd_known": "0.10"}, {"usd": "0.05", "usd_known": "0.05"}],
    )
    assert got == {
        "writer": consolidate._usd(Decimal("0.35")),
        "helpers": consolidate._usd(Decimal("0.15")),
    }


def test_an_unpriced_helper_call_leaves_the_helpers_share_unknown():
    got = consolidate._by_category(
        "0.50",
        [{"usd": "unknown", "usd_known": "0.10", "unknown_cost_calls": 1}],
    )
    assert got == {
        "writer": consolidate._usd(Decimal("0.40")),
        "helpers": consolidate.UNKNOWN,
    }


def test_unknown_spend_stays_unknown():
    got = consolidate._by_category(consolidate.UNKNOWN, [])
    assert got == {"writer": consolidate.UNKNOWN, "helpers": consolidate.UNKNOWN}

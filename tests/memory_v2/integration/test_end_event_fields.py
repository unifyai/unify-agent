"""Memory v2.1 r5: the end event's spend split, analysts, staging and S0 fields."""

from decimal import Decimal

from unify.memory_v2.integration import consolidate


def test_spend_by_category_splits_writer_and_analysts():
    got = consolidate._by_category("0.50", [{"usd": "0.10"}, {"usd": "0.05"}])
    assert got == {
        "writer": consolidate._usd(Decimal("0.35")),
        "analysts": consolidate._usd(Decimal("0.15")),
    }


def test_unknown_spend_stays_unknown():
    got = consolidate._by_category(consolidate.UNKNOWN, [])
    assert got == {"writer": consolidate.UNKNOWN, "analysts": consolidate.UNKNOWN}

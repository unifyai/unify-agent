"""Symbolic: the block an agent sees at its boundary."""

from unify.agents.delivery import render_block
from unify.agents.entry import Entry


def _e(seq, author, text, kind="post"):
    return Entry(seq, "t", author, kind, text, ())


def test_a_block_lists_mine_in_order_and_counts_the_rest():
    block = render_block(
        [_e(14, "user", "use 2024"), _e(17, "h1", '{"rows": 3}', "reply")],
        [_e(15, "h2", "x"), _e(16, "h2", "y")],
        returned=set(),
        since=13,
        cap_tokens=4000,
        others_line=True,
    )
    assert block.splitlines() == [
        "[record · 2 new for you · #14–#17]",
        "#14 user: use 2024",
        '#17 h1 (reply): {"rows": 3}',
        "(+2 other entries, #15–#16, by h2; record.read(since=13) shows them)",
    ]


def test_entries_returned_to_code_show_as_one_line_but_user_and_cancel_never_do():
    block = render_block(
        [
            _e(3, "h1", "long reply", "reply"),
            _e(4, "user", "stop that"),
            _e(5, "root", "@h1 stop", "cancel"),
        ],
        [],
        returned={3, 4, 5},
        since=2,
        cap_tokens=4000,
        others_line=True,
    )
    assert "#3 h1 (reply) — returned to your code by record.wait" in block
    assert "#4 user: stop that" in block and "#5 root (cancel): @h1 stop" in block


def test_the_cap_names_what_did_not_fit_and_keeps_priority_entries():
    mine = [_e(i, "h1", "w" * 400) for i in range(1, 6)]
    mine.append(_e(6, "user", "the instruction"))
    block = render_block(
        mine,
        [],
        returned=set(),
        since=0,
        cap_tokens=250,
        others_line=True,
    )
    assert "#6 user: the instruction" in block
    assert "more for you" in block and "mentions_me=True" in block


def test_others_line_can_be_turned_off():
    block = render_block(
        [_e(2, "user", "hi")],
        [_e(1, "h1", "x")],
        returned=set(),
        since=0,
        cap_tokens=4000,
        others_line=False,
    )
    assert "other entries" not in block

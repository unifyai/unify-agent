"""Symbolic: the edit engine behind ``UNIFY_FUNCTION_PATCH``'s patch tools.

``apply_edits`` applies ``{old, new}`` edits in order to the evolving text, all
or nothing. Each ``old`` is found on a ladder -- exact, then line endings and
trailing blanks ignored, then indentation compared relative to the block
(``new`` re-indented to it), then runs of spaces and tabs collapsed -- and
the first level that finds it decides: once, it is applied; more than once,
it is refused (unless ``replace_all``) without trying a looser level. No
level matches across a different number of lines. Refusals show the closest
block with a character diff, or the line of every occurrence.
``collect_edits`` turns the tools' arguments into edits. No model is called.
"""

from __future__ import annotations

import random
import re

import pytest

from unify.common.exact_patch import (
    LEVELS,
    PatchEdit,
    PatchRefused,
    apply_edits,
    collect_edits,
    occurrences,
    syntax_refusal,
)

SRC = (
    "def total_minor(rows: list) -> int:\n"
    '    """Return the sum of `amount` over the rows, in minor units."""\n'
    "    total = 0\n"
    "    for row in rows:\n"
    "        total += row['amount']\n"
    "    return total\n"
)


def edit(old: str, new: str, replace_all: bool = False) -> dict:
    return {"old": old, "new": new, "replace_all": replace_all}


def apply(text: str, *edits: dict) -> tuple[str, list[str]]:
    out, report = apply_edits(text, list(edits), what="t")
    return out, [row["match"] for row in report]


def refusal(text: str, *edits: dict) -> str:
    with pytest.raises(PatchRefused) as exc:
        apply_edits(text, list(edits), what="t")
    return str(exc.value)


def test_the_ladder_runs_strictest_first():
    assert LEVELS == (
        "exact",
        "trailing_whitespace",
        "indentation",
        "collapsed_whitespace",
    )


# --------------------------------------------------------------------------- #
#  Level (a): exact                                                            #
# --------------------------------------------------------------------------- #


def test_one_exact_occurrence_is_replaced():
    assert apply("a = 1\nb = 2\n", edit("b = 2", "b = 3")) == (
        "a = 1\nb = 3\n",
        ["exact"],
    )
    assert apply("keep\ndrop\n", edit("drop\n", "")) == ("keep\n", ["exact"])


def test_exact_wins_even_when_a_looser_level_would_also_match():
    text = "x = 1\n    x = 1  \n"
    # Exact finds the first line only inside both lines ("x = 1" twice), so
    # it is ambiguous; a whole-line match at a looser level is never tried.
    assert "occurs 2 times in t (at lines 1, 2)" in refusal(text, edit("x = 1", "y"))


def test_overlapping_occurrences_count():
    assert occurrences("aaa", "aa") == [0, 1]
    message = refusal("aaa", edit("aa", "b"))
    assert "`old` occurs 2 times in t (at lines 1 (2 times))" in message


# --------------------------------------------------------------------------- #
#  Level (b): line endings and trailing whitespace                             #
# --------------------------------------------------------------------------- #


def test_crlf_text_matches_an_lf_old_and_keeps_its_crlf_endings():
    text = SRC.replace("\n", "\r\n")
    out, levels = apply(
        text,
        edit(
            "    total = 0\n    for row in rows:\n",
            "    total = 1\n    for row in rows:\n",
        ),
    )
    assert levels == ["trailing_whitespace"]
    assert out == text.replace("total = 0", "total = 1")
    assert "\n" not in out.replace("\r\n", "")


def test_cr_only_text_matches():
    text = "one\rtwo\rthree\r"
    out, levels = apply(text, edit("two\nthree", "2\n3"))
    assert levels == ["trailing_whitespace"]
    assert out == "one\r2\r3\r"


def test_trailing_blanks_in_the_text_or_in_old_are_ignored():
    text = "alpha  \nbeta\t\ngamma\n"
    out, levels = apply(text, edit("alpha\nbeta\n", "ALPHA\nBETA\n"))
    assert (out, levels) == ("ALPHA\nBETA\ngamma\n", ["trailing_whitespace"])
    out, levels = apply("alpha\nbeta\n", edit("alpha   \nbeta", "A\nB"))
    assert (out, levels) == ("A\nB\n", ["trailing_whitespace"])


def test_trailing_whitespace_ambiguity_is_refused():
    text = "head\nstep one  \nmid\nstep one\ntail\n"
    # With the next line included the block is unique, so it is applied.
    out, levels = apply(text, edit("step one \nmid", "step two\nmid"))
    assert (out, levels) == (
        "head\nstep two\nmid\nstep one\ntail\n",
        ["trailing_whitespace"],
    )
    message = refusal(text, edit("step one \n", "x"))
    assert (
        "`old` occurs 2 times in t when line endings and trailing spaces are "
        "ignored (at lines 2, 4)"
    ) in message


# --------------------------------------------------------------------------- #
#  Level (c): relative indentation                                             #
# --------------------------------------------------------------------------- #


def test_a_dedented_old_matches_and_new_is_reindented():
    out, levels = apply(
        SRC,
        edit(
            "total = 0\nfor row in rows:\n    total += row['amount']\n",
            "total = 0\nfor row in rows:\n    if row:\n        total += row['amount']\n",
        ),
    )
    assert levels == ["indentation"]
    assert out == SRC.replace(
        "        total += row['amount']\n",
        "        if row:\n            total += row['amount']\n",
    )
    compile(out, "<patched>", "exec")


def test_an_over_indented_old_matches_and_new_follows_the_block():
    old = "            for row in rows:\n                total += row['amount']"
    new = "            for row in rows or []:\n                total += row['amount']"
    out, levels = apply(SRC, edit(old, new))
    assert levels == ["indentation"]
    assert out == SRC.replace("for row in rows:", "for row in rows or []:")


def test_tabs_against_spaces_match_and_new_takes_the_files_tabs():
    text = SRC.replace("        ", "\t\t").replace("    ", "\t")
    out, levels = apply(
        text,
        edit(
            "    for row in rows:\n        total += row['amount']\n",
            "    for row in rows:\n        if row:\n            total += row['amount']\n",
        ),
    )
    assert levels == ["indentation"]
    assert "\tfor row in rows:\n\t\tif row:\n\t\t\ttotal += row['amount']\n" in out
    assert "    " not in out


def test_spaces_in_the_file_against_tabs_in_old():
    old = "\tfor row in rows:\n\t\ttotal += row['amount']"
    new = "\tfor row in rows:\n\t\ttotal -= row['amount']"
    out, levels = apply(SRC, edit(old, new))
    assert levels == ["indentation"]
    assert out == SRC.replace("total += row", "total -= row")


def test_new_lines_keep_their_indentation_relative_to_old():
    text = "def f():\n    if a:\n        b()\n"
    out, levels = apply(text, edit("if a:\n    b()", "if a:\n    b()\nc()"))
    assert levels == ["indentation"]
    assert out == "def f():\n    if a:\n        b()\n    c()\n"
    # A line less indented than old's base moves out by the same amount.
    text = "class C:\n    def f(self):\n        if a:\n            b()\n"
    out, levels = apply(
        text,
        edit("    if a:\n        b()", "    if a:\n        b()\nc = 1"),
    )
    assert levels == ["indentation"]
    assert out == (
        "class C:\n    def f(self):\n        if a:\n            b()\n    c = 1\n"
    )
    # An exact match is applied as given, with no re-indentation.
    text = "def f():\n    if a:\n        b()\n"
    out, _ = apply(text, edit("    b()\n", "    b()\nreturn\n"))
    # "    b()" matched exactly inside "        b()"; the edit is applied there.
    assert out == "def f():\n    if a:\n        b()\nreturn\n"


def test_blank_lines_inside_the_block_match_whatever_blanks_they_hold():
    text = "def f():\n    a = 1\n    \n    return a\n"
    out, levels = apply(text, edit("a = 1\n\nreturn a", "a = 2\n\nreturn a"))
    assert levels == ["indentation"]
    assert out == "def f():\n    a = 2\n\n    return a\n"


def test_indentation_ambiguity_is_refused():
    text = "def f():\n    x()\n    y()\n\ndef g():\n        x()\n        y()\n"
    message = refusal(text, edit("  x()\n  y()", "z()"))
    assert (
        "`old` occurs 2 times in t when indentation is compared relative to "
        "the block (at lines 2-3, 6-7)"
    ) in message
    assert "replace_all" in message


# --------------------------------------------------------------------------- #
#  Level (d): collapsed whitespace                                             #
# --------------------------------------------------------------------------- #


def test_runs_of_spaces_and_tabs_count_as_one():
    out, levels = apply(
        SRC,
        edit("total  +=\trow['amount']", "total += int(row['amount'])"),
    )
    assert levels == ["collapsed_whitespace"]
    assert out == SRC.replace("row['amount']", "int(row['amount'])")


def test_collapsed_whitespace_ambiguity_is_refused():
    text = "a  =  1\nb = 2\na = 1\n"
    message = refusal(text, edit("a =   1", "a = 9"))
    assert (
        "`old` occurs 2 times in t when runs of spaces and tabs count as one "
        "(at lines 1, 3)"
    ) in message


def test_no_level_matches_across_a_different_number_of_lines():
    text = "call(a,\n     b)\n"
    message = refusal(text, edit("call(a, b)", "call(b, a)"))
    assert "occurs 0 times in t, even with whitespace differences ignored" in message
    assert "It does match when line breaks are ignored as well" in message
    message = refusal("x = 1\ny = 2\n", edit("x = 1\n\ny = 2", "z"))
    assert "occurs 0 times" in message


# --------------------------------------------------------------------------- #
#  Refusals that say how to retry                                              #
# --------------------------------------------------------------------------- #


def test_zero_matches_show_the_closest_block_and_a_character_diff():
    message = refusal(SRC, edit("total += row['amt']", "x"))
    assert message.startswith(
        "`old` occurs 0 times in t, even with whitespace differences ignored, "
        "so nothing was changed. The closest text is line 5",
    )
    assert "   5 |         total += row['amount']" in message
    assert "- total += row['amt']" in message
    assert "+ total += row['amount']" in message
    assert "\n? " in message  # the caret line under the stored text
    assert "line breaks" not in message


def test_zero_matches_of_a_block_name_its_line_range():
    old = "    for row in rows:\n        total += row['amt']\n    return totals\n"
    message = refusal(SRC, edit(old, "x"))
    assert "The closest text is lines 4-6" in message
    for number in (4, 5, 6):
        assert f"   {number} | " in message


def test_several_matches_list_every_line_and_suggest_a_neighbour_or_replace_all():
    text = "".join(f"x = 1\nline {n}\n" for n in range(5))
    message = refusal(text, edit("x = 1", "x = 9"))
    assert message.startswith("`old` occurs 5 times in t (at lines 1, 3, 5, 7, 9)")
    assert "Add a neighbouring line to `old`" in message
    assert "set `replace_all`" in message
    assert message.count("----") == 2  # three excerpts at most


def test_nothing_close_shows_the_start_of_the_text():
    message = refusal(SRC, edit("zzzzqqqq", "x"))
    assert "Nothing is close; it begins:" in message
    assert "   1 | def total_minor" in message


# --------------------------------------------------------------------------- #
#  Batches                                                                     #
# --------------------------------------------------------------------------- #


def test_edits_apply_in_order_to_the_evolving_text():
    out, levels = apply(
        SRC,
        edit("total = 0", "total = 1"),
        edit("total = 1", "total = 2"),  # only exists after edit 1
        edit("return total", "return total * 2"),
    )
    assert levels == ["exact", "exact", "exact"]
    assert out == SRC.replace("total = 0", "total = 2").replace(
        "return total",
        "return total * 2",
    )


def test_an_edit_that_was_ambiguous_can_be_made_unique_by_an_earlier_one():
    text = "x = 1\nx = 1\n"
    out, _ = apply(text, edit("x = 1\nx", "y = 1\nx"), edit("x = 1", "x = 2"))
    assert out == "y = 1\nx = 2\n"


def test_a_failing_edit_names_itself_and_nothing_is_applied():
    message = refusal(
        SRC,
        edit("total = 0", "total = 1"),
        edit("total = 0", "total = 5"),  # edit 1 consumed it
        edit("return total", "return -total"),
    )
    assert message.startswith(
        "Edit 2 of 3 (matched in the text as edit 1 left it; no edit was kept): "
        "`old` occurs 0 times in t",
    )
    # The excerpt shows the evolving text, where edit 1 applied.
    assert "   3 |     total = 1" in message


def test_edits_that_cancel_out_are_refused():
    message = refusal(
        SRC,
        edit("total = 0", "total = 1"),
        edit("total = 1", "total = 0"),
    )
    assert "the edits leave t as it was" in message


def test_levels_are_reported_per_edit():
    text = "def f():\r\n    a = 1\r\n    b  =  2\r\n    if a:\r\n        c()\r\n"
    out, levels = apply(
        text,
        edit("    a = 1\r\n", "    a = 10\r\n"),
        edit("a = 10\n", "a = 11\n"),
        edit("b = 2", "b = 20"),
        edit("if a:\n    c()\n", "if a:\n    d()\n"),
    )
    assert levels == [
        "exact",
        "trailing_whitespace",
        "collapsed_whitespace",
        "indentation",
    ]
    assert out == (
        "def f():\r\n    a = 11\r\n    b = 20\r\n    if a:\r\n        d()\r\n"
    )


# --------------------------------------------------------------------------- #
#  replace_all                                                                 #
# --------------------------------------------------------------------------- #


def test_replace_all_replaces_every_occurrence_and_counts_them():
    out, report = apply_edits(SRC, [edit("total", "acc", replace_all=True)], what="t")
    assert out == SRC.replace("total", "acc")
    assert report == [{"match": "exact", "replaced": 4}]


def test_replace_all_takes_occurrences_left_to_right_without_overlap():
    out, report = apply_edits("aaaa", [edit("aa", "b", replace_all=True)], what="t")
    assert (out, report) == ("bb", [{"match": "exact", "replaced": 2}])


def test_replace_all_uses_the_first_level_that_finds_old():
    text = "x  =  1\ny = 2\nx =\t1\n"
    out, report = apply_edits(
        text,
        [edit("x = 1", "x = 0", replace_all=True)],
        what="t",
    )
    assert out == "x = 0\ny = 2\nx = 0\n"
    assert report == [{"match": "collapsed_whitespace", "replaced": 2}]


def test_replace_all_still_needs_one_match():
    assert "occurs 0 times" in refusal(SRC, edit("nope", "x", replace_all=True))


# --------------------------------------------------------------------------- #
#  Arguments                                                                   #
# --------------------------------------------------------------------------- #


def test_old_and_new_are_one_edit():
    assert collect_edits(old="a", new="b") == [edit("a", "b")]
    assert collect_edits(old="a", new="", replace_all=True) == [edit("a", "", True)]


def test_the_edit_tool_names_are_accepted():
    assert collect_edits(old_string="a", new_string="b") == [edit("a", "b")]
    assert collect_edits(old="a", old_string="a", new_string="b") == [edit("a", "b")]
    assert collect_edits(edits=[{"old_string": "a", "new_string": "b"}]) == [
        edit("a", "b"),
    ]


def test_edits_arrive_as_objects_models_a_single_object_or_json():
    expected = [edit("a", "b"), edit("c", "d", True)]
    assert (
        collect_edits(
            edits=[
                {"old": "a", "new": "b"},
                {"old": "c", "new": "d", "replace_all": True},
            ],
        )
        == expected
    )
    assert (
        collect_edits(
            edits=[
                PatchEdit(old="a", new="b"),
                PatchEdit(old="c", new="d", replace_all=True),
            ],
        )
        == expected
    )
    assert collect_edits(edits={"old": "a", "new": "b"}) == [edit("a", "b")]
    # Empty top-level fields beside `edits`, as schema-filling models send.
    assert collect_edits(old="", new="", edits=[{"old": "a", "new": "b"}]) == [
        edit("a", "b"),
    ]
    assert (
        collect_edits(
            edits='[{"old": "a", "new": "b"}, {"old": "c", "new": "d", "replace_all": "true"}]',
        )
        == expected
    )


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({}, "give the text to change as `old` and `new`, or several"),
        ({"edits": []}, "give the text to change as `old` and `new`"),
        ({"old": "a"}, "`new` is missing"),
        ({"old": "", "new": "b"}, "`old` must be the non-empty text"),
        ({"old": "a", "new": "a"}, "nothing to change"),
        ({"old": "a", "old_string": "b", "new": "c"}, "`old` and `old_string` differ"),
        (
            {"old": "a", "new": "b", "edits": [{"old": "c", "new": "d"}]},
            "give either `old` and `new` or `edits`, not both",
        ),
        (
            {"edits": [{"old": "c", "new": "d"}], "replace_all": True},
            "set `replace_all` inside the edit",
        ),
        ({"edits": "not json"}, "`edits` must be a list of {old, new} objects"),
        ({"edits": [7]}, "each edit must be an object"),
        (
            {"edits": [{"old": "a", "new": "b"}, {"old": "c", "new": "d", "why": "x"}]},
            "edit 2 of 2: unknown keys 'why'",
        ),
        (
            {"edits": [{"old": "a", "new": "b"}, {"old": "c"}]},
            "edit 2 of 2: `new` is missing",
        ),
        (
            {"edits": [{"old": "a", "new": "b", "replace_all": "maybe"}]},
            "`replace_all` must be true or false",
        ),
    ],
)
def test_malformed_arguments_are_refused(kwargs, match):
    with pytest.raises(PatchRefused, match=re.escape(match)):
        collect_edits(**kwargs)


def test_syntax_refusal_names_the_line():
    assert syntax_refusal(SRC) is None
    broken = SRC.replace("    return total\n", "  return total\n")
    problem = syntax_refusal(broken)
    assert problem.startswith("line 6: unindent does not match any outer indentation")
    assert "   6 |   return total" in problem


def test_an_old_that_starts_with_a_line_break_takes_the_whole_crlf():
    out, levels = apply("a\r\nb  \r\nc\r\n", edit("\nb\n", "\nB\n"))
    assert (out, levels) == ("a\r\nB\r\nc\r\n", ["trailing_whitespace"])


# --------------------------------------------------------------------------- #
#  Randomised: a copied block, mangled as models mangle it, still patches      #
# --------------------------------------------------------------------------- #


def _random_text(rng: random.Random, tabs: bool) -> list[str]:
    """Lines each carrying a unique id token, at random nesting, some blank."""
    lines, depth = [], 0
    for number in range(rng.randint(3, 14)):
        if rng.random() < 0.1:
            lines.append("")
            continue
        depth = max(0, min(3, depth + rng.choice((-1, 0, 0, 1))))
        lead = "\t" * depth if tabs else "    " * depth
        words = " ".join(rng.choice("abcdef") for _ in range(rng.randint(0, 3)))
        lines.append(f"{lead}L{number} {words}".rstrip())
    return lines


def _mangle(block: list[str], how: str) -> list[str]:
    if how == "dedent":
        spaces = [row.expandtabs(4) for row in block]
        cut = min(len(r) - len(r.lstrip()) for r in spaces if r.strip())
        return [row[cut:] for row in spaces]
    if how == "shift":
        return ["  " + row.expandtabs(4) if row else row for row in block]
    if how == "trailing":
        return [row + " \t " for row in block]
    if how == "collapse":
        return [
            row[: len(row) - len(row.lstrip())] + row.lstrip().replace(" ", "  ")
            for row in block
        ]
    return list(block)


def test_a_mangled_copy_of_a_unique_block_patches_that_block():
    levels = set()
    for seed in range(400):
        levels.add(_patch_a_mangled_block(seed))
    assert levels == set(LEVELS)  # every level was exercised


def _patch_a_mangled_block(seed: int) -> str:
    rng = random.Random(seed)
    tabs = rng.random() < 0.3
    lines = _random_text(rng, tabs)
    ids = [i for i, row in enumerate(lines) if row.strip()]
    first = rng.choice(ids)
    last = min(len(lines) - 1, first + rng.randint(0, 3))
    target = rng.choice([i for i in ids if first <= i <= last])
    eol = rng.choice(("\n", "\r\n"))
    how = rng.choice(("none", "dedent", "shift", "trailing", "collapse"))
    whole = rng.random() < 0.5  # copy the block's final line break too

    text = eol.join(lines) + eol
    block = lines[first : last + 1]
    old_rows = _mangle(block, how)
    token = lines[target].split()[0]
    # The model's `new` is its mangled copy, edited -- without the stray
    # trailing blanks or doubled spaces, which it would not mean to store.
    clean = {"trailing": [r.rstrip(" \t") for r in old_rows], "collapse": block}
    new_rows = [row.replace(token, "M" + token[1:]) for row in clean.get(how, old_rows)]
    old = "\n".join(old_rows) + ("\n" if whole else "")
    new = "\n".join(new_rows) + ("\n" if whole else "")
    expected = text.replace(token + " ", "M" + token[1:] + " ").replace(
        token + eol,
        "M" + token[1:] + eol,
    )

    out, report = apply_edits(text, [edit(old, new)], what="t")
    assert out == expected, (seed, how, eol, old)
    assert report[0]["replaced"] == 1
    if how == "trailing":
        assert report[0]["match"] == "trailing_whitespace"
    if how == "collapse" and any("  " in row.lstrip() for row in old_rows):
        assert report[0]["match"] == "collapsed_whitespace"
    return report[0]["match"]

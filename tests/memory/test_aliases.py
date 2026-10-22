"""Aliases and calls in a library-mode library: pure AST."""

from unify.memory.aliases import forwards

MODS = {"memory.text", "memory.text.parse", "memory.text.split", "memory.text.dates"}


def test_the_three_alias_forms():
    src = (
        "from memory.text.parse import tokens\n"
        "from .parse import words as count_words\n"
        "import memory.text.dates as dates\n\n\n"
        "def split_words(text):\n    return text.split()\n\n\n"
        "split = split_words\n"
        "old_tokens = tokens\n"
        "when = dates.parse_date\n\n\n"
        'def comma_tokens(text, limit=-1):\n    """Tokens split on commas."""\n    return tokens(text, ",", limit)\n'
    )
    assert forwards(src, "memory.text.split", MODS) == {
        "comma_tokens": "memory.text.parse:tokens",
        "count_words": "memory.text.parse:words",
        "old_tokens": "memory.text.parse:tokens",
        "split": "memory.text.split:split_words",
        "tokens": "memory.text.parse:tokens",
        "when": "memory.text.dates:parse_date",
    }


def test_what_is_not_a_forward():
    src = (
        "import json\n\n\n"
        "def doubled(x):\n    return twice(x) + 1\n\n\n"  # more than a forwarded call
        "def twice(x):\n    return x * 2\n\n\n"
        "def stamped(x):\n    print(x)\n    return twice(x)\n\n\n"  # a statement before the return
        "def mixed(x, y):\n    return twice(x + y)\n\n\n"  # an argument that is an expression
        "def again(x):\n    return twice(x, x)\n\n\n"  # a parameter passed twice
        "dumps = json.dumps\n"  # not a library function
        "_private = twice\n"
    )
    assert forwards(src, "memory.text.calc", {"memory.text.calc"}) == {}
    assert forwards("def x(:\n", "memory.text.calc", set()) is None


def test_the_last_binding_wins():
    src = "from memory.text.parse import tokens\n\n\nold = tokens\n\n\ndef old(s):\n    return s\n"
    assert forwards(src, "memory.text.split", MODS) == {
        "tokens": "memory.text.parse:tokens",
    }

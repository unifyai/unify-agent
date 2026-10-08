"""The lean docstring standard (v2.1 stage 3): parsing, the gate's reasons, self-explaining refusals."""

import ast

from unify.memory_v2 import docstrings

FULL = """Read the ledger into rows.

The ledger is the accounts-payable CSV.

Args:
    path: the ledger CSV file (a path).
    strict: refuse rows with an empty
        amount.

Returns:
    A list of dicts, one per row.

Raises:
    MemoryInputError: when the file is not a CSV with the ledger's columns.

Example:
    >>> rows = read_ledger("env/w/tests/data/ledger.csv")
    >>> rows[0]["vendor_id"]
    'V-17'

Use when: you hold the ledger file.
Don't use when: the file is another register.

Effect: read
Input: path"""


def _fn(src):
    return ast.parse(src).body[0]


def test_parse_reads_every_section_and_the_line_fields():
    d = docstrings.parse(FULL)
    assert d.summary == "Read the ledger into rows."
    assert d.description == "The ledger is the accounts-payable CSV."
    assert d.args == [
        ("path", "the ledger CSV file (a path)."),
        ("strict", "refuse rows with an empty amount."),
    ]
    assert d.sections["Returns"] == "A list of dicts, one per row."
    assert d.sections["Raises"].startswith("MemoryInputError:")
    assert (
        d.sections["Example"].splitlines()[0]
        == '>>> rows = read_ledger("env/w/tests/data/ledger.csv")'
    )
    assert d.sections["Use when"] == "you hold the ledger file."
    assert d.sections["Don't use when"] == "the file is another register."
    assert d.lines == {"Effect": "read", "Input": "path"}
    assert docstrings.has_example(d)


def test_a_complete_docstring_has_no_problems():
    assert docstrings.problems(FULL, ["path", "strict"], "path") == []


def test_each_missing_required_part_is_named():
    no_returns = FULL.replace("Returns:\n    A list of dicts, one per row.\n\n", "")
    assert docstrings.problems(no_returns, ["path", "strict"], "path") == [
        "has no Returns: section",
    ]
    no_example = FULL.split("Example:")[0] + "Effect: read\nInput: path"
    assert docstrings.problems(no_example, ["path", "strict"], "path") == [
        "has no Example: section with a `>>>` example",
    ]
    assert docstrings.problems(FULL, ["path", "strict", "limit"], "path") == [
        "Args: does not describe parameter(s) limit",
    ]
    assert docstrings.problems(FULL, ["path", "strict"], "bytes") == [
        "Args: the entry of the first parameter `path` does not name its input form `bytes`",
    ]
    no_error = FULL.replace("MemoryInputError: when", "ValueError: when")
    assert docstrings.problems(no_error, ["path", "strict"], "path") == [
        "Raises: does not name MemoryInputError and when it is raised",
    ]
    bare = "Read it.\n\nEffect: read\nInput: path"
    assert docstrings.problems(bare, ["path"], "path") == [
        "has no Args: section (one `name: description` line per parameter)",
        "has no Returns: section",
        "has no Raises: section naming MemoryInputError and when it is raised",
        "has no Example: section with a `>>>` example",
    ]
    assert "has no one-line summary (its first line)" in docstrings.problems(
        "",
        [],
        None,
    )


def test_examples_heading_is_accepted_and_inline_text_is_kept():
    doc = FULL.replace("Example:", "Examples:").replace(
        "Returns:\n    A list of dicts, one per row.",
        "Returns: a list of dicts, one per row.",
    )
    d = docstrings.parse(doc)
    assert d.sections["Returns"] == "a list of dicts, one per row."
    assert docstrings.problems(doc, ["path", "strict"], "path") == []


def test_refusals_must_say_what_was_expected():
    good = _fn(
        "def f(path):\n"
        "    if not path:\n"
        "        raise MemoryInputError(f'expected a CSV path, got {path!r}; read the file directly')\n",
    )
    assert docstrings.refusal_problems(good) == []
    short = _fn("def f(x):\n    raise MemoryInputError('bad input')\n")
    (reason,) = docstrings.refusal_problems(short)
    assert "9-character message (line 2)" in reason and "at least 24" in reason
    bare = _fn("def f(x):\n    raise MemoryInputError\n")
    assert "without a message (line 2)" in docstrings.refusal_problems(bare)[0]
    empty_call = _fn("def f(x):\n    raise mod.MemoryInputError()\n")
    assert "without a message" in docstrings.refusal_problems(empty_call)[0]
    variable = _fn("def f(x, msg):\n    raise MemoryInputError(msg)\n")
    assert docstrings.refusal_problems(variable) == []  # unmeasurable: accepted
    other = _fn("def f(x):\n    raise ValueError('x')\n")
    assert docstrings.refusal_problems(other) == []


def test_params_and_catalog_form():
    fn = _fn("def f(a, /, b, *rest, c=1, **kw):\n    pass\n")
    assert docstrings.params(fn) == ["a", "b", "rest", "c", "kw"]
    cat = docstrings.as_catalog(docstrings.parse(FULL))
    assert set(cat) == {
        "description",
        "args",
        "returns",
        "raises",
        "example",
        "use_when",
        "dont_use_when",
        "arg_list",
    }
    assert cat["arg_list"][0] == {
        "name": "path",
        "text": "the ledger CSV file (a path).",
    }


def test_the_standard_text_is_generated_from_the_constants():
    text = docstrings.describe_standard()
    for name in docstrings.REQUIRED_SECTIONS + docstrings.OPTIONAL_SECTIONS:
        assert f"`{name}:`" in text
    assert str(docstrings.MIN_REFUSAL_CHARS) in text and "MemoryInputError" in text

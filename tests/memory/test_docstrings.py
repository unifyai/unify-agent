"""Docstring sections as the When-to-use reader parses them."""

from unify.memory import docstrings

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


def test_examples_heading_is_accepted_and_inline_text_is_kept():
    doc = FULL.replace("Example:", "Examples:").replace(
        "Returns:\n    A list of dicts, one per row.",
        "Returns: a list of dicts, one per row.",
    )
    d = docstrings.parse(doc)
    assert d.sections["Returns"] == "a list of dicts, one per row."


def test_section_aliases():
    d = docstrings.parse(
        "Read.\n\nParameters:\n    path: a path.\n\nReturn:\n    Rows.\n",
    )
    assert d.args == [("path", "a path.")] and d.sections["Returns"] == "Rows."

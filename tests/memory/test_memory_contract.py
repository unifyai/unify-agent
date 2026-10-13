"""Symbolic: every public call of the library package meets the contract, and its errors teach.

The contract (tests/contract_lint.py): type hints on every parameter and the return, a docstring with Args,
Returns, Raises and an Example, arguments checked at the call. Its errors carry the call as written, what was wrong,
then what was expected with an example call or the right call; nothing unfilled and no host path. The errors are
raised in a generated copy of the library, in a fresh interpreter, as a cell raises them.
"""

from __future__ import annotations

import json

from tests.contract_lint import contract_gaps, error_gaps
from tests.memory.test_library_helper import _library, _py
from unify.common import contract_runtime as cr
from unify.memory import library_ask, library_helper

CALLS = {"memory.get": library_helper.get, "memory.items": library_helper.items, "memory.ask": library_ask.ask}
#: Members of a result: documented and typed like a call; their arguments are the item's own (Item.__call__) or
#: free-form inputs (Item.fits), so they are not checked at the call.
MEMBERS = {"Item.fits": library_helper.Item.fits, "Item.__call__": library_helper.Item.__call__}


def test_every_public_call_meets_the_contract():
    gaps = [g for name, fn in CALLS.items() for g in contract_gaps(name, fn)]
    gaps += [
        g
        for name, fn in MEMBERS.items()
        for g in contract_gaps(name, fn)
        if "not checked at the call" not in g
    ]
    assert gaps == [], "\n".join(gaps)


#: Calls that break the contract in a generated copy, one per error kind.
WRONG = [
    "memory.get('stencil.md')",
    "memory.get('memory.text.dates:nope')",
    "memory.items('notes/text/dates.md')",
    "memory.get(3)",
    "memory.items(kind='notess')",
]


def test_every_error_teaches(tmp_path):
    root = _library(tmp_path)
    code = (
        "import json, memory\n"
        "out = []\n"
        f"for call in {WRONG!r}:\n"
        "    try:\n"
        "        eval(call)\n"
        "        out.append(None)\n"
        "    except memory.ContractError as exc:\n"
        "        out.append({'kind': type(exc).__name__, 'call': exc.call, 'problem': exc.problem, "
        "'expected': exc.expected, 'example': exc.example, 'nearest': list(exc.nearest), 'hint': exc.hint, "
        "'text': str(exc)})\n"
        "    except Exception as exc:\n"
        "        out.append({'raw': type(exc).__name__ + ': ' + str(exc)[:200]})\n"
        "print(json.dumps(out))\n"
    )
    for call, got in zip(WRONG, json.loads(_py(root, code))):
        assert got is not None, f"{call} raised nothing"
        assert "raw" not in got, f"{call} raised a raw exception: {got['raw']}"
        kind = getattr(cr, got["kind"])
        rebuilt = kind(
            got["call"],
            got["problem"],
            expected=got["expected"],
            example=got["example"],
            nearest=got["nearest"],
            hint=got["hint"],
        )
        assert str(rebuilt) == got["text"], call  # rendered by the one renderer
        assert error_gaps(rebuilt) == [], (call, error_gaps(rebuilt), got["text"])

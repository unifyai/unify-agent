"""A small library, its item records, its episodes' summaries and two captured procedures, shared by the librarian
tests (``UNIFY_MEMORY_LIBRARIAN``: ``memory.ask`` over the whole catalogue).

Planted on purpose, so the tests can show that none of them reaches what an agent sees:

* printed identifiers in the summaries' ``## Asked`` and ``## Approach`` sections and in one docstring: a UUID, an
  alias of the ``TASK-4471`` form and a recorded key of an identifier's form;
* an answer value, in the summaries' ``## Worked`` and ``## Outcome`` sections and in a recorded example call;
* a raw request text, as a module constant and in a captured procedure's code.

No model is called and nothing here imports a box.
"""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

from unify.memory import captures as cap
from unify.memory import item_records
from unify.memory import library_export as lx
from unify.memory import memory_helper

# ==== ASSUMED NAMES (the integrator adapts these to the code, here only) ======================================
#: The generated files of the librarian (design J(b).3), by path in the copy of the library.
CATALOGUE_FILE = ".memory/catalogue.md"
DISPATCH_FILE = ".memory/dispatch.json"
#: The item-record key naming a landed procedure's reader (an item id), as the manifest's ``reader`` (I.2.6).
RECORD_READER_KEY = "reader"


def generate(tree: Path, *, records=None, summaries=None, captures=None) -> dict[str, bytes]:
    """Every generated file of the copy at *tree* with the librarian on: :func:`.library_export.generated_library_mode`
    given the item records, each episode's ``summary.md`` (*summaries*: episode id -> text) and the capture store's
    snapshot (*captures*: ``captures.export_files`` paths -> bytes), so the catalogue and the dispatch file can be
    built from them."""
    summaries = dict(summaries or {})

    def summary_of(episode: str):
        return summaries.get(episode)

    return lx.generated_library_mode(
        tree,
        status_of=item_records.status_of(records) if records else None,
        records=records,
        saved_against=lx.saved_against_of(records, summary_of) if records else None,
        summary_of=summary_of,
        captures=dict(captures or {}),
        librarian=True,
    )


def generate_off(tree: Path, *, records=None, summaries=None, captures=None) -> dict[str, bytes]:
    """The same with the librarian off: the call as the build makes it without this feature."""
    summaries = dict(summaries or {})
    return lx.generated_library_mode(
        tree,
        status_of=item_records.status_of(records) if records else None,
        records=records,
        saved_against=lx.saved_against_of(records, summaries.get) if records else None,
    )


# ==============================================================================================================

UUID = "3f2c8a10-9b7e-4c51-a2d4-6e0f1b9c7d35"
ALIAS = "TASK-4471"
RECORDED_KEY = "acct_99812"
IDENTIFIERS = (UUID, ALIAS, RECORDED_KEY)
ANSWER = "731942"
RAW_REQUEST = "Please total the amounts in the attached order sheet and reply with one number only."
#: A value only a reader computes (it is not in any text the actor passes): base64 in the text, decoded by a reader.
PLANTED_VALUE = "PLANTED-VALUE-QUOKKA-5521"
PLANTED_B64 = base64.b64encode(PLANTED_VALUE.encode()).decode()

TOTAL = "memory.orders.sums:total_amount"
COUNT = "memory.orders.sums:count_rows"
READ_ORDERS = "memory.orders.readers:read_orders"
RECONCILE = "memory.billing.ledger:reconcile"
TOKENS = "memory.text.parse:tokens"
WORDS = "memory.text.parse:words"
NOTE = "notes/orders/totals.md"
#: Every function of the library, in module-path order (the catalogue's order).
FUNCTIONS = [RECONCILE, READ_ORDERS, COUNT, TOTAL, TOKENS, WORDS]

SUMS = f'''"""Sums over order tables."""

SAMPLE_REQUEST = {RAW_REQUEST!r}


def total_amount(rows: list) -> int:
    """Sum the amount column of an order table.
    Each row is a record keyed by column name.

    The second paragraph says more: SECOND-PARAGRAPH-MARK.

    Notes: {NOTE}
    """
    return sum(r["amount"] for r in rows)


def count_rows(rows: list) -> int:
    """Count the rows of a table."""
    return len(rows)


def _helper(x):
    return x
'''
READERS = '''"""Readers of request texts."""

import json


def read_orders(text: str) -> list:
    """Read the order table that follows the line TABLE in a request."""
    return json.loads(text.split("TABLE\\n", 1)[1].splitlines()[0])
'''
LEDGER = f'''"""Ledger work."""


def reconcile(ledger: list, payments: list) -> list:
    """Match ledger rows against payments, as for case {UUID}.

    Returns the unmatched rows.
    """
    paid = {{p["ref"] for p in payments}}
    return [r for r in ledger if r["ref"] not in paid]
'''
PARSE = '''def tokens(s):
    """Split text on whitespace."""
    return s.split()


def words(s):
    """Count the words of a text."""
    return len(s.split())
'''
NOTE_TEXT = (
    "---\ntitle: Order totals\ndescription: How order totals were worked out.\n"
    f"uses: [{TOTAL}]\n---\nSum the amount column; amounts are whole numbers.\n"
)
#: The library's files. Written in an order that is not the module-path order, so an order taken from the
#: filesystem or from insertion is caught.
FILES = {
    "memory/text/parse.py": PARSE,
    "memory/orders/sums.py": SUMS,
    "memory/billing/ledger.py": LEDGER,
    "memory/orders/readers.py": READERS,
    NOTE: NOTE_TEXT,
}

ROWS = [{"item": "pen", "amount": 3}, {"item": "ink", "amount": 4}]


def summary(asked: str, approach: str) -> str:
    """A ``summary.md`` as the episode-end fork writes it (design H.4), with the answer in Worked and Outcome."""
    return (
        f"## Asked\n\n{asked}\n\n## Approach\n\n{approach}\n\n## Worked\n\nThe total came to {ANSWER}.\n\n"
        f"## Failed\n\nnone\n\n## Outcome\n\naccepted: {ANSWER}\n"
    )


TOTAL_ASKED = f"Total the amount column of the order table for {UUID} ({ALIAS})."
TOTAL_APPROACH = f"Read the rows under the key {RECORDED_KEY} as records and summed the amount column."
LEDGER_ASKED = f"Find the ledger rows of {ALIAS} that no payment matches."
LEDGER_APPROACH = "Collected the payment refs and kept the ledger rows whose ref was not among them."
SUMMARIES = {
    "ep-orders-a": summary(TOTAL_ASKED, TOTAL_APPROACH),
    "ep-ledger-a": summary(LEDGER_ASKED, LEDGER_APPROACH),
    "ep-capture-a": summary(
        f"Turn the rows of a grid into columns for {UUID}.",
        f"Read the grid after the line GRID and zipped its rows; key {RECORDED_KEY}.",
    ),
    "ep-capture-b": summary("Say whether a marked text holds a code.", "Decoded the marked part of the text."),
}


def _signature(value, result, goal: str, applies: bool = False) -> dict:
    """An applicability signature as the capture store writes it (design I.2.5)."""
    return {
        "input": memory_helper.value_shape(value),
        "output": memory_helper.value_shape(result) if isinstance(result, (list, dict)) else {"kind": "scalar"},
        "relations": ["scalar_output"] if not isinstance(result, (list, dict)) else [],
        "constraints": {
            "applies": {"stated": applies, "held": True if applies else None, "sha256": "ab" * 32 if applies else None},
            "holds": {"stated": False, "held": None, "sha256": None},
        },
        "goal": goal,
    }


TOTAL_SIGNATURE = _signature(ROWS, 7, "Total the amount column of the order table.", applies=True)
LEDGER_SIGNATURE = _signature(
    {"ledger": [{"ref": "a", "amount": 1}], "payments": [{"ref": "a"}]},
    [{"ref": "b", "amount": 2}],
    "Find the ledger rows that no payment matches.",
)


def records() -> dict[str, dict]:
    """The item records at the commit: levels, statuses, provenance, use counts, signatures, a reader, and an
    example call quoting the answer."""
    out: dict[str, dict] = {}
    for item in FUNCTIONS:
        rec = item_records.empty_record(item, "function")
        rec["status"] = "stable" if item == TOTAL else "experimental"
        out[item] = rec
    total = out[TOTAL]
    total.update(
        level="reproduces-its-case",
        signature=TOTAL_SIGNATURE,
        provenance={"episodes": ["ep-orders-a"], "pass": None, "failure_only": False},
        use={"1" * 40: {"episodes": 3, "uses": 5, "seen": 5, "errors": 1, "refused": 0}},
        example={"test": "memory/orders/tests/test_sums.py", "line": 4, "call": f"total_amount(rows) == {ANSWER}"},
    )
    total[RECORD_READER_KEY] = READ_ORDERS
    out[RECONCILE].update(
        level="unchecked",
        signature=LEDGER_SIGNATURE,
        provenance={"episodes": ["ep-ledger-a"], "pass": None, "failure_only": False},
    )
    note = item_records.empty_record(NOTE, "note")
    out[NOTE] = note
    return out


# --- captured procedures ---------------------------------------------------------------------------------------

GRID = [[1, 2, 3], [4, 5, 6]]
CAPTURE_GRID = f'''"""Turn a grid's rows into its columns.

Kept from an attempt that was accepted.
"""

import json

SEEN = {RAW_REQUEST!r}


def read_input(text):
    """The grid that follows the line GRID in the request."""
    return json.loads(text.split("GRID\\n", 1)[1].splitlines()[0])


def applies(value):
    """A list of equal-length rows."""
    return isinstance(value, list) and len({{len(r) for r in value}}) == 1


def run(value):
    """Turn a grid's rows into its columns."""
    return [list(c) for c in zip(*value)]
'''
CAPTURE_MARKED = '''"""Say whether a marked text holds a code.

Kept from an attempt; not shown whether it was accepted.
"""

import base64


def read_input(text):
    """The decoded part of the text after the line MARKED."""
    return base64.b64decode(text.split("MARKED\\n", 1)[1].split()[0]).decode()


def run(value):
    """Whether the decoded part holds a code."""
    return "-" in value
'''


def capture_id(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]


def capture_files(sources: dict[str, tuple[str, dict]] | None = None) -> dict[str, bytes]:
    """The actor's snapshot of a capture store (``captures.export_files``' layout) holding the given procedures:
    episode id -> (procedure source, signature)."""
    sources = sources if sources is not None else {
        "ep-capture-a": (CAPTURE_GRID, _signature(GRID, [[1, 4], [2, 5], [3, 6]], "Turn the rows of a grid into columns.", True)),
        "ep-capture-b": (CAPTURE_MARKED, _signature("a-b", True, "Say whether a marked text holds a code.")),
    }
    out: dict[str, bytes] = {}
    for episode, (source, sig) in sources.items():
        cid = capture_id(source)
        rec = cap.empty_record(cid)
        rec["episodes"] = [episode]
        rec["cases"] = [{"episode": episode, "reproduces": "unknown", "deterministic": True, "outcome": "none"}]
        rec = cap.refresh(rec)
        base = f"{cap.EXPORT_DIR}/{cid}"
        out[f"{base}/{cap.PROCEDURE_FILE}"] = source.encode()
        out[f"{base}/{cap.RECORD_FILE}"] = (json.dumps(rec, sort_keys=True, indent=1) + "\n").encode()
        out[f"{base}/{cap.SIGNATURE_FILE}"] = (json.dumps(sig, sort_keys=True, indent=1) + "\n").encode()
    return out


def write_tree(root: Path, files: dict | None = None) -> Path:
    for rel, text in (FILES if files is None else files).items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    return root


def big_files(n: int = 60) -> dict[str, str]:
    """A larger library: *n* modules of two functions each, beside :data:`FILES`."""
    out = dict(FILES)
    for i in range(n):
        out[f"memory/bulk/part_{chr(97 + i // 26)}{chr(97 + i % 26)}.py"] = (
            f'def first(x):\n    """Return x plus {i}."""\n    return x + {i}\n\n\n'
            f'def second(x):\n    """Return x times {i}."""\n    return x * {i}\n'
        )
    return out

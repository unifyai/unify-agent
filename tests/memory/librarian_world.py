"""A small library, its item records, its episodes' summaries and two captured procedures, shared by the librarian
tests (``UNIFY_MEMORY_LIBRARIAN``: ``memory.ask`` over the whole catalogue).

The item records are what production writes (:mod:`tests.memory.gate_records`): a writer pass lands the capture of
episode ``ep-orders-a`` as :data:`TOTAL` and writes the rest of the library (its readers, functions, tests and parsed
fixtures), the library gate checks it, and the records are ``lifecycle.next_records`` over that tree with the gate's
verification and the capture store's landing. By what production writes, the library holds:

* a landed procedure, :data:`TOTAL`: its record names the capture and the reader it was landed with
  (:data:`READ_TOTAL`), and carries the capture's signature and its ``applies`` (:data:`TOTAL_APPLIES`); the gate
  finds a second reader of it, :data:`READ_ORDERS`, whose parsed fixture the writer's own test of it loads;
* a function the gate finds exactly one reader of, :data:`COUNT` (:data:`READ_ORDERS`);
* a function the gate finds two readers of, with no ``applies``, :data:`RECONCILE` (:data:`READ_LEDGER` and
  :data:`READ_PAYMENTS`);
* two functions and a note stored before the pass, which the gate does not check (:data:`TOKENS`, :data:`WORDS`,
  :data:`NOTE`).

Only :data:`TOTAL`'s status and use are set by hand, as three accepted episodes would leave them.

Planted on purpose, so the tests can show what reaches what an agent sees:

* printed identifiers in the summaries' ``## Asked`` and ``## Approach`` sections and in one docstring: a UUID, an
  alias of the ``TASK-4471`` form and a recorded key of an identifier's form (rev 2.1: shown as written, never
  stripped or masked);
* an answer value, in the summaries' ``## Worked`` and ``## Outcome`` sections and as the value the landed
  procedure's episode handed back (its recorded-result fixture);
* a raw request text, as a module constant, in the landed procedure's code and a captured procedure's, and in the
  landed episode's request.

No model is called. Building the records runs code confined (:data:`needs_box`), once per process.
"""

from __future__ import annotations

import atexit
import base64
import copy
import functools
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

from tests.memory.gate_records import Built, Landing, build, needs_box  # noqa: F401 (needs_box: for the tests)
from tests.memory.test_episodes import _ep
from unify.memory import captures as cap
from unify.memory import item_records
from unify.memory import library_export as lx
from unify.memory import memory_helper
from unify.memory.episodes import Action

# ==== ASSUMED NAMES (the integrator adapts these to the code, here only) ======================================
#: The generated files of the librarian (design J(b).3), by path in the copy of the library.
CATALOGUE_FILE = ".memory/catalogue.md"
DISPATCH_FILE = ".memory/dispatch.json"


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

#: The landed procedure, the reader it was landed with and its ``applies``, as ``land_capture`` names them.
TOTAL = "memory.orders.procedures:total_amount"
READ_TOTAL = "memory.orders.readers:read_total_amount"
TOTAL_APPLIES = "memory.orders.procedures:_total_amount_applies"
#: The landed procedure's recipe, the whole job on a request (``land_capture`` writes it beside the procedure).
TOTAL_RECIPE = "memory.orders.procedures:apply_total_amount"
COUNT = "memory.orders.sums:count_rows"
READ_ORDERS = "memory.orders.tables:read_orders"
RECONCILE = "memory.billing.ledger:reconcile"
READ_LEDGER = "memory.billing.readers:read_ledger"
READ_PAYMENTS = "memory.billing.readers:read_payments"
TOKENS = "memory.text.parse:tokens"
WORDS = "memory.text.parse:words"
NOTE = "notes/orders/totals.md"
#: Every function of the library, in module-path order (the catalogue's order).
FUNCTIONS = [RECONCILE, READ_LEDGER, READ_PAYMENTS, TOTAL_RECIPE, TOTAL, READ_TOTAL, COUNT, READ_ORDERS, TOKENS, WORDS]

#: The order table of the landed episode: its amounts add up to the answer it handed back.
ROWS = [{"item": "pen", "amount": int(ANSWER) - 4}, {"item": "ink", "amount": 4}]
LEDGER_ROWS = [{"ref": "a", "amount": 1}, {"ref": "b", "amount": 2}]
PAYMENTS = [{"ref": "a"}]
UNMATCHED = [{"ref": "b", "amount": 2}]
ORDERS_REQUEST = f"{RAW_REQUEST}\nTABLE\n{json.dumps(ROWS)}\n"
LEDGER_REQUEST = (
    f"Which ledger rows have no payment?\nLEDGER\n{json.dumps(LEDGER_ROWS)}\nPAYMENTS\n{json.dumps(PAYMENTS)}\n"
)


def _handed_back(eid: str, request: str, value) -> object:
    """A recorded episode: a text request, answered by one dialogue action that handed *value* back."""
    return _ep(
        episode_id=eid,
        request=[request],
        transcript=[{"role": "user", "content": request}],
        cells=[],
        actions=[Action(-1, "dialogue:user", "answer", [value], {}, "Accepted.", "ok", "unknown", kind="dialogue")],
    )


EPISODES = {
    "ep-orders-a": _handed_back("ep-orders-a", ORDERS_REQUEST, int(ANSWER)),
    "ep-ledger-a": _handed_back("ep-ledger-a", LEDGER_REQUEST, UNMATCHED),
}

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
#: The library stored before the pass.
BASE_FILES = {
    "memory/text/parse.py": PARSE,
    NOTE: NOTE_TEXT,
    "memory/orders/__init__.py": '"""Order tables: reading them, counting and totalling their rows."""\n',
    "memory/billing/__init__.py": '"""Ledgers and the payments that settle them."""\n',
}

#: The procedure the episode-end fork staged for ``ep-orders-a``, which the pass lands as :data:`TOTAL`.
TOTAL_PROCEDURE = f'''"""Total the amount column of an order table.

Kept from an attempt that was accepted.
"""

import json

SEEN = {RAW_REQUEST!r}


def read_input(text):
    """The order table that follows the line TABLE in a request."""
    return json.loads(text.split("TABLE\\n", 1)[1].splitlines()[0])


def applies(value):
    """A list of order rows, each with a whole amount."""
    return isinstance(value, list) and all(isinstance(r, dict) and isinstance(r.get("amount"), int) for r in value)


def run(value):
    """Sum the amount column of an order table.
    Each row is a record keyed by column name.

    The second paragraph says more: SECOND-PARAGRAPH-MARK.

    Notes: {NOTE}
    """
    return sum(r["amount"] for r in value)
'''
TABLES = '''"""Order tables read from request texts."""

import json


class MemoryInputError(ValueError):
    pass


def read_orders(text: str) -> list:
    """Read the order table that follows the line TABLE in a request.

    Example: memory/orders/tests/test_tables.py::test_reads_the_recorded_order_table
    """
    if not isinstance(text, str) or "TABLE\\n" not in text:
        raise MemoryInputError("a request with a line TABLE before its order table is needed")
    return json.loads(text.split("TABLE\\n", 1)[1].splitlines()[0])
'''
SUMS = f'''"""Counts over order tables."""

SAMPLE_REQUEST = {RAW_REQUEST!r}


class MemoryInputError(ValueError):
    pass


def count_rows(rows: list) -> int:
    """Count the rows of a table.

    Example: memory/orders/tests/test_sums.py::test_counts_the_recorded_rows
    """
    if not isinstance(rows, list):
        raise MemoryInputError("a table as a list of rows is needed")
    return len(rows)


def _helper(x):
    return x
'''
LEDGER = f'''"""Ledger work."""


class MemoryInputError(ValueError):
    pass


def reconcile(ledger: list, payments: list) -> list:
    """Match ledger rows against payments, as for case {UUID}.

    Returns the unmatched rows.

    Example: memory/billing/tests/test_ledger.py::test_keeps_the_recorded_rows_no_payment_matches
    """
    if not isinstance(ledger, list) or not isinstance(payments, list):
        raise MemoryInputError("a ledger and its payments, each a list of rows, are needed")
    paid = {{p["ref"] for p in payments}}
    return [r for r in ledger if r["ref"] not in paid]
'''
BILLING_READERS = '''"""Readers of ledger requests."""

import json


class MemoryInputError(ValueError):
    pass


def _rows_after(text, line):
    if not isinstance(text, str) or line + "\\n" not in text:
        raise MemoryInputError(f"a request with a line {line} before its rows is needed")
    return json.loads(text.split(line + "\\n", 1)[1].splitlines()[0])


def read_ledger(text: str) -> list:
    """The ledger rows that follow the line LEDGER in a request.

    Example: memory/billing/tests/test_readers.py::test_reads_the_recorded_ledger
    """
    return _rows_after(text, "LEDGER")


def read_payments(text: str) -> list:
    """The payments that follow the line PAYMENTS in a request.

    Example: memory/billing/tests/test_readers.py::test_reads_the_recorded_payments
    """
    return _rows_after(text, "PAYMENTS")
'''


def _test_module(imports: str, body: str) -> str:
    """A test module of the writer's: *imports* (the library's), the data helpers, then *body*."""
    return (
        "import json\nfrom pathlib import Path\n\nimport pytest\n\n"
        + imports
        + '\nDATA = Path(__file__).parent / "data"\n\n\n'
        + "def load(name):\n    return json.loads((DATA / name).read_text(encoding=\"utf-8\"))\n\n\n"
        + "def text(name):\n    return (DATA / name).read_text(encoding=\"utf-8\")\n"
        + body
    )


#: The reader's test calls it on the recorded text before any parsed fixture is opened: it has no reader itself.
TEST_TABLES = _test_module(
    "from memory.orders.tables import MemoryInputError, read_orders\n",
    f'''

def test_reads_the_recorded_order_table():
    assert read_orders(text("orders.txt")) == {ROWS!r}


def test_refuses_text_without_a_table():
    with pytest.raises(MemoryInputError):
        read_orders("no table here")
''',
)
#: Both functions on the order table :data:`READ_ORDERS` parsed: its one reader for :data:`COUNT`, a second one for
#: the landed :data:`TOTAL`.
TEST_SUMS = _test_module(
    "from memory.orders.procedures import total_amount\nfrom memory.orders.sums import MemoryInputError, count_rows\n",
    '''

def test_counts_the_recorded_rows():
    assert count_rows(load("orders.json")) == 2


def test_totals_the_recorded_rows_as_the_episode_did():
    assert total_amount(load("orders.json")) == load("total_amount.case1.result.json")


def test_refuses_what_is_not_a_table():
    with pytest.raises(MemoryInputError):
        count_rows("not a table")
''',
)
TEST_BILLING_READERS = _test_module(
    "from memory.billing.readers import MemoryInputError, read_ledger, read_payments\n",
    f'''

def test_reads_the_recorded_ledger():
    assert read_ledger(text("ledger.txt")) == {LEDGER_ROWS!r}


def test_reads_the_recorded_payments():
    assert read_payments(text("ledger.txt")) == {PAYMENTS!r}


def test_refuses_text_without_a_ledger():
    with pytest.raises(MemoryInputError):
        read_ledger("no ledger here")


def test_refuses_text_without_payments():
    with pytest.raises(MemoryInputError):
        read_payments("no payments here")
''',
)
#: Two parsed fixtures, of two readers, in one call: :data:`RECONCILE`'s two readers.
TEST_LEDGER = _test_module(
    "from memory.billing.ledger import MemoryInputError, reconcile\n",
    f'''

def test_keeps_the_recorded_rows_no_payment_matches():
    assert reconcile(load("ledger.json"), load("payments.json")) == {UNMATCHED!r}


def test_refuses_what_is_not_a_ledger():
    with pytest.raises(MemoryInputError):
        reconcile("no ledger", [])
''',
)
#: The modules the writer writes in the pass, beside the landing.
WRITER_MODULES = {
    "memory/billing/ledger.py": LEDGER,
    "memory/orders/sums.py": SUMS,
    "memory/billing/readers.py": BILLING_READERS,
    "memory/orders/tables.py": TABLES,
}
#: The library as written by hand: the stored library and the writer's modules, without the landed procedure (for
#: tests that need no item records). Written in an order that is not the module-path order, so an order taken from
#: the filesystem or from insertion is caught.
WRITTEN_FILES = {
    "memory/text/parse.py": PARSE,
    **WRITER_MODULES,
    **{k: v for k, v in BASE_FILES.items() if k != "memory/text/parse.py"},
}
#: The writer's files in the pass: its modules and their tests.
WRITER_FILES = {
    **WRITER_MODULES,
    "memory/orders/tests/test_tables.py": TEST_TABLES,
    "memory/orders/tests/test_sums.py": TEST_SUMS,
    "memory/billing/tests/test_readers.py": TEST_BILLING_READERS,
    "memory/billing/tests/test_ledger.py": TEST_LEDGER,
}


def _fixture(episode: str, dest: str, parser: str | None = None) -> dict:
    """The fixture tool's arguments: the episode's request text into *dest*, through *parser* when given."""
    return {"episode": episode, "source": "request:0", "dest": dest, **({"parser": parser} if parser else {})}


WRITER_FIXTURES = [
    _fixture("ep-orders-a", "memory/orders/tests/data/orders.txt"),
    _fixture("ep-orders-a", "memory/orders/tests/data/orders.json", READ_ORDERS),
    _fixture("ep-ledger-a", "memory/billing/tests/data/ledger.txt"),
    _fixture("ep-ledger-a", "memory/billing/tests/data/ledger.json", READ_LEDGER),
    _fixture("ep-ledger-a", "memory/billing/tests/data/payments.json", READ_PAYMENTS),
]


def _entry(item: str, episode: str, test: str, **more) -> dict:
    return {"item": item, "kind": "function", "source_episodes": [episode], "tests": [test], "covers": [], **more}


WRITER_ITEMS = [
    _entry(READ_ORDERS, "ep-orders-a", "memory/orders/tests/test_tables.py", role="reader"),
    _entry(COUNT, "ep-orders-a", "memory/orders/tests/test_sums.py"),
    # the landed procedure's entry gains the writer's test of it
    {"item": TOTAL, "tests": ["memory/orders/tests/test_sums.py"]},
    _entry(READ_LEDGER, "ep-ledger-a", "memory/billing/tests/test_readers.py", role="reader"),
    _entry(READ_PAYMENTS, "ep-ledger-a", "memory/billing/tests/test_readers.py", role="reader"),
    _entry(RECONCILE, "ep-ledger-a", "memory/billing/tests/test_ledger.py"),
]


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
    """An applicability signature as the capture store writes it (design I.2.5), for the captured procedures'
    snapshot."""
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


@functools.lru_cache(maxsize=1)
def built() -> Built:
    """The world, built once per process (module docstring): the capture of ``ep-orders-a`` landed as
    :data:`TOTAL`, the writer's files, fixtures and manifest entries, the gate's check and the item records."""
    root = Path(tempfile.mkdtemp(prefix="librarian-world-"))
    atexit.register(shutil.rmtree, root, True)
    return build(
        root,
        episodes=EPISODES,
        base=BASE_FILES,
        landings=[Landing("ep-orders-a", TOTAL_PROCEDURE, SUMMARIES["ep-orders-a"], "orders", "total_amount")],
        files=WRITER_FILES,
        fixtures=WRITER_FIXTURES,
        items=WRITER_ITEMS,
    )


def files() -> dict[str, str]:
    """The library the gate checked, by path: the stored library, the landing and the writer's files, with their
    tests and fixtures."""
    return dict(built().files)


def records() -> dict[str, dict]:
    """The item records of the checked library, as ``lifecycle.next_records`` writes them, with :data:`TOTAL`
    adopted after three accepted episodes. A fresh copy each call."""
    out = copy.deepcopy(built().records)
    out[TOTAL]["status"] = "adopted"
    out[TOTAL]["use"] = {"1" * 40: {"episodes": 3, "uses": 5, "seen": 5, "errors": 1, "refused": 0}}
    return out


def landed_capture() -> str:
    """The id of the capture the pass landed as :data:`TOTAL`."""
    return built().captures["ep-orders-a"]


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


def write_tree(root: Path, tree: dict | None = None) -> Path:
    """Write the library *tree* (path -> text; None: the library the gate checked, :func:`files`) under *root*."""
    for rel, text in (files() if tree is None else tree).items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    return root


def big_files(n: int = 60) -> dict[str, str]:
    """A larger library: *n* modules of two functions each, beside :data:`WRITTEN_FILES`."""
    out = dict(WRITTEN_FILES)
    for i in range(n):
        out[f"memory/bulk/part_{chr(97 + i // 26)}{chr(97 + i % 26)}.py"] = (
            f'def first(x):\n    """Return x plus {i}."""\n    return x + {i}\n\n\n'
            f'def second(x):\n    """Return x times {i}."""\n    return x * {i}\n'
        )
    return out

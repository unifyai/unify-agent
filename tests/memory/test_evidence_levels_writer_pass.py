"""Evidence levels end to end through a writer pass (design I.3, with the decisions of I.11).

One scripted writer pass, tools called from code, in real boxes, over recorded episodes of two forms:

* **structured**: ``e1`` and ``e2`` each recorded an ``acct.me`` response (a JSON object). The function
  ``memory.acct.users:user_id`` is tested on whole copies of them (``tests/memory/library_example_world.py``).
* **text**: ``e4`` and ``e5`` each recorded a request whose record is written as ``key: value`` lines. A stored reader,
  ``memory.fields.readers:read_fields(text)``, reads such text into a mapping; the fixtures of
  ``memory.fields.accounts:account_name(record)`` are made with ``tools.fixture(..., parser=<the reader's id>)``,
  so the gate runs the reader on the recorded bytes again to check them.

The gate treats the two alike (I.11 decision 4): both functions land, at the same evidence level. A function whose
test parses the recorded text inline (no stored reader) also lands, at a lower level, and the finding that capped it
is in its level reasons: evidence findings set a level, they do not refuse.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from tests.memory import library_example_world as w
from tests.memory.test_episodes import _ep
from tests.memory.test_tool_channel_box import needs_box
from tests.memory.test_writer_pass import _call
from tests.memory.writer_sections import README_TEXT, as_written
from unify.memory import fixtures as fx
from unify.memory.blobs import BlobStore
from unify.memory.evidence import EvidenceStore
from unify.memory.gate import Gate
from unify.memory.gitio import Repo
from unify.memory.library_checks import LibraryConfig
from unify.memory.trigger import PassRequest
from unify.memory.writer_pass import PassConfig, WriterPass

# ── assumed names: the integrator adapts these to the build in this one block ─────────────────────────────────
#: Design I.11 decision 3: the evidence levels, lowest first. I.3.1's ``experimental`` is ``unchecked`` (I.11: it
#: means the status ``on-trial`` at the level ``unchecked``) and its ``verified`` is ``reproduces-its-case``; the
#: gate gives only those two, and ``confirmed-by-outcomes`` comes from outcomes after use, never from the gate.
from unify.memory.item_states import LEVELS, REPRODUCES_ITS_CASE, UNCHECKED  # noqa: E402

LEVEL_LOW, LEVEL_TOP = UNCHECKED, REPRODUCES_ITS_CASE
#: Design I.3.3: the ``fixture`` tool's argument naming the stored reader.
PARSER_ARG = "parser"
#: The evidence finding a value computed from recorded text by the test itself is capped for (I.3.2).
GROUNDED_CODE = "[examples:grounded]"


def gate_level(result, item: str) -> str:
    """Design I.3.2: the gate result's per-item evidence level."""
    return result.levels[item]


def gate_level_reasons(result, item: str) -> list:
    return list(result.level_reasons.get(item, []))


# ── end of assumed names ──────────────────────────────────────────────────────────────────────────────────────

READER_ID = "memory.fields.readers:read_fields"
ACCOUNT_ID = "memory.fields.accounts:account_name"
READERS = "memory/fields/readers.py"
ACCOUNTS = "memory/fields/accounts.py"
READER_TEST = "memory/fields/tests/test_readers.py"
ACCOUNT_TEST = "memory/fields/tests/test_accounts.py"
DATA = "memory/fields/tests/data/"
ACCOUNT_INPUTS = fx.inputs_file(ACCOUNT_ID)


def _text_episode(eid: str, user: str, name: str):
    text = f"Account record\nuser_id: {user}\nname: {name}\n"
    return _ep(
        episode_id=eid,
        request=[text],
        transcript=[{"role": "user", "content": text}],
        cells=[],
        actions=[],
    )


E4 = _text_episode("e4", "u-4", "Di")
E5 = _text_episode("e5", "u-5", "Ed")
EPISODES = {**w.EPISODES, "e4": E4, "e5": E5}

PACKAGE_INIT = '"""Records written as text."""\n'
READERS_SOURCE = f'''"""Readers of records written as text."""


class MemoryInputError(ValueError):
    pass


def read_fields(text: str) -> dict:
    """The key: value lines of a text as a mapping; other lines are skipped.

    Effect: read
    Example: {READER_TEST}::test_reads_the_recorded_fields
    """
    if not isinstance(text, str):
        raise MemoryInputError("a text is needed")
    out = {{}}
    for line in text.splitlines():
        key, sep, value = line.partition(": ")
        if sep and key.strip() and " " not in key.strip():
            out[key.strip()] = value.strip()
    if not out:
        raise MemoryInputError("the text has no key: value line")
    return out
'''
ACCOUNTS_SOURCE = f'''"""Accounts read from records."""

from memory.fields.readers import MemoryInputError


def account_name(record: dict) -> str:
    """The account's name, read from a record.

    Effect: read
    Example: {ACCOUNT_TEST}::test_reads_the_recorded_names
    """
    if not isinstance(record, dict) or "name" not in record:
        raise MemoryInputError("a record with a name field is needed")
    return record["name"]
'''
READER_TEST_SOURCE = f'''from pathlib import Path

import pytest

from memory.fields.readers import MemoryInputError, read_fields

DATA = Path(__file__).parent / "data"


def test_reads_the_recorded_fields():
    assert read_fields((DATA / "request_4.txt").read_text()) == {{"user_id": "u-4", "name": "Di"}}
    assert read_fields((DATA / "request_5.txt").read_text()) == {{"user_id": "u-5", "name": "Ed"}}


def test_refuses_a_text_without_fields():
    with pytest.raises(MemoryInputError):
        read_fields("no fields here")
'''
ACCOUNT_TEST_PARSED = '''import json
from pathlib import Path

import pytest

from memory.fields.accounts import account_name
from memory.fields.readers import MemoryInputError

DATA = Path(__file__).parent / "data"
INPUTS = [json.loads(x) for x in (DATA / "accounts.account_name.inputs.jsonl").read_text().splitlines() if x]


def load(name):
    return json.loads((DATA / name).read_text())


def test_reads_the_recorded_names():
    assert account_name(load("record_4.json")) == "Di"
    assert account_name(load("record_5.json")) == "Ed"


@pytest.mark.parametrize("row", INPUTS)
def test_every_recorded_input_has_a_name(row):
    assert account_name(row["input"]) == row["input"]["name"]


def test_refuses_a_record_without_a_name():
    with pytest.raises(MemoryInputError):
        account_name({"user_id": "u-0"})
'''
#: The same function, its test reading the recorded text and parsing it inline (no stored reader).
ACCOUNT_TEST_INLINE = '''from pathlib import Path

import pytest

from memory.fields.accounts import account_name
from memory.fields.readers import MemoryInputError

DATA = Path(__file__).parent / "data"


def parse(text):
    out = {}
    for line in text.splitlines():
        key, sep, value = line.partition(": ")
        if sep:
            out[key.strip()] = value.strip()
    return out


def test_reads_the_recorded_names():
    assert account_name(parse((DATA / "request_4.txt").read_text())) == "Di"
    assert account_name(parse((DATA / "request_5.txt").read_text())) == "Ed"


def test_refuses_a_record_without_a_name():
    with pytest.raises(MemoryInputError):
        account_name({"user_id": "u-0"})
'''

ITEM_READER = {
    "item": READER_ID,
    "kind": "function",
    "source_episodes": ["e4", "e5"],
    "tests": [READER_TEST],
    "covers": [],
}
ITEM_ACCOUNT = {**ITEM_READER, "item": ACCOUNT_ID, "tests": [ACCOUNT_TEST]}


def _write(rel: str, text: str) -> str:
    # as a writer writes it: each function and note with its "when to use" section
    text = as_written({rel: text})[rel]
    return (
        f"_p = pathlib.Path('/memory') / {rel!r}\n"
        "_p.parent.mkdir(parents=True, exist_ok=True)\n"
        f"_p.write_text({text!r})\n"
    )


def _fixture(**args) -> str:
    return f"print('FIXTURE', tools.fixture(**{args!r})[:300].replace(chr(10), ' '))\n"


def _structured_cell() -> str:
    """The structured item of the shared world: its module, tests, whole copies of e1's and e2's responses and its
    recorded-inputs file, all made through the fixture tool."""
    out = ["import pathlib\n"]
    out += [_write(w.PACKAGE_INIT, '"""Accounts."""\n'), _write(w.MODULE, w.MODULE_SOURCE), _write(w.TEST_FILE, w.TEST_SOURCE)]
    out.append(_write(w.README, README_TEXT))
    for path, eid in zip(w.FIXTURE_PATHS, ("e1", "e2")):
        out.append(_fixture(episode=eid, dest=path, source="0"))
    for eid in ("e1", "e2"):
        out.append(_fixture(episode=eid, dest=w.INPUTS_PATH, source="0", append=True))
    return "".join(out)


def _text_cell(*, parsed: bool) -> str:
    """The reader and the account function, over e4's and e5's recorded text: through the reader (*parsed*), or
    with a test that parses the text inline."""
    out = [
        "import pathlib\n",
        _write("memory/fields/__init__.py", PACKAGE_INIT),
        _write("memory/fields/README.md", README_TEXT),
        _write(READERS, READERS_SOURCE),
        _write(ACCOUNTS, ACCOUNTS_SOURCE),
        _write(READER_TEST, READER_TEST_SOURCE),
        _write(ACCOUNT_TEST, ACCOUNT_TEST_PARSED if parsed else ACCOUNT_TEST_INLINE),
    ]
    for eid, n in (("e4", 4), ("e5", 5)):
        out.append(_fixture(episode=eid, dest=f"{DATA}request_{n}.txt", source="request:0"))
    if parsed:
        for eid, n in (("e4", 4), ("e5", 5)):
            out.append(_fixture(episode=eid, dest=f"{DATA}record_{n}.json", source="request:0", **{PARSER_ARG: READER_ID}))
            out.append(
                _fixture(episode=eid, dest=ACCOUNT_INPUTS, source="request:0", append=True, **{PARSER_ARG: READER_ID}),
            )
    return "".join(out)


def _manifest_cell(items: list[dict]) -> str:
    manifest = {"items": items, "deleted": [], "unlisted": [], "support": [], "summary": "stored the account readers"}
    return "import pathlib\n" + _write(".pass/manifest.json", json.dumps(manifest)) + "print('CHECK', tools.check()[:4000])\n"


FINISH = "tools.finish('stored the account functions')\nprint('NOT REACHED')\n"


class _Cells:
    def __init__(self, cells: list[str]) -> None:
        self.cells, self.outputs = list(cells), {}

    async def __call__(self, messages, tools):
        for m in messages:
            if m.get("role") == "tool":
                self.outputs[m["tool_call_id"]] = str(m.get("content"))
        n = len(self.outputs) + 1
        return _call(f"c{n}", "execute_code", {"code": self.cells.pop(0) if self.cells else FINISH}), "0.001"


def _pass(tmp_path: Path, cells: list[str], eids: list[str]):
    mem = Repo.init_bare(tmp_path / "mem.git")
    ev = EvidenceStore(tmp_path / "e.sqlite")
    for n, ep in enumerate(EPISODES.values()):
        ev.index_episode(ep, str(n + 1) * 40)
    gate = Gate(
        mem,
        ev,
        BlobStore(tmp_path / "b"),
        action_lookup=lambda eid, i: (
            EPISODES[eid].actions[i] if eid in EPISODES and 0 <= i < len(EPISODES[eid].actions) else None
        ),
        library_mode=LibraryConfig(episodes=EPISODES.get),
    )
    results: list = []
    merge = gate.merge

    def spy(*args, **kwargs):
        res = merge(*args, **kwargs)
        results.append(res)
        return res

    gate.merge = spy
    model = _Cells(cells)
    writer = WriterPass(
        mem,
        gate,
        ev,
        load=EPISODES.__getitem__,
        model_turn=model,
        config=PassConfig(run_home=str(tmp_path / "run-home"), 
            pass_record_root=str(tmp_path / "out"),
            # each fixture and check call made from code counts against the cap as a model call does: the
            # reader script alone makes 15 or more of them
            max_calls=24,
            cell_timeout_s=300.0,
            deadline_s=1500.0,
        ),
    )
    out = asyncio.run(writer.run(PassRequest("incremental", "acct", eids, False), "p1"))
    return out, model, results


def _why(model: _Cells) -> str:
    return "\n".join(f"{k}: {v[:2500]}" for k, v in sorted(model.outputs.items()))


@needs_box
@pytest.mark.timeout(1200)
def test_a_structured_fixture_lands_at_the_top_level_through_a_writer_pass(tmp_path, monkeypatch):
    cells = [_structured_cell(), _manifest_cell([w.item()]), FINISH]
    out, model, results = _pass(tmp_path, cells, ["e1", "e2"])
    assert out.finished and out.passed, (out.reasons, _why(model))
    assert w.ITEM_ID in out.items_merged, (out.items_merged, out.items_refused)
    assert gate_level(results[-1], w.ITEM_ID) == LEVEL_TOP, gate_level_reasons(results[-1], w.ITEM_ID)


@needs_box
@pytest.mark.timeout(1200)
def test_text_grounded_through_a_stored_reader_lands_alike_with_a_structured_fixture(tmp_path, monkeypatch):
    cells = [
        _structured_cell(),
        _text_cell(parsed=True),
        _manifest_cell([w.item(), ITEM_READER, ITEM_ACCOUNT]),
        FINISH,
    ]
    out, model, results = _pass(tmp_path, cells, ["e1", "e2", "e4", "e5"])
    shown = _why(model)
    text_cell = model.outputs.get("c2", "")
    assert "Traceback" not in text_cell and "Refused" not in text_cell, text_cell[-3000:]
    assert out.finished and out.passed, (out.reasons, shown)
    assert {w.ITEM_ID, READER_ID, ACCOUNT_ID} <= set(out.items_merged), (out.items_merged, out.items_refused)
    final = results[-1]
    structured, through_reader = gate_level(final, w.ITEM_ID), gate_level(final, ACCOUNT_ID)
    # structured and unstructured inputs treated alike: the same level, the top one
    assert through_reader == structured == LEVEL_TOP, (
        gate_level_reasons(final, ACCOUNT_ID),
        gate_level_reasons(final, w.ITEM_ID),
    )
    # the reader lands with them; its own level rests on its own tests (an exact assertion of its result), which
    # this script does not write, so any level is accepted here
    assert gate_level(final, READER_ID) in LEVELS
    # the parsed fixtures carry the reader that made them, so the gate can make them again
    verified = getattr(final, "fixtures_verified", {}) or {}
    entries = [entry for path, (_, entry) in verified.items() if path.startswith(DATA + "record_")]
    assert entries and all(e.get("kind") == "parsed" and e.get("parser") == READER_ID for e in entries), verified


@needs_box
@pytest.mark.timeout(1200)
def test_text_parsed_inline_by_the_test_lands_at_a_lower_level_with_its_finding(tmp_path, monkeypatch):
    cells = [_text_cell(parsed=False), _manifest_cell([ITEM_READER, ITEM_ACCOUNT]), FINISH]
    out, model, results = _pass(tmp_path, cells, ["e4", "e5"])
    assert out.finished and out.passed, (out.reasons, _why(model))
    assert {READER_ID, ACCOUNT_ID} <= set(out.items_merged), (out.items_merged, out.items_refused)
    final = results[-1]
    assert gate_level(final, ACCOUNT_ID) == LEVEL_LOW
    assert any(GROUNDED_CODE in str(r) for r in gate_level_reasons(final, ACCOUNT_ID)), gate_level_reasons(
        final,
        ACCOUNT_ID,
    )
    # a level, not a refusal: no refusal names the function
    # (the gate's ``note:`` lines report each item's evidence level, which names it; a refusal never does here)
    assert ACCOUNT_ID not in out.items_refused, out.items_refused
    assert not any(ACCOUNT_ID in r for r in out.reasons if not r.startswith("note:")), out.reasons

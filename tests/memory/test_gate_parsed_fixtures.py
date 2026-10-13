"""Evidence in any recorded form: evidence findings set an item's evidence level, and only integrity and safety
checks refuse; text reaches a function through a stored reader (a parsed fixture), which the gate re-derives.

The pure parts (parsed fixtures with a stand-in reader, the fixture tool) need no box; the gate's cases run in the
jail (bubblewrap) and are skipped without it.
"""

from __future__ import annotations

import hashlib
import json
import shutil

import pytest

from tests.memory import library_example_world as lw
from tests.memory.library_example_world import library_example_world  # noqa: F401 (fixture)
from tests.memory.test_episodes import _ep
from unify.memory import fixtures as fx
from unify.memory import writer_tools
from unify.memory.blobs import BlobStore
from unify.memory.evidence import EvidenceStore
from unify.memory.gate import Gate
from unify.memory.gitio import Repo
from unify.memory.item_states import REPRODUCES_ITS_CASE, UNCHECKED
from unify.memory.library_checks import LibraryConfig

needs_bwrap = pytest.mark.skipif(shutil.which("bwrap") is None, reason="bubblewrap required")

#: a recorded request that is text: `key: value` lines
REQUEST = "user: u-7\nname: Gil\n"
T1 = _ep(episode_id="t1", request=[REQUEST], actions=[], cells=[])
FORM = {"name": "Gil", "user": "u-7"}

READER = "memory.forms.read:fields"
USER_OF = "memory.forms.users:user_of"
READ_MODULE = "memory/forms/read.py"
USERS_MODULE = "memory/forms/users.py"
READ_TEST = "memory/forms/tests/test_read.py"
USERS_TEST = "memory/forms/tests/test_users.py"
TEXT_FIXTURE = "memory/forms/tests/data/request.txt"
PARSED_FIXTURE = "memory/forms/tests/data/request.json"

READ_SOURCE = '''"""Readers of recorded text."""


class MemoryInputError(ValueError):
    pass


def fields(text: str) -> dict:
    """The `key: value` lines of a text block, as a mapping.

    Example: memory/forms/tests/test_read.py::test_reads_the_recorded_fields
    """
    if not isinstance(text, str):
        raise MemoryInputError("text with key: value lines is needed")
    out = {}
    for line in text.splitlines():
        key, sep, value = line.partition(":")
        if sep and key.strip():
            out[key.strip()] = value.strip()
    if not out:
        raise MemoryInputError("text with key: value lines is needed")
    return out
'''
USERS_SOURCE = '''"""Users named in recorded forms."""


class MemoryInputError(ValueError):
    pass


def user_of(form: dict) -> str:
    """The user a form names.

    Example: memory/forms/tests/test_users.py::test_reads_the_user_of_the_request
    """
    if not isinstance(form, dict) or "user" not in form:
        raise MemoryInputError("a form with a user field is needed")
    return form["user"]
'''
READ_TEST_SOURCE = '''from pathlib import Path

import pytest

from memory.forms.read import MemoryInputError, fields

DATA = Path(__file__).parent / "data"


def test_reads_the_recorded_fields():
    assert fields((DATA / "request.txt").read_text()) == {"user": "u-7", "name": "Gil"}


def test_refuses_text_without_fields():
    with pytest.raises(MemoryInputError):
        fields("no fields here")
'''
_USERS_HEAD = '''import json
from pathlib import Path

import pytest

from memory.forms.users import MemoryInputError, user_of

DATA = Path(__file__).parent / "data"


def test_refuses_a_form_without_a_user():
    with pytest.raises(MemoryInputError):
        user_of({"name": "Gil"})
'''
#: the test parses the recorded text itself: the argument is computed from text, which the gate cannot perturb
USERS_TEST_INLINE = _USERS_HEAD + '''

def test_reads_the_user_of_the_request():
    text = (DATA / "request.txt").read_text()
    form = dict(line.split(": ", 1) for line in text.splitlines() if ": " in line)
    assert user_of(form) == "u-7"
'''
#: the test loads the fixture the stored reader made from the same recorded text
USERS_TEST_PARSED = _USERS_HEAD + '''

def test_reads_the_user_of_the_request():
    assert user_of(json.loads((DATA / "request.json").read_text())) == "u-7"
'''


def _parse_like_the_reader(reader, text):
    """A stand-in for the confined reader run: what ``fields`` returns, as canonical JSON, and a digest."""
    assert reader == READER
    out = {}
    for line in text.splitlines():
        key, sep, value = line.partition(":")
        if sep and key.strip():
            out[key.strip()] = value.strip()
    return fx.canonical(out), "d" * 64


# --- parsed fixtures, without a box -----------------------------------------------------------------------------


def test_a_parsed_fixture_is_made_and_verified_with_its_reader():
    data, entry = fx.make(
        T1,
        "request:0",
        PARSED_FIXTURE,
        blob=lambda s: b"",
        parser=READER,
        parse=_parse_like_the_reader,
    )
    assert json.loads(data) == FORM
    assert entry["kind"] == "parsed" and entry["parser"] == READER and entry["parser_sha256"] == "d" * 64
    assert entry["episode"] == "t1" and entry["source"] == "request:0" and entry["trust"] == 1
    assert entry["sha256"] == hashlib.sha256(data).hexdigest()
    load = {"t1": T1}.get
    assert fx.verify(PARSED_FIXTURE, data, entry, load, lambda s: b"", parse=_parse_like_the_reader) is None
    # one changed byte is refused, with what to do
    bad = data.replace(b"u-7", b"u-8")
    why = fx.verify(PARSED_FIXTURE, bad, entry, load, lambda s: b"", parse=_parse_like_the_reader)
    assert why is not None and READER in why and fx.REMAKE in why
    # without a reader run, it does not verify
    assert fx.verify(PARSED_FIXTURE, data, entry, load, lambda s: b"") is not None
    # its views are the JSON's: the mapping a function takes is grounded as a copy
    assert fx.value_views(PARSED_FIXTURE, data, entry)
    assert fx.readers_of(entry) == [READER]


def test_a_parsed_input_line_names_its_reader_and_verifies():
    dest = "memory/forms/tests/data/users.user_of.inputs.jsonl"
    line, entry = fx.make(
        T1,
        "request:0",
        dest,
        blob=lambda s: b"",
        append=True,
        parser=READER,
        parse=_parse_like_the_reader,
    )
    got = json.loads(line)
    assert got["input"] == FORM and got["source"]["parser"] == READER
    assert fx.assembled_entry(line)["lines"][0]["parser"] == READER
    load = {"t1": T1}.get
    assert fx.verify(dest, line, entry, load, lambda s: b"", parse=_parse_like_the_reader) is None
    assert fx.readers_of(entry) == [READER]


def test_a_parsed_fixture_must_be_json():
    with pytest.raises(fx.FixtureError, match="must end in .json"):
        fx.make(
            T1,
            "request:0",
            "memory/forms/tests/data/request.out",
            blob=lambda s: b"",
            parser=READER,
            parse=_parse_like_the_reader,
        )


def test_the_fixture_tool_takes_a_parser_and_lists_the_parsed_fixture(tmp_path):
    box = tmp_path / "box"
    box.mkdir()
    reply = writer_tools.fixture(
        {"episode": "t1", "dest": PARSED_FIXTURE, "source": "request:0", "parser": READER},
        {"t1": T1}.get,
        box,
        lambda s: b"",
        parse=_parse_like_the_reader,
    )
    assert reply.status == "ok", reply.text
    assert json.loads((box / PARSED_FIXTURE).read_text()) == FORM
    listed = fx.read_listing(box)[PARSED_FIXTURE]
    assert listed["kind"] == "parsed" and listed["parser"] == READER
    derived = fx.derive_ledger(box)[PARSED_FIXTURE]
    assert derived["parser"] == READER and derived["parser_sha256"] == "d" * 64


# --- the gate ---------------------------------------------------------------------------------------------------


@pytest.fixture
def text_world(tmp_path):
    """An empty library and an evidence store holding the text episode t1, with a library-mode gate factory."""
    mem = Repo.init_bare(tmp_path / "mem.git")
    ev = EvidenceStore(tmp_path / "e.sqlite")
    ev.index_episode(T1, "7" * 40)
    blobs = BlobStore(tmp_path / "b")
    episodes = {"t1": T1}

    def lookup(eid, i):
        return None

    def gate(**cfg):
        return Gate(
            mem,
            ev,
            blobs,
            action_lookup=lookup,
            library_mode=LibraryConfig(checks=True, episodes=episodes.get, **cfg),
        )

    return mem, ev, blobs, gate


def _items(users_tests):
    return [
        {"item": READER, "kind": "function", "source_episodes": ["t1"], "tests": [READ_TEST], "covers": []},
        {"item": USER_OF, "kind": "function", "source_episodes": ["t1"], "tests": users_tests, "covers": []},
    ]


def _text_files(blobs, users_test, parsed_bytes=None):
    """The candidate's files and fixture entries; with *parsed_bytes* (b"": the reader's own result) also the
    parsed fixture."""
    text, text_entry = fx.make(T1, "request:0", TEXT_FIXTURE, blob=blobs.get)
    files = {
        "memory/forms/__init__.py": '"""Recorded forms."""\n',
        READ_MODULE: READ_SOURCE,
        USERS_MODULE: USERS_SOURCE,
        READ_TEST: READ_TEST_SOURCE,
        USERS_TEST: users_test,
        TEXT_FIXTURE: text.decode(),
    }
    entries = {TEXT_FIXTURE: text_entry}
    if parsed_bytes is not None:
        data, entry = fx.make(
            T1,
            "request:0",
            PARSED_FIXTURE,
            blob=blobs.get,
            parser=READER,
            parse=_parse_like_the_reader,
        )
        files[PARSED_FIXTURE] = (parsed_bytes or data).decode()
        entries[PARSED_FIXTURE] = entry
    return files, entries


def _manifest(entries, items):
    return {"items": items, "deleted": [], "unlisted": [], "support": [], "fixtures": entries}


@needs_bwrap
def test_text_parsed_inline_lands_at_the_lowest_level(text_world):
    mem, ev, blobs, gate = text_world
    files, entries = _text_files(blobs, USERS_TEST_INLINE)
    res = gate().check(mem.head(), lw.candidate(mem, files), _manifest(entries, _items([USERS_TEST])))
    # the call returns on a value the test computed from recorded text: stored, at the lowest evidence level
    assert res.passed, res.reasons
    assert res.levels[USER_OF] == UNCHECKED
    assert any("[examples:grounded]" in r for r in res.level_reasons[USER_OF]), res.level_reasons
    assert res.verification[USER_OF]["level"] == UNCHECKED


@needs_bwrap
def test_text_grounded_through_a_stored_reader_reaches_the_grounding(text_world):
    mem, ev, blobs, gate = text_world
    files, entries = _text_files(blobs, USERS_TEST_PARSED, parsed_bytes=b"")
    res = gate().check(mem.head(), lw.candidate(mem, files), _manifest(entries, _items([USERS_TEST])))
    assert res.passed, res.reasons
    # the gate ran the reader confined on the recorded bytes and got the fixture's bytes again
    assert res.parsed[PARSED_FIXTURE]["rederived"] is True and res.parsed[PARSED_FIXTURE]["parser"] == [READER]
    # the mapping now comes from a fixture: no grounding finding, whatever else its evidence shows
    assert not any("[examples:grounded]" in r for r in res.level_reasons.get(USER_OF, [])), res.level_reasons
    assert res.levels[USER_OF] in (UNCHECKED, REPRODUCES_ITS_CASE)
    assert res.verification[READER]["reads_for"] == [PARSED_FIXTURE]
    assert res.verification[USER_OF]["readers"] == [READER]


@needs_bwrap
def test_a_parsed_fixture_that_differs_from_its_reader_is_refused(text_world):
    mem, ev, blobs, gate = text_world
    data, _ = fx.make(T1, "request:0", PARSED_FIXTURE, blob=blobs.get, parser=READER, parse=_parse_like_the_reader)
    files, entries = _text_files(blobs, USERS_TEST_PARSED, parsed_bytes=data.replace(b"u-7", b"u-8"))
    res = gate().check(mem.head(), lw.candidate(mem, files), _manifest(entries, _items([USERS_TEST])))
    assert not res.passed and "G1" in res.refused
    assert any(PARSED_FIXTURE in r and fx.REMAKE in r for r in res.reasons), res.reasons


@pytest.mark.parametrize(
    "reader_source, why",
    [
        (
            READ_SOURCE.replace("def fields(text: str) -> dict:", "def fields(text: str, strict: bool) -> dict:"),
            "does not take one positional text parameter",
        ),
        (
            READ_SOURCE.replace("    return out\n", "    _SEEN.append(1)\n    out['seen'] = len(_SEEN)\n    return out\n")
            + "\n_SEEN = []\n",
            "gives different results on two runs",
        ),
    ],
)
@needs_bwrap
def test_a_reader_that_breaks_its_conditions_is_refused(text_world, reader_source, why):
    mem, ev, blobs, gate = text_world
    files, entries = _text_files(blobs, USERS_TEST_PARSED, parsed_bytes=b"")
    files[READ_MODULE] = reader_source
    res = gate().check(mem.head(), lw.candidate(mem, files), _manifest(entries, _items([USERS_TEST])))
    assert not res.passed and "G1" in res.refused
    assert any(why in r for r in res.reasons), res.reasons


@needs_bwrap
def test_a_reader_neither_listed_nor_stored_is_refused(text_world):
    mem, ev, blobs, gate = text_world
    files, entries = _text_files(blobs, USERS_TEST_PARSED, parsed_bytes=b"")
    users_only = [it for it in _items([USERS_TEST]) if it["item"] == USER_OF]
    files.pop(READ_TEST)
    res = gate().check(mem.head(), lw.candidate(mem, files), _manifest(entries, users_only))
    assert not res.passed
    assert any("neither listed in this pass's manifest nor stored" in r for r in res.reasons), res.reasons


@needs_bwrap
def test_a_function_with_no_call_on_any_recorded_value_is_still_refused(library_example_world):
    mem, ev, blobs, gate = library_example_world
    literal_only = lw.TEST_SOURCE.replace('user_id(load("me_1.json"))', 'user_id({"user_id": "u-1"})').replace(
        'user_id(load("me_2.json"))',
        'user_id({"user_id": "u-2"})',
    )
    literal_only = literal_only.replace(
        "INPUTS = [json.loads(x) for x in (DATA / \"users.user_id.inputs.jsonl\").read_text().splitlines() if x]",
        "INPUTS = []",
    )
    files, fixtures = lw.files_and_fixtures(blobs.get, test=literal_only)
    res = gate().check(mem.head(), lw.candidate(mem, files), lw.manifest(fixtures))
    assert not res.passed
    assert any(r.startswith(f"G2: [examples:grounded] {lw.ITEM_ID}") for r in res.reasons), res.reasons


# --- safety and integrity still refuse ----------------------------------------------------------------------------


@needs_bwrap
def test_safety_and_integrity_findings_still_refuse(library_example_world):
    mem, ev, blobs, gate = library_example_world
    files, fixtures = lw.files_and_fixtures(blobs.get)
    man = lw.manifest(fixtures)
    # G6: a key-shaped string in the library
    key = "sk-or-v1-" + "ab" * 32  # pragma: allowlist secret
    res = gate().check(
        mem.head(),
        lw.candidate(mem, {**files, lw.MODULE: lw.MODULE_SOURCE + f'\nTOKEN = "{key}"\n'}),
        man,
    )
    assert not res.passed and "G6" in res.refused, res.reasons
    # a malformed manifest
    res = gate().check(mem.head(), lw.candidate(mem, files), {**man, "items": "not a list"})
    assert not res.passed and any("malformed manifest" in r for r in res.reasons), res.reasons
    # a reserved path the harness generates
    res = gate().check(mem.head(), lw.candidate(mem, {**files, "INDEX.md": "# mine\n"}), man)
    assert not res.passed and "G1" in res.refused, res.reasons
    # an example call that raises something other than MemoryInputError
    raising = lw.MODULE_SOURCE.replace(
        '        raise MemoryInputError("an acct.me response with a user_id field is needed")\n',
        '        raise KeyError("user_id")\n',
    )
    res = gate().check(mem.head(), lw.candidate(mem, {**files, lw.MODULE: raising}), man)
    assert not res.passed, res.reasons
    # a regression: the parent's tests fail against the candidate's library
    assert gate().merge(mem.head(), lw.candidate(mem, files), man, "base", "write", None, "0").passed
    upper = lw.MODULE_SOURCE.replace('    return response["user_id"]\n', '    return response["user_id"].upper()\n')
    res = gate().check(mem.head(), lw.candidate(mem, {lw.MODULE: upper}), lw.manifest({}, lw.item()))
    assert not res.passed and "G3" in res.refused, res.reasons

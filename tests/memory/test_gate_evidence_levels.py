"""Evidence in any recorded form: findings of the evidence kind set an item's evidence level; only integrity and
safety refuse (library mode, writer role, in the jail).

Structured and unstructured recorded inputs are treated alike. A function whose recorded input is text lands
whether its test parses the text inline (the lowest level, ``unchecked``) or loads a fixture parsed from the
recorded text by a stored reader (the same level as a structured fixture). An item the check with unseen values
flags lands at the lowest level, and the flag is shown. G6, a malformed manifest, a regression and the reserved
paths still refuse.
"""

from __future__ import annotations

import json
import shutil
from typing import Any

import pytest

from tests.memory import library_example_world as w
from tests.memory.library_example_world import World
from tests.memory.test_episodes import _ep
from unify.memory import fixtures as fx
from unify.memory.blobs import BlobStore
from unify.memory.episodes import Action
from unify.memory.evidence import EvidenceStore
from unify.memory.gate import Gate
from unify.memory.gitio import Repo
from unify.memory.library_checks import LibraryConfig

pytestmark = pytest.mark.skipif(
    shutil.which("bwrap") is None,
    reason="bubblewrap required",
)

# --- assumed names (adapt here) ---------------------------------------------------------------------------------
#: the evidence levels, lowest first
LOWEST = "unchecked"
#: the level a function reaches with every evidence check passing on a structured (or parsed) fixture
FULL = "reproduces-its-case"


def level_of(res: Any, item: str) -> str | None:
    """An item's evidence level in a gate result (``GateResult.levels``, else its verification record)."""
    levels = getattr(res, "levels", None)
    if isinstance(levels, dict) and item in levels:
        return levels[item]
    return (res.verification.get(item) or {}).get("level")


def level_reasons_of(res: Any, item: str) -> list[str]:
    """The findings that capped an item's level (``GateResult.level_reasons``, else its verification record)."""
    reasons = getattr(res, "level_reasons", None)
    if isinstance(reasons, dict) and item in reasons:
        return list(reasons[item])
    return list((res.verification.get(item) or {}).get("level_reasons") or [])


def make_parsed(ep, dest: str, blob, *, append: bool = False, existing: dict | None = None):
    """A fixture made from *ep*'s recorded text by the stored reader :data:`READER_ID` (the ``fixture`` tool's
    ``parser=``): ``fixtures.make`` with the reader's id and an injected ``parse(item, text)``."""
    return fx.make(
        ep,
        0,
        dest,
        blob=blob,
        append=append,
        existing=existing,
        parser=READER_ID,
        # fixtures.make's parse returns (the result as canonical JSON text, the reader's digest)
        parse=lambda item, text: (fx.canonical(read_pairs(item, text)), "0" * 64),
    )


# -----------------------------------------------------------------------------------------------------------------

READER_ID = "memory.acct.readers:key_values"
READER_MODULE = "memory/acct/readers.py"
READER_TEST = "memory/acct/tests/test_readers.py"
READER_INPUTS = fx.inputs_file(READER_ID)
TEXT_FIXTURES = ("memory/acct/tests/data/profile_1.txt", "memory/acct/tests/data/profile_2.txt")


def _profile(episode_id: str, user: str, name: str):
    return _ep(
        episode_id=episode_id,
        actions=[
            Action(0, "acct", "profile", [], {}, f"user_id: {user}\nname: {name}\n", "ok", "read"),
        ],
    )


T1 = _profile("t1", "u-1", "Ada")
T2 = _profile("t2", "u-2", "Bo")
T3 = _profile("t3", "u-3", "Cy")  # recorded, copied by no fixture
EPISODES = {**w.EPISODES, **{e.episode_id: e for e in (T1, T2, T3)}}


def _lookup(eid: str, i: int):
    ep = EPISODES.get(eid)
    return ep.actions[i] if ep is not None and 0 <= i < len(ep.actions) else None


def read_pairs(item: str, text: str) -> dict:
    """The reader's own result, computed here for the fixture tool's parse step (the gate re-derives it by
    running the stored reader in a confined box)."""
    assert item == READER_ID
    out = {}
    for line in text.splitlines():
        if line.strip():
            key, _, value = line.partition(": ")
            out[key.strip()] = value.strip()
    return out


READER_SOURCE = f'''"""Readers of recorded text."""


class MemoryInputError(ValueError):
    pass


def key_values(text: str) -> dict:
    """The `key: value` lines of a block of text, as a mapping.

    Effect: read
    Example: {READER_TEST}::test_reads_the_recorded_blocks
    """
    if not isinstance(text, str) or ": " not in text:
        raise MemoryInputError("a block of `key: value` lines is needed")
    out = {{}}
    for line in text.splitlines():
        if not line.strip():
            continue
        key, sep, value = line.partition(": ")
        if not sep:
            raise MemoryInputError("every line of the block is `key: value`")
        out[key.strip()] = value.strip()
    return out
'''

READER_TEST_SOURCE = """import json
from pathlib import Path

import pytest

from memory.acct.readers import MemoryInputError, key_values

DATA = Path(__file__).parent / "data"
INPUTS = [json.loads(x) for x in (DATA / "readers.key_values.inputs.jsonl").read_text().splitlines() if x]


def test_reads_the_recorded_blocks():
    assert key_values((DATA / "profile_1.txt").read_text()) == {"user_id": "u-1", "name": "Ada"}
    assert key_values((DATA / "profile_2.txt").read_text()) == {"user_id": "u-2", "name": "Bo"}


@pytest.mark.parametrize("row", INPUTS)
def test_every_recorded_block_has_a_user_id(row):
    assert "user_id: " + key_values(row["input"])["user_id"] in row["input"]


def test_refuses_text_without_key_value_lines():
    with pytest.raises(MemoryInputError):
        key_values("no pairs here")
"""

#: user_id's test when its recorded input is text that the test parses inline
INLINE_PARSE_TEST = """import json
from pathlib import Path

import pytest

from memory.acct.users import MemoryInputError, user_id

DATA = Path(__file__).parent / "data"
INPUTS = [json.loads(x) for x in (DATA / "users.user_id.inputs.jsonl").read_text().splitlines() if x]


def parse(text):
    return dict(line.split(": ", 1) for line in text.splitlines() if line.strip())


def test_reads_the_recorded_user_ids():
    assert user_id(parse((DATA / "profile_1.txt").read_text())) == "u-1"
    assert user_id(parse((DATA / "profile_2.txt").read_text())) == "u-2"


@pytest.mark.parametrize("row", INPUTS)
def test_every_recorded_input_has_a_user_id(row):
    assert user_id(parse(row["input"])) == parse(row["input"])["user_id"]


def test_refuses_a_response_without_a_user_id():
    with pytest.raises(MemoryInputError):
        user_id({"name": "Ada"})
"""

#: user_id's tests on literals only: no call returns on a recorded value
LITERAL_ONLY_TEST = """import pytest

from memory.acct.users import MemoryInputError, user_id


def test_reads_the_recorded_user_ids():
    assert user_id({"user_id": "u-1", "name": "Ada"}) == "u-1"


def test_refuses_a_response_without_a_user_id():
    with pytest.raises(MemoryInputError):
        user_id({"name": "Ada"})
"""

SPECIAL_CASED = w.MODULE_SOURCE.replace(
    '    return response["user_id"]\n',
    '    return {"u-1": "u-1", "u-2": "u-2"}[response["user_id"]]\n',
)
KEY = "sk-or-v1-" + "ab" * 32  # pragma: allowlist secret


@pytest.fixture
def text_world(tmp_path) -> World:
    """The shared world's stores, indexing the text episodes too."""
    mem = Repo.init_bare(tmp_path / "mem.git")
    ev = EvidenceStore(tmp_path / "e.sqlite")
    for n, ep in enumerate(EPISODES.values()):
        ev.index_episode(ep, f"{n + 1:x}" * 40)
    blobs = BlobStore(tmp_path / "b")

    def gate(**cfg: Any) -> Gate:
        return Gate(
            mem,
            ev,
            blobs,
            action_lookup=_lookup,
            library_mode=LibraryConfig(checks=True, episodes=EPISODES.get, **cfg),
        )

    return World(mem, ev, blobs, gate)


def _copies(blob, dests, item: str) -> tuple[dict[str, str], dict[str, dict]]:
    """Whole copies of t1's and t2's recorded text at *dests*, and *item*'s recorded-inputs file of the same."""
    files, fixtures = {}, {}
    for dest, ep in zip(dests, (T1, T2)):
        data, entry = fx.make(ep, 0, dest, blob=blob)
        files[dest], fixtures[dest] = data.decode(), entry
    lines, entry, rel = b"", None, fx.inputs_file(item)
    for ep in (T1, T2):
        line, entry = fx.make(ep, 0, rel, blob=blob, append=True, existing=entry)
        lines += line
    files[rel], fixtures[rel] = lines.decode(), entry
    return files, fixtures


def _check(world, files, man):
    mem, _, _, gate = world
    return gate().check(mem.head(), w.candidate(mem, files), man)


def _merge(world, files, man, pass_id="p1"):
    mem, _, _, gate = world
    return gate().merge(
        mem.head(),
        w.candidate(mem, files),
        man,
        pass_id,
        "write",
        None,
        "0.01",
    )


def _refusal_lines(res) -> list[str]:
    return [r for r in res.reasons if not r.startswith("note:")]


# --- levels -----------------------------------------------------------------------------------------------------


def test_a_structured_fixture_lands_at_the_full_level(text_world):
    _, _, blobs, _ = text_world
    files, fixtures = w.files_and_fixtures(blobs.get)
    res = _merge(text_world, files, w.manifest(fixtures))
    assert res.passed and w.ITEM_ID in res.items_merged, res.reasons
    assert level_of(res, w.ITEM_ID) == FULL
    assert level_reasons_of(res, w.ITEM_ID) == []


def test_a_text_input_parsed_inline_in_the_test_lands_at_the_lowest_level(text_world):
    """The same function on the same values recorded as text: its tests parse the text inline, so its structured
    argument is not taken from a fixture. Today that refuses the item; now it caps its level."""
    _, _, blobs, _ = text_world
    files, fixtures = _copies(blobs.get, TEXT_FIXTURES, w.ITEM_ID)
    files.update({w.PACKAGE_INIT: '"""Accounts."""\n', w.MODULE: w.MODULE_SOURCE, w.TEST_FILE: INLINE_PARSE_TEST})
    man = w.manifest(fixtures, w.item(source_episodes=["t1", "t2"]))
    res = _merge(text_world, files, man)
    assert res.passed and w.ITEM_ID in res.items_merged, res.reasons
    assert level_of(res, w.ITEM_ID) == LOWEST
    capped = level_reasons_of(res, w.ITEM_ID)
    assert any("[examples:grounded]" in r for r in capped), capped
    # a capped finding is shown, but it is not a refusal
    assert not any(r.startswith("G2: [examples:grounded]") for r in res.reasons), res.reasons
    assert not res.item_reasons.get(w.ITEM_ID), res.item_reasons


def test_a_text_input_grounded_through_a_stored_reader_lands_at_the_full_level(text_world):
    """The recorded text goes through a stored reader into a parsed fixture; the function takes the parsed value as
    it would take a structured one, and reaches the same level. The reader lands with it."""
    _, _, blobs, _ = text_world
    files, fixtures = _copies(blobs.get, TEXT_FIXTURES, READER_ID)
    for dest, ep in zip(w.FIXTURE_PATHS, (T1, T2)):
        data, entry = make_parsed(ep, dest, blobs.get)
        assert json.loads(data) == read_pairs(READER_ID, ep.actions[0].response)
        assert entry["kind"] == "parsed" and entry["parser"] == READER_ID
        files[dest], fixtures[dest] = data.decode(), entry
    lines, entry = b"", None
    for ep in (T1, T2):
        line, entry = make_parsed(ep, w.INPUTS_PATH, blobs.get, append=True, existing=entry)
        lines += line
    files[w.INPUTS_PATH], fixtures[w.INPUTS_PATH] = lines.decode(), entry
    files.update(
        {
            w.PACKAGE_INIT: '"""Accounts."""\n',
            w.MODULE: w.MODULE_SOURCE,
            w.TEST_FILE: w.TEST_SOURCE,
            READER_MODULE: READER_SOURCE,
            READER_TEST: READER_TEST_SOURCE,
        },
    )
    reader = {
        "item": READER_ID,
        "kind": "function",
        "source_episodes": ["t1", "t2"],
        "tests": [READER_TEST],
        "covers": [],
    }
    man = w.manifest(fixtures, reader, w.item(source_episodes=["t1", "t2"]))
    res = _merge(text_world, files, man)
    assert res.passed, res.reasons
    assert set(res.items_merged) >= {READER_ID, w.ITEM_ID}
    assert level_of(res, w.ITEM_ID) == FULL, level_reasons_of(res, w.ITEM_ID)
    assert level_reasons_of(res, w.ITEM_ID) == []
    assert level_of(res, READER_ID) in (LOWEST, FULL)


def test_a_parsed_fixture_that_does_not_re_derive_is_refused(text_world):
    """Integrity: a parsed fixture whose bytes differ from what the stored reader makes of the recorded text."""
    _, _, blobs, _ = text_world
    files, fixtures = _copies(blobs.get, TEXT_FIXTURES, READER_ID)
    for dest, ep in zip(w.FIXTURE_PATHS, (T1, T2)):
        data, entry = make_parsed(ep, dest, blobs.get)
        files[dest], fixtures[dest] = data.decode(), entry
    lines, entry = b"", None
    for ep in (T1, T2):
        line, entry = make_parsed(ep, w.INPUTS_PATH, blobs.get, append=True, existing=entry)
        lines += line
    files[w.INPUTS_PATH], fixtures[w.INPUTS_PATH] = lines.decode(), entry
    files[w.FIXTURE_PATHS[0]] = files[w.FIXTURE_PATHS[0]].replace("u-1", "u-7")
    files.update(
        {
            w.PACKAGE_INIT: '"""Accounts."""\n',
            w.MODULE: w.MODULE_SOURCE,
            w.TEST_FILE: w.TEST_SOURCE.replace('"u-1"', '"u-7"'),
            READER_MODULE: READER_SOURCE,
            READER_TEST: READER_TEST_SOURCE,
        },
    )
    reader = {
        "item": READER_ID,
        "kind": "function",
        "source_episodes": ["t1", "t2"],
        "tests": [READER_TEST],
        "covers": [],
    }
    man = w.manifest(fixtures, reader, w.item(source_episodes=["t1", "t2"]))
    res = _check(text_world, files, man)
    assert not res.passed and "G1" in res.refused, res.reasons
    assert any(w.FIXTURE_PATHS[0] in r for r in res.reasons if r.startswith("G1:")), res.reasons


def test_an_item_the_unseen_value_check_flags_lands_at_the_lowest_level_with_the_flag_shown(
    text_world,
):
    _, _, blobs, _ = text_world
    files, fixtures = w.files_and_fixtures(blobs.get, module=SPECIAL_CASED)
    res = _merge(text_world, files, w.manifest(fixtures))
    assert res.passed and w.ITEM_ID in res.items_merged, res.reasons
    assert level_of(res, w.ITEM_ID) == LOWEST
    capped = level_reasons_of(res, w.ITEM_ID)
    assert any("[examples:special-cased]" in r for r in capped), capped
    # the writer sees the flag in the gate's reply, as a note, never as a refusal
    shown = [r for r in res.reasons if "[examples:special-cased]" in r]
    assert shown and all(r.startswith("note:") for r in shown), res.reasons
    assert not res.item_reasons.get(w.ITEM_ID), res.item_reasons


def test_a_function_with_no_example_on_a_recorded_value_is_not_landed(text_world):
    _, _, blobs, _ = text_world
    files = {
        w.PACKAGE_INIT: '"""Accounts."""\n',
        w.MODULE: w.MODULE_SOURCE,
        w.TEST_FILE: LITERAL_ONLY_TEST,
    }
    res = _merge(text_world, files, w.manifest({}))
    assert w.ITEM_ID not in res.items_merged, res.reasons


# --- the safety and integrity checks still refuse ----------------------------------------------------------------


def test_g6_still_refuses_a_key_shaped_string(text_world):
    _, _, blobs, _ = text_world
    files, fixtures = w.files_and_fixtures(blobs.get)
    files[w.MODULE] = w.MODULE_SOURCE + f"\n# token {KEY}\n"
    res = _check(text_world, files, w.manifest(fixtures))
    assert not res.passed and "G6" in res.refused, res.reasons
    assert not any(KEY in r for r in res.reasons)


@pytest.mark.parametrize(
    "bad",
    [
        {"items": "nope"},
        {"items": [{"item": w.ITEM_ID, "kind": "magic"}]},
        {"items": [{"item": "memory.acct.users:user_id", "kind": "function", "tests": ["../escape.py"]}]},
    ],
)
def test_a_malformed_manifest_still_refuses(text_world, bad):
    _, _, blobs, _ = text_world
    files, _ = w.files_and_fixtures(blobs.get)
    res = _check(text_world, files, bad)
    assert not res.passed and not res.checks.get("G1", True), res.reasons


def test_a_regression_still_refuses(text_world):
    """The parent's tests fail against the candidate's library: refused even though the edit has its red test."""
    mem, _, blobs, _ = text_world
    files, fixtures = w.files_and_fixtures(blobs.get)
    assert _merge(text_world, files, w.manifest(fixtures), "p0").passed
    upper = w.MODULE_SOURCE.replace(
        '    return response["user_id"]\n',
        '    return response["user_id"].upper()\n',
    )
    red = "memory/acct/tests/test_users_upper.py"
    red_source = (
        "from memory.acct.users import user_id\n\n\n"
        "def test_the_id_is_upper_case():\n"
        '    assert user_id({"user_id": "u-1"}) == "U-1"\n'
    )
    man = w.manifest({}, w.item(tests=[w.TEST_FILE, red]))
    res = _check(text_world, {w.MODULE: upper, red: red_source}, man)
    assert not res.passed and "G3" in res.refused, res.reasons
    assert any("parent's tests" in r or "new failures" in r for r in _refusal_lines(res)), res.reasons


@pytest.mark.parametrize(
    "path, why",
    [
        ("INDEX.md", "the harness generates it"),
        ("scripts/run.py", "the library holds only memory/ and notes/"),
        ("memory/acct/tests/__init__.py", "tests/ takes no __init__.py"),
    ],
)
def test_the_reserved_paths_still_refuse(text_world, path, why):
    _, _, blobs, _ = text_world
    files, fixtures = w.files_and_fixtures(blobs.get)
    files[path] = "x = 1\n" if path.endswith(".py") else "# Memory index\n"
    res = _check(text_world, files, w.manifest(fixtures))
    assert not res.passed and "G1" in res.refused, res.reasons
    assert any(r.startswith(f"G1: file {path}:") and why in r for r in res.reasons), res.reasons

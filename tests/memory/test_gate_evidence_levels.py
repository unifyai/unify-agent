"""Only integrity and safety refuse, whatever form the recorded input takes (library mode, writer role, in the jail).

Structured and unstructured recorded inputs are treated alike. The gate runs no code (rev 2.1), so every function
lands at ``unchecked`` with labels saying what was not run (``test_gate_runs_nothing``); G6, a malformed manifest
and the reserved paths still refuse.
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
            library_mode=LibraryConfig(episodes=EPISODES.get, **cfg),
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


# --- levels -----------------------------------------------------------------------------------------------------


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

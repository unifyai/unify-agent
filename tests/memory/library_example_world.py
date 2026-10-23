"""A shared library-mode world whose function takes any arguments and whose tests are its examples.

The library holds one function, ``memory.acct.users:user_id(response: dict)``. Its test file loads two fixtures,
each a whole copy of a recorded ``acct.me`` response made with :func:`unify.memory.fixtures.make` (by default one from
episode ``e1`` and one from ``e2``), and calls the function on each. It also reads the function's recorded-inputs file
(:func:`unify.memory.fixtures.inputs_file`, assembled with ``fixtures.make(..., append=True)`` from the same
episodes). The manifest item names the function's tests and source
episodes only: it has no ``input`` key and no action cover. The manifest's ``fixtures`` key carries the provenance
entries ``fixtures.make`` returned.

Public names:

* episodes ``E1``, ``E2``, ``E3`` (three recorded ``acct.me`` responses; ``E3`` is copied by no default fixture), ``EPISODES`` (id to episode), and :func:`lookup` (the gate's ``action_lookup``);
* ids and paths ``ITEM_ID``, ``PACKAGE_INIT``, ``MODULE``, ``TEST_FILE``, ``FIXTURE_PATHS`` (the two copies),
  ``INPUTS_PATH`` (the recorded-inputs file), and the test ids ``EXAMPLE_TEST``, ``INPUTS_TEST``, ``REFUSAL_TEST``;
* sources ``MODULE_SOURCE``, ``TEST_SOURCE`` (:func:`example_test_source` for other fixture episodes) and
  ``WEAK_TEST_SOURCE`` (the same calls with no exact assertion in any test);
* builders :func:`make_fixtures`, :func:`files_and_fixtures`, :func:`item`, :func:`manifest`, :func:`candidate`
  (exactly the files given) and :func:`written_candidate` (with the sections a writer adds);
* the pytest fixture :func:`library_example_world` (a :class:`World`: memory repo, evidence store, blob store and a
  library-mode gate factory).
"""

from __future__ import annotations

import json
from typing import Any, Callable, NamedTuple

import pytest

from unify.memory import fixtures as fx
from unify.memory.blobs import BlobStore
from unify.memory.episodes import Action, Episode
from unify.memory.evidence import EvidenceStore
from unify.memory.gate import Gate
from unify.memory.gitio import Repo
from unify.memory.library_checks import LibraryConfig
from tests.memory.test_episodes import _ep

ITEM_ID = "memory.acct.users:user_id"
PACKAGE_INIT = "memory/acct/__init__.py"
#: The package folder's README (every library folder holds one, :mod:`unify.memory.when_to_use`).
README = "memory/acct/README.md"
MODULE = "memory/acct/users.py"
TEST_FILE = "memory/acct/tests/test_users.py"
EXAMPLE_TEST = f"{TEST_FILE}::test_reads_the_recorded_user_ids"
INPUTS_TEST = f"{TEST_FILE}::test_every_recorded_input_has_a_user_id"
REFUSAL_TEST = f"{TEST_FILE}::test_refuses_a_response_without_a_user_id"
#: The copied fixtures, in the order of the episodes they are copied from.
FIXTURE_PATHS = ("memory/acct/tests/data/me_1.json", "memory/acct/tests/data/me_2.json")
#: The recorded-inputs file: one line per fixture episode, assembled by fixtures.make(..., append=True).
INPUTS_PATH = fx.inputs_file(ITEM_ID)


def _me(episode_id: str, user: str, name: str) -> Episode:
    return _ep(
        episode_id=episode_id,
        actions=[
            Action(0, "acct", "me", [], {}, {"user_id": user, "name": name}, "ok", "read"),
        ],
    )


E1 = _me("e1", "u-1", "Ada")
E2 = _me("e2", "u-2", "Bo")
E3 = _me("e3", "u-3", "Cy")  # recorded, but no fixture of the default world copies it
EPISODES = {e.episode_id: e for e in (E1, E2, E3)}


def lookup(eid: str, i: int) -> Action | None:
    """The gate's ``action_lookup`` over :data:`EPISODES`."""
    ep = EPISODES.get(eid)
    return ep.actions[i] if ep is not None and 0 <= i < len(ep.actions) else None


MODULE_SOURCE = f'''"""Accounts."""


class MemoryInputError(ValueError):
    pass


def user_id(response: dict) -> str:
    """The account's user id, read from an acct.me response.

    Effect: read
    Example: {EXAMPLE_TEST}

    When to use:
        Saved for: reading the user id from an account response.
        Recognise: the request needs the id of the account an acct.me response describes.
        Not for: responses of other calls, or the account's other fields.
        Example request: Which user id does this account response hold?
    """
    if not isinstance(response, dict) or "user_id" not in response:
        raise MemoryInputError("an acct.me response with a user_id field is needed")
    return response["user_id"]
'''

_TEST_HEAD = """import json
from pathlib import Path

import pytest

from memory.acct.users import MemoryInputError, user_id

DATA = Path(__file__).parent / "data"
INPUTS = [json.loads(x) for x in (DATA / "users.user_id.inputs.jsonl").read_text().splitlines() if x]


def load(name):
    return json.loads((DATA / name).read_text())


def test_reads_the_recorded_user_ids():
"""
_TEST_TAIL = """

@pytest.mark.parametrize("row", INPUTS)
def test_every_recorded_input_has_a_user_id(row):
    assert user_id(row["input"]) == row["input"]["user_id"]


def test_refuses_a_response_without_a_user_id():
    with pytest.raises(MemoryInputError):
        user_id({"name": "Ada"})
"""


def _user(ep: Episode) -> str:
    return ep.actions[0].response["user_id"]


def example_test_source(sources: tuple[Episode, ...] = (E1, E2)) -> str:
    """The test file: one exact assertion per fixture, each calling ``user_id`` on the loaded fixture."""
    body = "".join(
        f'    assert user_id(load("{path.rsplit("/", 1)[1]}")) == "{_user(ep)}"\n'
        for path, ep in zip(FIXTURE_PATHS, sources)
    )
    return _TEST_HEAD + body + _TEST_TAIL


TEST_SOURCE = example_test_source()
#: The test file with no trusted exact assertion: each fixture's call, and each recorded-inputs line's, is only
#: checked to return a string (the recorded-inputs test's expected value comes from its fixture, so it is trusted).
WEAK_TEST_SOURCE = (
    TEST_SOURCE.replace(
        'assert user_id(load("me_1.json")) == "u-1"',
        'assert isinstance(user_id(load("me_1.json")), str)',
    )
    .replace(
        'assert user_id(load("me_2.json")) == "u-2"',
        'assert isinstance(user_id(load("me_2.json")), str)',
    )
    .replace(
        'assert user_id(row["input"]) == row["input"]["user_id"]',
        'assert isinstance(user_id(row["input"]), str)',
    )
)


def make_fixtures(
    blob: Callable[[str], bytes],
    sources: tuple[Episode, ...] = (E1, E2),
) -> dict[str, tuple[bytes, dict]]:
    """Each fixture path to its bytes and provenance entry: a whole copy of action 0's response of its episode,
    and the recorded-inputs file assembled from the same responses (one line per episode, in order).
    """
    made = {path: fx.make(ep, 0, path, blob=blob) for path, ep in zip(FIXTURE_PATHS, sources)}
    lines, entry = b"", None
    for ep in sources:
        line, entry = fx.make(ep, 0, INPUTS_PATH, blob=blob, append=True, existing=entry)
        lines += line
    made[INPUTS_PATH] = (lines, entry)
    return made


def files_and_fixtures(
    blob: Callable[[str], bytes],
    *,
    sources: tuple[Episode, ...] = (E1, E2),
    module: str = MODULE_SOURCE,
    test: str | None = None,
) -> tuple[dict[str, str], dict[str, dict]]:
    """The candidate's files (package, module, test file, fixtures) and the manifest's ``fixtures`` entries."""
    made = make_fixtures(blob, sources)
    files = {
        PACKAGE_INIT: '"""Accounts."""\n',
        README: "Reading account responses.\n",
        MODULE: module,
        TEST_FILE: example_test_source(sources) if test is None else test,
        **{path: data.decode() for path, (data, _) in made.items()},
    }
    return files, {path: entry for path, (_, entry) in made.items()}


def item(**over: Any) -> dict:
    """The function's manifest item: no ``input`` key and no action cover."""
    return {
        "item": ITEM_ID,
        "kind": "function",
        "source_episodes": ["e1", "e2"],
        "tests": [TEST_FILE],
        "covers": [],
        **over,
    }


def manifest(fixtures: dict[str, dict], *items: dict, **top: Any) -> dict:
    """A library-mode manifest of *items* (default: :func:`item`) with the given ``fixtures`` entries."""
    return {
        "items": list(items) or [item()],
        "deleted": [],
        "unlisted": [],
        "support": [],
        "fixtures": fixtures,
        **top,
    }


def candidate(mem: Repo, files: dict[str, str | None], base: str = "main") -> str:
    """Commit *files* (``None`` deletes) on *base*, exactly as given, and return the candidate commit. Nothing is
    added: a function or note without its sections, or a folder without its README.md, is committed so, and the gate
    sees it so (:func:`written_candidate` commits what a writer would leave)."""
    with mem.temp_checkout(base) as wt:
        for rel, text in files.items():
            if text is None:
                (wt / rel).unlink()
                continue
            (wt / rel).parent.mkdir(parents=True, exist_ok=True)
            (wt / rel).write_text(text)
        return mem.commit_all(wt, "pass", {"Pass": "p1"})


def written_candidate(mem: Repo, files: dict[str, str | None], base: str = "main") -> str:
    """:func:`candidate` of *files* as a writer leaves them: what they add or change gets its "when to use" section
    (and a note its ``## Rule`` section), each folder its README.md (:func:`tests.memory.writer_sections.as_written`);
    what they leave as *base* has it stays as it is. For tests about something else than those sections."""
    from tests.memory.writer_sections import as_written

    with mem.temp_checkout(base) as wt:
        written = {
            rel: text
            for rel, text in files.items()
            if text is None or not (wt / rel).is_file() or (wt / rel).read_text() != text
        }
        present = _Present(wt)
        completed = {**files, **as_written(written, present)}
    return candidate(mem, completed, base)


class _Present:
    """The paths of a checkout, for ``in``."""

    def __init__(self, root) -> None:
        self.root = root

    def __contains__(self, rel: object) -> bool:
        return isinstance(rel, str) and (self.root / rel).is_file()


class World(NamedTuple):
    mem: Repo
    ev: EvidenceStore
    blobs: BlobStore
    gate: Callable[..., Gate]  # gate(**LibraryConfig fields) -> a library-mode gate with checks on


@pytest.fixture
def library_example_world(tmp_path) -> World:
    """An empty memory repo, an evidence store indexing :data:`EPISODES`, a blob store and a gate factory."""
    mem = Repo.init_bare(tmp_path / "mem.git")
    ev = EvidenceStore(tmp_path / "e.sqlite")
    for n, ep in enumerate(EPISODES.values()):
        ev.index_episode(ep, str(n + 1) * 40)
    blobs = BlobStore(tmp_path / "b")

    def gate(**cfg: Any) -> Gate:
        return Gate(
            mem,
            ev,
            blobs,
            action_lookup=lookup,
            library_mode=LibraryConfig(episodes=EPISODES.get, **cfg),
        )

    return World(mem, ev, blobs, gate)

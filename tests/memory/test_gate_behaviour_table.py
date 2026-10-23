"""What the gate does with each rule the writer's brief states, checked by real merges of raw candidates.

Each row commits a candidate exactly as written (nothing adds a missing section or README) and merges it with the
library gate (``Gate.merge``), then asserts what the brief says happens: the item lands (with a label naming the
finding) or is refused. The brief's words come from :data:`unify.memory.gate_policy.GATE`
(``test_brief_states_the_gate.py``); these rows hold the gate to them.

No model, network or key. The procedure rows land a captured procedure first (``tests.memory.gate_records``), which
needs bubblewrap and prlimit.
"""

from __future__ import annotations

import pytest

from tests.memory import gate_records as gr
from tests.memory import library_example_world as lw
from tests.memory.library_example_world import library_example_world  # noqa: F401 - a fixture
from tests.memory.writer_sections import note_lines
from unify.memory import gate_policy as gp

NOTE = "notes/acct/ids.md"
NOTE_HEAD = "---\ntitle: Account ids\ndescription: where an account response keeps its id\nuses: []\n---\n"
NOTE_RULE = "## Rule\nThe id is the response's user_id field.\n\n"
#: A second function of the module, for the rows that remove one.
NAME_ID = "memory.acct.users:user_name"
SECOND = '''

def user_name(response: dict) -> str:
    """The account's name, read from an acct.me response.

    When to use:
        Saved for: reading the name from an account response.
        Recognise: the request needs the name an acct.me response holds.
        Not for: responses of other calls.
        Example request: Which name does this account response hold?
    """
    if not isinstance(response, dict) or "name" not in response:
        raise MemoryInputError("an acct.me response with a name field is needed")
    return response["name"]
'''
ALIAS = "memory.acct.users:old_user_id"


def _merge(world, files, manifest, pass_id="p-table"):
    mem, _, _, gate = world
    return gate().merge(mem.head(), lw.candidate(mem, files), manifest, pass_id, "write", None, "0")


def _base(world, *, module: str = lw.MODULE_SOURCE, items=None):
    """The default world's library merged first (the parent of the row's pass): its files and fixture entries."""
    files, fixtures = lw.files_and_fixtures(world.blobs.get, module=module)
    res = _merge(world, files, lw.manifest(fixtures, *(items or [lw.item()])), pass_id="p-base")
    assert res.passed, res.reasons
    return files, fixtures


def _labels(res, item):
    return res.verification.get(item, {}).get("label_findings", [])


def _note_item():
    return {"item": NOTE, "kind": "note", "source_episodes": ["e1"]}


# --- each row: (the world) -> (the result, the item the row is about, lands?) ----------------------------------------


def complete(world):
    files, fixtures = lw.files_and_fixtures(world.blobs.get)
    files[NOTE] = NOTE_HEAD + NOTE_RULE + note_lines()
    files["notes/acct/README.md"] = "Notes on account responses.\n"
    return _merge(world, files, lw.manifest(fixtures, lw.item(), _note_item())), NOTE, True


def missing_rule(world):
    files, fixtures = lw.files_and_fixtures(world.blobs.get)
    files[NOTE] = NOTE_HEAD + "The id is the response's user_id field.\n\n" + note_lines()
    files["notes/acct/README.md"] = "Notes on account responses.\n"
    return _merge(world, files, lw.manifest(fixtures, lw.item(), _note_item())), NOTE, False


def missing_when_to_use(world):
    module = lw.MODULE_SOURCE.split("    When to use:")[0].rstrip() + '\n    """' + lw.MODULE_SOURCE.split('    """', 2)[2]
    assert "When to use" not in module and "def user_id" in module
    files, fixtures = lw.files_and_fixtures(world.blobs.get, module=module)
    res = _merge(world, files, lw.manifest(fixtures))
    assert any("has no `When to use:` section" in r for r in _labels(res, lw.ITEM_ID)), _labels(res, lw.ITEM_ID)
    return res, lw.ITEM_ID, True


def missing_readme(world):
    files, fixtures = lw.files_and_fixtures(world.blobs.get)
    del files[lw.README]
    res = _merge(world, files, lw.manifest(fixtures))
    assert any("has no README.md" in r for r in _labels(res, lw.ITEM_ID)), _labels(res, lw.ITEM_ID)
    return res, lw.ITEM_ID, True


def fewer_items(world):
    _base(world, module=lw.MODULE_SOURCE + SECOND,
          items=[lw.item(), lw.item(item=NAME_ID, tests=[lw.TEST_FILE])])  # fmt: skip
    files, fixtures = lw.files_and_fixtures(world.blobs.get)
    res = _merge(world, files, lw.manifest(fixtures, lw.item(), deleted=[NAME_ID]))
    assert any("may not reduce the number of items" in r for r in res.reasons), res.reasons
    assert any("may not delete an item" in r for r in res.reasons), res.reasons
    return res, lw.ITEM_ID, False


def fewer_tests(world):
    files, fixtures = _base(world)
    test = lw.TEST_SOURCE.split("\n\ndef test_refuses_a_response_without_a_user_id")[0] + "\n"
    assert test.count("def test_") == 2
    res = _merge(world, {**files, lw.TEST_FILE: test}, lw.manifest(fixtures))
    assert any("fewer tests than its parent" in r for r in res.reasons), res.reasons
    return res, lw.ITEM_ID, True


def dropped_alias(world):
    files, fixtures = _base(world, module=lw.MODULE_SOURCE + "\n\nold_user_id = user_id\n")
    world.ev.record_curations(
        "p-base", world.mem.head(), [{"item": ALIAS, "action": "alias", "target": lw.ITEM_ID, "reason": "renamed"}],
    )
    res = _merge(world, {**files, lw.MODULE: lw.MODULE_SOURCE}, lw.manifest(fixtures))
    assert any("no longer forwarded" in r for r in res.reasons), res.reasons
    return res, lw.ITEM_ID, False


def fixture_provenance(world):
    files, fixtures = lw.files_and_fixtures(world.blobs.get)
    path = lw.FIXTURE_PATHS[0]
    files[path] = files[path].replace("u-1", "u-9")
    test = lw.TEST_SOURCE.replace('== "u-1"', '== "u-9"')
    return _merge(world, {**files, lw.TEST_FILE: test}, lw.manifest(fixtures)), lw.ITEM_ID, False


def cited_unknown_episode(world):
    files, fixtures = lw.files_and_fixtures(world.blobs.get)
    return _merge(world, files, lw.manifest(fixtures, lw.item(source_episodes=["e1", "e9"]))), lw.ITEM_ID, False


def no_source_episode(world):
    files, fixtures = lw.files_and_fixtures(world.blobs.get)
    return _merge(world, files, lw.manifest(fixtures, lw.item(source_episodes=[]))), lw.ITEM_ID, False


def key_shaped_string(world):
    key = "sk-" + "a1B2" * 10  # key-shaped, never a real key
    module = lw.MODULE_SOURCE.replace('"""Accounts."""', f'"""Accounts."""\n\nNOTE = "{key}"')
    files, fixtures = lw.files_and_fixtures(world.blobs.get, module=module)
    res = _merge(world, files, lw.manifest(fixtures))
    assert key not in "\n".join(res.reasons)
    return res, lw.ITEM_ID, False


def code_that_does_not_parse(world):
    files, fixtures = lw.files_and_fixtures(world.blobs.get)
    test = lw.TEST_SOURCE + "\ndef test_broken(:\n    pass\n"
    res = _merge(world, {**files, lw.TEST_FILE: test}, lw.manifest(fixtures))
    assert any(gp.NO_PARSE_MARK in r for r in res.reasons), res.reasons
    return res, lw.ITEM_ID, False


ROWS = {
    "complete": complete,
    "missing Rule": missing_rule,
    "missing When to use": missing_when_to_use,
    "missing README": missing_readme,
    "fewer items": fewer_items,
    "fewer tests": fewer_tests,
    "dropped alias": dropped_alias,
    "fixture provenance": fixture_provenance,
    "cited unknown episode": cited_unknown_episode,
    "no source episode": no_source_episode,
    "key-shaped string": key_shaped_string,
    "code that does not parse": code_that_does_not_parse,
}


@pytest.mark.parametrize("row", list(ROWS))
def test_the_gate_does_what_the_brief_says(row, library_example_world):  # noqa: F811
    res, item, lands = ROWS[row](library_example_world)
    if lands:
        assert res.passed and item in res.items_merged, (row, res.reasons)
        assert library_example_world.mem.head() == res.merged
    else:
        assert item not in res.items_merged, (row, res.reasons, res.items_merged)
        assert item in res.items_refused or not res.passed, (row, res.reasons)


# --- the procedure rows: a captured procedure landed with land_capture, then merged ---------------------------------

PROC = "memory.orders.procedures:line_price"


def _procedure(tmp_path, *, signals=(), edit_manifest=None):
    from tests.memory.test_recipe_gate import DERIVES, PRICING, SHOWN_CALL, _episode

    ep = _episode("p1", PRICING, 12)
    landing = gr.Landing("p1", DERIVES, "## Asked\nPrice an order line.\n", "orders", "line_price", (SHOWN_CALL,))
    return gr.build(
        tmp_path, episodes={"p1": ep}, base={}, landings=[landing], signals=signals, edit_manifest=edit_manifest,
        require_pass=False, merge=True,
    ).result


@gr.needs_box
@pytest.mark.timeout(900)
def test_a_procedure_from_a_failed_source_episode_lands_with_its_label(tmp_path):
    from unify.memory.signals import Signal

    contrary = Signal("p1.checker", "p1", "checker", "fail", "2026-10-08T01:02:00Z", regime="dense", visible_to_actor=True)
    res = _procedure(tmp_path, signals=[contrary])
    assert res.passed and PROC in res.items_merged, res.reasons
    want = gp.FAILED_SOURCE_LABEL.format(item=PROC, episodes="p1")
    assert any(want in r for r in _labels(res, PROC)), _labels(res, PROC)


@gr.needs_box
@pytest.mark.timeout(900)
def test_a_procedure_without_a_recipe_lands_with_its_label(tmp_path):
    def without_recipe(manifest: dict) -> dict:
        items = [
            {k: v for k, v in i.items() if not (i["item"] == PROC and k in ("recipe", "cases"))}
            for i in manifest["items"]
        ]
        return {**manifest, "items": items}

    res = _procedure(tmp_path, edit_manifest=without_recipe)
    assert res.passed and PROC in res.items_merged, res.reasons
    assert any(gp.NO_RECIPE_LABEL.format(item=PROC) in r for r in _labels(res, PROC)), _labels(res, PROC)

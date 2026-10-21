"""A library function's effect is derived from what it reproduces and recorded in its item record: the writer
declares nothing, and the environment-function ``Effect:`` checks are unchanged."""

from __future__ import annotations

import inspect
from types import SimpleNamespace


from unify.memory.library_checks import LibraryChecks, derive_effect
from unify.memory.procedures import Cover

WRITER, READER = "memory.shop.cart:add_all", "memory.shop.cart:total"


def _action(effect, kind="tool"):
    return SimpleNamespace(
        kind=kind,
        effect=effect,
        status="ok",
        error=None,
        response={"ok": True},
    )


def _episodes(**by_id):
    def get(eid):
        actions = by_id.get(eid)
        return None if actions is None else SimpleNamespace(actions=actions)

    return get


EPISODES = _episodes(
    wrote=[_action("read"), _action("write"), _action("read")],
    read=[_action("read")],
    undeclared=[_action("unknown"), _action("read")],
)


def _tool(eid):
    return Cover(eid, "episode", runner="tool")


#: derive_effect takes the typed covers and the episode reader only


def test_a_tool_procedure_over_recorded_writes_is_a_write():
    assert derive_effect([_tool("wrote")], EPISODES) == "write"
    # one write cover among others is enough
    assert derive_effect([_tool("read"), _tool("wrote")], EPISODES) == "write"


def test_a_tool_procedure_whose_calls_declare_no_effect_is_unknown():
    assert derive_effect([_tool("undeclared")], EPISODES) == "unknown"
    assert derive_effect([_tool("missing")], EPISODES) == "unknown"


def test_a_worktree_procedure_changes_files():
    cover = Cover("wrote", "episode", runner="worktree")
    assert derive_effect([cover], EPISODES) == "write"


def test_a_dialogue_procedure_gives_read():
    dialogue = Cover("read", "episode", runner="dialogue", action=0)
    assert derive_effect([dialogue], EPISODES) == "read"


def test_only_episode_covers_give_an_effect():
    """Replaces test_action_covers_carry_their_recorded_effect_unless_the_input_is_a_recorded_value (and the action
    cover and cell cases of test_read_only_covers_give_read). The rule that replaces it: action covers are refused in
    library mode, so a function's effect is derived from its ``episode`` covers only, and is None (never guessed)
    without one.
    """
    assert derive_effect([Cover("read", "cell", index=0)], EPISODES) is None
    assert derive_effect([Cover("wrote", "diff")], EPISODES) is None
    assert derive_effect([], EPISODES) is None  # nothing to derive from
    params = list(inspect.signature(derive_effect).parameters)
    assert params == ["typed_covers", "episode"]


def _checks(items, verification):
    cfg = SimpleNamespace(episodes=EPISODES)
    gate = SimpleNamespace(library_mode=cfg, lookup=lambda eid, idx: None)
    run = SimpleNamespace(man=SimpleNamespace(items=items), verification=verification)
    return LibraryChecks(gate, run)


def _item(item, *, typed=()):
    return SimpleNamespace(
        item=item,
        kind="function",
        typed_covers=list(typed),
        covers=[],
        input=None,
    )


def test_the_gate_records_the_effect_in_each_functions_verification_record():
    verification: dict = {}
    checks = _checks(
        [
            _item(WRITER, typed=[_tool("wrote")]),
            _item(
                READER,
                typed=[Cover("read", "episode", runner="dialogue", action=0)],
            ),
            _item("memory.shop.cart:count"),  # no episode cover: no effect recorded
            SimpleNamespace(item="notes/shop/cart.md", kind="note"),
        ],
        verification,
    )
    checks.effects()
    assert verification == {WRITER: {"effect": "write"}, READER: {"effect": "read"}}

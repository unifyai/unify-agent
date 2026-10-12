"""A library function's effect is derived from what it reproduces and recorded in its item record: the writer
declares nothing, and the environment-function ``Effect:`` checks are unchanged."""

from __future__ import annotations

from types import SimpleNamespace

from unify.memory_v2.library_checks import LibraryChecks, derive_effect
from unify.memory_v2.procedures import Cover

WRITER, READER = "memory.shop.cart:add_all", "memory.shop.cart:total"


def _action(effect, kind="tool"):
    return SimpleNamespace(kind=kind, effect=effect, status="ok")


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


def test_a_tool_procedure_over_recorded_writes_is_a_write():
    assert derive_effect([_tool("wrote")], [], None, EPISODES) == "write"
    # one write cover among others is enough
    assert derive_effect([_tool("read"), _tool("wrote")], [], None, EPISODES) == "write"


def test_a_tool_procedure_whose_calls_declare_no_effect_is_unknown():
    assert derive_effect([_tool("undeclared")], [], None, EPISODES) == "unknown"
    assert derive_effect([_tool("missing")], [], None, EPISODES) == "unknown"


def test_a_worktree_procedure_changes_files():
    cover = Cover("wrote", "episode", runner="worktree")
    assert derive_effect([cover], [], None, EPISODES) == "write"


def test_read_only_covers_give_read():
    cell = Cover("read", "cell", index=0)
    dialogue = Cover("read", "episode", runner="dialogue", action=0)
    assert derive_effect([cell], [], None, EPISODES) == "read"
    assert derive_effect([dialogue], [], None, EPISODES) == "read"
    assert derive_effect([], [("read", 0, _action("read"))], None, EPISODES) == "read"
    assert derive_effect([], [("f", 0, _action("unknown", kind="file"))], None, EPISODES) == "read"
    assert derive_effect([], [], None, EPISODES) is None  # nothing to derive from


def test_action_covers_carry_their_recorded_effect_unless_the_input_is_a_recorded_value():
    write = [("wrote", 1, _action("write"))]
    assert derive_effect([], write, None, EPISODES) == "write"
    assert derive_effect([], write, "env", EPISODES) == "write"
    # a parser of a recorded response makes no call itself
    assert derive_effect([], write, "text", EPISODES) == "read"
    assert derive_effect([], [("u", 0, _action("unknown"))], None, EPISODES) == "unknown"


def _checks(items, lookup, verification):
    cfg = SimpleNamespace(episodes=EPISODES)
    gate = SimpleNamespace(library_mode=cfg, lookup=lookup)
    run = SimpleNamespace(man=SimpleNamespace(items=items), verification=verification)
    return LibraryChecks(gate, run)


def _item(item, *, typed=(), covers=(), form=None):
    return SimpleNamespace(
        item=item,
        kind="function",
        typed_covers=list(typed),
        covers=list(covers),
        input=form,
    )


def test_the_gate_records_the_effect_in_each_functions_verification_record():
    recorded = {("read", 0): _action("read")}
    verification: dict = {}
    checks = _checks(
        [
            _item(WRITER, typed=[_tool("wrote")]),
            _item(READER, covers=[("read", 0)]),
            SimpleNamespace(item="notes/shop/cart.md", kind="note"),
        ],
        lambda eid, idx: recorded.get((eid, idx)),
        verification,
    )
    checks.effects()
    assert verification == {WRITER: {"effect": "write"}, READER: {"effect": "read"}}


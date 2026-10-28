"""Rev 2.1, fix 11: a use that went badly reaches the writer, so a harmful item is read again.

A call the item refused (``MemoryInputError``) counted as neutral and opened no ``use_went_badly`` entry, and an
actor's message to the writer about an item counted only with an ``item:`` ref, so a harmful item was never revisited.
Now a refused call, like one that raised, opens the entry for its item (the item's trust record still counts a refusal
as neutral: it was not built for that input), and an actor's message to the writer that names an item its episode used
by the item's library name is feedback on that item. The name is resolved from the harness's records (the items the
episode used) and the item's own names, whole; a description of what an item does names nothing.
"""

from __future__ import annotations

from unify.memory import attention as at
from unify.memory import trust
from unify.memory.attention import Ledger, UseFailure
from unify.memory.lifecycle import EpisodeUse, ItemUse

from tests.memory.test_attention_ledger import Clock, Fake

ITEM = "memory.acct.users:user_id"
NOTE = "notes/acct/ids.md"
OTHER = "memory.acct.users:user_name"


def _use(**k) -> ItemUse:
    base = dict(used=False, uses=1, refused=False, errored=False, unknown=False)
    base.update(k)
    return ItemUse(**base)


class _Home(at.HomeSources):
    """The home's sources over given use facts, record entries and episode items (no store opened)."""

    def __init__(self, tmp_path, facts=(), record=(), items_of=None):
        super().__init__(tmp_path / "home")
        self._facts_given, self._record, self._items_of = list(facts), list(record), dict(items_of or {})

    def _episode_facts(self):
        return self._facts_given

    def record_entries(self):
        return self._record

    def episode_items(self, episode):
        return tuple(self._items_of.get(episode, ()))


def _entry(seq, text, *, episode="e1", refs=(), role="actor", mentions=("writer",)):
    return {"v": 1, "seq": seq, "ts": "2026-10-22T10:00:00.000Z", "author": f"{role}-e1", "kind": "post",
            "text": text, "mentions": list(mentions), "role": role, "episode": episode, "pass_id": None,
            "reply_to": None, "refs": list(refs)}  # fmt: skip


def test_a_refused_call_and_a_raised_one_each_open_an_entry_for_their_item(tmp_path):
    facts = [
        EpisodeUse("e1", "m1", "implicit", False, False, {ITEM: _use(refused=True)}),
        EpisodeUse("e2", "m1", "implicit", False, False, {OTHER: _use(errored=True)}),
        EpisodeUse("e3", "m1", "implicit", False, False, {NOTE: _use(used=True)}),
    ]
    failures = _Home(tmp_path, facts).use_failures()
    assert failures == [UseFailure("e1", ITEM, "refused"), UseFailure("e2", OTHER, "error")]
    led = Ledger(tmp_path / "home", clock=Clock())
    made = led.sync(Fake(failures=failures))
    assert sorted((e.kind, e.refs.items, e.refs.episodes) for e in made) == [
        ("use_went_badly", (ITEM,), ("e1",)),
        ("use_went_badly", (OTHER,), ("e2",)),
    ]


def test_a_refusal_stays_neutral_in_the_items_trust_record():
    verdict, why, _ = trust.judge(trust.Use("e1", "implicit", "refused"))
    assert (verdict, why) == (trust.NEUTRAL, "refused")
    assert trust.judge(trust.Use("e1", "implicit", "error"))[0] == trust.FAILED


def test_an_actor_message_naming_a_used_item_by_its_library_name_is_feedback_on_it(tmp_path):
    record = [
        _entry(5, "@writer memory.acct.users.user_id returned the wrong id for the second account"),
        _entry(6, "@writer the id helper looks wrong to me"),  # a description names nothing
        _entry(7, "@writer memory.acct.users.user_id_v2 and memory.acct.users.user_id.cache"),  # neither is the item
        _entry(8, "@writer see memory.acct.users:user_name"),  # an item its episode never used
        _entry(9, f"@writer {NOTE} is out of date", episode="e2"),
        _entry(10, "@writer memory.acct.users.user_id", role="fork"),  # not an actor
        _entry(11, "memory.acct.users.user_id", mentions=()),  # not to the writer
        _entry(12, "@writer this one", refs=["item:" + OTHER]),  # an item ref is taken as written
    ]
    items_of = {"e1": (ITEM,), "e2": (NOTE, ITEM)}
    failures = _Home(tmp_path, record=record, items_of=items_of).use_failures()
    assert failures == [
        UseFailure("e1", ITEM, "actor_reported", (), 5),
        UseFailure("e2", NOTE, "actor_reported", (), 9),
        UseFailure("e1", OTHER, "actor_reported", (), 12),
    ]


def test_named_items_matches_whole_library_names_only():
    items = (ITEM, NOTE, "capture:" + "c" * 16)
    assert at.named_items("call memory.acct.users:user_id(x)", items) == [ITEM]
    assert at.named_items("`memory.acct.users.user_id` broke.", items) == [ITEM]
    assert at.named_items("read notes/acct/ids.md first", items) == [NOTE]
    assert at.named_items("xmemory.acct.users.user_id", items) == []
    assert at.named_items("memory.acct.users.user_idx", items) == []
    assert at.named_items("user_id and ids.md", items) == []
    assert at.named_items("", items) == []

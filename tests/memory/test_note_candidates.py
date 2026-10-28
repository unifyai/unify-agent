"""Lessons, not only procedures: what an episode showed reaches the writer without a capture.

* An episode whose fork staged lessons (``notes.md``) or a note candidate (``notes/<name>.md``) and no procedure is not
  closed unseen: its entry carries them (``refs.notes``), waits for its group as a capture does, and owes their read.
* The fork's brief lets it keep a note candidate (what someone stated, a correction, an outcome, a caution), with its
  scope, and no ``procedure.py``; it posts to the writer only with something to point at or to correct.
* A message to the writer that carries nothing of its own makes no pass: it goes along with the next pass.

Real components throughout (the ledger, the staging reader, the grouping, the fork's parser, the library's readers);
only the records a world would make are written by the test.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from unify.agents import run_record as rr
from unify.memory import attention as at
from unify.memory import attention_tools as tools
from unify.memory import capture_groups as cg
from unify.memory.attention import Ledger, SettledFork
from unify.memory.integration import fork, fork_worker

from tests.memory.test_attention_ledger import Clock, Fake

NOTE_CANDIDATE = (
    "---\nkind: preference\napplies_to: monthly reports for the planning team\nstated_by: the planning lead\n---\n"
    'Send reports as tab-separated files. From the reply: "please always send it as a .tsv".\n'
)


def _stage(home: Path, name: str, files: dict[str, str]) -> None:
    """A settled fork's staging dir, as the fork worker leaves it (its files, then ``fork.json`` last)."""
    d = home / "memory-state" / "memory-staging" / name
    for rel, text in files.items():
        (d / rel).parent.mkdir(parents=True, exist_ok=True)
        (d / rel).write_text(text)
    (d / "fork.json").write_text(json.dumps({"episode": name, "status": "ok", "files": sorted(files)}))


class Staged(Fake):
    """The ledger's sources with the settled forks read from the home's real staging dirs (:class:`at.HomeSources`)."""

    def __init__(self, home: Path, **kw) -> None:
        super().__init__(**kw)
        self.real = at.HomeSources(home)

    def settled_forks(self):
        return self.real.settled_forks()


# --- the fork keeps a note candidate, with no procedure ----------------------------------------------------------------


def test_the_fork_brief_offers_a_note_candidate_with_no_procedure_and_its_parser_keeps_it():
    text = fork.fork_instruction()
    assert fork.NOTE_BULLET in text and "It needs no `procedure.py`." in text
    assert "`applies_to:`" in text and "`stated_by:`" in text and "`kind:`" in text
    # no notes.md is written to say nothing is worth keeping: an empty episode leaves nothing for the writer
    assert "write `notes.md` saying so" not in text and "leave `notes.md` out" in text
    # the brief's own example parses, its note candidate included, as a staged note
    blocks, _record = fork_worker.parse_reply(fork.FORMAT_EXAMPLE)
    (note,) = [b for b in blocks if b["path"] == "notes/header_line.md"]
    assert note["status"] == "parsed" and note["kind"] == "note" and note["content"].startswith("---\nkind: caution\napplies_to: ")
    assert fork_worker.allowed("notes/header_line.md") is None
    assert fork_worker.allowed("notes/a/b.md") is not None and fork_worker.allowed("notes/x.py") is not None
    # the episode's summary keeps its own name: a note is never staged as notes/summary.md
    assert fork_worker.allowed("notes/summary.md") is not None and "is not `summary`" in fork.NOTE_BULLET


def test_the_fork_mentions_the_writer_only_with_something_to_point_at_or_to_correct():
    (line,) = [rr.mention_line(m["who"], m["when"]) for m in rr.SHOULD_MENTION["fork"]]
    assert line == (
        "Mention @writer to point at a `procedure.py`, candidate or note you wrote and say what you were unsure of, "
        "or to correct a served item, naming it as `item:<id>`; only then, never to say there is nothing to keep."
    )


# --- a capture-less episode with lessons reaches the writer --------------------------------------------------------------


def test_an_episode_with_lessons_and_no_capture_is_handed_for_them_and_one_without_is_closed(tmp_path):
    home = tmp_path / "home"
    clock = Clock()
    led = Ledger(home, clock=clock)
    _stage(home, "e1", {"summary.md": "## Asked\nSend the report.\n", "notes.md": "- The requester wants a .tsv file.\n",
                        "notes/tsv.md": NOTE_CANDIDATE, "broke.md": "none\n"})
    _stage(home, "e2", {"summary.md": "## Asked\nSay hello.\n", "notes.md": "\n  \n", "broke.md": "none\n"})
    made = {e.refs.episodes[0]: e for e in led.sync(Staged(home))}
    assert made["e1"].refs.captures == () and made["e1"].refs.notes == ("e1/notes.md", "e1/notes/tsv.md")
    assert made["e2"].refs.notes == ()  # a blank notes.md is no lesson
    # its material: the summary and the lessons, both owed a read
    parts = {p.name: p for p in at.material(made["e1"])}
    assert parts["notes"].required and parts["notes"].paths == ("/inputs/staging/e1/notes.md",
                                                                "/inputs/staging/e1/notes/tsv.md")

    def grouping(entries):
        return cg.groups(entries, cg.HomeProfiles(led.state_dir), now=clock())

    handed = led.hand("p1", 4, grouping=grouping)
    assert [e.entry for e in handed] == [made["e1"].entry]
    row = led.group_of("p1")
    assert row["due"] == cg.ENDED and row["group"] is None  # related to nothing: due once its request has ended
    gone = led.get(made["e2"].entry)
    assert gone.state == "closed" and gone.reason_ref == cg.NO_CAPTURE  # nothing to keep: closed, no pass
    # read only when its lessons were opened, then the writer lands a note from it with no capture behind it
    summary_only = at.PassReads(frozenset({"/inputs/staging/e1/summary.md"}))
    assert led.verify_reads("p1", summary_only) == set()
    both = at.PassReads(frozenset({"/inputs/staging/e1/summary.md", "/inputs/staging/e1/notes/tsv.md"}))
    led.decide("p1", [made["e1"].entry], "note", ["notes/reports/tsv.md"], "a stated preference", reads=both,
               decisions_path=tmp_path / "decisions.jsonl")  # fmt: skip
    done = led.get(made["e1"].entry)
    assert done.state == "closed" and done.decision == "note" and "notes" in done.parts_read


def test_a_note_entry_groups_with_a_capture_through_a_shared_item_and_the_pass_is_told_it_holds_entries(tmp_path):
    clock = Clock()
    led = Ledger(tmp_path / "home", clock=clock)
    cap = {"v": 2, "capture_id": "c" * 16, "episode": "e1", "own_check": {"ran": False}, "outcome": "none"}
    led.sync(Fake(forks=[SettledFork("e1", "e1", True, True, None),
                         SettledFork("e2", "e2", True, False, None, ("e2/notes/tsv.md",))],
                  captures={"e1": cap}))  # fmt: skip
    shared = frozenset({"memory.reports.files:write_tsv"})

    def profile(e):
        return cg.Profile(e.entry, tuple(e.refs.captures), items=shared)

    handed = led.hand("p1", 4, grouping=lambda entries: cg.groups(entries, profile, now=clock()))
    assert len(handed) == 2 and led.group_of("p1")["due"] == "formed"
    assert led.group_of("p1")["reasons"]["items"] == 1
    doc = cg.group_doc(cg.Group(id="x", open=handed), profile)
    by = {m["entry"]: m for m in doc["members"]}
    note = next(e for e in handed if not e.refs.captures)
    assert by[note.entry]["notes"] == ["/inputs/staging/e2/notes/tsv.md"] and by[note.entry]["captures"] == []
    text = tools.first_message(SimpleNamespace(pass_id="p1", group={"related_closed": [], "library_items": []}),
                               SimpleNamespace(attention=handed), "900")
    assert text.startswith("Pass p1: a group of 2 related entries needs your attention")


def test_a_note_entry_a_group_pass_cannot_finish_is_given_up_like_a_capture(tmp_path):
    clock = Clock()
    led = Ledger(tmp_path / "home", clock=clock)
    led.sync(Fake(forks=[SettledFork("e1", "e1", True, False, None, ("e1/notes.md",))]))

    def grouping(entries):
        return cg.groups(entries, lambda e: cg.Profile(e.entry, ()), now=clock())

    for n in range(1, 4):
        assert led.hand(f"p{n}", 4, grouping=grouping)
        led.return_unread(f"p{n}")
    (e,) = led.entries().values()
    assert e.state == "closed" and e.reason_ref == at.GIVEN_UP  # never unreadable, as a capture of a group


# --- a message with nothing of its own makes no pass ---------------------------------------------------------------------


def _post(seq, role, *, refs=(), reply_to=None, episode="e1"):
    return {"seq": seq, "author": f"{role}-3f9a", "role": role, "kind": "post", "mentions": ["writer"],
            "episode": episode, "refs": list(refs), "reply_to": reply_to, "ts": f"2026-10-20T00:00:0{seq % 10}.000Z"}


def test_a_message_with_nothing_of_its_own_makes_no_pass_and_goes_along_with_the_next(tmp_path):
    clock = Clock()
    led = Ledger(tmp_path / "home", clock=clock)
    (quiet,) = led.sync(Fake(record=[_post(1, "fork")]))
    assert quiet.kind == "message" and quiet.carries is False and not at.wants_pass(quiet)
    assert led.hand("p1", 4) == []  # alone it makes no pass
    assert led.get(quiet.entry).state == "open"
    # a post with something of its own: an item named by a typed ref, an answer to the writer, an actor's report
    record = [_post(1, "fork"), _post(2, "fork", refs=["item:memory.reports.files:write_tsv"]),
              {"seq": 3, "author": "writer", "role": "writer", "mentions": ["fork-3f9a"]},
              _post(4, "fork", reply_to=3), _post(5, "actor"), _post(6, "harness")]
    made = {e.refs.record_seq: e for e in led.sync(Fake(record=record))}
    assert {s: e.carries for s, e in made.items()} == {2: True, 4: True, 5: True, 6: False}
    handed = led.hand("p2", 4)
    assert {e.refs.record_seq for e in handed} == {1, 2, 4, 5, 6}
    along = {r["entry"] for r in led.rows() if r.get("op") == "hand" and r.get("pass_id") == "p2"
             and r.get("by") == "with_pass"}  # fmt: skip
    assert along == {quiet.entry, made[6].entry}
    assert at.carries_content(_post(7, "fork", refs=["e1/action:3"])) is False  # its own episode's parts: its entry's

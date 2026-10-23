"""One way to reach the library, the same for every agent allowed it, with the harness resolving where things are.

* The actor's ``tools.read`` opens a library file by either name agents are shown for the library (``/memory-library``,
  the actor's mount, and ``/memory``, as the librarian and the writer name it), read-only; a path outside every root
  is refused with the call that opens a library entry; a run with no library refuses exactly as before.
* A call the librarian suggests must go through ``memory`` (never a path read or another tool).
* The actor's teammates ask the librarian the same way the actor does (``memory.ask``).
* The actor is never told to mention ``@librarian`` (it asks with ``memory.ask``) and is never offered ``@fork`` as an
  address while its episode runs (the fork starts after it).
* In a writer pass, ``memory.get('capture:<id>')`` opens a capture handed to the pass as it opens an exported one.
"""

from __future__ import annotations

from types import SimpleNamespace

from unify.actor import tool_handlers as H
from unify.common import tool_registry as R
from unify.memory import librarian


def _ctx():
    return SimpleNamespace(roots={}, state=SimpleNamespace(policy=None))


def test_the_actor_reads_a_library_file_by_either_of_its_names(tmp_path, monkeypatch):
    lib = tmp_path / "copy"
    (lib / "notes" / "files").mkdir(parents=True)
    (lib / "notes" / "files" / "sort-rows.md").write_text("Sort by the integer id.\n")
    monkeypatch.setattr(H, "_library_root", lambda: str(lib))
    for path in ("/memory/notes/files/sort-rows.md", "/memory-library/notes/files/sort-rows.md"):
        got = H.read(_ctx(), {"path": path})
        assert got.status == "ok" and "Sort by the integer id." in got.text, (path, got.text)
    listing = H.read(_ctx(), {"path": "/memory/notes"})
    assert listing.status == "ok" and "files" in listing.text
    missing = H.read(_ctx(), {"path": "/memory/notes/none.md"})
    assert missing.status == "refused" and missing.form == "not_found"
    outside = H.read(_ctx(), {"path": "/etc/hostname"})
    assert outside.status == "refused" and outside.form == "outside_roots"
    assert "memory.get(<its id>)" in outside.text and "/memory-library" in outside.text
    escape = H.read(_ctx(), {"path": "/memory/../etc/hostname"})
    assert escape.status == "refused"


def test_without_a_library_the_refusal_is_as_before(monkeypatch):
    monkeypatch.setattr(H, "_library_root", lambda: None)
    got = H.read(_ctx(), {"path": "/memory/notes/x.md"})
    assert got.status == "refused" and got.form == "outside_roots"
    assert "memory.get" not in got.text and got.text.endswith("is outside /workspace")


def test_a_suggested_call_goes_through_memory():
    ids = ["notes/files/sort-rows.md", "memory.files.rows:sort_rows"]

    def check(call):
        return librarian.check_answer({"fits": [{"id": ids[1], "call": call, "why": "it sorts rows"}]}, ids)

    assert check("memory.get('memory.files.rows:sort_rows')(memory.request())")[0] is not None
    assert check("print(memory.get('notes/files/sort-rows.md').text)")[0] is not None
    for bad in ("tools.read('/memory/notes/files/sort-rows.md')", "open('/memory/notes/x.md').read()", "run(state)"):
        got, why = check(bad)
        assert got is None and "memory.get(" in why, (bad, why)


def test_teammates_ask_the_librarian_the_same_way_as_the_actor():
    for agent in ("actor", "teammate"):
        allow = R.allow_list(agent, H.actor_flags(agent, librarian=True))
        assert "ask_library" in allow, agent
    assert "ask_library" not in R.allow_list("teammate", H.actor_flags("teammate", librarian=False))


def test_the_actor_is_not_told_to_mention_the_librarian_or_offered_its_fork(tmp_path, monkeypatch):
    from unify.agents import run_record as rr

    assert all(m["who"] != "librarian" for m in rr.SHOULD_MENTION["actor"])
    monkeypatch.setattr(rr, "run_has_memory", lambda *a, **k: True)
    rec = rr.join_run_record(tmp_path)
    rec.join("actor-1a2b", "actor")
    rec.join("writer", "writer", state="idle")
    addresses = rr.RunRecordView(rec, "actor-1a2b", "actor").addresses()
    assert "@fork" not in addresses and "@writer" in addresses


def test_a_writer_pass_opens_a_capture_handed_to_it(tmp_path, monkeypatch):
    from unify.memory import library_helper as lh

    handed = tmp_path / "inputs" / "captures"
    cid = "a" * 16
    (handed / cid).mkdir(parents=True)
    (handed / cid / "procedure.py").write_text("def run(value):\n    return value\n")
    monkeypatch.setattr(lh, "HANDED_CAPTURES", str(handed))
    assert lh.CAPTURE_PREFIX + cid in lh._capture_ids()
    assert lh._capture_dir(cid) == str(handed / cid)

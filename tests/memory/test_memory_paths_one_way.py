"""One way to reach the library, the same for every agent allowed it, with the harness resolving where things are.

In real boxes (the actor, and a teammate it starts; ``UNIFY_MEMORY=on`` and ``in_episode``; ``real_request_box``):

* ``tools.read`` and ``tools.grep`` open the library copy the box mounts, by either name the texts give it
  (``/memory-library``, and ``/memory`` as the librarian and the writer name it), and read what a cell's own
  ``open()`` reads there; ``memory.get`` takes the same paths. Under in_episode both names are the copy served now,
  and a cell's own file calls find it behind the link the guide's location line names.
* Every path and handle the agent is shown (its memory section, first message, cards, ``memory.ask``'s answer, the
  refusals of read and grep) opens with its own ``tools.read`` or ``memory.get``.
* A teammate's ``tools.request()`` and ``memory.request()`` are the actor's request, never its own task.
* A path outside every root is refused with the roots the agent has, and nothing outside them is read.

Also here (no box needed): a call the librarian suggests goes through ``memory``; teammates ask the librarian as the
actor does; a writer pass opens a capture handed to it. What the record tells an agent inside a request is in
``tests/agents/test_record_texts_inside_a_request.py``.
"""

from __future__ import annotations

import pytest

from tests.actor.code_act.core_world import core_world  # noqa: F401 (fixture)
from tests.actor.code_act.sandbox_world import needs_bwrap, world  # noqa: F401 (fixture)
from tests.memory import real_request_box as B
from tests.memory.test_tool_channel_box import needs_box
from unify.actor import tool_handlers as H
from unify.common import tool_registry as R
from unify.memory import librarian

#: Each real session runs once per file run, by (role, mode): several tests read one session.
_SESSIONS: dict = {}


def _session(core_world, monkeypatch, role: str, mode: str) -> dict:  # noqa: F811
    key = (role, mode)
    if key not in _SESSIONS:
        _SESSIONS[key] = B.run_session(core_world, monkeypatch, role, mode)
    state = _SESSIONS[key]
    assert state.get("probe"), (state.get("error"), state.get("kinds"), state.get("probe output"))
    return state


ROLES_MODES = [(r, m) for r in ("actor", "teammate") for m in ("on", "in_episode")]


@needs_box
@needs_bwrap
@pytest.mark.timeout(1500)
@pytest.mark.parametrize("role, mode", ROLES_MODES)
def test_read_and_grep_open_the_copy_the_box_mounts_by_every_name(role, mode, core_world, monkeypatch):  # noqa: F811
    probe = _session(core_world, monkeypatch, role, mode)["probe"]
    for path, (status, text) in probe["reads"].items():
        assert status == "ok", (path, status, text)
    note = probe["reads"][f"/memory-library/{B.NOTE_ID}"][1]
    assert "Dates are YYYY-MM-DD." in note
    assert probe["reads"][f"/memory/{B.NOTE_ID}"][1] == note
    # the same bytes a cell's own open() reads where the guide says the copy is
    assert probe["own_open"][0] == "ok" and "Dates are YYYY-MM-DD." in probe["own_open"][1], probe["own_open"]
    # the library's root lists the copy itself (INDEX.md, memory/, notes/), in either mode
    listing = probe["reads"]["/memory-library"][1]
    assert "INDEX.md" in listing and "notes/" in listing and "memory/" in listing, listing
    assert probe["reads"]["/memory"][1] == listing
    if mode == "in_episode":
        assert probe["reads"][f"/memory-library/current/{B.NOTE_ID}"][1] == note
        # the package's own files are in the episode's pin, inside the mount: that path opens too
        assert probe["package root"].startswith("/memory-library/.pins/"), probe["package root"]
    else:
        assert probe["package root"] == "/memory-library", probe["package root"]
    for path, (status, text) in probe["greps"].items():
        assert status == "ok" and f"/memory-library/{B.LONG_NOTE_ID}:" in text, (path, status, text)
    for path, (status, got) in probe["gets"].items():
        assert status == "ok" and got == B.NOTE_ID, (path, status, got)


@needs_box
@needs_bwrap
@pytest.mark.timeout(1500)
@pytest.mark.parametrize("role, mode", ROLES_MODES)
def test_every_printed_path_opens(role, mode, core_world, monkeypatch):  # noqa: F811
    state = _session(core_world, monkeypatch, role, mode)
    printed = state["printed"]
    shown = state["shown texts"]
    # the texts print what the agent is told to open: the location line, the note's rest, the roots a refusal names,
    # the items the cards and the answer name
    guide_paths = printed["memory section"][0]
    assert "/memory-library" in guide_paths, shown["memory section"][-600:]
    assert f"/memory-library/{B.LONG_NOTE_ID}" in printed["long note card"][0], shown["long note card"]
    assert B.FUNCTION_ID in printed["ask answer"][1], shown["ask answer"]
    assert {"/workspace", "/memory-library"} <= set(printed["refusal of read /etc/hostname"][0])
    if role == "actor":
        assert B.FUNCTION_ID in printed["first message"][1], shown["first message"][-1500:]
    if mode == "in_episode":
        assert "/memory-library/current" in guide_paths, shown["memory section"][-600:]
    opened = state["opened"]
    assert opened, state.get("open output")
    failed = {k: v for k, v in opened.items() if v != "ok"}
    assert not failed, failed


@needs_box
@needs_bwrap
@pytest.mark.timeout(1500)
@pytest.mark.parametrize("role, mode", ROLES_MODES)
def test_a_path_outside_every_root_is_refused_with_the_roots_the_agent_has(role, mode, core_world, monkeypatch):  # noqa: F811
    probe = _session(core_world, monkeypatch, role, mode)["probe"]
    by_path = {(tool, asked): result for tool, asked, result in probe["refusals"]}
    for key in (("read", "/etc/hostname"), ("read", "/memory-library/../etc/hostname"), ("grep", "/etc")):
        form, text = by_path[key]
        assert form == "refused:outside_roots", (key, form, text)
        assert "/workspace and the library (/memory-library, read-only)" in text and "memory.get(" in text, text
    for key in (("read", "/workspace/none.txt"), ("grep", "/workspace/none")):
        form, text = by_path[key]
        assert form == "refused:not_found", (key, form, text)
        assert "/inputs" not in text and "attention" not in text and "memory" not in text, text
    for key in (("read", "/memory-library/notes/none.md"), ("read", "/memory/notes/none.md"),
                ("grep", "/memory-library/none")):
        form, text = by_path[key]
        # a missing library path names the calls that list the library's entries and open one
        assert form == "refused:not_in_library", (key, form, text)
        assert "memory.items()" in text and "memory.get(" in text, text
        # the actor's and a teammate's own text: never a writer's input or another role's root
        assert "/inputs" not in text and "attention" not in text, text


@needs_box
@needs_bwrap
@pytest.mark.timeout(1500)
@pytest.mark.parametrize("mode", ["on", "in_episode"])
def test_a_teammate_works_on_the_actors_request_not_its_own_task(mode, core_world, monkeypatch):  # noqa: F811
    state = _session(core_world, monkeypatch, "teammate", mode)
    probe = state["probe"]
    assert probe["request"] == ["ok", B.REQUEST_TEXT], probe["request"]
    assert probe["memory_request"] == ["ok", B.REQUEST_TEXT], probe["memory_request"]
    # its task stays its own first message
    assert B.TASK_TEXT in state["first"]["first message"]


@needs_box
@needs_bwrap
@pytest.mark.timeout(1500)
@pytest.mark.parametrize("role, mode", ROLES_MODES)
def test_help_memory_as_the_first_cell_shows_the_package(role, mode, core_world, monkeypatch):  # noqa: F811
    """The agent's first cell, and the request's librarian's first cell, run ``help(memory)`` before any other use of
    ``memory`` (the writer's and a helper's: ``test_help_memory_first_cell``)."""
    state = _session(core_world, monkeypatch, role, mode)
    B.assert_package_help(state["help"])
    B.assert_package_help(state.get("librarian help", ""))


# ── without a box ───────────────────────────────────────────────────────────────────────────────────────────────


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


def test_a_writer_pass_opens_a_capture_handed_to_it(tmp_path, monkeypatch):
    from unify.memory import library_helper as lh

    handed = tmp_path / "inputs" / "captures"
    cid = "a" * 16
    (handed / cid).mkdir(parents=True)
    (handed / cid / "procedure.py").write_text("def run(value):\n    return value\n")
    monkeypatch.setattr(lh, "HANDED_CAPTURES", str(handed))
    assert lh.CAPTURE_PREFIX + cid in lh._capture_ids()
    assert lh._capture_dir(cid) == str(handed / cid)


def test_natural_forms_resolve(tmp_path):
    """The forms of one note's path an agent may write (its id, its ``topic/slug``, and its path under
    ``/memory-library`` or ``/memory``) open the same entry: ``memory.get`` takes all four (one package for every
    role); the actor's and a teammate's ``tools.read`` takes both box names, the librarian's its ``/memory`` path, and
    the writer's ``tools.read`` its relative forms too, each giving the note's bytes. A path that leaves the roots is
    refused for each, and nothing outside is read."""
    from types import SimpleNamespace

    from tests.memory.test_capture_examples import _cell, _copy
    from unify.memory import safe_paths, views
    from unify.memory.writer_pass import _natural_path

    root, _ = _copy(tmp_path)
    note = "notes/text/dates.md"
    text = (root / note).read_text()
    forms = [note, "text/dates", f"{H.LIBRARY}/{note}", f"{H.MEMORY_ALIAS}/{note}"]
    got = _cell(root, f"print(json.dumps({{'ids': [memory.get(f).id for f in {forms!r}], "
                      f"'outside': err(memory.get, {H.LIBRARY + '/../etc/hostname'!r})}}))\n")  # fmt: skip
    assert got["ids"] == [note] * 4
    assert got["outside"][0] == "NotFound", got["outside"]

    # the actor's and a teammate's read: either box name is the copy the box mounts
    view = H.LibraryView(mount=root, copy=root, link="", mounts=())
    places = {H._library_place(view, f) for f in forms[2:]}
    assert len(places) == 1
    shown, host_root, _ = places.pop()
    assert views.read_entry(shown, {H.LIBRARY: str(host_root)})[0] == text
    with pytest.raises(safe_paths.Outside):
        H._library_place(view, f"{H.LIBRARY}/../etc/hostname")

    # the librarian's read: the library under /memory, and nothing outside it
    ctx = SimpleNamespace(state=SimpleNamespace(copy=root))
    assert librarian.read(ctx, {"path": forms[3]}).text == text
    refused = librarian.read(ctx, {"path": "/memory/../etc/hostname"})
    assert refused.status == "refused" and "is outside the library, /memory" in refused.text

    # the writer's read: what it naturally writes resolves to the one /memory path
    roots = {"/memory": root, "/inputs": tmp_path / "inputs"}
    (tmp_path / "inputs").mkdir()
    for form in (note, f"memory/{note}", forms[3]):
        resolved = _natural_path(form, roots)
        assert resolved == forms[3], (form, resolved)
        assert views.read_entry(resolved, {"/memory": str(root)})[0] == text
    assert _natural_path("../etc/hostname", roots) == "../etc/hostname"
    with pytest.raises(safe_paths.Outside):
        views.read_entry("/memory/../etc/hostname", {"/memory": str(root)})


@pytest.mark.asyncio
async def test_helper_ask(tmp_path, monkeypatch):
    """An actor-started helper (a teammate) asks the request's librarian as the actor does: its tools offer the ask only
    in a run with a librarian, its question reaches the librarian and comes back answered, and once the request's
    librarian sessions are at their bound its question is refused, naming where to send it instead."""
    import json
    from types import SimpleNamespace

    from tests.memory import test_librarian_answer_once as A
    from unify.memory.integration import request as RQ

    with_librarian = R.allow_list("teammate", H.actor_flags("teammate", librarian=True))
    without = R.allow_list("teammate", H.actor_flags("teammate", librarian=False))
    assert "ask_library" in with_librarian and "ask_library" not in without
    assert "ask_library" not in R.list_text(without, "teammate", memory=False)

    turn = A.Turn([A.FITS])
    session = A._session(tmp_path, turn)
    ctx = A._ctx(session)
    ctx.agent = "teammate"
    got = await H.ask_library(ctx, {"question": A.REQUEST})
    assert got.status == "ok", got
    assert [f["id"] for f in json.loads(got.text)["fits"]] == [A.TOTAL] and len(turn.calls) == 1

    # the bound: the actor's and its first teammates' sessions are open; a later teammate's question is refused
    full = SimpleNamespace(copy_root=A._copy(tmp_path / "bound"), episode_id=None, pin=None, in_episode=None,
                           librarian_sessions=[SimpleNamespace(mode="ask")] * librarian.MAX_ASK_SESSIONS)  # fmt: skip
    monkeypatch.setattr(RQ, "current", lambda: full)
    late = SimpleNamespace(agent="teammate", state=H.ActorRequestState(
        open_librarian=librarian.opener(turn_factory=lambda: A.Turn([])), request_text=A.REQUEST))
    refused = await H.ask_library(late, {"question": A.REQUEST})
    assert refused.status == "refused", refused
    assert "@actor" in refused.text and f"{librarian.MAX_ASK_SESSIONS} librarian sessions" in refused.text

"""Every agent on the run record (UNIFY_AGENTS_RUN_RECORD): the four comms tools in every role, ``who``, the common
section's bytes, the prompt-size report, no box binding the record, the cached prefix across rosters, and the
exchanges the mesh carries (the redesign's A1 tests 5, 6, 7, 10 and 11, and its acceptance exchanges).

Keyless: every model is scripted; nothing here opens a socket off this machine.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

from unify.agents import run_record as rr
from unify.agents.options import Options
from unify.common import tool_registry as R
from unify.settings import SETTINGS
from tests.memory.test_helpers import helper_boxes  # noqa: F401,F811 (fixture)
from tests.agents.roster import read_roster

RECORD_TOOLS = {"send", "messages", "wait", "who"}
FAST = Options(
    max_live=16,
    max_total=64,
    max_entries=None,
    max_record_bytes=rr.MAX_RECORD_BYTES,
    min_post_interval_s=0.0,
)


@pytest.fixture
def run_record_on(monkeypatch):
    return SETTINGS


# --- test 5: every role has the four comms tools, and who shows the roster --------------------------------------------


def test_who_shows_the_roster_and_whom_this_agent_should_mention(tmp_path):
    from unify.agents import run_record_tools as T

    home = tmp_path / "home"
    rec = rr.join_run_record(home, options=FAST)
    rec.join("actor-3f9a", "actor", episode="ep-1")
    rec.join("writer", "writer", state="idle")
    rec.join("librarian-3f9a", "librarian", episode="ep-1")
    rec.join("h1", "helper", spawner="actor-3f9a")
    rec.leave("h1")
    text = T.who_text(rr.RunRecordView(rec, "actor-3f9a", "actor"))
    assert (
        "- actor-3f9a (actor, running, episode ep-1): is doing the current request"
        in text
    )
    assert (
        "- writer (writer, idle)" in text
        and "- librarian-3f9a (librarian, running, episode ep-1)" in text
    )
    assert "h1 (" not in text and "(1 agents of this run have ended)" in text
    assert "Mention @writer when a served item was wrong" in text
    # rev 2: the librarian is a role the run seeds; until it does, nobody is told to mention it
    assert "Mention @librarian" not in text and "@librarian" not in text.splitlines()[-1]
    rr.seed_role(home, "librarian", by="test")
    text = T.who_text(rr.RunRecordView(rec, "actor-3f9a", "actor"))
    assert "Mention @librarian when you want to know whether stored items fit" in text
    assert "@librarian" in text.splitlines()[-1]


# --- test 6 and 7: the common section, and the prompt-size report -----------------------------------------------------


def test_the_common_section_is_the_same_bytes_in_every_roles_lines(run_record_on):
    for role in ("actor", "fork", "librarian", "writer", "helper"):
        lines = rr.prompt_lines(
            f"{role}-x" if role != "writer" else "writer", role, "writer"
        )
        assert lines.count(R.COMMON_SECTION) == 1 and lines.startswith(R.COMMON_SECTION)
    assert rr.COMMON_SECTION is R.COMMON_SECTION
    assert R.help_topics()["record"][1] == R.COMMON_SECTION


def test_the_actors_and_a_teammates_first_message_carry_the_common_section(
    run_record_on, tmp_path, monkeypatch
):
    from unify.agents import binding

    monkeypatch.setattr(binding, "_run_home", lambda: tmp_path / "home")
    main = binding.bind_for_act(request="Do the thing.", user_reads=False)
    section = main.prompt_section()
    assert (
        section.count(R.COMMON_SECTION) == 1
        and f"you are `{main.name}` (actor)" in section
    )
    assert main.name.startswith("actor-") and main.is_main and main.run_role == "actor"
    rows = rr.read_entries(tmp_path / "home")
    assert rows[0]["author"] == "user" and rows[0]["mentions"] == [main.name]
    main.leave()
    assert {r["name"]: r["state"] for r in read_roster(tmp_path / "home")["agents"]}[
        main.name
    ] == "ended"


def _tokens(text: str) -> int:
    from unify.memory.catalogue import estimate_tokens

    return int(estimate_tokens(text))


# --- test 10: no box binds the record --------------------------------------------------------------------------------


def test_no_helper_box_bind_reaches_the_run_record(tmp_path, helper_boxes):
    """A1 test 10: a helper on the run record runs a cell; every folder its box binds (read-only or writable) is
    outside ``memory-state/agents``, and none holds it."""
    from tests.memory.test_helper_messaging import Steps, _done
    from tests.memory.test_helpers import _ctx, _msg, _tc
    from unify.memory import helpers, messaging

    home = tmp_path / "home"
    pr = messaging.PassRecord(
        root="writer",
        deadline=time.monotonic() + 600,
        pass_id="ab" * 8,
        run_home=str(home),
    )
    model = Steps(
        {"t": [_msg(_tc("x", "execute_code", {"code": "print(6)"})), _done()]}
    )
    ctx = _ctx(tmp_path, model, pass_record=pr)

    async def go():
        out = await helpers.handle({"task": "t", "episodes": ["e1"]}, ctx)
        await helpers.result(ctx, out.split()[1])
        await messaging.close(pr, "done")

    asyncio.run(go())
    assert helper_boxes, "the helper ran no cell"
    agents = rr.agents_dir(home).resolve()
    assert rr.record_path(home).is_file()
    for call in helper_boxes:
        for kind in ("ro", "rw", "late_ro"):
            for src in call.get(kind) or {}:
                path = Path(src).resolve()
                assert not (
                    agents == path
                    or agents.is_relative_to(path)
                    or path.is_relative_to(agents)
                ), (kind, path)


@pytest.mark.skipif(__import__("shutil").which("bwrap") is None, reason="bubblewrap required")
def test_a_helper_box_that_tries_to_open_the_run_record_is_refused(tmp_path):
    """A1 test 10, in a real box: a helper on the run record runs a cell that opens the record, its delivery log and
    its roster by their host paths and lists the record's folder; every attempt fails inside the box, and the record
    is unchanged by the cell."""
    from tests.memory.test_helpers import Script, _ctx, _msg, _tc
    from unify.memory import helpers, messaging

    home = tmp_path / "home"
    pr = messaging.PassRecord(
        root="writer",
        deadline=time.monotonic() + 600,
        pass_id="cd" * 8,
        run_home=str(home),
    )
    targets = [rr.record_path(home), rr.deliveries_path(home), rr.agents_dir(home) / rr.ROSTER_FILE]
    code = (
        "import os\n"
        f"for p in {[str(t) for t in targets]!r}:\n"
        "    try:\n"
        "        open(p).read(1)\n"
        "        print('OPENED', p)\n"
        "    except OSError as e:\n"
        "        print('REFUSED', type(e).__name__)\n"
        "    try:\n"
        "        open(p, 'a').write('x')\n"
        "        print('WROTE', p)\n"
        "    except OSError as e:\n"
        "        print('REFUSED', type(e).__name__)\n"
        "try:\n"
        f"    print('LISTED', os.listdir({str(rr.agents_dir(home))!r}))\n"
        "except OSError as e:\n"
        "    print('REFUSED', type(e).__name__)\n"
    )
    script = Script(
        {
            "t": [
                _msg(_tc("x", "execute_code", {"code": code})),
                _msg(_tc("d", "done", {"summary": "s", "claims": []})),
            ],
        },
    )
    ctx = _ctx(tmp_path, script, pass_record=pr)

    async def go():
        out = await helpers.handle({"task": "t", "episodes": ["e1"]}, ctx)
        await helpers.result(ctx, out.split()[1])
        await messaging.close(pr, "done")

    before = rr.record_path(home).read_bytes() if rr.record_path(home).exists() else b""
    asyncio.run(go())
    assert rr.record_path(home).is_file()
    out = script.outputs["x"]
    assert out.count("REFUSED") == 2 * len(targets) + 1, out
    assert "OPENED" not in out and "WROTE" not in out and "LISTED" not in out, out
    # the cell wrote nothing to the record (the helper's own entries came through its tools, after the cell)
    assert rr.record_path(home).read_bytes().startswith(before)
    assert not any(row.get("text") == "x" for row in rr.read_entries(home))


def test_the_actors_sandbox_never_shows_the_run_record(
    run_record_on, tmp_path, monkeypatch
):
    """The actor's cells see the state directory only through its read-only views; the run record is none of them."""
    from unify import sandbox
    from unify.agents import binding

    monkeypatch.setenv("UNIFY_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(binding, "_run_home", lambda: tmp_path / "home")
    b = binding.bind_for_act(request="r", user_reads=False)
    assert binding.current_root_pool() is b.pool
    assert sandbox._current_run_records() is None
    policy = sandbox.build_policy(fresh=True)
    agents = rr.agents_dir(tmp_path / "home").resolve()
    for p in policy.readonly_state:
        assert not Path(p).resolve().is_relative_to(
            agents
        ) and not agents.is_relative_to(Path(p).resolve()), p


# --- the exchanges the mesh carries, all on one record ----------------------------------------------------------------


def test_a_writer_delegates_to_a_helper_and_the_helper_answers_on_the_run_record(
    tmp_path, helper_boxes
):
    """Acceptance: a writer -> helper delegation and a helper -> writer reply, on the run record."""
    from tests.memory.test_helper_messaging import Steps, _done
    from tests.memory.test_helpers import _ctx, _msg, _tc
    from unify.memory import helpers, messaging

    home = tmp_path / "home"
    pr = messaging.PassRecord(
        root="writer",
        deadline=time.monotonic() + 600,
        pass_id="ab" * 8,
        run_home=str(home),
    )
    model = Steps(
        {
            "t": [
                _msg(_tc("s", "send", {"text": "e1 computes w * h; source e1/cell:0"})),
                _done(summary="ok"),
            ]
        }
    )
    ctx = _ctx(tmp_path, model, pass_record=pr)

    async def go():
        out = await helpers.handle({"task": "t", "episodes": ["e1"]}, ctx)
        name = out.split()[1]
        info: dict = {}
        got = await messaging.call(pr, "writer", "wait", {}, info)
        assert "e1 computes w * h" in got and info["outcome"] == "message"
        await helpers.result(ctx, name)
        assert await messaging.close(pr, "the writer finished") == {name: "ended"}
        return name

    name = asyncio.run(go())
    assert name == "call-1"  # the first call- name no agent of the run had
    rows = rr.read_entries(home)
    asked = next(r for r in rows if r["author"] == "writer" and r["mentions"] == [name])
    answered = next(r for r in rows if r["author"] == name and r["kind"] == "post")
    assert asked["role"] == "writer" and asked["pass_id"] == "ab" * 8
    assert (
        answered["mentions"] == ["writer"]
        and answered["role"] == "helper"
        and answered["refs"] == ["e1/cell:0"]
    )
    # the helper's first message carried the common section after its task
    first = model.seen["t"][1]["content"]
    assert first.count(R.COMMON_SECTION) == 1 and f"you are `{name}` (helper)" in first


def test_a_helper_message_that_names_no_one_goes_to_the_writer(tmp_path, helper_boxes):
    from tests.memory.test_helper_messaging import Steps, _done
    from tests.memory.test_helpers import _ctx, _msg, _tc
    from unify.memory import helpers, messaging

    home = tmp_path / "home"
    pr = messaging.PassRecord(
        root="writer",
        deadline=time.monotonic() + 600,
        pass_id="cd" * 8,
        run_home=str(home),
    )
    model = Steps({"t": [_msg(_tc("s", "send", {"text": "no address here"})), _done()]})
    ctx = _ctx(tmp_path, model, pass_record=pr)

    async def go():
        out = await helpers.handle({"task": "t", "episodes": ["e1"]}, ctx)
        await helpers.result(ctx, out.split()[1])
        await messaging.close(pr, "done")

    asyncio.run(go())
    row = next(r for r in rr.read_entries(home) if r["text"] == "no address here")
    assert row["mentions"] == ["writer"]

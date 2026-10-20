"""Every agent on the run record (UNIFY_AGENTS_RUN_RECORD): the four comms tools in every role, ``who``, the common
section's bytes, the prompt-size report, no box binding the record, the cached prefix across rosters, and the
exchanges the mesh carries (the redesign's A1 tests 5, 6, 7, 10 and 11, and its acceptance exchanges).

Keyless: every model is scripted; nothing here opens a socket off this machine.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import pytest

from unify.agents import run_record as rr
from unify.agents.options import Options
from unify.common import tool_registry as R
from unify.settings import SETTINGS
from tests.memory.test_helpers import fake_box  # noqa: F401,F811 (fixture)

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
    monkeypatch.setattr(SETTINGS, "UNIFY_AGENTS_RUN_RECORD", "on")
    monkeypatch.setattr(SETTINGS, "UNIFY_TOOLS_IN_CODE", "on")
    return SETTINGS


# --- test 5: every role has the four comms tools, and who shows the roster --------------------------------------------


def test_every_role_has_the_four_comms_tools_on_the_run_record(run_record_on):
    from unify.actor.tool_handlers import actor_flags
    from unify.memory.helpers import helper_flags

    flags = {
        "actor": actor_flags("actor", can_store=False, librarian=True),
        "teammate": actor_flags("teammate", can_store=False, librarian=False),
        "writer": frozenset({"pool", "similar", "helper", "edit"}),
        "curator": frozenset({"edit"}),
        "helper": helper_flags(True),
        "librarian": frozenset(),
    }
    for agent, f in flags.items():
        allowed = set(R.allow_list(agent, f))
        assert RECORD_TOOLS <= allowed, (agent, sorted(allowed))
        stub = R.render_stub(R.allow_list(agent, f), agent)
        for name in RECORD_TOOLS:
            assert f"def {name}(" in stub, (agent, name)
        # one definition for every agent: the same spec and the same handler
        for name in RECORD_TOOLS:
            assert R.registry()[name] is R._RUN_SPECS[name]
            assert R.handler_for(agent, name) == (
                f"unify.agents.run_record_tools:{name}",
                False,
            )
    # the fork, which runs no code, makes the same calls through its relay
    from unify.memory.integration import fork, fork_worker

    assert fork_worker.comms_calls(
        'tools.send("@writer x")\nprint(tools.wait(timeout=5))\ntools.who()'
    ) == [
        ("send", {"text": "@writer x"}),
        ("wait", {"timeout": 5}),
        ("who", {}),
    ]
    assert fork_worker.comms_calls("tools.messages(all=True)") == [
        ("messages", {"all": True})
    ]
    assert fork_worker.comms_calls("import os\ntools.send('x')") is None
    assert fork_worker.comms_calls("tools.read('/memory')") is None
    assert (
        fork.RECORD_CALLS_ONLY in fork.fork_instruction()
        and fork.NO_TOOLS not in fork.fork_instruction()
    )


def test_with_the_switch_off_no_role_gains_or_loses_a_tool(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_AGENTS_RUN_RECORD", "")
    assert R.registry()["send"] is R.SEND and "librarian" not in R.SEND.roles
    assert not RECORD_TOOLS & set(R.allow_list("librarian", frozenset()))
    from unify.memory.integration import fork

    monkeypatch.setattr(SETTINGS, "UNIFY_TOOLS_IN_CODE", "on")
    assert fork.fork_instruction() == fork.INSTRUCTION_IN_CODE


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
    assert "Mention @librarian when you want to know whether stored items fit" in text


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
    assert {r["name"]: r["state"] for r in rr.read_roster(tmp_path / "home")["agents"]}[
        main.name
    ] == "ended"


def test_the_writers_first_message_carries_the_common_section_and_its_brief_points_at_it(
    run_record_on,
    tmp_path,
    monkeypatch,
):
    """A writer pass on the run record (a pass worker's, the mesh on with the service off): the brief stays a constant
    prefix and the first message carries the common section and the writer's roster lines.
    """
    from tests.memory.integration import test_consolidate as tc
    from unify.memory import messaging
    from unify.memory.integration import consolidate

    fake = tc.FakeWriter()
    monkeypatch.setattr(consolidate, "unillm_turn", fake)
    # the fake writer answers with JSON tools, so this pass runs with the tools in code off; the run record's own need
    # for them (a refusal tested in tests/memory/integration) is not what this checks
    from unify.memory.integration import switch

    monkeypatch.setattr(switch, "redesign_a_refusal", lambda settings: None)
    tc._no_records(monkeypatch)
    stores = tc._stores(tmp_path)
    sha, _ = tc._record(stores, "e1")
    settings = tc._settings_library()
    settings.UNIFY_AGENTS_RUN_RECORD = "on"
    asyncio.run(
        consolidate.run_due_passes(
            stores,
            "e1",
            sha,
            tc.State(stores.paths.state),
            effort="low",
            settings=settings,
            emit=None,
        ),
    )
    system, first = fake.seen[0][0]["content"], fake.seen[0][1]["content"]
    assert first.count(R.COMMON_SECTION) == 1 and "you are `writer` (writer)" in first
    assert messaging.RECORD_TEXT_RUN in system and R.COMMON_SECTION not in system
    # the pass's agent joined the run record and left it at the pass's end (no service holds its place)
    roster = {r["name"]: r for r in rr.read_roster(stores.paths.home)["agents"]}
    assert roster["writer"]["state"] == "ended" and roster["writer"]["role"] == "writer"


def _tokens(text: str) -> int:
    from unify.memory.index import estimate_tokens

    return int(estimate_tokens(text))


def test_prompt_size_report_per_role_before_and_after(
    run_record_on, monkeypatch, capsys
):
    """A1 test 7: per role, the tokens the record adds to its prompt and its tool JSON, before (switch off) and after
    (on). No role loses a tool, the tool JSON is the same, and no role grows by more than the common section and its
    roster lines; what does is listed."""
    from unify.memory import helpers, messaging
    from unify.memory.integration import fork

    def texts(on: bool) -> dict:
        monkeypatch.setattr(SETTINGS, "UNIFY_AGENTS_RUN_RECORD", "on" if on else "")
        lines = lambda name, role, spawner=None: (
            rr.prompt_lines(name, role, spawner) if on else ""
        )  # noqa: E731
        return {
            # the actor's opening (system) and its first message's record section
            "actor": R.preamble("actor", {"flags": frozenset({"helper"})})
            + lines("actor-3f9a", "actor"),
            "writer": messaging.record_text() + lines("writer", "writer"),
            "curator": messaging.record_text() + lines("curator", "writer"),
            "helper": helpers.helper_brief(True, in_code=True)
            + lines("call-1", "helper", "writer"),
            "librarian": R.preamble("librarian", {})
            + lines("librarian-3f9a", "librarian"),
            "fork": fork.fork_instruction() + lines("fork-3f9a", "fork"),
        }

    def tools(on: bool) -> dict:
        monkeypatch.setattr(SETTINGS, "UNIFY_AGENTS_RUN_RECORD", "on" if on else "")
        return {a: (len(json.dumps(R.json_tools(a))), set(R.allow_list(a, frozenset({"helper", "pool", "similar", "edit"}))))
                for a in ("actor", "teammate", "writer", "curator", "helper", "librarian")}  # fmt: skip

    before, after = texts(False), texts(True)
    t_before, t_after = tools(False), tools(True)
    report, excess = [], {}
    for role in before:
        # compared exactly, in characters; reported in tokens too
        grown = len(after[role]) - len(before[role])
        allowed = len(R.COMMON_SECTION) + len(
            after[role].split(R.COMMON_SECTION, 1)[-1]
        )
        report.append(
            f"{role}: {_tokens(before[role])} -> {_tokens(after[role])} tokens "
            f"({len(before[role])} -> {len(after[role])} chars; +{grown}, allowed +{allowed})",
        )
        if grown > allowed:
            excess[role] = grown - allowed
    for agent, (size_before, names_before) in t_before.items():
        size_after, names_after = t_after[agent]
        assert (
            size_after == size_before
        ), agent  # one model-API tool, execute_code, for every agent
        assert names_before <= names_after, (
            agent,
            names_before - names_after,
        )  # no role loses a tool
        report.append(
            f"{agent} tool JSON: {size_before} -> {size_after} bytes; tools +{sorted(names_after - names_before)}"
        )
    text = "\n".join(
        ["prompt-size report (UNIFY_AGENTS_RUN_RECORD off -> on):", *report, f"excess: {excess}"],
    )
    print(text)
    # kept beside the run's logs (gitignored) for the build's record
    out = Path(__file__).resolve().parents[2] / "logs" / "redesign-a-prompt-sizes.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n")
    # the fork's instruction names the calls it may make in place of "Do not call tools.": listed, by design
    assert set(excess) <= {"fork"}, excess


# --- test 10: no box binds the record --------------------------------------------------------------------------------


def test_no_helper_box_bind_reaches_the_run_record(tmp_path, fake_box):
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
    assert fake_box, "the helper ran no cell"
    agents = rr.agents_dir(home).resolve()
    assert rr.record_path(home).is_file()
    for call in fake_box:
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
    targets = [rr.record_path(home), rr.deliveries_path(home), rr.roster_path(home)]
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
    tmp_path, fake_box
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


def test_a_helper_message_that_names_no_one_goes_to_the_writer(tmp_path, fake_box):
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

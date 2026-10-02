"""Symbolic: ``UNIFY_FUNCTION_CASES`` records calls of stored functions and replays them before a change.

As shipped, an overwrite or patch replaces a stored function's source with
no check that it still does what it did: in past runs a function that removed
the tracks released *before* a year was patched, under the same name, to
remove those released *at or after* it, and a review patched one entry 33
times. The calls cannot be re-run against the real environment (a function
may change it, and must not run twice), so with the switch on each call is
recorded as a case -- the arguments, what it returned or raised, and the
environment calls it made with their answers -- and the cases that returned
are replayed against the new source with the recorded answers served in
place of the environment. A different environment call, one too many or too
few, or another return value refuses the change and names the case; storing
the behaviour under a new name, or retiring the case with a reason, lets it
through. A case that cannot be replayed faithfully (the clock, randomness, a
model call, a replay that runs too long) does not block and is reported.
Functions run in-process against a fake registered environment; no model or
network is called. With the switch off nothing is written and every call and
overwrite is as shipped.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from tests.helpers import _handle_project
from unify import db
from unify.function_manager import store_cases
from unify.function_manager.function_manager import FunctionManager
from unify.function_manager.primitives import (
    EnvironmentMethod,
    EnvironmentNamespace,
    EnvironmentSurface,
    register_environment,
)
from unify.function_manager.primitives.environment import (
    clear_environment_namespaces,
    namespace_object,
)
from unify.settings import ProductionSettings, SETTINGS

DOUBLE = "def double(x: int) -> int:\n    return x * 2\n"
TRIPLE_AS_DOUBLE = "def double(x: int) -> int:\n    return x * 3\n"
DOUBLE_REWRITTEN = "def double(x: int) -> int:\n    return x + x\n"
DIVIDE = "def divide(a: int, b: int) -> float:\n    return a / b\n"
SAFE_DIVIDE = (
    "def divide(a: int, b: int) -> float:\n"
    "    if b == 0:\n"
    "        return 0.0\n"
    "    return a / b\n"
)

REMOVE_BEFORE = (
    "def remove_tracks_before(year: int) -> int:\n"
    "    removed = 0\n"
    "    for track in primitives.music.list_tracks():\n"
    "        if track['year'] < year:\n"
    "            primitives.music.remove_track(track_id=track['id'])\n"
    "            removed += 1\n"
    "    return removed\n"
)
REMOVE_BEFORE_REWRITTEN = (
    "def remove_tracks_before(year: int) -> int:\n"
    "    old = [t for t in primitives.music.list_tracks() if t['year'] < year]\n"
    "    for track in old:\n"
    "        primitives.music.remove_track(track['id'])\n"
    "    return len(old)\n"
)
REMOVE_AT_OR_AFTER = REMOVE_BEFORE.replace(
    "track['year'] < year",
    "track['year'] >= year",
)
REMOVE_BEFORE_NO_CALL = (
    "def remove_tracks_before(year: int) -> int:\n"
    "    return len([t for t in primitives.music.list_tracks() if t['year'] < year])\n"
)
REMOVE_BEFORE_LISTS_TWICE = (
    "def remove_tracks_before(year: int) -> int:\n"
    "    primitives.music.list_tracks()\n"
    "    removed = 0\n"
    "    for track in primitives.music.list_tracks():\n"
    "        if track['year'] < year:\n"
    "            primitives.music.remove_track(track_id=track['id'])\n"
    "            removed += 1\n"
    "    return removed\n"
)
STAMPED = (
    "def stamped(x: int) -> int:\n"
    "    import time\n"
    "    return x + int(time.time() * 0)\n"
)
STAMPED_CHANGED = (
    "def stamped(x: int) -> int:\n"
    "    import time\n"
    "    return x + 1 + int(time.time() * 0)\n"
)
ASK = "def ask(q: str) -> str:\n    return q.upper()\n"
ASK_A_MODEL = "def ask(q: str) -> str:\n    return query_llm(q)\n"
SLOW = "async def slow(x: int) -> int:\n    return x\n"
SLOW_CHANGED = (
    "async def slow(x: int) -> int:\n" "    await asyncio.sleep(1.0)\n" "    return x\n"
)
CLEANUP = (
    "def cleanup(year: int) -> str:\n"
    "    return f'removed {remove_tracks_before(year)}'\n"
)

CALLS: list[tuple] = []


class FakeMusic:
    """A stand-in environment whose calls the test can count."""

    def __init__(self):
        self.tracks = [
            {"id": 1, "year": 1990},
            {"id": 2, "year": 2005},
            {"id": 3, "year": 2020},
        ]

    def list_tracks(self, **kwargs):
        CALLS.append(("list_tracks", kwargs))
        return [dict(t) for t in self.tracks]

    def remove_track(self, track_id: int):
        CALLS.append(("remove_track", {"track_id": track_id}))
        self.tracks = [t for t in self.tracks if t["id"] != track_id]
        return {"removed": track_id}


MUSIC = FakeMusic()


@pytest.fixture
def music_env():
    clear_environment_namespaces()
    global MUSIC
    MUSIC = FakeMusic()
    methods = (
        EnvironmentMethod(
            name="list_tracks",
            call=lambda **kw: MUSIC.list_tracks(**kw),
            effect="read",
        ),
        EnvironmentMethod(
            name="remove_track",
            call=lambda track_id: MUSIC.remove_track(track_id),
            effect="destructive",
            signature="(track_id: int)",
        ),
    )
    register_environment(
        EnvironmentSurface(
            namespaces=(EnvironmentNamespace(name="music", methods=methods),),
        ),
        source="tests:music",
    )
    from unify.function_manager import function_manager as fm_module

    fm_module._PRIMITIVES_SEEDED_FOR.clear()
    CALLS.clear()
    yield
    clear_environment_namespaces()
    fm_module._PRIMITIVES_SEEDED_FOR.clear()


@pytest.fixture
def cases_on(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_FUNCTION_CASES", True)


@pytest.fixture
def cases_off(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_FUNCTION_CASES", False)


def _FM() -> FunctionManager:
    return FunctionManager(include_primitives=False)


def _load(fm: FunctionManager) -> dict:
    """Load the library as the actor's list tool does, with the registered environment as ``primitives``."""
    namespace = {"primitives": SimpleNamespace(music=namespace_object("music"))}
    fm.list_functions(_return_callable=True, _namespace=namespace)
    return namespace


def _cases(fm: FunctionManager, name: str, **kw) -> list[store_cases.Case]:
    return store_cases.cases(int(fm.list_function_name_to_ids()[name]), **kw)


def _source(name: str) -> str:
    return db.query_one("SELECT implementation FROM functions WHERE name = ?", (name,))[
        "implementation"
    ]


def _row_count() -> int:
    return int(db.query_one("SELECT COUNT(*) AS n FROM function_cases")["n"])


def test_the_switch_is_off_by_default_and_parses_as_a_bool():
    assert ProductionSettings.model_fields["UNIFY_FUNCTION_CASES"].default is False
    assert ProductionSettings(UNIFY_FUNCTION_CASES="true").UNIFY_FUNCTION_CASES is True


# --------------------------------------------------------------------------- #
#  Recording                                                                   #
# --------------------------------------------------------------------------- #


@_handle_project
def test_a_pure_call_is_recorded_with_its_arguments_and_result(cases_on):
    fm = _FM()
    fm.add_functions(implementations=[DOUBLE])
    assert _load(fm)["double"](3) == 6
    (case,) = _cases(fm, "double")
    assert (case.kind, case.status, case.args_shown) == ("pass", "active", "x=3")
    assert case.call == {"args": [3], "kwargs": {}}
    assert case.result["shown"] == "6" and case.result["exact"] is True
    assert case.trace == () and case.trace_complete
    assert case.error is None


@_handle_project
def test_environment_calls_are_recorded_in_order_with_their_answers(
    cases_on,
    music_env,
):
    fm = _FM()
    fm.add_functions(implementations=[REMOVE_BEFORE])
    assert _load(fm)["remove_tracks_before"](2000) == 1
    (case,) = _cases(fm, "remove_tracks_before")
    assert [c["call"] for c in case.trace] == [
        "music.list_tracks",
        "music.remove_track",
    ]
    assert case.trace[0]["result"][0] == {"id": 1, "year": 1990}
    assert case.trace[1]["shown"] == "track_id=1"
    assert case.trace[1]["result"] == {"removed": 1}
    assert case.result["shown"] == "1"


@_handle_project
def test_a_stored_function_called_inside_another_is_a_case_of_its_own(
    cases_on,
    music_env,
):
    fm = _FM()
    fm.add_functions(implementations=[REMOVE_BEFORE])
    fm.add_functions(implementations=[CLEANUP])
    assert _load(fm)["cleanup"](2000) == "removed 1"
    (outer,) = _cases(fm, "cleanup")
    (inner,) = _cases(fm, "remove_tracks_before")
    # the callee's environment calls are in both traces
    assert [c["call"] for c in outer.trace] == [c["call"] for c in inner.trace]
    assert len(outer.trace) == 2


@_handle_project
def test_a_raise_is_recorded_as_a_failing_case(cases_on):
    fm = _FM()
    fm.add_functions(implementations=[DIVIDE])
    with pytest.raises(ZeroDivisionError):
        _load(fm)["divide"](1, 0)
    (case,) = _cases(fm, "divide")
    assert (case.kind, case.error) == ("fail", "ZeroDivisionError: division by zero")


@_handle_project
def test_a_call_that_does_not_fit_the_signature_is_not_recorded(cases_on):
    fm = _FM()
    fm.add_functions(implementations=[DIVIDE])
    with pytest.raises(TypeError):
        _load(fm)["divide"](1)
    assert _cases(fm, "divide") == []


@_handle_project
def test_at_most_three_passing_and_three_failing_cases_one_per_input(cases_on):
    fm = _FM()
    fm.add_functions(implementations=[DIVIDE])
    divide = _load(fm)["divide"]
    for a in (1, 2, 3, 4, 5):
        divide(a, 1)
    for a in (1, 2, 3, 4):
        with pytest.raises(ZeroDivisionError):
            divide(a, 0)
    passing = [c for c in _cases(fm, "divide") if c.kind == "pass"]
    failing = [c for c in _cases(fm, "divide") if c.kind == "fail"]
    assert [c.args_shown for c in passing] == ["a=3, b=1", "a=4, b=1", "a=5, b=1"]
    assert [c.args_shown for c in failing] == ["a=2, b=0", "a=3, b=0", "a=4, b=0"]
    # the same input again updates its case in place
    kept = passing[0].case_id
    divide(a=3, b=1)
    passing = [c for c in _cases(fm, "divide") if c.kind == "pass"]
    assert len(passing) == 3 and kept in {c.case_id for c in passing}


@_handle_project
@pytest.mark.asyncio
async def test_execute_function_and_proxy_calls_are_recorded(cases_on, music_env):
    fm = _FM()
    fm.add_functions(implementations=[DOUBLE, REMOVE_BEFORE])
    out = await fm.execute_function(function_name="double", call_kwargs={"x": 4})
    assert out["result"] == 8
    out = await fm.execute_function(
        function_name="remove_tracks_before",
        call_kwargs={"year": 2000},
        extra_namespaces={
            "primitives": SimpleNamespace(music=namespace_object("music")),
        },
    )
    assert out["error"] is None and out["result"] == 1
    proxies = fm.list_functions(_return_callable=True, _namespace={})
    assert await proxies["double"](5) == 10
    assert {c.args_shown for c in _cases(fm, "double")} == {"x=4", "x=5"}
    (case,) = _cases(fm, "remove_tracks_before")
    assert [c["call"] for c in case.trace] == [
        "music.list_tracks",
        "music.remove_track",
    ]


@_handle_project
@pytest.mark.asyncio
async def test_a_call_through_the_actors_execute_function_tool_is_recorded(cases_on):
    from unify.actor.code_act_actor import CodeActActor

    fm = _FM()
    fm.add_functions(implementations=[DIVIDE])
    actor = CodeActActor(function_manager=fm, can_store=False)
    tool = actor.get_tools("act")["execute_function"]
    tool = getattr(tool, "fn", tool)
    try:
        for b in (2, 0):
            await tool(
                thought="Reusing a stored function.",
                function_name="divide",
                call_kwargs={"a": 4, "b": b},
            )
    finally:
        await actor.close()
    passed, failed = sorted(_cases(fm, "divide"), key=lambda c: c.kind != "pass")
    assert (passed.args_shown, passed.result["shown"]) == ("a=4, b=2", "2.0")
    assert (failed.kind, failed.error) == (
        "fail",
        "ZeroDivisionError: division by zero",
    )


@_handle_project
def test_a_credential_argument_is_not_shown(cases_on):
    fm = _FM()
    fm.add_functions(
        implementations=[
            "def login(user: str, password: str) -> str:\n    return user\n",
        ],
    )
    _load(fm)["login"]("ann", "hunter2-secret")
    (case,) = _cases(fm, "login")
    assert case.args_shown == "user='ann', password=<withheld>"


@_handle_project
def test_deleting_a_function_deletes_its_cases(cases_on):
    fm = _FM()
    fm.add_functions(implementations=[DOUBLE])
    _load(fm)["double"](1)
    assert _row_count() == 1
    fm.delete_function(function_id=fm.list_function_name_to_ids()["double"])
    assert _row_count() == 0


# --------------------------------------------------------------------------- #
#  Replay on overwrite and patch                                               #
# --------------------------------------------------------------------------- #


@_handle_project
def test_a_change_that_keeps_every_case_is_stored_and_says_so(cases_on, music_env):
    fm = _FM()
    fm.add_functions(implementations=[DOUBLE, REMOVE_BEFORE])
    ns = _load(fm)
    ns["double"](3)
    ns["remove_tracks_before"](2000)
    calls_before = list(CALLS)
    out = fm.add_functions(
        implementations=[DOUBLE_REWRITTEN, REMOVE_BEFORE_REWRITTEN],
        overwrite=True,
    )
    assert out["double"] == "updated; cases: 1 recorded call(s) replayed unchanged"
    # a positional argument binds to the same parameter as the recorded keyword
    assert out["remove_tracks_before"] == (
        "updated; cases: 1 recorded call(s) replayed unchanged"
    )
    assert _source("remove_tracks_before") == REMOVE_BEFORE_REWRITTEN
    # the replay never reached the environment
    assert CALLS == calls_before


@_handle_project
def test_a_different_environment_call_refuses_the_change(cases_on, music_env):
    fm = _FM()
    fm.add_functions(implementations=[REMOVE_BEFORE])
    _load(fm)["remove_tracks_before"](2010)
    (case,) = _cases(fm, "remove_tracks_before")
    out = fm.add_functions(
        implementations=[REMOVE_AT_OR_AFTER],
        overwrite=True,
        raise_on_error=False,
    )
    error = out["remove_tracks_before"]
    assert error.startswith(
        "error: 'remove_tracks_before' was not changed: the new source does "
        "something else on 1 recorded call(s) that worked before:",
    )
    assert (
        f"- case {case.case_id}, remove_tracks_before(year=2010): diverges at "
        "environment call 2: recorded primitives.music.remove_track(track_id=1), "
        "now primitives.music.remove_track(track_id=3)"
    ) in error
    assert "under a new name" in error and "FunctionManager_retire_case" in error
    assert _source("remove_tracks_before") == REMOVE_BEFORE
    with pytest.raises(ValueError, match="was not changed"):
        fm.add_functions(implementations=[REMOVE_AT_OR_AFTER], overwrite=True)


@_handle_project
def test_an_extra_or_a_missing_environment_call_refuses_the_change(
    cases_on,
    music_env,
):
    fm = _FM()
    fm.add_functions(implementations=[REMOVE_BEFORE])
    _load(fm)["remove_tracks_before"](2000)
    fewer = fm.add_functions(
        implementations=[REMOVE_BEFORE_NO_CALL],
        overwrite=True,
        raise_on_error=False,
    )["remove_tracks_before"]
    assert "makes 1 of the 2 recorded environment calls and then returns" in fewer
    more = fm.add_functions(
        implementations=[REMOVE_BEFORE_LISTS_TWICE],
        overwrite=True,
        raise_on_error=False,
    )["remove_tracks_before"]
    assert "diverges at environment call 2" in more


@_handle_project
def test_a_changed_return_value_refuses_the_change(cases_on):
    fm = _FM()
    fm.add_functions(implementations=[DOUBLE])
    _load(fm)["double"](3)
    out = fm.add_functions(
        implementations=[TRIPLE_AS_DOUBLE],
        overwrite=True,
        raise_on_error=False,
    )
    assert "double(x=3): returned 6 before; now returns 9" in out["double"]
    assert _source("double") == DOUBLE


@_handle_project
def test_a_patch_that_changes_behaviour_is_refused(cases_on, monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_FUNCTION_PATCH", True)
    fm = _FM()
    fm.add_functions(implementations=[DOUBLE])
    _load(fm)["double"](3)
    refused = fm.patch_function(
        name="double",
        old="x * 2",
        new="x * 3",
        why="triple instead",
    )
    assert "returned 6 before; now returns 9" in refused["error"]
    assert _source("double") == DOUBLE
    kept = fm.patch_function(name="double", old="x * 2", new="2 * x", why="tidy")
    assert kept["status"] == "patched"
    assert kept["cases"] == "cases: 1 recorded call(s) replayed unchanged"


@_handle_project
def test_a_nondeterministic_function_is_inconclusive_and_does_not_block(cases_on):
    fm = _FM()
    fm.add_functions(implementations=[STAMPED])
    _load(fm)["stamped"](1)
    out = fm.add_functions(implementations=[STAMPED_CHANGED], overwrite=True)
    assert out["stamped"].startswith(
        "updated; cases: 0 recorded call(s) replayed unchanged; 1 could not be "
        "replayed faithfully:",
    )
    assert "it imports time" in out["stamped"] or "it reads time" in out["stamped"]
    assert _source("stamped") == STAMPED_CHANGED


@_handle_project
def test_a_model_call_is_inconclusive(cases_on):
    fm = _FM()
    fm.add_functions(implementations=[ASK])
    _load(fm)["ask"]("hi")
    out = fm.add_functions(implementations=[ASK_A_MODEL], overwrite=True)
    assert "could not be replayed faithfully" in out["ask"]
    assert "query_llm" in out["ask"]


@_handle_project
def test_a_replay_that_runs_too_long_is_inconclusive(cases_on, monkeypatch):
    monkeypatch.setattr(store_cases, "REPLAY_TIMEOUT_S", 0.2)
    fm = _FM()
    fm.add_functions(implementations=[SLOW])
    asyncio.run(_load(fm)["slow"](1))
    out = fm.add_functions(implementations=[SLOW_CHANGED], overwrite=True)
    assert "the replay took over 0.2s" in out["slow"]


@_handle_project
def test_a_failing_case_is_rerun_and_reported_when_it_now_returns(cases_on):
    fm = _FM()
    fm.add_functions(implementations=[DIVIDE])
    divide = _load(fm)["divide"]
    divide(4, 2)
    with pytest.raises(ZeroDivisionError):
        divide(1, 0)
    failing = next(c for c in _cases(fm, "divide") if c.kind == "fail")
    out = fm.add_functions(implementations=[SAFE_DIVIDE], overwrite=True)
    assert out["divide"] == (
        "updated; cases: 1 recorded call(s) replayed unchanged; 1 that raised "
        f"before now return:\n- case {failing.case_id}, divide(a=1, b=0): now "
        "returns 0.0"
    )


@_handle_project
def test_a_retired_case_no_longer_blocks(cases_on):
    fm = _FM()
    fm.add_functions(implementations=[DOUBLE])
    _load(fm)["double"](3)
    (case,) = _cases(fm, "double")
    assert "error" in fm.retire_case(
        function_name="double",
        case_id=case.case_id,
        why="",
    )
    missing = case.case_id + 1000
    assert (
        f"no active case #{missing}"
        in fm.retire_case(
            function_name="double",
            case_id=missing,
            why="wrong",
        )["error"]
    )
    done = fm.retire_case(
        function_name="double",
        case_id=case.case_id,
        why="the caller wanted triples all along",
    )
    assert done == {
        "function_name": "double",
        "case_id": case.case_id,
        "status": "retired",
    }
    assert fm.add_functions(implementations=[TRIPLE_AS_DOUBLE], overwrite=True) == {
        "double": "updated",
    }
    (retired,) = _cases(fm, "double", status="retired")
    assert retired.retired_why == "the caller wanted triples all along"


@_handle_project
def test_new_behaviour_under_a_new_name_is_stored(cases_on):
    fm = _FM()
    fm.add_functions(implementations=[DOUBLE])
    _load(fm)["double"](3)
    triple = "def triple(x: int) -> int:\n    return x * 3\n"
    assert fm.add_functions(implementations=[triple]) == {"triple": "added"}
    assert _source("double") == DOUBLE


# --------------------------------------------------------------------------- #
#  Visibility                                                                  #
# --------------------------------------------------------------------------- #


@_handle_project
def test_search_and_filter_results_carry_compact_cases(cases_on):
    fm = _FM()
    fm.add_functions(implementations=[DIVIDE, DOUBLE])
    ns = _load(fm)
    ns["divide"](4, 2)
    with pytest.raises(ZeroDivisionError):
        ns["divide"](1, 0)
    passed, failed = sorted(_cases(fm, "divide"), key=lambda c: c.kind != "pass")
    rows = {r["name"]: r for r in fm.filter_functions()}
    assert rows["divide"]["cases"] == (
        f"#{passed.case_id} divide(a=4, b=2) -> 2.0; #{failed.case_id} "
        "divide(a=1, b=0) raised ZeroDivisionError: division by zero"
    )
    assert "cases" not in rows["double"]  # never called
    loaded = fm.filter_functions(
        _return_callable=True,
        _namespace={},
        _also_return_metadata=True,
    )["metadata"]
    assert {r["name"]: r.get("cases") for r in loaded}["divide"] == rows["divide"][
        "cases"
    ]


@_handle_project
def test_a_long_cases_field_is_cut_to_its_bound(cases_on):
    fm = _FM()
    fm.add_functions(implementations=["def echo(s: str) -> str:\n    return s\n"])
    _load(fm)["echo"]("x" * 5000)
    (row,) = fm.filter_functions()
    assert len(row["cases"]) == store_cases.SUMMARY_LIMIT


def test_the_review_gets_the_retire_tool_only_while_on(monkeypatch):
    import unify.actor.code_act_actor as code_act_actor
    from unify.guidance_manager.guidance_manager import GuidanceManager

    actor = SimpleNamespace(
        function_manager=FunctionManager(),
        guidance_manager=GuidanceManager(),
    )
    monkeypatch.setattr(SETTINGS, "UNIFY_FUNCTION_CASES", False)
    off = set(code_act_actor._build_storage_tools(actor=actor, ask_tools={})[0])
    monkeypatch.setattr(SETTINGS, "UNIFY_FUNCTION_CASES", True)
    on = set(code_act_actor._build_storage_tools(actor=actor, ask_tools={})[0])
    assert on - off == {"FunctionManager_retire_case"}


# --------------------------------------------------------------------------- #
#  Redaction                                                                   #
# --------------------------------------------------------------------------- #

TOKEN = "tok-5f1e9a7c2b8d4e60"  # pragma: allowlist secret
PASSWORD = "hunter2-correct-horse"  # pragma: allowlist secret

# Logs in, passes the token on under a credential name and inside a header,
# and returns the token and the items.
SYNC = (
    "def sync(user: str, password: str) -> dict:\n"
    "    login = primitives.accounts.login(username=user, password=password)\n"
    "    token = login['access_token']\n"
    "    items = primitives.accounts.list_items(access_token=token)\n"
    "    primitives.accounts.ping(header='Bearer ' + token)\n"
    "    return {'session': token, 'count': len(items), 'first': items[0]}\n"
)
SYNC_REWRITTEN = (
    "def sync(user: str, password: str) -> dict:\n"
    "    t = primitives.accounts.login(username=user, password=password)['access_token']\n"
    "    found = list(primitives.accounts.list_items(access_token=t))\n"
    "    primitives.accounts.ping(header=f'Bearer {t}')\n"
    "    return {'first': found[0], 'count': len(found), 'session': t}\n"
)
SYNC_WRONG_TOKEN = SYNC.replace(
    "list_items(access_token=token)",
    "list_items(access_token=token[:-1])",
)
SYNC_COUNTS_AUTHORS = SYNC.replace(
    "'count': len(items)",
    "'count': len([i for i in items if i['author'] == user])",
)


class FakeAccounts:
    def login(self, username: str, password: str):
        CALLS.append(("login", {"username": username}))
        assert password == PASSWORD
        return {"access_token": TOKEN, "token_type": "Bearer"}

    def list_items(self, access_token: str):
        CALLS.append(("list_items", {}))
        assert access_token == TOKEN
        return [{"id": 7, "author": "ann"}, {"id": 8, "author": "bob"}]

    def ping(self, header: str):
        CALLS.append(("ping", {}))
        assert header == f"Bearer {TOKEN}"
        return {"ok": True}


@pytest.fixture
def accounts_env():
    clear_environment_namespaces()
    accounts = FakeAccounts()
    methods = (
        EnvironmentMethod(
            name="login",
            call=accounts.login,
            effect="read",
            signature="(username: str, password: str)",
        ),
        EnvironmentMethod(
            name="list_items",
            call=accounts.list_items,
            effect="read",
            signature="(access_token: str)",
        ),
        EnvironmentMethod(
            name="ping",
            call=accounts.ping,
            effect="read",
            signature="(header: str)",
        ),
    )
    register_environment(
        EnvironmentSurface(
            namespaces=(EnvironmentNamespace(name="accounts", methods=methods),),
        ),
        source="tests:accounts",
    )
    from unify.function_manager import function_manager as fm_module

    fm_module._PRIMITIVES_SEEDED_FOR.clear()
    CALLS.clear()
    yield
    clear_environment_namespaces()
    fm_module._PRIMITIVES_SEEDED_FOR.clear()


def _load_accounts(fm: FunctionManager) -> dict:
    namespace = {
        "primitives": SimpleNamespace(accounts=namespace_object("accounts")),
    }
    fm.list_functions(_return_callable=True, _namespace=namespace)
    return namespace


def _stored_case_text() -> str:
    rows = db.query("SELECT * FROM function_cases")
    return "\n".join(str(dict(row)) for row in rows)


def test_credential_keys_are_matched_word_by_word():
    from unify.function_manager.store_trust import credential_key

    for name in ("access_token", "apiKey", "X-Auth-Token", "Authorization", "password"):
        assert credential_key(name), name
    for name in ("author", "monkey", "keyword", "user", "header"):
        assert not credential_key(name), name


@_handle_project
def test_secrets_in_answers_and_arguments_are_stored_as_placeholders(
    cases_on,
    accounts_env,
):
    fm = _FM()
    fm.add_functions(implementations=[SYNC])
    out = _load_accounts(fm)["sync"]("ann", PASSWORD)
    assert out["session"] == TOKEN  # the caller still gets the real value
    (case,) = _cases(fm, "sync")
    stored = _stored_case_text()
    assert TOKEN not in stored and PASSWORD not in stored
    login, items, ping = case.trace
    token = login["result"]["access_token"]
    assert store_cases.REDACTED.fullmatch(token)
    # data under a key that only contains a credential word is kept
    assert items["result"][0] == {"id": 7, "author": "ann"}
    assert case.call["args"][0] == "ann"
    assert store_cases.REDACTED.fullmatch(case.call["args"][1])
    assert ping["shown"] == f"header='Bearer {token}'"
    assert f"'session': '{token}'" in case.result["shown"]
    # the same secret is one placeholder within a case, and another in the next
    assert case.salt
    _load_accounts(fm)["sync"]("ann", PASSWORD)
    (again,) = _cases(fm, "sync")
    assert again.trace[0]["result"]["access_token"] != token


@_handle_project
def test_a_function_that_passes_a_secret_on_still_replays(cases_on, accounts_env):
    fm = _FM()
    fm.add_functions(implementations=[SYNC])
    _load_accounts(fm)["sync"]("ann", PASSWORD)
    calls_before = list(CALLS)
    out = fm.add_functions(implementations=[SYNC_REWRITTEN], overwrite=True)
    assert out["sync"] == "updated; cases: 1 recorded call(s) replayed unchanged"
    assert CALLS == calls_before


@_handle_project
def test_a_change_to_what_is_done_with_a_secret_is_still_refused(
    cases_on,
    accounts_env,
):
    fm = _FM()
    fm.add_functions(implementations=[SYNC])
    _load_accounts(fm)["sync"]("ann", PASSWORD)
    out = fm.add_functions(
        implementations=[SYNC_WRONG_TOKEN],
        overwrite=True,
        raise_on_error=False,
    )["sync"]
    assert "diverges at environment call 2" in out
    assert "access_token=<withheld>" in out
    assert TOKEN not in out and PASSWORD not in out
    # data next to the secret is compared as recorded
    out = fm.add_functions(
        implementations=[SYNC_COUNTS_AUTHORS],
        overwrite=True,
        raise_on_error=False,
    )["sync"]
    assert "was not changed" in out and TOKEN not in out
    assert _source("sync") == SYNC


@_handle_project
def test_a_case_recorded_before_redaction_replays_unredacted(
    cases_on,
    accounts_env,
    monkeypatch,
):
    # as the first version of the switch stored cases: no salt, values in clear
    monkeypatch.setattr(
        store_cases._Redactor,
        "fresh",
        lambda: store_cases._Redactor(None),
    )
    fm = _FM()
    fm.add_functions(implementations=[SYNC])
    _load_accounts(fm)["sync"]("ann", PASSWORD)
    (old,) = _cases(fm, "sync")
    assert old.salt is None and old.call["args"][1] == PASSWORD
    assert old.trace[0]["result"]["access_token"] == TOKEN
    monkeypatch.undo()
    monkeypatch.setattr(SETTINGS, "UNIFY_FUNCTION_CASES", True)
    out = fm.add_functions(implementations=[SYNC_REWRITTEN], overwrite=True)
    assert out["sync"] == "updated; cases: 1 recorded call(s) replayed unchanged"


# --------------------------------------------------------------------------- #
#  Switch off                                                                  #
# --------------------------------------------------------------------------- #


@_handle_project
def test_switch_off_records_nothing_and_overwrites_as_shipped(cases_off, music_env):
    fm = _FM()
    fm.add_functions(implementations=[DOUBLE, REMOVE_BEFORE])
    ns = _load(fm)
    assert ns["double"](3) == 6
    assert ns["remove_tracks_before"](2010) == 2
    assert _row_count() == 0
    assert fm.add_functions(
        implementations=[TRIPLE_AS_DOUBLE, REMOVE_AT_OR_AFTER],
        overwrite=True,
    ) == {"double": "updated", "remove_tracks_before": "updated"}
    assert all("cases" not in row for row in fm.filter_functions())
    assert "error" in fm.retire_case(function_name="double", case_id=1, why="x")
    assert _row_count() == 0

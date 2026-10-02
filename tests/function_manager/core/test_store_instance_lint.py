"""Symbolic: ``UNIFY_STORE_INSTANCE_LINT`` keeps a task's own identifiers out of the libraries.

In offline replays of storage reviews (2 Oct, review-replay-v2) what was
stored sometimes carried values of the one task in front of the review: an
aliased ARC task id (``task-7a4cf12e``) in a docstring, in one or two reviews
per condition, an AppWorld playlist title from the request (``"R&B
Recommendation"``) hard-coded in the code, literal example data. No later task
shares them. With the switch on, a function whose name or code (literals,
defaults) carries an identifier from the session's request, or whose name has
the shape of a task id, is refused with the identifier named; a docstring or
a guidance entry that names one is stored with a warning. Domain words (an
app name, ``grid``, a quoted tool name) are not identifiers. The request's
tokens reach the task loop, its tools and its storage review through the task
context. With the switch off everything is stored as shipped. No model is
called.
"""

from __future__ import annotations

import asyncio
import contextvars

import pytest

from tests import cache_discipline_helpers as h
from tests.helpers import _handle_project
from unify.function_manager import instance_lint
from unify.function_manager.function_manager import FunctionManager
from unify.guidance_manager.guidance_manager import GuidanceManager
from unify.settings import ProductionSettings, SETTINGS

ARC = (
    "New instance. Task id: task-7a4cf12e\n"
    "Test input (2x2):\n0 1\n1 0\n"
    'Reply with {"action": "request_demonstration"} to see a demonstration.'
)
APPWORLD = (
    'Create a Spotify playlist titled "R&B Recommendation" with the songs '
    "my friends liked this week, and reply with its playlist_id. "
    "Order 4417823 is already paid."
)


@pytest.fixture
def lint(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_STORE_CHECK", "")
    monkeypatch.setattr(SETTINGS, "UNIFY_STORE_VERIFY", "")
    monkeypatch.setattr(SETTINGS, "UNIFY_FUNCTION_PATCH", True)

    def set_(on: bool) -> None:
        monkeypatch.setattr(SETTINGS, "UNIFY_STORE_INSTANCE_LINT", on)

    return set_


def _in_task(request, fn):
    """Run *fn* in a fresh context keyed by *request*, as ``act()`` keys its task."""

    def run():
        token = instance_lint.enter(request)
        try:
            return fn()
        finally:
            instance_lint.leave(token)

    return contextvars.copy_context().run(run)


def _add(source: str) -> str:
    fm = FunctionManager()
    name = source.split("def ", 1)[1].split("(", 1)[0]
    return fm.add_functions(implementations=[source], raise_on_error=False)[name]


# ── the tokens of a request ──────────────────────────────────────────────


def test_tokens_are_the_instance_identifiers_not_the_domain_words(lint):
    arc = instance_lint.tokens_of(ARC)
    assert "task-7a4cf12e" in arc.ids and "7a4cf12e" in arc.ids
    assert arc.quoted == ()  # a quoted tool name is vocabulary

    app = instance_lint.tokens_of(APPWORLD)
    assert app.quoted == ("r&b recommendation",)
    assert app.ids == ("4417823",)

    uuid = "3f2b8c1e-9a4d-4c2b-8e1f-0a9b8c7d6e5f"
    assert instance_lint.tokens_of({"messages": [f"Session {uuid}"]}).ids[0] == uuid

    plain = instance_lint.tokens_of(
        "Use the grid in the spotify app; stop after 1000000 steps, reply "
        '"no answer was found" and see primitives.spotify.search_songs.',
    )
    assert not plain


def test_a_sub_agent_keeps_the_tokens_of_its_task(lint):
    lint(True)

    def nested():
        inner = instance_lint.enter("Sub-task: look at order 99887766.")
        try:
            return instance_lint.current()
        finally:
            instance_lint.leave(inner)

    tokens = _in_task(ARC, nested)
    assert "task-7a4cf12e" in tokens.ids and "99887766" not in tokens.ids
    assert not instance_lint.current()


# ── functions ────────────────────────────────────────────────────────────


@_handle_project
def test_on_a_name_with_the_task_alias_is_refused(lint):
    lint(True)
    source = "def solve_task_7a4cf12e(grid: list) -> list:\n    return grid\n"
    status = _in_task(ARC, lambda: _add(source))
    assert status.startswith("error: 'solve_task_7a4cf12e' was not stored"), status
    assert "'task-7a4cf12e'" in status and "parameter" in status
    assert "solve_task_7a4cf12e" not in FunctionManager().list_functions()


@_handle_project
def test_on_a_task_id_shaped_name_is_refused_whatever_the_request(lint):
    lint(True)
    for source in (
        "def solve_task_0badc0de(grid: list) -> list:\n    return grid\n",
        "def run_3f2b8c1e_9a4d_4c2b_8e1f_0a9b8c7d6e5f() -> int:\n    return 1\n",
    ):
        status = _add(source)
        assert status.startswith("error: ") and "the shape of a task id" in status


@_handle_project
def test_on_a_docstring_with_the_task_alias_is_stored_with_a_warning(lint):
    lint(True)
    source = (
        "def mirror_rows(grid: list) -> list:\n"
        '    """Mirror each row, as task-7a4cf12e needs."""\n'
        "    return [row[::-1] for row in grid]\n"
    )
    status = _in_task(ARC, lambda: _add(source))
    assert status.startswith("added; warning: its docstring names 'task-7a4cf12e'")
    assert "mirror_rows" in FunctionManager().list_functions()


@_handle_project
def test_on_a_quoted_instance_value_hard_coded_in_code_is_refused(lint):
    lint(True)
    literal = (
        "def make_playlist(songs: list) -> dict:\n"
        '    return {"title": "R&B Recommendation", "songs": songs}\n'
    )
    default = (
        "def name_playlist(title: str = 'R&B recommendation') -> str:\n"
        "    return title\n"
    )
    number = "def pay(order: int = 4417823) -> int:\n    return order\n"
    for source in (literal, default, number):
        status = _in_task(APPWORLD, lambda: _add(source))
        assert status.startswith("error: ") and "hard-codes" in status, status
    stored = FunctionManager().list_functions()
    assert not {"make_playlist", "name_playlist", "pay"} & set(stored)


@_handle_project
def test_on_domain_words_and_parameters_are_stored(lint):
    lint(True)
    source = (
        "def spotify_playlist_from_grid(title: str, songs: list) -> dict:\n"
        '    """Make a Spotify playlist with the grid of songs; reply with its playlist_id."""\n'
        '    action = {"action": "request_demonstration"}\n'
        '    return {"title": title, "songs": songs, "app": "spotify", "next": action}\n'
    )
    assert _in_task(APPWORLD, lambda: _add(source)) == "added"
    assert (
        _in_task(ARC, lambda: _add(source.replace("def spotify", "def x_spotify")))
        == "added"
    )


@_handle_project
def test_on_a_patch_is_checked_like_any_write(lint):
    lint(True)
    fm = FunctionManager()
    base = "def make_playlist(title: str) -> dict:\n    return {'title': title}\n"
    assert fm.add_functions(implementations=[base]) == {"make_playlist": "added"}

    def patch(old, new):
        return fm.patch_function(name="make_playlist", old=old, new=new, why="fix")

    refused = _in_task(
        APPWORLD,
        lambda: patch("'title': title", "'title': 'R&B Recommendation'"),
    )
    assert "hard-codes 'r&b recommendation'" in refused["error"]
    warned = _in_task(
        ARC,
        lambda: patch(
            "-> dict:\n",
            '-> dict:\n    """Learned on task-7a4cf12e."""\n',
        ),
    )
    assert warned["status"] == "patched"
    assert "task-7a4cf12e" in warned["warning"]


@_handle_project
def test_off_the_same_functions_are_stored_as_shipped(lint):
    lint(False)
    named = "def solve_task_7a4cf12e(grid: list) -> list:\n    return grid\n"
    literal = "def make_playlist() -> str:\n    return 'R&B Recommendation'\n"
    assert _in_task(ARC, lambda: _add(named)) == "added"
    assert _in_task(APPWORLD, lambda: _add(literal)) == "added"
    assert instance_lint.current().ids == ()


# ── guidance ─────────────────────────────────────────────────────────────


@_handle_project
def test_on_guidance_naming_the_instance_is_stored_with_a_warning(lint):
    lint(True)
    gm = GuidanceManager()

    added = _in_task(
        APPWORLD,
        lambda: gm.add_guidance(
            title="Playlists",
            content='Name the playlist "R&B Recommendation" first.',
        ),
    )
    assert "its content names 'r&b recommendation'" in added["warning"]
    gid = added["details"]["guidance_id"]

    updated = _in_task(
        ARC,
        lambda: gm.update_guidance(guidance_id=gid, title="Mirror for task-7a4cf12e"),
    )
    assert "its title names 'task-7a4cf12e'" in updated["warning"]

    patched = _in_task(
        ARC,
        lambda: gm.patch_guidance(
            id_or_title=gid,
            old="first.",
            new="first (seen on 7a4cf12e).",
            why="note",
        ),
    )
    assert "7a4cf12e" in patched["warning"]

    clean = _in_task(
        APPWORLD,
        lambda: gm.add_guidance(title="Spotify", content="Take the title as given."),
    )
    assert "warning" not in clean


@_handle_project
def test_off_guidance_has_no_warning(lint):
    lint(False)
    added = _in_task(
        ARC,
        lambda: GuidanceManager().add_guidance(title="t", content="For task-7a4cf12e."),
    )
    assert "warning" not in added


# ── through the actor ────────────────────────────────────────────────────

# Not a task-id shape: only the request's tokens refuse it.
NAMED = "def mirror_7a4cf12e(grid: list) -> list:\n    return grid\n"
TASK = "New instance. Task id: task-7a4cf12e. Reply with the grid."


def _act_replies(provider: h.Provider, store: str):
    add = lambda: h.completion(  # noqa: E731
        calls=[("FunctionManager_add_functions", {"implementations": NAMED})],
    )

    def later():
        messages = provider.requests[-1]["messages"]
        review_start = any(
            "Review the trajectory" in str(m.get("content")) for m in messages
        ) and not any(m.get("role") == "assistant" for m in messages)
        return add() if store == "review" and review_start else h.completion("done")

    search = [
        ("FunctionManager_search_functions", {"query": "mirror"}),
        ("GuidanceManager_search", {}),
    ]
    first = [lambda: h.completion(calls=search)]
    if store == "actor":
        first.append(add)
    return [*first, *([later] * 12)]


async def _act(store: str) -> list[dict]:
    from unify.actor.code_act_actor import CodeActActor

    actor = CodeActActor()
    try:
        with h.scripted([]) as provider:
            provider.replies.extend(_act_replies(provider, store))
            handle = await actor.act(TASK, persist=False)
            await asyncio.wait_for(handle.result(), 60)
            await asyncio.wait_for(handle._lifecycle_task, 60)
    finally:
        await actor.close()
    return provider.requests


@pytest.mark.asyncio
@pytest.mark.timeout(180)
@pytest.mark.parametrize("store", ["actor", "review"])
@_handle_project
async def test_on_the_actor_and_its_review_are_refused_the_alias(
    lint,
    store,
    monkeypatch,
):
    lint(True)
    # The scripted session may end before the tool's result is sent back, so
    # the refusal is read where it is made.
    refused = []
    real = instance_lint.refusal

    def spy(name, problem):
        refused.append((name, problem))
        return real(name, problem)

    monkeypatch.setattr(instance_lint, "refusal", spy)
    await _act(store)
    assert refused and refused[0][0] == "mirror_7a4cf12e", refused
    assert "'7a4cf12e'" in refused[0][1]
    assert "mirror_7a4cf12e" not in FunctionManager().list_functions()
    assert not instance_lint.current()


def test_the_setting_defaults_off():
    assert ProductionSettings().UNIFY_STORE_INSTANCE_LINT is False
    assert (
        ProductionSettings(UNIFY_STORE_INSTANCE_LINT="1").UNIFY_STORE_INSTANCE_LINT
        is True
    )

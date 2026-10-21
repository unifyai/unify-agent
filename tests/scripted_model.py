"""A scripted model for deterministic tests: one script per kind of call.

A test that drives the harness with a model needs the model's decisions, not
the model. This module replaces unillm's provider call (the transport below
unillm's request building, so every request is built exactly as in
production) and answers each request from a script. Nothing leaves the
process, and no recording is read.

Besides the actor's own turns the harness makes other model calls (the
storage review's fork and its gate, context compression, ``query_llm`` from
cells, helpers, ...). A single ordered script breaks whenever one of those
appears, disappears or moves, so each request is classified by
:func:`kind_of`, from the request alone, and answered from that kind's
script. A test scripts only the kinds it is about; any other kind that
appears fails the test, naming the kind, unless the test allows it.

Typical use::

    model = ScriptedModel(
        actor=[reply(calls=[cell("x = 41\\nx + 1")]), reply("42")],
    )
    with scripted(model):
        handle = await actor.act("Compute.", can_store=False)
        assert await handle.result() == "42"
    assert model.kinds() == ["actor", "actor"]

Kinds, how each is recognised, and where the request is built
(``unify/`` paths, at harness-learning f64bfca2c):

* ``review``: the storage review forked from the session; a user message
  carries the fork's role (``code_act_actor._REVIEW_FORK_ROLE_UNIFIED``).
  Its tools and system prompt are the actor's own, so nothing else tells it
  apart.
* ``gate``: the review gate; no tools, system ``review_gate.GATE_SYSTEM_PROMPT``.
* ``compression_fork``: the summary turn of a compression; the last message
  is ``cache_discipline.COMPRESSION_FORK_INSTRUCTION``.
* ``compress_turn``: the forced turn at the context threshold; the loop's
  notice "Context window is nearly full." is the last message and the only
  tools are ``compress_context`` (and ``store_skills``). A request that only
  offers ``compress_context`` (a loop with no tools of its own) is not one.
* ``compactor``: ``context_compression.compress_messages``; system
  ``COMPRESSION_PROMPT``.
* ``standalone_review`` / ``proactive_storage``: the JSON-tool storage
  sessions (FunctionManager_/GuidanceManager_ tools), told apart by their
  system prompt.
* ``last_word``: ``UNIFY_STEP_CAP_REPLY=last_word``'s tool-less call.
* ``helper``: a helper's turn (``agents.spawn``); its first request names it
  "a helper in a team".
* ``query_llm``: any other request with no tools (cells' ``query_llm``).
* ``actor``: everything else.

Embeddings are not chat calls: fake them with
``tests/actor/code_act/library_world.install_fake_embed``.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import json
import os
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Optional, Union

import pytest
from openai.types.chat import ChatCompletion

MODEL = "openai/gpt-5.6-sol"
_IDS = itertools.count()

# Fixed fragments of requests whose text is not a module constant. Each is
# checked against the code that writes it (tests/test_scripted_model.py), so
# a reworded prompt fails that check instead of misrouting silently.
HELPER_FRAGMENT = ", a helper in a team. "
LAST_WORD_FRAGMENT = "The step limit for this request is reached"
# The notice the loop appends before its forced compress turn. Offering
# compress_context alone does not make a request one: a loop with no tools of
# its own offers it on demand.
COMPRESS_TURN_FRAGMENT = "Context window is nearly full."
STANDALONE_REVIEW_PREFIX = "You are the agent that just completed the task below."
PROACTIVE_STORAGE_PREFIX = "You are the agent executing the task below, and you asked"
COMPRESS_TOOLS = frozenset({"compress_context", "store_skills"})
STORAGE_TOOL_PREFIXES = ("FunctionManager_", "GuidanceManager_")

KINDS = (
    "review",
    "gate",
    "compression_fork",
    "compress_turn",
    "compactor",
    "standalone_review",
    "proactive_storage",
    "last_word",
    "helper",
    "query_llm",
    "actor",
)


# ── replies ──────────────────────────────────────────────────────────────────


def reply(
    content: Optional[str] = None,
    calls: Union[list, tuple] = (),
    *,
    finish_reason: Optional[str] = None,
    prompt_tokens: int = 100,
    cached_tokens: Optional[int] = None,
) -> ChatCompletion:
    """One provider reply. *calls* are ``(name, args)`` pairs, *args* a dict,
    or a string sent as the arguments as it is (malformed JSON)."""
    tool_calls = [
        {
            "id": f"call_{next(_IDS)}",
            "type": "function",
            "function": {
                "name": name,
                "arguments": args if isinstance(args, str) else json.dumps(args),
            },
        }
        for name, args in calls
    ]
    usage: dict[str, Any] = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": 5,
        "total_tokens": prompt_tokens + 5,
    }
    if cached_tokens is not None:
        usage["prompt_tokens_details"] = {"cached_tokens": cached_tokens}
    return ChatCompletion.model_validate(
        {
            "id": "cmpl-scripted",
            "object": "chat.completion",
            "created": 0,
            "model": MODEL,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": finish_reason
                    or ("tool_calls" if tool_calls else "stop"),
                    "message": {
                        "role": "assistant",
                        "content": content,
                        "tool_calls": tool_calls or None,
                    },
                },
            ],
            "usage": usage,
        },
    )


def cell(code: str, thought: str = "Running a step.") -> tuple[str, dict]:
    """An ``execute_code`` call for :func:`reply`'s *calls*."""
    return ("execute_code", {"thought": thought, "code": code})


@dataclass(frozen=True)
class Always:
    """A script entry that answers every request of its kind (it is never
    used up): ``ScriptedModel(gate=Always(reply("no")))``."""

    entry: Any


# ── classification ───────────────────────────────────────────────────────────


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(part.get("text", "")) if isinstance(part, dict) else str(part)
            for part in content
        )
    return "" if content is None else str(content)


def _markers() -> dict[str, str]:
    """The module constants that mark a kind, read from the code itself."""
    from unify.actor import code_act_actor, review_gate
    from unify.common._async_tool import cache_discipline, context_compression

    role = code_act_actor._REVIEW_FORK_ROLE_UNIFIED
    return {
        "review": role.split("\n", 1)[0],
        "gate": review_gate.GATE_SYSTEM_PROMPT[:80],
        "compression_fork": cache_discipline.COMPRESSION_FORK_INSTRUCTION[:80],
        "compactor": context_compression.COMPRESSION_PROMPT[:60],
    }


def kind_of(request: dict) -> str:
    """The kind of model call *request* (the transport's keyword arguments)
    is. Rules run most specific first; see the module docstring."""
    marks = _markers()
    messages = request.get("messages") or []
    tools = request.get("tools")
    names = {
        (t.get("function") or {}).get("name") or t.get("name")
        for t in (tools or [])
        if isinstance(t, dict)
    }
    system = "\n".join(
        _text(m.get("content")) for m in messages if m.get("role") == "system"
    )
    users = [_text(m.get("content")) for m in messages if m.get("role") == "user"]
    last = _text(messages[-1].get("content")) if messages else ""

    if any(marks["review"] in u for u in users):
        return "review"
    if not names and marks["gate"] in system:
        return "gate"
    if marks["compression_fork"] in last:
        return "compression_fork"
    if COMPRESS_TURN_FRAGMENT in last and names and names <= COMPRESS_TOOLS:
        return "compress_turn"
    if marks["compactor"] in system:
        return "compactor"
    if any(n and n.startswith(STORAGE_TOOL_PREFIXES) for n in names):
        if STANDALONE_REVIEW_PREFIX in system:
            return "standalone_review"
        if PROACTIVE_STORAGE_PREFIX in system:
            return "proactive_storage"
    if not names and LAST_WORD_FRAGMENT in last:
        return "last_word"
    if users and HELPER_FRAGMENT in users[0]:
        return "helper"
    if not names:
        return "query_llm"
    return "actor"


# ── the scripted transport ───────────────────────────────────────────────────


_RECORDED = (
    "messages",
    "tools",
    "tool_choice",
    "response_format",
    "reasoning_effort",
    "temperature",
    "parallel_tool_calls",
)


@dataclass
class Call:
    """One request the harness sent, its kind, and what answered it."""

    kind: str
    request: dict
    answer: Any = None

    @property
    def messages(self) -> list[dict]:
        return self.request.get("messages") or []

    @property
    def tool_names(self) -> list[str]:
        return [
            (t.get("function") or {}).get("name") or t.get("name")
            for t in self.request.get("tools") or []
        ]


class ScriptedModel:
    """Answers each request from its kind's script, in order.

    A script is a list of entries; an entry is a reply, an exception (raised,
    as a provider error is), a callable taking the :class:`Call` and
    returning either (or an awaitable of either), or :class:`Always`
    wrapping one of those. A request of a kind with no script, or whose
    script is used up, fails with an ``AssertionError`` naming the kind,
    unless the kind is in *allow*, whose requests are answered with
    ``reply("")`` (a turn with nothing to say).
    """

    def __init__(self, *, allow=(), **scripts: Any) -> None:
        unknown = (set(scripts) | set(allow)) - set(KINDS)
        if unknown:
            raise ValueError(f"unknown kinds {sorted(unknown)}; kinds are {KINDS}")
        self.scripts = {
            k: (list(v) if isinstance(v, (list, tuple)) else [v])
            for k, v in scripts.items()
        }
        self.allow = frozenset(allow)
        self.calls: list[Call] = []

    # The transport's signature: unillm calls it with the request's keyword
    # arguments (and its own session/client, ignored here).
    async def __call__(self, *, shared_session=None, client=None, **kw: Any) -> Any:
        # A JSON copy: what was sent, frozen at the time it was sent.
        request = json.loads(
            json.dumps({k: kw.get(k) for k in _RECORDED}, default=str),
        )
        call = Call(kind=kind_of(request), request=request)
        self.calls.append(call)
        entry = self._next(call)
        if callable(entry) and not isinstance(entry, ChatCompletion):
            entry = entry(call)
            if asyncio.iscoroutine(entry):
                entry = await entry
        call.answer = entry
        if isinstance(entry, BaseException):
            raise entry
        return entry

    def _next(self, call: Call) -> Any:
        script = self.scripts.get(call.kind)
        if script:
            head = script[0]
            if isinstance(head, Always):
                return head.entry
            return script.pop(0)
        if call.kind in self.allow:
            return reply("")
        seen = ", ".join(c.kind for c in self.calls)
        raise AssertionError(
            f"no scripted reply for a {call.kind!r} request (request "
            f"{len(self.calls)}; kinds so far: {seen})",
        )

    # ── reading what happened ────────────────────────────────────────────

    def kinds(self) -> list[str]:
        """The kind of every request, in order."""
        return [c.kind for c in self.calls]

    def of(self, kind: str) -> list[Call]:
        """The requests of *kind*, in order."""
        return [c for c in self.calls if c.kind == kind]

    def unused(self) -> dict[str, int]:
        """Scripted entries never asked for, per kind (``Always`` excluded)."""
        return {
            k: sum(1 for e in v if not isinstance(e, Always))
            for k, v in self.scripts.items()
            if any(not isinstance(e, Always) for e in v)
        }

    def assert_used_up(self) -> None:
        """Fail if any scripted entry was never asked for."""
        left = self.unused()
        assert not left, f"scripted replies never requested: {left}"


@contextlib.contextmanager
def scripted(
    model: ScriptedModel,
    *,
    below_retry: bool = False,
) -> Iterator[ScriptedModel]:
    """Install *model* as unillm's provider transport for the block.

    ``below_retry`` installs it as ``litellm.acompletion`` instead, under
    unillm's transient-retry policy, for tests of that policy. Recording is
    off inside the block: no client may read or write a recording.
    """
    import unillm.clients.uni_llm as uni_llm
    from unillm.settings import SETTINGS as unillm_settings

    old_env = os.environ.get("UNILLM_CACHE")
    old_setting = unillm_settings.UNILLM_CACHE
    os.environ["UNILLM_CACHE"] = "false"
    unillm_settings.UNILLM_CACHE = False
    if below_retry:
        original = uni_llm.litellm.acompletion
        uni_llm.litellm.acompletion = model
    else:
        original = uni_llm._acompletion_with_transient_retry
        uni_llm._acompletion_with_transient_retry = model
    try:
        yield model
    finally:
        if below_retry:
            uni_llm.litellm.acompletion = original
        else:
            uni_llm._acompletion_with_transient_retry = original
        unillm_settings.UNILLM_CACHE = old_setting
        if old_env is None:
            os.environ.pop("UNILLM_CACHE", None)
        else:
            os.environ["UNILLM_CACHE"] = old_env


@pytest.fixture
def scripted_model() -> Callable[..., ScriptedModel]:
    """``scripted_model(actor=[...], ...)`` builds a :class:`ScriptedModel`
    and installs it for the rest of the test."""
    stack = contextlib.ExitStack()

    def build(**kwargs: Any) -> ScriptedModel:
        return stack.enter_context(scripted(ScriptedModel(**kwargs)))

    with stack:
        yield build

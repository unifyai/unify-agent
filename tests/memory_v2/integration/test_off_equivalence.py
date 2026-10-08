"""With UNIFY_MEMORY_V2 off, a visit is byte-identical to the same visit with memory v2 absent (gate (b)).

One scripted visit runs through the CLI (``Act`` in process: ``--persist --jsonl``, a cell that reads a
workspace file, a reply, an outcome line, then ``{"quit": true}``) twice on fresh actors:

1. **off**: ``UNIFY_MEMORY_V2`` empty, the shipped code;
2. **absent**: every ``unify.memory_v2`` module is taken out of ``sys.modules`` and further imports of
   the package are refused (and counted). The harness's call sites (cli.py, code_act_actor.py,
   core_surface.py, worker.py) then reach a stand-in ``hooks`` whose every function returns its input
   unchanged or nothing, which is what the code did before the call sites existed. The one module left
   in place is ``integration.switch``, the settings validator, which only parses values.

Compared byte for byte, in order: every model request body (the canonical JSON, sorted keys, of all the
keyword arguments the harness hands the model transport: messages, tools and every parameter) across all
turns; and every ``--jsonl`` line the CLI wrote (the outcome answer, the response, the result, ended).
Cell outputs are inside the request bodies (each ``execute_code`` result is a tool message).

Named normalisations, and nothing else:

- The harness writes elapsed-time annotations into request bodies (``time_context``: user messages'
  ``[elapsed: ...]`` and tool results' ``called_at``/``duration``). Its clock,
  ``unify.common._async_tool.time_context.perf_counter``, is fixed at 0 in both runs, so those
  values are constant rather than excluded.
- Tool-call ids come from the scripted model, not the harness; its id counter restarts for each run.
- A transport argument that is not JSON (none is expected) is rendered as its type's name, never an
  address.
- The provider credential (``api_key``) is compared by a digest, so it is never kept or printed.

No request id or timestamp is excluded: the harness puts none in the body at this base (the body's
``extra_body.prompt_cache_key``/``session_id`` are derived from the request, and equal across the two
runs). If one appears, normalise that one field by name here, with the reason, rather than loosening the
comparison.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib.abc
import itertools
import json
import os
import sys
import types
from typing import Any

import pytest

import tests.scripted_model as scripted_model_mod
from tests.actor.code_act.core_world import (  # noqa: F401 (fixtures)
    core_world,
    new_actor,
    world,
)
from tests.actor.code_act.sandbox_world import needs_bwrap
from tests.helpers import _handle_project
from tests.scripted_model import ScriptedModel, cell, reply, scripted
from unify.common._async_tool import time_context
from unify.memory_v2.integration.paths import Paths
from unify.settings import SETTINGS

REQUEST = "How many lines does data.txt have?"
CODE = "print(len(open('data.txt').read().splitlines()))"
OUTCOME = {"outcome": {"solved": True, "checks": [{"name": "c", "passed": True}]}}


def _opaque(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return f"<{type(value).__module__}.{type(value).__qualname__}>"


#: Keys whose values are credentials, at any depth (transport arguments and headers): compared by digest, never
#: kept or printed. Matched on the key name, case-insensitively.
CREDENTIALS = ("api_key", "authorization", "x-api-key", "proxy-authorization")


def _redact_credentials(value: Any) -> Any:
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            if isinstance(k, str) and k.lower() in CREDENTIALS and v is not None:
                digest = hashlib.sha256(str(v).encode()).hexdigest()
                out[k] = f"<credential sha256:{digest[:16]}>"
            else:
                out[k] = _redact_credentials(v)
        return out
    if isinstance(value, (list, tuple)):
        return [_redact_credentials(v) for v in value]
    return value


def canonical(kw: dict) -> str:
    return json.dumps(
        _redact_credentials(dict(kw)),
        sort_keys=True,
        default=_opaque,
        ensure_ascii=False,
    )


def test_canonical_never_keeps_a_credential():
    body = {
        "api_key": "sk-or-v1-FAKE",  # pragma: allowlist secret
        "extra_headers": {"Authorization": "Bearer FAKE2"},  # pragma: allowlist secret
        "model": "m",
    }
    text = canonical(body)
    assert "FAKE" not in text and "<credential sha256:" in text


class _Recording(ScriptedModel):
    """The scripted transport, also keeping each request's full body as canonical JSON."""

    def __init__(self, **kw: Any) -> None:
        super().__init__(**kw)
        self.bodies: list[str] = []

    async def __call__(self, *, shared_session=None, client=None, **kw: Any) -> Any:
        self.bodies.append(canonical(kw))
        return await super().__call__(
            shared_session=shared_session,
            client=client,
            **kw,
        )


# ── memory v2 absent ─────────────────────────────────────────────────────────


def _identity_hooks() -> types.ModuleType:
    hooks = types.ModuleType("unify.memory_v2.integration.hooks")
    hooks.enabled = lambda: False
    hooks.can_store = lambda value: value
    hooks.system_prompt = lambda text: text
    hooks.sandbox_objects = lambda objects: objects
    hooks.worker_paths = lambda: []
    hooks.worker_mounts = lambda: []
    hooks.begin_request = lambda request: None
    hooks.worker_audit = lambda: None
    hooks.worker_cell_done = lambda events: None
    hooks.result_hook = lambda: None
    hooks.tool_result = lambda name, call_id, raw, *, raised=None: None

    def __getattr__(name: str):  # a new call site must be added here, deliberately
        raise AttributeError(
            f"the identity hooks have no {name!r}; add it to the OFF test",
        )

    hooks.__getattr__ = __getattr__
    return hooks


class _RefuseMemoryV2(importlib.abc.MetaPathFinder):
    def __init__(self) -> None:
        self.attempts: list[str] = []

    def find_spec(self, fullname, path=None, target=None):
        if fullname == "unify.memory_v2" or fullname.startswith("unify.memory_v2."):
            self.attempts.append(fullname)
            raise ImportError(f"memory v2 is absent in this run: {fullname}")
        return None


def _make_absent(monkeypatch) -> _RefuseMemoryV2:
    switch = sys.modules.get("unify.memory_v2.integration.switch")
    for name in [
        m
        for m in sys.modules
        if m == "unify.memory_v2" or m.startswith("unify.memory_v2.")
    ]:
        monkeypatch.delitem(sys.modules, name)
    pkg = types.ModuleType("unify.memory_v2")
    pkg.__path__ = []
    integration = types.ModuleType("unify.memory_v2.integration")
    integration.__path__ = []
    hooks = _identity_hooks()
    integration.hooks = hooks
    pkg.integration = integration
    stand_ins = {
        "unify.memory_v2": pkg,
        "unify.memory_v2.integration": integration,
        "unify.memory_v2.integration.hooks": hooks,
    }
    if switch is not None:
        integration.switch = switch
        stand_ins["unify.memory_v2.integration.switch"] = switch
    for name, mod in stand_ins.items():
        monkeypatch.setitem(sys.modules, name, mod)
    finder = _RefuseMemoryV2()
    monkeypatch.setattr(sys, "meta_path", [finder, *sys.meta_path])
    return finder


# ── one visit ────────────────────────────────────────────────────────────────


async def _visit(
    world,
    monkeypatch,
) -> tuple[list[str], list[str], list[str]]:  # noqa: F811
    """One scripted CLI visit; returns (request bodies, jsonl lines, call kinds)."""
    from unify.cli import Act, _parse_args

    monkeypatch.setattr(scripted_model_mod, "_IDS", itertools.count())
    model = _Recording(actor=[reply(calls=[cell(CODE)]), reply("10 lines.")])
    read_fd, write_fd = os.pipe()
    monkeypatch.setattr(sys, "stdin", os.fdopen(read_fd, "r"))
    args = _parse_args(
        [
            "act",
            "--persist",
            "--jsonl",
            "--no-clarify",
            "--no-store",
            "--quiet",
            REQUEST,
        ],
    )
    session = Act(args)
    lines: list[str] = []
    closed = False

    def emit(**payload: Any) -> None:
        nonlocal closed
        lines.append(json.dumps(payload, default=str))
        if payload.get("type") == "response" and not closed:
            closed = True
            os.write(write_fd, (json.dumps(OUTCOME) + "\n").encode())
            os.write(write_fd, b'{"quit": true}\n')
            os.close(write_fd)

    async def start() -> None:
        os.chdir(world["workspace"])
        session._actor = new_actor()

    monkeypatch.setattr(session, "_emit", emit)
    monkeypatch.setattr(session, "start", start)
    try:
        with scripted(model):
            code = await asyncio.wait_for(session.run(REQUEST), 180)
    finally:
        await session.close()
        if not closed:
            os.close(write_fd)
    assert code == 0, lines
    return model.bodies, lines, model.kinds()


def _first_difference(a: list[str], b: list[str]) -> str:
    if len(a) != len(b):
        return f"{len(a)} vs {len(b)} items"
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            at = (
                next(j for j in range(min(len(x), len(y))) if x[j] != y[j])
                if x[:1] == y[:1]
                else 0
            )
            return f"item {i} differs at {at}: {x[max(0, at - 120):at + 120]!r} vs {y[max(0, at - 120):at + 120]!r}"
    return ""


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(420)
@_handle_project
async def test_off_is_byte_identical_to_memory_v2_absent(core_world, monkeypatch):
    monkeypatch.setattr(time_context, "perf_counter", lambda: 0.0)
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY_V2", "")
    # the tool loop's memory-v2 hook: asked for at each finished call, None while off, never entered
    from unify.common._async_tool import tools_data
    from unify.memory_v2.integration import hooks as real_hooks

    resolved: list = []
    entered: list = []
    ask = real_hooks.result_hook

    def observed_result_hook():
        hook = ask()
        resolved.append(hook)
        return hook

    monkeypatch.setattr(real_hooks, "result_hook", observed_result_hook)
    monkeypatch.setattr(
        real_hooks,
        "tool_result",
        lambda *a, **k: entered.append(a),
    )
    off = await _visit(core_world, monkeypatch)
    assert resolved and all(hook is None for hook in resolved), resolved
    assert entered == []  # the hook was never entered
    assert tools_data._memory_v2_result_hook() is None

    # nothing of memory v2 was made under UNIFY_HOME
    paths = Paths.under(core_world["state"])
    for p in (paths.checkout, *paths.harness_only(), paths.events):
        assert not p.exists(), p

    finder = _make_absent(monkeypatch)
    absent = await _visit(core_world, monkeypatch)
    assert finder.attempts == []  # nothing of memory v2 was imported in the absent run

    off_bodies, off_lines, off_kinds = off
    absent_bodies, absent_lines, absent_kinds = absent
    assert off_kinds == absent_kinds == ["actor", "actor"]
    assert off_bodies == absent_bodies, _first_difference(off_bodies, absent_bodies)
    assert off_lines == absent_lines, _first_difference(off_lines, absent_lines)
    # the visit really ran: the cell's output reached the model, the outcome line was answered
    assert "10" in json.dumps(json.loads(off_bodies[1])["messages"][-1])
    assert [json.loads(x)["type"] for x in off_lines] == [
        "response",
        "outcome",
        "result",
        "ended",
    ]

"""With UNIFY_MEMORY off, a visit enters no memory hook and makes nothing of memory (gate (b)).

One scripted visit runs through the CLI (``Act`` in process: ``--persist --jsonl``, a cell that reads a
workspace file, a reply, an outcome line, then ``{"quit": true}``) with ``UNIFY_MEMORY=off``. The tool loop's memory
hook is asked for at each finished call and is None, it is never entered, nothing of memory is made under the home
(rev 2's run record, ``unify.agents``, is on in every run and is the one thing in the state dir), and the visit ran.

Rev 2 has no "memory absent" run to compare it with byte for byte: the actor's tool layer imports unify.memory's
reply, profile and redaction modules, so the package cannot be taken out of a run. :func:`canonical` (the request
bodies' canonical JSON, the provider credential compared by a digest only) is kept and tested on its own.
"""

from __future__ import annotations

import asyncio
import hashlib
import itertools
import json
import os
import sys
from typing import Any

import pytest

import tests.scripted_model as scripted_model_mod
from tests.actor.code_act.core_world import (  # noqa: F401 (fixtures)
    core_world,
    new_actor,
    world,
)
from tests.actor.code_act.sandbox_world import needs_bwrap
from tests.scripted_model import ScriptedModel, cell, reply, scripted
from unify.memory.integration.paths import Paths
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


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(420)
async def test_off_enters_no_memory_and_makes_nothing(core_world, monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY", "off")  # rev 2: empty means on
    # the tool loop's memory hook: asked for at each finished call, None while off, never entered
    from unify.common._async_tool import tools_data
    from unify.memory.integration import hooks as real_hooks

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
    assert tools_data._memory_result_hook() is None

    # nothing of memory v2 was made under UNIFY_HOME; rev 2's run record (unify.agents, on in every run, memory or
    # not) is the one thing in the state dir
    paths = Paths.under(core_world["state"])
    for p in (paths.checkout, *paths.harness_only(), paths.events):
        if p == paths.state_dir and p.exists():
            assert sorted(x.name for x in p.iterdir()) == ["agents"], sorted(p.rglob("*"))
            continue
        assert not p.exists(), p

    # rev 2: memory cannot be absent from a run (the actor's tool layer imports unify.memory's reply, profile and
    # redaction modules), so off is judged by the hook never entered and nothing of memory made, and by the visit
    off_bodies, off_lines, off_kinds = off
    assert off_kinds == ["actor", "actor"]
    # the visit really ran: the cell's output reached the model, the outcome line was answered
    assert "10" in json.dumps(json.loads(off_bodies[1])["messages"][-1])
    assert [json.loads(x)["type"] for x in off_lines] == [
        "response",
        "outcome",
        "result",
        "ended",
    ]

"""S0: no test of this suite can reach a real model: the guard holds in the test process and in the boxes, every
in-process model call goes to the scripted model, and every memory lane is a loopback route to the scripted lanes.

The lead's standing rule: tests never make real model API calls; no key, no proxy to a provider, no network to a model
API. The suite-wide guard is ``tests/conftest.py`` (``_no_provider_or_network``); this test fails if it stops holding.
"""

from __future__ import annotations

import os
import socket

import pytest

from tests.actor.code_act.sandbox_world import needs_bwrap
from tests.scripted_model import ScriptedModel, cell, reply
from tests.system import sysworld as SW

PROVIDER_KEYS = ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY",
                 "AZURE_API_KEY", "TOGETHER_API_KEY", "GROQ_API_KEY", "MISTRAL_API_KEY", "DEEPSEEK_API_KEY",
                 "XAI_API_KEY", "UNIFY_KEY")
#: Stand-ins the suite itself sets; never a credential.
PLACEHOLDERS = ("ci-placeholder-not-a-key",)
OFF_MACHINE = ("203.0.113.7", 443)  # TEST-NET-3: never a real host


def _no_key(env: dict) -> list[str]:
    return [k for k in PROVIDER_KEYS if env.get(k) and env.get(k) not in PLACEHOLDERS]


def test_the_test_process_cannot_reach_a_provider_or_the_network():
    import unillm.clients.uni_llm as uni_llm

    s = socket.socket()
    try:
        with pytest.raises(AssertionError):
            s.connect(OFF_MACHINE)
    finally:
        s.close()
    with pytest.raises(AssertionError):
        uni_llm.litellm.completion(model="openai/gpt-4o", messages=[{"role": "user", "content": "hi"}])
    assert not _no_key(dict(os.environ)), _no_key(dict(os.environ))


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(900)
@pytest.mark.parametrize("mode", ["on"])
async def test_every_model_call_of_a_memory_run_is_scripted_and_boxes_have_no_network(system, mode):
    import unillm.clients.uni_llm as uni_llm
    from unify.memory import writer_service as ws
    from unify.memory.integration import switch as sw
    from unify.settings import SETTINGS

    s = system
    seen: dict = {}
    box = SW.probe_json(
        "import socket, urllib.request\n"
        "_try('interfaces', lambda: sorted(n for _, n in socket.if_nameindex()))\n"
        "def _connect():\n"
        "    c = socket.socket(); c.settimeout(3)\n"
        f"    c.connect({OFF_MACHINE!r})\n"
        "    return 'connected'\n"
        "_try('connect', _connect)\n"
        "_try('provider', lambda: urllib.request.urlopen('https://openrouter.ai/api/v1/models', timeout=3).status)\n"
        "import os\n"
        f"_try('keys', lambda: [k for k in {PROVIDER_KEYS!r} if os.environ.get(k)])\n"
    )

    model = None

    def first(call):
        seen["transport_is_scripted"] = uni_llm._acompletion_with_transient_retry is model
        return reply(calls=[cell(box)])

    def second(call):
        seen["box"] = SW.marked(SW.last_tools(call))
        return reply(SW.words_answer(["a"]).__repr__())

    model = ScriptedModel(actor=[first, second, reply("Good.")])
    await s.request(SW.words_request(["a"]), model, SW.accepted_host())
    s.settle()
    assert seen["transport_is_scripted"], "the actor's model calls did not go to the scripted model"
    b = seen["box"]
    assert b["interfaces"] in (["lo"], []) or str(b["interfaces"]).startswith("RAISED"), b
    assert str(b["connect"]).startswith("RAISED") and str(b["provider"]).startswith("RAISED"), b
    assert b["keys"] == [], b
    # the memory lanes are loopback routes to the scripted lanes, and the fork really used one
    loop = f"http://127.0.0.1:{s.port}/"
    assert str(getattr(SETTINGS, sw.WRITER_BASE_URL, "") or os.environ.get(sw.WRITER_BASE_URL, "")).startswith(loop)
    assert s.lanes.of("fork"), "the fork's model call did not reach the scripted lane"
    assert all(c.path.startswith(("/fork/", "/writer/", "/helper/")) for c in s.lanes.calls)
    # the writer service (its own process) has no key and a loopback writer route
    doc = ws.read_state(s.home) or {}
    pid = doc.get("pid")
    if isinstance(pid, int) and os.path.exists(f"/proc/{pid}/environ"):
        env = dict(x.split("=", 1) for x in open(f"/proc/{pid}/environ", "rb").read().decode(errors="replace")
                   .split("\0") if "=" in x)
        assert not _no_key(env), _no_key(env)
        assert env.get(sw.WRITER_BASE_URL, loop).startswith(loop), env.get(sw.WRITER_BASE_URL)

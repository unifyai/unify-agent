"""A run without memory (``UNIFY_MEMORY=off``, the baseline) names no part of memory to the actor or a teammate.

Every text such an agent can receive is rendered with memory off: ``tools.list()``, every ``tools.help(name)`` and
help topic of its allow-list, the shared opening, every refusal its tools can give it (each template of its own texts,
or the shared ones where it has none, and the general forms), the record's common section, its roster lines, and what
``tools.who()`` and a refused ``tools.send`` say. None holds memory's vocabulary: memory, the writer, the librarian, a
fork, ``/inputs`` or attention. The texts with memory on are checked to hold it, so the scan reads real texts.
"""

from __future__ import annotations

import re
import string

import pytest

from unify.actor import tool_handlers as H
from unify.common import tool_registry as R
from unify.settings import SETTINGS

#: Memory's vocabulary: none of it reaches an agent of a run without memory.
MEMORY_WORDS = re.compile(r"(?i)memory|writer|librarian|\bforks?\b|/inputs|attention")
AGENTS = ("actor", "teammate")


def _fields(template: str) -> dict:
    """A value for every field *template* takes."""
    return {name: "x" for _, name, _, _ in string.Formatter().parse(template) if name}


def _refusal_texts(agent: str, memory: bool) -> dict[str, str]:
    """Every refusal *agent*'s tools can give it in a run with or without *memory*, rendered as its handlers render
    them (``tool_registry.role_refusal``)."""
    out = {}
    allow = R.allow_list(agent, H.actor_flags(agent, librarian=memory))
    for name in allow:
        spec = R.registry()[name]
        _own, texts = R._own_text(spec, "refusals", agent, memory)
        forms = {**R.GENERAL_REFUSALS, **(texts or {})}
        for form, template in forms.items():
            text = R.role_refusal(agent, name, form, memory=memory, **_fields(template))
            out[f"refusal {name}/{form}"] = str(text)
    return out


def _texts(agent: str, memory: bool, tmp_path) -> dict[str, str]:
    from unify.agents import run_record as rr
    from unify.agents import run_record_tools as rt
    from unify.common.tool_registry import RecordRefused

    flags = H.actor_flags(agent, librarian=memory)
    allow = R.allow_list(agent, flags)
    out = {
        "tools.list()": R.list_text(allow, agent, memory=memory),
        "opening": R.preamble(agent, {"flags": flags, "environment": "Your box: /workspace."}),
        "common section": R.common_section(memory),
    }
    for name in allow:
        out[f"tools.help({name})"] = R.help_text(name, agent, memory=memory)
    for topic, (roles, text) in R.help_topics(agent).items():
        out[f"tools.help({topic})"] = text
    out.update(_refusal_texts(agent, memory))
    rec = rr.join_run_record(tmp_path / f"{agent}-{memory}")
    rec.join("actor-1a2b", "actor", episode="e1")
    rec.join("h1", "helper", spawner="actor-1a2b")
    name, role, spawner = ("actor-1a2b", "actor", None) if agent == "actor" else ("h1", "helper", "actor-1a2b")
    view = rr.RunRecordView(rec, name, role, spawner=spawner)
    out["roster lines"] = rr.prompt_lines(name, role, spawner, home=rec.home)
    out["tools.who()"] = rt.who_text(view)
    # a message to no one: the refusal names the addresses that reach someone now
    if agent == "actor":
        try:
            view.post(name, "@nobody is anyone there?")
        except RecordRefused as exc:
            out["send refusal"] = exc.text
    return out


@pytest.mark.parametrize("agent", AGENTS)
def test_no_text_an_agent_of_a_run_without_memory_receives_names_memory(agent, tmp_path, monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY", "off")
    texts = _texts(agent, False, tmp_path)
    assert "send refusal" in texts or agent != "actor"
    found = {k: MEMORY_WORDS.findall(v) for k, v in texts.items() if MEMORY_WORDS.search(v)}
    assert not found, {k: (found[k], texts[k][:400]) for k in found}


@pytest.mark.parametrize("agent", AGENTS)
def test_the_scan_reads_texts_that_name_memory_when_the_run_has_it(agent, tmp_path, monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY", "on")
    texts = _texts(agent, True, tmp_path)
    assert MEMORY_WORDS.search(texts["tools.help(read)"]) and MEMORY_WORDS.search(texts["common section"])
    assert MEMORY_WORDS.search(texts["refusal read/outside_roots"])


@pytest.mark.parametrize("agent", AGENTS)
def test_a_read_refusal_is_the_agents_own_text_in_either_run(agent, monkeypatch):
    """The handlers give the role's own refusal (its ``tools.help`` shows the same forms): never the writer's
    template, which names its inputs and the library."""
    from types import SimpleNamespace

    for memory in (False, True):
        monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY", "on" if memory else "off")
        ctx = SimpleNamespace(agent=agent)
        text = H._refusal(ctx, "read", "not_found", path="/workspace/none.txt")
        assert text == "refused: /workspace/none.txt does not exist" and text.form == "not_found", text
        own = R._own_text(R.registry()["read"], "refusals", agent, memory)[1]
        assert set(own) >= {"outside_roots", "not_found", "unreadable"}
        assert "/inputs" not in str(H._refusal(ctx, "read", "outside_roots", path="/etc/x"))

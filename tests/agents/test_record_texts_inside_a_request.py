"""What the run's record tells the actor and its teammates, the agents inside a request.

* Neither is told to mention the librarian (they ask with ``memory.ask``) or offered an episode's fork as an address
  (it starts only after the episode): not in ``who``, a send refusal or the ``send`` help; the writer still is.
* ``who`` writes each "whom to mention, and when" line as the prompt's roster lines do.
* A send refusal offers only addresses that reach someone now, and words a missing role from the record as it is.
* The ``ask_library`` help states the bound on librarian sessions a later teammate can meet, and what to do then.
"""

from __future__ import annotations

import re

import pytest

from unify.common import tool_registry as R
from unify.memory import librarian


def test_the_session_bound_the_ask_help_states_is_the_librarians():
    assert R.MAX_ASK_SESSIONS == librarian.MAX_ASK_SESSIONS
    text = R.help_text("ask_library", "teammate", memory=True)
    assert f"at most {librarian.MAX_ASK_SESSIONS} librarian sessions" in text and "(@actor)" in text, text


def test_no_agent_inside_a_request_is_told_to_mention_the_librarian_or_offered_its_fork(tmp_path, monkeypatch):
    from unify.agents import run_record as rr
    from unify.agents import run_record_tools as rt

    assert all(m["who"] != "librarian" for m in rr.SHOULD_MENTION["actor"])
    monkeypatch.setattr(rr, "run_has_memory", lambda *a, **k: True)
    rec = rr.join_run_record(tmp_path)
    rec.join("actor-1a2b", "actor", episode="e1")
    rec.join("h1", "helper", spawner="actor-1a2b")
    rec.join("h2", "helper", spawner="h1")
    rec.join("writer", "writer", state="idle")
    rec.join("fork-1a2b", "fork", episode="e1")
    fork_role = re.compile(r"@fork(?![\w-])")  # the role address (a fork running now stays reachable by its name)
    for name, role, spawner in (("actor-1a2b", "actor", None), ("h1", "helper", "actor-1a2b"), ("h2", "helper", "h1")):
        view = rr.RunRecordView(rec, name, role, spawner=spawner)
        addresses = view.addresses()
        assert not fork_role.search(addresses) and "@writer" in addresses, (name, addresses)
        who = rt.who_text(view)
        assert not fork_role.search(who) and "@librarian" not in who, (name, who)
    writer = rr.RunRecordView(rec, "writer", "writer")
    assert "@fork" in rt.who_text(writer)
    for agent in ("actor", "teammate"):
        assert not fork_role.search(R.help_text("send", agent, memory=True))


def test_who_writes_each_mention_as_the_prompt_does(tmp_path, monkeypatch):
    from unify.agents import run_record as rr
    from unify.agents import run_record_tools as rt

    monkeypatch.setattr(rr, "run_has_memory", lambda *a, **k: True)
    rec = rr.join_run_record(tmp_path)
    rec.join("actor-1a2b", "actor", episode="e1")
    rec.join("h1", "helper", spawner="actor-1a2b")
    view = rr.RunRecordView(rec, "h1", "helper", spawner="actor-1a2b")
    who = rt.who_text(view)
    line = rr.mention_line("actor-1a2b", "to report findings, with their sources")
    assert line == "Mention @actor-1a2b to report findings, with their sources."
    assert line in who and "when to report" not in who, who
    assert line in rr.prompt_lines("h1", "helper", "actor-1a2b", home=tmp_path)


def test_a_send_refusal_offers_only_addresses_that_reach_someone(tmp_path, monkeypatch):
    from unify.agents import run_record as rr
    from unify.common.tool_registry import RecordRefused

    monkeypatch.setattr(rr, "run_has_memory", lambda *a, **k: True)
    rec = rr.join_run_record(tmp_path)
    rec.join("actor-1a2b", "actor", episode="e1")
    view = rr.RunRecordView(rec, "actor-1a2b", "actor")
    # no writer on the record: @writer matches no one, so it is never offered
    with pytest.raises(RecordRefused) as nobody:
        view.post("actor-1a2b", "@writer the reader failed")
    assert nobody.value.form == "no_recipient" and "no other agent is on the record now" in nobody.value.text
    rec.join("h1", "helper", spawner="actor-1a2b")
    with pytest.raises(RecordRefused) as named:
        view.post("actor-1a2b", "@writer the reader failed")
    offered = named.value.text.split("address it with ", 1)[1]
    assert offered == "@h1, @helpers", named.value.text
    # a librarian instance running while the role is not kept: the refusal says so, and offers it by name
    rec.join("librarian-b7f4", "librarian", episode="e1")
    with pytest.raises(RecordRefused) as no_role:
        view.post("actor-1a2b", "@librarian which item fits?")
    text = no_role.value.text
    assert no_role.value.form == "no_role" and "there is no librarian in this run" not in text, text
    assert "keeps no librarian role" in text and "@librarian-b7f4" in text, text

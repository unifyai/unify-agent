"""The fork is not told to mention the writer (rev 2.3).

In rev 2.2 the fork's roster line said "Mention @writer to point at your summary and drafts …"; its posts became
``message`` entries and 35–41 % of writer spend went to passes on them, most ending ``nothing_to_keep``. The writer
reads the episode, its summary and drafts through the episode's own entry. The fork may still post; nothing tells it to.
The other roles' lines are unchanged.
"""

from __future__ import annotations


def test_the_fork_has_no_mention_line_and_the_others_keep_theirs(tmp_path, monkeypatch):
    from unify.agents import run_record as rr

    monkeypatch.setattr(rr, "run_has_memory", lambda *a, **k: True)
    assert "fork" not in rr.SHOULD_MENTION
    fork = rr.prompt_lines("fork-1a2b", "fork", home=tmp_path)
    assert "Mention @" not in fork and fork.rstrip().endswith("On the record you are `fork-1a2b` (fork).")
    actor = rr.prompt_lines("actor-1a2b", "actor", home=tmp_path)
    assert "Mention @writer" in actor
    assert rr.SHOULD_MENTION["writer"] and rr.SHOULD_MENTION["actor"]

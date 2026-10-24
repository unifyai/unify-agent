"""Held back after a rejection: an in-request serving rule, never trust.

When the world rejects an answer that a request attributes to an entry (by exact provenance or by value identity), that
entry has a bad use, and every entry linked to it by a recorded derivation link (a shared recorded episode: the link a
card shows as "generalised by"; an item landed from a capture or merged from others carries their episodes) is held back
from the librarian's fits until an answer is accepted, or a rejection is attributed to another entry (whose links are
then held instead). The question names them and the answer may not. Inside the request only: nothing is stored as
trust.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from tests.memory.integration.test_request import _abort, _begin, mv2  # noqa: F401
from unify.memory import fixtures
from unify.memory import librarian as L
from unify.memory.integration import request as R

CAP = "capture:00112233445566aa"
GEN = "memory.shapes.fill:fill_empty"
OTHER = "memory.text.words:count_words"
EP = "20261024T000000Z-aaaaaaaa"


def _digest(value) -> str:
    return hashlib.sha256(fixtures.canonical(value).encode("utf-8", "surrogatepass")).hexdigest()[:16]


def _copy(root):
    (root / ".memory" / "captures" / CAP[8:]).mkdir(parents=True)
    (root / ".memory" / "captures" / CAP[8:] / "record.json").write_text(json.dumps({"episodes": [EP]}))
    rows = {GEN: {"kind": "function", "episodes": [EP, "20261024T000001Z-bbbbbbbb"]},
            OTHER: {"kind": "function", "episodes": ["20261024T000002Z-cccccccc"]}}  # fmt: skip
    (root / ".memory" / "items.json").write_text(json.dumps({"items": rows}))
    return root


def test_the_derivation_link_is_a_shared_recorded_episode(tmp_path):
    copy = _copy(tmp_path / "copy")
    assert R.derivation_links(copy, CAP) == {GEN}
    assert R.derivation_links(copy, GEN) == {CAP}
    assert R.derivation_links(copy, OTHER) == set()
    assert R.derivation_links(tmp_path / "missing", CAP) == set()


def _say(run, reply: str) -> None:
    from unify import transcripts

    path = transcripts.transcripts_dir() / f"{run.episode_id}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"type": "message", "message": {"role": "user", "content": "the request"}},
            {"type": "message", "message": {"role": "assistant", "content": reply}}]  # fmt: skip
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_a_rejection_holds_back_what_is_linked_until_the_next_verdict(mv2, monkeypatch, tmp_path):  # noqa: F811
    run = _begin(mv2, "the request")
    try:
        copy = _copy(tmp_path / "copy")
        monkeypatch.setattr(R.RequestRun, "copy_root", property(lambda self: copy))
        grid = [[7, 7, 7], [7, 7, 7]]
        run.note_calls({"calls": {"results": [{"item": CAP, "outcome": "ok", "digest": _digest(grid), "input": ""}]}})
        _say(run, json.dumps({"action": "submit", "grid": grid}))
        run.take_checker({"label": "fail"})
        assert list(run.bad_uses()) == [CAP] and run.held() == {GEN: CAP}
        # the standing fit was the generalising item: the gate opens on it (re-ask on new evidence)
        run.standing_fit = GEN
        assert run.reask_events("the request") == ["outcome", "failed_use"]
        # a rejection attributed to nothing keeps the hold (the loop it stops goes on through such answers)
        _say(run, json.dumps({"action": "submit", "grid": [[1]]}))
        run.take_checker({"label": "fail"})
        assert run.held() == {GEN: CAP}
        # an accepted answer: the linked item is offered again; the rejected entry's own bad use stands
        run.take_checker({"label": "pass"})
        assert run.held() == {} and list(run.bad_uses()) == [CAP]
    finally:
        _abort(mv2, run)


def test_a_rejection_attributed_to_another_entry_holds_its_links_instead(mv2, monkeypatch, tmp_path):  # noqa: F811
    run = _begin(mv2, "the request")
    try:
        copy = _copy(tmp_path / "copy")
        monkeypatch.setattr(R.RequestRun, "copy_root", property(lambda self: copy))
        first, second = [[7, 7, 7], [7, 7, 7]], [[8, 8, 8], [8, 8, 8]]
        run.note_calls({"calls": {"results": [{"item": CAP, "outcome": "ok", "digest": _digest(first), "input": ""},
                                              {"item": GEN, "outcome": "ok", "digest": _digest(second),
                                               "input": ""}]}})  # fmt: skip
        _say(run, json.dumps({"action": "submit", "grid": first}))
        run.take_checker({"label": "fail"})
        assert run.held() == {GEN: CAP}
        _say(run, json.dumps({"action": "submit", "grid": second}))
        run.take_checker({"label": "fail"})  # attributed to GEN: its own links are held now, and it has a bad use
        assert run.held() == {} and sorted(run.bad_uses()) == sorted([CAP, GEN])
    finally:
        _abort(mv2, run)


def test_the_ask_tool_sends_what_is_held(monkeypatch):
    from types import SimpleNamespace

    from unify.actor import tool_handlers as H

    monkeypatch.setattr(R, "current", lambda: SimpleNamespace(held=lambda: {GEN: CAP}))
    assert H._memory_run_held() == {GEN: CAP}
    monkeypatch.setattr(R, "current", lambda: None)
    assert H._memory_run_held() == {}


FIT = {"id": GEN, "call": f"memory.get({GEN!r})(grid)", "why": "w"}


def test_a_held_item_is_refused_as_a_fit_and_may_be_named_as_nearest():
    got, why = L.check_answer({"fits": [FIT]}, [GEN, CAP], held={GEN: CAP})
    assert got is None and "held back in this request until an answer is accepted" in why and CAP in why
    nearest = {"fits": [], "nearest": [{"id": GEN, "differs": "held back"}]}
    assert L.check_answer(nearest, [GEN, CAP], held={GEN: CAP})[1] == ""
    assert L.check_answer({"fits": [FIT]}, [GEN, CAP])[1] == ""  # nothing held: offered as before


def test_the_question_names_what_is_held_and_its_key_changes_with_it():
    one = L.sent_to_librarian("q", {"commit": "c"})
    two = L.sent_to_librarian("q", {"commit": "c"}, held={GEN: CAP})
    assert "Held back in this request" not in one["message"]
    assert f"{GEN} (with {CAP})" in two["message"] and L.answer_key(one) != L.answer_key(two)


@pytest.mark.parametrize("label", ["pass", "fail"])
def test_nothing_is_held_without_a_rejection_or_without_a_link(mv2, monkeypatch, tmp_path, label):  # noqa: F811
    run = _begin(mv2, "the request")
    try:
        copy = _copy(tmp_path / "copy")
        monkeypatch.setattr(R.RequestRun, "copy_root", property(lambda self: copy))
        item = CAP if label == "pass" else OTHER
        run.note_calls({"calls": {"results": [{"item": item, "outcome": "ok", "digest": _digest([1, 2, 3, 4, 5]),
                                               "input": ""}]}})  # fmt: skip
        _say(run, "[1, 2, 3, 4, 5]")
        run.take_checker({"label": label})
        assert run.held() == {}
    finally:
        _abort(mv2, run)

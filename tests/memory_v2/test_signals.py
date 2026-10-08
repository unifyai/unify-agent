import pytest
from unify.memory_v2.evidence import EvidenceStore
from unify.memory_v2.gitio import Repo
from unify.memory_v2.signals import (
    Signal,
    SignalMasked,
    job_item_status,
    post_signal,
    tainted_items,
)
from tests.memory_v2.test_episodes import _ep


@pytest.fixture
def store(tmp_path):
    ev = EvidenceStore(tmp_path / "evidence.sqlite")
    for i in range(4):
        ev.index_episode(_ep(episode_id=f"e{i}"), commit_sha=f"{i:040d}")
    return ev


def test_regime_masks_checker_in_implicit(tmp_path, store):
    repo = Repo.init_bare(tmp_path / "ep.git")
    store.index_episode(_ep(episode_id="imp", regime="implicit"), "5" * 40)
    store.index_episode(_ep(episode_id="non", regime="none"), "6" * 40)
    with pytest.raises(SignalMasked):
        post_signal(
            Signal("s1", "imp", "checker", "pass", "t", regime="implicit"),
            repo,
            repo.head(),
            store,
        )
    post_signal(
        Signal("s2", "non", "environment", "error", "t", regime="none"),
        repo,
        repo.head(),
        store,
    )
    assert repo.notes(repo.head()) and store.signals_for("non")[0].label == "error"


def test_promotion_needs_two_distinct_supporting_episodes(store):
    store.add_item_evidence("workflows/pay.md", "e0", "source")
    store.add_item_evidence("workflows/pay.md", "e1", "source")
    store.add_signal(Signal("a", "e0", "checker", "pass", "t"))
    assert (
        job_item_status("workflows/pay.md", store, reader_promotes=False)
        == "insufficient"
    )
    store.add_signal(Signal("b", "e1", "provenance", "support", "t", regime="implicit"))
    assert (
        job_item_status("workflows/pay.md", store, reader_promotes=False)
        == "promotable"
    )


def test_one_correction_hides_and_reader_accept_needs_flag(store):
    store.add_item_evidence("w", "e0", "source")
    store.add_item_evidence("w", "e1", "source")
    store.add_signal(Signal("a", "e0", "reader", "accept", "t", regime="implicit"))
    store.add_signal(Signal("b", "e1", "reader", "accept", "t", regime="implicit"))
    assert job_item_status("w", store, reader_promotes=False) == "insufficient"
    assert job_item_status("w", store, reader_promotes=True) == "promotable"
    store.add_signal(Signal("c", "e1", "provenance", "correct", "t", regime="implicit"))
    assert job_item_status("w", store, reader_promotes=True) == "hidden"
    assert tainted_items("e1", store) == ["w"]


def test_abandon_and_clarify_are_neutral(store):
    store.add_item_evidence("w", "e0", "source")
    store.add_signal(Signal("a", "e0", "reader", "abandon", "t", regime="implicit"))
    store.add_signal(Signal("b", "e0", "reader", "clarify", "t", regime="implicit"))
    assert job_item_status("w", store, reader_promotes=True) == "insufficient"


def test_regime_mismatch_unknown_regime_and_unknown_episode_refused(tmp_path):
    ev = EvidenceStore(tmp_path / "x.sqlite")
    ev.index_episode(_ep(episode_id="i", regime="implicit"), "1" * 40)
    ev.index_episode(_ep(episode_id="w", regime="weird"), "2" * 40)
    repo = Repo.init_bare(tmp_path / "ep2.git")
    with pytest.raises(SignalMasked):
        post_signal(
            Signal("m", "i", "checker", "pass", "t", regime="dense"),
            repo,
            repo.head(),
            ev,
        )
    with pytest.raises(SignalMasked):
        post_signal(
            Signal("u", "w", "checker", "pass", "t", regime="weird"),
            repo,
            repo.head(),
            ev,
        )
    with pytest.raises(ValueError):
        post_signal(Signal("n", "nope", "checker", "pass", "t"), repo, repo.head(), ev)
    assert ev.signals_for("i") == [] and ev.signals_for("w") == []


@pytest.mark.parametrize(
    "source,label",
    [
        ("checker", "fail"),
        ("provenance", "correct"),
        ("reader", "correct"),
        ("reader", "re_ask"),
        ("recurrence", "re_ask"),
    ],
)
def test_each_contrary_label_hides(store, source, label):
    store.add_item_evidence("w", "e0", "source")
    store.add_signal(Signal("c", "e0", source, label, "t"))
    assert job_item_status("w", store, reader_promotes=True) == "hidden"

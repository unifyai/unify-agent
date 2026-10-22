import pytest
from unify.memory.evidence import EvidenceStore
from unify.memory.gitio import Repo
from unify.memory.signals import (
    Signal,
    SignalMasked,
    post_signal,
    tainted_items,
)
from tests.memory.test_episodes import _ep


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


def test_a_correction_taints_the_items_of_its_episode(store):
    store.add_item_evidence("w", "e0", "source")
    store.add_item_evidence("w", "e1", "source")
    store.add_signal(Signal("c", "e1", "provenance", "correct", "t", regime="implicit"))
    assert tainted_items("e1", store) == ["w"]

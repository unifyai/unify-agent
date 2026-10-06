"""Symbolic: in record mode a sandboxed cell can read, not write, its own run's
record, and never another run's (no channel between runs)."""

from unify import sandbox
from unify.settings import SETTINGS


def test_off_mounts_no_records(tmp_path, monkeypatch):
    monkeypatch.setenv("UNIFY_HOME", str(tmp_path))
    monkeypatch.setattr(SETTINGS, "UNIFY_AGENTS", "")
    policy = sandbox.build_policy(fresh=True)
    assert not any("records" in str(p) for p in policy.readonly_state)
    assert not (tmp_path / "records").exists()


def test_record_mode_mounts_only_the_current_runs_record(tmp_path, monkeypatch):
    from unify.agents import binding

    monkeypatch.setenv("UNIFY_HOME", str(tmp_path))
    monkeypatch.setattr(SETTINGS, "UNIFY_AGENTS", "record")
    earlier = binding.bind_for_act(request="an earlier run", user_reads=False)
    current = binding.bind_for_act(request="this run", user_reads=False)
    policy = sandbox.build_policy(fresh=True)
    mounted = [p for p in policy.readonly_state if "records" in str(p)]
    assert mounted == [current.pool.record.path.parent.resolve()]
    assert earlier.pool.record.path.parent.resolve() not in policy.readonly_state
    assert current.pool.record.path.name == "record.jsonl"

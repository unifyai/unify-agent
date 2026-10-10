"""Symbolic: in record mode a sandboxed cell can read, not write, its own run's
record, and never another run's (no channel between runs)."""

from unify import sandbox


def test_record_mode_mounts_only_the_current_runs_record(tmp_path, monkeypatch):
    from unify.agents import binding

    monkeypatch.setenv("UNIFY_HOME", str(tmp_path))
    earlier = binding.bind_for_act(request="an earlier run", user_reads=False)
    current = binding.bind_for_act(request="this run", user_reads=False)
    policy = sandbox.build_policy(fresh=True)
    # (Matched by folder, not by name: a test's own store file may carry "records".)
    records = str((tmp_path / "records").resolve())
    mounted = [p for p in policy.readonly_state if str(p).startswith(records)]
    assert mounted == [current.pool.record.path.parent.resolve()]
    assert earlier.pool.record.path.parent.resolve() not in policy.readonly_state
    assert current.pool.record.path.name == "record.jsonl"

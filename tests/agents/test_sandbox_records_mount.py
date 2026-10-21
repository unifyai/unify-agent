"""Symbolic: no sandboxed cell can read the run's record, this run's or another's: the run record lives in the
harness-only state directory and no box binds it (agents reach it only through their tools)."""

from unify import sandbox


def test_no_run_record_is_mounted_in_a_cell(tmp_path, monkeypatch):
    from unify.agents import binding, run_record as rr

    monkeypatch.setenv("UNIFY_HOME", str(tmp_path))
    earlier = binding.bind_for_act(request="an earlier run", user_reads=False)
    current = binding.bind_for_act(request="this run", user_reads=False)
    policy = sandbox.build_policy(fresh=True)
    for b in (earlier, current):
        folder = rr.agents_dir(b.pool.record.home).resolve()
        folder.mkdir(parents=True, exist_ok=True)
        assert not any(str(p).startswith(str(folder)) for p in policy.readonly_state), (folder, policy.readonly_state)
        assert policy.readable_violation(folder / rr.RECORD_FILE) is not None

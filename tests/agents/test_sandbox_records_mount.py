"""Symbolic: in record mode a sandboxed cell can read, not write, the run records."""

import pytest

from unify import sandbox
from unify.settings import SETTINGS


@pytest.mark.parametrize("mode, mounted", [("record", True), ("", False)])
def test_the_records_folder_is_read_only_in_record_mode_only(
    tmp_path,
    monkeypatch,
    mode,
    mounted,
):
    monkeypatch.setenv("UNIFY_HOME", str(tmp_path))
    monkeypatch.setattr(SETTINGS, "UNIFY_AGENTS", mode)
    policy = sandbox.build_policy(fresh=True)
    records = (tmp_path / "records").resolve()
    assert (records in policy.readonly_state) is mounted
    assert records.exists() is mounted

"""Shutdown: stop the writer service and verify its end (rev 2 has no per-request pass worker to wait for), never
start a pass, report every pass that ended after the last request, uncounted, and archive the memory repo with its
notes once nothing writes."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from unify.memory.integration import async_pass as ap
from unify.memory.integration.paths import Paths

def test_drain_archives_the_repo_with_its_notes_once_no_worker_writes(tmp_path):
    from unify.memory.gitio import Repo

    paths = Paths.under(tmp_path)
    mem = Repo.init_bare(paths.memory)
    mem.add_note(
        mem.head(),
        "{}",
        ref="items",
    )  # the item records live on refs/notes/items
    report = ap.drain(paths, archive_to=tmp_path / "out" / "memory.bundle")
    refs = report["archive"]["refs"]
    assert "refs/notes/items" in refs and "refs/heads/main" in refs


@pytest.mark.timeout(180)
def test_the_cli_prints_the_report(tmp_path):
    out = subprocess.run(
        [
            sys.executable,
            "-m",
            "unify.memory.integration.async_pass",
            "drain",
            "--home",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert out.returncode == 0 and json.loads(out.stdout)["phase"] == "shutdown"

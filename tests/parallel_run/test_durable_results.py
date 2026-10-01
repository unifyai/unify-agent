"""Exit receipts survive session disappearance without guessing missing outcomes."""

import importlib.util
import os
import subprocess
import uuid
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "parallel_run.sh"
HELPER = SCRIPT.with_name("_parallel_result.py")


def publisher():
    spec = importlib.util.spec_from_file_location("parallel_result_fixture", HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.publish


def accounting(tmp_path, receipt=None, outcome=None, repeat=False):
    # Extract the real reporting functions, not the runner's bootstrap or env.
    source = SCRIPT.read_text()
    start = source.index("_is_reported() {")
    end = source.index("\n_cleanup_sessions() {", start)
    sid = "$synthetic"
    socket = "receipt-fixture-" + uuid.uuid4().hex
    (tmp_path / "starts").write_text(f"{sid} 1\n")
    (tmp_path / "results").touch()
    result = tmp_path / "0.exit"
    if receipt is not None:
        result.write_text(receipt)
    outcome_path = Path(f"/tmp/parallel_run_outcome_{socket}_{sid}.txt")
    if outcome is not None:
        outcome_path.write_text(outcome)
    shell = """set -euo pipefail
START_TIMES_FILE="$1/starts"
RESULTS_FILE="$1/results"
TMUX_SOCKET="$2"
REPORTED_COMPLETIONS=":"
CREATED_SESSION_IDS=('$synthetic')
CREATED_RESULT_PATHS=("$1/0.exit")
CREATED_SESSION_NAMES=('exact-node')
"""
    shell += source[start:end]
    shell += """
# A vanished tmux session must not prevent receipt accounting.
tmux_cmd() { return 1; }
date() { printf '%s\\n' 2; }
report_completed_sessions
if [[ "$3" == repeat ]]; then report_completed_sessions; fi
printf 'REPORTED=%s\\n' "$REPORTED_COMPLETIONS"
"""
    try:
        completed = subprocess.run(
            [
                "bash",
                "--noprofile",
                "--norc",
                "-c",
                shell,
                "fixture",
                str(tmp_path),
                socket,
                "repeat" if repeat else "once",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            env={"PATH": "/usr/bin:/bin", "LC_ALL": "C.UTF-8"},
        )
    finally:
        outcome_path.unlink(missing_ok=True)
    assert completed.returncode == 0, completed.stderr
    return (tmp_path / "results").read_text().splitlines(), completed.stdout


@pytest.mark.parametrize("code,status", [(0, "pass"), (1, "fail"), (124, "fail")])
def test_disappeared_session_uses_recorded_exit(tmp_path, code, status):
    rows, _ = accounting(tmp_path, f"{code}\n", repeat=True)
    assert len(rows) == 1
    assert f"|{status}|" in rows[0]
    assert rows[0].endswith("|exact-node")


def test_all_skipped_retains_existing_outcome_contract(tmp_path):
    rows, _ = accounting(tmp_path, "0\n", "0|1\n")
    assert len(rows) == 1
    assert "|skip|" in rows[0]


@pytest.mark.parametrize("receipt", [None, "unknown\n", "0\n1\n", "256\n", "-1\n"])
def test_absent_or_invalid_receipt_never_invents_outcome(tmp_path, receipt):
    rows, trace = accounting(tmp_path, receipt)
    assert rows == []
    assert "REPORTED=:$synthetic:" not in trace


def test_publication_is_create_only_and_keeps_original(tmp_path):
    target = tmp_path / "0.exit"
    publish = publisher()
    publish(target, 1)
    assert target.read_text() == "1\n"
    with pytest.raises(FileExistsError):
        publish(target, 0)
    assert target.read_text() == "1\n"
    assert list(tmp_path.glob(".exit-*")) == []


def test_publication_does_not_follow_receipt_symlink(tmp_path):
    foreign = tmp_path / "foreign"
    foreign.write_text("retained")
    target = tmp_path / "0.exit"
    target.symlink_to(foreign)
    with pytest.raises(FileExistsError):
        publisher()(target, 0)
    assert foreign.read_text() == "retained"


def test_atomic_publish_failure_leaves_no_receipt(tmp_path, monkeypatch):
    publish = publisher()

    def fail_link(*args):
        raise OSError("injected publication failure")

    monkeypatch.setattr(os, "link", fail_link)
    with pytest.raises(OSError, match="publication failure"):
        publish(tmp_path / "0.exit", 0)
    assert list(tmp_path.iterdir()) == []


def test_repeat_occurrences_keep_separate_receipts(tmp_path):
    publish = publisher()
    publish(tmp_path / "0.exit", 0)
    publish(tmp_path / "1.exit", 1)
    assert (tmp_path / "0.exit").read_text() == "0\n"
    assert (tmp_path / "1.exit").read_text() == "1\n"

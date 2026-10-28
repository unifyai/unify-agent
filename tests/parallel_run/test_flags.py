"""
Individual flag tests for parallel_run.sh.

Tests each flag in isolation:
- --timeout / -t
- --serial / -s
- --match / -m
- --env / -e
- --jobs / -j
- --repeat
- --help / -h
- -- (pytest passthrough)
"""

from __future__ import annotations

import shutil
import time

import pytest


class TestBlockingBehavior:
    """Tests for default blocking behavior (script always blocks until completion)."""

    def test_blocks_until_completion(self, runner):
        """Script should block until all tests complete."""
        start = time.time()
        result = runner.run(
            runner.fixture_path("test_always_pass.py"),
        )
        elapsed = time.time() - start

        # Should have blocked for some time (tests take at least a moment)
        # but not forever (300s upper bound to handle resource contention)
        assert elapsed > 0.5, "Should have waited for tests"
        assert elapsed < 300, "Should not wait forever"

    def test_returns_zero_on_all_pass(self, runner):
        """Script should return 0 when all tests pass."""
        result = runner.run(
            runner.fixture_path("test_always_pass.py"),
        )

        assert (
            result.exit_code == 0
        ), f"Should return 0 on success, got {result.exit_code}. stderr: {result.stderr}"

    def test_returns_nonzero_on_failure(self, runner):
        """Script should return non-zero when any test fails."""
        result = runner.run(
            runner.fixture_path("test_always_fail.py"),
        )

        assert (
            result.exit_code != 0
        ), f"Should return non-zero on failure, got {result.exit_code}"

    def test_returns_nonzero_on_mixed_results(self, runner):
        """Script should return non-zero when some tests fail."""
        result = runner.run(
            runner.fixture_path("test_mixed_results.py"),
        )

        assert result.exit_code != 0, "Should return non-zero on partial failure"

    def test_creates_log_files(self, runner):
        """Script should create log files in logs/pytest/{socket}/."""
        result = runner.run(
            runner.fixture_path("test_always_pass.py"),
        )

        # Should have created at least one log file
        assert len(result.log_files) >= 1, "Should create log files"


class TestTimeoutFlag:
    """Tests for --timeout / -t flag."""

    def test_timeout_completes_normally(self, runner):
        """--timeout N should complete normally if tests finish before timeout."""
        import time

        start = time.time()
        result = runner.run(
            "--timeout",
            "300",  # 5 minute timeout - handles stress test scenarios
            runner.fixture_path("test_always_pass.py"),
        )
        elapsed = time.time() - start

        # Should complete successfully (not timeout)
        assert result.exit_code == 0, f"Should pass within timeout: {result.stderr}"
        # Should complete within reasonable time (300s upper bound for stress test scenarios)
        assert elapsed < 300, f"Should complete within timeout, took {elapsed:.1f}s"

    def test_timeout_times_out(self, runner):
        """--timeout N should timeout and exit with code 2 if tests don't complete in time."""
        # Use a very short timeout (2 seconds) - tests won't complete in time
        result = runner.run(
            "--timeout",
            "2",  # 2 second timeout - too short
            runner.fixture_path("test_always_pass.py"),
        )

        # Should timeout with exit code 2
        assert (
            result.exit_code == 2
        ), f"Should timeout with code 2, got {result.exit_code}"
        # Should mention timeout in output
        assert "timeout" in result.stdout.lower() or "Timeout" in result.stdout

    def test_timeout_with_short_flag(self, runner):
        """-t N should work the same as --timeout N."""
        result = runner.run(
            "-t",
            "300",  # 5 minute timeout - handles stress test scenarios
            runner.fixture_path("test_always_pass.py"),
        )

        assert (
            result.exit_code == 0
        ), f"Short -t with timeout should work: {result.stderr}"

    def test_timeout_default_per_test(self, runner):
        """--timeout N should work correctly with default per-test mode."""
        result = runner.run(
            "--timeout",
            "300",  # 5 minute timeout - handles stress test scenarios
            runner.fixture_path("test_always_pass.py"),
        )

        assert result.exit_code == 0, f"Should pass: {result.stderr}"


class TestSessionTimeoutFlag:
    """Tests for --session-timeout (per-tmux-session hang kill)."""

    def test_session_timeout_completes_normally(self, runner):
        """--session-timeout N should not kill sessions that finish quickly."""
        result = runner.run(
            "--session-timeout",
            "120",
            runner.fixture_path("test_always_pass.py"),
        )
        assert (
            result.exit_code == 0
        ), f"Fast tests should pass under session-timeout: {result.stderr}"
        assert "Per-session hang guard" in result.stdout

    @pytest.mark.skipif(
        shutil.which("timeout") is None and shutil.which("gtimeout") is None,
        reason="the per-session guard wraps pytest in coreutils timeout(1)",
    )
    def test_session_timeout_kills_hung_session(self, runner):
        """--session-timeout N should fail a hung session instead of waiting forever."""
        result = runner.run(
            "--session-timeout",
            "3",
            "--timeout",
            "60",
            runner.hang_fixture_path("test_hang.py"),
        )
        # Hung session is killed by timeout(1) → non-zero pytest status → runner exit 1
        # (not whole-run exit 2, which only fires when sessions are still pending).
        assert result.exit_code != 0, (
            f"Hung session should fail under --session-timeout, got 0. "
            f"stdout={result.stdout[-500:]!r} stderr={result.stderr[-500:]!r}"
        )
        assert result.exit_code != 2, (
            "Should fail the hung session (exit 1), not burn the whole-run timeout (exit 2). "
            f"stdout={result.stdout[-500:]!r}"
        )


class TestSerialFlag:
    """Tests for --serial / -s flag."""

    def test_serial_creates_session_per_file(self, runner):
        """--serial should create one session per file (not per test)."""
        result = runner.run(
            "-s",
            runner.fixture_path("test_always_pass.py"),
            wait_for_completion=True,
        )

        # test_always_pass.py has 3 test functions, but -s creates 1 session per file
        assert (
            len(result.sessions_created) == 1
        ), f"Expected 1 session (one per file), got {len(result.sessions_created)}: {result.sessions_created}"

    def test_serial_session_names_include_file_name(self, runner):
        """Serial mode session names should include the file name."""
        result = runner.run(
            "-s",
            runner.fixture_path("test_always_pass.py"),
            wait_for_completion=True,
        )

        assert len(result.sessions_created) == 1
        session_name = result.sessions_created[0]
        # Session name should contain file identifier
        assert (
            "always_pass" in session_name or "test_always_pass" in session_name
        ), f"Session name should reference file name: {session_name}"

    def test_serial_long_flag(self, runner):
        """--serial should work same as -s."""
        result = runner.run(
            "--serial",
            runner.fixture_path("test_always_pass.py"),
            wait_for_completion=True,
        )

        assert len(result.sessions_created) == 1

    def test_serial_with_directory(self, runner, fixtures_dir):
        """--serial with directory should create session per file."""
        # Just test with subdirectory to keep it manageable
        result = runner.run(
            "-s",
            runner.fixture_path("subdir"),
            wait_for_completion=True,
        )

        # subdir has 1 test file (test_in_subdir.py)
        assert (
            len(result.sessions_created) == 1
        ), f"Expected 1 session, got {len(result.sessions_created)}"

    def test_serial_blocks_until_completion(self, runner):
        """--serial should block until all file sessions complete."""
        result = runner.run(
            "-s",
            runner.fixture_path("test_always_pass.py"),
        )

        # Should block and return success
        assert result.exit_code == 0


class TestDefaultPerTestMode:
    """Tests verifying default per-test behavior."""

    def test_default_creates_session_per_function(self, runner):
        """Default mode should create one session per test function."""
        result = runner.run(
            runner.fixture_path("test_always_pass.py"),
            wait_for_completion=True,
        )

        # test_always_pass.py has 3 test functions
        assert (
            len(result.sessions_created) == 3
        ), f"Expected 3 sessions (one per test), got {len(result.sessions_created)}: {result.sessions_created}"

    def test_default_session_names_include_test_name(self, runner):
        """Default mode session names should include the test function name."""
        result = runner.run(
            runner.fixture_path("test_always_pass.py"),
            wait_for_completion=True,
        )

        # Session names should contain test function identifiers
        session_names_joined = " ".join(result.sessions_created)
        assert (
            "test_pass" in session_names_joined
            or "pass" in session_names_joined.lower()
        ), f"Session names should reference test names: {result.sessions_created}"

    def test_default_with_directory(self, runner, fixtures_dir):
        """Default mode with directory should create session per test function."""
        # Just test with subdirectory to keep it manageable
        result = runner.run(
            runner.fixture_path("subdir"),
            wait_for_completion=True,
        )

        # subdir/test_in_subdir.py has 2 tests
        assert (
            len(result.sessions_created) == 2
        ), f"Expected 2 sessions, got {len(result.sessions_created)}"


class TestMatchFlag:
    """Tests for --match / -m flag."""

    def test_match_filters_by_filename(self, runner, fixtures_dir):
        """--match should filter files by name pattern."""
        result = runner.run(
            "-s",  # Serial mode for predictable session count
            "--match",
            "*docstring*",
            fixtures_dir,
            wait_for_completion=True,
        )

        # Should only find test_docstring_pattern.py (1 file)
        assert (
            len(result.sessions_created) == 1
        ), f"Expected 1 matching file, got {len(result.sessions_created)}"

    def test_match_with_wildcard(self, runner, fixtures_dir):
        """--match should support wildcard patterns."""
        result = runner.run(
            "-s",  # Serial mode for predictable session count
            "-m",
            "*always*",
            fixtures_dir,
            wait_for_completion=True,
        )

        # Should find test_always_pass.py and test_always_fail.py (2 files)
        assert (
            len(result.sessions_created) == 2
        ), f"Expected 2 matching files, got {len(result.sessions_created)}"

    def test_match_no_matches(self, runner, fixtures_dir):
        """--match with no matches should find no tests."""
        result = runner.run(
            "-m",
            "*nonexistent_pattern*",
            fixtures_dir,
        )

        # Should report no tests found
        assert "No tests" in result.stdout or len(result.sessions_created) == 0


class TestEnvFlag:
    """Tests for --env / -e flag."""

    def test_env_single_value(self, runner):
        """--env should pass environment variable to pytest sessions."""
        result = runner.run(
            "--env",
            "TEST_VAR=test_value",
            runner.fixture_path("test_always_pass.py"),
            wait_for_completion=True,
        )

        assert result.exit_code == 0

    def test_env_multiple_values(self, runner):
        """Multiple --env flags should all be passed."""
        result = runner.run(
            "-e",
            "VAR1=value1",
            "-e",
            "VAR2=value2",
            runner.fixture_path("test_always_pass.py"),
            wait_for_completion=True,
        )

        assert result.exit_code == 0

    def test_env_requires_value(self, runner):
        """--env without a value should error."""
        result = runner.run(
            "--env",
        )

        assert result.exit_code != 0 or "Error" in result.stderr


class TestRepeatFlag:
    """Tests for --repeat flag."""

    def test_repeat_creates_multiple_sessions(self, runner):
        """--repeat N should create N sessions per test target."""
        result = runner.run(
            "--repeat",
            "3",
            runner.fixture_path("test_single_test.py"),
            wait_for_completion=True,
        )

        # Should create 3 sessions for the single file
        assert (
            len(result.sessions_created) == 3
        ), f"Expected 3 sessions, got {len(result.sessions_created)}"

    def test_repeat_default_per_test(self, runner):
        """--repeat with default per-test mode should multiply correctly."""
        result = runner.run(
            "--repeat",
            "2",
            runner.fixture_path("test_single_test.py"),
            wait_for_completion=True,
        )

        # test_single_test.py has 1 test, repeated 2 times = 2 sessions
        assert (
            len(result.sessions_created) == 2
        ), f"Expected 2 sessions, got {len(result.sessions_created)}"

    def test_repeat_requires_number(self, runner):
        """--repeat without a number should error."""
        result = runner.run(
            "--repeat",
        )

        assert result.exit_code != 0

    def test_repeat_requires_positive(self, runner):
        """--repeat with 0 or negative should error."""
        result = runner.run(
            "--repeat",
            "0",
            runner.fixture_path("test_single_test.py"),
        )

        assert result.exit_code != 0


class TestJobsFlag:
    """Tests for --jobs / -j flag (concurrency limiting)."""

    def test_jobs_limits_concurrency(self, runner):
        """--jobs N should limit concurrent sessions."""
        result = runner.run(
            "-j",
            "2",
            runner.fixture_path("test_always_pass.py"),
            wait_for_completion=True,
        )

        # Should succeed with limited concurrency
        assert result.exit_code == 0
        # test_always_pass.py has 3 tests, all should still run
        # Use log_files count since sessions may auto-close before parsing
        assert len(result.log_files) == 3
        # Verify concurrency limit was applied
        assert "Concurrency limit: 2" in result.stdout

    def test_jobs_long_flag(self, runner):
        """--jobs N should work same as -j N."""
        result = runner.run(
            "--jobs",
            "5",
            runner.fixture_path("test_always_pass.py"),
            wait_for_completion=True,
        )

        assert result.exit_code == 0

    def test_jobs_with_value_one(self, runner):
        """--jobs 1 should run tests sequentially."""
        result = runner.run(
            "-j",
            "1",
            runner.fixture_path("test_always_pass.py"),
            wait_for_completion=True,
        )

        # Should still succeed, just slower
        assert result.exit_code == 0
        # Use log_files count since sessions may auto-close before parsing
        assert len(result.log_files) == 3
        # Verify sequential mode was applied
        assert "Concurrency limit: 1" in result.stdout

    def test_jobs_requires_number(self, runner):
        """--jobs without a number should error."""
        result = runner.run(
            "--jobs",
        )

        assert result.exit_code != 0

    def test_jobs_zero_means_unlimited(self, runner):
        """--jobs 0 should mean unlimited concurrency."""
        result = runner.run(
            "-j",
            "0",
            runner.fixture_path("test_always_pass.py"),
            wait_for_completion=True,
        )

        # 0 means no limit, not an error
        assert result.exit_code == 0
        assert "Concurrency limit: unlimited" in result.stdout


class TestHelpFlag:
    """Tests for --help / -h flag."""

    def test_help_shows_usage(self, runner):
        """--help should show usage information."""
        result = runner.run("--help")

        assert result.exit_code == 0
        assert "Usage:" in result.stdout
        assert "parallel_run" in result.stdout.lower()

    def test_help_shows_options(self, runner):
        """--help should list available options."""
        result = runner.run("--help")

        assert "--timeout" in result.stdout
        assert "--serial" in result.stdout
        assert "--match" in result.stdout
        assert "--env" in result.stdout
        assert "--jobs" in result.stdout
        assert "--repeat" in result.stdout

    def test_help_short_flag(self, runner):
        """-h should work same as --help."""
        result = runner.run("-h")

        assert result.exit_code == 0
        assert "Usage:" in result.stdout

    def test_help_shows_examples(self, runner):
        """--help should show usage examples."""
        result = runner.run("--help")

        assert "Examples:" in result.stdout


class TestPytestPassthrough:
    """Tests for -- pytest argument passthrough."""

    def test_passthrough_verbose(self, runner):
        """Arguments after -- should be passed to pytest."""
        result = runner.run(
            runner.fixture_path("test_always_pass.py"),
            "--",
            "-v",
            wait_for_completion=True,
        )

        # Should succeed
        assert result.exit_code == 0

    def test_passthrough_multiple_args(self, runner):
        """Multiple arguments after -- should all be passed."""
        result = runner.run(
            runner.fixture_path("test_always_pass.py"),
            "--",
            "-v",
            "--tb=short",
            wait_for_completion=True,
        )

        assert result.exit_code == 0

    def test_passthrough_with_other_flags(self, runner):
        """-- passthrough should work with other parallel_run flags."""
        result = runner.run(
            "-s",  # serial mode
            runner.fixture_path("test_always_pass.py"),
            "--",
            "-v",
            wait_for_completion=True,
        )

        assert result.exit_code == 0
        # Serial mode: 1 session for the file
        assert len(result.sessions_created) == 1

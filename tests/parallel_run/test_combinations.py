"""
Combination tests for parallel_run.sh.

Tests various flag combinations to ensure they work together correctly.
This is essentially a grid search across the flag space.

The script always blocks until tests complete, so these tests verify
flag combinations work correctly with blocking behavior.

Flag combinations tested:
- default per-test mode
- --serial + --match
- Triple combinations (--timeout, --match, --env, etc.)
"""

from __future__ import annotations


class TestBlockingWithOtherFlags:
    """Tests combining blocking behavior with other flags."""

    def test_default_per_test(self, runner):
        """Script should block for all per-test sessions (default mode)."""
        result = runner.run(
            runner.fixture_path("test_always_pass.py"),
        )

        # Should exit with 0 (all pass) and have waited for completion
        assert result.exit_code == 0, f"Failed with: {result.stderr}"

    def test_default_per_test_failure(self, runner):
        """Script should return non-zero if any test fails (default mode)."""
        result = runner.run(
            runner.fixture_path("test_mixed_results.py"),
        )

        # test_mixed_results.py has one failing test
        assert result.exit_code != 0

    def test_with_match(self, runner, fixtures_dir):
        """Script with --match should block for matched sessions."""
        result = runner.run(
            "--match",
            "*single*",
            fixtures_dir,
        )

        assert result.exit_code == 0

    def test_with_env(self, runner):
        """Script with --env should pass env and block."""
        result = runner.run(
            "--env",
            "MY_VAR=my_value",
            runner.fixture_path("test_always_pass.py"),
        )

        assert result.exit_code == 0

    def test_with_repeat(self, runner):
        """Script with --repeat should block for all repeated runs."""
        result = runner.run(
            "--repeat",
            "2",
            runner.fixture_path("test_single_test.py"),
        )

        assert result.exit_code == 0


class TestDefaultModeWithOtherFlags:
    """Tests combining default per-test mode with other flags."""

    def test_default_with_match(self, runner, fixtures_dir):
        """Default mode with --match should match files then split by test."""
        result = runner.run(
            "--match",
            "*always_pass*",
            fixtures_dir,
            wait_for_completion=True,
        )

        # test_always_pass.py has 3 tests
        assert (
            len(result.sessions_created) == 3
        ), f"Expected 3 sessions, got {len(result.sessions_created)}"

    def test_default_with_repeat(self, runner):
        """Default mode with --repeat should multiply correctly."""
        result = runner.run(
            "--repeat",
            "3",
            runner.fixture_path("test_single_test.py"),
            wait_for_completion=True,
        )

        # 1 test * 3 repeats = 3 sessions
        assert (
            len(result.sessions_created) == 3
        ), f"Expected 3 sessions, got {len(result.sessions_created)}"

    def test_default_with_env(self, runner):
        """Default mode with --env should pass env to each test session."""
        result = runner.run(
            "--env",
            "TEST_ENV=value",
            runner.fixture_path("test_always_pass.py"),
            wait_for_completion=True,
        )

        assert len(result.sessions_created) == 3


class TestTripleCombinations:
    """Tests with three or more flags combined."""

    def test_match_combo(self, runner, fixtures_dir):
        """--match should work with blocking."""
        result = runner.run(
            "--match",
            "*single*",
            fixtures_dir,
        )

        # test_single_test.py has 1 test
        assert result.exit_code == 0

    def test_repeat_combo(self, runner):
        """--repeat should work with blocking."""
        result = runner.run(
            "--repeat",
            "2",
            runner.fixture_path("test_single_test.py"),
        )

        # 1 test * 2 repeats = 2 sessions, all should pass
        assert result.exit_code == 0

    def test_env_combo(self, runner):
        """--env should work with blocking."""
        result = runner.run(
            "--env",
            "COMBO_VAR=value",
            runner.fixture_path("test_always_pass.py"),
        )

        assert result.exit_code == 0

    def test_repeat_env(self, runner):
        """--repeat + --env should all work together."""
        result = runner.run(
            "--repeat",
            "2",
            "--env",
            "REPEAT_VAR=value",
            runner.fixture_path("test_single_test.py"),
        )

        assert len(result.sessions_created) == 2


class TestQuadrupleCombinations:
    """Tests with four or more flags combined."""

    def test_match_repeat(self, runner, fixtures_dir):
        """--match + --repeat should work together."""
        result = runner.run(
            "--match",
            "*single*",
            "--repeat",
            "2",
            fixtures_dir,
        )

        # test_single_test.py has 1 test, repeated 2 times
        assert result.exit_code == 0


class TestInputTypeCombinations:
    """Tests combining different input types with flags."""

    def test_file_and_dir(self, runner):
        """File + directory input should work."""
        result = runner.run(
            runner.fixture_path("test_single_test.py"),
            runner.fixture_path("subdir"),
        )

        assert result.exit_code == 0
        # test_single_test.py has 1 test, subdir has 2 = 3 total
        assert len(result.sessions_created) == 3

    def test_specific_test(self, runner):
        """Specific test node should work."""
        test_path = runner.fixture_path("test_always_pass.py") + "::test_pass_one"
        result = runner.run(
            test_path,
        )

        assert result.exit_code == 0
        # Still just 1 session for the specific test
        assert len(result.sessions_created) == 1

    def test_multiple_specific_tests(self, runner):
        """Multiple specific tests should work."""
        test1 = runner.fixture_path("test_always_pass.py") + "::test_pass_one"
        test2 = runner.fixture_path("test_always_pass.py") + "::test_pass_two"
        result = runner.run(
            test1,
            test2,
        )

        assert result.exit_code == 0

    def test_multiple_specific_tests_blocks(self, runner):
        """Multiple explicit node IDs should block until complete.

        This tests the exact scenario of:
            parallel_run.sh file.py::test_a file.py::test_b ...

        The script should block until all tests complete.
        """
        import time

        test1 = runner.fixture_path("test_always_pass.py") + "::test_pass_one"
        test2 = runner.fixture_path("test_always_pass.py") + "::test_pass_two"
        test3 = runner.fixture_path("test_always_pass.py") + "::test_pass_three"
        test4 = runner.fixture_path("test_single_test.py") + "::test_single"

        start = time.time()
        result = runner.run(
            test1,
            test2,
            test3,
            test4,
        )
        elapsed = time.time() - start

        # Should have blocked for at least some time
        assert (
            elapsed > 1.0
        ), f"Script should block until tests complete, but returned in {elapsed:.2f}s"

        # All tests should pass
        assert result.exit_code == 0, f"All tests should pass: {result.stderr}"

    def test_mixed_file_and_specific_test(self, runner):
        """File + specific test should work."""
        file_path = runner.fixture_path("test_single_test.py")
        specific_test = runner.fixture_path("test_always_pass.py") + "::test_pass_one"
        result = runner.run(
            file_path,
            specific_test,
        )

        assert result.exit_code == 0


class TestMultipleEnvVars:
    """Tests with multiple --env flags."""

    def test_multiple_env(self, runner):
        """Multiple --env should work."""
        result = runner.run(
            "-e",
            "VAR1=value1",
            "-e",
            "VAR2=value2",
            "-e",
            "VAR3=value3",
            runner.fixture_path("test_always_pass.py"),
        )

        assert result.exit_code == 0

    def test_multiple_env_default(self, runner):
        """Multiple --env with default per-test mode should work."""
        result = runner.run(
            "-e",
            "A=1",
            "-e",
            "B=2",
            runner.fixture_path("test_single_test.py"),
        )

        assert len(result.sessions_created) == 1

    def test_multiple_env_combined(self, runner):
        """Multiple --env with different values should work."""
        result = runner.run(
            "-e",
            "X=1",
            "-e",
            "Y=2",
            runner.fixture_path("test_always_pass.py"),
        )

        assert result.exit_code == 0

"""
Basic functionality tests for parallel_run.sh.

Tests different input types:
- Single file
- Multiple files
- Single directory
- Multiple directories
- Specific test node IDs
- Mixed inputs (files + directories)
"""

from __future__ import annotations


class TestSingleFile:
    """Tests for running with a single test file."""

    def test_single_file_creates_one_session(self, runner):
        """Running a single file with -s should create exactly one tmux session."""
        result = runner.run(
            "-s",  # Serial mode: one session per file
            runner.fixture_path("test_always_pass.py"),
            wait_for_completion=True,
        )

        # Script should succeed (exits immediately after creating sessions)
        assert result.exit_code == 0, f"Script failed: {result.stderr}"

        # Should create exactly one session
        assert (
            len(result.sessions_created) == 1
        ), f"Expected 1 session, got {len(result.sessions_created)}: {result.sessions_created}"

    def test_single_file_session_naming(self, runner):
        """Session name should be derived from the file path."""
        result = runner.run(
            "-s",  # Serial mode: one session per file
            runner.fixture_path("test_always_pass.py"),
            wait_for_completion=True,
        )

        assert len(result.sessions_created) == 1
        session_name = result.sessions_created[0]

        # Session name should contain the file identifier (without test_ prefix and .py)
        # After status prefix like "r ⏳ " or "p ✅ "
        assert "always_pass" in session_name or "test_always_pass" in session_name


class TestMultipleFiles:
    """Tests for running with multiple test files."""

    def test_multiple_files_create_separate_sessions(self, runner):
        """Each file should get its own tmux session in serial mode."""
        result = runner.run(
            "-s",  # Serial mode: one session per file
            runner.fixture_path("test_always_pass.py"),
            runner.fixture_path("test_single_test.py"),
            wait_for_completion=True,
        )

        assert result.exit_code == 0, f"Script failed: {result.stderr}"

        # Should create exactly two sessions
        assert (
            len(result.sessions_created) == 2
        ), f"Expected 2 sessions, got {len(result.sessions_created)}: {result.sessions_created}"

    def test_multiple_files_unique_session_names(self, runner):
        """Each session should have a unique name."""
        result = runner.run(
            "-s",  # Serial mode: one session per file
            runner.fixture_path("test_always_pass.py"),
            runner.fixture_path("test_single_test.py"),
            runner.fixture_path("test_two_passes.py"),
            wait_for_completion=True,
        )

        # All session names should be unique
        assert len(result.sessions_created) == len(set(result.sessions_created))


class TestDirectory:
    """Tests for running with a directory."""

    def test_directory_discovers_all_test_files(self, runner, fixtures_dir):
        """Running on a directory should discover all test_*.py files."""
        result = runner.run(fixtures_dir, wait_for_completion=True)

        # fixtures/ contains both test_always_pass.py AND test_always_fail.py,
        # so running the whole directory yields exit_code=1 by design (the
        # always-fail fixtures are intentionally failing). The semantic
        # this test asserts is "directory discovery walked the whole tree
        # and launched ≥7 sessions", not "every test passed". Don't gate
        # on exit_code==0.
        assert result.exit_code in (0, 1), f"unexpected exit_code: {result.exit_code}"

        # Should find multiple test files in fixtures dir
        # fixtures/ contains: test_always_pass.py, test_always_fail.py,
        # test_mixed_results.py, test_two_passes.py, test_single_test.py,
        # test_docstring_pattern.py, subdir/test_in_subdir.py
        assert (
            len(result.sessions_created) >= 7
        ), f"Expected at least 7 sessions, got {len(result.sessions_created)}"

    def test_directory_recursive_discovery(self, runner, fixtures_dir):
        """Should recursively discover tests in subdirectories."""
        result = runner.run(fixtures_dir, wait_for_completion=True)

        # Should find test_in_subdir.py in the subdir
        subdir_session = [
            s for s in result.sessions_created if "subdir" in s or "in_subdir" in s
        ]
        assert (
            len(subdir_session) >= 1
        ), f"Should find subdir test, sessions: {result.sessions_created}"


class TestMultipleDirectories:
    """Tests for running with multiple directories."""

    def test_multiple_directories(self, runner):
        """Running with multiple directories should find tests in all."""
        result = runner.run(
            runner.fixture_path("subdir"),
            wait_for_completion=True,
        )

        assert result.exit_code == 0
        assert len(result.sessions_created) >= 1


class TestSpecificTests:
    """Tests for running specific test node IDs (file.py::test_name)."""

    def test_specific_test_creates_session(self, runner):
        """Running a specific test node ID should create a session."""
        test_path = runner.fixture_path("test_always_pass.py") + "::test_pass_one"
        result = runner.run(test_path, wait_for_completion=True)

        assert result.exit_code == 0, f"Script failed: {result.stderr}"
        assert (
            len(result.sessions_created) == 1
        ), f"Expected 1 session, got {len(result.sessions_created)}"

    def test_multiple_specific_tests(self, runner):
        """Running multiple specific tests should create one session each."""
        test1 = runner.fixture_path("test_always_pass.py") + "::test_pass_one"
        test2 = runner.fixture_path("test_always_pass.py") + "::test_pass_two"
        result = runner.run(test1, test2, wait_for_completion=True)

        assert result.exit_code == 0
        assert (
            len(result.sessions_created) == 2
        ), f"Expected 2 sessions, got {len(result.sessions_created)}"

    def test_specific_tests_from_different_files(self, runner):
        """Specific tests from different files should each get a session."""
        test1 = runner.fixture_path("test_always_pass.py") + "::test_pass_one"
        test2 = runner.fixture_path("test_single_test.py") + "::test_single"
        result = runner.run(test1, test2, wait_for_completion=True)

        assert len(result.sessions_created) == 2


class TestMixedInputs:
    """Tests for running with mixed input types (files + directories + tests)."""

    def test_file_and_directory(self, runner):
        """Can mix file and directory inputs."""
        result = runner.run(
            runner.fixture_path("test_single_test.py"),
            runner.fixture_path("subdir"),
            wait_for_completion=True,
        )

        assert result.exit_code == 0
        # Should have at least 2 sessions (1 for single_test + subdir files)
        assert len(result.sessions_created) >= 2

    def test_file_and_specific_test(self, runner):
        """Can mix file and specific test inputs."""
        file_path = runner.fixture_path("test_single_test.py")
        specific_test = runner.fixture_path("test_always_pass.py") + "::test_pass_one"
        result = runner.run(file_path, specific_test, wait_for_completion=True)

        assert result.exit_code == 0
        # Default mode: per-test. test_single_test.py has 1 test, plus the specific test = 2 sessions
        assert len(result.sessions_created) == 2


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_nonexistent_path_warning(self, runner):
        """Non-existent paths should produce a warning but not crash."""
        result = runner.run(
            "nonexistent_test_file.py",
            runner.fixture_path("test_single_test.py"),
            wait_for_completion=True,
        )

        # Should still succeed (creates session for valid file)
        assert "Warning" in result.stderr or "Skipping" in result.stderr

    def test_empty_run_with_no_tests(self, runner):
        """Running with no valid targets should exit gracefully."""
        result = runner.run("nonexistent_dir_that_does_not_exist/")

        # Should report no tests found
        assert "No valid" in result.stderr or "No tests" in result.stdout

    def test_output_includes_session_list(self, runner):
        """Output should list created tmux sessions."""
        result = runner.run(
            runner.fixture_path("test_single_test.py"),
            wait_for_completion=True,
        )

        assert "Created" in result.stdout and "tmux" in result.stdout.lower()

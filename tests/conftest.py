"""
tests/conftest.py
=================

Global pytest configuration for the Unify test suite.

Sections:
  1. Imports and logging guard
  2. Test stubs (DateTime, embeddings)
  3. Singleton isolation
  4. Command-line options
  5. Custom logging helpers
  6. Session lifecycle hooks
  7. Test run hooks
"""

from __future__ import annotations

import logging
import os
import random
import re

import pytest
from unify import db
from pytest_metadata.plugin import metadata_key

from datetime import datetime, timezone

# --------------------------------------------------------------------------- #
# 1. Early logging guard                                                      #
# --------------------------------------------------------------------------- #
# Ensure a handler exists before imports that might call logging.basicConfig()
_root_logger_early = logging.getLogger()
if not _root_logger_early.handlers:
    _root_logger_early.addHandler(logging.NullHandler())

from tests.settings import SETTINGS

# Diagnostic switch for the async tool loop's sent-watermark append-only
# transcript invariant. Prod behavior is identical whether this is set or
# not — it only gates an integrity
# assertion in generate_with_preprocess plus the below-watermark hashing
# that backs it, which would otherwise ship real per-turn CPU to prod under
# an `if __debug__` label. `setdefault` so an explicit override (e.g. a
# targeted rerun with it forced off) still wins.
os.environ.setdefault("UNIFY_TRANSCRIPT_INVARIANT_CHECKS", "1")


def _reset_singleton_registries() -> None:
    """Singletons must not leak across tests."""
    from unify.manager_registry import ManagerRegistry
    from unify.events.event_bus import EVENT_BUS

    ManagerRegistry.clear()
    EVENT_BUS.clear()


def _reset_store_for_test() -> None:
    """Give the test an empty store: the user tables are truncated and their
    id sequences restart, so a rerun in a reused store never sees a previous
    run's rows. The seeded catalogues (primitives, builtin guidance) stay."""
    db.clear()
    _reset_singleton_registries()


def _uses_unify_context(item: pytest.Item) -> bool:
    """Return whether this test needs the per-test store reset."""

    return item.get_closest_marker("no_unify_context") is None


def pytest_report_header(config):
    settings_str = [f"{k}={v}" for k, v in SETTINGS.model_dump().items()]
    return [
        f"unify_store={os.environ.get('UNIFY_STORE_PATH')}",
        f"UNILLM_CACHE={os.environ.get('UNILLM_CACHE', 'not set')}",
    ] + settings_str


# --------------------------------------------------------------------------- #
# 2. Test stubs (DateTime, embeddings)                                        #
# --------------------------------------------------------------------------- #

_FIXED_DATETIME = datetime(2025, 6, 13, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture(scope="session")
def static_now():
    return _FIXED_DATETIME


@pytest.fixture(autouse=True)
def stub_external_deps(monkeypatch):
    """
    This fixture automatically stubs heavy external dependencies for tests.
    It runs for every test.
    """

    # --- DateTime stub for prompts (centralized) -----------------------------------
    # All timestamps in prompts come from prompt_helpers.now() which returns either:
    # - A formatted string (as_string=True): "Friday, June 13, 2025 at 12:00 PM UTC"
    # - A datetime object (as_string=False): for timestamp comparisons
    #
    # When UNIFY_INCREMENTING_TIMESTAMPS is enabled (e.g., ConversationManager tests),
    # datetime objects auto-increment by microseconds so last_snapshot < message.timestamp
    # comparisons work correctly for **NEW** markers.

    from datetime import timedelta

    _timestamp_counter = {"value": 0}

    def _static_now(time_only: bool = False, as_string: bool = True):
        """Return a fixed timestamp for testing."""
        if SETTINGS.UNIFY_INCREMENTING_TIMESTAMPS and not as_string:
            # Return incrementing datetime for **NEW** marker comparisons
            _timestamp_counter["value"] += 1
            return _FIXED_DATETIME + timedelta(microseconds=_timestamp_counter["value"])

        if not as_string:
            return _FIXED_DATETIME

        label = "UTC"
        if time_only:
            return _FIXED_DATETIME.strftime("%I:%M %p ") + label
        return _FIXED_DATETIME.strftime("%A, %B %d, %Y at %I:%M %p ") + label

    # Patch prompt_helpers.now everywhere it's imported
    monkeypatch.setattr("unify.common.prompt_helpers.now", _static_now)
    monkeypatch.setattr("unify.conversation_manager.prompt_builders.now", _static_now)
    monkeypatch.setattr("unify.conversation_manager.events.prompt_now", _static_now)
    monkeypatch.setattr(
        "unify.conversation_manager.domains.chat_history.prompt_now",
        _static_now,
    )
    monkeypatch.setattr(
        "unify.conversation_manager.conversation_manager.prompt_now",
        _static_now,
    )

    def _static_perf_counter() -> float:
        return 1000.0

    monkeypatch.setattr(
        "unify.common._async_tool.time_context.perf_counter",
        _static_perf_counter,
    )


def _isolate_embeddings(monkeypatch, tmp_path) -> None:
    """A private embeddings cache, and fallback warnings that fire afresh."""
    from unify.common import embeddings, semantic_search

    monkeypatch.setenv("UNIFY_EMBED_CACHE", str(tmp_path / "embeddings.sqlite"))
    monkeypatch.setattr(semantic_search, "_REPORTED", set())
    embeddings.reset()


@pytest.fixture
def fake_embeddings(monkeypatch, tmp_path):
    """Embed from the deterministic concept space in ``tests.embedding_helpers``.

    No network: the provider is replaced, the model is named apart from any
    real one and the cache is private to the test.
    """
    from tests.embedding_helpers import FakeEmbeddings
    from unify.common import embeddings

    fake = FakeEmbeddings()
    _isolate_embeddings(monkeypatch, tmp_path)
    monkeypatch.setenv("UNIFY_EMBED_MODEL", "test/concepts")
    monkeypatch.setattr(embeddings, "_provider", lambda model: fake)
    yield fake
    embeddings.reset()


@pytest.fixture
def no_embedding_key(monkeypatch, tmp_path):
    """The settings hold no OpenRouter key, and a request would fail the test."""
    import requests
    from pydantic import SecretStr
    from unillm.settings import SETTINGS as UNILLM_SETTINGS
    from unify.common import embeddings

    def _refuse(*args, **kwargs):
        raise AssertionError("no embeddings request may be sent without a key")

    _isolate_embeddings(monkeypatch, tmp_path)
    monkeypatch.delenv("UNIFY_EMBED_MODEL", raising=False)
    monkeypatch.setattr(UNILLM_SETTINGS, "OPENROUTER_API_KEY", SecretStr(""))
    monkeypatch.setattr(requests, "post", _refuse)
    yield
    embeddings.reset()


@pytest.fixture
def semantic_search_warnings(caplog):
    """The warnings semantic search has logged so far in the test.

    The ``unify`` logger does not propagate to the root logger, so the
    capture handler is attached to it directly.
    """
    unify_logger = logging.getLogger("unify")
    unify_logger.addHandler(caplog.handler)
    caplog.set_level(logging.WARNING, logger="unify.common.semantic_search")
    yield lambda: [
        record.getMessage()
        for record in caplog.records
        if record.name == "unify.common.semantic_search"
        and record.levelno == logging.WARNING
    ]
    unify_logger.removeHandler(caplog.handler)


# --------------------------------------------------------------------------- #
# 3. Singleton isolation                                                      #
# --------------------------------------------------------------------------- #

from unify.manager_registry import ManagerRegistry


@pytest.fixture(autouse=True)
def _clear_singletons_between_tests():
    """Ensure *singleton* instances never leak from one test to the next."""
    yield
    ManagerRegistry.clear()  # Clear the registry after each test


@pytest.fixture(autouse=True)
def _exact_cache_keying_for_evals(request):
    """An eval never accepts a canonically-equivalent recording.

    Canonical keying deliberately lets a recording survive mundane prompt
    churn, and the invariance it buys includes description rewording --
    which is one of the ordinary ways to change which tool a model picks.
    For a functional test that trade is right: the assertion is about the
    code around the call. For an eval it inverts, because the model's
    behaviour *is* the subject, so a canonical hit would score the
    trajectory recorded before the change and report it green.

    Scoped per test rather than per shard: a shard is a directory, and
    directories hold both kinds.
    """
    if request.node.get_closest_marker("eval") is None:
        yield
        return

    from unillm.settings import SETTINGS as UNILLM_SETTINGS

    previous = UNILLM_SETTINGS.UNILLM_CACHE_KEYING
    UNILLM_SETTINGS.UNILLM_CACHE_KEYING = "exact"
    try:
        yield
    finally:
        UNILLM_SETTINGS.UNILLM_CACHE_KEYING = previous


# --------------------------------------------------------------------------- #
# 4. Command-line options                                                     #
# --------------------------------------------------------------------------- #


def pytest_addoption(parser):
    group = parser.getgroup("custom-logging")
    group.addoption(
        "--test-log-enable",
        action="store_true",
        default=False,
        help="Enable test-aware logging (adds test name to log records).",
    )
    group.addoption(
        "--test-log-file",
        action="store",
        default="tests.log",
        help="Filename to write test-aware logs to (only applies if --test-log-enable is used).",
    )
    group.addoption(
        "--test-log-format",
        action="store",
        default="[%(levelname)s] %(asctime)s - %(test_name)s: %(message)s",
        help="Custom log format string (only applies if --test-log-enable is used).",
    )


# --------------------------------------------------------------------------- #
# 5. Custom logging helpers                                                   #
# --------------------------------------------------------------------------- #


class TestNameLogFilter(logging.Filter):
    def __init__(self):
        super().__init__()
        self.test_name = None

    def set_test_name(self, test_name):
        self.test_name = test_name.split("tests/")[-1]

    def reset_test_name(self):
        self.test_name = ""

    def filter(self, record):
        record.test_name = self.test_name or "UNKNOWN"
        return True


test_name_log_filter = TestNameLogFilter()


@pytest.fixture(scope="session", autouse=True)
def configure_logging(request):
    config = request.config
    if not is_test_logging_enabled(config):
        return

    logger = logging.getLogger()
    file_handler = logging.FileHandler(get_test_log_file(config), mode="w")
    formatter = logging.Formatter(
        get_test_log_format(config),
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(formatter)
    file_handler.addFilter(test_name_log_filter)
    logger.addHandler(file_handler)


def is_test_logging_enabled(config):
    return config.getoption("--test-log-enable")


def get_test_log_file(config):
    return config.getoption("--test-log-file")


def get_test_log_format(config):
    return config.getoption("--test-log-format")


# --------------------------------------------------------------------------- #
# 6. Session lifecycle hooks                                                  #
# --------------------------------------------------------------------------- #


def pytest_sessionstart(session):
    if os.environ.get("SKIP_UNIFY_TEST_INIT"):
        return

    if os.environ.get("GITHUB_ACTIONS"):
        import unillm

        unillm.set_cache_backend("local_separate")

    import unify  # local import to avoid affecting stub installation order

    unify.init()


def pytest_sessionfinish(session, exitstatus):
    # Write cache stats to a temp file for parallel_run.sh to consume
    # The file is keyed by UNIFY_TMUX_SESSION_ID env var (set by parallel_run.sh)
    try:
        import unillm

        stats = unillm.get_cache_stats()
        session_id = os.environ.get("UNIFY_TMUX_SESSION_ID", "")
        if session_id:
            stats_file = f"/tmp/parallel_run_cache_{session_id}.txt"
            with open(stats_file, "w") as f:
                f.write(f"{stats.hits}|{stats.canonical_hits}|{stats.misses}\n")
    except Exception:
        pass  # Don't fail the test run if cache stats writing fails

    # Write LLM provider cost to a temp file for parallel_run.sh to consume
    try:
        session_id = os.environ.get("UNIFY_TMUX_SESSION_ID", "")
        if session_id:
            total_cost = sum(cost for _, cost in _session_costs)
            cost_file = f"/tmp/parallel_run_cost_{session_id}.txt"
            with open(cost_file, "w") as f:
                f.write(f"{total_cost:.6g}\n")
    except Exception:
        pass


def pytest_unconfigure(config):
    """Restore HOME (and HF_HOME if we set it).

    `/tmp/unity_test_home` is deliberately left in place: the path is
    shared by every parallel pytest session ``parallel_run.sh`` spawns
    (deterministic so LLM cache keys embedding the workspace root stay
    stable), so wiping it here would pull files out from under sessions
    still running. It accumulates only for the life of the CI runner;
    locally, ``rm -rf /tmp/unity_test_home`` gives a clean slate.
    """
    if _original_home is None:
        os.environ.pop("HOME", None)
    else:
        os.environ["HOME"] = _original_home
    if _hf_home_set_by_us:
        os.environ.pop("HF_HOME", None)


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    if SETTINGS.UNIFY_CACHE_STATS:
        import unillm

        stats = unillm.get_cache_stats()
        terminalreporter.section(
            f"Unify cache report | Hits ({stats.get_percentage_of_cache_hits():.2f}%): {stats.hits} ({stats.canonical_hits} canonical) | Misses ({stats.get_percentage_of_cache_misses():.2f}%): {stats.misses} | Reads: {stats.reads} | Writes: {stats.writes}",
        )

    total = sum(cost for _, cost in _session_costs)
    terminalreporter.write_sep("=", f"UNILLM Provider Cost Summary: ${total:.6g}")

    # Record outcome counts for parallel_run.sh. pytest exits 0 when every test
    # skips, so exit status alone cannot tell a session that passed from one
    # that ran nothing — and a summary that calls those the same thing hides
    # coverage silently disappearing.
    try:
        session_id = os.environ.get("UNIFY_TMUX_SESSION_ID", "")
        if session_id:
            passed = len(terminalreporter.stats.get("passed", []))
            skipped = len(terminalreporter.stats.get("skipped", []))
            outcome_file = f"/tmp/parallel_run_outcome_{session_id}.txt"
            with open(outcome_file, "w") as f:
                f.write(f"{passed}|{skipped}\n")
    except Exception:
        pass  # Don't fail the test run if outcome stats writing fails


# --------------------------------------------------------------------------- #
# 7. Test run hooks                                                           #
# --------------------------------------------------------------------------- #

from unillm.cost_tracker import capture_costs

_session_costs: list[tuple[str, float]] = []

_original_home: str | None = None
_hf_home_set_by_us: bool = False


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "no_unify_context: skip the per-test store reset for pure unit tests",
    )

    # ------------------------------------------------------------------
    # Isolate HOME so that tests never touch the real home directory.
    # get_local_root() resolves under the home directory, and the process
    # cwd is set to the same path at startup.  By pointing HOME at a temp
    # dir we keep Attachments/, .env, snapshots, etc. sandboxed.
    #
    # The path is deterministic (not random) so that CodeActActor system
    # prompts — which embed the resolved workspace path — produce
    # stable LLM cache keys across pytest sessions.  Actual test file
    # isolation is handled by pytest's tmp_path fixture, not HOME.
    # ------------------------------------------------------------------
    import tempfile

    global _original_home
    _original_home = os.environ.get("HOME")
    if _original_home:
        os.environ["UNIFY_REAL_HOME"] = _original_home
    test_home = os.path.join(tempfile.gettempdir(), "unity_test_home")
    os.makedirs(test_home, exist_ok=True)
    os.environ["HOME"] = test_home

    # Preserve access to the real HuggingFace model cache.  The HOME
    # override above moves ~/.cache/huggingface to a temp dir that won't
    # contain pre-downloaded models (e.g. SmolVLM used by docling's PDF
    # pipeline).  Pinning HF_HOME to the original location avoids
    # redundant multi-GB downloads and the .incomplete-blob hangs that
    # occur when the download is interrupted or raced across sessions.
    global _hf_home_set_by_us
    if "HF_HOME" not in os.environ and _original_home:
        original_hf = os.path.join(_original_home, ".cache", "huggingface")
        if os.path.isdir(original_hf):
            os.environ["HF_HOME"] = original_hf
            _hf_home_set_by_us = True

    config.addinivalue_line(
        "markers",
        "requires_real_unify: mark test as requiring the real unify implementation",
    )
    config.addinivalue_line(
        "markers",
        "eval: mark a test as a fuzzy evaluation test for English language "
        "APIs. Selects the eval tier in discover_test_paths.py, and pins cache "
        "lookups to exact keying so a canonical hit cannot score a trajectory "
        "recorded before the prompt changed. Distinct from llm_call, which "
        "says only that a model is reached: eval asks whether the answer was "
        "good, llm_call whether the call happens at all. The two are applied "
        "independently and neither implies the other.",
    )

    # Required to disable explicit log level if set from pytest.ini or command line options
    if os.environ.get("UNIFY_TESTS_CLI_LOGGING", "true").lower() == "false":
        config.option.log_cli_level = None
        config.option.showcapture = "no"
        config.option.capture = "no"

    config.stash[metadata_key]["Settings"] = SETTINGS.model_dump(mode="json")

    # ------------------------------------------------------------------ #
    # Prune non-pytest console handlers so only pytest live logs appear. #
    # Keeps any file handlers (e.g., when --test-log-enable is used).    #
    # ------------------------------------------------------------------ #
    try:
        root = logging.getLogger()
        kept_handlers: list[logging.Handler] = []
        for h in list(root.handlers):
            mod = getattr(h.__class__, "__module__", "")
            is_stream = isinstance(h, logging.StreamHandler)
            is_pytest = mod.startswith("_pytest.logging")
            # Retain pytest's handlers and any non-stream handlers (file, etc.)
            if is_stream and not is_pytest:
                continue
            kept_handlers.append(h)
        root.handlers = kept_handlers
    except Exception:
        # Never fail configuration due to logging hygiene adjustments.
        pass


def pytest_runtest_setup(item):
    test_name_log_filter.set_test_name(item.nodeid)
    if not os.environ.get("SKIP_UNIFY_TEST_INIT") and _uses_unify_context(item):
        _reset_store_for_test()


def _normalize_pytest_nodeid(nodeid):
    """
    Try to normalize the pytest nodeid to an alphanumeric string that is
    accepted for db.Context path. If not possible, return None.
    Will fallback to invocation count if empty.
    """
    bracket_match = re.search(r"\[([^\]]+)\]", nodeid)
    if bracket_match:
        bracket_content = bracket_match.group(1)
    else:
        bracket_content = ""

    # Try to normalize to alphanumeric
    normalized = re.sub(r"[^a-zA-Z0-9]", "", bracket_content)

    if len(normalized) == 0:
        return None

    return normalized[:24]


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    import types

    func_name = item.originalname

    # For class-based tests, item.obj is a bound method. We need to access
    # the underlying function via __func__ to set/get attributes.
    target_obj = item.obj
    if isinstance(target_obj, types.MethodType):
        target_obj = target_obj.__func__

    if "[" in item.nodeid:  # Any parametrization (markers, fixtures, etc.)
        # Need to keep track of invocation count for parametrized tests
        # In case of a later failure.
        current_count = getattr(target_obj, "_unity_pytest_invocation_count", 0)
        setattr(target_obj, "_unity_pytest_invocation_count", current_count + 1)

        normalized_id = _normalize_pytest_nodeid(item.nodeid)
        if normalized_id is None:
            normalized_id = f"_{current_count}_"
        func_name = f"{func_name}/{normalized_id}"

    setattr(target_obj, "_unity_pytest_nodeid", func_name)

    with capture_costs() as events:
        yield
    item._unillm_cost_events = events


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if call.when == "call":
        events = getattr(item, "_unillm_cost_events", [])
        total = sum(e.provider_cost for e in events)
        report._unillm_cost = total
        _session_costs.append((report.nodeid, total))


@pytest.hookimpl(hookwrapper=True)
def pytest_report_teststatus(report, config):
    outcome = yield
    if report.when == "call":
        result = outcome.get_result()
        if result and len(result) >= 3:
            category, shortletter, verbose = result
            cost = getattr(report, "_unillm_cost", 0.0)
            if isinstance(verbose, str):
                verbose = f"{verbose} [${cost:.6g}]"
            outcome.force_result((category, shortletter, verbose))


def pytest_runtest_teardown(item, nextitem=None):
    test_name_log_filter.reset_test_name()


def pytest_html_results_summary(prefix, summary, postfix):
    if SETTINGS.UNIFY_CACHE_STATS:
        import unillm

        stats = unillm.get_cache_stats()
        prefix.extend(
            [
                f"<h4>Unify Cache Stats Report:</h4>",
                f"<p>Hits ({stats.get_percentage_of_cache_hits():.2f}%): {stats.hits} ({stats.canonical_hits} canonical) | Misses ({stats.get_percentage_of_cache_misses():.2f}%): {stats.misses}</p>",
                f"<p>Reads: {stats.reads} | Writes: {stats.writes}</p>",
            ],
        )


@pytest.fixture(autouse=True)
def _set_random_seed():
    random.seed(42)

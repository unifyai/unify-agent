"""
tests/conftest.py
=================

Global pytest configuration for the Unify test suite.

Sections:
  1. Imports and logging guard
  2. Test stubs (DateTime)
  3. Model-cache keying
  4. Command-line options
  5. Custom logging helpers
  6. Session lifecycle hooks
  7. Test run hooks
"""

from __future__ import annotations

import hashlib
import itertools
import logging
import os
import random
import re
import shutil
import tempfile

import pytest
from pytest_metadata.plugin import metadata_key

from datetime import datetime, timezone
from pathlib import Path

# --------------------------------------------------------------------------- #
# 1. Early logging guard                                                      #
# --------------------------------------------------------------------------- #
# Ensure a handler exists before imports that might call logging.basicConfig()
_root_logger_early = logging.getLogger()
if not _root_logger_early.handlers:
    _root_logger_early.addHandler(logging.NullHandler())

from tests.helpers import _lock_file_nb
from tests.settings import SETTINGS


def pytest_report_header(config):
    settings_str = [f"{k}={v}" for k, v in SETTINGS.model_dump().items()]
    return [
        f"UNILLM_CACHE={os.environ.get('UNILLM_CACHE', 'not set')}",
    ] + settings_str


# --------------------------------------------------------------------------- #
# 2. Test stubs (DateTime)                                                    #
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

    def _static_now(time_only: bool = False, as_string: bool = True):
        """Return a fixed timestamp for testing."""
        if not as_string:
            return _FIXED_DATETIME

        label = "UTC"
        if time_only:
            return _FIXED_DATETIME.strftime("%I:%M %p ") + label
        return _FIXED_DATETIME.strftime("%A, %B %d, %Y at %I:%M %p ") + label

    # Patch prompt_helpers.now everywhere it's imported. A module that binds
    # it by name (``from unify.common.prompt_helpers import now as
    # prompt_now``) keeps its own reference, which patching prompt_helpers
    # alone misses, so every loaded ``unify`` module is searched for one.
    # A module first imported during a test binds that test's patched clock,
    # which the marker lets the next test find too.
    from unify.common import prompt_helpers

    _static_now._frozen_prompt_clock = True
    _patch_every_copy(monkeypatch, prompt_helpers.now, _static_now)

    # A transcript session's id is the wall-clock second and random hex
    # (unify/transcripts.py), and a compression summary names the session's
    # file in the request after it. Ids numbered in the order the test opens
    # its sessions keep that request the same on every run; the test's home
    # starts empty, so they never meet a file of an earlier run.
    from unify import transcripts

    _session_numbers = itertools.count(1)
    _session_stamp = _FIXED_DATETIME.strftime("%Y%m%dT%H%M%S")
    monkeypatch.setattr(
        transcripts,
        "_new_session_id",
        lambda: f"{_session_stamp}-{next(_session_numbers):08x}",
    )


def _patch_every_copy(monkeypatch, original, replacement) -> None:
    """Point ``original``'s defining attribute and every by-name copy of it
    in a loaded ``unify`` module at ``replacement``.

    An earlier test's replacement (one marked ``_frozen_prompt_clock``) is
    replaced as well."""
    import sys
    import types

    def _is_copy(value) -> bool:
        return value is original or (
            type(value) is types.FunctionType
            and getattr(value, "_frozen_prompt_clock", False)
        )

    for name, module in list(sys.modules.items()):
        if module is None or not (name == "unify" or name.startswith("unify.")):
            continue
        for attr, value in list(getattr(module, "__dict__", {}).items()):
            if _is_copy(value):
                monkeypatch.setattr(module, attr, replacement)


# --------------------------------------------------------------------------- #
# 3. Command-line options                                                     #
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
# 4. Custom logging helpers                                                   #
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
# 5. Session lifecycle hooks                                                  #
# --------------------------------------------------------------------------- #


def pytest_sessionstart(session):
    if os.environ.get("SKIP_UNIFY_TEST_INIT"):
        return

    if os.environ.get("GITHUB_ACTIONS"):
        import unillm

        unillm.set_cache_backend("local_separate")

    import unify  # local import to avoid affecting stub installation order

    unify.init()


def _parallel_run_result_file(kind: str) -> str | None:
    """The file parallel_run.sh reads this session's ``kind`` result from.

    None outside a parallel_run.sh session. It lives in the run's results
    directory (``UNIFY_TEST_RESULTS_DIR``), not /tmp: the session runs in the
    test sandbox, whose /tmp is private. The name carries the tmux socket as
    well as the session id: every tmux server numbers its sessions from $0,
    and each terminal running parallel_run.sh has a server of its own.
    """
    socket = os.environ.get("UNIFY_TEST_SOCKET")
    session_id = os.environ.get("UNIFY_TMUX_SESSION_ID")
    if not (socket and session_id):
        return None
    directory = os.environ.get("UNIFY_TEST_RESULTS_DIR") or "/tmp"
    return f"{directory}/parallel_run_{kind}_{socket}_{session_id}.txt"


def pytest_sessionfinish(session, exitstatus):
    # Write cache stats to a temp file for parallel_run.sh to consume
    try:
        import unillm

        stats = unillm.get_cache_stats()
        stats_file = _parallel_run_result_file("cache")
        if stats_file:
            with open(stats_file, "w") as f:
                f.write(f"{stats.hits}|{stats.canonical_hits}|{stats.misses}\n")
    except Exception:
        pass  # Don't fail the test run if cache stats writing fails

    # Write LLM provider cost to a temp file for parallel_run.sh to consume
    try:
        cost_file = _parallel_run_result_file("cost")
        if cost_file:
            total_cost = sum(cost for _, cost in _session_costs)
            with open(cost_file, "w") as f:
                f.write(f"{total_cost:.6g}\n")
    except Exception:
        pass


def pytest_unconfigure(config):
    """Restore HOME.

    The test HOME (``unity_test_home`` in the system temp directory) is
    deliberately left in place: every parallel pytest session
    ``parallel_run.sh`` spawns shares it, so wiping it here would pull files
    out from under sessions still running. It accumulates only for the life
    of the CI runner; locally, deleting it gives a clean slate.
    """
    if _original_home is None:
        os.environ.pop("HOME", None)
    else:
        os.environ["HOME"] = _original_home


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
        outcome_file = _parallel_run_result_file("outcome")
        if outcome_file:
            passed = len(terminalreporter.stats.get("passed", []))
            skipped = len(terminalreporter.stats.get("skipped", []))
            with open(outcome_file, "w") as f:
                f.write(f"{passed}|{skipped}\n")
    except Exception:
        pass  # Don't fail the test run if outcome stats writing fails


# --------------------------------------------------------------------------- #
# 6. Test run hooks                                                           #
# --------------------------------------------------------------------------- #

from unillm.cost_tracker import capture_costs

_session_costs: list[tuple[str, float]] = []

_original_home: str | None = None


def pytest_configure(config):
    # ------------------------------------------------------------------
    # Isolate HOME so that tests never touch the real home directory
    # (dotfiles, ~/.cache, the default ~/.unify). The path is fixed, not
    # random, so nothing derived from HOME varies between sessions. The
    # workspace the actor's system prompt embeds resolves under
    # UNIFY_HOME, not HOME; ``unify_home`` below gives each test its own.
    # ------------------------------------------------------------------
    global _original_home
    _original_home = os.environ.get("HOME")
    test_home = os.path.join(tempfile.gettempdir(), "unity_test_home")
    os.makedirs(test_home, exist_ok=True)
    os.environ["HOME"] = test_home

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


@pytest.fixture(autouse=True)
def unify_home(request, monkeypatch):
    """Point ``UNIFY_HOME`` at a home of the test's own, named by its node id.

    The actor's system prompt embeds the workspace path under the home, and
    the package installer's output names the environment under it, so the
    path must be the same on every run for recorded LLM responses to replay.
    The test holds a lock on its home while it runs: a concurrent run of the
    same test (``--repeat``, another checkout) takes the next free slot
    instead of sharing the directory. The home starts empty and is removed
    afterwards.
    """
    homes = Path(tempfile.gettempdir()) / "unity_test_homes"
    homes.mkdir(exist_ok=True)
    digest = hashlib.md5(request.node.nodeid.encode()).hexdigest()[:12]
    for slot in itertools.count():
        home = homes / (digest if slot == 0 else f"{digest}-{slot}")
        lock = open(f"{home}.lock", "w")
        try:
            _lock_file_nb(lock)
            break
        except BlockingIOError:
            lock.close()
    shutil.rmtree(home, ignore_errors=True)
    home.mkdir()
    monkeypatch.setenv("UNIFY_HOME", str(home))
    cwd = os.getcwd()
    yield home
    # ``unify act`` chdirs into the workspace.
    os.chdir(cwd)
    shutil.rmtree(home, ignore_errors=True)
    # The lock file stays: were it deleted, a run still holding the old file
    # and a run creating a new one could both take this home.
    lock.close()


# No test reaches a real model provider or the network (the lead's rule, 13 Oct 2026: "WE DO NOT ADD TESTS THAT
# NEED A REAL PROVIDER KEY AND ALWAYS MOCK THOSE TESTS WITH STUBBED LLM CALLS"). A test's model is scripted:
# tests/scripted_model.py (``ScriptedModel``, per agent kind) or tests/cache_discipline_helpers.py (``scripted(...)``,
# which replaces unillm's transport above the provider call). The guard below holds in every session, with or
# without a key loaded: unillm's provider calls raise :data:`PROVIDER_REFUSAL`, and a socket may connect only to
# this machine. The markers that once let a test reach a provider fail the test that carries them.
STUB_HINT = (
    "script the model with tests/scripted_model.py (ScriptedModel) or tests/cache_discipline_helpers.py "
    "(scripted(...)), which replace unillm's transport"
)
PROVIDER_REFUSAL = f"this test reached a real model provider; tests never call one: {STUB_HINT}"
NETWORK_REFUSAL = "this test opened a connection to {address!r}, off this machine; tests never use the network: {hint}"
REMOVED_MARKERS = ("live_provider", "requires_provider_key")


def _loopback(address) -> bool:
    """Whether a socket address is on this machine (a Unix path, localhost, 127.0.0.0/8 or ::1)."""
    import ipaddress

    if not isinstance(address, tuple):
        return True  # AF_UNIX paths and abstract names
    host = str(address[0])
    if host in ("localhost", ""):
        return True
    try:
        return ipaddress.ip_address(host.split("%", 1)[0]).is_loopback
    except ValueError:
        return host.endswith(".localhost")


@pytest.fixture(autouse=True)
def _no_provider_or_network(monkeypatch):
    """Refuse unillm's provider calls and any socket connection off this machine, in every test (the comment
    above)."""
    import socket

    import unillm.clients.uni_llm as uni_llm

    async def refuse_async(*args, **kwargs):
        raise AssertionError(PROVIDER_REFUSAL)

    def refuse(*args, **kwargs):
        raise AssertionError(PROVIDER_REFUSAL)

    for name in ("acompletion", "completion", "aembedding", "embedding"):
        if hasattr(uni_llm.litellm, name):
            monkeypatch.setattr(
                uni_llm.litellm,
                name,
                refuse_async if name.startswith("a") else refuse,
            )

    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex

    def connect(sock, address):
        if not _loopback(address):
            raise AssertionError(NETWORK_REFUSAL.format(address=address, hint=STUB_HINT))
        return real_connect(sock, address)

    def connect_ex(sock, address):
        if not _loopback(address):
            raise AssertionError(NETWORK_REFUSAL.format(address=address, hint=STUB_HINT))
        return real_connect_ex(sock, address)

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", connect_ex)
    yield


def pytest_runtest_setup(item):
    test_name_log_filter.set_test_name(item.nodeid)
    for marker in REMOVED_MARKERS:
        if item.get_closest_marker(marker) is not None:
            pytest.fail(
                f"{item.nodeid} is marked {marker}, but no test may need a provider key: {STUB_HINT}, and remove "
                "the marker",
                pytrace=False,
            )


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
        # a call that returned nothing has an unknown cost (None), which the total leaves out
        total = sum(e.provider_cost for e in events if e.provider_cost is not None)
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

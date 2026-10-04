#!/usr/bin/env bash
set -euo pipefail

# Resolve script directory first (needed for relative path resolution)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

# Source the repo root's .env (not committed) so LLM provider keys and any
# other configuration reach every pytest session.
_ENV_FILE="$SCRIPT_DIR/../.env"
if [ -f "$_ENV_FILE" ]; then
  # shellcheck disable=SC1090
  set -a
  . "$_ENV_FILE"
  set +a
fi
unset _ENV_FILE

# Source common utilities (socket derivation, locale, timeout handling)
source "$SCRIPT_DIR/_shell_common.sh"

# Source argument parsing
source "$SCRIPT_DIR/_parse_args.sh"

# ---- Increase file descriptor limit ----
# Parallel sessions open many files and connections at once. macOS defaults
# to 256 per process, which is easily exceeded. This setting is inherited by
# all child processes (tmux sessions, pytest).
ulimit -n 8192 2>/dev/null || true

TMUX_SOCKET="$UNIFY_TMUX_SOCKET"

# A runner started inside another run's session (the runner's own tests, or
# the shell a failed session leaves open) inherits that session's id. Nothing
# this process starts outside tmux is a session, so its pytest collection must
# not write result files under that id; each session exports its own.
unset UNIFY_TMUX_SESSION_ID

# Resolve repo root (parent of this script's directory)
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd -P)"

# ---- Log directory naming ----
# Log subdirectories use a datetime-prefixed format for natural time-based
# ordering in the filesystem. Format: YYYY-MM-DDTHH-MM-SS_{socket_name}
# This makes it easy to find recent test runs while preserving terminal isolation.
_derive_log_subdir() {
  local socket_name="$1"
  local datetime
  datetime=$(date +"%Y-%m-%dT%H-%M-%S")
  echo "${datetime}_${socket_name}"
}

# Generate log subdir once at script start (stable for this run)
LOG_SUBDIR="${UNIFY_LOG_SUBDIR:-$(_derive_log_subdir "$TMUX_SOCKET")}"
LOG_DIR="$REPO_ROOT/logs/pytest/$LOG_SUBDIR"

# ---- Per-session stores ----
# Every pytest process opens the SQLite store named by UNIFY_STORE_PATH.
# Each session gets its own file under the run's log directory, so sessions
# never share tables and each store stays on disk next to the session's log
# for post-mortem queries. A UNIFY_STORE_PATH the caller already set
# (environment, .env or --env) is honoured instead, and every session then
# shares that one store.
STORES_DIR="$LOG_DIR/stores"

# Wrapper for all tmux commands to use our isolated socket
# LC_ALL=en_US.UTF-8 ensures Unicode emojis work in session names
tmux_cmd() {
  LC_ALL=en_US.UTF-8 tmux -L "$TMUX_SOCKET" "$@"
}

# ---- Cleanup on interrupt ----
# Track session IDs for cleanup on SIGINT/SIGTERM
declare -a CREATED_SESSION_IDS=()
declare -a CREATED_RESULT_PATHS=()
declare -a CREATED_SESSION_NAMES=()

# ---- Inline pass/fail reporting ----
# Track which sessions we've already reported completion for (":$0:$3:")
REPORTED_COMPLETIONS=":"

# Check if a session ID has been reported. The match takes the delimiters on
# both sides: a bare substring test counts "$1" as reported once "$10" is.
_is_reported() {
  local sid="$1"
  [[ "$REPORTED_COMPLETIONS" == *":${sid}:"* ]]
}

# Mark a session ID as reported
_mark_reported() {
  local sid="$1"
  REPORTED_COMPLETIONS="${REPORTED_COMPLETIONS}${sid}:"
}

# Record one finished session's duration, cache stats and cost for the
# end-of-run sorted output. The per-session files are written by
# tests/conftest.py into this run's RESULTS_DIR (UNIFY_TEST_RESULTS_DIR in the
# session), keyed on the socket and the tmux session id: every tmux server
# numbers its sessions from $0, so the id alone is shared by overlapping runs
# from different terminals. They are not in /tmp: each pytest process runs in
# the test sandbox (tests/_test_sandbox.py), whose /tmp is its own.
_record_session_result() {
  local sid="$1" status="$2" base="$3"
  [[ -n "${START_TIMES_FILE:-}" && -f "$START_TIMES_FILE" ]] || return 0
  local start_time
  start_time=$(grep "^$sid " "$START_TIMES_FILE" 2>/dev/null | cut -d' ' -f2)
  [[ -n "$start_time" ]] || return 0
  local duration=$(( $(date +%s) - start_time ))
  local key="${TMUX_SOCKET}_${sid}"

  local hits=0 canonical=0 misses=0
  local stats_file="${RESULTS_DIR:-/tmp}/parallel_run_cache_${key}.txt"
  if [[ -f "$stats_file" ]]; then
    IFS='|' read -r hits canonical misses < "$stats_file" 2>/dev/null || true
    rm -f "$stats_file"
  fi

  local cost="0"
  local cost_file="${RESULTS_DIR:-/tmp}/parallel_run_cost_${key}.txt"
  if [[ -f "$cost_file" ]]; then
    cost=$(tr -d '[:space:]' < "$cost_file" 2>/dev/null || echo 0)
    rm -f "$cost_file"
  fi

  # pytest exits 0 when every test skips, so a green session may have run
  # nothing. Report those as skipped rather than passed.
  local outcome_passed outcome_skipped
  local outcome_file="${RESULTS_DIR:-/tmp}/parallel_run_outcome_${key}.txt"
  if [[ -f "$outcome_file" ]]; then
    IFS='|' read -r outcome_passed outcome_skipped < "$outcome_file" 2>/dev/null || true
    rm -f "$outcome_file"
    if [[ "$status" == "pass" && "${outcome_passed:-0}" == "0" && "${outcome_skipped:-0}" != "0" ]]; then
      status="skip"
    fi
  fi

  echo "$duration|$status|$hits|$canonical|$misses|${cost:-0}|$base" >> "$RESULTS_FILE"
}

# Report any sessions that have completed since last check
# Prints pass/fail status inline during the drip-feed phase
report_completed_sessions() {
  # Guard against empty array (set -u treats empty array expansion as unbound)
  (( ${#CREATED_SESSION_IDS[@]} == 0 )) && return 0

  local index sid receipt status base
  for (( index=0; index<${#CREATED_SESSION_IDS[@]}; index++ )); do
    sid="${CREATED_SESSION_IDS[$index]}"
    _is_reported "$sid" && continue
    receipt="${CREATED_RESULT_PATHS[$index]}"
    # The child publishes its exit receipt before it renames or closes the
    # session. A missing session is not evidence of a successful test.
    [[ -f "$receipt" && ! -L "$receipt" ]] || continue
    status=$(cat "$receipt") || continue
    if [[ ! "$status" =~ ^(0|[1-9][0-9]{0,2})$ ]] || (( status > 255 )); then
      echo "Invalid exit receipt for $sid: $receipt" >&2
      continue
    fi
    base="${CREATED_SESSION_NAMES[$index]}"
    if (( status == 0 )); then
      echo "  - p ✅ $base"
      _record_session_result "$sid" pass "$base"
    else
      echo "  - f ❌ $base"
      _record_session_result "$sid" fail "$base"
    fi
    _mark_reported "$sid"
  done
}

_cleanup_sessions() {
  local sig="${1:-}"
  if (( ${#CREATED_SESSION_IDS[@]} > 0 )); then
    echo ""
    echo "Caught signal${sig:+ ($sig)}. Cleaning up ${#CREATED_SESSION_IDS[@]} session(s)..."
    for sid in "${CREATED_SESSION_IDS[@]}"; do
      # Send SIGTERM to allow graceful Python shutdown, then kill session
      # First, send SIGTERM to all processes in the session
      tmux_cmd list-panes -t "$sid" -F '#{pane_pid}' 2>/dev/null | while read -r pid; do
        if [[ -n "$pid" ]]; then
          # Kill the entire process group to catch child processes
          kill -TERM "-$pid" 2>/dev/null || true
        fi
      done
      # Brief wait for graceful shutdown
      sleep 0.1
      # Then kill the tmux session
      tmux_cmd kill-session -t "$sid" 2>/dev/null || true
    done
    echo "Cleanup complete."
  fi
}

# Set up signal handlers for graceful cleanup
trap '_cleanup_sessions INT; exit 130' INT
trap '_cleanup_sessions TERM; exit 143' TERM

# ---- Configurable directory excludes (by name) ----
# Note: the fixture directories hold test data for the runner's own tests
# (tests/parallel_run/), not tests themselves; those tests hand the fixture
# files to this script explicitly. Excluding them by name keeps a directory
# sweep from spawning, say, the hang fixture that exists only to be killed by
# --session-timeout.
EXCLUDE_DIRS=( .git .hg .svn .venv venv .mypy_cache .pytest_cache __pycache__ .idea .vscode fixtures hang_fixtures store_fixtures report_fixtures )

# Parse arguments
# Returns: 0=success, 1=help requested, 2=error
#
# Capture the exit code via `|| _parse_result=$?`: a bare
# `parse_test_args "$@"; _parse_result=$?` aborts under `set -e` before the
# assignment runs, turning a help/usage exit into a silent exit-1.
_parse_result=0
parse_test_args "$@" || _parse_result=$?
if (( _parse_result == 1 )); then
  # Help requested
  HELP_SCRIPT_NAME="parallel_run.sh"
  print_help
  exit 0
elif (( _parse_result == 2 )); then
  # Error (already printed by parse_test_args)
  exit 2
fi
unset _parse_result

# Per-session hang guard: UNIFY_TEST_SESSION_TIMEOUT supplies the default
# when --session-timeout is not passed. Whole-run --timeout alone lets one
# hung session burn the entire budget.
if (( SESSION_TIMEOUT == 0 )); then
  if [[ -n "${UNIFY_TEST_SESSION_TIMEOUT:-}" && "${UNIFY_TEST_SESSION_TIMEOUT}" =~ ^[0-9]+$ && "${UNIFY_TEST_SESSION_TIMEOUT}" -ge 1 ]]; then
    SESSION_TIMEOUT="${UNIFY_TEST_SESSION_TIMEOUT}"
  fi
fi

# Build pytest marker filter based on flags
MARKER_FILTER=""
if (( EVAL_ONLY )); then
  MARKER_FILTER="-m eval"
elif (( SYMBOLIC_ONLY )); then
  MARKER_FILTER="-m 'not eval'"
elif (( DETERMINISTIC_ONLY )); then
  # "not eval" still leaves every llm_call test in, and those answer to
  # whatever the shared cache happens to hold: under UNILLM_CACHE=read-only a
  # request with no recorded entry raises, so a sweep meant to find broken
  # tests reports cache gaps instead. Excluding both markers leaves only the
  # tests whose outcome depends on this repository alone.
  MARKER_FILTER="-m 'not eval and not llm_call'"
fi

# ---------------------------------------------------------------------------
# Helper: check if a var name is in the --env overrides
# Usage: is_var_in_env_overrides VAR_NAME
# ---------------------------------------------------------------------------
is_var_in_env_overrides() {
  local var_name="$1"
  for kv in "${ENV_OVERRIDES[@]+"${ENV_OVERRIDES[@]}"}"; do
    if [[ "$kv" == "${var_name}="* ]]; then
      return 0
    fi
  done
  return 1
}

# ---------------------------------------------------------------------------
# Helper: resolve the store a session should open
# Usage: resolve_store_path SESSION_NAME
# ---------------------------------------------------------------------------
# --env wins over the environment (which includes .env); with neither set,
# the runner hands out one store per session under STORES_DIR.
RUNNER_OWNS_STORES=1
if is_var_in_env_overrides "UNIFY_STORE_PATH" || [[ -n "${UNIFY_STORE_PATH:-}" ]]; then
  RUNNER_OWNS_STORES=0
fi

resolve_store_path() {
  local session_name="$1"
  for kv in "${ENV_OVERRIDES[@]+"${ENV_OVERRIDES[@]}"}"; do
    if [[ "$kv" == "UNIFY_STORE_PATH="* ]]; then
      echo "${kv#UNIFY_STORE_PATH=}"
      return 0
    fi
  done
  if [[ -n "${UNIFY_STORE_PATH:-}" ]]; then
    echo "$UNIFY_STORE_PATH"
    return 0
  fi
  echo "$STORES_DIR/$(store_basename "$session_name").sqlite"
}

# A session name can carry a parametrised node id: characters the filesystem
# rejects and more bytes than one path component allows. The store file keeps
# a readable prefix and a hash of the full name, the same shape as the log
# filenames conftest.py writes.
store_basename() {
  local name="$1"
  name="${name//[^A-Za-z0-9._-]/_}"
  if (( ${#name} > 120 )); then
    local digest
    digest=$(printf '%s' "$1" | shasum | cut -c1-8)
    name="${name:0:111}_${digest}"
  fi
  printf '%s' "$name"
}

# ---------------------------------------------------------------------------
# Helper: build environment exports string from --env overrides and system env
# ---------------------------------------------------------------------------
build_env_exports() {
  local exports=""

  # Always export the socket name for tmux isolation
  exports="$exports UNIFY_TEST_SOCKET=$TMUX_SOCKET"

  # Export the log subdir for datetime-prefixed log directory naming
  exports="$exports UNIFY_LOG_SUBDIR=$LOG_SUBDIR"

  # ---------------------------------------------------------------------------
  # OpenTelemetry Configuration for Cross-Repo Traces
  # ---------------------------------------------------------------------------
  # Enable OTEL tracing in unify and unillm so the spans from a single test
  # are aggregated into one {trace_id}.jsonl file under logs/all/. Per-test
  # isolation comes from unique trace_ids, not subdirectories.
  local otel_log_dir="$REPO_ROOT/logs/all"

  # Enable OTEL master switches (unless explicitly disabled via --env)
  if ! is_var_in_env_overrides "UNIFY_OTEL"; then
    exports="$exports UNIFY_OTEL=true"
  fi
  if ! is_var_in_env_overrides "UNILLM_OTEL"; then
    exports="$exports UNILLM_OTEL=true"
  fi

  # Point both repos to the shared OTEL log directory (unless explicitly set via --env)
  if ! is_var_in_env_overrides "UNIFY_OTEL_LOG_DIR"; then
    exports="$exports UNIFY_OTEL_LOG_DIR=$otel_log_dir"
  fi
  if ! is_var_in_env_overrides "UNILLM_OTEL_LOG_DIR"; then
    exports="$exports UNILLM_OTEL_LOG_DIR=$otel_log_dir"
  fi

  # Add all --env flag overrides
  for kv in "${ENV_OVERRIDES[@]+"${ENV_OVERRIDES[@]}"}"; do
    exports="$exports $kv"
  done

  echo "$exports"
}

# Reset positional parameters safely under nounset (only expand if set)
set -- ${POSITIONAL_ARGS[@]+"${POSITIONAL_ARGS[@]}"}

# Always operate from the repo root for discovery, regardless of where the script was invoked
cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# Worktree dependency symlinks (for local editable packages)
# ---------------------------------------------------------------------------
# pyproject.toml references ../unillm as an editable local package. In the
# main repo at ~/unify that resolves to ~/unillm. In a worktree at
# ~/unify/.claude/worktrees/xyz it resolves to
# ~/unify/.claude/worktrees/unillm, which doesn't exist by default.
#
# This function creates symlinks at the worktree parent level so that
# `uv sync` works correctly in any worktree without manual setup.
ensure_worktree_dependency_symlinks() {
  local git_file="$REPO_ROOT/.git"

  # Only applies to git worktrees (where .git is a file, not a directory)
  if [[ ! -f "$git_file" ]]; then
    return 0
  fi

  # Parse the gitdir from .git file (format: "gitdir: /path/to/main/.git/worktrees/name")
  local gitdir
  gitdir=$(sed 's/^gitdir: //' "$git_file")

  # Derive the main repo path: /main/repo/.git/worktrees/name -> /main/repo
  local main_repo
  main_repo=$(dirname "$(dirname "$(dirname "$gitdir")")")

  # The worktree parent is where all worktrees live (e.g., ~/.cursor/worktrees/unity/)
  local worktree_parent
  worktree_parent=$(dirname "$REPO_ROOT")

  # For each local dependency, ensure a symlink exists at the worktree parent level
  local deps=("unillm")
  for dep in "${deps[@]}"; do
    local target="$main_repo/../$dep"  # e.g., ~/unillm (sibling of main repo)
    local link="$worktree_parent/$dep"  # e.g., ~/.cursor/worktrees/unity/unillm

    # Resolve to absolute path
    if [[ -d "$target" ]]; then
      target=$(cd "$target" && pwd -P)
    else
      # Local dependency doesn't exist; skip (will fail later in uv sync with clear error)
      continue
    fi

    # Create symlink if missing or pointing to wrong location
    if [[ -L "$link" ]]; then
      local current_target
      current_target=$(readlink "$link")
      if [[ "$current_target" != "$target" ]]; then
        echo "Updating worktree symlink: $link -> $target" >&2
        rm "$link"
        ln -s "$target" "$link"
      fi
    elif [[ ! -e "$link" ]]; then
      echo "Creating worktree symlink: $link -> $target" >&2
      ln -s "$target" "$link"
    fi
  done
}

# Ensure symlinks before venv setup (silent if not in a worktree)
ensure_worktree_dependency_symlinks

# ---------------------------------------------------------------------------
# Python environment (uv + repo-local .venv)
# ---------------------------------------------------------------------------
# Avoid hardcoding home-directory venv paths or relying on `python` being
# present on PATH. Instead, bootstrap and use the repo-local `.venv`.
VENV_DIR="$REPO_ROOT/.venv"
VENV_PY="$VENV_DIR/bin/python"
UV_BIN=""

ensure_uv() {
  # Prefer an existing uv on PATH.
  if command -v uv >/dev/null 2>&1; then
    UV_BIN="$(command -v uv)"
    return 0
  fi
  # Common pip --user install location.
  if [[ -x "${HOME}/.local/bin/uv" ]]; then
    UV_BIN="${HOME}/.local/bin/uv"
    return 0
  fi

  echo "uv not found; installing via pip (user install)..." >&2
  if ! command -v python3 >/dev/null 2>&1; then
    echo "Error: python3 not found; cannot install uv." >&2
    return 1
  fi
  if ! python3 -m pip install --user uv; then
    echo "Error: failed to install uv via pip." >&2
    return 1
  fi

  if command -v uv >/dev/null 2>&1; then
    UV_BIN="$(command -v uv)"
    return 0
  fi
  if [[ -x "${HOME}/.local/bin/uv" ]]; then
    UV_BIN="${HOME}/.local/bin/uv"
    return 0
  fi

  echo "Error: uv was installed but is not discoverable (PATH may be missing ~/.local/bin)." >&2
  echo "Try adding ~/.local/bin to PATH, or install uv manually: https://github.com/astral-sh/uv" >&2
  return 1
}

ensure_project_venv() {
  # Fast path: venv exists and can run pytest.
  if [[ -x "$VENV_PY" ]]; then
    if "$VENV_PY" -m pytest --version >/dev/null 2>&1; then
      return 0
    fi
  fi

  ensure_uv || return 1

  echo "Bootstrapping project virtualenv with uv (uv sync --all-groups)..." >&2
  if ! UV_PROJECT_ENVIRONMENT="$VENV_DIR" "$UV_BIN" sync --all-groups; then
    echo "Error: 'uv sync --all-groups' failed." >&2
    return 1
  fi

  if [[ ! -x "$VENV_PY" ]]; then
    echo "Error: expected venv python at '$VENV_PY' after uv sync." >&2
    return 1
  fi
  if ! "$VENV_PY" -m pytest --version >/dev/null 2>&1; then
    echo "Error: pytest is not available in the project venv after uv sync." >&2
    return 1
  fi
}

if ! ensure_project_venv; then
  exit 1
fi

# Build the command to run in each tmux session
run_cmd() {
  local target="$1"        # pytest target (file path or node id)
  local marker_arg="$2"    # optional marker filter (e.g., "-m eval")
  local session_name="$3"  # unique session name (names the session's store)
  local result_path="$4"   # controller-owned occurrence, never derived from a node
  # Build the inner script first with safe %q for path/target, then quote the whole script with %q
  local inner
  local env_exports
  # Every value the session depends on is exported inside the command itself
  # rather than only inherited: tmux launches the command through the user's
  # default shell, and zsh sources ~/.zshenv even for plain `zsh -c`, so a
  # dotfile exporting one of these would otherwise silently clobber the
  # inherited value. UTF-8 locale keeps emoji session names intact;
  # UNILLM_CACHE_STATS must be set before unillm is imported.
  env_exports="export LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 UNILLM_CACHE_STATS=true"
  env_exports="$env_exports $(printf 'UNIFY_STORE_PATH=%q' "$(resolve_store_path "$session_name")")"
  env_exports="$env_exports $(printf 'UNIFY_TEST_RESULTS_DIR=%q' "$RESULTS_DIR")"
  # Append user-provided --env overrides plus the socket/log-dir/OTEL exports
  env_exports="$env_exports$(build_env_exports)"
  # Build pytest command with optional marker filter and extra args
  local pytest_cmd
  local extra_args=""
  # Append any extra pytest args passed via --
  if (( ${#PYTEST_EXTRA_ARGS[@]} > 0 )); then
    for arg in "${PYTEST_EXTRA_ARGS[@]}"; do
      extra_args="$extra_args $(printf '%q' "$arg")"
    done
  fi
  if [[ -n "$marker_arg" ]]; then
    pytest_cmd=$(printf '%q -m pytest %s %s %q' "$VENV_PY" "$marker_arg" "$extra_args" "$target")
  else
    pytest_cmd=$(printf '%q -m pytest %s %q' "$VENV_PY" "$extra_args" "$target")
  fi
  # Wrap pytest in GNU/coreutils timeout so a hung session fails in minutes
  # instead of holding the whole-run --timeout budget. Prefer timeout(1);
  # fall back to gtimeout on macOS Homebrew. Skip wrapping if neither exists
  # (local mac without coreutils) — whole-run --timeout remains the backstop.
  if (( SESSION_TIMEOUT > 0 )); then
    local timeout_bin=""
    if command -v timeout >/dev/null 2>&1; then
      timeout_bin="timeout"
    elif command -v gtimeout >/dev/null 2>&1; then
      timeout_bin="gtimeout"
    fi
    if [[ -n "$timeout_bin" ]]; then
      # --foreground: forward signals to pytest.
      # --kill-after=30s: escalate to SIGKILL if pytest ignores SIGTERM.
      pytest_cmd=$(printf '%q --foreground --kill-after=30s %q %s' "$timeout_bin" "${SESSION_TIMEOUT}s" "$pytest_cmd")
    else
      echo "Warning: --session-timeout $SESSION_TIMEOUT requested but neither timeout nor gtimeout is on PATH; hanging sessions will not be killed per-session." >&2
    fi
  fi
  # Build inner command with socket name directly interpolated (not via env var)
  # This ensures tmux commands target the correct isolated server
  # Note: LC_ALL=en_US.UTF-8 is required for Unicode emoji support in tmux session names
  # Note: Log paths are auto-derived by conftest.py using UNIFY_TEST_SOCKET + semantic naming
  # Inner command runs inside tmux session after pytest completes.
  # The rename-session uses "|| true" to gracefully handle race conditions
  # where multiple sessions complete simultaneously or external agents interfere.
  # The session ID is captured BEFORE pytest runs and exported as UNIFY_TMUX_SESSION_ID
  # so pytest's conftest.py can write cache stats to a known temp file location.
  local receipt_cmd
  receipt_cmd=$(printf '%q -B %q %q "$status"' "$VENV_PY" "$SCRIPT_DIR/_parallel_result.py" "$result_path")
  inner=$(printf '%s; export UNIFY_TMUX_SESSION_ID=$(LC_ALL=en_US.UTF-8 tmux -L %q display-message -p -t "$TMUX_PANE" "#{session_id}"); cd %q && %s; status=$?; if ! %s; then echo "Failed to publish test exit receipt" >&2; exit 125; fi; sname=$(LC_ALL=en_US.UTF-8 tmux -L %q display-message -p -t "$TMUX_PANE" "#{session_name}"); base="$sname"; case "$sname" in "p ✅ "*) base="${sname#p ✅ }" ;; "f ❌ "*) base="${sname#f ❌ }" ;; "r ⏳ "*) base="${sname#r ⏳ }" ;; esac; if [ $status -eq 0 ]; then pfx="p ✅"; else pfx="f ❌"; fi; LC_ALL=en_US.UTF-8 tmux -L %q rename-session -t "$sname" "$pfx $base" 2>/dev/null || true; if [ $status -eq 0 ]; then (sleep 10; LC_ALL=en_US.UTF-8 tmux -L %q kill-session -t "$UNIFY_TMUX_SESSION_ID" 2>/dev/null; if ! LC_ALL=en_US.UTF-8 tmux -L %q ls >/dev/null 2>&1; then LC_ALL=en_US.UTF-8 tmux -L %q kill-server 2>/dev/null || true; fi) >/dev/null 2>&1 & disown; echo "All tests passed. This tmux session will close in 10s..."; fi; echo; echo "pytest exited with code: $status"; echo "(You are now in a shell. Press Ctrl-D to close this window.)"; exec bash -l' "$env_exports" "$TMUX_SOCKET" "$REPO_ROOT" "$pytest_cmd" "$receipt_cmd" "$TMUX_SOCKET" "$TMUX_SOCKET" "$TMUX_SOCKET" "$TMUX_SOCKET" "$TMUX_SOCKET")
  printf 'bash -c %q' "$inner"
}

# Ensure we don't collide with existing sessions.
# Checks status prefix variants (r ⏳, p ✅, f ❌) since sessions get renamed
# after completion, which could cause race conditions with subsequent runs.
#
# When called WITHOUT a prefix: checks unprefixed AND all prefixed variants
# When called WITH a prefix: only checks prefixed variants (the unprefixed
#   session is the one we just created and are about to rename)
unique_session_name() {
  local input="$1" n=1

  # Strip any status prefix to get the base name for collision checking
  local base="$input"
  local prefix=""
  case "$input" in
    "r ⏳ "*) base="${input#r ⏳ }"; prefix="r ⏳ " ;;
    "p ✅ "*) base="${input#p ✅ }"; prefix="p ✅ " ;;
    "f ❌ "*) base="${input#f ❌ }"; prefix="f ❌ " ;;
  esac

  local candidate="$base"
  while true; do
    local found=0
    # If input had NO prefix, check unprefixed version too
    # (If it had a prefix, skip - that's our just-created session we're renaming)
    if [[ -z "$prefix" ]]; then
      tmux_cmd has-session -t "$candidate" 2>/dev/null && found=1
    fi
    # Always check all prefixed versions to detect renamed sessions
    tmux_cmd has-session -t "r ⏳ $candidate" 2>/dev/null && found=1
    tmux_cmd has-session -t "p ✅ $candidate" 2>/dev/null && found=1
    tmux_cmd has-session -t "f ❌ $candidate" 2>/dev/null && found=1

    if (( found == 0 )); then
      break
    fi
    ((n++))
    candidate="${base}-${n}"
  done

  # Return with original prefix (if any)
  printf "%s%s" "$prefix" "$candidate"
}

# Rename session with retry on "duplicate session" errors.
# Race conditions (e.g., multiple agents, fast-completing tests) can cause
# the unique_session_name check to pass but the rename to fail. This helper
# retries with incrementing suffixes until success or max retries.
rename_session_with_retry() {
  local sid="$1"
  local target_name="$2"
  local max_retries=3
  local attempt=0
  local current_name="$target_name"

  while (( attempt < max_retries )); do
    if tmux_cmd rename-session -t "$sid" "$current_name" 2>/dev/null; then
      printf "%s" "$current_name"
      return 0
    fi
    # Rename failed - likely duplicate. Add/increment suffix and retry.
    ((attempt++)) || true
    # Strip any existing retry suffix and add new one
    local base="${target_name%-dup[0-9]*}"
    current_name="${base}-dup${attempt}"
    sleep 0.1  # Brief delay before retry
  done

  # All retries failed - log warning but don't crash the script.
  # The session exists (with unprefixed name), tests will still run.
  echo "Warning: rename-session failed after $max_retries retries for $target_name" >&2
  # Return the session's current name (query it since rename failed)
  local actual_name
  actual_name=$(tmux_cmd display-message -p -t "$sid" "#{session_name}" 2>/dev/null || echo "$target_name")
  printf "%s" "$actual_name"
  return 0  # Don't fail the script
}

# Count currently pending (running) sessions in our socket
count_pending_sessions() {
  local count=0
  while IFS= read -r name; do
    if [[ "$name" == "r"* ]]; then
      ((count++)) || true
    fi
  done < <(tmux_cmd list-sessions -F "#{session_name}" 2>/dev/null || true)
  echo "$count"
}

# Wait until we have fewer than MAX_JOBS pending sessions
# While waiting, report any sessions that have completed (inline pass/fail feedback)
wait_for_job_slot() {
  if (( MAX_JOBS == 0 )); then
    return 0  # No limit
  fi

  while true; do
    local pending
    pending=$(count_pending_sessions)
    if (( pending < MAX_JOBS )); then
      # Report any completions that freed up this slot BEFORE returning
      # (ensures pass/fail appears before the new pending session that fills this slot)
      report_completed_sessions
      return 0
    fi
    # Report completions while waiting for a slot
    report_completed_sessions
    sleep 0.5
  done
}

# Turn a file path (or pytest node id) into a session base name
#   ./animals/dogs/test_bark.py               -> animals-dogs-test_bark
#   ./animals/dogs/test_bark.py::test_woof    -> animals-dogs-test_bark--test_woof
session_basename_for() {
  local original="$1"
  local p
  local node_suffix=""

  # If a pytest node id is provided, split off the suffix after "::"
  if [[ "$original" == *"::"* ]]; then
    local base="${original%%::*}"
    node_suffix="${original#${base}::}"
    p="$base"
  else
    p="$original"
  fi

  # normalize to a relative-looking path for naming
  [[ "$p" = /* ]] || p="./${p#./}"
  p="${p%.py}"
  p="${p#./}"
  # Drop leading 'tests/' to avoid 'tests-' prefix in session names
  p="${p#tests/}"
  p="${p//\//-}"

  # If we have a node suffix, sanitize it and append
  if [[ -n "$node_suffix" ]]; then
    local ns="$node_suffix"
    ns="${ns//::/-}"
    ns="${ns// /-}"
    ns="${ns//[/}"
    ns="${ns//]/}"
    ns="${ns//(/}"
    ns="${ns//)/}"
    ns="${ns//,/}"
    ns="${ns//:/-}"
    ns="${ns//=/-}"
    ns="${ns//./-}"
    p="${p}--${ns}"
  fi

  printf "%s" "$p"
}

# Collect args: files and/or directories to search
declare -a roots=()
declare -a direct_files=()
declare -a direct_nodes=()

if (( $# == 0 )); then
  roots=( "." )
else
  for arg in "$@"; do
    if [[ "$arg" == *"::"* ]]; then
      # pytest node id: extract base file and suffix; resolve base relative to caller/tests/root
      base="${arg%%::*}"
      suffix="${arg#${base}::}"
      base_path=""
      if [[ -f "$base" ]]; then
        base_path="$base"
      elif [[ -f "$SCRIPT_DIR/$base" ]]; then
        base_path="$SCRIPT_DIR/$base"
      elif [[ -f "$REPO_ROOT/$base" ]]; then
        base_path="$REPO_ROOT/$base"
      fi
      if [[ -n "$base_path" ]]; then
        repo_rel="${base_path#$REPO_ROOT/}"
        if [[ "${repo_rel##*/}" == test_*.py ]]; then
          direct_nodes+=( "${repo_rel}::${suffix}" )
        else
          echo "Warning: Skipping node not under a test_*.py file: $arg" >&2
        fi
      else
        echo "Warning: Skipping non-existent test node (file missing): $arg" >&2
      fi
    elif [[ -f "$arg" || -f "$SCRIPT_DIR/$arg" || -f "$REPO_ROOT/$arg" ]]; then
      # only include Python test files directly (names starting with test_)
      file_path="$arg"
      if [[ ! -f "$file_path" ]]; then
        if [[ -f "$SCRIPT_DIR/$arg" ]]; then
          file_path="$SCRIPT_DIR/$arg"
        else
          file_path="$REPO_ROOT/$arg"
        fi
      fi
      repo_rel="${file_path#$REPO_ROOT/}"
      if [[ "${repo_rel##*/}" == test_*.py ]]; then
        direct_files+=( "$repo_rel" )
      fi
    elif [[ -d "$arg" || -d "$SCRIPT_DIR/$arg" || -d "$REPO_ROOT/$arg" ]]; then
      dir_path="$arg"
      if [[ ! -d "$dir_path" ]]; then
        if [[ -d "$SCRIPT_DIR/$arg" ]]; then
          dir_path="$SCRIPT_DIR/$arg"
        else
          dir_path="$REPO_ROOT/$arg"
        fi
      fi
      repo_rel="${dir_path#$REPO_ROOT/}"
      roots+=( "$repo_rel" )
    else
      echo "Warning: Skipping non-existent path: $arg" >&2
    fi
  done
  if (( ${#roots[@]} == 0 && ${#direct_files[@]} == 0 && ${#direct_nodes[@]} == 0 )); then
    echo "No valid directories, files, or tests provided." >&2
    exit 1
  fi
fi

# Build a safe find pipeline:
# find <roots> -mindepth 1 \( -type d \( -name EX1 -o EX2 ... \) -prune \) -o \( -type f -name "test_*.py" -print0 \)
# Note: -mindepth 1 ensures root directories aren't pruned even if they match EXCLUDE_DIRS
# (e.g., explicitly passing "fixtures/" should search it, not prune it)
build_find_cmd() {
  local -a cmd=( find )
  if (( ${#roots[@]} )); then
    cmd+=( "${roots[@]}" )
  else
    cmd+=( "." )
  fi

  # -mindepth 1: don't apply exclusions to root directories themselves
  cmd+=( -mindepth 1 "(" -type d "(" )
  local first=1
  for d in "${EXCLUDE_DIRS[@]}"; do
    if (( first )); then
      cmd+=( -name "$d" )
      first=0
    else
      cmd+=( -o -name "$d" )
    fi
  done
  cmd+=( ")" -prune ")" -o "(" -type f -name "test_*.py" -print0 ")" )

  printf '%q ' "${cmd[@]}"
}

# Collect pytest node ids for multiple targets at once (batch collection)
# This is much faster than calling pytest --collect-only per file, as pytest
# initialization (~10s) happens only once instead of per file.
collect_nodes_batch() {
  local marker_arg="$1"  # optional marker filter (e.g., "-m eval")
  shift
  local targets=("$@")

  if (( ${#targets[@]} == 0 )); then
    return 0
  fi

  local cmd
  # Collection only needs test ids, not an activated project: SKIP_UNIFY_TEST_INIT
  # keeps the harness from opening a store and seeding the builtin catalogues.
  local env_exports='export SKIP_UNIFY_TEST_INIT=1'
  # Append user-provided --env overrides
  local user_overrides
  user_overrides="$(build_env_exports)"
  if [[ -n "$user_overrides" ]]; then
    env_exports="$env_exports$user_overrides"
  fi

  # Build target list with proper quoting
  local quoted_targets=""
  for t in "${targets[@]}"; do
    quoted_targets+=" $(printf '%q' "$t")"
  done

  # Build collection filter args (marker filter + any -k/-m from PYTEST_EXTRA_ARGS)
  local collection_filters=""
  if [[ -n "$marker_arg" ]]; then
    collection_filters="$marker_arg"
  fi
  # Append collection-relevant args from PYTEST_EXTRA_ARGS (e.g., -k "pattern")
  if (( ${#PYTEST_COLLECTION_ARGS[@]} > 0 )); then
    for carg in "${PYTEST_COLLECTION_ARGS[@]}"; do
      collection_filters+=" $(printf '%q' "$carg")"
    done
  fi

  # Build collection command with filters
  if [[ -n "$collection_filters" ]]; then
    cmd=$(printf '%s; cd %q && %q -m pytest --collect-only -q %s %s' "$env_exports" "$REPO_ROOT" "$VENV_PY" "$collection_filters" "$quoted_targets")
  else
    cmd=$(printf '%s; cd %q && %q -m pytest --collect-only -q %s' "$env_exports" "$REPO_ROOT" "$VENV_PY" "$quoted_targets")
  fi
  # Remove color codes, keep only node ids (contain ::), ignore noise; never fail the script
  # Redirect stdin from /dev/null to prevent hangs when multiple processes compete for stdin
  # Note: stderr is captured to a temp file so collection errors can be surfaced if no tests found
  local stderr_file
  stderr_file=$(mktemp)
  local result
  result=$(bash -lc "$cmd" < /dev/null 2>"$stderr_file" | sed -E 's/\x1B\[[0-9;]*[mK]//g' | grep -E '::' || true)

  # If no tests collected and there was stderr output, show it for debugging
  if [[ -z "$result" && -s "$stderr_file" ]]; then
    echo "Warning: pytest collection produced no test nodes. stderr output:" >&2
    head -50 "$stderr_file" >&2
  fi
  rm -f "$stderr_file"

  echo "$result"
}

# Validate direct_nodes against pytest collection and add valid ones to $tmp.
# Extracts unique base files, runs collection, and checks each node exists.
# Prints warnings for nodes that don't exist (e.g., typos in test function names).
validate_and_add_direct_nodes() {
  if (( ${#direct_nodes[@]} == 0 )); then
    return 0
  fi

  # Extract unique base files from direct_nodes
  local -a base_files=()
  local seen_files=""
  for node in "${direct_nodes[@]}"; do
    local base_file="${node%%::*}"
    if [[ "$seen_files" != *"|${base_file}|"* ]]; then
      base_files+=( "$base_file" )
      seen_files="${seen_files}|${base_file}|"
    fi
  done

  # Collect all valid nodes from those files (no marker filter — just checking existence)
  local collected
  collected=$(collect_nodes_batch "" "${base_files[@]}")

  # Validate each direct_node against collected output
  for node in "${direct_nodes[@]}"; do
    if echo "$collected" | grep -qxF "$node"; then
      printf '%s\0' "$node" >> "$tmp"
    else
      echo "Error: Test node not found (skipping): $node" >&2
    fi
  done
}

# Gather recursive .py files from roots (NUL-delimited, sorted)
declare -a found_files=()
if (( ${#roots[@]} )); then
  found_files=()
  while IFS= read -r -d '' f; do
    found_files+=( "$f" )
  done < <(eval "$(build_find_cmd)")
fi

# Apply filename pattern filter (matches on basename) if provided
if [[ -n "$NAME_PATTERN" ]]; then
  if (( ${#direct_files[@]} )); then
    tmp_direct=()
    for f in "${direct_files[@]}"; do
      b="${f##*/}"
      if [[ "$b" == $NAME_PATTERN ]]; then
        tmp_direct+=( "$f" )
      fi
    done
    direct_files=( "${tmp_direct[@]}" )
  fi
  if (( ${#found_files[@]} )); then
    tmp_found=()
    for f in "${found_files[@]}"; do
      b="${f##*/}"
      if [[ "$b" == $NAME_PATTERN ]]; then
        tmp_found+=( "$f" )
      fi
    done
    found_files=( "${tmp_found[@]}" )
  fi
fi

# Combine targets based on mode; sort deterministically (and de-duplicate)
tmp="$(mktemp)"
START_TIMES_FILE="$(mktemp)"
RESULTS_FILE="$(mktemp)"
trap 'rm -f "$tmp" "$START_TIMES_FILE" "$RESULTS_FILE"' EXIT
if (( ! SERIAL )); then
  # Default mode (per-test): expand directories/files into node ids using batch collection
  # Combine all targets for a single pytest --collect-only call (much faster)
  all_targets=()
  if (( ${#direct_files[@]} )); then
    all_targets+=( "${direct_files[@]}" )
  fi
  if (( ${#found_files[@]} )); then
    all_targets+=( "${found_files[@]}" )
  fi
  if (( ${#all_targets[@]} )); then
    while IFS= read -r nid; do
      [[ -n "$nid" ]] && printf '%s\0' "$nid" >> "$tmp"
    done < <(collect_nodes_batch "$MARKER_FILTER" "${all_targets[@]}")
  fi
  validate_and_add_direct_nodes
elif [[ -n "$MARKER_FILTER" ]]; then
  # Serial mode WITH marker filter: collect nodes first to find which files
  # have matching tests, then create one session per file (not per-node).
  # This prevents creating sessions for files with 0 matching tests.
  all_targets=()
  if (( ${#direct_files[@]} )); then
    all_targets+=( "${direct_files[@]}" )
  fi
  if (( ${#found_files[@]} )); then
    all_targets+=( "${found_files[@]}" )
  fi
  if (( ${#all_targets[@]} )); then
    # Collect node ids, extract unique file paths (bash 3.x compatible)
    # Use a temp file + sort -u to get unique file paths
    tmp_files="$(mktemp)"
    while IFS= read -r nid; do
      if [[ -n "$nid" && "$nid" == *"::"* ]]; then
        # Extract file path (everything before first ::)
        file_path="${nid%%::*}"
        echo "$file_path" >> "$tmp_files"
      fi
    done < <(collect_nodes_batch "$MARKER_FILTER" "${all_targets[@]}")
    # Output unique files that have matching tests
    while IFS= read -r file_path; do
      [[ -n "$file_path" ]] && printf '%s\0' "$file_path" >> "$tmp"
    done < <(sort -u "$tmp_files")
    rm -f "$tmp_files"
  fi
  validate_and_add_direct_nodes
else
  # Serial mode without marker filter: one session per file
  if (( ${#direct_files[@]} )); then
    printf '%s\0' "${direct_files[@]}" >> "$tmp"
  fi
  if (( ${#found_files[@]} )); then
    printf '%s\0' "${found_files[@]}" >> "$tmp"
  fi
  validate_and_add_direct_nodes
fi

files=()
while IFS= read -r -d '' f; do
  files+=( "$f" )
done < <(tr '\0' '\n' < "$tmp" | LC_ALL=C sort -u | tr '\n' '\0')

if (( ${#files[@]} == 0 )); then
  echo "Error: No tests found for the given path(s)." >&2
  echo "This could mean:" >&2
  echo "  - The path doesn't exist or contains no test_*.py files" >&2
  echo "  - pytest --collect-only failed (check for import errors)" >&2
  echo "  - A marker filter (--eval-only/--symbolic-only) excluded all tests" >&2
  exit 1
fi

# Expand targets for repeat runs (statistical sampling)
if (( REPEAT_COUNT > 1 )); then
  original_files=( "${files[@]}" )
  files=()
  for (( r=1; r<=REPEAT_COUNT; r++ )); do
    for f in "${original_files[@]}"; do
      files+=( "$f" )
    done
  done
  echo "Repeating each test $REPEAT_COUNT times (${#files[@]} total sessions from ${#original_files[@]} unique targets)"
fi

declare -a made_sessions=()
declare -a session_ids=()

# Helper function to print log directory info (used at start and end)
print_log_directories() {
  echo "========================================================================"
  echo "📁 pytest logs:  logs/pytest/$LOG_SUBDIR/"
  if (( RUNNER_OWNS_STORES )); then
    echo "🗄️  stores:       logs/pytest/$LOG_SUBDIR/stores/ (one SQLite file per session)"
  else
    echo "🗄️  store:        $(resolve_store_path "") (UNIFY_STORE_PATH, shared by every session)"
  fi
  echo "🔗 OTel traces:  logs/{unify|unillm}/ (per-repo), logs/all/ (cross-repo)"
  echo "📖 Logging docs: logs/README.md"
  echo "========================================================================"
}

# Print log directory info first (before session creation starts)
echo
print_log_directories
echo

if (( MAX_JOBS > 0 )); then
  echo "Concurrency limit: $MAX_JOBS simultaneous sessions"
else
  echo "Concurrency limit: unlimited"
fi

# Sessions open their stores as soon as pytest starts, so the directory has
# to exist before the first session is created.
if (( RUNNER_OWNS_STORES )); then
  mkdir -p "$STORES_DIR"
fi

# Keep occurrence identities and terminal receipts after sessions and the
# runner exit. Each invocation gets a private fresh directory, including
# repeated runs using the same socket or log subdirectory.
mkdir -p "$LOG_DIR"
RESULTS_DIR=$(mktemp -d "$LOG_DIR/session-results.XXXXXX")

# Print header before drip-feeding session creation
echo "Creating ${#files[@]} tmux sessions..."

for target in "${files[@]}"; do
  # Report any completions before creating new sessions
  # (ensures passes/fails appear BEFORE the new pending sessions they freed slots for)
  report_completed_sessions

  # If job limit is set, wait for a slot before creating new session
  wait_for_job_slot

  base_sess="$(session_basename_for "$target")"
  session="$(unique_session_name "$base_sess")"

  # Window name = basename without .py
  fname="${target##*/}"
  wname="${fname%.py}"

  # Create the session first (no command), set remain-on-exit, then send the command.
  # Note: Log directory is created lazily by conftest.py only when a log file is
  # actually written, avoiding empty directories when sessions fail/are interrupted.
  # Note: Log paths are auto-derived by conftest.py (semantic name + timestamp in socket subdir)
  occurrence=${#CREATED_SESSION_IDS[@]}
  result_path="$RESULTS_DIR/$occurrence.exit"
  original_session_name="$session"
  printf '%s\n' "$target" > "$RESULTS_DIR/$occurrence.target"
  printf '%s\n' "$session" > "$RESULTS_DIR/$occurrence.name"
  cmd="$(run_cmd "$target" "$MARKER_FILTER" "$session" "$result_path")"

  # Capture session ID to track this specific run robustly
  sid=$(tmux_cmd new-session -d -P -F "#{session_id}" -s "$session" -n "$wname" "$cmd")

  # Record start time for duration tracking
  echo "$sid $(date +%s)" >> "$START_TIMES_FILE"

  # Rename to pending state with retry logic for race conditions
  pending_name="$(unique_session_name "r ⏳ $session")"
  session="$(rename_session_with_retry "$sid" "$pending_name")"

  # Print session as it's created (drip-feed)
  echo "  - $session"

  made_sessions+=( "$session" )
  session_ids+=( "$sid" )
  # Track for cleanup on interrupt (SIGINT/SIGTERM)
  CREATED_SESSION_IDS+=( "$sid" )
  CREATED_RESULT_PATHS+=( "$result_path" )
  CREATED_SESSION_NAMES+=( "$original_session_name" )
  printf '%s\n' "$sid" > "$RESULTS_DIR/$occurrence.session"
done

# ---- Wait for all tests to complete ----
# Always block until completion (or timeout). Continue showing drip-feed of
# pass/fail results as tests complete.

# Count how many have already completed during session creation
completed_count=0
pending_count=0
for sid in "${session_ids[@]}"; do
  current_name=$(tmux_cmd display-message -p -t "$sid" "#{session_name}" 2>/dev/null || echo "")
  if [[ "$current_name" == "r"* ]]; then
    ((pending_count++)) || true
  else
    ((completed_count++)) || true
  fi
done

# Print summary with completion status
if (( pending_count == 0 )); then
  echo "Created all ${#made_sessions[@]} tmux sessions. All completed!"
elif (( TIMEOUT > 0 )); then
  echo "Created all ${#made_sessions[@]} tmux sessions. $completed_count completed. Waiting for remaining $pending_count to complete (timeout: ${TIMEOUT}s)..."
else
  echo "Created all ${#made_sessions[@]} tmux sessions. $completed_count completed. Waiting for remaining $pending_count to complete..."
fi
if (( SESSION_TIMEOUT > 0 )); then
  echo "Per-session hang guard: pytest killed after ${SESSION_TIMEOUT}s if still running."
fi

wait_start=$(date +%s)
timed_out=0
while true; do
  # Report any newly completed sessions (drip-feed pass/fail inline)
  report_completed_sessions

  # Count remaining pending sessions
  pending_count=0
  for sid in "${session_ids[@]}"; do
    # Check name of our specific session IDs only
    current_name=$(tmux_cmd display-message -p -t "$sid" "#{session_name}" 2>/dev/null || echo "")
    # Look for "r" prefix to detect pending state (r ⏳)
    if [[ "$current_name" == "r"* ]]; then
      ((pending_count++)) || true
    fi
  done

  if (( pending_count == 0 )); then
    # Final report to catch any last completions
    report_completed_sessions
    break
  fi

  # Check timeout if specified
  if (( TIMEOUT > 0 )); then
    elapsed=$(( $(date +%s) - wait_start ))
    if (( elapsed >= TIMEOUT )); then
      timed_out=1
      echo ""
      echo "Timeout reached after ${TIMEOUT}s. ${pending_count} session(s) still running."
      break
    fi
  fi

  sleep 1
done

if (( timed_out )); then
  echo "Tests did not complete within timeout. Check tmux sessions manually."
  exit 2
fi

echo ""
incomplete_count=0
for sid in "${session_ids[@]}"; do
  if ! _is_reported "$sid"; then
    ((incomplete_count++)) || true
    echo "Missing terminal receipt for owned session $sid; outcome unknown." >&2
  fi
done
if (( incomplete_count > 0 )); then
  echo "Incomplete test accounting: $incomplete_count missing outcomes." >&2
else
  echo "All tests completed."
fi

# Collect failures
declare -a failed_sessions=()
for sid in "${session_ids[@]}"; do
  current_name=$(tmux_cmd display-message -p -t "$sid" "#{session_name}" 2>/dev/null || echo "")
  # Look for "f" prefix to detect failure (f ❌)
  if [[ "$current_name" == "f"* ]]; then
    failed_sessions+=( "$current_name" )
  fi
done

# Print test stats (duration, cache, and cost) sorted fastest to slowest
echo ""
echo "========================================================================"
echo "TEST STATS: DURATION, CACHE & COST (fastest → slowest)"
echo "========================================================================"

# Count passed, skipped and failed (use { grep || true; } to handle no-match case with pipefail)
pass_count=$( { grep '|pass|' "$RESULTS_FILE" || true; } 2>/dev/null | wc -l | tr -d ' ')
skip_count=$( { grep '|skip|' "$RESULTS_FILE" || true; } 2>/dev/null | wc -l | tr -d ' ')
fail_count=$( { grep '|fail|' "$RESULTS_FILE" || true; } 2>/dev/null | wc -l | tr -d ' ')

# Build duration summary for both stdout and file output
DURATION_SUMMARY_FILE="$LOG_DIR/duration_summary.txt"
mkdir -p "$(dirname "$DURATION_SUMMARY_FILE")"

# Helper to print to both stdout and file
print_duration_line() {
  echo "$1"
  echo "$1" >> "$DURATION_SUMMARY_FILE"
}

# Helper to format cache hit rate as percentage string
format_cache_rate() {
  local hits="$1"
  local misses="$2"
  local total=$((hits + misses))
  if (( total == 0 )); then
    echo "  --%"
  else
    local pct=$((hits * 100 / total))
    printf "%3d%%" "$pct"
  fi
}

# Clear/create the summary file
> "$DURATION_SUMMARY_FILE"

# Print passed tests sorted by duration (fastest first, slowest last)
# Format: duration|status|hits|canonical|misses|cost|name
if (( pass_count > 0 )); then
  echo ""
  print_duration_line "✅ PASSED ($pass_count tests):"
  print_duration_line "$(printf "  %6s  %6s  %10s  %s" "time" "cache" "cost" "test")"
  print_duration_line "$(printf "  %6s  %6s  %10s  %s" "----" "-----" "----" "----")"
  { grep '|pass|' "$RESULTS_FILE" || true; } | sort -t'|' -k1 -n | while IFS='|' read -r dur status hits canonical misses cost name; do
    cache_rate=$(format_cache_rate "$hits" "$misses")
    formatted_cost=$(printf '$%.6f' "$cost")
    print_duration_line "$(printf "  %5ds  %6s  %10s  %s" "$dur" "$cache_rate" "$formatted_cost" "$name")"
  done
fi

# Print skipped sessions — every test in these ran nothing, so the coverage
# they normally provide is absent from this run.
if (( skip_count > 0 )); then
  print_duration_line ""
  print_duration_line "⏭️  SKIPPED ($skip_count tests — no coverage from these):"
  print_duration_line "$(printf "  %6s  %6s  %10s  %s" "time" "cache" "cost" "test")"
  print_duration_line "$(printf "  %6s  %6s  %10s  %s" "----" "-----" "----" "----")"
  { grep '|skip|' "$RESULTS_FILE" || true; } | sort -t'|' -k1 -n | while IFS='|' read -r dur status hits canonical misses cost name; do
    cache_rate=$(format_cache_rate "$hits" "$misses")
    formatted_cost=$(printf '$%.6f' "$cost")
    print_duration_line "$(printf "  %5ds  %6s  %10s  %s" "$dur" "$cache_rate" "$formatted_cost" "$name")"
  done
fi

# Print failed tests sorted by duration (fastest first, slowest last)
if (( fail_count > 0 )); then
  print_duration_line ""
  print_duration_line "❌ FAILED ($fail_count tests):"
  print_duration_line "$(printf "  %6s  %6s  %10s  %s" "time" "cache" "cost" "test")"
  print_duration_line "$(printf "  %6s  %6s  %10s  %s" "----" "-----" "----" "----")"
  { grep '|fail|' "$RESULTS_FILE" || true; } | sort -t'|' -k1 -n | while IFS='|' read -r dur status hits canonical misses cost name; do
    cache_rate=$(format_cache_rate "$hits" "$misses")
    formatted_cost=$(printf '$%.6f' "$cost")
    print_duration_line "$(printf "  %5ds  %6s  %10s  %s" "$dur" "$cache_rate" "$formatted_cost" "$name")"
  done
fi

# Calculate and print aggregated totals
total_duration=0
total_hits=0
total_canonical=0
total_misses=0
total_cost="0"
while IFS='|' read -r dur status hits canonical misses cost name; do
  total_duration=$((total_duration + dur))
  total_hits=$((total_hits + hits))
  total_canonical=$((total_canonical + canonical))
  total_misses=$((total_misses + misses))
  total_cost=$(awk "BEGIN {printf \"%.6g\", $total_cost + $cost}")
done < "$RESULTS_FILE"

total_tests=$((pass_count + skip_count + fail_count))
if (( total_tests > 0 )); then
  print_duration_line ""
  print_duration_line "========================================================================"
  print_duration_line "TOTALS ($total_tests tests)"
  print_duration_line "========================================================================"
  # Format duration as minutes:seconds if > 60s
  if (( total_duration >= 60 )); then
    mins=$((total_duration / 60))
    secs=$((total_duration % 60))
    duration_str="${mins}m ${secs}s"
  else
    duration_str="${total_duration}s"
  fi
  total_cache_rate=$(format_cache_rate "$total_hits" "$total_misses")
  total_calls=$((total_hits + total_misses))
  print_duration_line "  Serial duration: $duration_str"
  # Canonical hits are recordings accepted through canonical keying rather
  # than byte-identical raw keys — the share of prompt drift being absorbed.
  print_duration_line "  LLM cache: $total_cache_rate ($total_hits hits [$total_canonical canonical], $total_misses misses, $total_calls total)"
  print_duration_line "  LLM cost: \$$total_cost"
fi

echo ""
print_log_directories

if (( incomplete_count > 0 || total_tests != ${#session_ids[@]} )); then
  echo "Test accounting is incomplete; preserved receipts: $RESULTS_DIR" >&2
  exit 2
elif (( fail_count > 0 || ${#failed_sessions[@]} > 0 )); then
  exit 1
else
  echo ""
  echo "All tests passed!"
  exit 0
fi

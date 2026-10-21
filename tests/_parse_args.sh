#!/usr/bin/env bash
# Argument parsing for parallel_run.sh.
#
# Usage:
#   SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
#   source "$SCRIPT_DIR/_parse_args.sh"
#   parse_test_args "$@"
#
# After calling parse_test_args, these variables are populated:
#   SERIAL, TIMEOUT, SESSION_TIMEOUT, NAME_PATTERN, EVAL_ONLY, SYMBOLIC_ONLY,
#   DETERMINISTIC_ONLY, KEYLESS, LIVE_ONLY, REPEAT_COUNT, MAX_JOBS,
#   ENV_OVERRIDES[], PYTEST_EXTRA_ARGS[], PYTEST_COLLECTION_ARGS[],
#   POSITIONAL_ARGS[]
#
# Additional functions:
#   print_help   - Prints usage (caller can override HELP_SCRIPT_NAME)

# ---- Detect CPU cores for default MAX_JOBS ----
if [[ "$(uname)" == "Darwin" ]]; then
  _PARSE_ARGS_NUM_CORES=$(sysctl -n hw.ncpu 2>/dev/null || echo 4)
else
  _PARSE_ARGS_NUM_CORES=$(nproc 2>/dev/null || echo 4)
fi

# Export for use in help text
DETECTED_CPU_CORES=$_PARSE_ARGS_NUM_CORES

# ---- Main argument parsing function ----
# Parses command line arguments and populates global variables.
# Returns 0 on success, 1 if help was requested, 2 on error.
parse_test_args() {
  # Initialize/reset all variables
  SERIAL=0
  TIMEOUT=0
  SESSION_TIMEOUT=0
  NAME_PATTERN=""
  EVAL_ONLY=0
  SYMBOLIC_ONLY=0
  DETERMINISTIC_ONLY=0
  KEYLESS=0
  LIVE_ONLY=0
  REPEAT_COUNT=1
  MAX_JOBS=$_PARSE_ARGS_NUM_CORES
  ENV_OVERRIDES=()
  PYTEST_EXTRA_ARGS=()
  PYTEST_COLLECTION_ARGS=()
  POSITIONAL_ARGS=()

  while (( "$#" )); do
    case "$1" in
      -t|--timeout)
        if [[ -n "${2-}" && "$2" =~ ^[0-9]+$ && "$2" -ge 1 ]]; then
          TIMEOUT="$2"
          shift 2
        else
          echo "Error: --timeout requires a positive integer (seconds)." >&2
          return 2
        fi
        ;;
      --session-timeout)
        if [[ -n "${2-}" && "$2" =~ ^[0-9]+$ && "$2" -ge 1 ]]; then
          SESSION_TIMEOUT="$2"
          shift 2
        else
          echo "Error: --session-timeout requires a positive integer (seconds)." >&2
          return 2
        fi
        ;;
      -s|--serial)
        SERIAL=1
        shift
        ;;
      -m|--match)
        if [[ -n "${2-}" ]]; then
          NAME_PATTERN="$2"
          shift 2
        else
          echo "Error: -m|--match requires a pattern argument (e.g., \"*_tool_docstring*\")." >&2
          return 2
        fi
        ;;
      -e|--env)
        if [[ -n "${2-}" && "$2" == *=* ]]; then
          ENV_OVERRIDES+=( "$2" )
          shift 2
        else
          echo "Error: -e|--env requires KEY=VALUE argument (e.g., --env UNILLM_CACHE=false)." >&2
          return 2
        fi
        ;;
      --no-cache)
        # Shorthand for --env UNILLM_CACHE=false: every LLM call goes to the
        # provider, so use it to re-evaluate model behaviour, not to "fix" a
        # failing cached test.
        ENV_OVERRIDES+=( "UNILLM_CACHE=false" )
        shift
        ;;
      --eval-only)
        EVAL_ONLY=1
        shift
        ;;
      --deterministic-only)
        DETERMINISTIC_ONLY=1
        shift
        ;;
      --symbolic-only)
        SYMBOLIC_ONLY=1
        shift
        ;;
      --keyless)
        # Every run is keyless: no test reaches a model provider or the
        # network (tests/conftest.py). Kept so that older commands parse.
        shift
        ;;
      --live-only)
        echo "Error: --live-only is gone: no test reaches a model provider; tests script the model (tests/scripted_model.py)." >&2
        exit 2
        ;;
      --repeat)
        if [[ -n "${2-}" && "$2" =~ ^[0-9]+$ && "$2" -ge 1 ]]; then
          REPEAT_COUNT="$2"
          shift 2
        else
          echo "Error: --repeat requires a positive integer argument (e.g., --repeat 5)." >&2
          return 2
        fi
        ;;
      -j|--jobs)
        if [[ -z "${2-}" ]]; then
          echo "Error: -j|--jobs requires an argument (e.g., --jobs 8, --jobs 0, --jobs none)." >&2
          return 2
        fi
        # Accept positive integers, 0, or keywords for unlimited
        local arg_lower
        arg_lower=$(echo "$2" | tr '[:upper:]' '[:lower:]')
        if [[ "$2" =~ ^[0-9]+$ ]]; then
          MAX_JOBS="$2"
        elif [[ "$arg_lower" == "none" || "$arg_lower" == "unlimited" || "$arg_lower" == "inf" ]]; then
          MAX_JOBS=0
        else
          echo "Error: -j|--jobs requires a non-negative integer or 'none'/'unlimited' (e.g., --jobs 8, --jobs 0, --jobs none)." >&2
          return 2
        fi
        shift 2
        ;;
      -h|--help)
        # Return 1 to signal help was requested; caller should print help and exit
        return 1
        ;;
      --)
        shift
        PYTEST_EXTRA_ARGS=("$@")
        # Extract collection-relevant args (-k, -m) for use during test discovery
        # These filters affect which tests are collected, not just how they run
        local _coll_i=0
        while (( _coll_i < ${#PYTEST_EXTRA_ARGS[@]} )); do
          local _coll_arg="${PYTEST_EXTRA_ARGS[_coll_i]}"
          case "$_coll_arg" in
            -k|-m)
              # Next arg is the value (e.g., -k "pattern")
              if (( _coll_i + 1 < ${#PYTEST_EXTRA_ARGS[@]} )); then
                PYTEST_COLLECTION_ARGS+=( "$_coll_arg" "${PYTEST_EXTRA_ARGS[_coll_i+1]}" )
                ((_coll_i+=2))
              else
                ((_coll_i++))
              fi
              ;;
            -k=*|-m=*)
              # Value is attached (e.g., -k="pattern")
              PYTEST_COLLECTION_ARGS+=( "$_coll_arg" )
              ((_coll_i++))
              ;;
            --keyword=*|--markers=*)
              # Long form with attached value
              PYTEST_COLLECTION_ARGS+=( "$_coll_arg" )
              ((_coll_i++))
              ;;
            *)
              ((_coll_i++))
              ;;
          esac
        done
        break
        ;;
      -*)
        echo "Error: Unknown option: $1" >&2
        echo "To pass pytest options (like -k, -v, -x), use -- before them:" >&2
        echo "  Example: parallel_run.sh tests/foo.py -- -k 'pattern'" >&2
        echo "Run with -h for all options." >&2
        return 2
        ;;
      *)
        POSITIONAL_ARGS+=( "$1" )
        shift
        ;;
    esac
  done

  # Validate mutually exclusive flags
  if (( EVAL_ONLY + SYMBOLIC_ONLY + DETERMINISTIC_ONLY > 1 )); then
    echo "Error: --eval-only, --symbolic-only and --deterministic-only are mutually exclusive." >&2
    return 2
  fi
  if (( KEYLESS + LIVE_ONLY > 1 )); then
    echo "Error: --keyless and --live-only are mutually exclusive." >&2
    return 2
  fi

  return 0
}

# ---- Print help text ----
# Caller can set HELP_SCRIPT_NAME before calling to customize the script name shown.
# Caller can set HELP_EXTRA_OPTIONS to add script-specific options to the help.
print_help() {
  local script_name="${HELP_SCRIPT_NAME:-parallel_run.sh}"
  cat << EOF
Usage: $script_name [options] [targets...]

Run pytest tests in parallel tmux sessions. Always blocks until all tests
complete (or timeout). Each session gets its own Unify home (tests/conftest.py).

Options:
  -t, --timeout N      Abort the whole run if any sessions remain after N seconds
  --session-timeout N  Kill each tmux session's pytest if it exceeds N seconds
                       (UNIFY_TEST_SESSION_TIMEOUT supplies the default)
  -s, --serial         One session per file (default: one per test)
  -m, --match PATTERN  Filter files by glob pattern
  -e, --env KEY=VALUE  Set environment variable (repeatable)
  --no-cache           Call the LLM provider for every request (--env UNILLM_CACHE=false)
  -j, --jobs N         Max concurrent sessions (default: CPU cores, currently $DETECTED_CPU_CORES)
  --eval-only          Run only @pytest.mark.eval tests
  --symbolic-only      Run only non-eval tests
  --deterministic-only Run only tests with no model in the loop
  --keyless            Accepted and ignored: every run refuses model-provider and
                       network calls (tests/conftest.py)
  --repeat N           Run each test N times
  -h, --help           Show this help
  --                   Pass remaining args directly to pytest
${HELP_EXTRA_OPTIONS:-}
Examples:
  $script_name tests/                    # Run all tests
  $script_name tests/foo.py             # Run one file
  $script_name --timeout 300 tests/     # 5-minute whole-run timeout
  $script_name --session-timeout 600 tests/  # 10-minute per-session kill
  $script_name -s tests/                # Serial mode (per-file)
  $script_name -j 8 tests/              # Limit to 8 concurrent
  $script_name --eval-only tests/       # Only eval tests
  $script_name --no-cache tests/foo.py  # Fresh LLM inference
  $script_name --keyless tests/async_tool_loop    # Without a provider key
  $script_name --live-only tests/async_tool_loop  # The key-needing tests only
  $script_name tests/ -- -v --tb=short  # Pass args to pytest
  $script_name tests/ -- -k 'gpt-5'     # Filter by test name pattern
EOF
}

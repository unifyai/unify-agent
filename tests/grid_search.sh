#!/usr/bin/env bash
set -euo pipefail

# Grid Search Runner
# ==================
# Runs tests across all combinations of settings values.
#
# Usage:
#   ./grid_search.sh --env KEY=val1|val2 --env KEY2=a|b|c [options] [targets...]
#
# The pipe character "|" separates multiple values for a setting.
# A full Cartesian product (grid) of all combinations is generated.
#
# Note: Each combination runs sequentially (parallel_run.sh always blocks until
#       completion). For truly parallel grid runs, launch in separate terminals.
#
# Example:
#   ./grid_search.sh --env UNIFY_MODEL="gpt-4o|claude-3" --env UNILLM_CACHE="true|false" tests/
#
# Generates 4 runs:
#   1. UNIFY_MODEL=gpt-4o   UNILLM_CACHE=true
#   2. UNIFY_MODEL=gpt-4o   UNILLM_CACHE=false
#   3. UNIFY_MODEL=claude-3 UNILLM_CACHE=true
#   4. UNIFY_MODEL=claude-3 UNILLM_CACHE=false

# Resolve script directory and repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd -P)"
PARALLEL_RUN="$SCRIPT_DIR/parallel_run.sh"

# Check that parallel_run.sh exists
if [[ ! -x "$PARALLEL_RUN" ]]; then
  echo "Error: parallel_run.sh not found or not executable at $PARALLEL_RUN" >&2
  exit 1
fi

# Arrays to collect grid variables and pass-through options
declare -a GRID_VARS=()      # KEY=val1|val2|val3 entries (grid search)
declare -a SINGLE_ENV_VARS=()  # KEY=value entries (no pipe, held constant across runs)
declare -a PASSTHROUGH=()    # All other args (non-env flags like -j)
declare -a TARGETS=()        # Test targets (files/directories)

DRY_RUN=0

usage() {
  cat <<'USAGE'
Grid Search Runner
==================

Run tests across all combinations of settings values.
Each combination runs sequentially (parallel_run.sh always blocks until completion).

Usage:
  ./grid_search.sh [options] --env KEY=val1|val2 [--env KEY2=a|b] [targets...]

Options:
  --env KEY=val1|val2   Specify multiple values for a setting (pipe-separated)
  --env KEY=value       Single value (included in all runs and tags)
  -n, --dry-run         Show generated commands without executing
  -h, --help            Show this help

All other options and arguments are passed through to parallel_run.sh.

Auto-Tagging:
  Each run is automatically tagged with all --env values passed to this script,
  formatted as "KEY1=val1,KEY2=val2,...". This makes post-hoc filtering easy.
  Only explicitly passed --env values are tagged; values from .env files or
  other sources appear in the full settings dict but not in tags.

Examples:
  # Grid search across memory modes and step limits
  # (auto-tags: UNIFY_MEMORY=off,UNIFY_MAX_TOOL_LOOP_STEPS=50 etc.)
  ./grid_search.sh --env UNIFY_MEMORY="off|on" --env UNIFY_MAX_TOOL_LOOP_STEPS="50|300" tests/

  # With additional pass-through options
  ./grid_search.sh --env UNIFY_MEMORY="off|on" -j 4 tests/

  # Dry run to see what would be executed
  ./grid_search.sh -n --env UNIFY_MEMORY="off|on" tests/

  # Add a constant variable to all runs
  ./grid_search.sh --env UNIFY_MEMORY="off|on" --env EXPERIMENT_ID="exp-42" tests/

Notes:
  - Each combination spawns a separate parallel_run.sh invocation
  - Combinations run sequentially (parallel_run.sh blocks until tests complete)
  - Each run writes its own log directory under logs/pytest/
USAGE
}

# Parse arguments
while (( "$#" )); do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    -n|--dry-run)
      DRY_RUN=1
      shift
      ;;
    -e|--env)
      if [[ -n "${2:-}" ]]; then
        kv="$2"
        if [[ "$kv" == *"|"* ]]; then
          # Contains pipe - this is a grid variable
          GRID_VARS+=( "$kv" )
        else
          # No pipe - single value, held constant across all runs
          SINGLE_ENV_VARS+=( "$kv" )
        fi
        shift 2
      else
        echo "Error: --env requires KEY=VALUE argument" >&2
        exit 2
      fi
      ;;
    -*)
      # Other flags - pass through
      PASSTHROUGH+=( "$1" )
      shift
      ;;
    *)
      # Positional args (targets)
      TARGETS+=( "$1" )
      shift
      ;;
  esac
done

# If no grid variables, just run parallel_run.sh directly
if (( ${#GRID_VARS[@]} == 0 )); then
  echo "No grid variables specified (use --env KEY=val1|val2). Running single invocation..."
  cmd=( "$PARALLEL_RUN" )
  for kv in "${SINGLE_ENV_VARS[@]+"${SINGLE_ENV_VARS[@]}"}"; do
    cmd+=( "--env" "$kv" )
  done
  cmd+=( "${PASSTHROUGH[@]+"${PASSTHROUGH[@]}"}" )
  cmd+=( "${TARGETS[@]+"${TARGETS[@]}"}" )
  exec "${cmd[@]}"
fi

# Parse grid variables into arrays
# GRID_KEYS[i] = key name
# GRID_VALUES[i] = "val1 val2 val3" (space-separated)
declare -a GRID_KEYS=()
declare -a GRID_VALUES=()

for entry in "${GRID_VARS[@]}"; do
  key="${entry%%=*}"
  values_str="${entry#*=}"
  # Replace | with space for iteration
  values_space="${values_str//|/ }"
  GRID_KEYS+=( "$key" )
  GRID_VALUES+=( "$values_space" )
done

# Generate all combinations using recursive expansion
# We'll build an array of "KEY1=val KEY2=val ..." strings
generate_combinations() {
  local depth="$1"
  local prefix="$2"

  if (( depth >= ${#GRID_KEYS[@]} )); then
    # Base case: output the combination
    echo "$prefix"
    return
  fi

  local key="${GRID_KEYS[$depth]}"
  local values="${GRID_VALUES[$depth]}"

  for val in $values; do
    local new_prefix
    if [[ -z "$prefix" ]]; then
      new_prefix="$key=$val"
    else
      new_prefix="$prefix $key=$val"
    fi
    generate_combinations $((depth + 1)) "$new_prefix"
  done
}

# Collect all combinations (bash 3.x compatible - no mapfile)
COMBINATIONS=()
while IFS= read -r combo; do
  COMBINATIONS+=( "$combo" )
done < <(generate_combinations 0 "")

echo "Grid Search Configuration"
echo "========================="
echo "Grid variables:"
for i in "${!GRID_KEYS[@]}"; do
  echo "  ${GRID_KEYS[$i]}: ${GRID_VALUES[$i]// / | }"
done
if (( ${#SINGLE_ENV_VARS[@]} > 0 )); then
  echo ""
  echo "Constant variables (included in all runs):"
  for kv in "${SINGLE_ENV_VARS[@]}"; do
    echo "  $kv"
  done
fi
echo ""
echo "Total combinations: ${#COMBINATIONS[@]}"
echo ""

# Show what we'll run
echo "Generated runs:"
for i in "${!COMBINATIONS[@]}"; do
  combo="${COMBINATIONS[$i]}"
  echo "  [$((i+1))/${#COMBINATIONS[@]}] $combo"
done
echo ""

if (( DRY_RUN )); then
  echo "Dry run - commands that would be executed:"
  echo ""
  for combo in "${COMBINATIONS[@]}"; do
    cmd=( "$PARALLEL_RUN" )
    for kv in $combo; do
      cmd+=( "--env" "$kv" )
    done
    for kv in "${SINGLE_ENV_VARS[@]+"${SINGLE_ENV_VARS[@]}"}"; do
      cmd+=( "--env" "$kv" )
    done
    cmd+=( "${PASSTHROUGH[@]+"${PASSTHROUGH[@]}"}" )
    cmd+=( "${TARGETS[@]+"${TARGETS[@]}"}" )
    echo "  ${cmd[*]}"
  done
  exit 0
fi

# Execute all combinations sequentially
# (parallel_run.sh always blocks until tests complete)
echo "Running ${#COMBINATIONS[@]} combinations sequentially..."
echo ""

for i in "${!COMBINATIONS[@]}"; do
  combo="${COMBINATIONS[$i]}"

  cmd=( "$PARALLEL_RUN" )
  for kv in $combo; do
    cmd+=( "--env" "$kv" )
  done
  for kv in "${SINGLE_ENV_VARS[@]+"${SINGLE_ENV_VARS[@]}"}"; do
    cmd+=( "--env" "$kv" )
  done
  cmd+=( "${PASSTHROUGH[@]+"${PASSTHROUGH[@]}"}" )
  cmd+=( "${TARGETS[@]+"${TARGETS[@]}"}" )

  echo "[$((i+1))/${#COMBINATIONS[@]}] Running: ${cmd[*]}"

  "${cmd[@]}"
  status=$?
  if (( status != 0 )); then
    echo "Warning: Run [$((i+1))] exited with status $status"
  fi
  unset all_env_for_tags
done

echo ""
echo "Grid search complete."

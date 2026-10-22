---
description: How to navigate and read log files (logs/pytest/, logs/unillm/)
---

# Log Directory Navigation

## Tool Behavior for Logs

The `logs/` directory is gitignored, which affects tool availability:

| Tool | Works? | Notes |
|------|--------|-------|
| **Read** | ✅ Yes | Preferred for reading log file contents |
| **Shell** | ✅ Yes | Use `ls` to explore directory structure |
| **Glob** | ❌ No | Git-aware index excludes gitignored paths |
| **Grep** | ❌ No | Git-aware index excludes gitignored paths; use `rg -uu` from the shell |

## Log Directories

| Directory | Purpose |
|-----------|---------|
| `logs/pytest/` | Test output logs (datetime-prefixed subdirs per run, one file per tmux session) |
| `logs/unillm/` | Raw LLM request/response traces |

`unify act` writes its runtime logs to `UNIFY_LOG_DIR`, by default `<UNIFY_HOME>/logs` (`~/.unify/logs`), not under the repo.

## Practical Steps

**Step 1: Explore with Shell**
```bash
# List log directories (sorted by time, newest last)
ls logs/pytest/

# List contents of a specific run
ls logs/pytest/2025-12-05T14-30-45_unify_dev_ttys042/
```

**Step 2: Read with Read tool**
```
Read: logs/pytest/2025-12-05T14-30-45_unify_dev_ttys042/function_manager-primitives-test_environment_namespaces.txt
```

## Worktree Symlinks

In worktrees, log directories contain a `_root` symlink pointing to the main repository's logs. Use this when looking for logs from tests run in the main repo.

```bash
# List main repo's logs from a worktree
ls logs/pytest/_root/
```

## Example: Debugging a Test Failure

```bash
# 1. Find recent log directories
ls logs/pytest/

# 2. List logs in the most recent run
ls logs/pytest/2025-12-21T16-00-00_unify_dev_ttys042/
```

Then use the Read tool:
```
Read: logs/pytest/2025-12-21T16-00-00_unify_dev_ttys042/function_manager-primitives-test_environment_namespaces.txt
```

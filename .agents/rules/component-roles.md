---
description: Which component owns what (the CLI, the actor, the run record, memory, the lanes) and where each one's jurisdiction ends
---

# Component Roles

Use this to decide which component owns a change and where its jurisdiction ends. ARCHITECTURE.md and docs/memory.md
are canonical for how each one works.

### The CLI (`unify/cli.py`)
- **Role**: `unify act`: one request, or a session (`--persist`, `--jsonl`), through one actor.
- **Owns**: argument parsing, the driver's stdin/stdout contract, and, with memory on, starting the lane relay
  (`unify/lanes/launch.py`) and the episode line.

### The actor (`unify/actor/`)
- **Role**: answers a request by writing Python, one cell per `execute_code` call.
- **Scope**: the tool loop (`unify/common/async_tool_loop.py`), its prompt (`prompt_builders.py`), its in-code tools
  (`in_code_tools.py`, `tool_handlers.py`) and the worker that runs cells in the sandbox (`execution/`).
- **Reaches**: `tools.<name>` (the registry), the embedding application's `primitives.<name>` namespaces, the run
  record, and with memory on the `memory` package and `memory.ask`.

### The run record (`unify/agents/`)
- **Role**: one append-only record per run for the actor, its teammates, the requester, the librarian and the harness.
- **Owns**: `send`, `messages`, `wait` and `who`, the same tools for every agent (`run_record_tools.py`); teammates
  and their lifetimes (`pool.py`).

### Memory (`unify/memory/`)
- **Role**: turns finished requests into a tested library that later requests read. One switch, `UNIFY_MEMORY`
  (`off`, `on`, `in_episode`).
- **Parts**: episodes and the request's memory (`integration/request.py`), the episode-end fork and captures
  (`integration/fork.py`, `captures.py`, `capture_trace.py`; a capture is never run), the attention ledger
  (`attention.py`), the writer service and its passes (`writer_service.py`, `writer_pass.py`), the gate (`gate.py`,
  `gate_policy.py`: labels, refusing only for safety, paths, syntax and layout) and the librarian (`librarian.py`).
- **Boundary**: only the writer edits the library; the actor reads a pinned copy.

### The lanes (`unify/lanes/`)
- **Role**: with memory on and `OPENROUTER_API_KEY` set, a local relay that holds the key and gives the writer, its
  helpers, the episode-end fork and embeddings their own lanes and tokens under one cap, `UNIFY_MEMORY_CAP_USD`.
- **Boundary**: the key stays in the relay; memory's processes get only their lane's token, on an inherited
  descriptor.

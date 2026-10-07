# cleanslate-v1 worker run kit (for OPS)

**For the frozen office prereg (PREREG-office-v1),** use the snapshot in `frozen/office-v1/`
(`frozen/README-office-v1.txt`). This working tree has moved on, and its files no longer match that prereg's
hashes.

A cell is one run of one arm over the office-v1 core stream, in the frozen order. `run_cell.sh` does
everything on the worker:

1. It creates a virtual environment from `/usr/bin/python3`. Only the standard library is used, so nothing
   is installed.
2. It sets the **worker** limits profile.
3. It runs a preflight: it starts one confined workspace, reads back its cgroup limits, and stops it.
4. It runs the cell.
5. It verifies cleanup: no task folders are left and no workspace process or scope survives.
6. It writes everything to the run folder, where `bench_workers sync` and `cost_ledger.py` already look.

## Commands

```
# zero-cost rehearsal with the scripted fake (do this first on each worker)
kit/run_cell.sh --arm full --run-index 0 --tasks ofc-d01,ofc-r01 --order frozen --fake

# paid cell (only after MAIN freezes the prereg and gives GO)
OPENROUTER_API_KEY=... kit/run_cell.sh --arm full|no-memory --run-index 1|2 --tasks all --order frozen \
    --confirm-paid --prereg prereg/PREREG-office-v1.md --prereg-sha256 <sha256 of the frozen file>
```

- **The key:** provide it the same way as for the Unify office cells, in the environment of this process
  only. The kit checks that it is present, and nothing else. It never prints, logs or writes it, and it never
  passes it to the workspace.
- **Defaults** (matched to the Unify office cells, `artifacts/everyday-v1/cells-conf-office-core-p*.json`):
  - model `openai/gpt-6-luna`, effort `low`, through OpenRouter;
  - per task: USD 0.50, 200 calls and 900 s;
  - the runaway guard stops the cell above 40 USD/h over 10 minutes (exit 4).
- **Optional:**
  - `--max-fs-mb N` refuses to start unless the workbench sits on a filesystem of at most N MB. This is how a
    loop-mounted or size-limited run folder becomes a hard disk bound.
  - `--dedicated-user NAME` enables the prlimit fallback.

**Exit codes:**

| Code | Meaning |
|---|---|
| 0 | done and clean |
| 1 | cleanup not verified (the evidence is kept) |
| 2 | refused before starting |
| 4 | runaway stop |

## Output

- **The cell folder,** `<workbench>/appworld-r0/<cell>/`, holds:
  - `attempt.json`, `cell.json` and `run.json`, with source hashes, model, caps, limits read back and prereg hash;
  - `records.jsonl`, one line per instance;
  - `summary.json` and `cleanup.json`;
  - `cost-record.jsonl`;
  - `procedures-before-NN.json`;
  - `workspaces/` and `held/`, which hold the scored states of held deliveries.
- **The lab journal** is at `<workbench>/runs/<cell>/attempts/<attempt>/costs.jsonl`, in the format
  `journal_accounting.read_journal` reads.
- **Cell names:**
  - `everyday-office-cleanslate-<arm>-r<N>-<UTC>` for paid cells, filed under `everyday`;
  - `fake-…` and `rehearse-…` prefixes for dry runs, filed under `everyday-dryrun`.
- **The workbench** defaults to `~/.local/share/continual-harness-research/runtime-appworld/continual-arc-baselines/.workbench`.

## Worker prerequisites (verified only on the worker; the kit refuses to start without them)

- **bwrap** must be installed, and unprivileged user namespaces must be allowed.
- **cgroup limits.** One of the following:
  - **Preferred: a user systemd that can create scopes, with the `cpu`, `memory` and `pids` controllers
    delegated to the user manager.** On this laptop only `memory` and `pids` are delegated, so the worker
    profile refuses here. Delegation is root configuration, for example
    `systemctl edit user@.service` with `[Service] Delegate=cpu cpuset io memory pids`.
  - **Otherwise: a dedicated unprivileged user** (for example `cleanslate-run`), passed as `--dedicated-user`.
    The per-uid `RLIMIT_NPROC` is acceptable only because that user runs nothing else.
- **The kit's location:** the prototype is not committed. OPS copies `prototypes/cleanslate-v1/` into a
  research checkout on the worker. The adapter finds `benchmarks/everyday` relative to that folder, and
  `cell.json` records the sha256 of every source file. These must match the prereg's source hashes.
- **Office data and source:** the office-v1 data must be at `~/.local/share/continual-harness-research/everyday/office-v1`, or at the path in `EVERYDAY_DATA`.
  The repository's `benchmarks/everyday/src` must be present, for the scorer.
- **Optional hard disk bound:** a size-limited filesystem for the workbench (loop mount, set up as root). Then
  pass `--max-fs-mb`.

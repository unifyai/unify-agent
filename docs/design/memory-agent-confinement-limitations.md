# Memory agents: host-side reads of box-written directories (limitations)

The writer, the curator and their helpers run at the same time: a helper's model calls and tools run while the writer
runs cells. Every directory a box can write is therefore read and written by the host only through descriptors opened without following links.

## Directories a box writes, and who reads them on the host

| directory | written by | read on the host by | discipline |
|---|---|---|---|
| the writer's (or curator's) `/memory` box | that agent's cells, read-write | that agent's own tools (`read`, `grep`, `check`, `fixture`, the gate at `finish`), the cancel path, and a helper's library snapshot (`_mirror`, from the helper's task) | descriptor-relative opens (`safe_paths`, below), so the tools may run during a cell (in-code tools); the cancel path waits for a running cell's box to end; the snapshot is copied from directory descriptors with `O_NOFOLLOW` (`writer_pass._mirror`); `check` judges its own copy and `finish` stops the box before the gate |
| the writer's `/outputs` | the writer's cells | the writer's `read` and `grep` | descriptor-relative opens (`safe_paths`) |
| a helper's `/work` | that helper's cells | that helper's `read`, `grep` and its quota walk (`helpers.work_over`) | descriptor-relative opens and mode changes (`safe_paths`) |

`/inputs` (the writer's, also bound read-only in every helper box), the pass record, the pool, the drafts index, the
similar index and a helper's staged output are written by the host only; every box binds `/inputs` read-only.

## Host reads and writes under a box-writable directory

Every host-side open of a path under a box-writable root (`/memory`, `/work`, `/outputs`) goes through
`unify/memory/safe_paths.py`: the path is checked lexically (one of the role's root prefixes, no `..`, no NUL,
bounded length), then each directory on the way is opened from the one before it with `O_NOFOLLOW | O_DIRECTORY`,
the leaf with `O_NOFOLLOW | O_NONBLOCK`, and the opened descriptor is checked with `fstat` (a regular file or a
directory, else refused). Listings use `os.scandir` on an opened descriptor; writes create a new `O_EXCL` file and
rename it over the target. Any link is refused, wherever it points. The quota measures change a mode only on an
entry opened with `O_PATH | O_NOFOLLOW`, through `/proc/self/fd/<n>`, and refuse to measure without `/proc`.

So a box may write the same directory while a host tool reads it: tool calls made from a running cell (the
in-code tools), a box process kept alive between cells (the persistent interpreter), and several helpers at once no
longer depend on the order of calls. `tests/memory/test_resolver_race.py` swaps directories for links and files for
FIFOs and links in a tight loop while every host reader and writer runs, and checks that no reply carries decoy
bytes, no decoy changes, and no call hangs.

Still read by path, and safe only because no box can write the directory at that moment: the library layout
discovery at `finish` (the box is stopped before the gate runs) and the read-only bind check when a box starts.

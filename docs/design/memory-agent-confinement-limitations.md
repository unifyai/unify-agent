# Memory agents: host-side reads of box-written directories (limitations)

The writer, the curator and their helpers run at the same time: a helper's model calls and tools run while the writer
runs cells. Every directory a box can write is therefore read by the host only under one of two disciplines.

## Directories a box writes, and who reads them on the host

| directory | written by | read on the host by | discipline |
|---|---|---|---|
| the writer's (or curator's) `/memory` box | that agent's cells, read-write | that agent's own tools (`read`, `grep`, `check`, `fixture`, the gate at `finish`), the cancel path, and a helper's library snapshot (`_mirror`, from the helper's task) | the agent's own tools run between its cells; the cancel path waits for a running cell's box to end; the snapshot is copied from directory descriptors with `O_NOFOLLOW` (`writer_pass._mirror`), because it runs while the writer's cells run |
| the writer's `/outputs` | the writer's cells | the writer's `read` and `grep` | between its own cells |
| a helper's `/work` | that helper's cells | that helper's `read`, `grep` and its quota walk (`helpers.work_over`, `lstat` only) | between its own cells |

`/inputs` (the writer's, also bound read-only in every helper box), the pass record, the pool, the drafts index, the
similar index and a helper's staged output are written by the host only; every box binds `/inputs` read-only.

## Precondition of the read and grep resolver

`views.resolve` checks a path's containment with `realpath` and then opens it by path. That is safe only while each
agent's resolver reads directories that the host writes or that the agent's own boxes write, and the agent's tool
calls run one at a time with every box ended when its cell returns, so nothing can swap a link between the check and
the open. Running one agent's tool calls in parallel, keeping a box process alive past its cell, or letting one
agent's box write a directory another agent's resolver reads would break it: the resolver must first open every
path component from a root descriptor with `O_NOFOLLOW` (as `edit_tools` and `_mirror` do) and check the opened file,
rather than the path.

# Memory

Memory lets later requests use what earlier ones learned. One setting controls it, `UNIFY_MEMORY`, which takes
`off`, `on` (the default) or `in_episode`. Any other value is refused when the settings are read, and the error lists
the three. The rest of memory's configuration is in [CONFIG.md](../CONFIG.md); the code is under `unify/memory/`.

## What happens with `UNIFY_MEMORY=on`

1. **An episode is recorded.** Each request is an episode in the memory home, `<UNIFY_HOME>/memory`. The episode
   records the request, the cells the actor ran with their outputs, its tool calls and its reply, plus any signal that
   arrives about it later, such as a verdict on the result.
2. **A fork stages what it did.** When the episode ends, the actor's own conversation is forked once, on the fork
   lane. The fork writes a summary and, where the episode computed something, a procedure that redoes it. Its
   capture is a record of what the episode did. **Nothing is ever run in order to classify, deduplicate or rank a
   capture.**
3. **The ledger says what needs reading.** The attention ledger opens an entry for each piece of new experience:
   - a settled fork;
   - a use of a library item that went badly;
   - a signal that arrived after its episode ended;
   - an item called again in new episodes.

   An entry stays open until a writer pass decides it.
4. **The writer keeps the library.** The writer is a detached service, one per memory home. It hands open entries to
   writer passes, and each pass is a model with tools in a confined box:
   - it reads the entries' material;
   - it may start helpers;
   - it lands, changes, merges or retires library items (tested functions and linked notes) and decides each entry.

   A gate checks every candidate before it lands. Safety, paths and syntax refuse a candidate. Every other finding
   is recorded on the item's label. The gate runs code only for an item that is a pure function of its input.
5. **The next episode is served from the library.** An episode starts with a read-only copy of the library as it is
   then, bound as the Python package `memory` in its cells. The librarian answers questions about it (`memory.ask`).

Spend is bounded by `UNIFY_MEMORY_CAP_USD`, which covers memory's calls over the memory home's life. Every call is
booked at its worst case until it is priced. Once the cap is reached, the library stops changing and requests are still
served from it.

## `UNIFY_MEMORY=in_episode`

`in_episode` does everything `on` does. It also lets memory change **during** an episode:

- The actor may pause at a checkpoint (`tools.checkpoint(note=...)`). A pause also happens at a compaction of its
  context and before a new user message joins the conversation.
- Each pause forks the cells since the last pause into a segment of the episode. The segment's capture becomes a
  ledger entry, just as a whole episode's does. A segment that captured the same procedure as an earlier entry is added
  to that entry instead of opening its own.
- At each cell boundary, the episode's copy of the library is re-pinned to what the writer has landed meanwhile. The
  next cell is told, in one notice, which items changed.

In evaluation, `in_episode` cost about twice as much per visit as `on`, with no measured gain. Use it only when the
episodes are long enough that something learned early can pay off within the same episode.

## `UNIFY_MEMORY=off`

No memory home, no fork, no writer and no memory tools. Every request runs as if nothing had come before it. This is
the baseline that the other two settings are measured against.

## Where to look

| Part | Module |
|---|---|
| The switch, the writer's model, the cap and the lanes | `unify/memory/integration/switch.py` |
| A request's memory (recording, serving, the fork) | `unify/memory/integration/request.py`, `integration/fork.py` |
| Memory during an episode | `unify/memory/integration/in_episode.py` |
| The attention ledger | `unify/memory/attention.py` |
| The writer service and its passes | `unify/memory/writer_service.py`, `unify/memory/writer_pass.py` |
| The gate | `unify/memory/gate.py`, `unify/memory/gate_policy.py` |
| Captures | `unify/memory/captures.py`, `unify/memory/capture_trace.py` |
| The librarian | `unify/memory/librarian.py` |

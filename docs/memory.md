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

   An entry stays open until a writer pass decides it. A new capture waits until its **capture group** is due:
   captures whose records show they act through the same tools, touch the same library items or read the same shape
   of input (never an id, a word or the order they came in). A group is due once two of them are open, once a new one
   relates to captures an earlier pass decided, or, for a capture still alone, after 8 later captures or 10 minutes
   with none. Every other kind of entry is handed at once.
4. **The writer keeps the library.** The writer is a detached service, one per memory home. It hands open entries to
   writer passes, and each pass is a model with tools in a confined box:
   - it reads the entries' material: for a capture group, its members together with the related captures earlier
     passes decided and the library items the group touches;
   - it may start helpers;
   - it lands new library items (tested functions and linked notes), changes or renames existing ones, and decides
     each entry. A pass never removes an item or merges two in the library: the gate refuses aliases, declared
     retirements and a smaller library. The writer retires or replaces an item it has learnt is harmful by deciding an
     entry about it (`retire`, or `merged_into` the item that replaces it): the item stays in the library, marked
     retired, and is no longer served.

   A gate checks every candidate before it lands. Safety, paths and syntax refuse a candidate. Every other finding
   is recorded on the item's label. The gate runs no code: the only code that runs is a test the writer states
   and runs itself (`tools.test`). Every item shows what it rests on: its kind (a curated item, a recorded
   procedure or a fixed answer), how many episodes it comes from and how many uses confirmed it.
5. **The next episode is served from the library.** An episode starts with a read-only copy of the library as it is
   then, bound as the Python package `memory` in its cells. The librarian answers questions about it (`memory.ask`).

`UNIFY_MEMORY_CAP_USD` bounds what memory spends in one run, meaning one `unify act` including the writer service it
ran. The run's count starts when the `unify act` makes its first memory request; a writer service that stops and starts
again within the run (its 36-hour lifetime, a crash) keeps counting. A pass has no cap of its own. Every call is
booked at its worst case until it is priced. Once the cap is reached, the library stops changing for
the rest of the run, and requests are still served from it.

The writer service is started by the harness, at a request's start and end. A harness that must own it (for example
one whose requests run in sandboxes that end with them) may run it as its own foreground process,
`python -m unify.memory.writer_service --home <UNIFY_HOME>` (`--daemon` detaches it instead); requests then find it
running and never start a second one.

When `unify act` ends, the writer's pass in flight gets at most 5 minutes to finish. Then it is suspended: its turns
are kept, and every entry it held stays owed and is handed again in the next run, never dropped. Ctrl-C stops the
waiting.

## `UNIFY_MEMORY=in_episode`

`in_episode` does everything `on` does. It also lets memory change **during** an episode:

- The actor may pause at a checkpoint (`tools.checkpoint(note=...)`). A pause also happens at a compaction of its
  context and before a new user message joins the conversation.
- Each pause forks the cells since the last pause into a segment of the episode. The segment's capture becomes a
  ledger entry, just as a whole episode's does. A segment that captured the same procedure as an earlier entry is added
  to that entry instead of opening its own.
- At each cell boundary, the episode's copy of the library is re-pinned to what the writer has landed meanwhile. The
  next cell is told, in one notice, which items changed.

`in_episode` is not the default because of cost and evidence, in any world. Each pause adds a fork and a writer
read, so an episode costs more: about twice as much per visit as `on` in the one evaluation so far (Continual-ARC).
No world has yet shown a gain within an episode. Use it only when the episodes are long enough that something learned
early can pay off within the same episode, and measure it there.

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
| Capture groups (which captures a pass reads together, and when) | `unify/memory/capture_groups.py` |
| The writer service and its passes | `unify/memory/writer_service.py`, `unify/memory/writer_pass.py` |
| The gate | `unify/memory/gate.py`, `unify/memory/gate_policy.py` |
| Captures | `unify/memory/captures.py`, `unify/memory/capture_trace.py` |
| The librarian | `unify/memory/librarian.py` |
| The lane relay (memory's model calls, under `UNIFY_MEMORY_CAP_USD`) | `unify/lanes/` |

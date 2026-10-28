# Memory

Memory lets later requests use what earlier ones learned. One setting controls it, `UNIFY_MEMORY`, which takes
`off` or `on` (the default). Any other value is refused when the settings are read, and the error lists the two. The rest of memory's configuration is in [CONFIG.md](../CONFIG.md); the code is under `unify/memory/`.

## What happens with `UNIFY_MEMORY=on`

1. **An episode is recorded.** Each request is an episode, committed to `<UNIFY_HOME>/episodes.git`: the request, the
   cells the actor ran with their outputs, its tool calls and its reply, plus any signal that arrives about it later,
   such as a verdict on the result. A verdict line may carry the verdict's text exactly as the world showed it; the
   text passes the run's redactor and is kept beside its signal for the writer, and the label alone is the verdict. A
   SQLite evidence store, `<UNIFY_HOME>/memory-evidence.sqlite`, indexes the
   episodes and every use of a library item.
2. **A fork writes down what it did.** When the episode ends, the actor's own conversation is continued once, on the
   fork lane, with a fixed instruction. The fork writes a summary (`summary.md`) and, where the episode computed
   something, a procedure that redoes it (`procedure.py`) with what broke on the way (`broke.md`). It also writes its
   lessons (`notes.md`) and, with no procedure needed, **note candidates** (`notes/<name>.md`): a preference or rule
   someone stated, a correction, an outcome or a caution the episode proved, each with what it applies to
   (`applies_to`) and, when someone stated it, who (`stated_by`). It posts to `@writer` on the run record only to point
   at a procedure, candidate or note it wrote, or to correct a served item (named as `item:<id>`); never to say there is
   nothing to keep. The harness turns
   the procedure into a **capture**: the procedure plus a signature and a record the harness computes, with the
   outcome taken from the signals the actor could see (`positive`, `none` or `contrary`). **Nothing is ever run in
   order to classify, deduplicate or rank a capture.** A capture is served at once to later requests, before any
   writer pass has read it. The fork needs a fork lane, which `unify act` starts with memory on and
   `OPENROUTER_API_KEY` set (or a launcher gives). Without one no fork runs and no capture is made, so
   `ANTHROPIC_API_KEY` alone stores none.
3. **The ledger says what needs reading.** The attention ledger opens an entry for each piece of new experience:
   - a settled fork;
   - a use of a library item that went badly;
   - a signal that arrived after its episode ended;
   - an item called again in new episodes;
   - a run-record message to the writer;
   - an entry the writer deferred, or a draft the gate refused.

   An entry stays open until a writer pass decides it. A new capture waits until its **capture group** is due. An
   episode that left no capture but left lessons or note candidates is an entry with the same standing: it joins a
   group through the library items its episode used (or a judge's reading), or is due on its own once its request has
   ended, and its lessons reach the writer as note candidates. An episode that left neither is closed by the harness
   and no pass runs on it.
   Captures are related when their records show they make the same calls, when they use the same library items, or
   when the writer's model, asked as a judge, reads them as the same kind of work; a shared shape of input alone never
   relates two captures, and no id, word or order is read. A group is due once two of its captures are open, or once
   a new one relates to captures an earlier pass decided. A capture with no related capture is due once its own
   request has ended: it goes with the pass that runs then, or, when none runs, in one pass with every other such
   capture. No count of later captures and no idle time makes anything due. Every other kind of entry is handed at
   once, except a run-record message that carries nothing of its own (no `item:`, `capture:` or `entry:` ref and no
   answer to the writer, posted by a fork or the harness): it makes no pass, and goes along with the next pass that
   runs for other entries.
4. **The writer keeps the library.** The writer is a detached service, one per memory home. It hands open entries to
   writer passes, and each pass is a model with tools in a confined box (no network, no credentials):
   - it reads the entries' material: for a capture group, its members together with the related captures earlier
     passes decided and the library items the group touches;
   - it may start helpers;
   - it lands new library items, changes or renames existing ones, and decides each entry.

   The library, `<UNIFY_HOME>/memory`, is a git repository of **functions** (`memory/<package>/<module>.py`, each with
   its tests) and **notes** (`notes/<topic>/<slug>.md`). A note needs no capture behind it. Its front matter may
   carry its kind (`kind`: `preference`, `rule`, `caution` or `lesson`; optional) and its scope: `applies_to` (what
   it holds for) and `stated_by` (who stated it, by name or role), shown wherever
   the note is served (`memory.get`, `memory.items()`, the index and search). Notes and functions link both ways when
   they are related: a note lists the functions it uses in its `uses:` front matter, and a function's docstring names
   its notes on a `Notes:` line; a note that is about no function stands on its own.
   Every copy also carries generated files that are never committed: `INDEX.md` and `links.json`, which show both
   directions of every link, and the package files that bind the library as `memory`.

   A pass never removes an item or merges two in the library: the gate refuses aliases, declared retirements and a
   smaller library. The writer retires or replaces an item it has learnt is harmful by deciding an entry about it
   (`retire`, or `merged_into` the item that replaces it): the item stays in the library, marked retired, and is no
   longer served.

   A gate checks every candidate before it lands. Safety, paths and syntax refuse a candidate. Every other finding
   is recorded on the item's label. The gate runs no code: the only code that runs is a test the writer states
   and runs itself (`tools.test`). Every item shows what it rests on: its kind (a curated item or a recorded
   procedure), how many episodes it comes from and how many uses confirmed it. An item's status
   follows its uses: it starts `on-trial`, becomes `adopted` after clean uses in at least three episodes with no
   failed use, and is `retired` after failed uses in at least two episodes, or by the writer's decision. The writer's
   brief tells it that an item is stale when a later outcome contradicts it (a verdict after an episode that used it,
   with the verdict's text when the world gave one, or a failed use after it was served), and that it then changes the
   item to what the outcome shows or retires it.
5. **The next episode is served from the library.** An episode starts with a read-only copy of the library as it is
   then, bound as the Python package `memory` in its cells, and with the captures served beside it. Before the actor's
   first turn, the librarian is asked once what in the library fits the request; its answer is part of the actor's first
   message. In its cells the actor calls `memory.ask(...)` to ask again, `memory.items()` to list the library and
   `memory.get(id)` to read an item or call a function; `help(memory)` documents them. The librarian checks every
   call it suggests against the item's real interface, and never suggests an item whose call failed earlier in the
   same request, or one the request had used before a `fail` verdict the actor was shown (a
   `{"checker": {"label": "fail"}}` line; the items are the ones its cells called and the ids they passed to
   `memory.get`, read from the cells' code, never from text). When a driver marks each new request of a persistent
   session (`{"message": ..., "request": "new"}`), such a verdict bears only on the items of the request it followed,
   and its exclusion holds for that request and the next; without the marker it holds for the whole episode.

   The librarian judges the fit of functions and procedures only. **Standing notes** are shown with no fit judgement:
   the actor's first message also carries a digest of the notes whose front matter's `kind` is `preference`, `rule`
   or `caution` (a note with no `kind` and a `lesson` are not in it), each with what it `applies_to` and
   who `stated_by`, when the note says so. It is ordered by status (`adopted` first), then by positive less negative
   uses, then by the newest change, and holds at most 8 entries of about 200 characters each. That is a display limit
   only: the digest ends with how many it shows of how many, and `memory.items(kind='note')` and `memory.get(id)`
   reach every note. A persistent session re-sends the digest at a later request only when it has changed, so the
   cached prompt prefix holds (`unify/memory/standing_notes.py`).

   A capture whose `run` never reads its input, or whose every `return` is a fixed value, is labelled a **record**
   when it is captured, from its code's syntax alone (`capture_shape.record_kind`: no word, value or task id is read).
   A record is served as what was done last time, never as a fit for a different request: its card, its catalogue
   row, its search row and the request-start block say so, and the librarian is told the same. It is labelled, never
   refused. Every card states how its uses went beside its status: `used N times: M positive, K negative`.

All of memory's agents share the run's record with the actor and its teammates: an append-only log they post to and
read, addressing one another as `@writer`, `@librarian`, `@actor`, `@fork` or `@helpers`. It is how the fork points the
writer at what it wrote, and how the actor tells the writer that a served item was wrong.

`UNIFY_MEMORY_CAP_USD` bounds what memory spends in one run, meaning one `unify act` including the writer service it
ran. The run's count starts when the `unify act` makes its first memory request; a writer service that stops and starts
again within the run (its 36-hour lifetime, a crash) keeps counting. A pass has no cap of its own. Every call is
booked at its worst case until it is priced. Once the cap is reached, the library stops changing for
the rest of the run, and requests are still served from it. HTTP 402 (the account is out of credit) stops the writer
the same way. A key proxy's refusal for its own budget or reserve, which is not the writer's lane's own cap, does not
stop it at once: its next pass waits 1, 2, 4 and then 8 minutes after each such refusal in a row, and a fifth refusal
in a row stops it as the cap does.

The writer service is started by the harness, at a request's start and end. A harness that must own it (for example
one whose requests run in sandboxes that end with them) may run it as its own foreground process,
`python -m unify.memory.writer_service --home <UNIFY_HOME>` (`--daemon` detaches it instead); requests then find it
running and never start a second one.

When `unify act` ends, the writer's pass in flight gets at most 5 minutes to finish. Then it is suspended: its turns
are kept, and every entry it held stays owed and is handed again in the next run, never dropped. Ctrl-C stops the
waiting.

## `UNIFY_MEMORY=off`

No memory home, no fork, no writer and no memory tools. Every request runs as if nothing had come before it. This is
the baseline that `on` is measured against. [memory-off.md](memory-off.md) describes it in full: what runs, what is
switched off and where, and what the actor reads differently.

## Where to look

| Part | Module |
|---|---|
| The switch, the writer's model, the cap and the lanes | `unify/memory/integration/switch.py` |
| A request's memory (recording, serving, the fork) | `unify/memory/integration/request.py`, `integration/fork.py` |
| The attention ledger | `unify/memory/attention.py` |
| Capture groups (which captures a pass reads together, and when) | `unify/memory/capture_groups.py` |
| The writer service and its passes | `unify/memory/writer_service.py`, `unify/memory/writer_pass.py` |
| The gate | `unify/memory/gate.py`, `unify/memory/gate_policy.py` |
| Captures | `unify/memory/captures.py`, `unify/memory/capture_trace.py` |
| The librarian | `unify/memory/librarian.py` |
| The standing-note digest | `unify/memory/standing_notes.py` |
| The lane relay (memory's model calls, under `UNIFY_MEMORY_CAP_USD`) | `unify/lanes/` |

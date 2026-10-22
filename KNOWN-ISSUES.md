# Known issues

Behaviour that main ships as rev 2 was evaluated, recorded here with the fix proposed for a later run, or the direction
when no fix is settled. Main does not change any of it until a later run has tested the fix.

## A parsed fixture's bytes are not checked against its reader

A parsed fixture (`fixture(..., parser=...)`) is a recorded text block parsed by a stored reader. Before rev 2 the gate
re-derived it: it ran the reader confined on the recorded text and refused a fixture whose bytes differed. In rev 2
the gate runs no reader to check a fixture (`library_checks`: "re-deriving it would run its reader"), so it labels
the fixture's owners `not run` and leaves the bytes unverified. An unverified parsed fixture grounds no argument, so
its item stays at the lowest evidence level.

Direction: open. Any fix must be static, with no execution: no mechanism may run a capture or a function to classify,
dedupe, rank or test it.

## A landed procedure's recorded input is the request's data, which `memory.match` does not match

Under the trace contract a capture is never run, so its signature cannot describe the value its `read_input` would
read. It describes the recorded data instead: for a value-form procedure, the list of JSON values the request holds
(`capture_trace.signature_input`). A procedure landed from such a capture keeps that signature. `memory.match`
builds its structural query from each JSON value of the new request, so a request with the same structure as the
landed procedure's own request gives a value of the rows' shape, while the procedure's input is a list holding that
value. The default matcher does not list it. A capture saved against the read value itself is matched.

Direction: open. Any fix must be static, with no execution: for example, compare the query with each value of a
recorded data list, from the shapes alone.

## Heavy forking can reach the relay's cap before the writer's own budget

When `unify act` starts the lane relay, the relay holds one cap, `UNIFY_MEMORY_CAP_USD`, over all of memory's lanes:
the writer, its helpers, the episode-end fork and embeddings (`unify/lanes/launch.py`). The writer service plans its
passes against the same cap, but it does not see the fork's and the embedding lane's spend as it happens. A run with
many episode-end forks can therefore use up the relay's cap while the writer's own accounting still shows room. The
relay then refuses the writer's next call (a refusal the pass reports as the cap), so memory stops changing for the
rest of the run. Nothing is spent past the cap: this is an early stop, never an overspend.

Direction: open. A later run can show the writer the relay's remaining room before it plans a pass.

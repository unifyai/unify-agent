# Known issues

Known behaviour of the harness, each with the fix proposed for it, or the direction when no fix is settled. None of
it changes until a run has tested the fix.

## A parsed fixture's bytes are not checked against its reader

A parsed fixture (`fixture(..., parser=...)`) is a recorded text block parsed by a stored reader. The gate runs no
reader to check a fixture (`library_checks`: "re-deriving it would run its reader"), so it labels
the fixture's owners `not run` and leaves the bytes unverified. An unverified parsed fixture grounds no argument, so
its item stays at the lowest evidence level.

Direction: open. Any fix must be static, with no execution: no mechanism may run a capture or a function to classify,
dedupe, rank or test it.

## A landed procedure's recorded input is the request's data, which `Item.fits` does not match

Under the trace contract a capture is never run, so its signature cannot describe the value its `read_input` would
read. It describes the recorded data instead: for a value-form procedure, the list of JSON values the request holds
(`capture_trace.signature_input`). A procedure landed from such a capture keeps that signature. `Item.fits` compares
the structure of what it is given with that recorded input, so a value of the rows' shape, taken from a request with
the same structure as the procedure's own, does not fit: the procedure's recorded input is a list holding such a
value. A capture saved against the read value itself fits.

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

## Capture grouping rereads every pair on every service turn

The writer service groups new captures on every turn of its loop (`capture_groups.groups`, about every 2 s). Each
time, it compares every open capture with every other live grouped entry and reads each entry's item uses from the
evidence store again; only a capture's own facts are cached (`HomeProfiles`). The cost grows with the square of a
run's entries. That is fine at a few hundred entries, but a cost risk over long campaigns.

Direction: cache the item uses per episode and compare only what changed since the last turn.

## A pass can start just before a stop marker

The service checks the stop marker at the top of each turn and then hands a pass, so a stop (a suspend or a drain)
set just after that check finds a pass already running. That pass runs to its deadline; its entries then return as
`suspend`, with no miss. An evaluated run showed about 5.8 minutes of dead time per stage boundary when the route
was already gone.

Proposed fix, to evaluate in a later run: check the marker again just before handing, and end a pass whose first
call fails after the marker.

## A wall-clock test can time out under the full suite's load

`tests/memory/test_adapter_worktree.py::test_benign_yaml_near_the_parse_cap_shapes_within_the_timeout` checks that
a YAML file near the parse cap is shaped within the adapter's `PARSE_TIMEOUT`. In one full run of the suite (ten
files at a time), it timed out. It passed when run alone on the same tree, and in the two full
runs before it, and no change touched the adapter. The test measures wall-clock time, so a loaded host can fail it.

Direction: open. Do not raise the timeout to make it pass. Run it alone to tell a load failure from a real slowdown.


# Known issues

Behaviour that main ships as rev 2 was evaluated, recorded here with the fix proposed for a later run, or the direction
when no fix is settled. Main does not change any of it until a later run has tested the fix.

## A long read can use up the writer's repair reserve

A writer pass prices its first repair round on round 0's write phase, so that reading the batch does not use up the
budget a repair round needs (`repair.round_reserve`, `unify/memory/writer_pass.py`). The write phase starts at the
writer's first write. In rev 2 every call is made from a cell, and the pass marks the write phase when any
`execute_code` cell starts (`writing()` in the cell branch of the pass loop). A cell that only reads therefore starts
the write phase, and a long read in round 0 is priced into the reserve. A pass that reads for many cells before it
writes can find no repair round fits in what is left, even though the reading was never a write.

Proposed fix, to evaluate in a later run: start the write phase only when a cell changes the writer's box or makes a
writer action (`check`, `fixture`, `test`, `land_capture`), as a tool call did before tools moved into code.

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

# Known issues

Behaviour that main ships as rev 2 was evaluated, recorded here with the fix proposed for a later run. Main does not
change any of it until that run has tested the fix.

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
the most it can do is leave its item at the lowest evidence level; it cannot raise an item's level. A writer can still
store a parsed fixture whose bytes its reader would not produce, and readers of the library see it as recorded data.

Proposed fix, to evaluate in a later run: re-derive a parsed fixture when its reader is statically pure
(`gate_policy.purity`), as the gate already runs a pure item's own tests, and keep the `not run` label otherwise.

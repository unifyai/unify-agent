"""The shared agent record.

One append-only thread per ``unify act`` session that every agent of the run,
the user (or the benchmark driving the run) and the harness post to. An agent
sees new entries that concern it only at its turn boundary, never during a
model call or a code cell. It was the ``UNIFY_AGENTS=record`` switch until the
code freeze baked it in.
"""

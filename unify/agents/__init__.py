"""The shared agent record (``UNIFY_AGENTS=record``).

One append-only thread per ``unify act`` session that every agent of the run,
the user (or the benchmark driving the run) and the harness post to. An agent
sees new entries that concern it only at its turn boundary, never during a
model call or a code cell.
"""


def enabled() -> bool:
    """Whether this process runs with the shared record."""
    from unify.settings import SETTINGS

    return SETTINGS.UNIFY_AGENTS == "record"

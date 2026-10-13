"""Memory v2 (spec v0, docs/design/memory-redesign-spec.md in continual-harness-research).

Episodes are git-tracked and append-only; shared memory is a git repo of environment code and notes that
only gated consolidation passes may extend; a SQLite evidence store indexes both. Nothing here is wired
into the actor yet.
"""

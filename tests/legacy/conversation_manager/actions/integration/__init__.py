"""
ConversationManager → CodeActActor integration tests.

This package holds production-like end-to-end tests that validate:
user message → ConversationManager → CodeActActor → real state managers → observable outcomes.

Testing philosophy (high-signal constraints):
- No fixed sleeps; all waits are condition-driven with explicit timeouts.
- Prefer verifying durable side effects (DB rows, emitted events) over brittle wording.
- Use unique/unguessable tokens for tasks so “correct” answers require real lookups.

Gotchas we account for:
- Some LLM/provider stacks bind background logging workers to an event loop; we reset per test.
- Fresh projects can 404 on brand-new contexts; we pre-create contexts during test setup.
"""

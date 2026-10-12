# Memory agents' compaction: what is shared with the agent loop, and the gap

The writer, the curator and each helper compact their conversations with the agent loop's policy
(`UNIFY_MEMORY_LIBRARY_COMPACT=on`). They do not yet run on the agent loop's own compaction code path. This note
records exactly what is shared, what is not, and why, with file:line evidence at the commit that adds it.

## Shared (same functions, same constants, same settings)

- Trigger: `context_compression.context_over_threshold` at `COMPACTION_THRESHOLD` (0.7; named here, the agent loop
  reads the same constant: `unify/common/_async_tool/context_compression.py:19`, `unify/common/_async_tool/loop.py:2661`)
  against the window `unillm.get_max_input_tokens(endpoint)` gives for the agent's endpoint (the agent loop's
  source: `loop.py:1080`; the memory agents' turn: `unify/memory/writer_pass.py:4904`). A model with no known
  window is never compacted, in both. The memory side calls it through `writer_pass.compaction_due`
  (`writer_pass.py:373`), from the writer's loop (`writer_pass.py:4076`) and each helper's (`helpers.py:1324`).
- Summary request: the agent's last request, unchanged, plus `cache_discipline.COMPRESSION_FORK_INSTRUCTION`, the
  agent loop's `FORK_SUMMARY` path (`unify/common/async_tool_loop.py:534`); the reply's text read with
  `cache_discipline.completion_text`.
- Restart: the leading system message and `context_compression.restart_text(summary, pointer)`; with
  `UNIFY_COMPACTION_KEEP_PREFIX=on`, `kept_prefix` and `loop_user_notice` as the agent loop's keep-prefix restart
  does. The memory agents' pointer says their files under /memory, /inputs and /outputs are unchanged
  (`writer_pass.RESTART_POINTER`), where the agent loop's names its session transcript file.
- No new setting: `UNIFY_MEMORY_LIBRARY_COMPACT` (existing) switches it on; the agent loop's
  `UNIFY_COMPACTION_KEEP_PREFIX` applies as it is.

## The gap

1. Not the agent loop's method. `AsyncToolLoopHandle._compact_context` (`async_tool_loop.py:464`) and
   `_summarise_as_fork` (`async_tool_loop.py:534`) are methods of the agent loop's handle: they read the last
   request from the handle's unillm client (`_cache_discipline.last_sent_request(self._client)`,
   `async_tool_loop.py:560`), send the summary on a fork of that client (`fork_llm_client`, `async_tool_loop.py:587`)
   and restart the loop task. The memory agents run their own loops (`WriterPass.run`, `helpers._run_helper`), whose
   model calls go through a turn function that carries each agent's route, call-kind header, route check and cost
   hook (`writer_pass.unillm_turn`). Moving them onto the handle would mean running the writer, the curator and the
   helpers as agent-loop sessions. Until then, `writer_pass.compact_by_summary` (`writer_pass.py:400`) repeats the
   fork path's request and restart from the shared pieces above, and sends the summary through the agent's own
   turn, so it takes the agent's route and model and is booked like any of its calls.
2. No fallback compressor. When the fork yields no summary, the agent loop falls back to `compress_and_rebuild`
   (`async_tool_loop.py:492` and `:500`), whose compressor opens its own client on the endpoint
   (`new_llm_client(endpoint, origin="compress_messages")`, `context_compression.py:547`): a call off the agent's
   route, unpriced by its cost hook. The memory agents do not take it: a reply with no summary text (or with tool
   calls) leaves the conversation as it was (`writer_pass.py:421`); the call is still booked, and the next call over
   the threshold tries again.
3. Transcripts. The agent loop syncs its session transcript and records the compaction there
   (`transcripts.record_compaction`, `async_tool_loop.py:622`, `:650`). The memory agents keep what a compaction took
   out, with the summary call marked `"compaction_call": true`, in their own whole transcripts
   (`unify/memory/transcripts.py`), and write a `compaction` row (usage and charge) to the pass's
   `tool-events.jsonl`.
4. Step-cap compaction. The agent loop also compacts at its step limit (`UNIFY_STEP_CAP_COMPACT`); the memory agents
   have no step-cap compaction (their operational step guard ends a helper; the writer's calls are bounded by
   `max_calls`).

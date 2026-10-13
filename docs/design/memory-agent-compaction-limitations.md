# Memory agents' compaction: what is shared with the agent loop, and the gap

The writer, the curator and each helper compact their conversations with the agent loop's policy
(`UNIFY_MEMORY_LIBRARY_COMPACT=on`). They do not yet run on the agent loop's own compaction code path. This note
records exactly what is shared, what is not, and why, with file:line evidence at the commit that adds it.

## Shared (same functions, same constants, same settings)

- Trigger: `context_compression.context_over_threshold` with `COMPACTION_THRESHOLD` (0.85) and
  `COMPACTION_AT_TOKENS` (150,000, `UNIFY_COMPACTION_AT_TOKENS`), against the window `unillm.get_max_input_tokens`
  gives for the agent's endpoint; with no known window the fixed point applies alone (item 5 below). The memory side
  calls it through `writer_pass.compaction_due`, from the writer's and curator's loop and each helper's. The prompt
  size it reads is the call's own usage report: unillm hands it on the call's event (`LLMEvent.response["usage"]`),
  the memory agents' turn (`writer_pass.unillm_turn`) attaches it to the message it returns, and
  `writer_pass.split_usage` takes it off again before the message joins the conversation, so no request sends it
  back. A call that reports no usage never triggers compaction.
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
5. The trigger. Every agent, the agent loop included, compacts when a model call's prompt reaches whichever comes
   first: 85% of the model's context window (`COMPACTION_THRESHOLD`, from `unillm.get_max_input_tokens`) or a fixed
   150,000 prompt tokens (`COMPACTION_AT_TOKENS`, set by `UNIFY_COMPACTION_AT_TOKENS`); with no known window the fixed
   point applies alone (`context_compression.compaction_point`, read by the loop through `context_over_threshold`
   and by the memory agents through `writer_pass.compaction_due`). The fixed point is where a conversation is
   summarised and restarted, not a limit on what an agent reads or writes.

Parity. The agent loop and the memory agents decide with the same function and constants. With `UNIFY_TOOLS_IN_CODE=on` the agent loop also compacts the way the memory agents do: before the next model call, with no forced `compress_context` turn and no notice, by the same fork summary and restart. Each pass records the
largest prompt each memory agent sent (`max_prompt_tokens` in the end event and in each `helper.json`) beside the
window the trigger read (`context_window`, null when unknown).

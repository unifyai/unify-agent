from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field

import unillm
from pydantic import BaseModel
from unify import transcripts
from unify.common.llm_client import new_llm_client
from unify.common.token_utils import count_tokens
from unify.function_manager.execution_env import create_base_globals


def context_over_threshold(
    n_tokens: int,
    threshold: float,
    max_input_tokens: int,
) -> bool:
    return n_tokens >= max_input_tokens * threshold


_COMPRESSED_HEADER = "## Compressed Prior Context\n"


@dataclass
class CompressionState:
    """Mutable state tracked across compression passes."""

    raw_archives: list[list[dict]] = field(default_factory=list)
    entries: list[tuple[int, str]] = field(default_factory=list)
    count: int = 0
    image_registry: dict[int, dict] = field(default_factory=dict)
    live_image_ids: set[int] = field(default_factory=set)
    next_image_id: int = 0


@dataclass
class RebuildResult:
    """Everything the handle needs to restart a loop after compression."""

    system_msgs: list[dict]
    tools: dict[str, callable]


COMPRESSION_PROMPT = (
    "You are a context compactor with surgical editing tools. You receive a "
    "conversation transcript as a sequence of JSON-serialized messages, each "
    "prefixed with an `[N]` index. You must compress it by selectively editing "
    "the most verbose entries.\n"
    "\n"
    "## Strategy\n"
    "- Focus on the LARGEST and most verbose entries first — large tool results, "
    "full tracebacks, verbose assistant reasoning, extended thinking blocks. "
    "These yield the most savings.\n"
    "- Leave concise messages untouched. They cost nothing.\n"
    "- You may issue multiple `update` calls per turn (they execute in parallel).\n"
    "- Each tool response includes the current token usage as a percentage of "
    "the context window. Use this to gauge progress and stop when usage is low "
    "enough.\n"
    "- When you have compressed enough, stop calling tools and respond with any "
    "short message to finish.\n"
    "\n"
    "## Compression Rules\n"
    '- Error messages and tracebacks: reduce to "error: <one-line cause>". '
    "Keep the traceback only if later messages reference it.\n"
    "- Tool results: keep only data that was actually used or referenced later. "
    "Discard decorative formatting, repeated schema keys, and unreferenced fields.\n"
    "- Image tags (`[img:N]`): see the Images section below.\n"
    "- Thinking blocks: compress extended reasoning to key conclusions and "
    "decisions. Remove exploratory tangents that didn't affect the outcome.\n"
    "- Assistant messages with tool_calls: compact to tool names and key arguments.\n"
    '- Narration-only assistant text ("Let me look that up"): replace with the '
    "tool call summary. Keep reasoning text only if it informs later steps.\n"
    '- System messages (role "system"): preserve instructions faithfully. '
    "The exceptions are parent chat context sections "
    '("Parent Chat Context" headers) and compressed prior context sections '
    '("Compressed Prior Context" headers) — both can be heavily pruned '
    "as they contain summaries, not instructions.\n"
    "- Every entry must remain non-empty after transformation.\n"
    "- Do NOT invent information that was not in the original message.\n"
    "\n"
    "## `update` Tool — Transformation Code\n"
    "The `transformation` argument is Python code that transforms the message. "
    "The variable `x` holds the full JSON string of the message. The final "
    "value of `x` after execution becomes the new entry.\n"
    "\n"
    "Patterns:\n"
    '- Surgical replace: `x = x.replace("verbose section", "summary")`\n'
    '- Regex: `x = re.sub(r"Traceback[\\s\\S]*?", "traceback omitted", x)`\n'
    "- Truncate: `x = x[:500]`\n"
    "- Parse and rebuild:\n"
    "  ```\n"
    "  msg = json.loads(x)\n"
    '  msg["content"] = "error: connection timeout"\n'
    "  x = json.dumps(msg)\n"
    "  ```\n"
    "- String operations work directly on the JSON string — no need to\n"
    "  parse/serialize for simple replacements.\n"
    "  (`re` and `json` are available in the execution environment)\n"
    "\n"
    "## Images\n"
    "Messages may contain `[img:N]` tags referencing images visible in the "
    "accompanying image blocks below the transcript.\n"
    "- To KEEP an image: leave its `[img:N]` tag in the entry text.\n"
    "- To REMOVE an image: remove its `[img:N]` tag via `update`. "
    "Images whose tags no longer appear in any entry are discarded, "
    "freeing significant context space.\n"
    "- Remove images that are no longer relevant to the conversation's "
    "current direction. Keep images that are still actively referenced "
    "or needed for upcoming work.\n"
    "- When no `[img:N]` tags are present, ignore this section."
)

COMPRESSION_MULTI_PASS_ADDENDUM = (
    "\n\n## Multi-Pass Compression\n"
    "Some entries are from prior compression passes (already compressed once or "
    "more). You have a `get_raw` tool to retrieve the original uncompressed "
    "content of any entry.\n"
    "\n"
    "- Prior entries may still be verbose or contain information now irrelevant "
    "given later context.\n"
    "- Use `get_raw(index)` to inspect the original before deciding whether to "
    "re-compress.\n"
    "- You can `update` any entry (prior or new) using the same transformation "
    "rules.\n"
    "- If an old compression lost important detail, use `get_raw` to recover it "
    "and write a better summary."
)


# Returned by the loop in place of a result when compression is requested.
_COMPRESSION_SIGNAL = object()


def compress_context() -> str:
    """Compress the conversation history to free up context window space.

    This tool is always visible. When the context window is nearly full, this
    becomes the **only** available tool and you **must** call it.

    Cannot be called while other tools are in-flight -- finish or stop all
    running tools first.
    """
    return "compression acknowledged"


def tag_images_in_messages(
    messages: list[dict],
    start_id: int = 0,
) -> tuple[list[dict], dict[int, dict], int]:
    """Replace every ``image``/``image_url`` content block with a text block
    holding a unique ``[img:N]`` tag; other blocks and plain-string messages
    pass through unchanged.

    Returns the tagged copy of *messages*, the registry mapping each
    ``img_id`` to its original block, and the next free id.
    """
    result: list[dict] = []
    registry: dict[int, dict] = {}
    current_id = start_id

    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            result.append(msg)
            continue

        new_blocks: list[dict] = []
        for block in content:
            btype = block.get("type", "")
            if btype in ("image", "image_url"):
                registry[current_id] = block
                new_blocks.append(
                    {"type": "text", "text": f"[img:{current_id}]"},
                )
                current_id += 1
            else:
                new_blocks.append(block)

        new_msg = {k: v for k, v in msg.items() if k != "content"}
        new_msg["content"] = new_blocks
        result.append(new_msg)

    return result, registry, current_id


_IMG_TAG_RE = re.compile(r"\[img:(\d+)\]")


def _scan_surviving_image_ids(entries: dict[int, str]) -> set[int]:
    """Scan compressed entries for ``[img:N]`` tags and return surviving IDs."""
    ids: set[int] = set()
    for content in entries.values():
        ids.update(int(m) for m in _IMG_TAG_RE.findall(content))
    return ids


class CompressedMessage(BaseModel):
    content: str


class CompressedMessages(BaseModel):
    messages: list[CompressedMessage]
    surviving_image_ids: set[int] = set()


_SANDBOX_GLOBALS = None


def _get_sandbox_globals() -> dict:
    """Sandbox globals for transformation exec: ``create_base_globals()``
    plus ``re`` and ``json``, built once."""
    global _SANDBOX_GLOBALS
    if _SANDBOX_GLOBALS is None:
        g = create_base_globals()
        g["re"] = re
        g["json"] = json
        _SANDBOX_GLOBALS = g
    return _SANDBOX_GLOBALS


def _eval_transformation(transformation_str: str, content: str) -> str:
    """Execute transformation code that receives the content as ``x`` and
    leaves the result in ``x``. Raises on syntax or runtime errors; the
    caller reports them to the LLM.
    """
    sandbox = dict(_get_sandbox_globals())
    sandbox["x"] = content

    code = transformation_str.strip()

    # A bare expression such as ``x.replace("old", "new")`` is wrapped as
    # ``x = <expr>``; multi-line code or code containing ``=`` runs as-is
    # and must assign ``x`` itself.
    if "\n" not in code and "=" not in code:
        try:
            compile(code, "<transformation>", "eval")
            code = f"x = {code}"
        except SyntaxError:
            pass

    exec(code, sandbox)  # noqa: S102
    result = sandbox["x"]
    if not isinstance(result, str):
        result = str(result)
    return result


def _compute_token_usage(entries: dict[int, str], endpoint: str) -> str:
    """Compute token usage of all entries as a percentage of the context window."""
    total_text = "\n".join(entries.values())
    tokens = count_tokens(total_text)
    max_input = unillm.get_max_input_tokens(endpoint) or 0
    if max_input > 0:
        pct = tokens / max_input * 100
        return f"[{tokens:,} tokens | {pct:.1f}% of context window]"
    return f"[{tokens:,} tokens]"


def _make_update_tool(entries: dict[int, str], endpoint: str) -> callable:

    def update(index: int, transformation: str) -> str:
        """Transform a message to compress it in-place.

        Args:
            index: The [N] index of the message to transform.
            transformation: Python code that transforms the message.
                The variable ``x`` holds the full JSON string of the
                message. The final value of ``x`` after execution becomes
                the new entry.

                String operations (work directly on the JSON string):
                  x = x.replace("verbose section", "summary")
                  x = re.sub(r"Traceback[\\s\\S]*", "omitted", x)
                  x = x[:500]

                Structured operations (parse, modify, serialize):
                  msg = json.loads(x)
                  msg["content"] = "error: timeout"
                  x = json.dumps(msg)

                A bare expression (no ``=``) is auto-assigned to ``x``:
                  x.replace("verbose", "short")

                ``re`` and ``json`` are available.

        Returns:
            The new entry after transformation, followed by current
            token usage of the compressed context.
        """
        if index not in entries:
            valid = sorted(entries.keys())
            return f"Error: index {index} not found. Valid indices: {valid}"
        try:
            new_content = _eval_transformation(
                transformation,
                entries[index],
            )
        except Exception as exc:
            return f"Error applying transformation: {type(exc).__name__}: {exc}"
        if not new_content:
            new_content = "(empty)"
        entries[index] = new_content
        usage = _compute_token_usage(entries, endpoint)
        return f"{new_content}\n{usage}"

    return update


def _make_archive_lookup_tool(
    raw_archives: list[list[dict]],
    *,
    for_compression: bool = False,
) -> callable:
    """Archive-lookup tool over the raw message archives: ``get_raw`` for
    the compression sub-loop when *for_compression*, else ``unpack_messages``
    for the restarted main loop."""

    def _lookup(index: int, n: int = 1) -> str:
        flat = [msg for archive in raw_archives for msg in archive]
        if index < 0 or index >= len(flat):
            return json.dumps(
                {"error": f"Index {index} out of range (0-{len(flat) - 1})"},
            )
        end = min(index + n, len(flat))
        return json.dumps(flat[index:end], default=str)

    if for_compression:
        _lookup.__name__ = "get_raw"
        _lookup.__doc__ = (
            "Retrieve the original uncompressed content of a message.\n\n"
            "Use this to inspect what a previously-compressed entry looked "
            "like before compression.  Helpful when re-evaluating whether "
            "an old compression decision was too aggressive or when the "
            "conversation focus has shifted.\n\n"
            "Args:\n"
            "    index: The [N] index of the message.\n"
            "    n: Number of consecutive messages to retrieve (default 1)."
        )
    else:
        _lookup.__name__ = "unpack_messages"
        _lookup.__doc__ = (
            "Retrieve one or more uncompressed messages by index.\n\n"
            "``index`` corresponds to the ``[N]`` label in the compressed "
            "context summary.  Returns up to ``n`` consecutive original "
            "messages as a JSON array.\n\n"
            "Args:\n"
            "    index: The [N] index of the message.\n"
            "    n: Number of consecutive messages to retrieve (default 1)."
        )
    return _lookup


async def compress_messages(
    messages: list[dict],
    endpoint: str,
    *,
    image_blocks: dict[int, dict] | None = None,
    prior_entries: list[tuple[int, str]] | None = None,
    raw_archives: list[list[dict]] | None = None,
    new_indices: list[int] | None = None,
) -> CompressedMessages:
    """Compact a message list using a tool-loop that surgically edits entries.

    Messages should already have images replaced by ``[img:N]`` tags via
    ``tag_images_in_messages``.  When ``image_blocks`` is provided, the
    tagged image content blocks are delivered inline as multimodal content
    so the compression LLM can reason about which images to keep or remove.

    For multi-pass compression, ``prior_entries`` provides already-compressed
    entries from previous passes (as ``(global_index, content)`` tuples),
    ``raw_archives`` enables the ``get_raw`` tool for peeking at originals,
    and ``new_indices`` maps each message to its global archive index.
    """
    from unify.common.async_tool_loop import start_async_tool_loop

    if new_indices is not None and len(new_indices) != len(messages):
        raise ValueError(
            f"new_indices length ({len(new_indices)}) must match "
            f"messages length ({len(messages)})",
        )

    entries: dict[int, str] = {}
    if prior_entries:
        for idx, content in prior_entries:
            entries[idx] = content

    indices = new_indices if new_indices is not None else list(range(len(messages)))
    for i, msg in enumerate(messages):
        entries[indices[i]] = json.dumps(msg, default=str)

    serialized = "\n\n".join(f"[{idx}] {e}" for idx, e in sorted(entries.items()))

    update_tool = _make_update_tool(entries, endpoint)
    tools: dict[str, callable] = {"update": update_tool}

    if prior_entries and raw_archives:
        tools["get_raw"] = _make_archive_lookup_tool(
            raw_archives,
            for_compression=True,
        )

    initial_usage = _compute_token_usage(entries, endpoint)

    if prior_entries:
        prior_indices = sorted(idx for idx, _ in prior_entries)
        new_msg_indices = sorted(indices)
        text_prompt = (
            f"Compress the following transcript. "
            f"Entries [{prior_indices[0]}]-[{prior_indices[-1]}] are from "
            f"prior passes (already compressed). "
            f"Entries [{new_msg_indices[0]}]-[{new_msg_indices[-1]}] are "
            f"from the current session.\n"
            f"Focus on the largest entries first.\n"
            f"Current usage: {initial_usage}\n\n"
            f"{serialized}"
        )
    else:
        text_prompt = (
            f"Compress the following {len(messages)} message transcript. "
            f"Focus on the largest entries first.\n"
            f"Current usage: {initial_usage}\n\n"
            f"{serialized}"
        )

    if image_blocks:
        content_blocks: list[dict] = [{"type": "text", "text": text_prompt}]
        for img_id in sorted(image_blocks):
            content_blocks.append(
                {"type": "text", "text": f"--- [img:{img_id}] ---"},
            )
            content_blocks.append(image_blocks[img_id])
        user_prompt: str | list[dict] = content_blocks
    else:
        user_prompt = text_prompt

    sys_prompt = COMPRESSION_PROMPT
    if prior_entries:
        sys_prompt += COMPRESSION_MULTI_PASS_ADDENDUM

    client = new_llm_client(endpoint, origin="compress_messages")
    client.set_system_message(sys_prompt)

    handle = start_async_tool_loop(
        client,
        user_prompt,
        tools,
        loop_id="compress_messages",
        timeout=120,
        log_steps=True,
        enable_compression=False,
    )

    await handle.result()

    surviving = _scan_surviving_image_ids(entries) if image_blocks else set()

    return CompressedMessages(
        messages=[
            CompressedMessage(content=entries[idx]) for idx in sorted(entries.keys())
        ],
        surviving_image_ids=surviving,
    )


async def compress_and_rebuild(
    state: CompressionState,
    all_messages: list[dict],
    endpoint: str,
    original_tools: dict[str, callable],
) -> RebuildResult:
    """Archive messages, compress, and prepare everything for a loop restart.

    Mutates *state* in place (archives, entries, image registry, counters).
    Returns the rebuilt system messages and augmented tools dict needed to
    start a new loop iteration.
    """
    # With UNIFY_TRANSCRIPTS on, the session whose history this is gets
    # everything not yet on disk before that history is replaced.
    session = transcripts.session_for_messages(all_messages)
    if session is not None:
        session.sync()
    all_messages = copy.deepcopy(all_messages)
    state.raw_archives.append(all_messages)
    archive_base = sum(len(a) for a in state.raw_archives[:-1])

    # The compressed-context system message is already represented by the
    # prior entries; every other message gets a global index so numbering
    # stays continuous across passes.
    new_messages: list[dict] = []
    new_msg_global_indices: list[int] = []
    for i, msg in enumerate(all_messages):
        if msg.get("_compressed_message"):
            continue
        new_messages.append(msg)
        new_msg_global_indices.append(archive_base + i)

    tagged_messages, new_image_blocks, next_id = tag_images_in_messages(
        new_messages,
        start_id=state.next_image_id,
    )
    state.next_image_id = next_id
    state.image_registry.update(new_image_blocks)
    state.live_image_ids.update(new_image_blocks.keys())

    live_images = (
        {iid: state.image_registry[iid] for iid in state.live_image_ids}
        if state.live_image_ids
        else None
    )

    # The compressor's own loop is a child of the session it compacts.
    with transcripts.as_current(session):
        compressed = await compress_messages(
            tagged_messages,
            endpoint,
            image_blocks=live_images,
            prior_entries=state.entries or None,
            raw_archives=state.raw_archives,
            new_indices=new_msg_global_indices,
        )

    if live_images:
        state.live_image_ids = compressed.surviving_image_ids

    # The first n_prior results are the re-compressed prior entries.
    n_prior = len(state.entries)
    prior_results = compressed.messages[:n_prior]
    new_results = compressed.messages[n_prior:]

    conversation_entries: list[tuple[int, str]] = []
    system_msgs: list[dict] = []

    for (orig_idx, _), comp in zip(state.entries, prior_results):
        conversation_entries.append((orig_idx, comp.content))

    for global_idx, orig_msg, comp in zip(
        new_msg_global_indices,
        new_messages,
        new_results,
    ):
        if orig_msg.get("role") == "system":
            new_msg = dict(orig_msg)
            try:
                compressed_dict = json.loads(comp.content)
                new_msg["content"] = compressed_dict.get(
                    "content",
                    comp.content,
                )
            except (json.JSONDecodeError, TypeError):
                new_msg["content"] = comp.content
            system_msgs.append(new_msg)
        else:
            conversation_entries.append((global_idx, comp.content))

    state.entries = conversation_entries

    body = "\n".join(f"[{idx}] {content}" for idx, content in state.entries)
    combined = _COMPRESSED_HEADER + body

    _instructions = (
        "When you need details from a compressed message, call "
        "`unpack_messages(index)` with its `[N]` index to "
        "retrieve the full original content. Pass `n` to "
        "retrieve a range of consecutive messages."
    )
    if session is not None:
        _instructions += "\n\n" + session.pointer_line()

    if state.live_image_ids:
        content_blocks: list[dict] = [
            {"type": "text", "text": f"{combined}\n\n{_instructions}"},
        ]
        for img_id in sorted(state.live_image_ids):
            if img_id in state.image_registry:
                content_blocks.append(
                    {"type": "text", "text": f"[img:{img_id}]"},
                )
                content_blocks.append(state.image_registry[img_id])
        compressed_content: str | list[dict] = content_blocks
    else:
        compressed_content = f"{combined}\n\n{_instructions}"

    system_msgs.append(
        {
            "role": "system",
            "_compressed_message": True,
            "content": compressed_content,
        },
    )

    tools = dict(original_tools)
    tools["unpack_messages"] = _make_archive_lookup_tool(state.raw_archives)

    state.count += 1
    transcripts.record_compaction(
        session,
        archived=len(all_messages),
        pass_number=state.count,
        context=system_msgs,
    )

    return RebuildResult(system_msgs=system_msgs, tools=tools)

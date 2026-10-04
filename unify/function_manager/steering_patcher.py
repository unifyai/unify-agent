"""Turning a correction into a code change, without leaving the execution engine.

When someone corrects work that is already running, something has to decide
what the correction means for the code. That decision is made here rather than
by the actor, for two reasons: the block is suspended while it happens, so the
round trip through an outer tool loop is latency the correction cannot afford;
and the decision is about *this source*, which the actor does not have in front
of it.

The output is deliberately narrow. A correction may rewrite one or more of the
functions the block defines and may declare which cached calls are now stale,
or it may stop the remaining work when continuing would contradict the
correction. It may not do anything else — not call tools, not reach outside the
block, not touch anything the block did not define. A correction is a decision
about work in progress, not a new instruction.
"""

from __future__ import annotations

import ast
import json
import logging
from typing import Any, List, Optional, Sequence

from .steering import InterruptionRequest, Patch, SteeringSession

logger = logging.getLogger(__name__)

_SYSTEM = """\
You are correcting a code block that is running right now. It has been \
suspended mid-execution so your edit can be applied.

You are given a message that arrived while it runs, the source, which of its \
functions you may rewrite, and which calls have already completed. Decide what \
the message means for the code that has not run yet.

Not every message is a correction. A question (how long it takes, what it is \
doing, how far it has got), an acknowledgement or encouragement leaves the \
task exactly as it was: return false for `stop` and no patches. You cannot \
reply here — questions are answered elsewhere while the block keeps \
running — so stopping is never a way to answer one.

Rules that matter:

- Work already listed as completed HAS ALREADY HAPPENED. Sends have been sent. \
You cannot undo it. Write the remaining code as someone would who knows those \
steps are done.
- Rewrite whole functions. Return the complete new definition, not a diff or a \
fragment. Only functions the block itself defines can be rewritten.
- Rewriting a function does NOT repeat its completed calls: identical calls \
replay from a record instead of executing again. So a rewritten loop that \
still covers earlier items is safe.
- If the correction means an earlier call should now happen *differently*, \
name it in `invalidate` so its record is discarded and it runs again. Use this \
sparingly and never for something irreversible that already happened.
- Set `stop` to true only when the message revokes the task, or when it says \
remaining work would now be wrong and there is no listed function you can \
rewrite. A stop leaves completed work in place and abandons everything still \
pending.
- Completed calls are listed with the arguments they were given. Write \
rewritten code against the data as it actually is, as shown by those \
arguments and by how the block was called, not against a shape you assume.
- If the message does not change any remaining work, return false for \
`stop` and no patches.
- A stop and patches are mutually exclusive.

Respond with JSON only:

{"reason": "<short restatement of the message>",
 "stop": false,
 "patches": [{"function_name": "<name>",
              "source": "<complete def or async def>",
              "reason": "<what changed>",
              "invalidate": ["<tool path prefix>", ...]}]}
"""


def _defined_functions(source: str) -> List[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    return [
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]


def _build_user_prompt(
    *,
    source: str,
    interjections: Sequence[str],
    completed: Sequence[str],
    defined: Sequence[str],
    invocation: str = "",
) -> str:
    parts = [
        "The message:",
        *(f"  {text}" for text in interjections),
        "",
        "The running block:",
        "```",
        source,
        "```",
    ]
    if invocation:
        parts += ["", "It was called as:", f"  {invocation}"]
    parts += [
        "",
        f"Functions you may rewrite: {', '.join(defined) or '(none)'}",
    ]
    if completed:
        parts += [
            "",
            "Calls that have ALREADY COMPLETED (do not plan to repeat these):",
            *(f"  {call}" for call in completed),
        ]
    else:
        parts += ["", "No calls have completed yet."]
    return "\n".join(parts)


def _parse_decision(
    raw: str,
    *,
    defined: Sequence[str],
) -> Optional[InterruptionRequest]:
    """Read the model's JSON, discarding anything it was not allowed to say."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        text = text[len("json") :] if text.startswith("json") else text
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("steering: patch author returned non-JSON")
        return None

    stop = bool(payload.get("stop"))
    patches: List[Patch] = []
    for entry in payload.get("patches") or []:
        name = entry.get("function_name")
        source = entry.get("source") or ""
        # A patch naming a function this block does not define cannot be
        # spliced, and letting it through would burn a retry to discover that.
        if name not in defined:
            logger.warning("steering: patch author named unknown function %s", name)
            continue
        patches.append(
            Patch(
                function_name=str(name),
                source=str(source),
                reason=str(entry.get("reason") or ""),
                invalidate=tuple(entry.get("invalidate") or ()),
            ),
        )

    if stop:
        return InterruptionRequest(
            reason=str(payload.get("reason") or "stopped"),
            stop=True,
        )
    if not patches:
        return None
    return InterruptionRequest(
        reason=str(payload.get("reason") or "steered"),
        patches=patches,
    )


class LLMPatchAuthor:
    """Decides what a correction means for a running block.

    Held by the execution engine and given the source of whatever it is
    currently running. Returning ``None`` means the correction changes nothing
    about the remaining work, which leaves the block to continue untouched
    rather than paying for a retry that would produce identical code.
    """

    def __init__(self, *, client_factory: Any) -> None:
        self._client_factory = client_factory

    async def __call__(
        self,
        *,
        interjections: Sequence[str],
        session: SteeringSession,
    ) -> Optional[InterruptionRequest]:
        source = session.source
        if not source:
            return None
        defined = _defined_functions(source)

        prompt = _build_user_prompt(
            source=source,
            interjections=interjections,
            completed=session.cache.completed_calls(),
            defined=defined,
            invocation=session.invocation,
        )
        client = self._client_factory()
        raw = await client.generate(user_message=prompt, system_message=_SYSTEM)
        return _parse_decision(str(raw), defined=defined)


def build_patch_author(*, model: Optional[str] = None) -> LLMPatchAuthor:
    """A patch author backed by the project's usual LLM client."""

    def _factory() -> Any:
        from unify.common.llm_client import new_llm_client

        return new_llm_client(model, async_client=True, origin="steering_patch")

    return LLMPatchAuthor(client_factory=_factory)

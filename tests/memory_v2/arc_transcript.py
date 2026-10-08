"""A Continual-ARC visit as the memory v2 dialogue adapter reads it: the transcript lines of one session.

The texts follow the benchmark's text protocol (continual-arc-baselines ``systems/protocol.py`` at
99605e6: ``RULES_FINISH``, ``render_observation``, ``render_feedback``, ``render_pairs``) and the agent
record's delivery block (``unify/agents/delivery.py`` ``render_block``): the visit's first message is the
request; every later message the runner sends (``{"message": ...}`` on stdin) reaches the actor as a record
block. The agent's action is a JSON object on the last line of its reply. The adapter reads none of these
words; the tests check that it records structure only. Stdlib only.
"""

from __future__ import annotations

import json
from typing import Any

RULES = """\
You are being evaluated on Continual-ARC, a lifelong stream of ARC-style grid puzzles.
Each puzzle instance has a task id and a test input grid. Tasks recur across the
stream with fresh inputs. Each instance is a separate conversation: the next one
starts without this one's messages.

On every message you must choose exactly one action and put it as a JSON object
on the last line of your reply:
  {"action": "submit", "grid": [[0, 1, 2], [3, 4, 5]]}   submit an output grid
  {"action": "request_demos"}                            ask for one more demonstration pair for this task
  {"action": "finish"}                                   end the instance, once it is over
Grids are lists of rows; cells are integers 0-9; dimensions are 1-30.

Each demo request returns one more input-output pair of the task, up to the demo cap shown
with the input.

Scoring: every wrong submission costs 1 and every demo request costs 1. An
instance is over when a submission is correct or when the attempt cap is reached,
and reaching the cap is charged the full cap. Feedback on a submission is only
correct/incorrect. There is no way to abandon an instance, and finish is refused until
the instance is over. Each instance is scored on its own, and lower cost is better.
"""


def grid(rows: int, cols: int, seed: int = 0) -> list[list[int]]:
    return [[(r * 7 + c * 3 + seed) % 10 for c in range(cols)] for r in range(rows)]


def render_grid(g: list[list[int]]) -> str:
    return "\n".join(" ".join(str(c) for c in row) for row in g)


def _shape(g: list[list[int]]) -> str:
    return f"{len(g)}x{len(g[0]) if g else 0}"


def render_pairs(pairs: list[tuple[list, list]], start: int = 1) -> str:
    parts = []
    for k, (inp, out) in enumerate(pairs, start):
        parts.append(
            f"Demo pair {k} input ({_shape(inp)}):\n{render_grid(inp)}\n"
            f"Demo pair {k} output ({_shape(out)}):\n{render_grid(out)}",
        )
    return "\n\n".join(parts)


def demos_feedback(pairs: list[tuple[list, list]], used: int) -> str:
    return f"Demos received (request {used}):\n\n{render_pairs(pairs, start=used)}"


def submit_feedback(correct: bool, attempts: int, failed: bool = False) -> str:
    if correct:
        return f"Your submission was CORRECT. Instance solved with {attempts} wrong attempt(s)."
    tail = " The attempt cap is reached; the instance is over." if failed else ""
    return f"Your submission was INCORRECT. Wrong attempts so far: {attempts}.{tail}"


def observation(
    task: str,
    test_input: list[list[int]],
    *,
    attempts: int = 0,
    demos: int = 0,
    attempt_cap: int = 3,
    demo_cap: int = 3,
    pending: tuple[str, ...] = (),
    first: bool = False,
) -> str:
    lines: list[str] = []
    if first:
        lines += [RULES, ""]
    if pending:
        lines.append("Feedback since your last action:")
        lines.extend(pending)
        lines.append("")
    fresh = attempts == 0 and demos == 0
    lines.append(f"{'New instance' if fresh else 'Same instance'}. Task id: {task}")
    lines.append(f"Test input ({_shape(test_input)}):")
    lines.append(render_grid(test_input))
    lines.append(
        f"Wrong attempts used: {attempts}/{attempt_cap}. Demo requests used: {demos}/{demo_cap}.",
    )
    lines.append("Reply with exactly one action JSON on the last line.")
    return "\n".join(lines)


def block(seq: int, text: str) -> str:
    """A runner message as the record delivers it to the main agent."""
    return f"[record · 1 new for you · #{seq}]\n#{seq} user: {text}"


def line(seq: int, role: str, content: Any = "", **extra: Any) -> dict:
    """One ``message`` line of a session transcript (unify/transcripts.py)."""
    return {
        "seq": seq,
        "ts": f"2026-10-08T12:{seq // 60:02d}:{seq % 60:02d}+00:00",
        "session": "s",
        "type": "message",
        "loop": "",
        "in_context": True,
        "message": {"role": role, "content": content, **extra},
    }


def code_call(seq: int, call_id: str, code: str) -> dict:
    return line(
        seq,
        "assistant",
        "",
        tool_calls=[
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": "execute_code",
                    "arguments": json.dumps({"code": code}),
                },
            },
        ],
    )


def action_reply(thought: str, action: dict, *, fenced: bool = False) -> str:
    body = json.dumps(action)
    return f"{thought}\n```json\n{body}\n```" if fenced else f"{thought}\n{body}"


def visit(
    *,
    task: str = "task-3f2a9c1e",
    size: tuple[int, int] = (3, 3),
    pair_size: tuple[int, int] = (3, 3),
    seed: int = 0,
    structured: bool = False,
    wrapped: bool = True,
) -> dict:
    """One visit: request_demos, a cell, a wrong submit, a right submit, then finish (no answer).

    ``structured``: the counterpart answers with JSON objects (a typed counterpart) instead of the
    rendered text, so the observations' shape changes. ``wrapped``: the messages reach the actor as record
    blocks (as through the CLI); unwrapped, each is the whole user message. Returns ``{"lines", "messages", "grids",
    "observations"}``: the transcript lines, the runner messages (before the record wraps them), the two
    submitted grids, and the counterpart messages as the transcript holds them.
    """
    test_input = grid(*size, seed=seed)
    pair = (grid(*pair_size, seed=seed + 1), grid(*pair_size, seed=seed + 2))
    wrong, right = grid(*size, seed=seed + 3), grid(*size, seed=seed + 4)
    if structured:
        messages = [
            json.dumps(
                {
                    "type": "DemosFeedback",
                    "pairs": [list(pair)],
                    "refused": False,
                    "demo_requests_used": 1,
                },
            ),
            json.dumps(
                {
                    "type": "SubmitFeedback",
                    "correct": False,
                    "valid": True,
                    "attempts_used": 1,
                    "failed": False,
                },
            ),
            json.dumps(
                {
                    "type": "SubmitFeedback",
                    "correct": True,
                    "valid": True,
                    "attempts_used": 1,
                    "failed": False,
                },
            ),
        ]
    else:
        messages = [
            observation(
                task,
                test_input,
                demos=1,
                pending=(demos_feedback([pair], 1),),
            ),
            observation(
                task,
                test_input,
                attempts=1,
                demos=1,
                pending=(submit_feedback(False, 1),),
            ),
            observation(
                task,
                test_input,
                attempts=1,
                demos=1,
                pending=(submit_feedback(True, 1),),
            ),
        ]
    observations = (
        [block(3, messages[0]), block(8, messages[1]), block(10, messages[2])]
        if wrapped
        else list(messages)
    )
    lines = [
        line(0, "user", observation(task, test_input, first=True)),
        line(
            1,
            "assistant",
            action_reply(
                "No demonstration yet; I ask for one.",
                {"action": "request_demos"},
            ),
        ),
        line(2, "user", "Working on it...", _loop_authored=True, _progress_msg=True),
        line(3, "user", observations[0]),
        code_call(4, "c1", "print('rows', 3)"),
        line(
            5,
            "tool",
            "--- stdout ---\nrows 3\n",
            tool_call_id="c1",
            name="execute_code",
        ),
        line(
            6,
            "assistant",
            action_reply(
                "Each cell shifts by one.",
                {"action": "submit", "grid": wrong},
                fenced=True,
            ),
        ),
        {
            "seq": 7,
            "type": "message_update",
            "message": {"role": "assistant", "content": "x"},
        },
        line(8, "user", observations[1]),
        line(
            9,
            "assistant",
            action_reply("Second try.", {"action": "submit", "grid": right}),
        ),
        line(10, "user", observations[2]),
        line(11, "assistant", json.dumps({"action": "finish"})),
        {"seq": 12, "type": "outcome", "accepted": True, "solved": True},
    ]
    return {
        "lines": lines,
        "messages": messages,
        "grids": (wrong, right),
        "observations": observations,
    }


# --- a scripted Sol's item on the dialogue channel ----------------------------------------------------

ITEM = "env/env:observation_lines"
ITEM_MODULE = '''__all__ = ["observation_lines"]


def observation_lines(observation):
    """Split one counterpart observation into its non-empty lines; raise on another input.

    Effect: read
    Input: text
    """
    if not isinstance(observation, str):
        raise ValueError("expected the observation as text")
    return [line for line in observation.splitlines() if line.strip()]
'''
ITEM_TEST = """import json
import pathlib

from env.env import observation_lines

REC = pathlib.Path(__file__).parent / "rec"


def test_every_recorded_observation_has_lines():
    observations = json.loads((REC / "observations.json").read_text())
    assert observations
    for observation in observations:
        lines = observation_lines(observation)
        assert lines and all(line.strip() for line in lines)
"""
# One Sol cell: read the exported episodes, cover every recorded dialogue observation on the memory
# channel ``env``, write the item, its test and fixture, and the manifest.
SOL_CELL = f"""import json, pathlib
covers, observations, sources = [], [], []
for f in sorted(pathlib.Path("/inputs/episodes").glob("*.json")):
    ep = json.loads(f.read_text())
    sources.append(ep["episode_id"])
    for a, ch in zip(ep["actions"], ep["memory_channels"]):
        if ch == "env" and a["kind"] == "dialogue" and a["status"] == "ok" and isinstance(a["response"], str):
            covers.append([ep["episode_id"], a["index"]])
            observations.append(a["response"])
root = pathlib.Path("/memory/env/env")
(root / "tests" / "rec").mkdir(parents=True, exist_ok=True)
(root / "__init__.py").write_text({ITEM_MODULE!r})
(root / "tests" / "test_observation_lines.py").write_text({ITEM_TEST!r})
(root / "tests" / "rec" / "observations.json").write_text(json.dumps(observations))
pathlib.Path("/memory/.pass").mkdir(parents=True, exist_ok=True)
manifest = {{
    "items": [{{
        "item": {ITEM!r},
        "kind": "env_function",
        "input": "text",
        "source_episodes": sources,
        "tests": ["env/env/tests/test_observation_lines.py"],
        "covers": covers,
    }}],
    "support": ["env/env/tests/rec/observations.json"],
    "summary": "split env observations into lines",
}}
pathlib.Path("/memory/.pass/manifest.json").write_text(json.dumps(manifest))
print("COVERS", len(covers), "EPISODES", len(sources))
"""

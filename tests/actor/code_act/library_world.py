"""The world the end-to-end library tests run in, and their helpers.

* A hierarchy of stored functions: an entry point composed of three helpers,
  each a function of its own, written and stored from code.
* A deterministic embedder (hashed bag of words), so a library search ranks
  by shared words without a model or the network.
* :class:`Cells`: cells of one ``UNIFY_TOOL_SURFACE=core`` session, run
  through the actor's own ``execute_code`` tool in the real sandboxed worker.
* :func:`run_in_another_process`: the same cells in a separate actor process
  on the same ``UNIFY_HOME`` store, which exits before the test goes on, so
  what a later session finds came through the store file and nothing else.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

REPO = Path(__file__).resolve().parents[3]

# ── the hierarchy ───────────────────────────────────────────────────────────

PARSE = (
    "def parse_pairs(text: str) -> list:\n"
    '    """Parse comma-separated key=value pairs into [key, int] pairs."""\n'
    "    pairs = []\n"
    "    for part in text.split(','):\n"
    "        part = part.strip()\n"
    "        if not part:\n"
    "            continue\n"
    "        key, _, value = part.partition('=')\n"
    "        pairs.append([key.strip(), int(value)])\n"
    "    return pairs\n"
)
TOTAL = (
    "def total_by_key(pairs: list) -> dict:\n"
    '    """Sum the values of [key, value] pairs per key."""\n'
    "    totals = {}\n"
    "    for key, value in pairs:\n"
    "        totals[key] = totals.get(key, 0) + value\n"
    "    return totals\n"
)
FORMAT = (
    "def format_totals(totals: dict, sep: str = '; ') -> str:\n"
    '    """Render per-key totals as key: value, sorted by key."""\n'
    "    return sep.join(f'{k}: {totals[k]}' for k in sorted(totals))\n"
)
ENTRY = (
    "def summarize_pairs(text: str, sep: str = '; ') -> str:\n"
    '    """Summarise comma-separated key=value pairs as per-key totals.\n'
    "\n"
    "    Composes parse_pairs, total_by_key and format_totals.\n"
    '    """\n'
    "    return format_totals(total_by_key(parse_pairs(text)), sep=sep)\n"
)
HELPERS = ("parse_pairs", "total_by_key", "format_totals")
HIERARCHY = (PARSE, TOTAL, FORMAT, ENTRY)
TEXT = "a=1, b=2, a=3"
SUMMARY = "a: 4; b: 2"

GUIDANCE_TITLE = "Summarising key=value pairs"
GUIDANCE_CONTENT = (
    "Call summarize_pairs(text) on comma-separated key=value pairs; it parses "
    "them (parse_pairs), totals them per key (total_by_key) and renders the "
    "totals (format_totals). Pass sep to change the separator."
)


# ── a deterministic embedder ────────────────────────────────────────────────

DIM = 512


def fake_embed(texts: Any) -> np.ndarray:
    """Unit vectors of hashed lower-case words: texts that share words are close."""
    rows = []
    for text in texts:
        vector = np.zeros(DIM, dtype=np.float32)
        for word in re.findall(r"[a-z0-9]+", str(text).lower()):
            digest = hashlib.md5(word.encode()).hexdigest()
            vector[int(digest, 16) % DIM] += 1.0
        if not vector.any():
            vector[0] = 1.0
        rows.append(vector / np.linalg.norm(vector))
    return np.stack(rows) if rows else np.zeros((0, DIM), dtype=np.float32)


def install_fake_embed(setattr_: Any = setattr) -> None:
    """Route every embedding through :func:`fake_embed` (``monkeypatch.setattr`` in tests)."""
    import unify.common.embeddings as embeddings
    import unify.common.semantic_search as semantic_search

    setattr_(embeddings, "embed", fake_embed)
    setattr_(semantic_search, "embed", fake_embed)


# ── cells of one core session ───────────────────────────────────────────────


def new_actor(**kwargs: Any):
    """A CodeActActor over the store's function and guidance libraries."""
    from unify.actor.code_act_actor import CodeActActor
    from unify.function_manager.function_manager import FunctionManager
    from unify.guidance_manager.guidance_manager import GuidanceManager

    kwargs.setdefault("environments", [])
    kwargs.setdefault("function_manager", FunctionManager(include_primitives=False))
    kwargs.setdefault("guidance_manager", GuidanceManager())
    return CodeActActor(**kwargs)


class Cells:
    """Cells of one core session, run through the actor's ``execute_code`` tool."""

    def __init__(self, actor: Any, policy: Any = None, extra: Optional[dict] = None):
        from unify.actor import core_surface
        from unify.actor.execution import PythonExecutionSession, _CURRENT_SANDBOX

        self.actor = actor
        self.tools = actor.get_tools("act")
        self.sandbox = PythonExecutionSession(environments={})
        self.sandbox.global_state.update(extra or {})
        objects = core_surface.sandbox_objects(
            actor,
            policy=policy or core_surface.WritePolicy(),
        )
        self.sandbox.global_state.update(objects)
        self.sandbox.core_globals = objects
        self._token = _CURRENT_SANDBOX.set(self.sandbox)

    async def __call__(self, code: str, **kwargs: Any) -> Any:
        return await self.tools["execute_code"].fn(
            thought="A step.",
            code=code,
            **kwargs,
        )

    async def close(self) -> None:
        from unify.actor.execution import _CURRENT_SANDBOX

        try:
            _CURRENT_SANDBOX.reset(self._token)
        except ValueError:
            pass
        await self.sandbox.close()
        await self.actor.close()


def stdout(out: Any) -> str:
    from unify.actor.execution.types import parts_to_text

    return (
        parts_to_text(out.stdout) if isinstance(out.stdout, list) else str(out.stdout)
    )


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return repr(value)


# ── a session in another process ────────────────────────────────────────────

#: The switches a separate session runs under, as environment variables.
CORE_ENV = {
    "UNIFY_WORKSPACE": "sandboxed",
    "UNIFY_WORKSPACE_PYTHON": "worker",
    "UNIFY_TOOL_SURFACE": "core",
    "UNIFY_DISCOVERY_GATE": "false",
    "UNIFY_VALIDATE_LLM_PROVIDERS": "false",
    "UNIFY_FUNCTION_CASES": "true",
    "UNIFY_STORE_TRUST": "ramp",
}


def run_in_another_process(
    cells: List[str],
    *,
    request: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
    timeout: float = 240,
) -> List[dict]:
    """Run *cells* in one core session of a new Python process on this ``UNIFY_HOME``.

    Returns, per cell, its ``result`` (as JSON, else its repr), ``error`` and
    ``stdout``. The process has exited when this returns.
    """
    child_env = {**os.environ, **CORE_ENV, **(env or {})}
    child_env.pop("UNIFY_STORE_PATH", None)
    child_env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO), *filter(None, [os.environ.get("PYTHONPATH")])],
    )
    workspace = Path(child_env["UNIFY_HOME"]) / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [sys.executable, "-m", "tests.actor.code_act.library_world"],
        input=json.dumps({"cells": cells, "request": request}),
        capture_output=True,
        text=True,
        env=child_env,
        cwd=str(workspace),
        timeout=timeout,
    )
    if proc.returncode != 0:
        raise AssertionError(
            f"the session process failed ({proc.returncode}):\n{proc.stderr[-4000:]}",
        )
    lines = [l for l in proc.stdout.splitlines() if l.startswith("{")]
    return json.loads(lines[-1])["cells"]


async def _session(spec: dict) -> dict:
    from unify.function_manager import task_origin

    install_fake_embed()
    token = task_origin.enter(spec["request"]) if spec.get("request") else None
    cells = Cells(new_actor())
    outs = []
    try:
        for code in spec["cells"]:
            out = await cells(code)
            outs.append(
                {
                    "result": _jsonable(out.result),
                    "error": out.error,
                    "stdout": stdout(out),
                },
            )
    finally:
        await cells.close()
        if token is not None:
            task_origin.leave(token)
    return {"cells": outs, "pid": os.getpid()}


def main() -> None:
    spec = json.loads(sys.stdin.read())
    result = asyncio.run(_session(spec))
    sys.stdout.write("\n" + json.dumps(result) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()

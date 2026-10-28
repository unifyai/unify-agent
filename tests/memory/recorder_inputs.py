"""The library switch's off-path inputs: one deterministic over-the-cap recording per recorder (dialogue, shell, tool,
worktree), using only the recorders' default arguments, as they exist at 4675a3c45.

:func:`off_records` is run once at 4675a3c45 (on a keyless worker) to write ``golden/recorder_cuts_off_4675a3c45.json``;
``test_recorders_library_mode.py`` runs it again on this build and requires the same bytes, so with ``UNIFY_MEMORY_LIBRARY`` off
every recorder cuts exactly as at 4675a3c45. Standard library plus unify only; no model, no network.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from tests.memory import store_reads

#: A key-shaped string the redactor knows (planted past every cap in the library_mode tests).
KEY = "sk-or-v1-" + "ab" * 32


def big_text(n: int, tag: str) -> str:
    """*n* characters of deterministic text, with *tag* at its very end (so a cut drops it)."""
    line = "x" * 63 + "\n"
    body = (line * (n // 64 + 1))[: n - len(tag)]
    return body + tag


@dataclass(frozen=True)
class EnvCall:
    namespace: str
    method: str
    effect: str
    args: tuple
    kwargs: dict = field(default_factory=dict)
    via: str = "primitives"


def _line(seq: int, role: str, content: str) -> dict:
    return {
        "seq": seq,
        "ts": f"2026-10-09T03:00:{seq:02d}+00:00",
        "type": "message",
        "message": {"role": role, "content": content},
    }


def dialogue_lines() -> list[dict]:
    return [
        _line(0, "user", "go"),
        _line(
            1,
            "assistant",
            json.dumps({"action": "submit", "grid": big_text(20000, "PAYLOAD_END")}),
        ),
        _line(2, "user", big_text(70000, "OBSERVATION_END")),
    ]


def tool_call() -> tuple[EnvCall, dict]:
    return EnvCall("svc", "export", "read", ("all",), {}), {
        "rows": big_text(20000, "RESPONSE_END"),
    }


def worktree_file() -> str:
    return big_text(2 * 1024 * 1024 + 4096, "FILE_END")


def _setup_worktree(tmp: Path, **kw):
    from unify.memory.blobs import BlobStore
    from unify.memory.gitio import Repo
    from unify.memory.integration.adapters.worktree import WorkTreeRecorder

    wt = tmp / "work"
    wt.mkdir()
    (wt / "big.txt").write_text(worktree_file())
    repo = store_reads.init_snapshot(tmp / "worktree.git", wt)
    rec = WorkTreeRecorder(wt, repo, BlobStore(tmp / "blobs"), **kw)
    return wt, rec


def _rows(actions) -> list[dict]:
    return [dataclasses.asdict(a) for a in actions]


def off_records(tmp: Path) -> dict:
    """Each recorder's actions over the cap, with its default (4675a3c45) arguments, plus the blob ids the
    worktree recorder stored."""
    from unify.memory.integration.adapters.dialogue import dialogue_actions
    from unify.memory.integration.adapters.tool import RecordingObserver

    out: dict = {"dialogue": _rows(dialogue_actions(dialogue_lines(), "env"))}
    obs = RecordingObserver()
    call, result = tool_call()
    obs.before(call)
    obs.after(
        call,
        result=result,
        error=None,
        intercepted=False,
        started=0.0,
        elapsed_s=0.0,
    )
    out["tool"] = _rows(obs.drain().actions)
    wt, rec = _setup_worktree(Path(tmp))
    try:
        rec.begin()
        out["worktree"] = _rows(
            rec.record_cell(
                0,
                [{"event": "open", "path": "big.txt", "mode": "r", "cell": 0}],
            ),
        )
    finally:
        rec.close()  # its blob reader is a process
    out["worktree_blobs"] = sorted(
        p.name for p in (Path(tmp) / "blobs").rglob("*") if p.is_file()
    )
    return out


def dumps(records: dict) -> str:
    return json.dumps(records, sort_keys=True, default=str, indent=1) + "\n"


if (
    __name__ == "__main__"
):  # write the golden: python -m tests.memory.recorder_inputs OUT
    import sys
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        Path(sys.argv[1]).write_text(dumps(off_records(Path(d))))

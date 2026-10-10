"""P1 Task 7 (Amendment B; spec v2.1 P5): under UNIFY_MEMORY_V21 the recorders keep full values, redacted whole.

Off, each recorder cuts exactly as at 4675a3c45 (the golden was written there by :mod:`p1_t7_inputs`). On, each one
keeps the value over its old cap whole, and a key planted past that cap is redacted. RUNTIME's checklist points 1
and 5; the writer and loader (points 2–4, 6) are in test_episodes_v21.py."""

import dataclasses
import json
from pathlib import Path

from unify.memory_v2.integration.adapters.dialogue import dialogue_actions
from unify.memory_v2.integration.adapters.shell import shell_actions
from unify.memory_v2.integration.adapters.tool import RecordingObserver
from unify.memory_v2.redact import Redactor
from tests.memory_v2 import p1_t7_inputs as inp

GOLDEN = Path(__file__).parent / "golden" / "p1_t7_off_4675a3c45.json"
RED = Redactor(secrets={"OPENROUTER_API_KEY": inp.KEY})


def _planted(n: int, tag: str) -> str:
    """Text over the cap with the key planted past it (and *tag* at its end)."""
    return inp.big_text(n, inp.KEY + tag)


def test_off_records_equal_the_4675a3c45_golden(tmp_path):
    assert inp.dumps(inp.off_records(tmp_path)) == GOLDEN.read_text()


def test_dialogue_keeps_the_full_observation_and_payload(tmp_path):
    lines = inp.dialogue_lines()
    obs = _planted(70000, "OBSERVATION_END")
    lines[2]["message"]["content"] = obs
    (a,) = dialogue_actions(
        lines,
        "env",
        redactor=RED,
        max_observation_chars=None,
        max_payload_chars=None,
    )
    assert a.response == obs.replace(inp.KEY, RED.text(inp.KEY)) and len(
        a.response,
    ) >= 70000 - len(inp.KEY)
    assert inp.KEY not in json.dumps(dataclasses.asdict(a))
    assert a.method == "submit" and a.args == [inp.big_text(20000, "PAYLOAD_END")]


def test_tool_keeps_the_full_response():
    call, result = inp.tool_call()
    result = {"rows": _planted(20000, "RESPONSE_END")}
    obs = RecordingObserver(RED, max_value_bytes=None)
    obs.before(call)
    obs.after(
        call,
        result=result,
        error=None,
        intercepted=False,
        started=0.0,
        elapsed_s=0.0,
    )
    (a,) = obs.drain().actions
    assert (
        a.response["rows"].endswith("RESPONSE_END") and len(a.response["rows"]) > 19000
    )
    assert "__truncated__" not in json.dumps(a.response) and inp.KEY not in json.dumps(
        a.response,
    )


def test_shell_keeps_full_arguments():
    audit = inp.shell_audit()
    audit[0]["argv"][2] = _planted(9000, "ARG_END")
    (a,) = shell_actions(audit, redactor=RED, full=True)
    assert a.args[2].endswith("ARG_END") and "<truncated" not in a.args[2]
    assert inp.KEY not in json.dumps(a.args)


def test_worktree_stores_a_file_over_the_old_blob_cap_whole_and_redacted(tmp_path):
    from unify.memory_v2.blobs import BlobStore

    big = _planted(2 * 1024 * 1024 + 4096, "FILE_END")
    wt, rec = inp._setup_worktree(tmp_path, blob_cap=None, redactor=RED)
    (wt / "big.txt").write_text(big)
    rec.begin()
    (row,) = rec.record_cell(
        0,
        [{"event": "open", "path": "big.txt", "mode": "r", "cell": 0}],
    )
    blob = row.response["blob"]
    assert blob is not None
    data = BlobStore(tmp_path / "blobs").get(blob).decode()
    assert data.endswith("FILE_END") and len(data) > 2 * 1024 * 1024
    assert inp.KEY not in data
    assert not any(
        inp.KEY.encode() in p.read_bytes()
        for p in (tmp_path / "blobs").rglob("*")
        if p.is_file()
    )

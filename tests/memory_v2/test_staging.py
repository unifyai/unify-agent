"""Memory v2.1 r5 (§2-§4, §6): staging reaches the writer read-only as untrusted input; Sol analysts (arm C)."""

import asyncio
import json
import os

from unify.memory_v2 import analysts, staging
from unify.memory_v2.episodes import Cell
from unify.memory_v2.trigger import PassRequest
from tests.memory_v2.test_sol_pass import Turns, _call, _sol
from tests.memory_v2.test_sol_v21_tools import _episode


def _staged(root, eid, status="fork.json"):
    d = root / eid
    (d / "candidates").mkdir(parents=True)
    (d / "candidates" / "f.py").write_text("def f(x):\n    return x\n")
    if status:
        (d / status).write_text(json.dumps({"status": "ok"}))
    return d


def test_staging_export_refuses_paths_links_and_sizes(tmp_path):
    d = _staged(tmp_path, "e1")
    os.symlink("/etc/passwd", d / "candidates" / "link.py")
    (d / "evil.sh").write_text("rm -rf /")
    (d / "notes.md").write_bytes(b"x" * (staging.QUOTA_BYTES + 1))
    v = staging.read(tmp_path, "e1")
    assert (
        v.source == "fork" and v.status == "ok" and set(v.files) == {"candidates/f.py"}
    )
    assert {r["path"] for r in v.refused} == {
        "candidates/link.py",
        "evil.sh",
        "notes.md",
    }
    entry = staging.export(v, tmp_path / "out" / "e1")
    assert (
        entry["files"] == ["candidates/f.py"]
        and (tmp_path / "out" / "e1" / "candidates" / "f.py").is_file()
    )
    assert not (tmp_path / "out" / "e1" / "evil.sh").exists()


def test_a_staging_dir_without_its_status_file_is_still_being_written(tmp_path):
    _staged(tmp_path, "e2", status=None)
    assert staging.read(tmp_path, "e2") is None and staging.export(
        None,
        tmp_path / "x",
    ) == {"status": "none"}


def test_a_staging_dir_that_is_a_link_is_ignored(tmp_path):
    real = _staged(tmp_path / "elsewhere", "e3")
    (tmp_path / "root").mkdir()
    os.symlink(real, tmp_path / "root" / "e3")
    assert staging.read(tmp_path / "root", "e3") is None


def _turns_for(eid):
    return [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "s",
                    "type": "function",
                    "function": {
                        "name": "stage",
                        "arguments": json.dumps(
                            {"path": "notes.md", "text": f"lesson from {eid}"},
                        ),
                    },
                },
            ],
        },
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "d",
                    "type": "function",
                    "function": {"name": "done", "arguments": "{}"},
                },
            ],
        },
    ]


def test_analysts_share_one_prefix_and_write_staging(tmp_path):
    seen = []
    queues = {}

    async def model_turn(messages, tools):
        eid = messages[2]["content"].split("`")[1]
        seen.append([json.dumps(m, sort_keys=True) for m in messages[:2]])
        q = queues.setdefault(eid, _turns_for(eid))
        return (q.pop(0) if q else {"role": "assistant", "content": "done"}), "0.01"

    async def reader(name, args):
        return "text"

    rows = asyncio.run(
        analysts.run(
            {"e1": ["signal"], "e2": ["novel_code"], "e3": []},
            model_turn=model_turn,
            reader=reader,
            reader_tools=[],
            index_view="Current library index: (empty)",
            root=tmp_path,
            step_guard=10,
        ),
    )
    assert sorted(r["episode"] for r in rows) == [
        "e1",
        "e2",
    ]  # e3 was not flagged: no analyst
    assert (
        len({tuple(s) for s in seen}) == 1
    )  # every sibling's prefix is the same bytes
    v = staging.read(tmp_path, "e1")
    assert v.source == "sol_analyst" and v.files == {"notes.md": b"lesson from e1"}


def test_flags_come_from_code_never_ids():
    ep = _episode(
        "e9",
        [
            Cell(0, "x = 1/0", "Traceback (most recent call last):\n", None),
            Cell(1, "y = [i * 2 for i in range(9)]\nprint(y)", "", None),
        ],
    )
    why = analysts.flagged(ep, clustered=set(), seen_keys=set())
    assert "fail_then_succeed" in why and "novel_code" in why and "signal" not in why
    assert analysts.flagged(_episode("e8"), clustered=set(), seen_keys=set()) == []


def test_the_pass_lists_each_episodes_staging_in_the_batch_map(tmp_path):
    root = tmp_path / "staging-root"
    _staged(root, "e2")
    model = Turns(
        [
            _call("m", "read", {"path": "/inputs/batch_map.json"}),
            _call("f", "read", {"path": "/inputs/staging/e2/candidates/f.py"}),
        ],
    )
    _, _, sol = _sol(tmp_path, model, v21=True, max_calls=6, staging_root=str(root))
    sol.load = {"e2": _episode("e2")}.__getitem__
    asyncio.run(sol.run(PassRequest("incremental", "svc", ["e2"], False), "p1"))
    assert '"source": "fork"' in model.outputs["m"] and "def f(x)" in model.outputs["f"]

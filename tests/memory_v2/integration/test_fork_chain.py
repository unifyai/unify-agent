"""The fork to writer chain: admission of settled episodes only, the reply parser over the recorded corpus and every
block variant, the block ledger, and what the writer sees of parse-time refusals. Keyless: real subprocesses here
are the system python sleeping or the stdlib worker against a closed local port."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from unify.memory_v2 import staging
from unify.memory_v2.evidence import EvidenceStore
from unify.memory_v2.integration import fork as fork_mod
from unify.memory_v2.integration import fork_worker as fw
from unify.memory_v2.trigger import Trigger
from tests.memory_v2.test_episodes import _ep

CORPUS = Path(__file__).resolve().parents[1] / "data" / "fork_replies" / "corpus.json"
PY = "/usr/bin/python3"
CODE = 'def double(x):\n    """Twice x."""\n    return 2 * x\n'
ON = SimpleNamespace(UNIFY_MEMORY_LIBRARY="on", UNIFY_MEMORY_LIBRARY_FORK="on")
OFF = SimpleNamespace(UNIFY_MEMORY_LIBRARY="on", UNIFY_MEMORY_LIBRARY_FORK="off")


def _paths(tmp_path):
    state = tmp_path / "state"
    state.mkdir(exist_ok=True)
    return SimpleNamespace(state_dir=state)


def _status(paths, eid):
    return json.loads((fork_mod.staging_dir(paths, eid) / "fork.json").read_text())


# --- the parser over the recorded corpus ----------------------------------------------------------------------------


def _corpus():
    return json.loads(CORPUS.read_text())


def test_the_corpus_fixture_is_the_labelled_set():
    doc = _corpus()
    assert len(doc) == 271
    assert sum(len(e["files"]) for e in doc.values()) == 612
    assert sum(1 for e in doc.values() if e["form"] == "no_text") == 29
    for e in doc.values():
        assert hashlib.sha256(e["text"].encode()).hexdigest() == e["text_sha256"]


def test_every_labelled_file_is_recovered_with_its_path_and_content():
    by_form: dict[str, int] = {}
    renamed = []
    for rid, e in sorted(_corpus().items()):
        blocks, record = fw.parse_reply(e["text"])
        c = record["counts"]
        assert c["blocks"] == len(blocks) == c["parsed"] + c["refused"] + c["dropped"], rid
        assert set(record) <= {"blocks", "counts", "note"}  # the episode and digest come from write_record
        if not e["files"]:
            assert blocks == [] and record["note"] == "no blocks", rid
            continue
        taken = set()
        for f in e["files"]:
            hits = [
                b
                for b in blocks
                if b["index"] not in taken
                and b["status"] == "parsed"
                and b["form"] == f["form"]
                and b["content_sha256"] == f["sha256"]
                and f["path"] in (b["path"], b.get("renamed_from"))
            ]
            assert hits, (rid, f["path"], f["form"])
            b = hits[0]
            taken.add(b["index"])
            by_form[f["form"]] = by_form.get(f["form"], 0) + 1
            assert hashlib.sha256(b["content"].rstrip("\n").encode()).hexdigest() == f["sha256"]
            if f.get("marker"):
                assert b["marker"] == f["marker"]
            if b.get("renamed_from"):
                renamed.append((b["renamed_from"], b["path"], b["reason"]))
            else:
                assert b["path"] == f["path"] and b["reason"] is None
        # nothing else in the reply became a file or a silent loss
        assert [b for b in blocks if b["index"] not in taken] == [], rid
    assert by_form == {
        "fence": 434,
        "inner": 108,
        "split": 39,
        "line_before": 27,
        "split_unfenced": 4,
    }
    assert sorted(renamed) == [
        ("candidates/fill_single-edge_regions.py", "candidates/fill_single_edge_regions.py", fw.RENAMED),
        ("cases/fill_single-edge_regions.json", "cases/fill_single_edge_regions.json", fw.RENAMED),
    ]


def test_staged_cases_and_candidates_from_the_corpus_parse():
    for e in _corpus().values():
        for b in fw.parse_reply(e["text"])[0]:
            if b["status"] != "parsed":
                continue
            if b["kind"] == "case":
                json.loads(b["content"])  # a `// path:` line in a JSON block is removed
            elif b["kind"] == "candidate":
                ast.parse(b["content"])


# --- every block variant --------------------------------------------------------------------------------------------

_VARIANTS = {
    "fence": ("```path: candidates/double.py\n" + CODE + "```\n", "fence"),
    "lang_path": ("```python path: candidates/double.py\n" + CODE + "```\n", "fence"),
    "tilde": ("~~~path: candidates/double.py\n" + CODE + "~~~\n", "fence"),
    "inner_plain": ("```python\npath: candidates/double.py\n" + CODE + "```\n", "inner"),
    "inner_hash": ("```python\n# path: candidates/double.py\n" + CODE + "```\n", "inner"),
    "inner_bold": ("```python\n**path: candidates/double.py**\n" + CODE + "```\n", "inner"),
    "line_before": ("Here.\npath: candidates/double.py\n```python\n" + CODE + "```\n", "line_before"),
    "bold_before": ("**path:** `candidates/double.py`\n```python\n" + CODE + "```\n", "line_before"),
    "split": ("```text\npath: candidates/double.py\n```\n```python\n" + CODE + "```\n", "split"),
    "split_python_blank": (
        "```python\npath: candidates/double.py\n```\n\n\n```python\n" + CODE + "```\n",
        "split",
    ),
    "indented": (
        "Steps:\n  ```python\n  path: candidates/double.py\n"
        + "".join("  " + ln + "\n" for ln in CODE.splitlines())
        + "  ```\n",
        "inner",
    ),
    "unclosed": ("```path: candidates/double.py\n" + CODE + "\n", "fence"),
}


@pytest.mark.parametrize("name", sorted(_VARIANTS))
def test_every_block_variant_stages_the_same_file(name):
    text, form = _VARIANTS[name]
    blocks, record = fw.parse_reply(text)
    assert [(b["path"], b["form"], b["status"], b["content"]) for b in blocks] == [
        ("candidates/double.py", form, "parsed", CODE),
    ]
    assert record["counts"] == {"blocks": 1, "parsed": 1, "refused": 0, "dropped": 0}


def test_a_json_block_loses_its_slashes_path_line_and_html_marks_work():
    text = (
        "```json\n// path: cases/double.json\n[{\"inputs\": [21], \"expected\": 42}]\n```\n"
        "```markdown\n<!-- path: notes.md -->\nA lesson.\n```\n"
    )
    blocks, _ = fw.parse_reply(text)
    assert [(b["path"], b["marker"]) for b in blocks] == [
        ("cases/double.json", "slashes"),
        ("notes.md", "html"),
    ]
    assert json.loads(blocks[0]["content"]) == [{"inputs": [21], "expected": 42}]


def test_nested_fences_resolve_by_fence_length():
    text = "````path: notes.md\nUse:\n```python\nx = 1\n```\nDone.\n````\n```path: cases/x.json\n{}\n```\n"
    blocks, _ = fw.parse_reply(text)
    assert [(b["path"], b["content"]) for b in blocks] == [
        ("notes.md", "Use:\n```python\nx = 1\n```\nDone.\n"),
        ("cases/x.json", "{}\n"),
    ]


def test_split_unfenced_runs_to_the_end_and_a_later_block_wins():
    blocks, record = fw.parse_reply("```path: notes.md\nold\n```\n```text\npath: notes.md\n```\n\nNew.\nMore.\n")
    assert [(b["form"], b["status"], b["reason"], b["content"]) for b in blocks] == [
        ("fence", "dropped", "replaced_by_block_1", None),
        ("split_unfenced", "parsed", None, "New.\nMore.\n"),
    ]
    assert record["counts"] == {"blocks": 2, "parsed": 1, "refused": 0, "dropped": 1}


def test_refused_paths_carry_a_reason_and_no_content():
    blocks, record = fw.parse_reply("```path: ../x.py\nx\n```\n```path: other.txt\nx\n```\n")
    assert [(b["status"], b["reason"], b["content"]) for b in blocks] == [
        ("refused", "path_not_plain", None),
        ("refused", "path_not_allowed", None),
    ]
    assert record["counts"]["refused"] == 2 and "content" not in record["blocks"][0]


# --- the worker's records and what the writer sees -----------------------------------------------------------------


def _reply(content=None, calls=None):
    msg = {"role": "assistant", "content": content}
    if calls:
        msg["tool_calls"] = calls
    return {"choices": [{"message": msg}], "usage": {"cost": "0.001"}}


def _req(eid="e1"):
    return {"episode": eid, "url": "u", "headers": {}, "body": {"model": "m", "messages": []}}


def test_the_worker_writes_the_ledger_and_the_raw_reply_with_its_digest(tmp_path):
    text = (
        "```python\n# path: candidates/double.py\n" + CODE + "```\n"
        "```path: other.txt\nx\n```\n```\nprint(1)\n```\n"
    )
    doc = fw.run(_req(), tmp_path, post=lambda u, h, b: _reply(content=text))
    ledger = json.loads((tmp_path / fw.BLOCKS_JSON).read_text())
    raw = (tmp_path / fw.REPLY_JSON).read_bytes()
    assert ledger["reply_sha256"] == hashlib.sha256(text.encode()).hexdigest() == doc["reply_sha256"]
    assert ledger["reply_file_sha256"] == hashlib.sha256(raw).hexdigest()
    assert json.loads(raw)["replies"][0]["content"] == json.loads(raw)["text"] == text
    assert ledger["counts"] == doc["blocks"] == {"blocks": 3, "parsed": 1, "refused": 1, "dropped": 1}
    assert [f["path"] for f in doc["files"]] == ["candidates/double.py"]  # files written = parsed
    assert ledger["episode"] == "e1" and all("content" not in b for b in ledger["blocks"])
    # the writer sees the parse-time refusal and drop beside the staged file; record files are never staged files
    view = staging.read(tmp_path.parent, tmp_path.name)
    assert set(view.files) == {"candidates/double.py"}
    assert {(r["path"], r["why"], r["stage"]) for r in view.refused} == {
        ("other.txt", "path_not_allowed", "parse"),
        (None, "no_path", "parse"),
    }
    entry = staging.export(view, tmp_path.parent / "out" / tmp_path.name)
    assert entry["blocks"] == ledger["counts"] and len(entry["refused"]) == 2
    assert not (tmp_path.parent / "out" / tmp_path.name / fw.REPLY_JSON).exists()


def test_the_quota_refusal_reaches_the_ledger(tmp_path):
    many = "".join(f"```path: cases/c{i}.json\n{{}}\n```\n" for i in range(fw.QUOTA_FILES + 2))
    fw.run(_req(), tmp_path, post=lambda u, h, b: _reply(content=many))
    ledger = json.loads((tmp_path / fw.BLOCKS_JSON).read_text())
    assert ledger["counts"]["parsed"] == fw.QUOTA_FILES
    assert sum(b["reason"] == "quota" for b in ledger["blocks"]) == 2


# --- the instruction -------------------------------------------------------------------------------------------------


def test_the_instruction_shows_one_format_whose_example_parses():
    text = fork_mod.fork_instruction()
    assert text == fork_mod.INSTRUCTION
    assert "Do not call any tool" in text and "fenced block" in text
    assert "fenced blocks" in fw.TOOL_REPLY
    blocks, record = fw.parse_reply(fork_mod.FORMAT_EXAMPLE)
    assert [(b["path"], b["form"], b["status"]) for b in blocks] == [
        ("candidates/double.py", "fence", "parsed"),
        ("cases/double.json", "fence", "parsed"),
        ("notes.md", "fence", "parsed"),
    ]
    cases = json.loads(blocks[1]["content"])
    ns: dict = {}
    exec(compile(blocks[0]["content"], "double.py", "exec"), ns)  # the example's own function, from the constant
    assert all(ns["double"](*c["inputs"]) == c["expected"] for c in cases)


@pytest.mark.parametrize(
    "choice,expected",
    [
        (None, None),
        ("auto", "auto"),
        ("none", "none"),
        ("required", "auto"),
        ({"type": "function", "function": {"name": "execute_code"}}, "auto"),
    ],
)
def test_a_forcing_tool_choice_lets_the_fork_answer_in_text(choice, expected):
    assert fork_mod.fork_tool_choice(choice) == expected


# --- settling: timeout, lost, never started, stale ------------------------------------------------------------------


def test_the_wall_bound_records_timeout_between_turns(tmp_path):
    now = [0.0]

    def post(u, h, b):
        now[0] += 10.0
        return _reply(
            content="```path: notes.md\nn\n```\n",
            calls=[{"id": "c", "function": {"name": "x", "arguments": "{}"}}],
        )

    doc = fw.run(_req(), tmp_path, post=post, wall_s=15.0, clock=lambda: now[0])
    assert doc["status"] == "timeout" and doc["turns"] == 2
    assert [f["path"] for f in doc["files"]] == ["notes.md"]  # what it had is staged
    assert json.loads((tmp_path / "fork.json").read_text())["status"] == "timeout"


def test_the_wall_timer_interrupts_a_call_and_records_timeout(tmp_path):
    old = signal.signal(signal.SIGALRM, fw._alarm)
    try:
        signal.setitimer(signal.ITIMER_REAL, 0.2)
        doc = fw.run(_req(), tmp_path, post=lambda u, h, b: time.sleep(5))
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)
    assert doc["status"] == "timeout" and (tmp_path / "fork.json").exists()


def _sleeper(reservation, tmp_path):
    return fork_mod.launch(
        reservation,
        [PY, "-I", "-c", "import time; time.sleep(120)"],
        b"{}",
        tmp_path / "worker.log",
    )


def test_a_killed_worker_is_recorded_lost_and_then_admitted(tmp_path):
    paths = _paths(tmp_path)
    res = fork_mod.reserve(paths, "e1", ON)
    adm = fork_mod.admission(paths, ON)
    assert not adm.settled("e1")  # reserved by this process: unsettled
    fork = _sleeper(res, tmp_path)
    assert res.fd is None  # this process no longer holds the lock; the worker does
    assert not adm.settled("e1")
    os.killpg(fork.proc.pid, signal.SIGKILL)
    fork.proc.wait(timeout=30)
    assert adm.settled("e1")
    doc = _status(paths, "e1")
    assert doc["status"] == "lost" and "without writing its record" in doc["detail"]
    assert staging.read(Path(paths.state_dir) / "memory-staging", "e1").status == "lost"


def test_the_real_worker_settles_its_episode_itself(tmp_path):
    paths = _paths(tmp_path)
    res = fork_mod.reserve(paths, "e1", ON)
    req = {
        "episode": "e1",
        "url": "http://127.0.0.1:9/v1/chat/completions",  # closed: the call fails at once, keyless
        "headers": {},
        "body": {"model": "m", "messages": []},
    }
    fork = fork_mod.launch(
        res,
        [PY, "-I", str(fork_mod.WORKER_FILE), "--staging", str(res.staging)],
        json.dumps(req).encode(),
        tmp_path / "worker.log",
    )
    fork.proc.wait(timeout=60)
    assert fork_mod.settled(Path(paths.state_dir) / "memory-staging", "e1")
    doc = _status(paths, "e1")
    assert doc["status"].startswith("error:") and doc["blocks"] == {
        "blocks": 0,
        "parsed": 0,
        "refused": 0,
        "dropped": 0,
    }
    assert json.loads((res.staging / fw.BLOCKS_JSON).read_text())["note"] == "no blocks"


def test_finish_records_lost_when_the_waited_worker_exits_without_a_record(tmp_path):
    paths = _paths(tmp_path)
    res = fork_mod.reserve(paths, "e1", ON)
    fork = fork_mod.launch(res, [PY, "-I", "-c", "raise SystemExit(3)"], b"", tmp_path / "w.log")
    fork_mod.finish(fork, SimpleNamespace(UNIFY_MEMORY_LIBRARY_WAIT_SLOT="on"))
    doc = _status(paths, "e1")
    assert doc["status"] == "lost" and "code 3" in doc["detail"]


def test_a_dir_never_started_and_a_stale_lock_settle(tmp_path, monkeypatch):
    paths = _paths(tmp_path)
    root = Path(paths.state_dir) / "memory-staging"
    (root / "bare").mkdir(parents=True)
    assert not fork_mod.settled(root, "bare")  # just made: its request may still be locking it
    monkeypatch.setattr(fork_mod, "UNSTARTED_S", 0.0)
    assert fork_mod.settled(root, "bare") and _status(paths, "bare")["status"] == "lost"
    res = fork_mod.reserve(paths, "slow", ON)
    assert not fork_mod.settled(root, "slow")
    monkeypatch.setattr(fork_mod, "STALE_S", 0.0)
    assert fork_mod.settled(root, "slow") and _status(paths, "slow")["status"] == "timeout"
    res.close()
    assert fork_mod.settled(root, "no-dir")  # no staging dir: the episode had no fork


def test_admission_is_off_with_the_fork_off(tmp_path):
    paths = _paths(tmp_path)
    assert fork_mod.admission(paths, OFF) is None
    assert fork_mod.reserve(paths, "e1", OFF) is None
    assert not (Path(paths.state_dir) / "memory-staging").exists()
    assert isinstance(fork_mod.admission(paths, ON), fork_mod.Admission)


# --- the trigger under admission --------------------------------------------------------------------------------------


def _index(ev, eid):
    ev.index_episode(_ep(episode_id=eid, transcript=[], actions=[]), hashlib.sha1(eid.encode()).hexdigest())


def test_admission_holds_an_unsettled_episode_then_admits_it(tmp_path):
    paths = _paths(tmp_path)
    ev = EvidenceStore(tmp_path / "e.sqlite")
    adm = fork_mod.admission(paths, ON)
    tr = Trigger(ev, experience_budget=1, admission=adm)
    res = {e: fork_mod.reserve(paths, e, ON) for e in ("a", "b", "c")}
    for e in ("a", "b", "c"):
        _index(ev, e)
    res["a"].abandon("ok")
    res["c"].abandon("ok")
    [req] = tr.after_episode("c")
    assert req.episodes == ["a", "c"] and req.held == ["b"]
    tr.mark_done(req)
    assert adm.rolled() == ["b"]
    assert tr.after_episode("c") == []  # b is still unsettled: nothing is due
    _index(ev, "d")  # d has no staging dir: it had no fork and is admitted
    [req] = tr.after_episode("d")
    assert req.episodes == ["d"] and req.held == ["b"]
    tr.mark_done(req)
    res["b"].abandon("ok")  # b settles
    [req] = tr.after_episode("d")
    assert req.episodes == ["b"] and req.held == []
    tr.mark_done(req)
    assert adm.rolled() == [] and tr.pending() == []


def test_without_admission_the_trigger_is_unchanged(tmp_path):
    ev = EvidenceStore(tmp_path / "e.sqlite")
    tr = Trigger(ev, experience_budget=1)
    for e in ("a", "b"):
        _index(ev, e)
    [req] = tr.after_episode("b")
    assert req.episodes == ["a", "b"] and req.held == []


def test_the_pass_request_never_admits_by_stream_position(tmp_path):
    """A settled episode recorded after an unsettled one enters the batch; the order of recording decides nothing."""
    paths = _paths(tmp_path)
    ev = EvidenceStore(tmp_path / "e.sqlite")
    tr = Trigger(ev, experience_budget=1, admission=fork_mod.admission(paths, ON))
    first = fork_mod.reserve(paths, "first", ON)
    later = fork_mod.reserve(paths, "later", ON)
    _index(ev, "first")
    _index(ev, "later")
    later.abandon("ok")
    [req] = tr.after_episode("later")
    assert (req.episodes, req.held) == (["later"], ["first"])
    first.close()


@pytest.mark.parametrize("choice", ["required", {"type": "function", "function": {"name": "execute_code"}}])
def test_the_fork_request_keeps_the_tools_but_never_a_forcing_choice(monkeypatch, choice):
    import unify.actor.code_act_actor as caa

    tools = [{"type": "function", "function": {"name": "execute_code", "parameters": {}}}]
    source = {
        "client": SimpleNamespace(endpoint="provider/model@route", reasoning_effort="low"),
        "sent_messages": [{"role": "user", "content": "double 21"}],
        "tools": tools,
        "tool_choice": choice,
    }
    monkeypatch.setattr(caa, "_session_fork_source", lambda inner, actor: (source, None))
    doc, why = fork_mod.request_doc(SimpleNamespace(), "e1", "http://127.0.0.1:9/v1", "t")
    assert why is None and doc["body"]["tools"] == tools and doc["body"]["tool_choice"] == "auto"
    assert doc["body"]["messages"][-1] == {"role": "user", "content": fork_mod.fork_instruction()}


def test_write_record_keeps_the_raw_reply_and_the_ledger_in_the_staging_dir(tmp_path):
    text = "```path: notes.md\nA lesson.\n```\n```path: x.txt\ny\n```"
    blocks, hop = fw.parse_reply(text)
    doc = fw.write_record(tmp_path, "e9", text, hop)
    assert json.loads((tmp_path / fw.BLOCKS_JSON).read_text()) == doc
    assert doc["episode"] == "e9" and doc["reply_sha256"] == hashlib.sha256(text.encode()).hexdigest()
    assert doc["counts"] == {"blocks": 2, "parsed": 1, "refused": 1, "dropped": 0}
    assert json.loads((tmp_path / fw.REPLY_JSON).read_text())["text"] == text
    (tmp_path / "fork.json").write_text('{"status": "ok"}')
    view = staging.read(tmp_path.parent, tmp_path.name)
    assert view.files == {} and [r["path"] for r in view.refused] == ["x.txt"]  # record files are never content


def test_admit_settled_is_the_pure_rule(tmp_path):
    from unify.memory_v2.trigger import admit_settled

    for eid, status in (("a", "ok"), ("b", None), ("c", "timeout"), ("d", "lost")):
        (tmp_path / eid).mkdir()
        if status:
            (tmp_path / eid / "fork.json").write_text(json.dumps({"status": status}))
    assert admit_settled(tmp_path, ["b", "a", "c", "d", "e"]) == (["a", "c", "d", "e"], ["b"])
    assert not (tmp_path / "b" / "fork.json").exists()  # pure: nothing recorded


def test_request_from_source_builds_the_fork_request_directly():
    source = {
        "client": SimpleNamespace(endpoint="provider/model@route", reasoning_effort="max"),
        "sent_messages": [{"role": "user", "content": "double 21"}],
        "tools": [{"type": "function", "function": {"name": "execute_code", "parameters": {}}}],
        "tool_choice": "required",
    }
    doc = fork_mod.request_from_source("e1", source, "http://127.0.0.1:9/fork/v1", "tok")
    assert doc["url"] == "http://127.0.0.1:9/fork/v1/chat/completions"
    assert doc["headers"]["Authorization"] == "Bearer tok" and doc["headers"]["x-unify-session"] == "fork.e1"
    body = doc["body"]
    assert body["model"] == "provider/model" and body["tool_choice"] == "auto" and body["reasoning_effort"] == "xhigh"
    assert body["messages"] == [*source["sent_messages"], {"role": "user", "content": fork_mod.fork_instruction()}]
    same = fork_mod.request_from_source("e1", source, doc["url"], "tok")
    assert same["url"] == doc["url"]

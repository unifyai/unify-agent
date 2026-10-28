"""The fork to writer chain: admission of settled episodes only, the reply parser over the recorded corpus and every
block variant, the block ledger, and what the writer sees of parse-time refusals. Keyless: real subprocesses here
are the system python sleeping or the stdlib worker against a closed local port."""

from __future__ import annotations

import ast
import hashlib
import json
import signal
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from unify.memory import staging
from unify.memory.integration import fork as fork_mod
from unify.memory.integration import fork_worker as fw

CORPUS = Path(__file__).resolve().parents[1] / "data" / "fork_replies" / "corpus.json"
PY = "/usr/bin/python3"
CODE = 'def double(x):\n    """Twice x."""\n    return 2 * x\n'


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
        assert (
            c["blocks"] == len(blocks) == c["parsed"] + c["refused"] + c["dropped"]
        ), rid
        assert set(record) <= {
            "blocks",
            "counts",
            "note",
        }  # the episode and digest come from write_record
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
            assert (
                hashlib.sha256(b["content"].rstrip("\n").encode()).hexdigest()
                == f["sha256"]
            )
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
        (
            "candidates/fill_single-edge_regions.py",
            "candidates/fill_single_edge_regions.py",
            fw.RENAMED,
        ),
        (
            "cases/fill_single-edge_regions.json",
            "cases/fill_single_edge_regions.json",
            fw.RENAMED,
        ),
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
    "inner_plain": (
        "```python\npath: candidates/double.py\n" + CODE + "```\n",
        "inner",
    ),
    "inner_hash": (
        "```python\n# path: candidates/double.py\n" + CODE + "```\n",
        "inner",
    ),
    "inner_bold": (
        "```python\n**path: candidates/double.py**\n" + CODE + "```\n",
        "inner",
    ),
    "line_before": (
        "Here.\npath: candidates/double.py\n```python\n" + CODE + "```\n",
        "line_before",
    ),
    "bold_before": (
        "**path:** `candidates/double.py`\n```python\n" + CODE + "```\n",
        "line_before",
    ),
    "split": (
        "```text\npath: candidates/double.py\n```\n```python\n" + CODE + "```\n",
        "split",
    ),
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
        '```json\n// path: cases/double.json\n[{"inputs": [21], "expected": 42}]\n```\n'
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
    blocks, record = fw.parse_reply(
        "```path: notes.md\nold\n```\n```text\npath: notes.md\n```\n\nNew.\nMore.\n",
    )
    assert [(b["form"], b["status"], b["reason"], b["content"]) for b in blocks] == [
        ("fence", "dropped", "replaced_by_block_1", None),
        ("split_unfenced", "parsed", None, "New.\nMore.\n"),
    ]
    assert record["counts"] == {"blocks": 2, "parsed": 1, "refused": 0, "dropped": 1}


def test_refused_paths_carry_a_reason_and_no_content():
    blocks, record = fw.parse_reply(
        "```path: ../x.py\nx\n```\n```path: other.txt\nx\n```\n",
    )
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
    return {
        "episode": eid,
        "url": "u",
        "headers": {},
        "body": {"model": "m", "messages": []},
    }


def test_the_worker_writes_the_ledger_and_the_raw_reply_with_its_digest(tmp_path):
    text = (
        "```python\n# path: candidates/double.py\n" + CODE + "```\n"
        "```path: other.txt\nx\n```\n```\nprint(1)\n```\n"
    )
    doc = fw.run(_req(), tmp_path, post=lambda u, h, b: _reply(content=text))
    ledger = json.loads((tmp_path / fw.BLOCKS_JSON).read_text())
    raw = (tmp_path / fw.REPLY_JSON).read_bytes()
    assert (
        ledger["reply_sha256"]
        == hashlib.sha256(text.encode()).hexdigest()
        == doc["reply_sha256"]
    )
    reply = json.loads(raw)
    assert (
        reply["reply_text"] == text
        and reply["turns"] == [text]
        and reply["reply_sha256"] == doc["reply_sha256"]
    )
    # the worker's ledger is the parse of the joined reply text, as the offline replay computes it
    assert {
        k: v for k, v in ledger.items() if k not in ("episode", "reply_sha256")
    } == fw.parse_reply(text)[1]
    assert (
        ledger["counts"]
        == doc["blocks"]
        == {"blocks": 3, "parsed": 1, "refused": 1, "dropped": 1}
    )
    assert [f["path"] for f in doc["files"]] == [
        "candidates/double.py",
    ]  # files written = parsed
    assert ledger["episode"] == "e1" and all(
        "content" not in b for b in ledger["blocks"]
    )
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
    many = "".join(
        f"```path: cases/c{i}.json\n{{}}\n```\n" for i in range(fw.QUOTA_FILES + 2)
    )
    fw.run(_req(), tmp_path, post=lambda u, h, b: _reply(content=many))
    ledger = json.loads((tmp_path / fw.BLOCKS_JSON).read_text())
    assert ledger["counts"]["parsed"] == fw.QUOTA_FILES
    assert sum(b["reason"] == "quota" for b in ledger["blocks"]) == 2


# --- the instruction -------------------------------------------------------------------------------------------------


def test_the_instruction_shows_one_format_whose_example_parses():
    text = fork_mod.fork_instruction()
    assert text == fork_mod.INSTRUCTION
    # rev 2: the fork calls only the run record's tools, and its instruction says so
    assert fork_mod.RECORD_CALLS_ONLY in text and "fenced block" in text
    assert "fenced blocks" in fw.TOOL_REPLY
    blocks, record = fw.parse_reply(fork_mod.FORMAT_EXAMPLE)
    assert [(b["path"], b["kind"], b["form"], b["status"]) for b in blocks] == [
        ("summary.md", "summary", "fence", "parsed"),
        ("candidates/double.py", "candidate", "fence", "parsed"),
        ("cases/double.json", "case", "fence", "parsed"),
        ("notes.md", "notes", "fence", "parsed"),
        # a note candidate, kept with no procedure behind it
        ("notes/header_line.md", "note", "fence", "parsed"),
    ]
    # the example summary has the five headed sections, in order, and a pointer on every line under the last four
    summary = blocks[0]["content"]
    heads = [ln for ln in summary.splitlines() if ln.startswith("## ")]
    assert heads == ["## Asked", "## Approach", "## Worked", "## Failed", "## Outcome"]
    from unify.memory.summaries import pointers

    assert len(pointers(summary)) == 5
    cases = json.loads(blocks[2]["content"])
    ns: dict = {}
    exec(
        compile(blocks[1]["content"], "double.py", "exec"),
        ns,
    )  # the example's own function, from the constant
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


def test_the_real_worker_settles_its_episode_itself(tmp_path):
    paths = _paths(tmp_path)
    res = fork_mod.reserve(paths, "e1")
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
    res = fork_mod.reserve(paths, "e1")
    fork = fork_mod.launch(
        res,
        [PY, "-I", "-c", "raise SystemExit(3)"],
        b"",
        tmp_path / "w.log",
    )
    fork_mod.finish(fork)
    doc = _status(paths, "e1")
    assert doc["status"] == "lost" and "code 3" in doc["detail"]


@pytest.mark.parametrize(
    "choice",
    ["required", {"type": "function", "function": {"name": "execute_code"}}],
)
def test_the_fork_request_keeps_the_tools_but_never_a_forcing_choice(
    monkeypatch,
    choice,
):
    import unify.actor.code_act_actor as caa

    tools = [
        {"type": "function", "function": {"name": "execute_code", "parameters": {}}},
    ]
    source = {
        "client": SimpleNamespace(
            endpoint="provider/model@route",
            reasoning_effort="low",
        ),
        "sent_messages": [{"role": "user", "content": "double 21"}],
        "tools": tools,
        "tool_choice": choice,
    }
    monkeypatch.setattr(
        caa,
        "_session_fork_source",
        lambda inner, actor: (source, None),
    )
    doc, why = fork_mod.request_doc(
        SimpleNamespace(),
        "e1",
        "http://127.0.0.1:9/v1",
        "t",
    )
    assert (
        why is None
        and doc["body"]["tools"] == tools
        and doc["body"]["tool_choice"] == "auto"
    )
    assert doc["body"]["messages"][-1] == {
        "role": "user",
        "content": fork_mod.fork_instruction(),
    }


def test_write_record_keeps_the_raw_reply_and_the_ledger_in_the_staging_dir(tmp_path):
    text = "```path: notes.md\nA lesson.\n```\n```path: x.txt\ny\n```"
    blocks, hop = fw.parse_reply(text)
    doc = fw.write_record(tmp_path, "e9", text, hop)
    assert json.loads((tmp_path / fw.BLOCKS_JSON).read_text()) == doc
    assert (
        doc["episode"] == "e9"
        and doc["reply_sha256"] == hashlib.sha256(text.encode()).hexdigest()
    )
    assert doc["counts"] == {"blocks": 2, "parsed": 1, "refused": 1, "dropped": 0}
    reply = json.loads((tmp_path / fw.REPLY_JSON).read_text())
    assert (reply["episode"], reply["reply_text"], reply["turns"]) == (
        "e9",
        text,
        [text],
    )
    assert reply["reply_sha256"] == doc["reply_sha256"]
    (tmp_path / "fork.json").write_text('{"status": "ok"}')
    view = staging.read(tmp_path.parent, tmp_path.name)
    assert view.files == {} and [r["path"] for r in view.refused] == [
        "x.txt",
    ]  # record files are never content


def test_request_from_source_builds_the_fork_request_directly():
    source = {
        "client": SimpleNamespace(
            endpoint="provider/model@route",
            reasoning_effort="max",
        ),
        "sent_messages": [{"role": "user", "content": "double 21"}],
        "tools": [
            {
                "type": "function",
                "function": {"name": "execute_code", "parameters": {}},
            },
        ],
        "tool_choice": "required",
    }
    doc, why = fork_mod.request_from_source(
        "e1",
        source,
        "http://127.0.0.1:9/fork/v1",
        "tok",
    )
    assert why is None and doc["url"] == "http://127.0.0.1:9/fork/v1/chat/completions"
    assert (
        doc["headers"]["Authorization"] == "Bearer tok"
        and doc["headers"]["x-unify-session"] == "fork.e1"
    )
    body = doc["body"]
    assert (
        body["model"] == "provider/model"
        and body["tool_choice"] == "auto"
        and body["reasoning_effort"] == "xhigh"
    )
    assert body["messages"] == [
        *source["sent_messages"],
        {"role": "user", "content": fork_mod.fork_instruction()},
    ]
    same, _ = fork_mod.request_from_source("e1", source, doc["url"], "tok")
    assert same["url"] == doc["url"]
    assert fork_mod.request_from_source(
        "e1",
        {**source, "client": SimpleNamespace()},
        "u",
        "t",
    ) == (
        None,
        "the session's client names no model",
    )


def test_the_reply_text_joins_every_turn_and_tool_call_blocks_are_kept(tmp_path):
    turns = [
        {
            "role": "assistant",
            "content": "```path: notes.md\nfirst\n```",
            "tool_calls": [
                {
                    "id": "c",
                    "function": {
                        "name": "execute_code",
                        "arguments": json.dumps(
                            {"code": "```path: cases/x.json\n{}\n```"},
                        ),
                    },
                },
            ],
        },
        {
            "role": "assistant",
            "content": "```path: candidates/double.py\n" + CODE + "```",
        },
    ]
    blocks, hop, text = fw.parse_turns(turns)
    assert text == turns[0]["content"] + "\n" + turns[1]["content"]
    assert [(b["path"], b["source"]) for b in blocks] == [
        ("notes.md", "text"),
        ("candidates/double.py", "text"),
        ("cases/x.json", "tool_call:execute_code"),
    ]
    fw.write_record(tmp_path, "e1", text, hop, turns)
    reply = json.loads((tmp_path / fw.REPLY_JSON).read_text())
    assert (
        reply["turns"] == [t["content"] for t in turns] and reply["tool_calls"][1] == []
    )


def test_helpers_are_a_staging_source(tmp_path):
    (tmp_path / "h").mkdir()
    (tmp_path / "h" / "helper.json").write_text('{"status": "ok"}')
    assert staging.read(tmp_path, "h").source == "helper"


def test_the_fork_sends_the_messages_as_the_actor_sent_them():
    """The history the harness holds carries its own annotations (keys that start with ``_``) and keys with no
    value; the actor's request carried neither (unillm strips the one, LiteLLM leaves out the other), so the fork's
    prefix leaves them out too and is the actor's, byte for byte."""
    held = [
        {"role": "user", "content": "double 21", "_ctx_header": "x"},
        {
            "role": "assistant",
            "content": "42",
            "tool_calls": None,
            "function_call": None,
            "provider_specific_fields": None,
            "_reply_source": "cell",
            "_reply_from_value": True,
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "kept", "_static": True}],
            "reasoning_details": [{"type": "reasoning.encrypted", "data": "e"}],
        },
    ]
    source = {
        "client": SimpleNamespace(endpoint="provider/model@route"),
        "sent_messages": held,
        "tools": [],
    }

    doc, why = fork_mod.request_from_source("e1", source, "http://127.0.0.1:9/fork/v1", "tok")

    assert why is None
    assert doc["body"]["messages"][:-1] == [
        {"role": "user", "content": "double 21"},
        {"role": "assistant", "content": "42"},
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "kept"}],
            "reasoning_details": [{"type": "reasoning.encrypted", "data": "e"}],
        },
    ]
    # the history itself is left as it is
    assert held[1]["_reply_source"] == "cell" and "_static" in held[2]["content"][0]

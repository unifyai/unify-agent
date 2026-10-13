"""The episode-end fork's ``summary.md``: the fork writes it with pointers into its episode, given the episode's part
list; the pointers that do not resolve are counted per episode; the writer brief sends the writer to the summaries
first; a fork that did not finish hands the writer nothing, and the writer is told; and the evidence fields are
written. Keyless: scripted replies and scripted writer turns only."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from unify.common import in_code_switch
from unify.memory import library_prompts as lp
from unify.memory import staging, summaries
from unify.memory.episodes import Action, Cell
from unify.memory.integration import consolidate
from unify.memory.integration import fork as fork_mod
from unify.memory.integration import fork_worker as fw
from unify.memory.trigger import PassRequest
from unify.memory.writer_tools import part_names
from tests.memory.test_writer_library_tools import _episode
from tests.memory.test_writer_pass import Turns, _call, _writer

# --- one recorded episode and a summary of it --------------------------------------------------------------------


def _counting_episode(eid="e1"):
    ep = _episode(
        eid,
        [
            Cell(0, "n = int(lines[0])\n", "", "ValueError: invalid literal for int() with base 10: 'rows'"),
            Cell(1, "rows = [x for x in lines if x.strip()]\nprint(len(rows))\n", "12\n"),
        ],
        ("Count the rows in data.txt.", "The file has 13 lines."),
    )
    ep.actions = [Action(1, "files", "submit", [12], {}, response="received: 12", status="ok")]
    return ep


SUMMARY = (
    "## Asked\n"
    "Count the non-empty rows of a text file.\n"
    "## Approach\n"
    '- Parsed the first line as a number (`cell:0` "int(lines[0])").\n'
    '- Counted the non-blank lines (`cell:1` "print(len(rows))").\n'
    "## Worked\n"
    '- Counting non-blank lines (`cell:1` "12").\n'
    "## Failed\n"
    '- Reading a count from the first line (`cell:0` "ValueError: invalid literal for int()").\n'
    "## Outcome\n"
    'Submitted 12; whether it was right is not shown (`action:0` "received: 12").\n'
)


def _reply(content):
    return {
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 1000, "completion_tokens": 50, "cost": "0.001"},
    }


def _source():
    return {
        "client": SimpleNamespace(endpoint="provider/model@route", reasoning_effort="low"),
        "sent_messages": [{"role": "user", "content": "Count the rows in data.txt."}],
        "tools": [],
        "tool_choice": "auto",
    }


# --- the fork ----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("in_code", [False, True])
def test_the_fork_instruction_asks_for_the_summary_first_in_the_format(monkeypatch, in_code):
    monkeypatch.setattr(in_code_switch, "tools_in_code", lambda: in_code)
    text = fork_mod.fork_instruction()
    assert text == (fork_mod.INSTRUCTION_IN_CODE if in_code else fork_mod.INSTRUCTION)
    wanted = text[text.index("What to write:\n") :].splitlines()[1:5]
    assert [ln.split("`")[1] for ln in wanted] == [
        "summary.md",
        "candidates/<name>.py",
        "cases/<name>.json",
        "notes.md",
    ]
    assert lp.SUMMARY_FORMAT in text and fork_mod.NUMBERING in text
    assert text.index(lp.SUMMARY_FORMAT) < text.index("Limits:")
    assert not lp.example_checks(text) and not lp.benchmark_words(text)
    # the format and its example name no benchmark or environment: a general task only
    for part in (fork_mod.FORMAT_EXAMPLE, lp.SUMMARY_FORMAT, fork_mod.NUMBERING, fork_mod.PARTS_HEAD):
        assert lp.benchmark_words(part) == [] and lp.example_checks(part) == []
    assert "grid" not in fork_mod.FORMAT_EXAMPLE.lower()
    assert fw.allowed("summary.md") is None and fw.kind("summary.md") == "summary"


def test_the_fork_is_given_the_episodes_part_list():
    ep = _counting_episode()
    parts = fork_mod.part_list(ep)
    lines = parts.splitlines()
    assert lines[0] == fork_mod.PARTS_HEAD
    # every part the writer can read, in the writer's order, each with its first line or what it was
    assert [ln[2:].split(": ", 1)[0] for ln in lines[1:]] == part_names(ep)
    assert lines[1:] == [
        "- request: Count the rows in data.txt.",
        "- observation:0: The file has 13 lines.",
        "- cell:0: n = int(lines[0])  (raised an error)",
        "- cell:1: rows = [x for x in lines if x.strip()]",
        "- action:0: tool files.submit: ok",
    ]
    doc, why = fork_mod.request_from_source("e1", _source(), "http://127.0.0.1:9/fork/v1", "tok", parts=parts)
    assert why is None
    # the part list follows the instruction in the one message after the cached prefix
    assert doc["body"]["messages"][:-1] == _source()["sent_messages"]
    assert doc["body"]["messages"][-1] == {
        "role": "user",
        "content": fork_mod.fork_instruction() + "\n\n" + parts,
    }
    same, _ = fork_mod.request_from_source("e1", _source(), "http://127.0.0.1:9/fork/v1", "tok")
    assert same["body"]["messages"][-1]["content"] == fork_mod.fork_instruction()


def test_the_fork_writes_summary_md_whose_pointers_resolve(tmp_path):
    ep = _counting_episode()
    content = f"```path: summary.md\n{SUMMARY}```\n\n```path: notes.md\n- A header line is not a count.\n```\n"
    doc = fw.run(
        {"episode": "e1", "url": "u", "headers": {}, "body": {"model": "m", "messages": []}},
        tmp_path,
        post=lambda u, h, b: _reply(content),
    )
    assert doc["status"] == "ok"
    assert (tmp_path / "summary.md").read_text() == SUMMARY
    assert doc["summary_bytes"] == len(SUMMARY.encode()) and isinstance(doc["seconds"], float)
    assert json.loads((tmp_path / fw.FORK_JSON).read_text())["summary_bytes"] == len(SUMMARY.encode())
    ledger = json.loads((tmp_path / fw.BLOCKS_JSON).read_text())
    assert [(b["path"], b["kind"], b["status"]) for b in ledger["blocks"]] == [
        ("summary.md", "summary", "parsed"),
        ("notes.md", "notes", "parsed"),
    ]
    # staged with the fork's other files, and its path in the batch-map entry
    view = staging.read(tmp_path.parent, tmp_path.name)
    assert set(view.files) == {"summary.md", "notes.md"} and view.finished
    entry = staging.export(view, tmp_path.parent / "out" / tmp_path.name)
    assert entry["summary"] == f"/inputs/staging/{tmp_path.name}/summary.md" and entry["status"] == "ok"
    # every pointer names a recorded part and quotes it
    assert summaries.check(view.files["summary.md"].decode(), ep) == {
        "pointers": 5,
        "resolved": 5,
        "unresolved": 0,
        "unresolved_pointers": [],
    }


def test_the_pointers_that_do_not_resolve_are_counted():
    ep = _counting_episode()
    text = (
        SUMMARY
        + '- A part that does not exist (`cell:7` "print(len(rows))").\n'
        + '- A quote the part does not hold (`cell:1` "len(lines)").\n'
        + '- No quote (`request` "").\n'
        + "- Spacing differs, still the same text (`observation:0` “The file  has 13 \t lines.”).\n"
    )
    got = summaries.check(text, ep)
    assert (got["pointers"], got["resolved"], got["unresolved"]) == (9, 6, 3)
    assert got["unresolved_pointers"] == [
        {"part": "cell:7", "quote": "print(len(rows))", "why": summaries.NO_SUCH_PART},
        {"part": "cell:1", "quote": "len(lines)", "why": summaries.NOT_IN_PART},
        {"part": "request", "quote": "", "why": summaries.EMPTY_QUOTE},
    ]
    assert summaries.check("no pointers here", ep)["pointers"] == 0


# --- the writer ----------------------------------------------------------------------------------------------------


def _brief(monkeypatch, in_code, staging_on, helper=True):
    monkeypatch.setattr(in_code_switch, "tools_in_code", lambda: in_code)
    return lp.write_brief_now(staging=staging_on, pool=True, helper=helper, similar=True)


def _step(brief, n):
    work = brief[brief.index("# How to work\n") :]
    return next(ln for ln in work.splitlines() if ln.startswith(f"{n}. "))


@pytest.mark.parametrize("in_code", [False, True])
def test_the_writer_brief_names_the_summaries_first(monkeypatch, in_code):
    brief = _brief(monkeypatch, in_code, True)
    one, two = _step(brief, 1), _step(brief, 2)
    # step 1: the batch, then each episode's summary, then the proposals
    assert one.index("overview()") < one.index("summary.md") < one.index("proposals")
    assert "/inputs/staging/<episode>/summary.md" in one
    # step 2: a full episode only where the summary is not enough, read by the parts a pointer names
    assert two.startswith("2. Open a full episode only where you need more than its summary")
    assert "read_episode(episode, parts)" in two and "`end_parts`" in two
    assert lp.SUMMARY_NOT_EVIDENCE in _step(brief, 3)
    # the inputs say what a summary holds, and what a row without reader output means
    staged = next(ln for ln in brief.splitlines() if ln.startswith("- /inputs/staging/<episode>/"))
    assert "`summary.md`" in staged and lp.STAGING_UNFINISHED in staged
    # delegate stays, unchanged; the brief stays complete
    assert "hand a reading or drafting task to a helper" in _step(brief, 5)
    assert all(head in brief for head in ("# Role\n", "# Inputs\n", "# How to work\n", "# Done\n"))
    assert not lp.example_checks(brief) and not lp.benchmark_words(brief)
    # without staging, nothing about summaries, and step 2 is as before
    plain = _brief(monkeypatch, in_code, False)
    assert "summary.md" not in plain and lp.STAGING_UNFINISHED not in plain
    assert _step(plain, 2).startswith("2. Read the episodes behind what may be worth keeping")


def test_the_staging_row_keys_are_the_built_rows_keys_with_the_summary_and_no_reader_name(tmp_path):
    from unify.memory.writer_pass import _staged_candidates

    doc = dict(lp.row_keys(staging=True))["staging"]
    assert "no reader output" in doc and "`source`" not in doc
    named = [k for k in lp.STAGING_ROW_FIELDS if f"`{k}`" in doc]
    assert named == list(lp.STAGING_ROW_FIELDS) and "summary" in named and "source" not in named
    # a row built as the pass builds it has exactly these keys, and never names the reader
    root = tmp_path / "staging-root"
    d = _stage(root, "e4", "ok", {"summary.md": SUMMARY, "candidates/f.py": "def f(x):\n    return x\n"})
    (d / fw.BLOCKS_JSON).write_text(json.dumps({"counts": {"blocks": 2, "parsed": 2, "refused": 0, "dropped": 0}}))
    view = staging.read(root, "e4")
    assert view.source == "fork"  # kept host-side
    entry = staging.export(view, tmp_path / "out" / "e4")
    entry["candidates"] = _staged_candidates(entry)
    assert set(entry) == set(lp.STAGING_ROW_FIELDS)
    assert "fork" not in json.dumps({k: v for k, v in entry.items() if k != "refused"})


def _stage(root, eid, status, files):
    d = root / eid
    for rel, text in files.items():
        (d / rel).parent.mkdir(parents=True, exist_ok=True)
        (d / rel).write_text(text)
    (d / fw.FORK_JSON).write_text(json.dumps({"status": status}))
    return d


def _with_usage(msg, prompt=None, cached=None):
    usage = {}
    if prompt is not None:
        usage["prompt_tokens"] = prompt
    if cached is not None:
        usage["prompt_tokens_details"] = {"cached_tokens": cached}
    return {**msg, "usage": usage}


@pytest.mark.parametrize("status", ["timeout", "lost", "error: TimeoutError", "refused: no fork lane"])
def test_an_unfinished_fork_stages_nothing_and_the_writer_is_told(tmp_path, status):
    root = tmp_path / "staging-root"
    _stage(root, "e2", status, {"summary.md": SUMMARY, "candidates/f.py": "def f(x):\n    return x\n"})
    view = staging.read(root, "e2")
    assert view.files == {} and not view.finished
    assert sorted(view.withheld) == ["candidates/f.py", "summary.md"]
    assert {(r["path"], r["why"]) for r in view.refused} == {
        ("candidates/f.py", staging.WITHHELD_WHY),
        ("summary.md", staging.WITHHELD_WHY),
    }
    model = Turns(
        [
            _call("m", "read", {"path": "/inputs/batch_map.json"}),
            _call("f", "read", {"path": "/inputs/staging/e2/candidates/f.py"}),
            _call("s", "read", {"path": "/inputs/staging/e2/summary.md"}),
        ],
    )
    _, _, writer = _writer(tmp_path, model, library_mode=True, max_calls=6, staging_root=str(root))
    writer.load = {"e2": _episode("e2")}.__getitem__
    out = asyncio.run(writer.run(PassRequest("incremental", "svc", ["e2"], False), "p1"))
    # the writer's batch-map row says no reader output and why, and lists the withheld files as refused
    shown = model.outputs["m"]
    assert f'"status": "{staging.NO_OUTPUT}: {status}"' in shown
    assert '"summary": null' in shown and f'"why": "{staging.WITHHELD_WHY}"' in shown
    # the files stay on the host for the record, and none reaches the writer
    assert (root / "e2" / "summary.md").read_text() == SUMMARY
    assert "def f(x)" not in model.outputs["f"] and "## Asked" not in model.outputs["s"]
    (ep_row,) = out.staging_episodes
    assert ep_row["status"].startswith(staging.NO_OUTPUT) and ep_row["withheld"] == 2
    assert ep_row["files"] == 0 and ep_row["summary_bytes"] is None and ep_row["summary_read"] is False
    # the writer is told what such a row means
    assert lp.STAGING_UNFINISHED in lp.write_brief_now(staging=True)


def test_a_finished_fork_with_the_quota_note_still_hands_its_files_on(tmp_path):
    root = tmp_path / "staging-root"
    _stage(root, "e3", "ok: operational quota reached", {"notes.md": "- a lesson\n"})
    view = staging.read(root, "e3")
    assert view.finished and set(view.files) == {"notes.md"} and view.withheld == []
    assert staging.export(view, tmp_path / "out" / "e3")["status"] == "ok: operational quota reached"


def test_the_evidence_fields_are_written(tmp_path, monkeypatch):
    root = tmp_path / "staging-root"
    summary = (
        "## Asked\nReturn the input.\n## Approach\n"
        '- Wrote the function (`cell:0` "return x").\n'
        "## Worked\n"
        '- It ran (`cell:0` "ok").\n'
        "## Failed\nnone\n## Outcome\n"
        'not shown (`action:0` "x").\n'
    )
    _stage(root, "e2", "ok", {"summary.md": summary, "candidates/f.py": "def f(x):\n    return x\n"})
    ep = _episode("e2", [Cell(0, "def f(x):\n    return x\n", "ok\n")], ("Return the input.",))
    model = Turns(
        [
            _with_usage(_call("s", "read", {"path": "/inputs/staging/e2/summary.md"}), 100, 40),
            _with_usage(_call("m", "read", {"path": "/inputs/batch_map.json"}), 150),
        ],
    )
    _, _, writer = _writer(tmp_path, model, library_mode=True, max_calls=6, staging_root=str(root))
    writer.load = {"e2": ep}.__getitem__
    out = asyncio.run(writer.run(PassRequest("incremental", "svc", ["e2"], False), "p1"))
    assert "## Asked" in model.outputs["s"]
    assert '"summary": "/inputs/staging/e2/summary.md"' in model.outputs["m"]
    (row,) = out.staging_episodes
    assert row == {
        "episode": "e2",
        "source": "fork",
        "status": "ok",
        "files": 2,
        "candidates": 1,
        "withheld": 0,
        "summary_bytes": len(summary.encode()),
        "summary_pointers": 3,
        "summary_pointers_resolved": 2,
        "summary_pointers_unresolved": 1,
        "unresolved_pointers": [{"part": "action:0", "quote": "x", "why": summaries.NO_SUCH_PART}],
        "summary_read": True,
    }
    # the writer's input tokens: summed over the calls that reported them, with how many did, out of all its calls
    assert out.input_tokens == 250 and out.input_tokens_calls == 2
    assert out.cache_read_tokens == 40 and out.cache_read_calls == 1
    assert out.writer_calls > 2 and out.writer_first_prompt_tokens == 100
    monkeypatch.setattr(consolidate, "_library_after", lambda stores: (None, None))
    end = consolidate._end_event(None, "p1", out, [], 1.0, None)
    assert end["staging_episodes"] == out.staging_episodes
    assert (end["input_tokens"], end["input_tokens_calls"], end["cache_read_tokens"], end["cache_read_calls"]) == (
        250,
        2,
        40,
        1,
    )
    assert end["writer_calls"] == out.writer_calls and end["writer_first_prompt_tokens"] == 100

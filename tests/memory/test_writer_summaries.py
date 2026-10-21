"""The writer reads each episode's summary first, opens a full episode only when it needs to, is told when an
episode's fork did not finish, keeps ``delegate``, and has its own input tokens recorded.

Keyless: the writer is a scripted model; the episodes are in memory; no box runs a cell.
"""

from __future__ import annotations

import asyncio
import copy
import json
from pathlib import Path


from unify.memory import library_prompts as lp
from unify.memory.blobs import BlobStore
from unify.memory.episodes import Cell
from unify.memory.evidence import EvidenceStore
from unify.memory.gate import Gate
from unify.memory.library_checks import LibraryConfig
from unify.memory.gitio import Repo
from unify.memory.integration import fork_worker as fw
from unify.memory.trigger import PassRequest
from unify.memory.writer_pass import PassConfig, WriterPass
from tests.memory.test_writer_library_tools import _episode
from tests.memory.test_writer_pass import _call

# --- assumed names (the spec leaves these open; the integrator adapts them here, in one place) ---------------------
SUMMARY_PATH = "summary.md"
#: the sentences the WRITE brief gains (H.4), quoted from the spec
SUMMARY_IN_BOX = "/inputs/staging/<episode>/summary.md"
OPEN_ONLY_WHEN_NEEDED = "Open a full episode only where you need more than its summary"
NOT_EVIDENCE = "A summary is a reader's account, not evidence"
WITHHELD_SENTENCE = (
    "A row whose `staging.status` is not `ok` has no reader output and says why; read that episode yourself."
)
#: the batch-map row's new key and an unfinished reader's status prefix (H.4, H.3.5)
ROW_SUMMARY = "summary"
NO_OUTPUT = "no reader output: "
#: PassOutcome fields that the end event carries (H.5, narrowed to the fork)
STAGING_EPISODES = "staging_episodes"
POINTERS = "summary_pointers"
RESOLVED = "summary_pointers_resolved"
INPUT_TOKENS_CALLS = "input_tokens_calls"
WRITER_CALLS = "writer_calls"
FIRST_PROMPT = "writer_first_prompt_tokens"
#: the staging counts' key for an episode whose reader's output was withheld (H.5)
WITHHELD_COUNT = "withheld"
# ------------------------------------------------------------------------------------------------------------------

EPS = {
    "e1": _episode(
        "e1",
        [Cell(0, "def area(w, h):\n    return w * h\nprint(area(2, 3))", "6\n")],
        ("Find the area of a 2 by 3 rectangle.",),
    ),
    "e2": _episode("e2", [Cell(0, "print('two')", "two\n")], ("Print two.",)),
}
SUMMARY = (
    "## Asked\n"
    "Compute a rectangle's area from its two sides.\n"
    "## Approach\n"
    '- Wrote a function of the two sides (`cell:0` "return w * h").\n'
    '- Checked a value the code never printed (`cell:0` "area is 7").\n'
    "## Worked\n"
    '- Calling it on the request\'s sides (`cell:0` "print(area(2, 3))").\n'
    "## Failed\n"
    '- A step the record does not hold (`cell:7` "x").\n'
    "## Outcome\n"
    'Handed back 6; not shown (`action:0` "area: 6").\n'
)
#: five pointers: two resolve (cell:0 holds both quotes), one quotes text cell:0 lacks, cell:7 and action:0 do not
#: exist in e1
EXPECTED_POINTERS = (5, 2)
NOTES_E2 = "A lesson that must not reach the writer.\n"


class Model:
    """A scripted writer: each call returns the next step (then ``finish``), with the usage given for that call."""

    def __init__(self, steps, usages=()):
        self.steps, self.usages = list(steps), list(usages)
        self.calls = 0
        self.first: list[dict] | None = None
        self.tools: list[dict] | None = None
        self.outputs: dict[str, str] = {}

    async def __call__(self, messages, tools):
        if self.first is None:
            self.first, self.tools = copy.deepcopy(messages), copy.deepcopy(tools)
        for m in messages:
            if m.get("role") == "tool":
                self.outputs[m["tool_call_id"]] = m["content"]
        k = self.calls
        self.calls += 1
        msg = dict(self.steps.pop(0) if self.steps else _call("fin", "finish", {"summary": "done"}))
        if k < len(self.usages) and self.usages[k] is not None:
            msg["usage"] = self.usages[k]
        return msg, "0.001"


def _staging(root: Path) -> None:
    d1 = root / "e1"
    d1.mkdir(parents=True)
    (d1 / SUMMARY_PATH).write_text(SUMMARY)
    (d1 / "notes.md").write_text("A lesson.\n")
    (d1 / fw.FORK_JSON).write_text(
        json.dumps(
            {
                "episode": "e1",
                "status": "ok",
                "usd": "0.0012",
                "usage": {"prompt_tokens": 1000, "cached_tokens": 900, "completion_tokens": 50, "reasoning_tokens": 10},
                "seconds": 3.5,
                "summary_bytes": len(SUMMARY.encode()),
            },
        ),
    )
    d2 = root / "e2"
    d2.mkdir(parents=True)
    (d2 / "notes.md").write_text(NOTES_E2)
    (d2 / SUMMARY_PATH).write_text(SUMMARY)
    (d2 / fw.FORK_JSON).write_text(
        json.dumps({"episode": "e2", "status": "timeout", "usd": None, "usage": {}}),
    )


def _pass(tmp_path, model, *, helper=False, eids=("e1", "e2")):
    root = tmp_path / "staging-root"
    _staging(root)
    mem = Repo.init_bare(tmp_path / "mem.git")
    ev = EvidenceStore(tmp_path / "e.sqlite")
    gate = Gate(mem, ev, BlobStore(tmp_path / "b"), library_mode=LibraryConfig())
    writer = WriterPass(
        mem,
        gate,
        ev,
        load=EPS.__getitem__,
        model_turn=model,
        config=PassConfig(max_calls=6, staging_root=str(root)),
        **({"helper_turn": lambda session=None: model} if helper else {}),
    )
    out = asyncio.run(writer.run(PassRequest("incremental", "svc", list(eids), False), "p1"))
    return out, root


def _system(model) -> str:
    return next(m["content"] for m in model.first if m.get("role") == "system")


# --- the brief ------------------------------------------------------------------------------------------------------


def _section(text: str, heading: str) -> str:
    start = text.index(heading)
    nxt = text.find("\n# ", start + len(heading))
    return text[start : nxt if nxt >= 0 else len(text)]


def test_the_brief_points_to_the_summaries_first():
    text = lp.write_brief_now()
    inputs = _section(text, "# Inputs")
    assert SUMMARY_PATH in inputs and WITHHELD_SENTENCE in inputs
    how = _section(text, "# How to work")
    one, two = how.find("1. "), how.find("\n2. ")
    assert 0 <= one < two
    step1 = how[one:two]
    # step 1 reads the summaries, before the proposals and before any full episode
    assert SUMMARY_IN_BOX in step1
    assert step1.find(SUMMARY_PATH) < step1.find("proposals")
    assert OPEN_ONLY_WHEN_NEEDED in how[two:]
    assert NOT_EVIDENCE in how
    assert lp.benchmark_words(text) == [] and lp.example_checks(text) == []


def test_the_brief_stays_complete_with_summaries():
    text = lp.write_brief_now(helper=True)
    assert SUMMARY_PATH in text
    for heading in ("# Role", "# Inputs", "# How to work", "# Done"):
        assert heading in text, heading
    # the reader's other files are still described, and every tool the writer uses is named where it is used
    for name in ("`candidates/<name>.py`", "`cases/<name>.json`", "`notes.md`", "staging:<episode>/candidates"):
        assert name in text, name
    for tool in ("overview", "read_episode", "fixture", "check", "finish", "delegate"):
        assert tool in _section(text, "# How to work"), tool
    assert "untrusted" in text


def test_delegate_is_still_offered_beside_the_summaries():
    text = lp.write_brief_now(helper=True)
    assert "delegate" in text
    assert "delegate" in lp._tool_names("write", lp.tool_flags(helper=True))
    assert SUMMARY_PATH in text


# --- the pass ---------------------------------------------------------------------------------------------------------


def test_the_writer_reads_the_summary_and_the_row_names_it(tmp_path):
    model = Model(
        [
            _call("m", "read", {"path": "/inputs/batch_map.json"}),
            _call("s", "read", {"path": f"/inputs/staging/e1/{SUMMARY_PATH}"}),
        ],
    )
    out, _root = _pass(tmp_path, model)
    assert f'"{ROW_SUMMARY}": "/inputs/staging/e1/{SUMMARY_PATH}"' in model.outputs["m"]
    assert "## Asked" in model.outputs["s"] and "return w * h" in model.outputs["s"]
    system = _system(model)
    assert SUMMARY_PATH in system and NOT_EVIDENCE in system


def test_the_pass_counts_each_summarys_pointers(tmp_path):
    out, _root = _pass(tmp_path, Model([]))
    rows = {r["episode"]: r for r in getattr(out, STAGING_EPISODES)}
    assert set(rows) == {"e1", "e2"}
    e1 = rows["e1"]
    assert e1["reader"] == "fork" and e1["status"] == "ok"
    assert e1["summary_bytes"] == len(SUMMARY.encode())
    assert (e1[POINTERS], e1[RESOLVED]) == EXPECTED_POINTERS
    assert e1["files"] == 2
    # the fork's own record, carried per episode: Decimal string USD, its seconds, and its token counts
    assert e1["reader_usd"] == "0.0012" and e1["reader_seconds"] == 3.5
    assert e1["reader_tokens"] == {"prompt": 1000, "cached": 900, "completion": 50, "reasoning": 10}


def test_an_unfinished_fork_reaches_the_writer_as_a_row_that_says_why(tmp_path):
    model = Model(
        [
            _call("m", "read", {"path": "/inputs/batch_map.json"}),
            _call("n", "read", {"path": "/inputs/staging/e2/notes.md"}),
        ],
    )
    out, root = _pass(tmp_path, model)
    # the writer is told in its row, and in its brief what such a row means
    assert f'"status": "{NO_OUTPUT}timeout"' in model.outputs["m"]
    assert WITHHELD_SENTENCE in _system(model)
    # nothing the fork wrote for e2 reaches the writer; it stays on the host
    assert NOTES_E2.strip() not in model.outputs["n"]
    assert (root / "e2" / "notes.md").read_text() == NOTES_E2
    # the pass still runs, and its records say the episode's reader output was withheld
    rows = {r["episode"]: r for r in getattr(out, STAGING_EPISODES)}
    e2 = rows["e2"]
    assert e2["status"].startswith(NO_OUTPUT) and e2["files"] == 0
    assert e2["summary_bytes"] is None
    assert e2["reader_usd"] == "unknown"
    assert out.staging[WITHHELD_COUNT] == 1


def test_delegate_is_offered_to_the_writer_with_the_fork_on(tmp_path):
    model = Model([])
    _pass(tmp_path, model, helper=True)
    names = {t["function"]["name"] for t in model.tools or []}
    system = _system(model)
    assert "delegate" in names or "tools.delegate" in system
    assert "delegate" in system
    assert SUMMARY_PATH in system


def test_the_writers_input_tokens_are_summed_over_the_calls_that_reported_them(tmp_path):
    model = Model(
        [_call("m", "read", {"path": "/inputs/batch_map.json"})],
        usages=[
            {"prompt_tokens": 1000, "prompt_tokens_details": {"cached_tokens": 600}},
            {"prompt_tokens": 1500},
        ],
    )
    out, _root = _pass(tmp_path, model)
    assert model.calls >= 3
    # a partial sum is never read as a whole one: the count of reporting calls and of all calls go with it
    assert out.input_tokens == 2500
    assert getattr(out, INPUT_TOKENS_CALLS) == 2
    assert getattr(out, WRITER_CALLS) == model.calls
    assert out.cache_read_tokens == 600
    assert getattr(out, FIRST_PROMPT) == 1000


def test_no_reported_tokens_stay_unknown(tmp_path):
    model = Model([])
    out, _root = _pass(tmp_path, model)
    assert out.input_tokens is None and getattr(out, INPUT_TOKENS_CALLS) == 0
    assert getattr(out, FIRST_PROMPT) is None
    assert getattr(out, WRITER_CALLS) == model.calls >= 1

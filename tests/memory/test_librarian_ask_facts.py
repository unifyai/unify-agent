"""``memory.ask(text)`` on the cell's side (design J(b).4 steps 1, 2 and 5): the recogniser facts are computed where
the cell runs, and only their outcomes go to the host.

1. Each distinct reader the dispatch file lists runs once on the text, then each ``applies`` whose reader parsed it,
   in the same confined, time-bounded child process as ``memory.run``. A reader that raises means "did not parse";
   nothing is retried, and its ``applies`` is never called.
2. The question and the outcomes go to the host through the in-code tool ``ask_library``. A value a reader parsed
   never leaves the cell's side.
3. What the cell gets back is the librarian's answer as a plain object: no ranking, score or top-k of the harness's
   own, and nothing word-matched added.

The generated package runs here as a cell imports it, in a fresh isolated interpreter with the library copy first on
its path. The host is a stand-in ``tools`` module that records what it is sent and returns a fixed answer. No model is
called.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tests.memory import librarian_world as w
from unify.memory.catalogue import write_files

# ==== ASSUMED NAMES (the integrator adapts these to the code, here only) ======================================
#: How ``memory.ask`` reaches the host: the box module ``tools`` (``box_interpreter.TOOLS_MODULE``), its in-code tool
#: ``ask_library``, called once per question; what it returns is the librarian's checked answer.
TOOLS_MODULE = "tools"
ASK_TOOL = "ask_library"
#: The helper's audit id and the item-call event (``library_helper.HELPER_EVENT`` / ``CALL_EVENT``).
HELPER_EVENT, CALL_EVENT, ASK_ID = "memory.helper_call", "memory.call", "memory:ask"
#: The answer's keys (design J(b).4 step 4).
FITS, NEAREST = "fits", "nearest"
#: How a fact's outcomes may be written in what is sent (any one of each group).
PARSED_WORDS = ("parsed",)
HELD_WORDS = ("held",)


def as_plain(reply_expr: str) -> str:
    """Code turning the value ``memory.ask`` returned into plain JSON data in the cell."""
    return f"(lambda r: r if isinstance(r, dict) else dict(r))({reply_expr})"


# ==============================================================================================================

PRELUDE = """
import json, sys
events = []
sys.addaudithook(lambda e, a: events.append([e, *[str(x) for x in a]]) if e in ("memory.call", "memory.helper_call") else None)
import memory
"""


def _stand_in_host(tools_dir: Path, sent: Path, answer: dict) -> None:
    tools_dir.mkdir(parents=True, exist_ok=True)
    (tools_dir / f"{TOOLS_MODULE}.py").write_text(
        "import json\n\n\n"
        f"def {ASK_TOOL}(*args, **kwargs):\n"
        f"    with open({str(sent)!r}, 'a') as fh:\n"
        "        fh.write(json.dumps({'args': args, 'kwargs': kwargs}, default=repr) + '\\n')\n"
        f"    return json.loads({json.dumps(answer)!r})\n"
    )


def _capture(mark: Path, name: str, reader_body: str, applies: bool = True) -> str:
    """A captured procedure whose reader and ``applies`` append *name*'s marks to *mark* when called."""
    src = (
        f'"""Procedure {name}."""\n\nimport base64, json\n\n\n'
        "def read_input(text):\n"
        f'    """Read the {name} part of the text."""\n'
        f"    open({str(mark)!r}, 'a').write({name + '-read'!r} + '\\n')\n"
        f"{reader_body}\n\n\n"
        "def run(value):\n"
        '    """Return the value."""\n'
        "    return value\n"
    )
    if applies:
        src += (
            "\n\ndef applies(value):\n"
            '    """Always."""\n'
            f"    open({str(mark)!r}, 'a').write({name + '-applies'!r} + '\\n')\n"
            "    return True\n"
        )
    return src


def _world(tmp_path: Path, answer: dict | None = None):
    mark = tmp_path / "marks.txt"
    sources = {
        "ep-grid": (
            _capture(mark, "grid", "    return json.loads(text.split('GRID\\n', 1)[1].splitlines()[0])"),
            w._signature(w.GRID, [[1, 4], [2, 5], [3, 6]], "Turn rows into columns.", applies=True),
        ),
        "ep-bad": (
            _capture(mark, "bad", "    raise ValueError('no such part')"),
            w._signature("x", "x", "Read a part that is never there.", applies=True),
        ),
        "ep-marked": (
            _capture(mark, "marked", "    return base64.b64decode(text.split('MARKED\\n', 1)[1].split()[0]).decode()"),
            w._signature("a-b", "a-b", "Decode the marked part.", applies=True),
        ),
    }
    caps = w.capture_files(sources)
    root = w.write_tree(tmp_path / "copy")
    write_files(root, w.generate(root, records=w.records(), summaries=w.SUMMARIES, captures=caps))
    write_files(root, caps)
    sent = tmp_path / "sent.jsonl"
    _stand_in_host(tmp_path / "host", sent, answer if answer is not None else {FITS: []})
    ids = {e: f"capture:{w.capture_id(src)}" for e, (src, _) in sources.items()}
    return root, mark, sent, ids


def _py(root: Path, tools_dir: Path, code: str) -> dict:
    out = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            f"import sys\nsys.path.insert(0, {str(tools_dir)!r})\nsys.path.insert(0, {str(root)!r})\n{PRELUDE}\n{code}",
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert out.returncode == 0, out.stderr[-4000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


TEXT = (
    "Please work out what the marked part says and turn the grid around.\n"
    f"MARKED\n{w.PLANTED_B64}\n"
    "GRID\n[[1, 2, 3], [4, 5, 6]]\n"
)


def _sent(sent: Path) -> list[dict]:
    return [json.loads(line) for line in sent.read_text().splitlines()] if sent.exists() else []


def test_facts_are_computed_on_the_cells_side_and_only_outcomes_are_sent(tmp_path):
    root, mark, sent, ids = _world(tmp_path)
    got = _py(root, tmp_path / "host", f"r = memory.ask({TEXT!r})\nprint(json.dumps({{'r': {as_plain('r')}, 'events': events}}))")
    calls = _sent(sent)
    assert len(calls) == 1, calls  # one question, one call to the host
    blob = json.dumps(calls)
    # the value a reader parsed never leaves the cell's side; the text the actor passed does
    assert w.PLANTED_VALUE not in blob
    assert w.PLANTED_B64 in blob or "MARKED" in blob
    # the outcomes name the procedures whose readers parsed it, and that an applies held
    assert ids["ep-grid"].split(":")[1] in blob and ids["ep-marked"].split(":")[1] in blob
    assert any(word in blob for word in PARSED_WORDS)
    assert any(word in blob for word in HELD_WORDS)
    # each reader ran once; the applies of the reader that raised was never called; nothing was retried
    marks = mark.read_text().splitlines()
    assert marks.count("grid-read") == 1 and marks.count("bad-read") == 1 and marks.count("marked-read") == 1
    assert marks.count("grid-applies") == 1 and marks.count("marked-applies") == 1
    assert "bad-applies" not in marks
    assert got["r"] == {FITS: []}


def test_ask_is_audited_as_a_helper_and_is_never_a_use(tmp_path):
    root, _, _, _ = _world(tmp_path)
    got = _py(root, tmp_path / "host", f"memory.ask({TEXT!r})\nprint(json.dumps(events))")
    helpers = [e for e in got if e[0] == HELPER_EVENT]
    assert [e[1] for e in helpers] == [ASK_ID], got
    assert helpers[0][-1] == "ok"
    assert [e for e in got if e[0] == CALL_EVENT] == []  # running the readers is not a use of any item


def test_the_answer_comes_back_as_the_librarian_gave_it_with_nothing_added_or_reordered(tmp_path):
    every = [
        {"id": item, "call": f"{item.split(':')[1]}(...)", "why": f"saved against a line that fits ({n})"}
        for n, item in enumerate(w.FUNCTIONS)
    ]
    root, _, _, _ = _world(tmp_path, {FITS: every})
    got = _py(root, tmp_path / "host", f"r = memory.ask({TEXT!r})\nprint(json.dumps({as_plain('r')}))")
    # every fit, in the order given, with exactly its fields: no top-k, no score, no order of the harness's own
    assert got == {FITS: every}


def test_nothing_fits_comes_back_with_the_nearest_and_what_differs(tmp_path):
    answer = {FITS: [], NEAREST: [{"id": w.TOTAL, "differs": "it sums a table; this text has none"}]}
    root, _, _, _ = _world(tmp_path, answer)
    got = _py(root, tmp_path / "host", f"r = memory.ask({TEXT!r})\nprint(json.dumps({as_plain('r')}))")
    assert got == answer


def test_text_sharing_only_words_with_a_procedure_adds_nothing_to_the_answer(tmp_path):
    """The words of a card's docstring and goal, copied into the question, never make a fit on the cell's side:
    the answer is the librarian's, and no reader parses plain words."""
    root, mark, sent, _ = _world(tmp_path)
    words = "Turn rows into columns. Decode the marked part. Sum the amount column of an order table."
    got = _py(root, tmp_path / "host", f"r = memory.ask({words!r})\nprint(json.dumps({as_plain('r')}))")
    assert got == {FITS: []}
    marks = mark.read_text().splitlines() if mark.exists() else []
    assert "grid-applies" not in marks and "marked-applies" not in marks  # no reader parsed it
    assert len(_sent(sent)) == 1


def test_a_question_with_no_procedures_to_recognise_still_reaches_the_host(tmp_path):
    root = w.write_tree(tmp_path / "copy", {"memory/text/parse.py": w.PARSE})
    write_files(root, w.generate(root))
    sent = tmp_path / "sent.jsonl"
    _stand_in_host(tmp_path / "host", sent, {FITS: []})
    got = _py(root, tmp_path / "host", "r = memory.ask('Which item counts words?')\nprint(json.dumps(" + as_plain("r") + "))")
    assert got == {FITS: []}
    assert len(_sent(sent)) == 1

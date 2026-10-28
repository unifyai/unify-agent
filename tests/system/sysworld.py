"""The system suite's two toy worlds and its scripted models.

Only the model and the world are faked here. Everything else in a system test is the real harness: the CLI session,
the actor's confined box, the request's memory run, the episode-end fork (its relay and box, reaching its model over a
loopback lane), the capture step, the writer service in its own process (its kinship judge and its passes, through the
real gate), the librarian, the record and the ledgers.

The two worlds have different shapes and name no benchmark:

* **words** (a text transform): the request carries a list of words; the work hands back each word transformed. Its
  recorded procedure is the ``value`` form (``read_input(text)`` + ``run(value)``), so a call with no arguments can run
  it on a new request.
* **tally** (a small stateful world with an action API): the actor reads a file of moves and updates a tally file in
  its workspace. Its recorded procedure acts through the episode's tools (``run(state, env)``).

The models are deterministic scripts: the actor, its helpers and the librarian through the in-process scripted
transport (:mod:`tests.scripted_model`); the fork, the writer, the writer's helpers and the kinship judge through the
loopback lanes (:mod:`tests.memory.conformance.fake_lanes`), which report a fixed declared cost per call. No key exists
and no connection leaves the machine (the suite's guard, ``tests/conftest.py``; :mod:`.test_s00_no_real_model`).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

from tests.memory.conformance import fake_lanes as FL
from tests.scripted_model import cell, reply

# ── the words world ───────────────────────────────────────────────────────────────────────────────────────────────

WORDS_MARK = "WORDS\n"


def _mark(op: str) -> int | None:
    return int(op[4:]) if op.startswith("mark") and op[4:].isdigit() else None


def words_request(words: list[str], *, op: str = "reverse") -> str:
    if _mark(op) is not None:
        return f"Write each word in capitals followed by {_mark(op)} marks.\n{WORDS_MARK}{json.dumps(words)}\nEND"
    how = {"reverse": "Reverse each word and write it in capitals.", "upper": "Write each word in capitals.",
           "title": "Write each word with a capital first letter.", "lower": "Write each word in small letters."}[op]
    return f"{how}\n{WORDS_MARK}{json.dumps(words)}\nEND"


def words_answer(words: list[str], *, op: str = "reverse") -> list[str]:
    if _mark(op) is not None:
        return [w.upper() + "!" * _mark(op) for w in words]
    return {"reverse": [w[::-1].upper() for w in words], "upper": [w.upper() for w in words],
            "title": [w.title() for w in words], "lower": [w.lower() for w in words]}[op]


def words_cell(words: list[str], *, op: str = "reverse") -> str:
    if _mark(op) is not None:
        return f"import json\nwords = {words!r}\nout = [w.upper() + {'!' * _mark(op)!r} for w in words]\nprint(json.dumps(out))\n"
    expr = {"reverse": "[w[::-1].upper() for w in words]", "upper": "[w.upper() for w in words]",
            "title": "[w.title() for w in words]", "lower": "[w.lower() for w in words]"}[op]
    return f"import json\nwords = {words!r}\nout = {expr}\nprint(json.dumps(out))\n"


_READ = (
    "import json\n\n\n"
    "def read_input(text):\n"
    '    """The list of words the request carries."""\n'
    '    return json.loads(text.split("WORDS\\n", 1)[1].split("\\nEND", 1)[0])\n\n\n'
)
WORDS_PROC = {
    "reverse": '"""Reverse each word of the request\'s list and write it in capitals."""\n\n' + _READ
    + 'def run(value):\n    """Each word reversed, in capitals."""\n    return [w[::-1].upper() for w in value]\n',
    "upper": '"""Write each word of the request\'s list in capitals."""\n\n' + _READ
    + 'def run(value):\n    """Each word in capitals."""\n    return [w.upper() for w in value]\n',
    "title": '"""Write each word of the request\'s list with a capital first letter."""\n\n' + _READ
    + 'def run(value):\n    """Each word with a capital first letter."""\n    return [w.title() for w in value]\n',
    "lower": '"""Write each word of the request\'s list in small letters."""\n\n' + _READ
    + 'def run(value):\n    """Each word in small letters."""\n    return [w.lower() for w in value]\n',
}


def _summary(asked: str, approach: str) -> str:
    return (
        "```path: summary.md\n"
        f"## Asked\n{asked}\n"
        f"## Approach\n- {approach} (`cell:0`).\n"
        "## Worked\n- The first try.\n"
        "## Failed\n- none\n"
        "## Outcome\nnot shown\n"
        "```\n\n"
    )


#: Ops ``mark<k>`` with k at or above this carry a long comment block, so their code is longer than any bound on the
#: code a judge is shown (kinship: 4,000 characters for the new capture, 600 for each earlier one).
LONG_MARK = 100
#: Real code, not a comment (the judge is shown code with its comments stripped): a long table of the steps.
PADDING = "_STEPS = (\n" + "".join(f"    \"step {i}: every word is kept in its place and only its letters change\",\n"
                                     for i in range(90)) + ")\n"


def words_proc(op: str) -> str:
    if _mark(op) is not None:
        k = _mark(op)
        return (f'"""Write each word of the request\'s list in capitals followed by {k} marks."""\n\n'
                + (PADDING + "\n" if k >= LONG_MARK else "") + _READ
                + f'def run(value):\n    """Each word in capitals with {k} marks."""\n'
                + f'    return [w.upper() + {"!" * k!r} for w in value]\n')
    return WORDS_PROC[op]


def fork_words_reply(op: str) -> str:
    return (
        _summary("Transform each word of a list.", "Read the list and transformed each word")
        + f"```path: procedure.py\n{words_proc(op)}```\n\n"
        + "```path: broke.md\nnone\n```\n"
    )


# ── the tally world ───────────────────────────────────────────────────────────────────────────────────────────────

TALLY = {"apples": 3, "pears": 1}
MOVES = [["apples", 2], ["pears", -1]]
TALLY_REQUEST = "Apply the moves in moves.json to tally.json."
TALLY_CELL = (
    "import json\n"
    "tally = json.load(open('tally.json'))\n"
    "for name, n in json.load(open('moves.json')):\n"
    "    tally[name] = tally.get(name, 0) + n\n"
    "json.dump(tally, open('tally.json', 'w'))\n"
    "print(tally)\n"
)
TALLY_PROC = (
    '"""Apply the moves file to the tally file."""\n\n\n'
    "def run(state, env):\n"
    '    """Read the moves and the tally, write the new tally; nothing is handed back."""\n'
    '    env.worktree_workspace.read("moves.json")\n'
    '    env.worktree_workspace.read("tally.json")\n'
    '    env.worktree_workspace.write("tally.json")\n'
    "    return None\n"
)


#: The same kind of work again, in another request's words: its procedure makes the same calls (a capture of its own,
#: related to :data:`TALLY_PROC`'s by their call sequence).
TALLY_REQUEST_B = "Apply the moves in moves.json to tally.json once more."
TALLY_PROC_B = TALLY_PROC.replace('"""Apply the moves file to the tally file."""', '"""Apply a file of moves to the tally."""').replace(
    '    env.worktree_workspace.read("moves.json")\n', '    moves = env.worktree_workspace.read("moves.json")\n')


def seed_tally(workspace: Path) -> None:
    (Path(workspace) / "tally.json").write_text(json.dumps(TALLY))
    (Path(workspace) / "moves.json").write_text(json.dumps(MOVES))


def fork_tally_reply() -> str:
    return (
        _summary("Apply a file of moves to a tally file.", "Read the moves and the tally, wrote the tally")
        + f"```path: procedure.py\n{TALLY_PROC}```\n\n"
        + "```path: broke.md\nnone\n```\n"
    )


# ── a third shape: numbers in a JSON object (only to have a capture no harness relation ties to the others) ──

SUM_REQUEST = 'Add up the values.\nNUMBERS\n{"values": [3, 4, 5]}\nEND'
SUM_CELL = "values = [3, 4, 5]\nprint(sum(values))\n"
SUM_PROC = (
    '"""Add up the values the request carries."""\n\nimport json\n\n\n'
    "def read_input(text):\n"
    '    """The object after the line NUMBERS."""\n'
    '    return json.loads(text.split("NUMBERS\\n", 1)[1].split("\\nEND", 1)[0])\n\n\n'
    "def run(value):\n"
    '    """The sum of its values."""\n'
    '    return sum(value["values"])\n'
)


def fork_sum_reply() -> str:
    return (_summary("Add up a list of values.", "Summed the values") + f"```path: procedure.py\n{SUM_PROC}```\n\n"
            + "```path: broke.md\nnone\n```\n")


SUMMARY_REPLY = _summary("Part of a piece of work.", "Worked on it") + "```path: broke.md\nnone\n```\n"


# ── a stated preference and no code: the fork keeps a note candidate, with no procedure ─────────────────────────

PREF_STATED = "always keep the words in the order given"
PREF_REQUEST = f'Give the words back as they are.\nThe requester says: "{PREF_STATED}".\n{WORDS_MARK}["oak", "ash"]\nEND'
PREF_ANSWER = ["oak", "ash"]
PREF_SCOPE = "lists of words a request carries"
PREF_WHO = "the requester"
NOTE_CANDIDATE = (
    "```path: notes/order.md\n---\nkind: preference\n"
    f"applies_to: {PREF_SCOPE}\nstated_by: {PREF_WHO}\n---\n"
    f'Keep the words in the order given. From the request: "{PREF_STATED}".\n```\n\n'
)


def fork_note_reply() -> str:
    """The fork of an episode with no code: a summary and a note candidate, no ``procedure.py``."""
    return (_summary("Give a list of words back.", "Gave the words back in their order") + NOTE_CANDIDATE
            + "```path: broke.md\nnone\n```\n")


def _all_text(body: dict) -> str:
    return json.dumps(body.get("messages") or [], ensure_ascii=False)


def fork_script(body: dict) -> str:
    """The fork's model: the procedure of the world whose request the episode holds; a summary for any other."""
    text = _all_text(body)
    if PREF_STATED in text:
        return fork_note_reply()
    if "to tally.json once more" in text:
        return fork_tally_reply().replace(TALLY_PROC, TALLY_PROC_B)
    if "moves.json" in text:
        return fork_tally_reply()
    if "Add up the values." in text:
        return fork_sum_reply()
    if "Write each word in capitals." in text:
        return fork_words_reply("upper")
    if "Write each word with a capital first letter." in text:
        return fork_words_reply("title")
    if "Write each word in small letters." in text:
        return fork_words_reply("lower")
    m = re.search(r"Write each word in capitals followed by (\d+) marks\.", text)
    if m:
        return fork_words_reply(f"mark{m.group(1)}")
    if "Reverse each word" in text:
        return fork_words_reply("reverse")
    return SUMMARY_REPLY


# ── the writer lane: the kinship judge and the writer's passes ────────────────────────────────────────────────────

JUDGE_MARK = "You compare recorded procedures by what they do."
PASS_LINE = re.compile(
    r"^Pass (\S+): (?:\d+ entr(?:y|ies)|a group of \d+ related (?:captures?|entr(?:y|ies))) needs? your attention: (.*?)"
    r"(?:, with \d+ related captures? an earlier pass decided)?\.$",
    re.M,
)
NOTE_ID = "notes/text/words.md"
NOTE_RULE = (
    "---\ntitle: Transforming the words of a list\ndescription: how a list of words is read and transformed\n"
    "uses: []\n---\n"
    "## Rule\n- the list follows the line WORDS and ends at END\n- transform every word, keep the order\n\n"
)
NOTE_WHEN = (
    "## When to use\n"
    "- **Saved for:** transforming each word of a list a request carries.\n"
    "- **Recognise:** a request with a WORDS list.\n"
    "- **Not for:** sorting words.\n"
    "- **Example request:** Write each word in capitals.\n"
)
#: An evidence appendix far longer than the bound on what ``memory.get`` serves of a note (P: the Rule, ≤ 2,000).
NOTE_EVIDENCE = "## Evidence\n" + "".join(f"- case {i}: the list was read after WORDS and every word kept its place.\n"
                                          for i in range(120))
NOTE_TEXT = NOTE_RULE + NOTE_WHEN + "\n" + NOTE_EVIDENCE
#: The same note with its scope in its front matter (what it applies to, who stated it), landed with no capture.
NOTE_SCOPED_TEXT = NOTE_TEXT.replace("uses: []\n", f"uses: []\nkind: preference\napplies_to: {PREF_SCOPE}\n"
                                                    f"stated_by: {PREF_WHO}\n")
#: The same note with no ``## Rule`` section (the gate must refuse it on a raw candidate).
NOTE_NO_RULE = NOTE_TEXT.replace("## Rule\n- the list follows the line WORDS and ends at END\n"
                                 "- transform every word, keep the order\n\n", "The list follows the line WORDS.\n\n")
NOTE_README = "Notes on work with text.\n"


def _first_user(body: dict) -> str:
    for m in body.get("messages") or []:
        if isinstance(m, dict) and m.get("role") == "user":
            c = m.get("content")
            return c if isinstance(c, str) else json.dumps(c)
    return ""


def _system(body: dict) -> str:
    for m in body.get("messages") or []:
        if isinstance(m, dict) and m.get("role") == "system":
            c = m.get("content")
            return c if isinstance(c, str) else json.dumps(c)
    return ""


def is_judge(body: dict) -> bool:
    return JUDGE_MARK in _system(body) or JUDGE_MARK in _first_user(body)


def handed(body: dict) -> list[str]:
    m = PASS_LINE.search(_first_user(body))
    return [part.split(" ", 1)[0] for part in m.group(2).split(", ")] if m else []


def _n_assistant(body: dict) -> int:
    return sum(1 for m in body.get("messages") or [] if isinstance(m, dict) and m.get("role") == "assistant")


def _cell(thought: str, code: str) -> dict:
    return {"content": "", "tool_calls": [FL.tool_call("execute_code", {"thought": thought, "code": code})]}


def _strings(value: Any) -> str:
    if isinstance(value, str):
        return flat(value)
    if isinstance(value, dict):
        return "\n".join(_strings(v) for v in value.values())
    if isinstance(value, list):
        return "\n".join(_strings(v) for v in value)
    return ""


def tool_texts(body: dict) -> list[str]:
    out = []
    for m in body.get("messages") or []:
        if isinstance(m, dict) and m.get("role") == "tool":
            out.append(flat(m.get("content")))
    return out


def flat(value: Any) -> str:
    """Every string inside *value* (a message content: a string, a list of parts, or a string that is itself JSON of
    those, as some roles' tool results are), joined."""
    if isinstance(value, str):
        if value[:1] in ("{", "["):
            try:
                inner = json.loads(value)
            except ValueError:
                return value
            return value + "\n" + _strings(inner)
        return value
    if isinstance(value, dict):
        return "\n".join(flat(v) for k, v in value.items() if k in ("text", "content", "output"))
    if isinstance(value, list):
        return "\n".join(flat(v) for v in value)
    return "" if value is None else str(value)


READ_ALL = (
    "import json, os\n"
    "seen = []\n"
    "for top in ('/inputs', '/memory'):\n"
    "    for root, dirs, files in os.walk(top):\n"
    "        dirs[:] = [d for d in dirs if d != '.git']\n"
    "        for name in sorted(files):\n"
    "            path = os.path.join(root, name)\n"
    "            try:\n"
    "                with open(path, 'rb') as fh:\n"
    "                    fh.read()\n"
    "                seen.append(path)\n"
    "            except OSError:\n"
    "                pass\n"
    "print(len(seen), 'files read')\n"
)


DELEGATE = (
    "import json\n"
    "idx = json.load(open('/inputs/attention/index.json'))\n"
    "eps = [e for r in idx['entries'] for e in ((r.get('refs') or {}).get('episodes') or [])]\n"
    "print(tools.delegate(task='Check your own box and roots, then say done.', episodes=eps[:1]))\n"
)
WAIT_HELPER = (
    "got = ''\n"
    "for _ in range(12):\n"
    "    got += str(tools.wait(timeout=60))\n"
    "    if '(done)' in got or 'ended' in got:\n"
    "        break\n"
    "print(got[-1500:])\n"
)


def decide_code(ids: list[str], plan: str) -> str:
    """The writer's deciding cell. *plan*: ``land`` (a note citing every new capture's episode, with its Rule
    section), ``land_scoped`` (the same note with its scope), ``land_no_rule`` (the same note without its Rule section), ``delete`` (remove the note and declare it
    deleted), ``keep`` (nothing to keep). Every entry that is not new experience is decided ``nothing_to_keep``, one at
    a time; then the manifest and ``tools.finish``."""
    note = NOTE_NO_RULE if plan == "land_no_rule" else NOTE_SCOPED_TEXT if plan == "land_scoped" else NOTE_TEXT
    return (
        "import json, os\n"
        f"handed = {ids!r}\n"
        "idx = json.load(open('/inputs/attention/index.json'))\n"
        "rows = [r for r in idx['entries'] if r['entry'] in handed]\n"
        f"plan = {plan!r}\n"
        f"path = '/memory/{NOTE_ID}'\n"
        "new = [r for r in rows if r['kind'] == 'new_experience']\n"
        "rest = [r for r in rows if r['kind'] != 'new_experience']\n"
        "items, deleted = [], []\n"
        "eps = sorted({e for r in new for e in ((r.get('refs') or {}).get('episodes') or [])})\n"
        "if plan in ('land', 'land_no_rule', 'land_scoped') and new:\n"
        "    os.makedirs(os.path.dirname(path), exist_ok=True)\n"
        f"    open(path, 'w').write({note!r})\n"
        f"    open(os.path.join(os.path.dirname(path), 'README.md'), 'w').write({NOTE_README!r})\n"
        f"    items = [{{'item': {NOTE_ID!r}, 'kind': 'note', 'source_episodes': eps, 'form': 'checklist'}}]\n"
        "if plan == 'delete' and os.path.exists(path):\n"
        "    os.remove(path)\n"
        f"    deleted = [{NOTE_ID!r}]\n"
        "os.makedirs('/memory/.pass', exist_ok=True)\n"
        "json.dump({'items': items, 'deleted': deleted, 'support': [], 'why': 'as decided', 'summary': 'as decided'},"
        " open('/memory/.pass/manifest.json', 'w'))\n"
        "if new and items:\n"
        f"    print(tools.decide([r['entry'] for r in new], 'new_item', items=[{NOTE_ID!r}], reason='one note for both'))\n"
        "elif new:\n"
        "    rest = new + rest\n"
        "for r in rest:\n"
        "    try:\n"
        "        if plan == 'retire_bad' and r['kind'] == 'use_went_badly':\n"
        "            print(tools.decide([r['entry']], 'retire', items=list((r.get('refs') or {}).get('items') or []),"
        " reason='its recorded use failed and no change repairs it'))\n"
        "            continue\n"
        "        print(tools.decide([r['entry']], 'nothing_to_keep', reason='nothing general'))\n"
        "    except Exception as exc:\n"
        "        print(type(exc).__name__, exc)\n"
        "print(tools.finish('as decided'))\n"
    )


class WriterLane:
    """The writer lane's model: the kinship judge (answers *kin*: every candidate, or none) and the attention passes
    (cell 1 reads what it was handed; cell 2 decides by :attr:`plan`; cell 3 finishes). Every body is kept, and an
    optional *probe* cell runs first in each pass (its output is kept in :attr:`probes`)."""

    def __init__(self, plan: str | Callable[[list[str]], str] = "land", *, kin: bool | str = "same_world",
                 probe: str | Callable[[dict], str] = "") -> None:
        self.plan = plan
        self.kin = kin
        self.probe = probe
        self.judge_bodies: list[dict] = []
        self.pass_bodies: list[dict] = []
        self.probes: list[str] = []

    @property
    def passes(self) -> list[str]:
        """The pass ids the writer lane served, in order (from each pass's first message)."""
        out: list[str] = []
        for body in self.pass_bodies:
            m = PASS_LINE.search(_first_user(body))
            if m and m.group(1) not in out:
                out.append(m.group(1))
        return out

    def __call__(self, body: dict) -> Any:
        if is_judge(body):
            self.judge_bodies.append(body)
            q = _first_user(body)
            new, _, rest = q.partition("The earlier procedures:")
            blocks = re.split(r"\n\[(\d+)\]:\n", "\n" + rest)
            cands = {int(blocks[i]): blocks[i + 1] for i in range(1, len(blocks) - 1, 2)}
            if self.kin == "same_world":  # kin only between procedures of the same world (reader vs acting)
                numbers = [n for n, b in cands.items() if ("read_input" in b) == ("read_input" in new)]
            else:
                numbers = sorted(cands) if self.kin else []
            kin = [{"number": n, "why": "the same kind of work"} for n in numbers]
            return json.dumps({"kin": kin})
        self.pass_bodies.append(body)
        n, ids = _n_assistant(body), handed(body)
        probe = self.probe(body) if callable(self.probe) else self.probe
        probe = [probe] if isinstance(probe, str) and probe else list(probe or [])
        steps = probe + [READ_ALL, None, "print(tools.finish('done'))\n"]
        if probe and n == len(probe):
            self.probes.append(tool_texts(body)[-1] if tool_texts(body) else "")
        if n >= len(steps):
            return _cell("Finish.", "print(tools.finish('done'))\n")
        code = steps[n]
        if code is None:
            plan = self.plan(ids) if callable(self.plan) else self.plan
            code = decide_code(ids, plan)
        return _cell(f"Step {n}.", code)


class HelperLane:
    """The writer's helpers' model: *probe* (a cell, or a callable of the request) runs first and its output is kept in
    :attr:`out`; then ``tools.done`` with no claims. Every request body is kept."""

    def __init__(self, probe: str | Callable[[dict], str] = "") -> None:
        self.probe = probe
        self.bodies: list[dict] = []
        self.out: list[str] = []

    def __call__(self, body: dict) -> Any:
        self.bodies.append(body)
        n = _n_assistant(body)
        probe = self.probe(body) if callable(self.probe) else self.probe
        if probe and n == 0:
            return _cell("Probe.", probe)
        if probe and n == 1:
            self.out.append(tool_texts(body)[-1] if tool_texts(body) else "")
        if n <= (1 if probe else 0):
            return _cell("Done.", "print(tools.done('checked', []))\n")
        return "done"


# ── the in-process agents: actor, helpers, librarian ─────────────────────────────────────────────────────────────

MARK = "SYSJSON:"


def probe_json(expr_code: str) -> str:
    """A cell that runs *expr_code* (which fills the dict ``out``) and prints it as one marked JSON line."""
    return (
        "import json\nout = {}\n"
        "def _try(key, fn):\n"
        "    try:\n"
        "        v = fn()\n"
        "        out[key] = v if isinstance(v, (str, int, float, bool, list, dict, type(None))) else repr(v)\n"
        "    except BaseException as e:\n"
        "        out[key] = 'RAISED ' + type(e).__name__ + ': ' + str(e)\n"
        + expr_code
        + f"\nprint({MARK!r} + json.dumps({{k: (v[:6000] if isinstance(v, str) else v) for k, v in out.items()}},"
        " default=repr))\n"
    )


def marked(text: str) -> dict:
    """The last marked JSON line in *text*."""
    found = None
    for line in text.splitlines():
        at = line.find(MARK)
        if at >= 0:
            try:
                found = json.loads(line[at + len(MARK):])
            except ValueError:
                continue
    if found is None:
        raise AssertionError("no marked output in:\n" + text[-3000:])
    return found


def last_tools(call: Any) -> str:
    return "\n".join(flat(m.get("content")) for m in call.messages if m.get("role") == "tool")


def first_user(call: Any) -> str:
    for m in call.messages:
        if m.get("role") == "user":
            return flat(m.get("content"))
    return ""


def system_text(call: Any) -> str:
    for m in call.messages:
        if m.get("role") == "system":
            return flat(m.get("content"))
    return ""


def probe_turns(code: str, sink: dict, *, answer: str = "Done.", then: tuple = (),
                on_first: Callable[[Any], Any] | None = None) -> list:
    """Actor turns: the first runs *code* (a :func:`probe_json` cell) and keeps the request it was sent (and what
    *on_first* returns, in ``sink['seen']``: a read of the request's own copy while it is mounted); the second keeps the
    probe's output (``sink['out']``) and answers *answer*; *then* are later turns (e.g. after a message)."""

    def first(call):
        sink["first"] = call
        if on_first is not None:
            sink["seen"] = on_first(call)
        return reply(calls=[cell(code)])

    def second(call):
        sink["out"] = marked(last_tools(call))
        return reply(answer)

    return [first, second, *then]


def librarian_answer(fits: list[dict] | None = None, nearest: list[dict] | None = None):
    """A librarian turn that answers with ``tools.finish``."""
    body = {"fits": fits or []}
    if not fits:
        body["nearest"] = nearest or []
    return reply(calls=[cell(f"tools.finish({body!r})\n")])


def fit(item: str, why: str = "it re-runs the same work on this request", call: str | None = None) -> dict:
    return {"id": item, "call": call or f"memory.get({item!r})()", "why": why}


# ── accepted episodes ─────────────────────────────────────────────────────────────────────────────────────────────

ACCEPTED_MESSAGE = "The check passed."


def accepted_host():
    """The host side of an accepted episode: a verdict the actor sees (on its answer), then a message showing it."""
    from tests.memory.conformance import harness as H

    return H.replies([{"checker": {"label": "pass"}}, {"message": ACCEPTED_MESSAGE}])


ID16 = re.compile(r"[0-9a-f]{16}")


# ── every path or handle shown to a role opens for it; every refusal names a call that works ─────────────────────
# A role runs at most a bounded number of reader calls per model turn, so the opening runs in steps: OPEN_PREP finds
# every path and handle (kept in the cell's globals), then OPEN_STEP opens the next few until nothing is left.

#: What a probe asks for on purpose although it does not exist (to get a refusal): never "shown" to the agent.
PROBED_MISSING = ("/memory-library/notes/none.md", "/memory/notes/none.md", "/inputs/attention/none/use.json",
                  "notes/none.md", "capture:0000000000000000")

OPEN_PREP = r"""
import re as _re
_PATH = _re.compile(r"(?<![\w.:/-])(/(?:memory-library|memory|inputs)(?:/[\w.@-]+)*)")
_HANDLES = (_re.compile(r"capture:[0-9a-f]{16}"), _re.compile(r"(?<![\w/])notes/[\w./-]+\.md"),
            _re.compile(r"(?<![\w.])memory(?:\.[A-Za-z_]\w*)+:[A-Za-z_]\w*"))
_CALL = _re.compile(r"(memory\.get\(\s*'[^'<>]*'\s*\)(?:\(\))?|memory\.items\(\)|memory\.ask\(\)|"
                    r"tools\.take\(\s*'[^'<>]*'\s*\)|tools\.read\(\s*'[^'<>]*'\s*\)|"
                    r"exec\(memory\.get\(\s*'[^'<>]*'\s*\)\.text,\s*ns\))")
try:
    _COND = tuple(CONDITIONAL)
except NameError:
    _COND = ()
_SKIP = ("/memory/.pass", "/outputs") + _COND
_found = {}
for _src, _t in list(TEXTS.items()) + list(REFUSALS.items()):
    for _m in _PATH.finditer(_t):
        _p = _m.group(1).rstrip(".,);:'")
        if _t[_m.end():_m.end() + 1] in ("<", "{", "*") or _p.endswith("-") or _p.startswith(_SKIP):
            continue
        if _p not in MISSING:
            _found.setdefault(("path", _p), _src)
    for _h in _HANDLES:
        for _m in _h.finditer(_t):
            _x = _m.group(0).rstrip(".,);:'")
            if _x not in MISSING:
                _found.setdefault(("handle", _x), _src)
_TODO = [("open", k, x, src) for (k, x), src in sorted(_found.items())]
_REF = {}
for _src, _t in REFUSALS.items():
    _calls = sorted({_m.group(1) for _m in _CALL.finditer(_t)})
    _REF[_src] = {"text": _t[:600], "calls": {}}
    _TODO += [("call", _src, _c, None) for _c in _calls if not any(("'" + _m + "'") in _c for _m in MISSING)]
_FAILS, _OPENED = [], 0
out["left"] = len(_TODO)
"""

OPEN_STEP = r"""
_n = 0
while _TODO and _n < 12:
    _what, _a, _b, _c = _TODO.pop(0)
    _n += 1
    _arg = None
    try:
        if _what == "open":
            _OPENED += 1
            if _a == "path":
                tools.read(_b)
            else:
                str(memory.get(_b))
            continue
        _arg = _re.search(r"'([^']*)'", _b)
        _arg = _arg.group(1) if _arg else None
        if _b.startswith("exec("):
            ns = {}
            exec(memory.get(_arg).text, ns)  # the refusal's own recipe, as written
        elif _b.startswith("memory.get(") and _b.endswith(")()"):
            memory.get(_arg)()
        elif _b.startswith("memory.get("):
            str(memory.get(_arg))
        elif _b == "memory.items()":
            str(memory.items())
        elif _b == "memory.ask()":
            str(memory.ask())
        elif _b.startswith("tools.take("):
            tools.take(_arg)
        elif _b.startswith("tools.read("):
            tools.read(_arg)
        _REF[_a]["calls"][_b] = "ok"
    except BaseException as _e:
        _err = (type(_e).__name__ + ": " + str(_e))[:300]
        if _what == "open":
            _FAILS.append({"kind": _a, "x": _b, "from": _c, "err": _err})
        else:
            _REF[_a]["calls"][_b] = _err
out["left"] = len(_TODO)
out["opened"] = _OPENED
out["fails"] = _FAILS
out["refusals"] = _REF
"""


def open_prep_cell(texts: dict[str, str], refusals: dict[str, str]) -> str:
    """The first opening cell: every path and handle in *texts* and *refusals*, and every call each refusal names,
    queued in the cell's globals."""
    return probe_json(f"TEXTS = {texts!r}\nREFUSALS = {refusals!r}\nMISSING = {PROBED_MISSING!r}\n" + OPEN_PREP)


OPEN_STEP_CELL = probe_json(OPEN_STEP)


def open_step_cell(k: int) -> str:
    """The *k*-th opening step: the same work, a cell of its own (an agent repeating one cell is stopped)."""
    return probe_json(f"_step = {k}\n" + OPEN_STEP)


def opening_turns(first_cell: Callable[[Any], str], sink: dict, *, answer: Callable[[Any], Any] | None = None):
    """A role's script that opens everything its first cell's texts show: *first_cell(call)* gives the prep cell
    (built from that call's texts), then step cells until nothing is left; the last output goes to ``sink['out']``,
    then *answer(call)* (or a plain reply) ends it."""
    state = {"n": 0}

    def turn(call):
        state["n"] += 1
        if state["n"] == 1:
            sink["first"] = call
            return reply(calls=[cell(first_cell(call))])
        if "out" in sink:
            return answer(call) if answer else reply("Checked.")
        got = marked(last_tools(call))
        if got.get("left", 0) > 0 or "fails" not in got:
            return reply(calls=[cell(open_step_cell(state["n"]))])
        sink["out"] = got
        return answer(call) if answer else reply("Checked.")

    return turn


GATHER_REFUSALS = r"""
def _err(fn):
    try:
        fn()
        return ""
    except BaseException as e:
        return type(e).__name__ + ": " + str(e)
"""


class SteppedProbe:
    """A writer probe that opens everything its first cell found: the first cell (*first(body)*), then a step cell of its
    own each turn while anything is left, then nothing more (the pass goes on). Per pass (by its first message)."""

    def __init__(self, first: Callable[[dict], str], most: int = 40) -> None:
        self.first = first
        self.most = most
        self.done: dict[str, int] = {}

    def __call__(self, body: dict) -> list[str]:
        key = _first_user(body)[:200]
        n = _n_assistant(body)
        if n == 0:
            self.done.pop(key, None)
            return [self.first(body)]
        if key not in self.done:
            texts = tool_texts(body)
            try:
                left = marked(texts[-1] if texts else "").get("left", 0)
            except AssertionError:
                left = 0
            if left <= 0 or n >= self.most:
                self.done[key] = n
        sent = self.done.get(key, n + 1)
        return [self.first(body)] + [open_step_cell(k) for k in range(1, sent)]

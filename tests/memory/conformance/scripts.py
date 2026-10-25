"""The scripted agents and texts of the conformance surfaces (:mod:`.test_surfaces`): every memory lane's model,
answered from the request it is sent (the fork's, the writer's, the helpers'), and the surfaces' own texts. Nothing here
names a world: the surfaces are a conversation, a file of rows, a long piece of work and a late verdict.

The scripted writer works as a writer must: it reads what it was handed (every file under ``/inputs`` and the record's
messages), decides each entry (:func:`decide_code`), writes its manifest and finishes; on the record it asks the
episode's fork a question when a fork's message was handed and delegates a reading to a helper for new experience
(:func:`writer_mesh`).
"""

from __future__ import annotations

import json
import re

from tests.memory.conformance import fake_lanes as FL

FORK_REPLY = (
    "```path: summary.md\n"
    "## Asked\nPlan a weekly reading schedule for a list of novels.\n"
    "## Approach\n- Split the list over seven days by length (`cell:0` \"schedule\").\n"
    "## Worked\n- An even split by page count.\n"
    "## Failed\n- none\n"
    "## Outcome\nnot shown\n"
    "```\n\n"
    "```path: broke.md\nnone\n```\n"
)
#: an attention pass's first line: its entries, or (rev 2.1) the capture group it is handed
#: (attention_tools.first_message)
PASS_LINE = re.compile(
    r"^Pass (\S+): (?:\d+ entr(?:y|ies)|a group of \d+ related captures?) needs? your attention: (.*?)"
    r"(?:, with \d+ related captures? an earlier pass decided)?\.$",
    re.M,
)


def _first_user(body: dict) -> str:
    for m in body.get("messages") or []:
        if isinstance(m, dict) and m.get("role") == "user":
            c = m.get("content")
            return c if isinstance(c, str) else json.dumps(c)
    return ""


def handed(body: dict) -> list[str]:
    """The entry ids an attention pass's first message hands (``Pass <id>: n entries need your attention: ...``, or
    ``Pass <id>: a group of n related captures needs your attention: ...``)."""
    m = PASS_LINE.search(_first_user(body))
    assert m, _first_user(body)[:400]
    return [part.split(" ", 1)[0] for part in m.group(2).split(", ")]


READ_ALL = (
    "import json, os\n"
    "seen = []\n"
    "for root, dirs, files in os.walk('/inputs'):\n"
    "    for name in sorted(files):\n"
    "        path = os.path.join(root, name)\n"
    "        with open(path, 'rb') as fh:\n"
    "            fh.read()\n"
    "        seen.append(path)\n"
    "print(len(seen), 'files read')\n"
)


def _all_text(body: dict) -> str:
    return json.dumps(body.get("messages") or [], ensure_ascii=False)


def fork_by_request(procedure_reply: str, *, when: str, other: str = FORK_REPLY):
    """The fork's script: the procedure reply for the episode-end fork of an episode whose record mentions *when*, a
    summary-only reply for any other episode."""

    def answer(body: dict) -> str:
        return procedure_reply if when in _all_text(body) else other

    return answer


INBOX = "id,amount\n3,1.50\n1,9.00\n2,4.25\n"
TOOL_PROC = '''"""Sort the rows of the inbox file by id into the sorted file. Accepted: not shown."""


def run(state, env):
    """Read the inbox file and write the sorted file; nothing is handed back."""
    env.worktree_workspace.read("inbox.csv")
    env.worktree_workspace.write("sorted.csv")
    return None
'''
TOOL_FORK_REPLY = (
    "```path: summary.md\n"
    "## Asked\nSort the rows of a file by their id into another file.\n"
    "## Approach\n- Read the rows, sorted them by id, wrote them out (`cell:0` \"sorted(rows\").\n"
    "## Worked\n- Sorting by the integer id.\n"
    "## Failed\n- none\n"
    "## Outcome\nnot shown\n"
    "```\n\n"
    "```path: procedure.py\n" + TOOL_PROC + "```\n\n"
    "```path: broke.md\nnone\n```\n"
)
#: Another episode's procedure for the same kind of work: the same calls, another output file (so its shape differs and
#: it is a capture of its own, related to :data:`TOOL_PROC`'s by their shared call sequence).
SECOND_FILE = "sorted_rows.csv"
TOOL_PROC_B = '''"""Sort the inbox rows by id into a file of sorted rows. Accepted: not shown."""


def run(state, env):
    """Read the inbox file and write its rows, sorted by id, to a new file; nothing is handed back."""
    env.worktree_workspace.read("inbox.csv")
    env.worktree_workspace.write("sorted_rows.csv")
    return None
'''
TOOL_FORK_REPLY_B = (
    "```path: summary.md\n"
    "## Asked\nSort the rows of a file by their id into a new file.\n"
    "## Approach\n- Read the rows, sorted them by id, wrote them to a new file (`cell:0` \"sorted(\").\n"
    "## Worked\n- Sorting by the integer id.\n"
    "## Failed\n- none\n"
    "## Outcome\nnot shown\n"
    "```\n\n"
    "```path: procedure.py\n" + TOOL_PROC_B + "```\n\n"
    "```path: broke.md\nnone\n```\n"
)


def fork_sorting():
    """The fork's script when two requests sort the inbox: the second procedure for the request that writes
    :data:`SECOND_FILE`, the first for any other request on the inbox, a summary-only reply otherwise."""

    def answer(body: dict) -> str:
        text = _all_text(body)
        if SECOND_FILE in text:
            return TOOL_FORK_REPLY_B
        return TOOL_FORK_REPLY if "inbox.csv" in text else FORK_REPLY

    return answer


NOTE_ID = "notes/files/sort-rows.md"
NOTE_TEXT = (
    "---\ntitle: Sorting the rows of a file\ndescription: what to check before writing sorted rows\nuses: []\n---\n"
    "## Rule\n- read the id column as an integer, never as text\n- write every row once, the header first\n\n"
    "## When to use\n"
    "- **Saved for:** sorting the rows of a delimited file by a numeric column.\n"
    "- **Recognise:** a file of rows with an id column and a file to write them to.\n"
    "- **Not for:** merging two files.\n"
    "- **Example request:** Sort the rows of the file by id into another file.\n"
)
NOTE_README = "Checks for work on files of rows.\n"


OUTCOME_SOLVED = {"solved": True, "checks": [{"name": "rows", "passed": True, "reason": "every row in place"}]}


def decide_code(ids: list[str], *, land: bool = True) -> str:
    """A writer cell that decides each handed entry from its kind: the ``new_experience`` entries of a pass (the
    related captures it was handed together) land the note while it is not in the library (one ``new_item``
    decision closing them all, citing their episodes); a
    ``late_evidence`` while it is retires it (``retire``); every other entry is ``nothing_to_keep``, each decided on
    its own (so one the ledger refuses holds no other back); then the manifest and ``tools.finish``."""
    return (
        "import json, os\n"
        f"handed = {ids!r}\n"
        "idx = json.load(open('/inputs/attention/index.json'))\n"
        "rows = [r for r in idx['entries'] if r['entry'] in handed]\n"
        "rows += [{'entry': e, 'kind': 'unlisted'} for e in handed if e not in {r['entry'] for r in rows}]\n"
        f"note = '/memory/{NOTE_ID}'\n"
        "there = os.path.exists(note)\n"
        "land, retire, keep = [], [], []\n"
        "for r in rows:\n"
        f"    if r['kind'] == 'new_experience' and not there and {land!r}:\n"
        "        land.append(r)\n"
        "    elif r['kind'] == 'late_evidence' and there:\n"
        "        retire.append(r)\n"
        "    else:\n"
        "        keep.append(r)\n"
        "items = []\n"
        "if land:\n"
        "    os.makedirs('/memory/notes/files', exist_ok=True)\n"
        f"    open(note, 'w').write({NOTE_TEXT!r})\n"
        f"    open('/memory/notes/files/README.md', 'w').write({NOTE_README!r})\n"
        "    eps = sorted({e for r in land for e in (r.get('refs') or {}).get('episodes') or []})\n"
        f"    items = [{{'item': {NOTE_ID!r}, 'kind': 'note', 'source_episodes': eps, 'form': 'checklist'}}]\n"
        "os.makedirs('/memory/.pass', exist_ok=True)\n"
        "json.dump({'items': items, 'support': [], 'why': 'as decided', 'summary': 'as decided'},"
        " open('/memory/.pass/manifest.json', 'w'))\n"
        "if land:\n"
        f"    print(tools.decide([r['entry'] for r in land], 'new_item', items=[{NOTE_ID!r}], reason='a reusable check'))\n"
        "if retire:\n"
        f"    print(tools.decide([r['entry'] for r in retire], 'retire', items=[{NOTE_ID!r}],"
        " reason='a later verdict on its episode failed'))\n"
        "for r in keep:  # one at a time: an entry the ledger refuses (unread) never holds the others back\n"
        "    try:\n"
        "        print(tools.decide([r['entry']], 'nothing_to_keep', reason='nothing general'))\n"
        "    except Exception as exc:\n"
        "        print(type(exc).__name__, exc)\n"
        "print(tools.finish('as decided'))\n"
    )


def _cell(thought: str, code: str) -> dict:
    return {"content": "", "tool_calls": [FL.tool_call("execute_code", {"thought": thought, "code": code})]}


def _n_assistant(body: dict) -> int:
    return sum(1 for m in body.get("messages") or [] if isinstance(m, dict) and m.get("role") == "assistant")


def writer_policy(*, land: bool = True):
    """A scripted writer pass: cell 1 reads everything handed (every file under /inputs, and the record's messages);
    cell 2 decides (:func:`decide_code`)."""

    def turn(body: dict) -> dict:
        n, ids = _n_assistant(body), handed(body)
        if n == 0:
            return _cell("Read.", READ_ALL + "print(tools.messages())\n")
        if n == 1:
            return _cell("Decide.", decide_code(ids, land=land))
        return _cell("Finish.", "print(tools.finish('done'))\n")

    return turn


SUMMARY = "Sorted the inbox rows by id and wrote them out; the work goes on with the totals."


# --- the mesh on one record (A1): fork -> writer, writer -> fork answered, actor -> writer, writer -> helper -> writer

ACTOR_REPORT = "@writer the ids in inbox.csv are text; I sorted them as integers"
FORK_NOTE = "@writer the inbox ids are text and sort right only as integers"
FORK_QUESTION = "which cell converted the ids?"
FORK_ANSWER = "@writer cell 0 converted them with int() before sorting"
HELPER_TASK = "Say which cell of episode {ep} sorted the rows, with its source."
EP_IN_TASK = re.compile(r"episode (\S+?) sorted")


def _kinds(body: dict) -> list[str]:
    m = PASS_LINE.search(_first_user(body))
    return [part.split(" (", 1)[1].rstrip(")") for part in m.group(2).split(", ")] if m else []


def _last_tool_text(body: dict) -> str:
    tools = [m for m in body.get("messages") or [] if isinstance(m, dict) and m.get("role") == "tool"]
    return json.dumps(tools[-1].get("content"), ensure_ascii=False) if tools else ""


def _fork_turns(body: dict) -> int:
    """The fork's own turns: the assistant messages after its instruction (the fork continues the actor's
    conversation, so the actor's turns come before it)."""
    msgs = [m for m in body.get("messages") or [] if isinstance(m, dict)]
    at = max((i for i, m in enumerate(msgs) if m.get("role") == "user" and "fenced block" in json.dumps(m.get("content"))),
             default=-1)
    return sum(1 for m in msgs[at + 1:] if m.get("role") == "assistant")


def fork_mesh(body: dict):
    """The episode-end fork of the file work, on the record: it tells the writer what it saw, waits for the writer's
    question (the fake lane holds each wait turn 2 s while the writer is idle), answers it, then writes its blocks."""
    import time as _t

    if "inbox.csv" not in _all_text(body):
        return FORK_REPLY
    if SECOND_FILE in _all_text(body):
        return TOOL_FORK_REPLY_B  # the second sorting episode's fork writes its blocks and says nothing on the record
    n = _fork_turns(body)
    if n == 0:
        return _cell("Tell the writer.", f"tools.send({FORK_NOTE!r})")
    msgs = [m for m in body.get("messages") or [] if isinstance(m, dict)]
    received = json.dumps([m.get("content") for m in msgs if m.get("role") != "assistant"], ensure_ascii=False)
    answered = FORK_ANSWER in json.dumps([m for m in msgs if m.get("role") == "assistant"], ensure_ascii=False)
    if FORK_QUESTION in received and not answered:
        _t.sleep(1.2)  # the record's post interval per agent (1 s)
        return _cell("Answer the writer.", f"tools.send({FORK_ANSWER!r})")
    if not answered and n < 12:
        _t.sleep(2.0)
        return _cell("Wait for the writer.", "tools.wait()")
    return TOOL_FORK_REPLY


def writer_mesh(body: dict) -> dict:
    """A writer pass on the record: cell 1 reads what it was handed (files, messages) and, for new experience,
    delegates a reading to a helper; cell 2 asks the episode's fork a question when a fork's message was handed
    (``who`` for its name, ``send``, ``wait``) and waits for the helper's reply; cell 3 decides."""
    n, ids, kinds = _n_assistant(body), handed(body), _kinds(body)
    if n == 0:
        code = READ_ALL + "print(tools.messages())\n"
        if "new_experience" in kinds:
            code += (
                "import json\n"
                "idx = json.load(open('/inputs/attention/index.json'))\n"
                "eps = [e for r in idx['entries'] if r['kind'] == 'new_experience'"
                " for e in (r.get('refs') or {}).get('episodes') or []]\n"
                "if eps:\n"
                f"    print(tools.delegate(task={HELPER_TASK!r}.format(ep=eps[0]), episodes=[eps[0]]))\n"
            )
        return _cell("Read and delegate.", code)
    if n == 1:
        code = ""
        if "message" in kinds:
            code += (
                "import re\n"
                "w = tools.who()\n"
                "print(w)\n"
                "forks = [f for f in re.findall(r'^- (\\S+) \\(fork, ', str(w), re.M) if not re.search(r'-s[0-9]+$', f)]\n"
                "if forks:\n"
                f"    print(tools.send('@' + forks[0] + ' ' + {FORK_QUESTION!r}))\n"
                "    print(tools.wait())\n"
            )
        if "new_experience" in kinds:
            code += "print(tools.wait())\n"
        return _cell("Ask and wait.", code or "print(tools.messages())\n")
    if n == 2:
        return _cell("Decide.", decide_code(ids))
    return _cell("Finish.", "print(tools.finish('done'))\n")


def helper_mesh(body: dict) -> dict:
    """The helper the writer delegated to (tools in code): one message to the writer with its source, then done."""
    m = EP_IN_TASK.search(_all_text(body))
    ep = m.group(1) if m else "unknown"
    if _n_assistant(body) == 0:
        return _cell("Tell the writer.", f"print(tools.send({'@writer cell 0 sorted the rows; source ' + ep + '/cell:0'!r}))")
    claims = [{"claim": "cell 0 sorts by integer id", "sources": [f"{ep}/cell:0"]}]
    return _cell("Done.", f"print(tools.done(summary='cell 0 sorts the rows by integer id', claims={claims!r}))")

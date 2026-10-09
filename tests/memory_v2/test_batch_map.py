from unify.memory_v2.episodes import Action, Cell, Episode
from unify.memory_v2 import batch_map as bm


def _ep(cells=(), actions=(), diff="", regime="none", eid="e1"):
    return Episode(
        episode_id=eid,
        started_at="2026-10-09T00:00:00Z",
        ended_at="2026-10-09T00:01:00Z",
        build="b",
        model="m",
        effort="low",
        regime=regime,
        memory_main="abc123",
        worktree_before=None,
        worktree_after=None,
        request=["Do the thing", "obs 1"],
        transcript=[],
        cells=list(cells),
        actions=list(actions),
        worktree_diff=diff,
        memory_use={"imported": ["memory.x"], "called": []},
    )


def test_signals_cell_error_action_error_and_retry():
    ep = _ep(
        cells=[Cell(0, "x=1", "", error="Traceback ... ValueError")],
        actions=[
            Action(
                cell=0,
                channel="svc",
                method="get",
                args=[1],
                kwargs={},
                status="error",
                error="boom",
            ),
            Action(
                cell=0,
                channel="svc",
                method="get",
                args=[1],
                kwargs={},
                status="ok",
            ),
            Action(
                cell=0,
                channel="svc",
                method="get",
                args=[2],
                kwargs={},
                status="ok",
            ),
        ],
    )
    sig = bm.structural_signals(ep)
    assert {"kind": "cell_error", "cell": 0, "action": None} in sig
    assert {"kind": "action_error", "cell": 0, "action": 0} in sig
    assert {"kind": "retry_after_error", "cell": 0, "action": 1} in sig
    assert not any(s["action"] == 2 for s in sig)


def test_actor_functions_from_cells_and_later_calls():
    code0 = "def total(rows, key='amt'):\n    return sum(r[key] for r in rows)\n"
    code1 = "print(total([{'amt': 2}]))\nprint(total([]))\n"
    ep = _ep(cells=[Cell(0, code0, ""), Cell(1, code1, "2\n0\n")])
    fns = bm.actor_functions(ep)
    assert fns == [
        {
            "name": "total",
            "signature": "(rows, key='amt')",
            "source": "cell",
            "cell": 0,
            "lineno": 1,
            "cell_error": False,
            "called_later": 2,
        },
    ]


def test_function_in_failed_cell():
    ep = _ep(
        cells=[
            Cell(
                0,
                "def f(x):\n    return x\nraise SystemExit(1)\n",
                "",
                error="SystemExit: 1",
            ),
        ],
    )
    assert bm.actor_functions(ep)[0]["cell_error"] is True


def test_actor_functions_from_worktree_diff():
    diff = (
        "diff --git a/run.py b/run.py\n--- a/run.py\n+++ b/run.py\n@@ -0,0 +1,3 @@\n"
        "+def export_employees(path, month):\n+    return []\n+\n"
    )
    fns = bm.actor_functions(_ep(diff=diff))
    assert (
        fns[0]["name"] == "export_employees"
        and fns[0]["source"] == "diff"
        and fns[0]["cell"] is None
    )


def test_non_python_and_empty():
    ep = _ep(
        cells=[
            Cell(0, "ls -la", "", language="bash"),
            Cell(1, "def broken(:\n", "", error="SyntaxError"),
        ],
    )
    assert bm.actor_functions(ep) == []
    assert bm.actor_functions(_ep()) == []


def test_required_parts_and_row():
    ep = _ep(
        cells=[Cell(0, "def f():\n    return 1\n", ""), Cell(1, "print(1)", "1")],
        actions=[
            Action(
                cell=1,
                channel="svc",
                method="get",
                args=[],
                kwargs={},
                status="error",
                error="x",
            ),
        ],
    )
    sig = bm.structural_signals(ep)
    fns = bm.actor_functions(ep)
    assert bm.required_parts(ep, sig, fns) == [
        "request",
        "observation:0",
        "cell:0",
        "cell:1",  # T8: the source cell of an action that is not a read
        "action:0",
    ]
    m = bm.build_batch_map(lambda e: ep, ["e1"])
    row = m["episodes"][0]
    assert (
        m["version"] == 1
        and row["episode_id"] == "e1"
        and row["memory_main"] == "abc123"
    )
    assert row["regime"] == "none" and row["request"] == "Do the thing"
    assert row["calls"] == {"svc.get": 1} and row["items_used"] == {
        "imported": ["memory.x"],
        "called": [],
    }
    assert row["errors"] == [{"action": 0, "error": "x", "next_call": None}]
    assert row["required_parts"] == [
        "request",
        "observation:0",
        "cell:0",
        "cell:1",
        "action:0",
    ]


def test_part_text_is_canonical_json():
    ep = _ep(cells=[Cell(0, "x=1", "out")])
    assert bm.part_text(ep, "request") == '"Do the thing"'
    assert '"code": "x=1"' in bm.part_text(ep, "cell:0")


def test_error_cell_without_function_is_required():
    ep = _ep(
        cells=[Cell(0, "x = 1", "", error="Traceback ... boom"), Cell(1, "y = 2", "")],
    )
    sig = bm.structural_signals(ep)
    assert bm.required_parts(ep, sig, bm.actor_functions(ep)) == [
        "request",
        "observation:0",
        "cell:0",
    ]


_TWO_FILES = (
    "diff --git a/good.py b/good.py\n--- /dev/null\n+++ b/good.py\n@@ -0,0 +1,2 @@\n"
    "+def good(x):\n+    return x\n"
    "diff --git a/bad.py b/bad.py\n--- /dev/null\n+++ b/bad.py\n@@ -0,0 +1,1 @@\n"
    "+def bad(:\n"
)


def test_diff_parsed_per_file_and_unparsable_files_listed():
    ep = _ep(diff=_TWO_FILES)
    fns = bm.actor_functions(ep)
    assert [(f["name"], f["path"], f["lineno"]) for f in fns] == [
        ("good", "good.py", 1),
    ]
    row = bm.build_batch_map(lambda e: ep, ["e1"])["episodes"][0]
    assert row["diff_unparsed"] == ["bad.py"]
    assert row["required_parts"] == ["request", "observation:0", "diff"]


def test_diff_fragment_of_a_modified_file():
    diff = (
        "diff --git a/svc.py b/svc.py\n--- a/svc.py\n+++ b/svc.py\n@@ -10,2 +10,5 @@ class Svc:\n"
        "     x = 1\n"
        "+    def total(self, rows):\n"
        "+        return sum(rows)\n"
        "+\n"
        "     y = 2\n"
    )
    fns = bm.actor_functions(_ep(diff=diff))
    assert [(f["name"], f["path"], f["lineno"], f["signature"]) for f in fns] == [
        ("total", "svc.py", 11, "(self, rows)"),
    ]


def test_diff_with_python_lines_but_no_function_is_still_required():
    diff = "diff --git a/c.py b/c.py\n--- a/c.py\n+++ b/c.py\n@@ -1,1 +1,2 @@\n x = 1\n+y = 2\n"
    ep = _ep(diff=diff)
    assert bm.actor_functions(ep) == []
    assert bm.required_parts(ep, bm.structural_signals(ep), []) == [
        "request",
        "observation:0",
        "diff",
    ]


def _obs_ep(observations):
    ep = _ep()
    ep.request = ["Do the thing", *observations]
    return ep


def test_every_observation_is_required_and_readable():
    ep = _obs_ep(["obs A", "Verdict: wrong total"])
    sig = bm.structural_signals(ep)
    assert bm.required_parts(ep, sig, []) == [
        "request",
        "observation:0",
        "observation:1",
    ]
    assert bm.part_text(ep, "observation:1") == '"Verdict: wrong total"'
    row = bm.build_batch_map(lambda e: ep, ["e1"])["episodes"][0]
    assert row["observations"] == 2


def test_identical_observations_are_required_once_and_any_copy_credits_it():
    ep = _obs_ep(["same", "other", "same", "same"])
    assert bm.required_parts(ep, [], []) == [
        "request",
        "observation:0",
        "observation:1",
    ]
    assert bm.canonical_part(ep, "observation:2") == "observation:0"
    assert bm.canonical_part(ep, "observation:3") == "observation:0"
    assert bm.canonical_part(ep, "observation:1") == "observation:1"
    assert bm.canonical_part(ep, "cell:0") == "cell:0"


def test_row_states_which_observations_are_copies():
    ep = _obs_ep(["same", "other", "same"])
    row = bm.build_batch_map(lambda e: ep, ["e1"])["episodes"][0]
    assert row["observation_copies"] == {"2": 0}


# --- P1 T8 (a1's blocker): what the actor did is required -----------------------------------------------------


def _row(seq, role, content="", **extra):
    return {
        "seq": seq,
        "type": "message",
        "message": {"role": role, "content": content, **extra},
    }


def _cell_call(seq, cid):
    call = {
        "id": cid,
        "type": "function",
        "function": {"name": "execute_code", "arguments": "{}"},
    }
    return [
        _row(seq, "assistant", "", tool_calls=[call]),
        _row(seq + 1, "tool", "out", tool_call_id=cid),
    ]


def test_a_dialogue_action_and_the_cell_before_its_reply_are_required():
    ep = _ep(
        cells=[Cell(0, "x = 1", "1"), Cell(1, "grid = [[1]]", "[[1]]")],
        actions=[
            Action(
                cell=-1,
                channel="env",
                method="submit",
                args=[[[1]]],
                kwargs={},
                response="Verdict: wrong",
                status="ok",
                kind="dialogue",
            ),
        ],
    )
    ep.request = ["Solve the grid", "Verdict: wrong"]
    ep.transcript = [
        _row(0, "user", "Solve the grid"),
        *_cell_call(1, "c0"),
        *_cell_call(3, "c1"),
        _row(5, "assistant", '{"action": "submit", "grid": [[1]]}'),
        _row(6, "user", "Verdict: wrong"),
    ]
    assert bm.required_parts(ep, bm.structural_signals(ep), bm.actor_functions(ep)) == [
        "request",
        "observation:0",
        "cell:1",
        "action:0",
    ]


def test_a_write_action_and_its_source_cell_are_required():
    ep = _ep(
        cells=[Cell(0, "x = 1", ""), Cell(1, "open('out.json', 'w').write('{}')", "")],
        actions=[
            Action(
                cell=0,
                channel="svc",
                method="get",
                args=[],
                kwargs={},
                status="ok",
                effect="read",
            ),
            Action(
                cell=1,
                channel="workspace",
                method="write",
                args=["out.json"],
                kwargs={},
                status="ok",
                effect="write",
                kind="worktree",
            ),
        ],
    )
    assert bm.required_parts(ep, bm.structural_signals(ep), []) == [
        "request",
        "observation:0",
        "cell:1",
        "action:1",
    ]


def test_a_pure_read_episode_requires_nothing_extra():
    ep = _ep(
        cells=[Cell(0, "x = 1", "")],
        actions=[
            Action(
                cell=0,
                channel="svc",
                method="get",
                args=[],
                kwargs={},
                status="ok",
                effect="read",
            ),
        ],
    )
    assert bm.required_parts(ep, bm.structural_signals(ep), []) == [
        "request",
        "observation:0",
    ]

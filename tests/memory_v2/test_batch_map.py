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
    assert bm.required_parts(ep, sig, fns) == ["request", "cell:0", "action:0"]
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
    assert row["required_parts"] == ["request", "cell:0", "action:0"]


def test_part_text_is_canonical_json():
    ep = _ep(cells=[Cell(0, "x=1", "out")])
    assert bm.part_text(ep, "request") == '"Do the thing"'
    assert '"code": "x=1"' in bm.part_text(ep, "cell:0")

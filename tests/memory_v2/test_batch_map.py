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


def test_structural_events_cell_error_action_error_and_retry():
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
    sig = bm.structural_events(ep)
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


def test_row_keys_end_parts_and_no_reading_list_or_telemetry():
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
    m = bm.build_batch_map(lambda e: ep, ["e1"])
    row = m["episodes"][0]
    assert (
        m["version"] == 1
        and row["episode_id"] == "e1"
        and row["memory_main"] == "abc123"
    )
    assert row["regime"] == "none" and row["request"] == "Do the thing"
    assert row["request_bytes"] == len("Do the thing")
    assert row["calls"] == {"svc.get": 1}
    assert row["errors"] == [{"action": 0, "error": "x", "next_call": None}]
    assert row["structural_events"] == [
        {"kind": "action_error", "cell": 1, "action": 0},
    ]
    # where the episode ended: its last clean cell, its last action, its last observation
    assert row["end_parts"] == ["cell:1", "action:0", "observation:0"]
    # removed: the reading list, the coverage fields and the use telemetry (now /inputs/usage/<episode>.json)
    for gone in ("required_parts", "items_used", "signals", "memory_use"):
        assert gone not in row


def test_row_keys_match_the_brief_exactly():
    from unify.memory_v2.library_prompts import BATCH_ROW_KEYS

    row = bm.build_batch_map(lambda e: _ep(), ["e1"])["episodes"][0]
    assert list(row) == [k for k, _ in BATCH_ROW_KEYS]


def test_end_parts_skip_a_failed_last_cell_and_match_solution_first():
    ep = _ep(
        cells=[Cell(0, "x = 1", ""), Cell(1, "y = 2", "", error="Traceback ... boom")],
    )
    assert bm.end_parts(ep) == ["cell:0", "observation:0"]
    assert bm.solution_first(ep)[1] == bm.end_parts(ep)
    empty = _ep()
    empty.request = ["only the request"]
    assert bm.end_parts(empty) == []


def test_request_is_a_short_head_with_a_pointer():
    ep = _ep()
    long = "row " * 400
    ep.request = [long, "obs"]
    row = bm.build_batch_map(lambda e: ep, ["e1"])["episodes"][0]
    assert row["request"].startswith(long[: bm.REQUEST_HEAD_CHARS])
    assert len(row["request"]) < bm.REQUEST_HEAD_CHARS + 100
    assert "request part" in row["request"]
    assert row["request_bytes"] == len(long.encode())


def test_part_text_is_canonical_json():
    ep = _ep(cells=[Cell(0, "x=1", "out")])
    assert bm.part_text(ep, "request") == '"Do the thing"'
    assert '"code": "x=1"' in bm.part_text(ep, "cell:0")


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


def _obs_ep(observations):
    ep = _ep()
    ep.request = ["Do the thing", *observations]
    return ep


def test_every_observation_is_readable():
    ep = _obs_ep(["obs A", "Verdict: wrong total"])
    assert bm.part_text(ep, "observation:1") == '"Verdict: wrong total"'
    row = bm.build_batch_map(lambda e: ep, ["e1"])["episodes"][0]
    assert row["observations"] == 2


def test_identical_observations_name_their_first_copy():
    ep = _obs_ep(["same", "other", "same", "same"])
    assert bm.canonical_part(ep, "observation:2") == "observation:0"
    assert bm.canonical_part(ep, "observation:3") == "observation:0"
    assert bm.canonical_part(ep, "observation:1") == "observation:1"
    assert bm.canonical_part(ep, "cell:0") == "cell:0"


def test_row_states_which_observations_are_copies():
    ep = _obs_ep(["same", "other", "same"])
    row = bm.build_batch_map(lambda e: ep, ["e1"])["episodes"][0]
    assert row["observation_copies"] == {"2": 0}

import inspect
import shutil
from pathlib import Path

import pytest

from unify.memory_v2 import procedures as pr
from unify.memory_v2.episodes import Action
from unify.memory_v2.sandbox_run import PYTHON, SandboxResult
from unify.memory_v2.signals import Signal
from tests.memory_v2.test_episodes import _ep

needs_bwrap = pytest.mark.skipif(
    shutil.which("bwrap") is None,
    reason="bubblewrap required",
)


def _tree(tmp_path, source):
    tree = tmp_path / "lib"
    (tree / "memory/proc").mkdir(parents=True)
    (tree / "memory/__init__.py").write_text("")
    (tree / "memory/proc/__init__.py").write_text('"""Procedures."""\n')
    (tree / "memory/proc/jobs.py").write_text(source)
    return tree


def _none(eid):
    return []


def _nofile(sha):
    raise KeyError(sha)


# --- parse_cover -----------------------------------------------------------------------------------------


def test_parse_cover_forms_and_refusals():
    assert pr.parse_cover(["e1", 3]) == ("e1", 3)
    assert pr.parse_cover({"episode": "e1", "type": "cell", "cell": 2}) == pr.Cover(
        "e1",
        "cell",
        index=2,
    )
    c = pr.parse_cover(
        {"episode": "e1", "type": "episode", "runner": "tool", "params": {"n": 1}},
    )
    assert (c.runner, c.params, c.action) == ("tool", {"n": 1}, None)
    for bad, why in [
        ({"episode": "e1", "type": "episode", "runner": "shell"}, "goes to drafts"),
        (
            {"episode": "e1", "type": "episode", "runner": "dialogue"},
            "names the action",
        ),
        ({"episode": "e1", "type": "cell"}, "cell index"),
        ({"episode": "../x", "type": "diff"}, "episode id"),
        ({"episode": "e1", "type": "diff", "extra": 1}, "unknown cover keys"),
        (["e1", True], "a cover is"),
    ]:
        with pytest.raises(ValueError, match=why):
            pr.parse_cover(bad)


def test_param_problems_need_recorded_values():
    ep = _ep(
        episode_id="e1",
        request=["Copy in.csv"],
        actions=[
            Action(
                0,
                "worktree:ws",
                "read",
                ["in.csv"],
                {},
                {"blob_before": "a" * 64},
                "ok",
                kind="worktree",
            ),
        ],
    )
    assert pr.param_problems({"src": "in.csv", "flag": True}, ep, _nofile) == []
    assert pr.param_problems({"dst": "never.txt"}, ep, _nofile) == [
        "parameter dst holds a value the episode did not record; a procedure's parameters come from its recording",
    ]


# --- the work-tree runner ------------------------------------------------------------------------------------

WT_OK = """from pathlib import Path


def total_amounts(root, src, dst):
    rows = [line.split(",") for line in (Path(root) / src).read_text().splitlines()]
    (Path(root) / dst).write_text(str(sum(int(r[1]) for r in rows)) + "\\n")
"""
WT_WRONG = WT_OK.replace(
    "sum(int(r[1]) for r in rows)",
    "sum(int(r[1]) for r in rows) + 1",
)


def _wt(tmp_path):
    before, after = tmp_path / "snap-b", tmp_path / "snap-a"
    before.mkdir()
    (before / "in.csv").write_text("a,1\nb,2\n")
    shutil.copytree(before, after)
    (after / "out.txt").write_text("3\n")
    snaps = {"w-before": before, "w-after": after}

    def files(sha, dest):
        return Path(shutil.copytree(snaps[sha], dest)) if sha in snaps else None

    ep = _ep(
        episode_id="e1",
        worktree_before="w-before",
        worktree_after="w-after",
        request=["Total in.csv"],
        actions=[
            Action(
                0,
                "worktree:ws",
                "read",
                ["in.csv"],
                {},
                {"blob_before": "a" * 64},
                "ok",
                kind="worktree",
            ),
            Action(
                1,
                "worktree:ws",
                "write",
                ["out.txt"],
                {},
                {"blob_after": "b" * 64},
                "ok",
                "write",
                kind="worktree",
            ),
        ],
    )
    cover = pr.parse_cover(
        {
            "episode": "e1",
            "type": "episode",
            "runner": "worktree",
            "params": {"src": "in.csv", "dst": "out.txt"},
        },
    )
    return ep, cover, files


@needs_bwrap
def test_worktree_runner_compares_the_changed_paths(tmp_path):
    ep, cover, files = _wt(tmp_path)
    run = lambda src, n: pr.run_procedure(
        "memory.proc.jobs:total_amounts",
        cover,
        ep=ep,
        tree=_tree(tmp_path / f"t{n}", src),
        python=PYTHON,
        work=tmp_path / f"w{n}",
        blob=_nofile,
        worktree_files=files,
        signals=_none,
    )
    assert run(WT_OK, 1).ok
    bad = run(WT_WRONG, 2)
    assert (
        not bad.ok
        and "differs from the recorded work tree at 1 of 1 paths" in bad.reason
    )


# --- the tool runner -----------------------------------------------------------------------------------------

TOOL_OK = """def post_total(env):
    rows = env.ledger.list()["rows"]
    env.ledger.post(total=sum(rows))
"""
TOOL_MISS = TOOL_OK.replace("sum(rows)", "sum(rows) + 1")
TOOL_NO_WRITE = """def post_total(env):
    return env.ledger.list()["rows"]
"""


@needs_bwrap
def test_tool_runner_replays_the_recorded_calls(tmp_path):
    ep = _ep(
        episode_id="e2",
        request=["Post the total"],
        actions=[
            Action(0, "ledger", "list", [], {}, {"rows": [1, 2]}, "ok", "read"),
            Action(1, "ledger", "post", [], {"total": 3}, {"ok": True}, "ok", "write"),
        ],
    )
    cover = pr.parse_cover({"episode": "e2", "type": "episode", "runner": "tool"})
    run = lambda src, n: pr.run_procedure(
        "memory.proc.jobs:post_total",
        cover,
        ep=ep,
        tree=_tree(tmp_path / f"t{n}", src),
        python=PYTHON,
        work=tmp_path / f"w{n}",
        blob=_nofile,
        worktree_files=lambda s, d: None,
        signals=_none,
    )
    assert run(TOOL_OK, 1).ok
    assert "made 1 call(s) the episode did not record" in run(TOOL_MISS, 2).reason
    assert (
        "issued 0 write call(s); the episode recorded 1" in run(TOOL_NO_WRITE, 3).reason
    )


# --- the dialogue runner -------------------------------------------------------------------------------------

DIALOGUE = """def solve(observation):
    return {"grid": [[1, 2]]}
"""


@needs_bwrap
def test_dialogue_runner_needs_the_accepted_action_and_a_positive_signal(tmp_path):
    ep = _ep(
        episode_id="e3",
        request=["Solve the puzzle"],
        actions=[
            Action(
                -1,
                "dialogue:user",
                "submit",
                [],
                {"grid": [[1, 2]]},
                "Your submission was CORRECT",
                "ok",
                "unknown",
                kind="dialogue",
            ),
        ],
    )
    cover = pr.parse_cover(
        {"episode": "e3", "type": "episode", "runner": "dialogue", "action": 0},
    )
    good = [
        Signal(
            "s1",
            "e3",
            "checker",
            "pass",
            "2026-10-09T00:00:00Z",
            visible_to_actor=True,
        ),
    ]
    run = lambda src, n, sig, visible=True: pr.run_procedure(
        "memory.proc.jobs:solve",
        cover,
        ep=ep,
        tree=_tree(tmp_path / f"t{n}", src),
        python=PYTHON,
        work=tmp_path / f"w{n}",
        blob=_nofile,
        worktree_files=lambda s, d: None,
        signals=lambda e: sig,
        checker_visible=visible,
    )
    assert run(DIALOGUE, 1, good).ok
    assert "no positive signal" in run(DIALOGUE, 2, []).reason
    assert (
        "differs from the recorded action"
        in run(DIALOGUE.replace("[[1, 2]]", "[[2, 1]]"), 3, good).reason
    )
    # Amendment D: a checker verdict the bed does not declare visible to the actor is never evidence
    assert "no positive signal" in run(DIALOGUE, 4, good, visible=False).reason
    # review R1: with the switch on, a checker verdict the actor never saw (a hidden grader) is not evidence either
    hidden = [Signal("s3", "e3", "checker", "pass", "2026-10-09T00:00:00Z")]
    assert "no positive signal" in run(DIALOGUE, 6, hidden).reason
    support = [Signal("s2", "e3", "provenance", "support", "2026-10-09T00:00:00Z")]
    assert run(
        DIALOGUE,
        5,
        support,
        visible=False,
    ).ok  # a non-checker support signal still counts


# --- confinement ---------------------------------------------------------------------------------------------


def test_runners_run_model_code_only_through_run_confined(tmp_path):
    seen = []

    def fake(argv, *, ro=None, rw=None, cwd="/tmp", timeout_s=120.0, env=None):
        seen.append((argv, ro, rw, cwd, env))
        return SandboxResult(1, "", "", False)

    ep = _ep(
        episode_id="e2",
        request=["Post"],
        actions=[
            Action(1, "ledger", "post", [], {"total": 3}, {"ok": True}, "ok", "write"),
        ],
    )
    cover = pr.parse_cover({"episode": "e2", "type": "episode", "runner": "tool"})
    tree = _tree(tmp_path, TOOL_OK)
    out = pr.run_procedure(
        "memory.proc.jobs:post_total",
        cover,
        ep=ep,
        tree=tree,
        python=PYTHON,
        work=tmp_path / "w",
        blob=_nofile,
        worktree_files=lambda s, d: None,
        signals=_none,
        runner=fake,
    )
    assert not out.ok and out.reason == "it could not be run (no result)"
    [(argv, ro, rw, cwd, env)] = seen
    assert argv == [str(PYTHON), "-I", "/case/run.py"] and env == {}
    assert ro[tree] == "/memory" and sorted(ro.values()) == ["/case", "/kit", "/memory"]
    assert sorted(rw.values()) == ["/out"]
    text = inspect.getsource(pr)
    assert "subprocess" not in text and "os.system" not in text and "exec(" not in text

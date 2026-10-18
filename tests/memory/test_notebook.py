"""The world notebook (``UNIFY_MEMORY_NOTEBOOK``): consequence moments from observable changes only, lessons kept as
counted rates with the alternative's rate, trials, and the established lessons pushed at each request's start; with
the switch off every prompt is byte for byte as before."""

from __future__ import annotations

import asyncio
import json
import re
from types import SimpleNamespace

import pytest

from tests.actor.code_act.sandbox_world import needs_bwrap
from unify.memory import notebook as nb
from unify.memory.episodes import Action
from unify.memory.integration.switch import notebook_enabled, parse_notebook
from unify.settings import SETTINGS

# --- synthetic traces ---------------------------------------------------------------------------------------------


def _frame(**q):
    """A text observation with ``name: a/b`` status lines (bound 9) and free text around them."""
    lines = ["You see something.", "", "Status:"]
    lines += [f"- {k}: {v}/9" for k, v in q.items()]
    return "\n".join(lines)


def _steps(frames, actions=None, channel="dialogue:env"):
    actions = actions or ["a"] * len(frames)
    return [
        nb.Step(
            part=f"action:{i}",
            key=nb.action_key("env", "act", [a]),
            channel=channel,
            observation=f,
        )
        for i, (a, f) in enumerate(zip(actions, frames))
    ]


def _kinds(moments, kind):
    return [m for m in moments if m["kind"] == kind]


def test_quantities_read_status_lines_counts_and_json_by_structure():
    q = nb.quantities(
        "You see a tree.\n\nYour status:\n- health: 7/9\n- food: 3/9\n\nInventory:\n- stone: 4\nscore: 1.5"
    )
    assert q == {
        "health": (7, 9),
        "food": (3, 9),
        "stone": (4, None),
        "score": (1.5, None),
    }
    assert nb.quantities(
        {"balance": 12, "items": [{"qty": 2}], "ok": True, "name": "x"}
    ) == {
        "balance": (12, None),
        "items[0].qty": (2, None),
    }
    assert nb.quantities('{"left": 3}') == {"left": (3, None)}
    assert nb.quantities("no numbers here") == {} and nb.quantities(None) == {}
    assert nb.quantities({"__blob__": "abc", "bytes": 9}) == {}


def test_every_fall_counts_before_the_run_knows_a_quantitys_drift():
    det = nb.MomentDetector()
    frames = [_frame(hp=9, food=9), _frame(hp=7, food=9), _frame(hp=7, food=8)]
    moments = det.episode(_steps(frames), {"dialogue:env": _frame(hp=9, food=9)})
    falls = _kinds(moments, "fall")
    assert [
        (m["part"], m["field"], m["before"], m["after"], m["why"]) for m in falls
    ] == [
        ("action:1", "hp", 9, 7, "early"),
        ("action:2", "food", 9, 8, "early"),
    ]
    assert moments[-1] == {"part": "action:2", "kind": "end"}


def _drifting_episode(n, food_every=10, hit_at=None, hit=3):
    """food falls by 1 every *food_every* steps and is refilled at 6 (a steady drift around its level); hp falls by
    *hit* once, at step *hit_at*."""
    frames, food, hp = [], 9, 9
    for t in range(1, n + 1):
        if t % food_every == 0:
            food = food - 1 if food > 6 else 9
        if hit_at is not None and t == hit_at:
            hp -= hit
        frames.append(_frame(hp=hp, food=food))
    return frames


def test_a_fall_at_the_drift_is_quiet_and_a_fall_faster_than_it_is_a_moment():
    det = nb.MomentDetector()
    start = {"dialogue:env": _frame(hp=9, food=9)}
    det.episode(
        _steps(_drifting_episode(nb.MIN_HISTORY + 20)), start
    )  # the drift is learned (ends at full hp)
    moments = det.episode(_steps(_drifting_episode(60, hit_at=33)), start)
    falls = _kinds(moments, "fall")
    assert [(m["field"], m["part"], m["why"]) for m in falls] == [
        ("hp", "action:32", "faster-than-drift")
    ]
    assert all(
        m["field"] != "food" for m in falls
    )  # food fell 6 times at its usual pace


def test_a_quantity_episodes_end_far_below_its_level_makes_any_fall_a_moment():
    det = nb.MomentDetector()
    start = {"dialogue:env": _frame(hp=9, food=9)}
    det.episode(_steps(_drifting_episode(nb.MIN_HISTORY + 20)), start)
    # an episode that ends with hp far below where it stood
    dying = [_frame(hp=9, food=9)] * 30 + [
        _frame(hp=8, food=9),
        _frame(hp=4, food=9),
        _frame(hp=1, food=9),
    ]
    det.episode(_steps(dying), start)
    assert nb.MomentDetector.end_linked(det.fields["hp"]) is True
    assert nb.MomentDetector.end_linked(det.fields["food"]) is False
    moments = det.episode(_steps([_frame(hp=9, food=9), _frame(hp=8, food=9)]), start)
    assert [(m["field"], m["why"]) for m in _kinds(moments, "fall")] == [
        ("hp", "ended-low")
    ]


def test_an_action_that_usually_changes_something_and_does_not_is_a_moment():
    det = nb.MomentDetector()
    frames, actions, wood = [], [], 0
    for t in range(12):  # "do" changes wood every time; "look" never changes anything
        wood += 1
        frames += [_frame(wood=wood), _frame(wood=wood)]
        actions += ["do", "look"]
    frames.append(_frame(wood=wood))
    actions.append("do")  # the 13th "do" changes nothing
    moments = det.episode(_steps(frames, actions), {"dialogue:env": _frame(wood=0)})
    nc = _kinds(moments, "no-change")
    assert [m["part"] for m in nc] == [f"action:{len(frames) - 1}"]
    assert nc[0]["uses"] == 12 and nc[0]["changed"] == 12


def test_a_failed_call_is_a_moment_and_channels_are_compared_apart():
    det = nb.MomentDetector()
    steps = [
        nb.Step("action:0", "env.act", "dialogue:env", _frame(hp=9)),
        nb.Step(
            "action:1",
            "tool.q",
            "tool:db",
            {"rows": 3},
            failed=True,
            error="ValueError: bad",
        ),
        nb.Step("action:2", "env.act", "dialogue:env", _frame(hp=6)),
    ]
    moments = det.episode(steps)
    assert {"part": "action:1", "kind": "failed", "error": "ValueError: bad"} in moments
    # the tool's reply has no hp: hp is compared within the dialogue channel only
    assert [(m["part"], m["before"], m["after"]) for m in _kinds(moments, "fall")] == [
        ("action:2", 9, 6)
    ]


def test_the_detector_state_round_trips_through_json():
    det = nb.MomentDetector()
    det.episode(_steps(_drifting_episode(30)), {"dialogue:env": _frame(hp=9, food=9)})
    again = nb.MomentDetector(json.loads(json.dumps(det.state())))
    assert again.state() == json.loads(json.dumps(det.state())) and again.episodes == 1


def test_steps_of_a_recorded_episode_start_a_dialogue_from_the_request():
    from tests.memory.test_episodes import _ep

    ep = _ep(
        request=[_frame(hp=9), _frame(hp=7)],
        actions=[
            Action(
                0,
                "env",
                "act",
                ["go"],
                {},
                _frame(hp=7),
                "ok",
                "unknown",
                kind="dialogue",
            ),
            Action(
                0,
                "venmo",
                "pay",
                [],
                {"to": "a"},
                None,
                "error",
                "write",
                error="refused",
            ),
        ],
    )
    steps, start = nb.steps_of(ep)
    assert start == {"dialogue:env": _frame(hp=9)}
    assert [(s.part, s.channel, s.failed) for s in steps] == [
        ("action:0", "dialogue:env", False),
        ("action:1", "tool:venmo", True),
    ]
    assert (
        steps[0].key
        == nb.action_key("env", "act", ["go"])
        != nb.action_key("env", "act", ["stay"])
    )


def test_the_fork_text_folds_consecutive_falls_and_states_only_facts():
    moments = [
        {
            "part": "action:3",
            "kind": "fall",
            "field": "hp",
            "before": 9,
            "after": 7,
            "bound": 9,
            "why": "early",
        },
        {
            "part": "action:4",
            "kind": "fall",
            "field": "hp",
            "before": 7,
            "after": 5,
            "bound": 9,
            "why": "early",
        },
        {"part": "action:9", "kind": "no-change", "uses": 12, "changed": 10},
        {"part": "action:12", "kind": "end"},
    ]
    text = nb.fork_moments_text(moments)
    assert text.startswith(nb.MOMENTS_HEAD + "\n")
    assert "- `action:3` to `action:4`: hp fell on each action, from 9/9 to 5/9" in text
    assert "`action:9`: the action changed none of the numbers shown" in text
    assert "`action:12`: the episode's last action" in text
    assert nb.fork_moments_text([]) == ""


# --- lessons as rates ---------------------------------------------------------------------------------------------

LESSON = {
    "id": "rest-near-threat",
    "situation": "a threat is within reach",
    "action": "resting",
    "consequence": "a loss before waking",
    "alternative": "resting with no threat near",
}


def _judge(lesson, episode, n, k, n2=0, k2=0):
    return {
        "lesson": lesson,
        "episode": episode,
        "occurrences": n,
        "followed": k,
        "alternative_occurrences": n2,
        "alternative_followed": k2,
    }


def _row(state, lid):
    return {r["id"]: r for r in nb.load(state).rows()}[lid]


def test_counts_are_sums_and_majority_verdicts_never_a_true_false_rule(tmp_path):
    rep = nb.absorb(
        tmp_path,
        {"lessons": [LESSON], "judgements": [_judge(LESSON["id"], "e1", 3, 3, 4, 1)]},
        ["e1"],
    )
    assert rep == {"lessons": 1, "judgements": 1, "retired": 0, "skipped": []}
    row = _row(tmp_path, LESSON["id"])
    assert (row["support"], row["contradict"], row["status"]) == (1, 0, "weak")
    nb.absorb(
        tmp_path, {"judgements": [_judge(LESSON["id"], "e2", 2, 1)]}, ["e2"]
    )  # a tie contradicts
    nb.absorb(
        tmp_path, {"judgements": [_judge(LESSON["id"], "e3", 4, 3, 2, 0)]}, ["e3"]
    )
    row = _row(tmp_path, LESSON["id"])
    assert (row["support"], row["contradict"], row["status"]) == (2, 1, "established")
    assert (
        row["occurrences"],
        row["followed"],
        row["alt_occurrences"],
        row["alt_followed"],
    ) == (9, 7, 6, 1)


def test_a_mostly_true_consequence_stays_a_rate(tmp_path):
    """A consequence that follows 9 times in 10 is established, and shown as 9 of 10, not as a rule."""
    lesson = {
        "id": "grass",
        "situation": "facing grass",
        "action": "using it",
        "consequence": "nothing gained",
    }
    nb.absorb(
        tmp_path,
        {"lessons": [lesson], "judgements": [_judge("grass", "e1", 5, 5)]},
        ["e1"],
    )
    nb.absorb(tmp_path, {"judgements": [_judge("grass", "e2", 5, 4)]}, ["e2"])
    section = nb.render_for(tmp_path)
    assert (
        "- When facing grass, using it was followed by nothing gained in 9 of 10 times"
        in section
    )
    assert section.startswith(nb.SECTION_HEAD + "\n")


def test_a_rejudged_episode_replaces_its_judgement_and_other_episodes_are_refused(
    tmp_path,
):
    nb.absorb(
        tmp_path,
        {"lessons": [LESSON], "judgements": [_judge(LESSON["id"], "e1", 3, 3)]},
        ["e1"],
    )
    nb.absorb(tmp_path, {"judgements": [_judge(LESSON["id"], "e1", 3, 0)]}, ["e1"])
    assert _row(tmp_path, LESSON["id"])["followed"] == 0
    rep = nb.absorb(
        tmp_path,
        {
            "judgements": [
                _judge(LESSON["id"], "e9", 1, 1),  # not in this batch
                _judge("nope", "e2", 1, 1),  # no such lesson
                _judge(LESSON["id"], "e2", 1, 2),  # followed > occurrences
                {
                    "lesson": LESSON["id"],
                    "episode": "e2",
                    "occurrences": True,
                    "followed": 0,
                },
            ],
            "lessons": [
                {"id": "Bad Id!", "situation": "s", "action": "a", "consequence": "c"},
                {"id": "x"},
            ],
        },
        ["e2"],
    )
    assert rep["judgements"] == 0 and rep["lessons"] == 0 and len(rep["skipped"]) == 6
    assert _row(tmp_path, LESSON["id"])["episodes"] == 1


def test_contradicted_lessons_drop_and_retired_ones_are_not_shown(tmp_path):
    lesson = dict(LESSON, id="never-fight", consequence="death")
    doc = {
        "lessons": [lesson],
        "judgements": [_judge("never-fight", e, 2, 0) for e in ("e1", "e2")],
    }
    nb.absorb(tmp_path, doc, ["e1", "e2"])
    assert _row(tmp_path, "never-fight")["status"] == "dropped"
    nb.absorb(
        tmp_path,
        {
            "lessons": [LESSON],
            "judgements": [_judge(LESSON["id"], e, 1, 1) for e in ("e3", "e4")],
        },
        ["e3", "e4"],
    )
    assert "resting was followed by" in nb.render_for(
        tmp_path
    ) and "death" not in nb.render_for(tmp_path)
    nb.absorb(tmp_path, {"retire": {LESSON["id"]: "same as another"}}, [])
    assert (
        _row(tmp_path, LESSON["id"])["status"] == "retired"
        and nb.render_for(tmp_path) == ""
    )


def test_a_trial_is_shown_labelled_until_its_counts_decide(tmp_path):
    trial = {
        "id": "close-in-first",
        "situation": "about to rest with open sides",
        "action": "closing the open sides first",
        "consequence": "no loss before waking",
        "alternative": "resting as it is",
        "trial": True,
        "remedy_for": "rest-near-threat",
    }
    nb.absorb(tmp_path, {"lessons": [trial]}, [])
    section = nb.render_for(tmp_path)
    assert (
        nb.TRIALS_HEAD in section
        and "- Trial: when about to rest with open sides" in section
    )
    assert "not tried yet" in section and nb.SECTION_HEAD in section
    nb.absorb(
        tmp_path, {"judgements": [_judge("close-in-first", "e1", 2, 2, 5, 1)]}, ["e1"]
    )
    assert (
        "tried 2 times, 2 with that result; resting as it is gave it in 1 of 5 times"
        in nb.render_for(tmp_path)
    )
    nb.absorb(tmp_path, {"judgements": [_judge("close-in-first", "e2", 1, 1)]}, ["e2"])
    section = nb.render_for(tmp_path)  # established: a lesson now, no longer a trial
    assert (
        "- Trial:" not in section
        and "closing the open sides first was followed by" in section
    )


def test_the_writer_input_carries_counts_and_the_batch_moments(tmp_path):
    nb.absorb(
        tmp_path,
        {"lessons": [LESSON], "judgements": [_judge(LESSON["id"], "e1", 1, 1)]},
        ["e1"],
    )
    (tmp_path / "notebook" / "moments").mkdir(parents=True)
    (tmp_path / "notebook" / "moments" / "e2.json").write_text(
        json.dumps([{"part": "action:0", "kind": "end"}])
    )
    doc = nb.writer_input(tmp_path, ["e2", "e3"])
    assert doc["moments"] == {"e2": [{"part": "action:0", "kind": "end"}], "e3": []}
    assert doc["lessons"][0]["id"] == LESSON["id"] and doc["lessons"][0]["support"] == 1


# --- the switch and the texts -------------------------------------------------------------------------------------

WORLD_NAMES = re.compile(
    r"\b(crafter|minecraft|zombie|skeleton|sapling|tau2?|airline|retail|office|cl-?bench|arc|balrog|alfworld|"
    r"appworld|scienceworld|grid|demonstrations?)\b",
    re.IGNORECASE,
)


@pytest.mark.parametrize(
    "name", ["MOMENTS_HEAD", "SECTION_HEAD", "TRIALS_HEAD", "WRITER_BRIEF"]
)
def test_the_added_texts_are_general_world(name):
    from tests.actor.code_act.test_prompt_generality import _findings

    text = getattr(nb, name)
    assert _findings({name: text}) == []
    assert WORLD_NAMES.search(text) is None
    assert "example" not in text.lower()


def test_the_switch_parses_off_by_default():
    assert parse_notebook("") == "off" and parse_notebook("ON") == "on"
    with pytest.raises(ValueError, match="UNIFY_MEMORY_NOTEBOOK"):
        parse_notebook("maybe")
    assert notebook_enabled(SimpleNamespace()) is False
    assert (
        getattr(SETTINGS, "UNIFY_MEMORY_NOTEBOOK") == "off"
        and nb.notebook_on() is False
    )
    assert nb.notebook_on(SimpleNamespace(UNIFY_MEMORY_NOTEBOOK="on")) is True


# --- the request: moments for the fork, the pushed section ---------------------------------------------------------


def _run_with_episode(tmp_path):
    from tests.memory.test_episodes import _ep
    from unify.memory.integration import request as request_mod
    from unify.memory.integration.paths import Paths

    paths = Paths.under(tmp_path / "home")
    run = request_mod.RequestRun("r", paths)
    ep = _ep(
        episode_id="e-0001",
        request=[_frame(hp=9)],
        actions=[
            Action(
                0,
                "env",
                "act",
                ["go"],
                {},
                _frame(hp=6),
                "ok",
                "unknown",
                kind="dialogue",
            ),
            Action(
                0,
                "env",
                "act",
                ["go"],
                {},
                _frame(hp=6),
                "ok",
                "unknown",
                kind="dialogue",
            ),
        ],
    )
    run._recorded = (ep, "sha")
    return run, paths


def test_with_the_switch_off_the_fork_gets_no_moments_and_nothing_is_written(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY_NOTEBOOK", "off")
    run, paths = _run_with_episode(tmp_path)
    assert run._notebook_moments(lambda _: None) == ""
    assert not (paths.state_dir / "notebook").exists()


def test_with_the_switch_on_the_fork_message_ends_with_the_moments(
    tmp_path, monkeypatch
):
    from unify.memory.integration import fork

    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY_NOTEBOOK", "on")
    run, paths = _run_with_episode(tmp_path)
    text = run._notebook_moments(lambda _: None)
    assert text.startswith(nb.MOMENTS_HEAD)
    assert (
        "`action:0`: hp fell from 9/9 to 6/9" in text
        and "`action:1`: the episode's last action" in text
    )
    assert nb.read_moments(paths.state_dir, "e-0001")[-1] == {
        "part": "action:1",
        "kind": "end",
    }
    assert (paths.state_dir / "notebook" / "detector.json").is_file()
    message = fork.fork_message("PARTS\n\n" + text)
    assert message.startswith(fork.fork_instruction()) and message.endswith(text)


def _open(tmp_path, monkeypatch, notebook):
    from tests.memory.test_layout import LIB
    from tests.memory.test_library_helper import _commit
    from unify.memory.gitio import Repo
    from unify.memory.integration import request as request_mod
    from unify.memory.integration.paths import Paths
    from unify.memory.integration.state import State

    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY", "on")
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY_NOTEBOOK", notebook)
    paths = Paths.under(tmp_path / "home")
    if not paths.memory.exists():
        _commit(Repo.init_bare(paths.memory), LIB, "seed")
    mem = Repo(paths.memory)
    run = request_mod.RequestRun("r", paths)
    run.state = State.load(paths.state)
    run.pin = mem.head()
    run.library_mode = True
    run._shape_rows = lambda consolidate, library_mode=False: {}
    run._open_library_mode(None)
    return run, paths


def test_established_lessons_are_pushed_at_the_start_and_off_is_byte_identical(
    tmp_path, monkeypatch
):
    from unify.memory.integration import prompt

    off, paths = _open(tmp_path, monkeypatch, "off")
    base = prompt.render_memory_library_mode(paths.checkout)[0]
    assert base and off.index == base
    nb.absorb(
        paths.state_dir,
        {"lessons": [LESSON], "judgements": [_judge(LESSON["id"], "e1", 1, 1)]},
        ["e1"],
    )
    # a weak lesson is not pushed, and nothing changes the bytes
    assert _open(tmp_path, monkeypatch, "on")[0].index == base
    assert _open(tmp_path, monkeypatch, "off")[0].index == base
    nb.absorb(
        paths.state_dir, {"judgements": [_judge(LESSON["id"], "e2", 2, 2)]}, ["e2"]
    )
    on = _open(tmp_path, monkeypatch, "on")[0]
    assert on.index == base + "\n" + nb.render_for(paths.state_dir)
    assert "resting was followed by a loss before waking in 3 of 3 times" in on.index
    assert on.shown["renderer"].endswith("+notebook") and on.shown["bytes"] == len(
        on.index.encode()
    )
    off = _open(tmp_path, monkeypatch, "off")[
        0
    ]  # the same notebook with the switch off: the bytes as before
    assert off.index == base and not off.shown["renderer"].endswith("+notebook")


# --- the writer pass ----------------------------------------------------------------------------------------------


def test_the_writer_brief_gains_the_notebook_section_only_with_the_switch(tmp_path):
    from unify.memory import library_prompts as pv
    from tests.memory.test_writer_library_brief import _briefs_integrated
    from tests.memory.test_writer_pass import Turns, _writer

    _briefs_integrated()
    (tmp_path / "off").mkdir()
    (tmp_path / "on").mkdir()
    _, _, off = _writer(tmp_path / "off", Turns([]), library_mode=True)
    assert off._brief_library_mode() == pv.write_brief_now(finish_capped=True)
    _, _, on = _writer(
        tmp_path / "on", Turns([]), library_mode=True, notebook_state=str(tmp_path)
    )
    assert (
        on._brief_library_mode()
        == pv.write_brief_now(finish_capped=True) + nb.WRITER_BRIEF
    )
    on.cfg.role = "curate"
    assert nb.WRITER_BRIEF not in on._brief_library_mode()


def test_the_pass_absorbs_the_writers_notebook_file_from_its_box(tmp_path):
    from unify.memory.trigger import PassRequest
    from tests.memory.test_writer_pass import Turns, _writer

    _, _, writer = _writer(
        tmp_path, Turns([]), library_mode=True, notebook_state=str(tmp_path / "state")
    )
    box = tmp_path / "box"
    (box / ".pass").mkdir(parents=True)
    doc = {
        "lessons": [LESSON],
        "judgements": [
            _judge(LESSON["id"], "e1", 2, 2),
            _judge(LESSON["id"], "e7", 1, 1),
        ],
    }
    (box / ".pass" / "notebook.json").write_text(json.dumps(doc))
    req = PassRequest("incremental", "svc", ["e1"], False)
    writer._role = "write"
    writer._absorb_notebook(box, req, "p1")
    row = _row(tmp_path / "state", LESSON["id"])
    assert (row["occurrences"], row["episodes"]) == (2, 1)  # e7 is not in the batch
    # a link in place of the file is never followed
    (box / ".pass" / "notebook.json").unlink()
    (tmp_path / "elsewhere.json").write_text(
        json.dumps({"lessons": [dict(LESSON, id="other")]})
    )
    (box / ".pass" / "notebook.json").symlink_to(tmp_path / "elsewhere.json")
    writer._absorb_notebook(box, req, "p2")
    assert "other" not in nb.load(tmp_path / "state").lessons


@needs_bwrap
@pytest.mark.timeout(240)
def test_a_store_pass_stages_the_notebook_and_merges_what_its_cell_writes(tmp_path):
    from tests.memory.test_writer_library_brief import _briefs_integrated
    from tests.memory.test_writer_library_tools import EPS
    from tests.memory.test_writer_pass import Turns, _call, _writer
    from unify.memory.trigger import PassRequest

    _briefs_integrated()
    state = tmp_path / "state"
    nb.absorb(state, {"lessons": [LESSON]}, [])
    doc = {"judgements": [_judge(LESSON["id"], "e1", 3, 2)]}
    code = (
        "import json, os\n"
        "print(sorted(json.load(open('/inputs/notebook.json'))))\n"
        "os.makedirs('/memory/.pass', exist_ok=True)\n"
        f"open('/memory/.pass/notebook.json', 'w').write({json.dumps(json.dumps(doc))})\n"
    )
    model = Turns(
        [
            _call("c", "execute_code", {"code": code}),
            _call("f", "finish", {"summary": "x"}),
        ]
    )
    _, _, writer = _writer(
        tmp_path, model, library_mode=True, notebook_state=str(state)
    )
    writer.load = EPS.__getitem__
    asyncio.run(
        writer.run(PassRequest("incremental", "svc", ["e1", "e2"], False), "b1")
    )
    assert "['lessons', 'moments']" in model.outputs["c"]
    row = _row(state, LESSON["id"])
    assert (row["occurrences"], row["followed"], row["support"]) == (3, 2, 1)

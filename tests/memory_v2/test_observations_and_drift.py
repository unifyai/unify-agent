"""``UNIFY_MEMORY_V2_DIALOGUE_DRIFT`` and ``UNIFY_MEMORY_V2_OBSERVATIONS`` (both off by default).

Drift: with ``structure`` a dialogue counterpart whose messages only change length (a larger grid) keeps one
fingerprint shape, so its channel never turns suspect for that; a change of structure still bumps it.
Observations: before each cell the export holds the counterpart's messages so far, as the recorder keeps an
observation, and the ``memory`` helper reads them back; off, nothing is written and the helper says so.
"""

import json
from pathlib import Path

import pytest

from unify.memory_v2 import catalogue
from unify.memory_v2.fingerprint import Generations
from unify.memory_v2.integration import hooks, switch
from unify.memory_v2.integration.adapters.dialogue import (
    counterpart_messages,
    dialogue_actions,
    observation_fingerprint,
    observation_shape,
    response_shape,
)
from unify.memory_v2.integration.checkout import checkout_diff
from unify.memory_v2.redact import Redactor
from unify.settings import SETTINGS
from tests.memory_v2.integration.test_request import _abort, _begin, mv2  # noqa: F401
from tests.memory_v2.test_catalogue import _helper, library  # noqa: F401


def _line(seq, role, content="", **extra):
    return {
        "seq": seq,
        "ts": f"2026-10-09T00:00:{seq:02d}+00:00",
        "type": "message",
        "message": {"role": role, "content": content, **extra},
    }


def _grid(n: int) -> str:
    rows = "\n".join(" ".join("0" for _ in range(n)) for _ in range(n))
    return f"Test input:\n{rows}\nReply with a JSON submit."


def _session(*observations: str) -> list[dict]:
    lines = [_line(0, "user", "Solve the puzzle.")]
    for i, obs in enumerate(observations):
        lines += [
            _line(2 * i + 1, "assistant", '{"action": "request_demos"}'),
            _line(2 * i + 2, "user", obs),
        ]
    return lines


# --- drift ----------------------------------------------------------------------------------------------


def test_the_switches_default_to_the_frozen_behaviour():
    assert switch.parse_dialogue_drift("") == "lines"
    assert switch.parse_dialogue_drift("Structure") == "structure"
    assert switch.parse_observations("") == switch.parse_observations("off") == ""
    assert switch.parse_observations("ON") == "on"
    for bad in ("both", "1"):
        with pytest.raises(ValueError):
            switch.parse_dialogue_drift(bad)
        with pytest.raises(ValueError):
            switch.parse_observations(bad)
    assert SETTINGS.UNIFY_MEMORY_V2_DIALOGUE_DRIFT == "lines"
    assert SETTINGS.UNIFY_MEMORY_V2_OBSERVATIONS == ""
    assert switch.dialogue_drift_lines() is True and switch.observations_on() is False


def test_structure_shapes_leave_out_the_line_bucket_only():
    small, large = _grid(3), _grid(30)
    assert observation_shape(small) != observation_shape(large)  # the frozen default
    assert (
        observation_shape(small, lines=False)
        == observation_shape(large, lines=False)
        == "counter=n;json=none"
    )
    assert observation_shape(small) == "lines=4-7;counter=n;json=none"  # unchanged
    assert (
        response_shape({"correct": True}, lines=False)
        == "counter=n;json=object:{correct:bool}"
    )
    assert observation_shape("x\n(step 3/9)", lines=False) == "counter=y;json=none"


def test_grid_size_alone_never_turns_the_channel_suspect_under_structure():
    def fp(*obs, lines):
        return observation_fingerprint(
            dialogue_actions(_session(*obs), "env"),
            lines=lines,
        )

    frozen, structure = Generations(), Generations()
    assert frozen.observe(fp(_grid(3), lines=True)) == set()
    assert structure.observe(fp(_grid(3), lines=False)) == set()
    assert frozen.observe(fp(_grid(30), lines=True)) == {
        "env",
    }  # the false drift seen online
    assert structure.observe(fp(_grid(30), lines=False)) == set()
    # a real change of structure (a JSON counterpart) still drifts
    assert structure.observe(
        fp('{"correct": false, "attempts_left": 1}', lines=False),
    ) == {"env"}


# --- observations: the adapter's view ----------------------------------------------------------------------


def test_counterpart_messages_are_the_request_then_each_observation_as_recorded():
    lines = _session(_grid(2), '{"correct": false}')
    lines.insert(2, _line(90, "user", "Progress: 1 step", _loop_authored=True))
    got = counterpart_messages(lines)
    assert got == ["Solve the puzzle.", _grid(2), {"correct": False}]
    recorded = [a.response for a in dialogue_actions(lines, "env")]
    assert got[1:] == recorded  # exactly the recorder's observation values


def test_counterpart_messages_are_redacted_capped_and_bounded():
    key = "sk-or-v1-" + "a" * 64
    red = Redactor({"APP_PASSWORD": "hunter2-very-secret"})
    lines = _session(f"your key is {key}; password hunter2-very-secret", "row\n" * 500)
    got = counterpart_messages(lines, redactor=red, max_observation_chars=100)
    assert key not in json.dumps(got) and "hunter2-very-secret" not in json.dumps(got)
    assert "<secret:APP_PASSWORD>" in got[1] and "<redacted:key-shaped>" in got[1]
    assert len(got[2]) <= 100 and "elided" in got[2]
    many = _session(*[f"obs {i}" for i in range(10)])
    kept = counterpart_messages(many, max_messages=4)
    assert kept == ["Solve the puzzle.", "obs 7", "obs 8", "obs 9"]


# --- observations: the helper ---------------------------------------------------------------------------


def test_off_the_exported_helper_is_the_frozen_file_byte_for_byte(
    library,
):  # noqa: F811
    _, _, export, _, _ = library  # written with the default: observations off
    here = Path(catalogue.__file__).parent / "memory_helper.py"
    assert (export / catalogue.HELPER).read_bytes() == here.read_bytes()
    assert catalogue.helper_bytes() == here.read_bytes()
    memory = _helper(export)
    assert not hasattr(memory, "observation")
    assert "memory.observation()" not in memory.catalog()
    assert "memory.observation()" not in (memory.__doc__ or "")


def test_on_the_helper_reads_the_observations_the_harness_kept(library):  # noqa: F811
    _, _, export, _, _ = library
    catalogue.write_generated(export, observations=True)
    memory = _helper(export)
    assert memory.OBSERVATIONS_FILE == catalogue.OBSERVATIONS
    assert memory.observations() == [] and memory.observation() is None
    target = export / catalogue.OBSERVATIONS
    target.write_text(
        json.dumps({"version": 1, "messages": ["Solve.", {"grid": [[1]]}]}),
    )
    assert memory.observations() == ["Solve.", {"grid": [[1]]}]
    assert memory.observation() == {"grid": [[1]]}
    assert "`memory.observation()` is the latest message" in memory.catalog()
    assert "memory.observation()" in memory.__doc__
    assert "The current one: memory.observation()." in memory.describe("parse_feedback")
    assert "memory.observation()" not in memory.describe("read_ledger")
    assert catalogue.reserved(catalogue.OBSERVATIONS)  # Sol can never write it


def test_a_broken_observations_file_says_so(library):  # noqa: F811
    _, _, export, _, _ = library
    catalogue.write_generated(export, observations=True)
    (export / catalogue.OBSERVATIONS).write_text("{not json")
    with pytest.raises(RuntimeError, match="observations"):
        _helper(export).observations()


# --- observations: the request run ----------------------------------------------------------------------


def _catalogue_run(mv2, monkeypatch, observations="on"):  # noqa: F811
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY_V2_SURFACING", "catalogue")
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY_V2_OBSERVATIONS", observations)
    return _begin(mv2)


def _write_transcript(run, lines) -> Path:
    path = run.paths.home / "transcripts" / f"{run.episode_id}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in lines))
    return path


def test_each_cell_sees_the_latest_observation_and_the_diff_leaves_it_out(
    mv2,
    monkeypatch,
):  # noqa: F811
    run = _catalogue_run(mv2, monkeypatch)
    try:
        target = Path(run.paths.checkout) / catalogue.OBSERVATIONS
        helper = (Path(run.paths.checkout) / catalogue.HELPER).read_bytes()
        assert helper == catalogue.helper_bytes(observations=True)
        _write_transcript(run, _session(_grid(2)))
        mv2.ctx.run(hooks.before_cell)
        assert json.loads(target.read_text())["messages"] == [
            "Solve the puzzle.",
            _grid(2),
        ]
        _write_transcript(run, _session(_grid(2), _grid(4)))
        mv2.ctx.run(hooks.before_cell)
        assert json.loads(target.read_text())["messages"][-1] == _grid(4)
        diff = checkout_diff(
            run.paths.memory,
            run.pin,
            run.paths.checkout,
            run.stores.blobs,
            generated=run.generated,
        )
        assert catalogue.OBSERVATIONS not in diff
        target.write_text("edited by a cell")
        diff = checkout_diff(
            run.paths.memory,
            run.pin,
            run.paths.checkout,
            run.stores.blobs,
            generated=run.generated,
        )
        assert (
            catalogue.OBSERVATIONS in diff
        )  # a cell's edit is the request's, as for any generated file
    finally:
        _abort(mv2, run)


def test_off_or_index_surfacing_writes_nothing(mv2, monkeypatch):  # noqa: F811
    run = _catalogue_run(mv2, monkeypatch, observations="")
    try:
        _write_transcript(run, _session(_grid(2)))
        mv2.ctx.run(hooks.before_cell)
        assert not (Path(run.paths.checkout) / catalogue.OBSERVATIONS).exists()
    finally:
        _abort(mv2, run)
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY_V2_SURFACING", "index")
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY_V2_OBSERVATIONS", "on")
    run = _begin(mv2)
    try:
        _write_transcript(run, _session(_grid(2)))
        mv2.ctx.run(hooks.before_cell)
        assert not (Path(run.paths.checkout) / catalogue.OBSERVATIONS).exists()
    finally:
        _abort(mv2, run)


def test_before_cell_is_inert_without_a_run_and_never_raises(
    mv2,
    monkeypatch,
):  # noqa: F811
    hooks.before_cell()  # no run in this context
    run = _catalogue_run(mv2, monkeypatch)
    try:
        monkeypatch.setattr(type(run), "refresh_observations", lambda self: 1 / 0)
        mv2.ctx.run(hooks.before_cell)  # logged, not raised
    finally:
        _abort(mv2, run)

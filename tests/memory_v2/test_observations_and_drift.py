"""``UNIFY_MEMORY_V2_DIALOGUE_DRIFT`` and ``UNIFY_MEMORY_V2_OBSERVATIONS`` (both off by default).

Drift: with ``structure`` a dialogue counterpart whose messages only change length (a larger grid) keeps one
fingerprint shape, so its channel never turns suspect for that; a change of structure still bumps it.
Observations: the exported ``memory`` helper says where an ``observation`` input comes from (the team record's
entries from ``user``); off, the helper is the frozen file byte for byte.
"""

from pathlib import Path

import pytest

from unify.memory_v2 import catalogue
from unify.memory_v2.fingerprint import Generations
from unify.memory_v2.integration import switch
from unify.memory_v2.integration.adapters.dialogue import (
    dialogue_actions,
    observation_fingerprint,
    observation_shape,
    response_shape,
)
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


# --- observations: the helper's pointer --------------------------------------------------------------------

POINTER = 'record.read(author="user", limit=1)'
ALL = 'record.read(author="user", limit=10**6)'


def test_off_the_exported_helper_is_the_frozen_file_byte_for_byte(
    library,
):  # noqa: F811
    _, _, export, _, _ = library  # written with the default: observations off
    here = Path(catalogue.__file__).parent / "memory_helper.py"
    assert (export / catalogue.HELPER).read_bytes() == here.read_bytes()
    assert catalogue.helper_bytes() == here.read_bytes()
    memory = _helper(export)
    assert POINTER not in memory.catalog()
    assert POINTER not in memory.describe("parse_feedback")
    assert POINTER not in (memory.__doc__ or "")


def test_on_the_helper_points_observation_inputs_at_the_record(library):  # noqa: F811
    _, _, export, _, _ = library
    catalogue.write_generated(export, observations=True)
    memory = _helper(export)
    assert POINTER in memory.catalog()  # the input-forms line
    assert POINTER in memory.__doc__
    assert f'The current one: `{POINTER}[0]["text"]`' in memory.describe(
        "parse_feedback",
    )
    assert (
        ALL in memory.catalog() and ALL in memory.__doc__
    )  # every entry, past the default newest 50
    assert POINTER not in memory.describe(
        "read_ledger",
    )  # a path-input function: no pointer
    # nothing else of the helper changed: the frozen catalogue lines are all still there
    frozen = _helper_off(library)
    for line in frozen.catalog().splitlines():
        assert line in memory.catalog()


def _helper_off(library):  # noqa: F811
    _, _, export, _, _ = library
    catalogue.write_generated(export)
    return _helper(export)


def _catalogue_run(mv2, monkeypatch, observations):  # noqa: F811
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY_V2_SURFACING", "catalogue")
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY_V2_OBSERVATIONS", observations)
    return _begin(mv2)


@pytest.mark.parametrize("observations", ["on", ""])
def test_the_request_exports_the_helper_the_switch_names(
    mv2,
    monkeypatch,
    observations,
):  # noqa: F811
    run = _catalogue_run(mv2, monkeypatch, observations)
    try:
        helper = (Path(run.paths.checkout) / catalogue.HELPER).read_bytes()
        assert helper == catalogue.helper_bytes(observations=observations == "on")
        assert not (
            Path(run.paths.checkout) / ".memory" / "observations.json"
        ).exists()  # no data, no file
    finally:
        _abort(mv2, run)

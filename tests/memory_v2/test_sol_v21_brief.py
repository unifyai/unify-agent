"""P7 Task 3: under v2.1 the writer's system message is its role's brief alone (spec §12); commits carry
Why/Episodes/Items; a v2.1 run with contradicting v2 switches is refused by name."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from unify.memory_v2 import prompts_v21 as pv
from unify.memory_v2 import qa as _qa
from unify.memory_v2.integration import prompt
from unify.memory_v2.integration.request import MemoryV2Unavailable, check_v21_switches
from unify.memory_v2.sol_pass import _v21_trailers, sol_system
from unify.memory_v2.trigger import PassRequest
from tests.memory_v2.test_sol_pass import Turns, _call, _sol
from tests.memory_v2.test_sol_v21_tools import EPS


def _finishing_pass(tmp_path, **cfg):
    _, _, sol = _sol(tmp_path, Turns([_call("f", "finish", {"summary": "x"})]), **cfg)
    sol.load = EPS.__getitem__
    asyncio.run(sol.run(PassRequest("incremental", "svc", ["e1", "e2"], False), "b1"))
    return sol


def _briefs_integrated():
    """write_brief_now and curate_brief_now read P2's, P4's and P6's constants: run once integrated."""
    for name in (
        "unify.memory_v2.repair",
        "unify.memory_v2.code_lint",
        "unify.memory_v2.library_index",
        "unify.memory_v2.curate",
    ):
        pytest.importorskip(name)


def test_the_prompt_section_uses_the_reviewed_guide():
    assert prompt.GUIDE_V21 is pv.GUIDE_V21


def test_v2_system_message_is_unchanged(tmp_path):
    sol = _finishing_pass(tmp_path, v21=False)
    assert sol.messages[0]["content"] == _qa.system(
        sol_system(**sol._switches()),
        sol._qa,
    )


def test_v21_system_message_is_the_write_brief(tmp_path):
    _briefs_integrated()
    sol = _finishing_pass(tmp_path, v21=True)
    assert sol.messages[0] == {"role": "system", "content": pv.write_brief_now()}
    assert "Manifest rules for consolidators" not in sol.messages[0]["content"]


def test_a_curate_pass_gets_the_curate_brief(tmp_path):
    _briefs_integrated()
    _, _, sol = _sol(tmp_path, Turns([]), v21=True)
    sol.cfg.role = "curate"  # P6's PassConfig field
    assert sol._brief_v21() == pv.curate_brief_now()


def test_v21_trailers_from_the_manifest():
    m = {
        "items": [
            {"item": "memory.text.dates:parse_date"},
            {"item": "notes/text/dates.md"},
            {"x": 1},
        ],
        "why": "dates repeat\nin four episodes",
    }
    assert _v21_trailers(m, ["e1", "e2"]) == {
        "Why": ["dates repeat in four episodes"],
        "Episodes": ["e1", "e2"],
        "Items": ["memory.text.dates:parse_date", "notes/text/dates.md"],
    }
    assert _v21_trailers(None, []) == {
        "Why": ["(not given)"],
        "Episodes": [],
        "Items": ["(none)"],
    }
    assert len(_v21_trailers({"why": "w" * 900}, [])["Why"][0]) == 200


def test_a_request_with_contradicting_switches_is_refused_by_name():
    check_v21_switches(SimpleNamespace(UNIFY_MEMORY_V21="on"))  # nothing to refuse
    check_v21_switches(  # v2.1 off: v2's settings are v2's business
        SimpleNamespace(UNIFY_MEMORY_V21="off", UNIFY_MEMORY_V2_DOCSTRINGS="on"),
    )
    with pytest.raises(MemoryV2Unavailable, match="UNIFY_MEMORY_V2_DOCSTRINGS=on"):
        check_v21_switches(
            SimpleNamespace(UNIFY_MEMORY_V21="on", UNIFY_MEMORY_V2_DOCSTRINGS="on"),
        )

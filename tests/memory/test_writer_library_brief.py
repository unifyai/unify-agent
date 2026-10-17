"""In library mode the writer's system message is its role's brief alone; commits carry
Why/Episodes/Items; a library-mode run with contradicting v2 switches is refused by name.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from unify.memory import library_prompts as pv
from unify.memory import qa as _qa
from unify.memory.integration import prompt
from unify.memory.integration.request import (
    MemoryUnavailable,
    check_library_switches,
)
from unify.memory.writer_pass import _library_trailers, writer_system
from unify.memory.trigger import PassRequest
from tests.memory.test_writer_pass import Turns, _call, _writer
from tests.memory.test_writer_library_tools import EPS


def _finishing_pass(tmp_path, **cfg):
    _, _, writer = _writer(
        tmp_path,
        Turns([_call("f", "finish", {"summary": "x"})]),
        **cfg,
    )
    writer.load = EPS.__getitem__
    asyncio.run(
        writer.run(PassRequest("incremental", "svc", ["e1", "e2"], False), "b1"),
    )
    return writer


def _briefs_integrated():
    """write_brief_now and curate_brief_now read the repair and index modules' constants: run once integrated."""
    for name in (
        "unify.memory.repair",
        "unify.memory.library_index",
        "unify.memory.curate",
    ):
        pytest.importorskip(name)


def test_the_prompt_section_uses_the_reviewed_guide():
    assert prompt.GUIDE_LIBRARY is pv.GUIDE_LIBRARY


def test_v2_system_message_is_unchanged(tmp_path):
    writer = _finishing_pass(tmp_path, library_mode=False)
    assert writer.messages[0]["content"] == _qa.system(
        writer_system(**writer._switches()),
        writer._qa,
    )


def test_library_system_message_is_the_write_brief(tmp_path):
    _briefs_integrated()
    writer = _finishing_pass(tmp_path, library_mode=True)
    # the default configuration caps the pass's spend, so finish's refusal cap is stated
    assert writer.messages[0] == {
        "role": "system",
        "content": pv.write_brief_now(finish_capped=True),
    }
    assert "Manifest rules for consolidators" not in writer.messages[0]["content"]


def test_a_curate_pass_gets_the_curate_brief(tmp_path):
    _briefs_integrated()
    _, _, writer = _writer(tmp_path, Turns([]), library_mode=True)
    writer.cfg.role = "curate"  # a PassConfig field
    assert writer._brief_library_mode() == pv.curate_brief_now(finish_capped=True)


def test_library_trailers_from_the_manifest():
    m = {
        "items": [
            {"item": "memory.text.dates:parse_date"},
            {"item": "notes/text/dates.md"},
            {"x": 1},
        ],
        "why": "dates repeat\nin four episodes",
    }
    assert _library_trailers(m, ["e1", "e2"]) == {
        "Why": ["dates repeat in four episodes"],
        "Episodes": ["e1", "e2"],
        "Items": ["memory.text.dates:parse_date", "notes/text/dates.md"],
    }
    assert _library_trailers(None, []) == {
        "Why": ["(not given)"],
        "Episodes": [],
        "Items": ["(none)"],
    }
    assert len(_library_trailers({"why": "w" * 900}, [])["Why"][0]) == 200


def test_a_request_with_contradicting_switches_is_refused_by_name():
    check_library_switches(
        SimpleNamespace(UNIFY_MEMORY_LIBRARY="on"),
    )  # nothing to refuse
    check_library_switches(  # library mode off: v2's settings are v2's business
        SimpleNamespace(UNIFY_MEMORY_LIBRARY="off", UNIFY_MEMORY_DOCSTRINGS="on"),
    )
    with pytest.raises(MemoryUnavailable, match="UNIFY_MEMORY_DOCSTRINGS=on"):
        check_library_switches(
            SimpleNamespace(UNIFY_MEMORY_LIBRARY="on", UNIFY_MEMORY_DOCSTRINGS="on"),
        )

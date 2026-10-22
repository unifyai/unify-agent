"""In library mode the writer's system message is its role's brief alone; commits carry
Why/Episodes/Items; a library-mode run with contradicting v2 switches is refused by name.
"""

from __future__ import annotations

import asyncio


from unify.memory import library_prompts as pv
from unify.memory.writer_pass import _library_trailers
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


def test_library_system_message_is_the_write_brief(tmp_path):
    writer = _finishing_pass(tmp_path)
    # the default configuration caps the pass's spend, so finish's refusal cap is stated
    assert writer.messages[0] == {
        "role": "system",
        "content": pv.write_brief_now(finish_capped=True),
    }
    assert "Manifest rules for consolidators" not in writer.messages[0]["content"]


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



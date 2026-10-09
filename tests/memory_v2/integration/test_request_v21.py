"""P1 Task 7, part 3: UNIFY_MEMORY_V21 reaches the online recorders (dialogue, worktree) and the episode writer.

Off, every call is made exactly as at 4675a3c45 (no new keyword is passed); on, the dialogue keeps its full
observation, the work-tree capture stores files whole, and the writer writes record format 2.
"""

from types import SimpleNamespace

from unify.memory_v2 import episodes as episodes_mod
from unify.memory_v2.integration import request as request_mod
from unify.memory_v2.integration import worktree_capture as capture_mod
from unify.memory_v2.integration.adapters.worktree import BLOB_CAP
from unify.memory_v2.integration.paths import Paths
from unify.memory_v2.redact import Redactor
from unify.settings import SETTINGS
from tests.memory_v2 import p1_t7_inputs as inp
from tests.memory_v2.integration.test_request import (
    _begin,
    _finish,
    _transcript,
    mv2,
)  # noqa: F401


def test_memory_v21_on_reads_the_setting(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY_V21", "off")
    assert request_mod.memory_v21_on() is False
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY_V21", "on")
    assert request_mod.memory_v21_on() is True


def test_recorded_dialogue_keeps_the_full_observation_only_under_v21():
    lines = inp.dialogue_lines()
    (off,) = request_mod.recorded_dialogue(lines, "env", Redactor())
    (on,) = request_mod.recorded_dialogue(lines, "env", Redactor(), v21=True)
    assert len(off.response) <= 65536 and "middle elided" in off.response
    assert on.response == inp.big_text(70000, "OBSERVATION_END")


def test_the_worktree_capture_stores_files_whole_under_v21(tmp_path, monkeypatch):
    monkeypatch.setattr(capture_mod, "_ACTIVE", None)
    paths = Paths.under(tmp_path / "home")
    ws = tmp_path / "ws"
    ws.mkdir()
    monkeypatch.setattr(request_mod, "_CURRENT", SimpleNamespace(index="", paths=paths))
    caps = []
    for kw in ({}, {"v21": True}):
        cap = capture_mod.WorktreeCapture(
            paths,
            ws,
            lambda: Redactor({}),
            hidden=lambda rel: False,
            **kw,
        )
        cap.begin()
        caps.append(cap._recorder.blob_cap)
        cap.abort()
        monkeypatch.setattr(capture_mod, "_ACTIVE", None)
    assert caps == [BLOB_CAP, None]


def test_finish_writes_record_format_2_only_under_v21(mv2, monkeypatch):  # noqa: F811
    made = []
    real = episodes_mod.EpisodeWriter

    def spy(*args, **kw):
        made.append(kw)
        return real(*args, **kw)

    monkeypatch.setattr(episodes_mod, "EpisodeWriter", spy)
    for value in ("off", "on"):
        monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY_V21", value)
        run = _begin(mv2, f"Say hi ({value}).")
        _transcript(run)
        _finish(mv2, run)
    assert made == [{}, {"v21": True}]  # off: the call as at 4675a3c45


def test_a_long_multi_line_trailing_object_is_the_payload_under_v21():
    import json

    obj = {
        "action": "submit",
        "grid": [[i % 10 for i in range(40)] for _ in range(400)],
    }
    reply = "Here is my answer:\n" + json.dumps(
        obj,
        indent=1,
    )  # multi-line, about 100 K
    assert len(reply) > 65536 and reply.count("\n") > 100
    lines = inp.dialogue_lines()
    lines[1]["message"]["content"] = reply
    (off,) = request_mod.recorded_dialogue(lines, "env", Redactor())
    (on,) = request_mod.recorded_dialogue(lines, "env", Redactor(), v21=True)
    assert off.method == "act" and off.args == [
        "}",
    ]  # as at 4675a3c45: the reply's last line
    assert on.method == "submit" and on.args == [obj["grid"]]

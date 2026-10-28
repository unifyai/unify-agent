"""The online recorders and the episode writer: the dialogue keeps its full observation, the work-tree capture stores
files whole, and the writer writes the current record format."""

import json
from types import SimpleNamespace

from unify.memory import episodes as episodes_mod
from unify.memory.evidence import EvidenceStore
from unify.memory.gitio import Repo
from unify.memory.integration import request as request_mod
from unify.memory.integration import worktree_capture as capture_mod
from unify.memory.integration.paths import Paths
from unify.memory.redact import Redactor
from tests.memory import recorder_inputs as inp
from tests.memory.integration.test_request import (  # noqa: F401
    _begin,
    _finish,
    _transcript,
    mv2,
)


def test_recorded_dialogue_keeps_the_full_observation():
    lines = inp.dialogue_lines()
    (got,) = request_mod.recorded_dialogue(lines, Redactor())
    assert got.response == inp.big_text(70000, "OBSERVATION_END")


def test_the_worktree_capture_stores_files_whole(tmp_path, monkeypatch):
    monkeypatch.setattr(capture_mod, "_ACTIVE", None)
    paths = Paths.under(tmp_path / "home")
    ws = tmp_path / "ws"
    ws.mkdir()
    monkeypatch.setattr(request_mod, "_CURRENT", SimpleNamespace(index="", paths=paths))
    cap = capture_mod.WorktreeCapture(paths, ws, lambda: Redactor({}), hidden=lambda rel: False)
    cap.begin()
    try:
        assert cap._recorder.blob_cap is None  # never cut at the older recorders' BLOB_CAP
    finally:
        cap.abort()


def test_finish_writes_the_current_record_format(mv2):  # noqa: F811
    run = _begin(mv2, "Say hi.")
    _transcript(run)
    _finish(mv2, run)
    sha = EvidenceStore(mv2.paths.evidence).episode_ref(run.episode_id)[0]
    repo = Repo(mv2.paths.episodes)
    (path,) = [p for p in repo.run("ls-tree", "-r", "--name-only", sha).split() if p.endswith("meta.json")]
    meta = json.loads(repo.run("show", f"{sha}:{path}"))
    assert meta["record_format"] == episodes_mod.RECORD_FORMAT


def test_a_long_multi_line_trailing_object_is_the_payload():
    obj = {"action": "submit", "grid": [[i % 10 for i in range(40)] for _ in range(400)]}
    reply = "Here is my answer:\n" + json.dumps(obj, indent=1)  # multi-line, about 100 K
    assert len(reply) > 65536 and reply.count("\n") > 100
    lines = inp.dialogue_lines()
    lines[1]["message"]["content"] = reply
    (got,) = request_mod.recorded_dialogue(lines, Redactor())
    assert got.method == "submit" and got.args == [obj["grid"]]

"""``memlab.inputs`` (memory v2.1 stage 5): recorded inputs for library tests, read on the host here.

The module's box paths (``/inputs``, ``/qa/samples.json``, ``/qa-out/reads.jsonl``) are redirected to temporary
directories; nothing here runs model code.
"""

import hashlib
import json

import pytest

from unify.memory_v2 import inputs as memlab_inputs
from unify.memory_v2.inputs import (
    RecordedInput,
    blob,
    from_action,
    from_blob,
    inputs,
    samples,
)
from unify.memory_v2.replay import RecordedError, ReplayMiss

CLOCK = {"date": "Tuesday, October 07, 2026", "time": "09:15 AM"}
CALL = {
    "index": 3,  # an exported row carries its index; it is ignored
    "cell": 0,
    "channel": "phone",
    "method": "get_current_date_and_time",
    "args": [],
    "kwargs": {},
    "response": CLOCK,
    "status": "ok",
    "kind": "tool",
}
CSV = b"vendor_id,invoice_no,amount\nV-17,INV-0042,1250.50\n"


@pytest.fixture
def box(tmp_path, monkeypatch):
    inputs_dir = tmp_path / "inputs"
    (inputs_dir / "blobs").mkdir(parents=True)
    qa = tmp_path / "qa"
    qa.mkdir()
    out = tmp_path / "qa-out"
    out.mkdir()
    monkeypatch.setattr(memlab_inputs, "INPUTS", inputs_dir)
    monkeypatch.setattr(memlab_inputs, "SAMPLES", qa / "samples.json")
    monkeypatch.setattr(memlab_inputs, "READS", out / "reads.jsonl")
    monkeypatch.setattr(memlab_inputs, "_cache", {})
    monkeypatch.setattr(memlab_inputs, "_reads", set())
    monkeypatch.setattr(memlab_inputs.tempfile, "tempdir", str(tmp_path / "tmp"))
    (tmp_path / "tmp").mkdir()
    return inputs_dir, qa, out


def _put(inputs_dir, data):
    sha = hashlib.sha256(data).hexdigest()
    (inputs_dir / "blobs" / sha).write_bytes(data)
    return sha


def test_env_form_is_the_exact_call_replay(box):
    x = from_action(CALL, form="env")
    env = x.value()
    assert env.phone.get_current_date_and_time() == CLOCK
    with pytest.raises(ReplayMiss):
        env.phone.get_current_date_and_time(tz="UTC")  # another call is not answered
    failed = {**CALL, "status": "error", "error": "busy", "response": None}
    ctx = from_action(CALL, form="env", context=[failed]).value()
    with pytest.raises(RecordedError):
        ctx.phone.get_current_date_and_time()  # the context, not the action, answers
    assert (
        x.response == CLOCK and x.kwargs == {} and x.status == "ok" and x.kind == "tool"
    )
    assert (
        repr(x) == "tool-fixture" and repr(from_action(CALL, label="clock")) == "clock"
    )


def test_observation_text_path_and_bytes_forms(box):
    inputs_dir, _, _ = box
    obs = from_action(
        {**CALL, "kind": "dialogue", "response": {"type": "SubmitFeedback"}},
    )
    assert obs.form == "observation" and obs.value() == {"type": "SubmitFeedback"}
    assert (
        from_action({**CALL, "kind": "dialogue", "response": "You see: tree"}).value(
            "text",
        )
        == "You see: tree"
    )
    with pytest.raises(ValueError):
        obs.value("text")
    sha = _put(inputs_dir, CSV)
    f = from_blob(sha, "invoices.csv")
    path = f.value()
    assert (
        path.endswith("/invoices.csv") and open(path, "rb").read() == CSV
    )  # the reader sees the name
    assert f.value("bytes") == CSV and f.value("text").startswith("vendor_id")
    wt = from_action(
        {
            "kind": "worktree",
            "channel": "worktree:workspace",
            "method": "read",
            "args": ["ap/inv.csv"],
            "kwargs": {},
            "response": {"blob_before": sha, "blob_after": sha},
            "status": "ok",
        },
    )
    assert wt.form == "path" and wt.value().endswith("/inv.csv")
    assert blob(sha) == CSV
    with pytest.raises(ValueError):
        blob("../../etc/passwd")
    with pytest.raises(ValueError):
        from_blob("not-a-blob-id")
    with pytest.raises(ValueError):
        obs.value("yaml")


def test_inputs_append_the_gates_samples_only_where_it_draws_them(box):
    _, qa, _ = box
    fixtures = [from_action(CALL, form="env")]
    assert (
        inputs("env/phone:current_datetime", fixtures) == fixtures
    )  # the sandbox, or an ordinary gate run
    rows = [
        {"id": 0, "role": "cover", "form": "env", "action": CALL},
        {
            "id": 1,
            "role": "sample",
            "form": "env",
            "action": {**CALL, "response": {**CLOCK, "time": "10:00 AM"}},
        },
        {"id": 2, "role": "sample", "form": "env", "action": CALL},
    ]
    (qa / "samples.json").write_text(
        json.dumps({"items": {"env/phone:current_datetime": {"rows": rows}}}),
    )
    memlab_inputs._cache.clear()
    got = inputs("env/phone:current_datetime", fixtures)
    assert got[0] is fixtures[0] and [repr(x) for x in got[1:]] == [
        "sample-1",
        "sample-2",
    ]  # covers stay out
    assert all(x.sampled for x in got[1:]) and not got[0].sampled
    assert samples("env/other:fn") == []


def test_a_sample_read_inside_a_test_is_recorded_once(box, monkeypatch):
    _, qa, out = box
    rows = [{"id": 7, "role": "sample", "form": "env", "action": CALL}]
    (qa / "samples.json").write_text(
        json.dumps({"items": {"env/phone:current_datetime": {"rows": rows}}}),
    )
    (x,) = samples("env/phone:current_datetime")
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    x.value()  # outside a test (collection time): nothing recorded
    assert not (out / "reads.jsonl").exists()
    monkeypatch.setenv(
        "PYTEST_CURRENT_TEST",
        "env/phone/tests/test_clock.py::test_clock[sample-7] (call)",
    )
    x.value()
    _ = x.response
    lines = [json.loads(ln) for ln in (out / "reads.jsonl").read_text().splitlines()]
    assert lines == [
        {
            "item": "env/phone:current_datetime",
            "sample": 7,
            "test": "env/phone/tests/test_clock.py::test_clock[sample-7]",
        },
    ]
    from_action(CALL).value()  # a fixture's read is never recorded
    assert len((out / "reads.jsonl").read_text().splitlines()) == 1


def test_a_row_without_an_action_is_refused():
    with pytest.raises(ValueError):
        RecordedInput({"id": 1})

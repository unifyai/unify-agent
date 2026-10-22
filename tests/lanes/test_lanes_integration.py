"""The memory lanes end to end through the local relay, with a stand-in model behind it: the capture fork (the real
fork worker, run by path with the box's environment, its request built by the real fork code) and a writer and a
helper turn (unillm's real call path, the writer's route installed as the writer pass installs it, with only the last
hop, litellm's transport, replaced by a plain HTTP post to the route). Then the readers: the writer pass's
reconciliation reads the writer lane's journal, the run report the fork lane's.

Nothing leaves the machine, the provider key is a fake, and no process but the relay ever holds it.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import subprocess
import sys
import urllib.request
from decimal import Decimal
from types import SimpleNamespace

import pytest
from pydantic import SecretStr

from tests.lanes.upstream import FAKE_KEY, MODEL, Upstream, reply
from unify.lanes.relay import LaneRelay
from unify.memory import metering, reconcile
from unify.memory.integration import fork
from unify.memory.writer_pass import WriterRoute, unillm_turn

EPISODE = "20261021T120000-ab12cd34"
SUMMARY = "```path: summary.md\n## Asked\nCount the rows.\n## Did\nRead data.csv and counted 3 rows.\n```\n"


def _model(body: dict):
    """The stand-in model: the fork gets a reply that stages a summary; the writer and the helper a plain answer."""
    text = json.dumps(body.get("messages") or [])
    content = SUMMARY if "Count the rows" in text else "noted"
    return 200, reply(
        content, cost=0.0021 if content == SUMMARY else 0.0007, gid=f"gen-{len(text)}"
    )


@pytest.fixture
def lanes(tmp_path):
    up = Upstream(_model)
    relay = LaneRelay.start(
        key=FAKE_KEY,
        lanes={"writer": [MODEL], "helper": [MODEL], "fork": [MODEL]},
        cap_usd="2.00",
        worst_case_usd={MODEL: "0.65625"},
        journal_dir=tmp_path / "lanes",
        upstream=up.base,
        log_path=tmp_path / "lanes" / "relay.log",
    )
    yield SimpleNamespace(relay=relay, upstream=up, folder=tmp_path / "lanes")
    relay.stop()
    up.close()


@pytest.fixture
def unillm_over_http(monkeypatch):
    """litellm's transport replaced by a plain HTTP post of the request unillm built, to the base URL and with the
    credential unillm chose for it; everything above it is unillm's own path. The response cache and OTel are off.
    """
    import unillm.clients.uni_llm as uni_llm
    from openai.types.chat import ChatCompletion
    from unillm.settings import SETTINGS as unillm_settings

    monkeypatch.setattr(
        uni_llm, "_llm_gateway", uni_llm._llm_gateway
    )  # undone after the test
    monkeypatch.setenv("UNILLM_CACHE", "false")
    monkeypatch.setattr(unillm_settings, "UNILLM_CACHE", False)
    monkeypatch.delenv("UNILLM_OTEL", raising=False)
    unillm_logger = importlib.import_module("unillm.logger")
    if hasattr(unillm_logger, "_OTEL_ENABLED"):
        monkeypatch.setattr(unillm_logger, "_OTEL_ENABLED", False)
    sent: list[dict] = []

    def post(kw: dict) -> dict:
        model = str(kw["model"])
        body = {
            "model": (
                model.split("/", 1)[1] if model.startswith("openrouter/") else model
            ),
            "messages": kw["messages"],
            **({"tools": kw["tools"]} if kw.get("tools") else {}),
            **(kw.get("extra_body") or {}),
        }
        assert not kw.get("stream")
        headers = {
            "Authorization": f"Bearer {kw['api_key']}",
            "Content-Type": "application/json",
        }
        headers.update(kw.get("extra_headers") or {})
        sent.append(
            {
                "api_base": kw.get("api_base"),
                "call_kind": headers.get("X-Unify-Call-Kind"),
            }
        )
        req = urllib.request.Request(
            str(kw["api_base"]).rstrip("/") + "/chat/completions",
            data=json.dumps(body).encode(),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read())

    async def acompletion(*, shared_session=None, client=None, **kw):
        return ChatCompletion.model_validate(await asyncio.to_thread(post, kw))

    monkeypatch.setattr(uni_llm.litellm, "acompletion", acompletion)
    return sent


def test_the_capture_fork_runs_through_its_lane_and_stages_its_capture(tmp_path, lanes):
    source = {
        "client": SimpleNamespace(
            endpoint=f"{MODEL}@openrouter", reasoning_effort="low"
        ),
        "sent_messages": [
            {"role": "system", "content": "You are an agent."},
            {"role": "user", "content": "Count the rows of data.csv"},
            {"role": "assistant", "content": "There are 3 rows."},
        ],
        "tools": [],
    }
    doc, why = fork.request_from_source(
        EPISODE, source, lanes.relay.base_url("fork"), lanes.relay.token("fork")
    )
    assert why is None
    staging = tmp_path / "staging"
    staging.mkdir()
    box_env = {
        "PATH": "/usr/bin:/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
    }  # the fork box's whole environment
    done = subprocess.run(
        [sys.executable, "-I", str(fork.WORKER_FILE), "--staging", str(staging)],
        input=json.dumps(doc).encode(),
        env=box_env,
        capture_output=True,
        timeout=300,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    record = json.loads((staging / "fork.json").read_text())
    assert record["status"] == "ok" and record["turns"] == 1
    assert record["usd"] == "0.0021" and record["model"] == MODEL
    assert "Count the rows." in (staging / "summary.md").read_text()
    # the provider saw the key and the fork's model only; the box never had the key
    (seen,) = lanes.upstream.requests
    assert (
        seen["headers"]["authorization"] == f"Bearer {FAKE_KEY}"
        and seen["body"]["model"] == MODEL
    )
    assert (
        FAKE_KEY not in json.dumps(doc)
        and FAKE_KEY.encode() not in done.stdout + done.stderr
    )
    # the fork lane's journal, read as the run report reads it
    by_episode = metering.fork_by_episode(
        lanes.folder / "fork.jsonl", Decimal("0.65625")
    )
    assert by_episode == {
        EPISODE: {
            "requests": 1,
            "known_usd": "0.0021",
            "unknown": 0,
            "booked_usd": "0.0021",
        }
    }
    assert metering.lane_total(
        "fork", lanes.folder / "fork.jsonl", Decimal("0.65625")
    ).known_usd == Decimal("0.0021")


def test_a_writer_and_a_helper_turn_run_through_their_lanes_and_the_pass_reconciles(
    lanes, unillm_over_http
):
    journal = lanes.folder / "writer.jsonl"
    start = reconcile.journal_offset(str(journal))
    writer = unillm_turn(
        MODEL,
        "low",
        origin="memory_writer",
        route=WriterRoute(
            lanes.relay.base_url("writer"), SecretStr(lanes.relay.token("writer"))
        ),
    )
    msg, usd = asyncio.run(
        writer(
            [{"role": "system", "content": "s"}, {"role": "user", "content": "write"}],
            [],
        )
    )
    assert msg["content"] == "noted" and usd == "0.0007"
    helper = unillm_turn(
        MODEL,
        "low",
        origin="memory_helper",
        route=WriterRoute(
            lanes.relay.base_url("helper"), SecretStr(lanes.relay.token("helper"))
        ),
    )
    msg, usd = asyncio.run(helper([{"role": "user", "content": "help"}], []))
    assert msg["content"] == "noted" and usd == "0.0007"
    end = reconcile.journal_offset(str(journal))
    assert [s["call_kind"] for s in unillm_over_http] == [
        "memory_writer",
        "memory_helper",
    ]
    # the writer pass's settle: its window of the writer lane's journal, priced by charge; the helper's call is in
    # its own journal, so the writer's never books it
    summary = asyncio.run(
        reconcile.reconcile_window(
            str(journal),
            start,
            end,
            models=[MODEL],
            worst_case_usd=Decimal("0.65625"),
            budget_s=5,
        )
    )
    assert (
        summary["calls"],
        summary["priced"],
        summary["unknown"],
        summary["usd"],
    ) == (1, 1, 0, "0.0007")
    helper_rows = [
        json.loads(x) for x in (lanes.folder / "helper.jsonl").read_text().splitlines()
    ]
    assert [r["unify_meta"]["call_kind"] for r in helper_rows] == ["memory_helper"] * 2
    for seen in lanes.upstream.requests:
        assert seen["headers"]["authorization"] == f"Bearer {FAKE_KEY}"
        assert seen["body"]["model"] == MODEL
        assert not [h for h in seen["headers"] if h.startswith("x-unify")]
    written = b"".join(p.read_bytes() for p in lanes.folder.iterdir() if p.is_file())
    assert FAKE_KEY.encode() not in written


def test_the_writers_route_refuses_a_model_off_its_lane(lanes, unillm_over_http):
    """A writer turn for a model the lane does not serve is refused by the relay before anything goes upstream, and
    the writer sees the refusal as its category only."""
    from unify.memory.writer_pass import WriterCallError

    writer = unillm_turn(
        "openai/gpt-6-sol",
        "low",
        origin="memory_writer",
        route=WriterRoute(
            lanes.relay.base_url("writer"), SecretStr(lanes.relay.token("writer"))
        ),
    )
    token = lanes.relay.token("writer")
    with pytest.raises(WriterCallError) as err:
        asyncio.run(writer([{"role": "user", "content": "write"}], []))
    assert lanes.upstream.requests == []
    assert FAKE_KEY not in str(err.value) and token not in str(err.value)
    refused = [
        json.loads(x)
        for x in (lanes.folder / "refusals.jsonl").read_text().splitlines()
    ]
    assert refused and {(r["lane"], r["type"], r["reason"]) for r in refused} == {
        ("writer", "off_model_refused", "not_allowed")
    }

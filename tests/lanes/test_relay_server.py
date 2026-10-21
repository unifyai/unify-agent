"""The lane relay's rules, served in this process against a stand-in provider (tests/lanes/upstream.py): per-lane
tokens, the model allow-list, refusal shapes the writer reads, the journals in the shape the memory code reads, the
one spend cap with worst-case booking, redaction of every secret from replies, streams, and concurrency.

The provider key is a fake and nothing leaves the machine. The relay process itself (its start, its lifeline, its
stop, what its process shows) is tests/lanes/test_relay_process.py.
"""

from __future__ import annotations

import asyncio
import contextlib
import http.client
import json
import secrets
import socket
import threading
from decimal import Decimal
from pathlib import Path

import pytest

from tests.lanes.upstream import FAKE_KEY, MODEL, Upstream
from unify.lanes import relay_server as rs
from unify.memory import metering, reconcile
from unify.memory.writer_pass import _refusal_kind

OTHER_MODEL = "openai/gpt-6-sol"
WORST = {MODEL: "0.65625", OTHER_MODEL: "13.125"}
LANES = ("writer", "helper", "fork")


def _doc(
    folder: Path, upstream: str, *, cap="2.00", prices=None, lanes=LANES, tokens=None
) -> tuple[dict, dict]:
    tokens = tokens or {lane: secrets.token_urlsafe(32) for lane in lanes}
    folder.mkdir(mode=0o700, parents=True, exist_ok=True)
    doc = {
        "format": rs.FORMAT,
        "key": FAKE_KEY,
        "upstream": upstream,
        "cafile": None,
        "journal_dir": str(folder),
        "cap_usd": cap,
        "worst_case_usd": dict(WORST),
        "prices": prices or {},
        "lanes": {lane: {"token": tokens[lane], "models": [MODEL]} for lane in lanes},
    }
    return doc, tokens


@contextlib.contextmanager
def serving(doc: dict):
    server = rs.bind(rs.Config(doc))
    threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True
    ).start()
    try:
        yield server.relay
    finally:
        server.relay.drain(server)


def call(
    port,
    lane,
    token,
    body=None,
    *,
    headers=None,
    method="POST",
    path=None,
    host=None,
    raw=None,
):
    """One request to the relay: ``(status, parsed body or raw bytes)``. *headers* is a list of pairs, so a header
    may repeat."""
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
    data = (
        raw
        if raw is not None
        else (json.dumps(body).encode() if body is not None else b"")
    )
    conn.putrequest(
        method,
        path or f"/{lane}/v1/chat/completions",
        skip_host=True,
        skip_accept_encoding=True,
    )
    conn.putheader("Host", host or f"127.0.0.1:{port}")
    pairs = [("Authorization", f"Bearer {token}")] if token is not None else []
    for name, value in pairs + list(headers or []):
        conn.putheader(name, value)
    conn.putheader("Content-Type", "application/json")
    conn.putheader("Content-Length", str(len(data)))
    conn.endheaders(data)
    resp = conn.getresponse()
    out = resp.read()
    conn.close()
    try:
        return resp.status, json.loads(out)
    except ValueError:
        return resp.status, out


def _rows(path: Path) -> list[dict]:
    return (
        [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if path.exists()
        else []
    )


BODY = {"model": MODEL, "messages": [{"role": "user", "content": "hi"}]}


@pytest.fixture
def upstream():
    up = Upstream()
    yield up
    up.close()


def test_a_call_on_its_lane_goes_upstream_with_the_key_and_returns_with_every_secret_redacted(
    tmp_path, upstream
):
    doc, tokens = _doc(tmp_path / "lanes", upstream.base)
    upstream.echo = (FAKE_KEY + " " + tokens["writer"] + " " + tokens["fork"]).encode()
    with serving(doc) as relay:
        status, reply = call(
            relay.port,
            "writer",
            tokens["writer"],
            BODY,
            headers=[
                ("X-Unify-Call-Kind", "memory_writer"),
                ("X-Unify-Pass-Id", "ab" * 8),
            ],
        )
    assert status == 200
    content = reply["choices"][0]["message"]["content"]
    assert FAKE_KEY not in content and not any(t in content for t in tokens.values())
    assert content.count("[redacted]") == 3
    (seen,) = upstream.requests
    assert seen["path"] == "/api/v1/chat/completions"
    assert (
        seen["headers"]["authorization"] == f"Bearer {FAKE_KEY}"
    )  # the relay alone adds the key
    assert not [
        h for h in seen["headers"] if h.startswith("x-unify")
    ]  # never sent upstream
    assert seen["body"]["model"] == MODEL and seen["body"]["usage"] == {"include": True}
    assert not any(t.encode() in seen["raw"] for t in tokens.values())


@pytest.mark.parametrize(
    "case, reason",
    [
        ("wrong", "no_lane_token"),
        ("other_lane", "other_lane_token"),
        ("missing", "no_lane_token"),
        ("twice", "no_lane_token"),
        ("lower_case_scheme", "no_lane_token"),
        ("key_itself", "no_lane_token"),
    ],
)
def test_each_token_is_valid_on_its_own_lane_only(tmp_path, upstream, case, reason):
    doc, tokens = _doc(tmp_path / "lanes", upstream.base)
    token, headers = {
        "wrong": (secrets.token_urlsafe(32), []),
        "other_lane": (tokens["fork"], []),
        "missing": (None, []),
        "twice": (tokens["writer"], [("Authorization", f"Bearer {tokens['writer']}")]),
        "lower_case_scheme": (None, [("Authorization", f"bearer {tokens['writer']}")]),
        "key_itself": (FAKE_KEY, []),
    }[case]
    with serving(doc) as relay:
        status, reply = call(relay.port, "writer", token, BODY, headers=headers)
    assert status == 403
    assert reply == {
        "error": {
            "code": 403,
            "type": "writer_route_refused",
            "message": "this lane needs its own token",
            "reason": reason,
        }
    }
    assert upstream.requests == []
    text = (tmp_path / "lanes" / "refusals.jsonl").read_text()
    assert not any(t in text for t in tokens.values()) and FAKE_KEY not in text


@pytest.mark.parametrize(
    "method, path, host, status, reason",
    [
        ("GET", "/writer/v1/chat/completions", None, 405, "method"),
        ("POST", "/writer/v1/models", None, 404, "path"),
        ("GET", "/api/v1/generation?id=gen-1", None, 404, "path"),
        ("POST", "/writer/v1/chat/completions?x=1", None, 404, "path"),
        ("POST", "/writer/v1/chat/completions/", None, 404, "path"),
        ("POST", "/other/v1/chat/completions", None, 404, "path"),
        ("POST", "/writer/v1/chat/completions", "localhost:1", 403, "host"),
    ],
)
def test_only_each_lanes_one_route_is_served(
    tmp_path, upstream, method, path, host, status, reason
):
    doc, tokens = _doc(tmp_path / "lanes", upstream.base)
    with serving(doc) as relay:
        got, reply = call(
            relay.port,
            "writer",
            tokens["writer"],
            BODY,
            method=method,
            path=path,
            host=host,
        )
    assert got == status and reply["error"]["reason"] == reason
    assert upstream.requests == []


def test_a_lane_the_run_does_not_serve_is_refused(tmp_path, upstream):
    doc, tokens = _doc(tmp_path / "lanes", upstream.base, lanes=("fork",))
    with serving(doc) as relay:
        status, reply = call(relay.port, "writer", tokens["fork"], BODY)
    assert status == 403 and reply["error"]["reason"] == "lane_off"
    assert upstream.requests == []


def test_a_malformed_request_is_answered_without_echoing_it(tmp_path, upstream):
    doc, tokens = _doc(tmp_path / "lanes", upstream.base)
    with serving(doc) as relay:
        with socket.create_connection(("127.0.0.1", relay.port), timeout=10) as s:
            s.sendall(f"GET /{tokens['writer']} extra HTTP/1.1\r\n\r\n".encode())
            data = b""
            while chunk := s.recv(
                65536
            ):  # to the end: the relay closes the connection after its answer
                data += chunk
    assert (
        b" 400 " in data.split(b"\r\n", 1)[0] and tokens["writer"].encode() not in data
    )
    assert b"malformed request" in data


@pytest.mark.parametrize(
    "body, reason",
    [
        ({"model": "openai/gpt-6-sol", "messages": []}, "not_allowed"),
        ({"model": "OpenAI/GPT-6-Luna", "messages": []}, "not_allowed"),
        ({"model": MODEL + ":batch", "messages": []}, "not_allowed"),
        (
            {"model": MODEL, "models": ["openai/gpt-6-sol"], "messages": []},
            "fallback_models_field",
        ),
        ({"model": MODEL, "plugins": [{"id": "web"}], "messages": []}, "plugins_field"),
        ({"model": MODEL, "web_search_options": {}, "messages": []}, "plugins_field"),
        ({"model": MODEL, "n": 128, "messages": []}, "several_completions"),
        ({"model": MODEL, "n": 0, "messages": []}, "several_completions"),
        ({"model": MODEL, "n": "1", "messages": []}, "several_completions"),
        ({"model": MODEL, "n": True, "messages": []}, "several_completions"),
        ({"model": MODEL, "best_of": 4, "messages": []}, "several_completions"),
        ({"messages": []}, "missing_model"),
        ({"model": ["x"], "messages": []}, "model_not_a_string"),
        ({"model": MODEL, "stream": "yes", "messages": []}, "stream_not_a_boolean"),
        ([MODEL], "not_an_object"),
        (
            ('{"model": "%s", "model": "%s"}' % (MODEL, OTHER_MODEL)).encode(),
            "duplicate_keys",
        ),
        (b"not json", "unparseable_body"),
    ],
)
def test_a_model_off_the_lanes_list_is_refused_before_anything_goes_upstream(
    tmp_path, upstream, body, reason
):
    doc, tokens = _doc(tmp_path / "lanes", upstream.base)
    raw = body if isinstance(body, bytes) else None
    with serving(doc) as relay:
        status, reply = call(
            relay.port, "fork", tokens["fork"], None if raw else body, raw=raw
        )
    assert status == 403
    assert (
        reply["error"]["type"] == "off_model_refused"
        and reply["error"]["reason"] == reason
    )
    assert reply["error"]["message"] == f"this lane may call only {MODEL}"
    assert upstream.requests == []
    assert not (
        tmp_path / "lanes" / "fork.jsonl"
    ).read_text()  # nothing journaled as a call
    (row,) = _rows(tmp_path / "lanes" / "refusals.jsonl")
    assert (row["lane"], row["type"], row["reason"]) == (
        "fork",
        "off_model_refused",
        reason,
    )


def test_each_call_is_journaled_in_its_lanes_journal_in_the_shape_the_memory_code_reads(
    tmp_path, upstream
):
    doc, tokens = _doc(tmp_path / "lanes", upstream.base)
    meta = [
        ("X-Unify-Call-Kind", "memory_writer"),
        ("X-Unify-Session", "s-1"),
        ("X-Unify-Pass-Id", "0123456789abcdef"),
        ("X-Unify-Entries", "0123456789abcdef.fedcba9876543210"),
        ("X-Unify-Msg-Count", "not-a-number"),
    ]
    with serving(doc) as relay:
        assert (
            call(relay.port, "writer", tokens["writer"], BODY, headers=meta)[0] == 200
        )
        fork_meta = [
            ("X-Unify-Call-Kind", "actor_fork"),
            ("X-Unify-Session", "fork.20261021T1-ab"),
            ("X-Unify-Parent", "20261021T1-ab"),
        ]
        assert (
            call(relay.port, "fork", tokens["fork"], BODY, headers=fork_meta)[0] == 200
        )
    journal = tmp_path / "lanes" / "writer.jsonl"
    started, done = _rows(journal)
    assert started["origin"] == "request_started" and started["status"] == "started"
    assert (
        started["request_attempt_id"]
        == done["request_attempt_id"]
        == started["call_id"]
    )
    assert started["requested_model"] == MODEL and started["account_charge"] is None
    assert done["status"] == "completed" and done["status_code"] == 200
    assert done["account_charge"] == "0.00125" and done["generation_id"] == "gen-stub-1"
    assert (done["booked_usd"], done["charge_class"]) == ("0.00125", "charge")
    assert done["tokens"] == {
        "input": 120,
        "output": 7,
        "cache_read": 100,
        "cache_write": 20,
        "reasoning": 3,
    }
    assert done["unify_meta"]["pass_id"] == "0123456789abcdef"
    assert done["unify_meta"]["entries"] == ["0123456789abcdef", "fedcba9876543210"]
    assert (
        done["unify_meta"]["call_kind"] == "memory_writer"
        and "msg_count: malformed" in done["unify_meta"]["flags"]
    )
    # one lane per journal: the writer's journal never holds the fork's call
    assert {r["lane"] for r in _rows(journal)} == {"writer"}
    assert {r["lane"] for r in _rows(tmp_path / "lanes" / "fork.jsonl")} == {"fork"}
    # the readers: the writer pass's reconciliation and the run report
    summary = asyncio.run(
        reconcile.reconcile_window(
            str(journal),
            0,
            journal.stat().st_size,
            models=[MODEL],
            worst_case_usd=Decimal("0.65625"),
            budget_s=1,
        )
    )
    assert (
        summary["calls"],
        summary["priced"],
        summary["unknown"],
        summary["usd"],
    ) == (1, 1, 0, "0.00125")
    by_episode = metering.fork_by_episode(
        tmp_path / "lanes" / "fork.jsonl", Decimal("0.65625")
    )
    assert by_episode == {
        "20261021T1-ab": {
            "requests": 1,
            "known_usd": "0.00125",
            "unknown": 0,
            "booked_usd": "0.00125",
        }
    }


def test_a_stream_is_relayed_as_it_arrives_metered_and_redacted(tmp_path, upstream):
    doc, tokens = _doc(tmp_path / "lanes", upstream.base)
    upstream.echo = FAKE_KEY.encode()
    with serving(doc) as relay:
        status, reply = call(
            relay.port, "writer", tokens["writer"], {**BODY, "stream": True}
        )
    assert status == 200 and isinstance(reply, bytes)
    assert (
        b"data: [DONE]" in reply
        and FAKE_KEY.encode() not in reply
        and b"hel[redacted]" in reply
    )
    _, done = _rows(tmp_path / "lanes" / "writer.jsonl")
    assert done["stream"] is True and done["status"] == "completed"
    assert (
        done["account_charge"] == "0.0004" and done["generation_id"] == "gen-stream-1"
    )
    assert done["finish_reason"] == "stop" and done["tokens"]["input"] == 40


@pytest.mark.parametrize("status", sorted(rs.REJECTED_STATUSES))
def test_a_refusal_before_generating_reaches_the_caller_and_is_booked_at_nothing(
    tmp_path, status
):
    """A 4xx the provider answers without usage (it refused before generating) costs nothing: booked at 0, its class
    journaled, and the caller gets the provider's own status and body, redacted."""
    up = Upstream("error", status=status, echo=FAKE_KEY.encode())
    try:
        doc, tokens = _doc(tmp_path / "lanes", up.base, cap="1.00")
        with serving(doc) as relay:
            got, reply = call(relay.port, "helper", tokens["helper"], BODY)
            again = call(relay.port, "helper", tokens["helper"], BODY)[0]
            state = relay.ledger.state()
    finally:
        up.close()
    assert got == again == status and FAKE_KEY not in json.dumps(reply)
    done = [
        r
        for r in _rows(tmp_path / "lanes" / "helper.jsonl")
        if r["origin"] == "response"
    ]
    assert {
        (r["status"], r["error_type"], r["account_charge"], r["measurement"])
        for r in done
    } == {("failed", f"http_{status}", "0", "rejected_before_generation")}
    assert {(r["charge_class"], r["booked_usd"]) for r in done} == {("rejected", "0")}
    assert (
        state["booked_usd"] == "0"
    )  # under a cap that holds one worst case, both were admitted


@pytest.mark.parametrize(
    "mode, status",
    [
        ("error", 408),  # a timeout: the provider may have been generating
        ("error", 500),
        ("error", 503),
        ("stream_status", 429),  # streamed output: billing may have happened
    ],
)
def test_a_failure_that_may_have_been_billed_keeps_the_worst_case(
    tmp_path, mode, status
):
    up = Upstream(mode, status=status, cost=None)
    try:
        doc, tokens = _doc(tmp_path / "lanes", up.base, cap="1.00")
        with serving(doc) as relay:
            got = call(relay.port, "writer", tokens["writer"], BODY)[0]
            state = relay.ledger.state()
    finally:
        up.close()
    assert got == status
    _, done = _rows(tmp_path / "lanes" / "writer.jsonl")
    assert (done["account_charge"], done["charge_class"], done["booked_usd"]) == (
        None,
        "worst_case",
        "0.65625",
    )
    assert state["booked_usd"] == "0.65625"


def test_a_refusal_that_reports_usage_is_not_taken_for_one_before_generating(tmp_path):
    up = Upstream(
        lambda body: (429, {"error": {"code": 429}, "usage": {"prompt_tokens": 50}})
    )
    try:
        doc, tokens = _doc(tmp_path / "lanes", up.base)
        with serving(doc) as relay:
            assert call(relay.port, "fork", tokens["fork"], BODY)[0] == 429
    finally:
        up.close()
    _, done = _rows(tmp_path / "lanes" / "fork.jsonl")
    assert (done["account_charge"], done["charge_class"], done["booked_usd"]) == (
        None,
        "worst_case",
        "0.65625",
    )


def test_an_unreachable_upstream_is_a_call_that_cost_nothing(tmp_path):
    closed = socket.socket()
    closed.bind(("127.0.0.1", 0))
    port = closed.getsockname()[1]
    closed.close()
    doc, tokens = _doc(tmp_path / "lanes", f"http://127.0.0.1:{port}/api/v1")
    with serving(doc) as relay:
        status, reply = call(relay.port, "writer", tokens["writer"], BODY)
        state = relay.ledger.state()
    assert status == 503 and reply["error"]["type"] == "writer_upstream_unreachable"
    _, done = _rows(tmp_path / "lanes" / "writer.jsonl")
    assert (done["status"], done["measurement"], done["account_charge"]) == (
        "failed",
        "not_sent",
        "0",
    )
    assert done["charge_class"] == "not_sent"
    assert state["booked_usd"] == "0"


def test_an_unpriced_call_is_booked_at_its_worst_case_and_the_cap_then_refuses_every_lane(
    tmp_path,
):
    up = Upstream(cost=None)
    try:
        doc, tokens = _doc(tmp_path / "lanes", up.base, cap="1.00")
        with serving(doc) as relay:
            assert call(relay.port, "writer", tokens["writer"], BODY)[0] == 200
            assert relay.ledger.state()["booked_usd"] == "0.65625"
            refused = {
                lane: call(relay.port, lane, tokens[lane], BODY) for lane in LANES
            }
    finally:
        up.close()
    for lane, (status, reply) in refused.items():
        assert status == 403
        assert reply["error"]["type"] == f"{lane}_budget_exhausted"
        assert reply["error"]["reason"] == f"{lane}_cap_reached"
        assert (
            _refusal_kind(reply["error"]) == "cap"
        )  # what the writer reads: the cap, no further pass
    assert len(up.requests) == 1


def test_pinned_prices_price_a_call_the_provider_did_not(tmp_path):
    up = Upstream(cost=None)
    try:
        doc, tokens = _doc(
            tmp_path / "lanes",
            up.base,
            cap="1.00",
            prices={MODEL: {"input": "1.25", "output": "10", "cache_write": "2"}},
        )
        with serving(doc) as relay:
            for _ in range(3):
                assert call(relay.port, "writer", tokens["writer"], BODY)[0] == 200
            state = relay.ledger.state()
    finally:
        up.close()
    # (120 x 1.25 + 7 x 10 + 20 x 2) / 1e6 per call, as unify/memory/reconcile.estimate prices it
    assert Decimal(state["booked_usd"]) == 3 * Decimal("0.00026")


def test_a_call_in_flight_holds_its_worst_case_until_it_ends(tmp_path):
    up = Upstream("hold")
    try:
        doc, tokens = _doc(tmp_path / "lanes", up.base, cap="1.00")
        with serving(doc) as relay:
            first = {}
            t = threading.Thread(
                target=lambda: first.update(
                    r=call(relay.port, "fork", tokens["fork"], BODY)
                )
            )
            t.start()
            while not up.requests:
                threading.Event().wait(0.05)
            status, reply = call(relay.port, "writer", tokens["writer"], BODY)
            assert status == 403 and reply["error"]["reason"] == "writer_cap_reached"
            up.release()
            t.join(30)
            assert first["r"][0] == 200
            assert call(relay.port, "writer", tokens["writer"], BODY)[0] == 200
    finally:
        up.close()


@pytest.mark.parametrize("stream", [False, True])
def test_a_call_made_right_after_another_is_admitted_against_what_that_one_cost(
    tmp_path, stream
):
    """The terminal row is journaled before the caller has the whole reply: under a cap that holds one call at its
    worst case and not two, back-to-back calls are never refused for the one before."""
    up = Upstream()
    try:
        doc, tokens = _doc(tmp_path / "lanes", up.base, cap="1.00")
        with serving(doc) as relay:
            got = [
                call(
                    relay.port, "writer", tokens["writer"], {**BODY, "stream": stream}
                )[0]
                for _ in range(25)
            ]
    finally:
        up.close()
    assert got == [200] * 25


def test_each_run_starts_with_the_whole_cap(tmp_path):
    """The cap holds one run (one relay): what an earlier run booked, in the same journals, does not count."""
    up = Upstream(cost=None)
    try:
        doc, tokens = _doc(tmp_path / "lanes", up.base, cap="1.00")
        with serving(doc) as relay:
            assert call(relay.port, "writer", tokens["writer"], BODY)[0] == 200
            assert (
                call(relay.port, "fork", tokens["fork"], BODY)[0] == 403
            )  # this run's cap is spent
        doc2, tokens2 = _doc(tmp_path / "lanes", up.base, cap="1.00")
        with serving(doc2) as relay:
            status, _reply = call(relay.port, "fork", tokens2["fork"], BODY)
            state = relay.ledger.state()
    finally:
        up.close()
    assert status == 200 and state == {
        "booked_usd": "0.65625",
        "cap_usd": "1.00",
        "calls": 1,
    }


def test_a_call_with_no_free_slot_is_refused_for_now(tmp_path, monkeypatch):
    monkeypatch.setattr(rs, "MAX_IN_FLIGHT", 1)
    monkeypatch.setattr(rs, "BUSY_WAIT_S", 0.3)
    up = Upstream("hold")
    try:
        doc, tokens = _doc(tmp_path / "lanes", up.base)
        with serving(doc) as relay:
            t = threading.Thread(
                target=call, args=(relay.port, "writer", tokens["writer"], BODY)
            )
            t.start()
            while not up.requests:
                threading.Event().wait(0.05)
            status, reply = call(relay.port, "helper", tokens["helper"], BODY)
            up.release()
            t.join(30)
    finally:
        up.close()
    assert status == 403 and reply["error"] == {
        "code": 403,
        "type": "helper_route_busy",
        "message": "every slot of the lane relay is in flight",
        "reason": "helper_in_flight",
    }
    assert (
        _refusal_kind(reply["error"]) == "busy"
    )  # the writer tries again and books nothing for it


def test_concurrent_calls_on_every_lane_are_each_served_and_journaled_once(tmp_path):
    up = Upstream(delay=0.2)
    try:
        # room for every call in flight at its worst case (8 x 0.65625), so only the slots bound them
        doc, tokens = _doc(tmp_path / "lanes", up.base, cap="20.00")
        results: list = []
        with serving(doc) as relay:
            threads = [
                threading.Thread(
                    target=lambda lane=lane: results.append(
                        call(relay.port, lane, tokens[lane], BODY)
                    )
                )
                for lane in LANES * 4
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join(60)
            state = relay.ledger.state()
    finally:
        up.close()
    assert [r[0] for r in results] == [200] * 12
    assert 1 < up.most_in_flight <= rs.MAX_IN_FLIGHT
    ids = []
    for lane in LANES:
        rows = _rows(tmp_path / "lanes" / f"{lane}.jsonl")
        assert [r["origin"] for r in rows].count("request_started") == 4
        assert (
            sorted(r["status"] for r in rows if r["origin"] == "response")
            == ["completed"] * 4
        )
        ids += [r["request_attempt_id"] for r in rows if r["origin"] == "response"]
    assert len(set(ids)) == 12
    assert Decimal(state["booked_usd"]) == 12 * Decimal("0.00125")


@pytest.mark.parametrize(
    "change, rule",
    [
        ({"key": "has a space"}, "the provider key is missing or malformed"),
        (
            {"upstream": "http://openrouter.ai/api/v1"},
            "the upstream must be https, or plain http to 127.0.0.1",
        ),
        (
            {"upstream": "https://user:pw@openrouter.ai/api/v1"},
            "no user, password, query or fragment",
        ),
        ({"cap_usd": "1e3"}, "the spend cap must be a plain decimal string"),
        ({"cap_usd": "0"}, "the spend cap must be above 0"),
        (
            {"worst_case_usd": {"openai/gpt-6-sol": "13.125"}},
            "every lane model needs a worst case per call",
        ),
        (
            {"lanes": {"other": {"token": "t" * 43, "models": [MODEL]}}},
            "the lanes must be some of writer, helper and fork",
        ),
        (
            {"lanes": {"writer": {"token": "short", "models": [MODEL]}}},
            "a lane token is malformed",
        ),
        ({"format": "other/1"}, "the configuration's format differs"),
    ],
)
def test_a_configuration_is_refused_by_rule_and_never_quotes_a_value(
    tmp_path, change, rule
):
    doc, _ = _doc(tmp_path / "lanes", "https://openrouter.ai/api/v1")
    doc.update(change)
    with pytest.raises(rs.ConfigError) as err:
        rs.Config(doc)
    assert (
        rule in str(err.value)
        and FAKE_KEY not in str(err.value)
        and "has a space" not in str(err.value)
    )


def test_tokens_must_differ_from_each_other_and_from_the_key(tmp_path):
    same = secrets.token_urlsafe(32)
    doc, _ = _doc(
        tmp_path / "lanes",
        "https://openrouter.ai/api/v1",
        tokens={"writer": same, "helper": same, "fork": same},
    )
    with pytest.raises(rs.ConfigError, match="every lane token must be distinct"):
        rs.Config(doc)


def test_a_journal_folder_others_can_write_is_refused(tmp_path):
    doc, _ = _doc(tmp_path / "lanes", "https://openrouter.ai/api/v1")
    (tmp_path / "lanes").chmod(0o777)
    with pytest.raises(rs.ConfigError, match="no one else can write"):
        rs.Config(doc)


def test_connections_beyond_the_bound_are_closed_unread_and_the_relay_serves_on(
    tmp_path, upstream, monkeypatch
):
    monkeypatch.setattr(rs, "MAX_CONNECTIONS", 2)
    doc, tokens = _doc(tmp_path / "lanes", upstream.base)
    with serving(doc) as relay:
        idle = [
            socket.create_connection(("127.0.0.1", relay.port), timeout=10)
            for _ in range(2)
        ]
        try:
            threading.Event().wait(0.3)  # both accepted, each holding its thread
            with pytest.raises((ConnectionError, http.client.HTTPException, OSError)):
                call(relay.port, "writer", tokens["writer"], BODY)
        finally:
            for s in idle:
                s.close()
        threading.Event().wait(0.3)
        assert call(relay.port, "writer", tokens["writer"], BODY)[0] == 200
    assert len(upstream.requests) == 1


def test_a_body_that_asks_for_exactly_one_completion_is_served(tmp_path, upstream):
    doc, tokens = _doc(tmp_path / "lanes", upstream.base)
    with serving(doc) as relay:
        status, _reply = call(relay.port, "writer", tokens["writer"], {**BODY, "n": 1})
    assert status == 200 and upstream.requests[0]["body"]["n"] == 1


def test_a_refused_large_body_is_read_in_pieces_and_the_refusal_still_arrives(
    tmp_path, upstream, monkeypatch
):
    """A refused body is drained in pieces of DISCARD_CHUNK (never one read of the whole bound), and the caller gets
    the refusal rather than a reset connection."""
    sizes: list[int] = []
    real = rs.Relay._discard_body

    def counting(handler):
        rfile = handler.rfile

        class Counting:
            def read(self, n=-1):
                sizes.append(n)
                return rfile.read(n)

        handler.rfile = Counting()
        try:
            real(handler)
        finally:
            handler.rfile = rfile

    monkeypatch.setattr(rs.Relay, "_discard_body", staticmethod(counting))
    doc, tokens = _doc(tmp_path / "lanes", upstream.base)
    big = json.dumps({**BODY, "padding": "x" * (5 << 20)}).encode()
    with serving(doc) as relay:
        status, reply = call(relay.port, "writer", "not-this-lanes-token", raw=big)
    assert status == 403 and reply["error"]["reason"] == "no_lane_token"
    assert (
        sizes
        and max(sizes) == rs.DISCARD_CHUNK
        and sum(sizes) >= len(big) - rs.DISCARD_CHUNK
    )
    assert upstream.requests == []

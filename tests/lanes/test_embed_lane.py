"""The embedding lane: the relay's ``/embed/v1/embeddings`` route (its own token, one allow-listed model, exactly
``{model, input}``, bounded size, the run's ledger), the client's token read once from ``UNIFY_EMBED_TOKEN_FD`` and
sent in place of the key, and a writer process that holds no real key with the relay serving embeddings.
Stand-in provider, fake key, no network."""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest

from tests.lanes.test_relay_server import _doc, _rows, call, serving
from tests.lanes.upstream import FAKE_KEY, Upstream
from unify.lanes import embed_token

EMBED = "openai/text-embedding-3-small"
OK = {"model": EMBED, "input": ["a text", "another"]}


def _embedding(body):
    return 200, {
        "data": [
            {"index": i, "embedding": [0.5, 0.25]} for i, _ in enumerate(body["input"])
        ],
        "usage": {"prompt_tokens": 4, "cost": 0.00000008},
    }


@pytest.fixture
def lane(tmp_path):
    up = Upstream(_embedding)
    doc, tokens = _doc(tmp_path / "lanes", up.base, lanes=("writer", "embed"))
    doc["lanes"]["embed"]["models"] = [EMBED]
    doc["worst_case_usd"][EMBED] = "0.05"
    with serving(doc) as relay:
        yield SimpleNamespace(
            relay=relay, tokens=tokens, up=up, folder=tmp_path / "lanes"
        )
    up.close()


def _embed(lane, body, token=None, raw=None):
    return call(
        lane.relay.port,
        "embed",
        token or lane.tokens["embed"],
        body,
        raw=raw,
        path="/embed/v1/embeddings",
    )


def test_an_embedding_goes_upstream_unchanged_and_is_booked_on_the_run_ledger(lane):
    status, reply = _embed(lane, OK)
    assert status == 200 and len(reply["data"]) == 2
    (seen,) = lane.up.requests
    assert seen["path"] == "/api/v1/embeddings" and seen["body"] == OK  # nothing added
    assert seen["headers"]["authorization"] == f"Bearer {FAKE_KEY}"
    _, done = _rows(lane.folder / "embed.jsonl")
    assert (done["lane"], done["account_charge"], done["charge_class"]) == (
        "embed",
        "0.00000008",
        "charge",
    )
    assert lane.relay.ledger.state()["booked_usd"] == "0.00000008"


@pytest.mark.parametrize(
    "body, reason",
    [
        ({**OK, "dimensions": 256}, "embedding_fields"),
        ({**OK, "encoding_format": "base64"}, "embedding_fields"),
        ({"model": EMBED}, "embedding_fields"),
        ({"model": EMBED, "input": "one text"}, "embedding_input"),
        ({"model": EMBED, "input": []}, "embedding_input"),
        ({"model": EMBED, "input": ["x"] * 129}, "embedding_input"),
        ({"model": EMBED, "input": [1, 2]}, "embedding_input"),
        ({"model": EMBED, "input": ["x" * (2 << 20) + "x"]}, "embedding_input_size"),
        ({"model": "openai/text-embedding-3-large", "input": ["x"]}, "not_allowed"),
    ],
)
def test_an_embedding_body_off_the_rules_is_refused_before_upstream(lane, body, reason):
    status, reply = _embed(lane, body)
    assert status == 403 and reply["error"] == {
        "code": 403,
        "type": "off_model_refused",
        "message": f"this lane may call only {EMBED}",
        "reason": reason,
    }
    assert lane.up.requests == []


def test_the_embedding_lane_takes_its_own_token_only(lane):
    status, reply = _embed(lane, OK, token=lane.tokens["writer"])
    assert status == 403 and reply["error"]["reason"] == "other_lane_token"
    status, reply = call(
        lane.relay.port,
        "writer",
        lane.tokens["embed"],
        {"model": EMBED, "messages": []},
    )
    assert status == 403 and reply["error"]["reason"] == "other_lane_token"
    assert lane.up.requests == []


@pytest.fixture
def fresh_token(monkeypatch):
    for name, value in (("_TOKEN", None), ("_REFUSAL", None), ("_TRIED", False)):
        monkeypatch.setattr(embed_token, name, value)


def _pipe(data: bytes) -> int:
    r, w = os.pipe()
    os.write(w, data)
    os.close(w)
    os.set_inheritable(r, True)
    return r


def test_the_token_is_read_once_from_its_descriptor_which_is_closed_and_its_name_removed(
    fresh_token,
):
    token = "embed-lane-token-" + "0" * 32
    r = _pipe((token + "\n").encode())
    environ = {"UNIFY_EMBED_TOKEN_FD": str(r), "OTHER": "x"}
    embed_token.settle(environ)
    assert embed_token.token() == token and environ == {"OTHER": "x"}
    with pytest.raises(OSError):
        os.fstat(r)


def test_an_unusable_descriptor_refuses_every_embedding_and_never_falls_back_to_the_key(
    fresh_token, monkeypatch
):
    from unify.common import embeddings
    from unify.settings import SETTINGS

    embed_token.settle({"UNIFY_EMBED_TOKEN_FD": "99999"})
    monkeypatch.setattr(
        SETTINGS, "UNIFY_EMBED_URL", "http://127.0.0.1:9/embed/v1/embeddings"
    )
    posted = []
    monkeypatch.setattr(embeddings.httpx, "post", lambda *a, **k: posted.append(k))
    with pytest.raises(embeddings.EmbeddingRouteRefused, match="UNIFY_EMBED_TOKEN_FD"):
        embeddings._openrouter(["a text"])
    assert posted == []


def test_the_client_sends_the_lanes_token_never_the_key(fresh_token, monkeypatch):
    import unillm
    from pydantic import SecretStr

    from unify.common import embeddings
    from unify.settings import SETTINGS

    token = "embed-lane-token-" + "1" * 32
    embed_token.settle({"UNIFY_EMBED_TOKEN_FD": str(_pipe(token.encode()))})
    monkeypatch.setattr(
        SETTINGS, "UNIFY_EMBED_URL", "http://127.0.0.1:9/embed/v1/embeddings"
    )
    monkeypatch.setattr(unillm.SETTINGS, "OPENROUTER_API_KEY", SecretStr(FAKE_KEY))
    sent = []

    class _Reply:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "data": [{"index": 0, "embedding": [1.0]}],
                "usage": {"prompt_tokens": 2},
            }

    monkeypatch.setattr(
        embeddings.httpx, "post", lambda url, **k: sent.append(k["headers"]) or _Reply()
    )
    embeddings._openrouter(["a text"])
    assert sent == [{"Authorization": f"Bearer {token}"}]


def test_a_writer_process_holds_no_real_key_with_the_relay_serving_embeddings(
    fresh_token, monkeypatch
):
    """The relay-served embedding route sends the lane's token, so the writer processes' keys stay masked; only an
    embedding route the user set (no lane token) keeps them, as before."""
    from unify.memory import writer_service as ws
    from unify.memory.integration import async_pass, switch

    settings = SimpleNamespace(UNIFY_EMBED_URL="http://127.0.0.1:9/embed/v1/embeddings")
    user_set = ws.without_provider_keys({"OPENROUTER_API_KEY": FAKE_KEY}, settings)
    assert user_set == {
        "OPENROUTER_API_KEY": FAKE_KEY
    }  # no lane token: the user's own route
    token = "embed-lane-token-" + "2" * 32
    embed_token.settle({"UNIFY_EMBED_TOKEN_FD": str(_pipe(token.encode()))})
    masked = ws.without_provider_keys({"OPENROUTER_API_KEY": FAKE_KEY}, settings)
    assert masked["OPENROUTER_API_KEY"] == ws.NO_PROVIDER_KEY
    monkeypatch.setenv("OPENROUTER_API_KEY", FAKE_KEY)
    monkeypatch.setattr(
        async_pass, "_route_token", lambda s: "writer-route-token-for-a-test"
    )
    monkeypatch.setattr(switch, "helper_lane_route", lambda: (None, "no helper lane"))
    monkeypatch.setattr(ws, "_settings", lambda: settings)
    env, fds = ws._token_pipes()
    try:
        assert (
            FAKE_KEY not in json.dumps(env)
            and env["OPENROUTER_API_KEY"] == ws.NO_PROVIDER_KEY
        )
        fd = int(env["UNIFY_EMBED_TOKEN_FD"])
        assert fd in fds and os.read(fd, 4096) == (token + "\n").encode()
    finally:
        for fd in fds:
            os.close(fd)

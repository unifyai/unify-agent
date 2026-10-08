"""The tool adapter: a record-only observer on the environment observer seam."""

from __future__ import annotations

import threading
import time

import pytest

from unify.memory_v2.integration.adapters.tool import (
    CYCLE,
    TRUNCATED,
    RecordingObserver,
    action_fingerprints,
    jsonable,
    split_channel,
)
from unify.memory_v2.redact import Redactor

try:  # the seam's own dataclass and dispatch when this checkout has them
    from unify.function_manager.primitives import observers as seam

    EnvCall = seam.EnvCall
except ImportError:  # pragma: no cover - a stand-in with the same fields
    from dataclasses import dataclass, field

    seam = None

    @dataclass(frozen=True)
    class EnvCall:  # type: ignore[no-redef]
        namespace: str
        method: str
        effect: str
        args: tuple
        kwargs: dict = field(default_factory=dict)
        via: str = "primitives"


def _after(obs, call, *, result=None, error=None, intercepted=False):
    obs.after(
        call,
        result=result,
        error=error,
        intercepted=intercepted,
        started=0.0,
        elapsed_s=0.01,
    )


def test_ok_call_is_one_tool_action_with_a_deep_copy_of_the_result():
    obs = RecordingObserver()
    obs.set_cell(3)
    result = {"user": "ada", "friends": [1, 2]}
    call = EnvCall(
        "appworld",
        "show_profile",
        "read",
        ("ada",),
        {"full": True},
        "primitives",
    )
    _after(obs, call, result=result)
    result["friends"].append(3)  # the caller mutates its own object afterwards
    [a] = obs.drain().actions
    assert (a.kind, a.cell, a.channel, a.method) == (
        "tool",
        3,
        "appworld",
        "show_profile",
    )
    assert (a.args, a.kwargs) == (["ada"], {"full": True})
    assert a.response == {"user": "ada", "friends": [1, 2]}
    assert (a.status, a.effect, a.error) == ("ok", "read", None)
    assert obs.drain().actions == []


def test_raw_global_call_takes_the_app_as_channel_and_unknown_effect():
    call = EnvCall("apis", "spotify.login", "", (), {"username": "ada"}, "global")
    assert split_channel(call) == ("spotify", "login")
    assert split_channel(EnvCall("apis", "login", "", (), {}, "global")) == (
        "apis",
        "login",
    )
    obs = RecordingObserver()
    _after(obs, call, result={"access_token": "t"})
    [a] = obs.drain().actions
    assert (a.channel, a.method, a.effect, a.kwargs) == (
        "spotify",
        "login",
        "unknown",
        {"username": "ada"},
    )


def test_error_call_records_type_and_first_line_only():
    obs = RecordingObserver()
    call = EnvCall("venmo", "send", "write", (), {"amount": 5}, "primitives")
    _after(
        obs,
        call,
        result=None,
        error=ValueError("amount too high\ntraceback detail"),
    )
    [a] = obs.drain().actions
    assert (a.status, a.response, a.effect) == ("error", None, "write")
    assert a.error == "ValueError: amount too high"
    assert action_fingerprints([a]) == {
        "venmo.send": {"shapes": [], "errors": ["ValueError: amount too high"]},
    }


def test_size_cap_replaces_a_large_value_with_a_bounded_marker_and_keeps_its_shape():
    obs = RecordingObserver(max_value_bytes=512)
    big = {"rows": [{"id": i, "name": "x" * 50} for i in range(200)]}
    _after(
        obs,
        EnvCall("db", "query", "read", ("q" * 2000,), {}, "primitives"),
        result=big,
    )
    [a] = obs.drain().actions
    assert set(a.response) == {TRUNCATED}
    marker = a.response[TRUNCATED]
    assert marker["bytes"] > 512 and len(marker["preview"].encode()) <= 256
    assert isinstance(a.args, list) and set(a.args[0]) == {
        TRUNCATED,
    }  # capped, still a list
    fp = action_fingerprints([a])
    assert fp["db.query"]["shapes"] == ["{rows:[{id:int,name:str}]}"]


def test_count_cap_keeps_the_first_calls_and_counts_the_rest():
    obs = RecordingObserver(max_calls=2)
    for i in range(5):
        _after(obs, EnvCall("ns", f"m{i}", "read", (), {}, "primitives"), result=i)
    drained = obs.drain()
    assert [a.method for a in drained.actions] == ["m0", "m1"]
    assert drained.dropped == 3
    assert obs.drain().dropped == 0  # counts reset with each drain


def test_redacts_a_fake_key_in_args_response_and_error():
    secret = "fake-secret-" + "z" * 12  # pragma: allowlist secret
    shaped = "sk-or-v1-" + "0" * 64  # pragma: allowlist secret
    obs = RecordingObserver(Redactor({"FAKE_API_KEY": secret}))
    _after(
        obs,
        EnvCall(
            "http",
            "get",
            "read",
            (f"Bearer {secret}",),
            {"k": shaped},
            "primitives",
        ),
        result={"echo": secret},
    )
    _after(
        obs,
        EnvCall("http", "get", "read", (), {}, "primitives"),
        error=RuntimeError(f"bad key {shaped}"),
    )
    ok, bad = obs.drain().actions
    text = repr((ok, bad))
    assert secret not in text and shaped not in text
    assert ok.args == ["Bearer <secret:FAKE_API_KEY>"]
    assert ok.kwargs == {"k": "<redacted:key-shaped>"}
    assert ok.response == {"echo": "<secret:FAKE_API_KEY>"}
    assert bad.error == "RuntimeError: bad key <redacted:key-shaped>"


def test_never_intercepts_and_drops_calls_another_observer_answered():
    obs = RecordingObserver()
    call = EnvCall("ns", "m", "write", (), {}, "primitives")
    assert obs.complete is False
    assert obs.before(call) is None
    _after(obs, call, result="speculated", intercepted=True)
    drained = obs.drain()
    assert drained.actions == [] and drained.intercepted == 1


def test_jsonable_handles_objects_tuples_bytes_and_pydantic_models():
    class Model:
        def model_dump(self, mode="python"):
            return {"a": (1, 2)}

    out = jsonable(
        {
            "t": (1, 2),
            "b": b"abc",
            "m": Model(),
            "o": object(),
            1: "k",
            "f": float("nan"),
        },
    )
    assert (
        out["t"] == [1, 2]
        and out["b"] == {"__bytes__": 3}
        and out["m"] == {"a": [1, 2]}
    )
    assert "__repr__" in out["o"] and out["1"] == "k" and out["f"] == "nan"


@pytest.mark.skipif(seam is None, reason="the observer seam is not in this checkout")
def test_through_the_seam_dispatch_and_a_raw_global_proxy():
    obs = RecordingObserver()
    call = EnvCall("ns", "m", "read", (1,), {}, "primitives")
    assert seam.dispatch([obs], call, lambda: {"v": 1}) == {"v": 1}
    with pytest.raises(KeyError):
        seam.dispatch([obs], call, lambda: {}["missing"])

    class Spotify:
        def login(self, username):
            return {"user": username}

    class Apis:
        spotify = Spotify()

    g = seam.observed_globals({"apis": Apis()}, enabled=True)
    with seam.observing(obs):
        assert not seam.complete_required()
        obs.set_cell(1)
        assert g["apis"].spotify.login(username="ada") == {"user": "ada"}
    ok, err, glob = obs.drain().actions
    assert (ok.status, ok.response, err.status, err.error) == (
        "ok",
        {"v": 1},
        "error",
        "KeyError: 'missing'",
    )
    assert (glob.cell, glob.channel, glob.method, glob.kwargs, glob.response) == (
        1,
        "spotify",
        "login",
        {"username": "ada"},
        {"user": "ada"},
    )


def test_a_self_referencing_result_is_cut_at_the_cycle_quickly():
    loop: dict = {}
    for i in range(30):
        loop[f"k{i}"] = loop
    t0 = time.monotonic()
    obs = RecordingObserver()
    _after(obs, EnvCall("ns", "m", "read", (), {}, "primitives"), result=loop)
    assert time.monotonic() - t0 < 2.0
    [a] = obs.drain().actions
    assert a.response == {f"k{i}": CYCLE for i in range(30)}
    lst: list = [1]
    lst.append(lst)
    assert jsonable(lst) == [1, CYCLE]


def test_a_wide_shared_dag_is_bounded_by_the_node_budget_quickly():
    node: dict = {"leaf": 1}
    for _ in range(12):  # 30**12 paths if expanded
        node = {f"k{i}": node for i in range(30)}
    t0 = time.monotonic()
    out = jsonable(node, max_nodes=10_000)
    assert time.monotonic() - t0 < 2.0

    def count(v):
        if isinstance(v, dict):
            return 1 + sum(count(x) for x in v.values())
        if isinstance(v, list):
            return 1 + sum(count(x) for x in v)
        return 1

    assert count(out) <= 3 * 10_000  # each budgeted node is at most a 3-node marker
    assert "__more__" in str(out)  # the rest is counted, not copied
    obs = RecordingObserver(max_value_bytes=4096)
    t0 = time.monotonic()
    _after(obs, EnvCall("ns", "m", "read", (), {}, "primitives"), result=node)
    assert time.monotonic() - t0 < 2.0
    [a] = obs.drain().actions
    assert set(a.response) == {TRUNCATED}


def test_depth_cap_turns_deep_values_into_bounded_reprs():
    deep: list = []
    cur = deep
    for _ in range(50):
        nxt: list = []
        cur.append(nxt)
        cur = nxt
    out = jsonable(deep, depth=3)
    assert out[0][0][0] == {"__depth__": "list", "len": 1}

    class Opaque:
        def __repr__(self):
            return "o" * 5000

    assert len(jsonable([[[Opaque()]]], depth=3)[0][0][0]["__repr__"]) == 2000


def test_a_secret_straddling_a_cut_leaves_no_prefix():
    secret = "straddle-secret-" + "q" * 24  # pragma: allowlist secret
    shaped = "sk-proj-" + "A" * 40  # pragma: allowlist secret
    red = Redactor({"FAKE_TOKEN": secret})

    class Hostile:
        def __repr__(self):
            return "x" * 1990 + shaped

    out = jsonable(
        {"s": "y" * 95 + secret, "r": Hostile()},
        redactor=red,
        max_chars=100,
    )
    text = repr(out)
    for s in (secret, shaped):
        for n in range(4, len(s) + 1):
            assert s[:n] not in text, (s[:n], text[-80:])
    out = jsonable({"r": Hostile()}, redactor=red)
    assert shaped[:4] not in repr(out)
    obs = RecordingObserver(red, max_error_chars=40)
    _after(
        obs,
        EnvCall("ns", "m", "read", (), {}, "primitives"),
        error=RuntimeError("z" * 30 + secret),
    )
    [a] = obs.drain().actions
    assert len(a.error) <= 40 and secret[:4] not in a.error


def test_many_secrets_in_one_long_string_leave_no_prefix_at_any_cut():
    # Each replacement shortens the text, so a window redacted before the cut can come up short and
    # let the occurrence that straddles the window's end through. Every alignment is tried, so one
    # occurrence straddles each cut.
    secret = "straddle-secret-" + "q" * 24  # pragma: allowlist secret
    shaped = "sk-ant-" + "A" * 100  # pragma: allowlist secret
    red = Redactor({"FAKE_TOKEN": secret})

    class Text:
        def __init__(self, text):
            self.text = text

        def __repr__(self):
            return self.text

    units = (secret + " ", shaped + " ", secret + " " + shaped + " ")
    for unit in units:
        for pad in range(len(unit)):
            text = "." * pad + unit * 200
            obs = RecordingObserver(red, max_value_bytes=500, max_error_chars=2000)
            _after(
                obs,
                EnvCall("ns", "m", "read", (text,), {text: text}, "primitives"),
                result=text,
            )
            _after(
                obs,
                EnvCall("ns", "m", "read", (), {}, "primitives"),
                error=RuntimeError(text),
            )
            outs = [
                repr(jsonable(text, redactor=red, max_chars=2000)),
                repr(jsonable({text: Text(text)}, redactor=red)),
                repr(obs.drain().actions),
            ]
            for out in outs:
                for s in (secret, shaped):  # no prefix of 4 or more characters
                    assert s[:4] not in out, (unit[:8], pad, s[:4])


def test_a_multi_line_secret_in_an_error_leaves_no_first_line():
    secret = "first-line-of-a-key\nsecond-line-of-a-key"  # pragma: allowlist secret
    red = Redactor({"FAKE_PRIVATE_KEY": secret})
    obs = RecordingObserver(red)
    _after(
        obs,
        EnvCall("ns", "m", "read", (), {}, "primitives"),
        error=RuntimeError("bad key " + secret),
    )
    [a] = obs.drain().actions
    assert a.error == "RuntimeError: bad key <secret:FAKE_PRIVATE_KEY>"


def test_a_call_keeps_the_cell_it_started_in():
    obs = RecordingObserver()
    obs.set_cell(1)
    call = EnvCall("ns", "slow", "read", (), {}, "primitives")
    assert obs.before(call) is None
    obs.set_cell(2)  # the next cell starts before the slow call finishes
    _after(obs, call, result=1)
    _after(obs, EnvCall("ns", "fast", "read", (), {}, "primitives"), result=2)
    assert [(a.method, a.cell) for a in obs.drain().actions] == [
        ("slow", 1),
        ("fast", 2),
    ]


def test_concurrent_calls_from_threads_are_all_counted():
    obs = RecordingObserver(max_calls=500)

    def work(t):
        for i in range(100):
            call = EnvCall("ns", f"t{t}", "read", (i,), {}, "primitives")
            obs.before(call)
            _after(obs, call, result={"i": i})

    threads = [threading.Thread(target=work, args=(t,)) for t in range(8)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    drained = obs.drain()
    assert len(drained.actions) == 500 and drained.dropped == 300


@pytest.mark.skipif(seam is None, reason="the observer seam is not in this checkout")
def test_a_failing_recording_is_swallowed_and_the_call_outcome_stands():
    class Broken(Redactor):
        def obj(self, o):
            raise RuntimeError("redactor down")

        def text(self, s):
            raise RuntimeError("redactor down")

    obs = RecordingObserver(Broken())
    call = EnvCall("ns", "m", "read", ("a",), {}, "primitives")
    result = {"v": [1]}
    assert seam.dispatch([obs], call, lambda: result) is result
    with pytest.raises(KeyError):
        seam.dispatch([obs], call, lambda: {}["missing"])

    class BadStr(Exception):
        def __str__(self):
            raise RuntimeError("no str")

    plain = RecordingObserver()
    with pytest.raises(BadStr):
        seam.dispatch([plain], call, lambda: (_ for _ in ()).throw(BadStr()))
    drained = obs.drain()
    assert drained.actions == [] and drained.failed == 2
    [a] = plain.drain().actions
    assert a.error == "BadStr: <unprintable>"

import json

import pytest
from unify.memory_v2.episodes import Action
from unify.memory_v2.replay import (
    RecordedEnv,
    RecordedError,
    ReplayMiss,
    UnknownEffect,
    call_of,
    calls,
    env_from,
)


def test_serves_recorded_and_raises_on_miss_and_error():
    env = RecordedEnv(
        [
            Action(0, "venmo", "login", [], {"username": "a"}, {"token": "t"}, "ok"),
            Action(
                0,
                "venmo",
                "pay",
                [],
                {"amount": 5},
                None,
                "error",
                error="HTTP 422",
            ),
        ],
    )
    assert env.venmo.login(username="a") == {"token": "t"}
    with pytest.raises(ReplayMiss):
        env.venmo.login(username="b")
    with pytest.raises(RecordedError, match="422"):
        env.venmo.pay(amount=5)
    assert env.served == [("venmo", "login"), ("venmo", "pay")]


# --- the harness's replay helper (memlab.replay.env_from) ----------------------------------------------------

LOGIN = {
    "cell": 0,
    "channel": "venmo",
    "method": "login",
    "args": [],
    "kwargs": {"username": "a", "password": "p"},
    "response": {"token": "t"},
    "status": "ok",
    "effect": "read",
    "index": 0,  # an exported row's extra keys are ignored
}
POST = {
    "channel": "slack",
    "method": "post",
    "args": ["#ops"],
    "kwargs": {"text": "deploy done"},
    "response": {"ok": True},
    "status": "ok",
    "effect": "write",
}
DELETE = {
    "channel": "slack",
    "method": "delete",
    "args": ["#ops"],
    "kwargs": {"ts": "1"},
    "response": None,
    "status": "error",
    "effect": "write",
    "error": "HTTP 403",
}
UNRECORDED = {**POST, "kwargs": {"text": "never sent"}, "status": "unrecorded"}
ROWS = [LOGIN, POST, DELETE, UNRECORDED]


def test_env_from_serves_only_the_exact_recorded_call():
    env = env_from(ROWS)
    # keyword order does not matter; every value does
    assert env.venmo.login(password="p", username="a") == {"token": "t"}
    assert env.slack.post("#ops", text="deploy done") == {"ok": True}
    for call in (
        lambda: env.venmo.login(username="a"),  # an argument missing
        lambda: env.venmo.login(username="a", password="q"),  # another value
        # the channel was recorded positionally, here it is a keyword
        lambda: env.slack.post(channel="#ops", text="deploy done"),
        # recorded, but never as ok or error
        lambda: env.slack.post("#ops", text="never sent"),
        lambda: env.slack.edit("#ops", text="deploy done"),  # another method
        lambda: env.teams.post("#ops", text="deploy done"),  # another channel
    ):
        with pytest.raises(ReplayMiss):
            call()
    with pytest.raises(RecordedError, match="403"):
        env.slack.delete("#ops", ts="1")
    assert len(env.misses) == 6 and env.misses[0] == {
        "channel": "venmo",
        "method": "login",
        "args": [],
        "kwargs": {"username": "a"},
    }


def test_env_from_records_the_calls_it_served_in_order():
    env = env_from(ROWS)
    env.slack.post("#ops", text="deploy done")
    env.venmo.login(username="a", password="p")
    with pytest.raises(RecordedError):
        env.slack.delete("#ops", ts="1")
    with pytest.raises(ReplayMiss):
        env.slack.post("#ops", text="other")
    assert env.issued() == [
        call_of("slack", "post", ["#ops"], {"text": "deploy done"}),
        call_of("venmo", "login", [], {"username": "a", "password": "p"}),
        call_of("slack", "delete", ["#ops"], {"ts": "1"}),
    ]  # a recorded error was issued; a miss was not served
    assert env.issued(effect="write") == calls(ROWS, effect="write")
    assert env.served == [("slack", "post"), ("venmo", "login"), ("slack", "delete")]
    # Actions and rows replay alike, and from_jsonl reads exported rows
    login = Action(0, "venmo", "login", [], LOGIN["kwargs"], {}, "ok")
    assert calls([login]) == calls([LOGIN])


def test_env_from_reads_exported_rows_from_jsonl(tmp_path):
    path = tmp_path / "calls.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in ROWS) + "\n")
    env = RecordedEnv.from_jsonl(path)
    assert env.venmo.login(username="a", password="p") == {"token": "t"}


def announce(env, texts):
    """A write function under test: posts each text once to #ops."""
    for text in texts:
        env.slack.post("#ops", text=text)


def announce_twice(env, texts):
    """The same with a careless extra write: every text is posted twice."""
    for text in texts:
        env.slack.post("#ops", text=text)
        env.slack.post("#ops", text=text)


def test_a_write_function_is_held_to_exactly_its_recorded_write_calls():
    """The assertion a library test makes about a write function: it passes when the function issues exactly
    the recorded write calls, and fails when it issues an extra write, recorded or not.
    """
    recorded = [LOGIN, POST]
    env = env_from(recorded)
    announce(env, ["deploy done"])
    assert env.issued(effect="write") == calls(recorded, effect="write")

    extra = env_from(recorded)
    # the extra write repeats a recorded call, so it is served
    announce_twice(extra, ["deploy done"])
    assert extra.issued(effect="write") != calls(recorded, effect="write")
    assert len(extra.issued(effect="write")) == 2

    unrecorded = env_from(recorded)
    with pytest.raises(ReplayMiss):  # a write nobody recorded is never answered
        announce(unrecorded, ["deploy done", "and again"])
    assert unrecorded.misses == [
        call_of("slack", "post", ["#ops"], {"text": "and again"}),
    ]


# --- effects: unknown is a class of its own (I-Q1) -----------------------------------------------------------

# dialogue actions are recorded with effect "unknown" (adapters/dialogue.py); an exported row without an effect
# reads as unknown too
SUBMIT = {
    "channel": "dialogue:user",
    "method": "reply",
    "args": ["submit"],
    "kwargs": {},
    "response": {"valid": True},
    "status": "ok",
    "kind": "dialogue",
}


def submit_nothing(env):
    """A careless write function under test: it should submit once and issues nothing."""
    return None


def test_an_effect_filter_fails_loudly_where_the_recorded_effect_is_unknown():
    rows = [SUBMIT]
    env = env_from(rows)
    submit_nothing(env)
    assert env.issued(effect="unknown") == []
    # the vacuous comparison ([] == []) cannot be made: both sides refuse to filter unknown effects
    with pytest.raises(UnknownEffect, match="dialogue:user"):
        calls(rows, effect="write")
    with pytest.raises(UnknownEffect):
        calls(rows, effect="read")
    # the unfiltered comparison over the function's own rows judges it: nothing was issued
    assert env.issued() != calls(rows)
    # unknown is selected as its own class
    assert calls(rows, effect="unknown") == calls(rows)
    assert calls([LOGIN, POST], effect="unknown") == []
    # a served call recorded unknown makes issued's filter refuse as well
    served = env_from(rows + [LOGIN])
    getattr(served, "dialogue:user").reply("submit")
    with pytest.raises(UnknownEffect):
        served.issued(effect="write")
    assert served.issued() == calls(rows)
    assert served.issued(effect="unknown") == calls(rows)
    # mixed rows: the tool channel's effects are known, the dialogue's are not, so a write filter refuses
    with pytest.raises(UnknownEffect, match="dialogue:user"):
        calls([LOGIN, POST, SUBMIT], effect="write")
    # rows that all carry an effect filter as before
    assert calls([LOGIN, POST], effect="write") == calls([POST])
    with pytest.raises(ValueError, match="effect is one of"):
        calls([LOGIN], effect="writes")
    with pytest.raises(ValueError, match="effect is one of"):
        env_from([LOGIN]).issued(effect="writes")

import pytest
from unify.memory_v2.episodes import Action
from unify.memory_v2.replay import RecordedEnv, RecordedError, ReplayMiss


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

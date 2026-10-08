import json

import pytest

from unify.memory_v2.fingerprint import Generations
from unify.memory_v2.integration.adapters.dialogue import (
    action_payload,
    cap_text,
    dialogue_actions,
    observation_fingerprint,
    observation_shape,
    observation_value,
    response_shape,
    split_action,
)
from unify.memory_v2.redact import Redactor


def _line(seq, role, content="", **extra):
    msg = {"role": role, "content": content, **extra}
    return {
        "seq": seq,
        "ts": f"2026-10-08T03:00:{seq:02d}+00:00",
        "type": "message",
        "message": msg,
    }


def _code_call(seq, call_id, code="print(1)"):
    return _line(
        seq,
        "assistant",
        "",
        tool_calls=[
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": "execute_code",
                    "arguments": json.dumps({"code": code}),
                },
            },
        ],
    )


def _crafter_obs(step, inv="wood: 1"):
    return (
        "You see grass, tree and water.\n"
        "You face tree at your front.\n"
        f"Inventory: {inv}\n"
        f"(step {step}/2000)"
    )


def test_crafter_typed_actions_pair_with_the_next_observation():
    lines = [
        _line(0, "user", _crafter_obs(1, "empty")),
        _line(1, "assistant", "A tree is in front of me, so I collect it.\ndo"),
        _line(2, "user", _crafter_obs(2), _interjection=True),
        _line(3, "assistant", "move_left"),
        _line(4, "user", _crafter_obs(3), _interjection=True),
    ]
    acts = dialogue_actions(lines, "crafter")
    assert [a.args for a in acts] == [["do"], ["move_left"]]
    assert all(a.kind == "dialogue" and a.channel == "crafter" for a in acts)
    assert all(a.method == "act" and a.kwargs == {} and a.cell == -1 for a in acts)
    assert [a.status for a in acts] == ["ok", "ok"]
    # the observation is the counterpart's message itself, as the offline Crafter import records it
    assert acts[0].response == _crafter_obs(2)
    assert response_shape(acts[0].response) == "lines=4-7;counter=y;json=none"
    assert acts[1].response == _crafter_obs(3)


def test_arc_json_submit_and_request_demos_with_feedback():
    submit = {"action": "submit", "grid": [[0, 1], [1, 0]]}
    lines = [
        _line(0, "user", "Solve this puzzle. Reply with a JSON action."),
        _line(
            1,
            "assistant",
            "I need more examples.\n" + json.dumps({"action": "request_demos"}),
        ),
        _line(2, "user", "Demonstrations:\ninput [[1]] -> output [[0]]"),
        _line(
            3,
            "assistant",
            "The rule swaps colours.\n```json\n"
            + json.dumps(submit, indent=1)
            + "\n```",
        ),
        _line(4, "user", json.dumps({"correct": False, "attempts_left": 1})),
        _line(
            5,
            "assistant",
            "Retrying.\n" + json.dumps({"action": "submit", "grid": [[1]]}),
        ),
    ]
    acts = dialogue_actions(lines, "arc")
    # the method is the object's first identifier-valued member; the rest are its arguments
    assert [(a.method, a.args, a.kwargs) for a in acts] == [
        ("request_demos", [], {}),
        ("submit", [[[0, 1], [1, 0]]], {}),
        ("submit", [[[1]]], {}),
    ]
    assert acts[0].response.startswith("Demonstrations:")
    # a counterpart message that is one JSON object is recorded structured
    assert acts[1].response == {"correct": False, "attempts_left": 1}
    assert [a.status for a in acts] == ["ok", "ok", "unrecorded"]
    assert acts[2].response is None


def test_mixed_transcript_ignores_tool_call_turns_and_non_message_lines():
    lines = [
        {"seq": 0, "type": "session_start", "session": "s"},
        {"seq": 1, "type": "system_prompt", "content": "be helpful"},
        _line(2, "user", _crafter_obs(1, "empty")),
        _code_call(3, "c1", "print(env_notes)"),
        _line(
            4,
            "tool",
            "--- stdout ---\nnotes\n",
            tool_call_id="c1",
            name="execute_code",
        ),
        # text plus a tool call is mid-turn reasoning, not a reply
        _line(
            5,
            "assistant",
            "Let me look first.",
            tool_calls=[
                {"id": "c2", "function": {"name": "execute_code", "arguments": "{}"}},
            ],
        ),
        _line(6, "tool", "", tool_call_id="c2", name="execute_code"),
        {
            "seq": 7,
            "type": "message_update",
            "message": {
                "role": "tool",
                "tool_call_id": "c2",
                "content": "--- stdout ---\nok\n",
            },
        },
        _line(8, "assistant", "noop"),
        _line(
            9,
            "user",
            "Working on it...",
            _loop_authored=True,
            _progress_msg=True,
        ),  # harness notice: skipped
        _line(10, "user", _crafter_obs(2, "empty"), _interjection=True),
        {"seq": 11, "type": "outcome", "accepted": True, "solved": True},
    ]
    acts = dialogue_actions(lines, "crafter")
    assert len(acts) == 1
    assert acts[0].args == ["noop"]
    assert acts[0].response == _crafter_obs(2, "empty")


def test_reply_followed_by_assistant_before_any_user_message_is_unrecorded():
    lines = [
        _line(0, "user", "go"),
        _line(1, "assistant", "first"),
        _line(2, "user", "keep going", _loop_authored=True),
        _line(3, "assistant", "second"),
        _line(4, "user", "obs"),
    ]
    acts = dialogue_actions(lines, "user")
    assert [(a.args[0], a.status) for a in acts] == [
        ("first", "unrecorded"),
        ("second", "ok"),
    ]


def test_outcome_line_never_becomes_an_observation():
    lines = [
        _line(0, "user", "task"),
        _line(1, "assistant", "done"),
        {
            "seq": 2,
            "type": "outcome",
            "accepted": True,
            "solved": True,
            "message": {"role": "user", "content": "x"},
        },
    ]
    (act,) = dialogue_actions(lines, "user")
    assert act.status == "unrecorded" and act.response is None


def test_observation_is_capped_keeping_head_and_tail():
    big = "row\n" * 5000 + "(step 9/2000)"
    lines = [
        _line(0, "user", "go"),
        _line(1, "assistant", "noop"),
        _line(2, "user", big),
    ]
    (act,) = dialogue_actions(lines, "crafter", max_observation_chars=500)
    obs = act.response
    assert len(obs) <= 500
    assert obs.startswith("row\n") and obs.endswith("(step 9/2000)")
    assert "elided" in obs
    assert cap_text("short", 500) == "short"


def test_observation_and_payload_are_redacted():
    key = "sk-or-v1-" + "a" * 64
    red = Redactor({"APP_PASSWORD": "hunter2-very-secret"})
    lines = [
        _line(0, "user", "go"),
        _line(
            1,
            "assistant",
            "login with\n"
            + json.dumps({"action": "login", "pw": "hunter2-very-secret"}),
        ),
        _line(2, "user", f"welcome; your key is {key}"),
    ]
    (act,) = dialogue_actions(lines, "app", redactor=red)
    dumped = json.dumps([act.args, act.kwargs, act.response])
    assert "hunter2-very-secret" not in dumped and key not in dumped
    assert act.method == "login" and act.args == ["<secret:APP_PASSWORD>"]
    assert "<redacted:key-shaped>" in act.response
    # the default redactor still removes key-shaped strings
    (act2,) = dialogue_actions(lines, "app")
    assert key not in json.dumps(act2.response)


def test_payload_is_structural_not_keyword_based():
    assert action_payload('{"a": 1} then prose') == '{"a": 1} then prose'
    assert action_payload("thinking\n\n  move_up  \n\n") == "move_up"
    assert action_payload(
        'note {"x": 1}\n{"action": "submit", "grid": [[{"k": 1}]]}',
    ) == {
        "action": "submit",
        "grid": [[{"k": 1}]],
    }
    assert action_payload('broken {"a": }') == 'broken {"a": }'
    assert (
        action_payload("[1, 2]") == "[1, 2]"
    )  # only an object counts as a trailing JSON action


def test_counterpart_comes_from_the_caller_only():
    lines = [_line(0, "user", "go"), _line(1, "assistant", "x"), _line(2, "user", "y")]
    assert dialogue_actions(lines, "scienceworld")[0].channel == "scienceworld"
    for bad in ("", "   ", None):
        with pytest.raises(ValueError):
            dialogue_actions(lines, bad)


def test_multimodal_content_parts_are_read_as_text():
    lines = [
        _line(0, "user", "go"),
        _line(
            1,
            "assistant",
            [{"type": "text", "text": "move_"}, {"type": "text", "text": "right"}],
        ),
        _line(
            2,
            "user",
            [
                {"type": "text", "text": "obs"},
                {"type": "image_url", "image_url": {"url": "x"}},
            ],
        ),
    ]
    (act,) = dialogue_actions(lines, "crafter")
    assert act.args == ["move_right"] and act.response == "obs"


def test_observation_shape_has_no_values():
    s1 = observation_shape(_crafter_obs(1, "empty"))
    s2 = observation_shape(_crafter_obs(1999, "wood: 4, stone: 2"))
    assert s1 == s2 == "lines=4-7;counter=y;json=none"
    assert (
        observation_shape('{"correct": true, "attempts_left": 2}')
        == "lines=1;counter=n;json=object:{attempts_left:int,correct:bool}"
    )
    assert (
        observation_shape('Feedback below\n{"correct": false}')
        == "lines=2-3;counter=n;json=trailing:{correct:bool}"
    )
    assert observation_shape("") == "lines=0;counter=n;json=none"
    assert "1999" not in s2 and "wood" not in s2


def test_observation_fingerprint_feeds_generations():
    def acts(obs_list):
        lines = [_line(0, "user", "go")]
        for i, obs in enumerate(obs_list):
            lines += [
                _line(2 * i + 1, "assistant", "noop"),
                _line(2 * i + 2, "user", obs),
            ]
        lines.append(_line(99, "assistant", "noop"))  # unrecorded: not fingerprinted
        return dialogue_actions(lines, "crafter")

    fp = observation_fingerprint(acts([_crafter_obs(1), _crafter_obs(2)]))
    assert fp == {
        "crafter.act": {"shapes": ["lines=4-7;counter=y;json=none"], "errors": []},
    }
    g = Generations()
    assert g.observe(fp) == set()
    assert g.observe(observation_fingerprint(acts([_crafter_obs(5)]))) == set()
    drifted = observation_fingerprint(acts(['{"obs": "grass", "step": 3}']))
    assert g.observe(drifted) == {"crafter"}


def test_deeply_nested_reply_never_raises():
    deep = "{" * 60000 + "}"
    assert (
        action_payload(deep) == deep
    )  # no trailing object: falls back to the last line
    nested = '{"a": ' * 100 + "1" + "}" * 100  # parses, but deeper than the bound
    assert action_payload("x\n" + nested) == nested
    lines = [
        _line(0, "user", "go"),
        _line(1, "assistant", deep),
        _line(2, "user", "[" * 60000 + "]"),
        _line(3, "assistant", "ok\n" + nested),
        _line(4, "user", nested),
    ]
    acts = dialogue_actions(lines, "arc", max_payload_chars=200000)
    assert [a.status for a in acts] == ["ok", "ok"]
    # too deep to keep parsed: kept as text, shaped as deep JSON
    assert acts[1].response == nested
    assert response_shape(acts[1].response) == "lines=1;counter=n;json=deep"
    assert observation_fingerprint(acts)  # shapes computed without recursion errors


def test_long_trailing_segment_is_not_parsed():
    huge = '{"action": "submit", "grid": [' + ", ".join(["0"] * 40000) + "]}"
    assert len(huge) > 65536
    assert action_payload(huge) == huge  # too long to parse: last line


@pytest.mark.parametrize("bad", ["my.game", "Crafter", "a:b", "1arc", "arc game", "-x"])
def test_counterpart_must_be_a_dot_free_channel_name(bad):
    lines = [_line(0, "user", "go"), _line(1, "assistant", "x"), _line(2, "user", "y")]
    with pytest.raises(ValueError):
        dialogue_actions(lines, bad)
    assert dialogue_actions(lines, "science_world-2")[0].channel == "science_world-2"


def test_object_payload_is_redacted_then_capped():
    grid = [[7] * 30 for _ in range(30)]
    planted = "hunter2-very-secret"
    reply = json.dumps({"action": "submit", "pw": planted, "grid": grid})
    lines = [_line(0, "user", "go"), _line(1, "assistant", reply)]
    red = Redactor({"APP_PASSWORD": planted})
    (small,) = dialogue_actions(lines, "arc", redactor=red, max_payload_chars=300)
    assert small.method == "submit" and small.kwargs == {}
    assert isinstance(small.args[0], str) and len(small.args[0]) <= 300
    assert planted not in small.args[0]
    assert small.args[0].startswith('{"grid": [[7, 7')
    (full,) = dialogue_actions(lines, "arc", redactor=red)
    assert full.method == "submit" and full.args == []
    assert full.kwargs == {"pw": "<secret:APP_PASSWORD>", "grid": grid}


def test_observation_is_kept_whole_under_the_cap_and_shaped_from_what_is_recorded():
    obs = "\n".join(f"row {i}" for i in range(40)) + "\n(step 9/2000)"
    lines = [
        _line(0, "user", "go"),
        _line(1, "assistant", "noop"),
        _line(2, "user", obs),
    ]
    (act,) = dialogue_actions(lines, "crafter")
    assert act.response == obs
    assert (
        response_shape(act.response)
        == observation_shape(obs)
        == "lines=32-63;counter=y;json=none"
    )
    assert observation_fingerprint([act]) == {
        "crafter.act": {"shapes": ["lines=32-63;counter=y;json=none"], "errors": []},
    }
    # over the adapter's bound, the head and tail are what is recorded and shaped
    (small,) = dialogue_actions(lines, "crafter", max_observation_chars=120)
    assert len(small.response) <= 120 and small.response.endswith("(step 9/2000)")
    assert observation_fingerprint([small])["crafter.act"]["shapes"] == [
        observation_shape(small.response),
    ]


def test_split_action_reads_structure_never_member_names():
    assert split_action({"action": "submit", "grid": [[1]]}) == ("submit", [[[1]]], {})
    assert split_action({"grid": [[1]], "action": "submit"}) == ("submit", [[[1]]], {})
    assert split_action({"type": "request_demos"}) == ("request_demos", [], {})
    assert split_action({"op": "move", "dx": 1, "dy": -1}) == (
        "move",
        [],
        {"dx": 1, "dy": -1},
    )
    # a free-text value is no method; the first identifier-valued member is
    assert split_action({"why": "two words", "do": "noop"}) == (
        "noop",
        ["two words"],
        {},
    )
    # no identifier-valued member, or not an object: act(payload)
    assert split_action({"grid": [[1]]}) == ("act", [{"grid": [[1]]}], {})
    assert split_action({"n": "42"}) == ("act", [{"n": "42"}], {})
    assert split_action("move_left") == ("act", ["move_left"], {})
    assert split_action({"x": "a" * 65}) == ("act", [{"x": "a" * 65}], {})


def test_observation_value_parses_only_a_whole_json_message():
    assert observation_value('{"correct": true}') == {"correct": True}
    assert observation_value(" [1, 2] ") == [1, 2]
    assert (
        observation_value('Feedback\n{"correct": true}')
        == 'Feedback\n{"correct": true}'
    )
    assert observation_value('"just a string"') == '"just a string"'
    assert observation_value("{broken") == "{broken"
    big = json.dumps({"rows": ["x" * 100] * 10})
    assert observation_value(big, cap=50) == cap_text(
        big,
        50,
    )  # over the bound: capped text


def test_reply_followed_by_a_tool_call_turn_is_unrecorded():
    lines = [
        _line(0, "user", "go"),
        _line(1, "assistant", "move_up"),
        _code_call(2, "c1"),
        _line(3, "tool", "--- stdout ---\n1\n", tool_call_id="c1", name="execute_code"),
        _line(4, "user", "obs"),
    ]
    (act,) = dialogue_actions(lines, "crafter")
    assert act.status == "unrecorded" and act.response is None

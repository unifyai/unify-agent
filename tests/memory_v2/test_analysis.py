import json
from unify.memory_v2.analysis.antiunify import antiunify
from unify.memory_v2.analysis.cells import Hole, call_sites, cells_from_transcript
from unify.memory_v2.analysis.provenance import def_use_edges, value_edges
from unify.memory_v2.analysis.slicing import backward_slice, generic_methods, prelude
from unify.memory_v2.episodes import Cell


def _tx(code, out):
    return [
        {
            "type": "message",
            "message": {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "c1",
                        "function": {
                            "name": "execute_code",
                            "arguments": json.dumps({"thought": "t", "code": code}),
                        },
                    },
                ],
            },
        },
        {
            "type": "message",
            "message": {
                "role": "tool",
                "tool_call_id": "c1",
                "name": "execute_code",
                "content": [
                    {"type": "text", "text": '{"duration_ms": 1}'},
                    {"type": "text", "text": "\n--- stdout ---\n"},
                    {"type": "text", "text": out},
                ],
            },
        },
    ]


def test_cells_from_transcript():
    cells = cells_from_transcript(_tx("print(1)", "1\n"))
    assert cells == [Cell(0, "print(1)", "1\n", None)]


def test_call_sites_literal_and_hole():
    cells = [
        Cell(
            0,
            "tok = apis.venmo.login(username='a', password=pw)\napis.venmo.pay(token=tok, amount=5)",
            "",
        ),
    ]
    s = call_sites(cells)
    assert [(x.channel, x.method) for x in s] == [("venmo", "login"), ("venmo", "pay")]
    assert s[0].kwargs["username"] == "a" and isinstance(s[0].kwargs["password"], Hole)


def test_def_use_and_slice():
    cells = [
        Cell(0, "a = 1", ""),
        Cell(1, "b = 2", ""),
        Cell(2, "c = a + 1", ""),
        Cell(3, "print(c)", ""),
    ]
    assert (0, 2) in def_use_edges(cells) and (2, 3) in def_use_edges(cells)
    assert backward_slice(cells, 3) == {0, 2, 3}


def test_value_edges_from_parsed_output():
    cells = [
        Cell(0, "print(apis.venmo.me())", "{'user_id': 'u-77'}"),
        Cell(1, "apis.venmo.friends(user_id='u-77')", ""),
    ]
    assert (0, 1, "suspected") in value_edges(cells, call_sites(cells))


def test_generic_and_prelude():
    login = lambda o: call_sites([Cell(0, "apis.venmo.login(username='a')", "")])[0]
    eps = [
        [login(0)] + call_sites([Cell(1, f"apis.venmo.job{i}(x=1)", "")])
        for i in range(6)
    ]
    gen = generic_methods(eps, share=0.2)
    assert ("venmo", "login") in gen and ("venmo", "job0") not in gen
    assert [(s.channel, s.method) for s in prelude(eps[0], gen)] == [("venmo", "login")]


def test_antiunify_lifts_differing_constants():
    a = call_sites(
        [
            Cell(
                0,
                "apis.venmo.login(username='a')\napis.venmo.pay(amount=5, to='x')",
                "",
            ),
        ],
    )
    b = call_sites(
        [
            Cell(
                0,
                "apis.venmo.login(username='a')\napis.venmo.pay(amount=7, to='y')",
                "",
            ),
        ],
    )
    t = antiunify([a, b])
    assert t.steps[0] == ("venmo", "login", {"username": "a"})
    assert (
        t.steps[1][2]["amount"] == "$p0"
        and t.steps[1][2]["to"] == "$p1"
        and t.params == ["$p0", "$p1"]
    )
    assert antiunify([a]) is None


def test_stderr_and_substring_never_links():
    tx = _tx("print(1)", "1\n")
    tx[1]["message"]["content"].append({"type": "text", "text": "\n--- stderr ---\n"})
    tx[1]["message"]["content"].append({"type": "text", "text": "boom"})
    assert cells_from_transcript(tx)[0].error == "boom"
    cells = [
        Cell(
            0,
            "print('hello world user_id u-77 here')",
            "hello world user_id u-77 here",
        ),
        Cell(1, "apis.venmo.friends(user_id='u-77')", ""),
    ]
    assert value_edges(cells, call_sites(cells)) == set()


def test_generic_share_boundary_is_inclusive():
    eps = [
        call_sites([Cell(0, f"apis.v.{'a' if i == 0 else 'b'}()", "")])
        for i in range(5)
    ]
    assert ("v", "a") in generic_methods(eps, share=0.2)


def test_call_sites_follow_evaluation_order():
    s = call_sites([Cell(0, "apis.a.f(x=apis.b.g())", "")])
    assert [(x.method, x.order) for x in s] == [("g", 0), ("f", 1)]
    s = call_sites([Cell(0, "print(apis.x.y())\napis.x.z(\n  k=apis.x.w(),\n)", "")])
    assert [x.method for x in s] == ["y", "w", "z"]


def test_dict_arguments_dropped_call_and_string_content():
    tx = [
        {
            "type": "message",
            "message": {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "d",
                        "function": {
                            "name": "execute_code",
                            "arguments": {"code": "print(2)"},
                        },
                    },
                    {
                        "id": "lost",
                        "function": {
                            "name": "execute_code",
                            "arguments": '{"code": "x=1"}',
                        },
                    },
                ],
            },
        },
        {
            "type": "message",
            "message": {
                "role": "tool",
                "tool_call_id": "d",
                "content": "meta\n--- stdout ---\n2\n",
            },
        },
    ]
    assert cells_from_transcript(tx) == [Cell(0, "print(2)", "2\n", None)]

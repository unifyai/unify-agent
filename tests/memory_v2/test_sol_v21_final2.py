"""Memory v2.1 final2 (design r2 §1, as the lead kept it on 10 Oct): identical parts credited once with their
source. Built on test_sol_v21_tools' helpers."""

from tests.memory_v2.test_sol_v21_tools import EPS, _episode, _pass

# --- §1: identical parts (RUNTIME P2) ----------------------------------------------------------------------


def test_an_identical_part_is_credited_once_with_its_source(tmp_path):
    turns = [
        _call("a", "read_episode", {"episode": "e1", "part": "request"}),
        _call(
            "b",
            "read_episode",
            {"episode": "e2", "part": "request"},
        ),  # the same bytes as e1's
        _call("d", "dismiss", {"episode": "e1", "reason": "nothing reusable"}),
    ]
    out, model = _pass(tmp_path, turns, max_calls=10)
    assert model.outputs["a"] == '"req"'
    assert model.outputs["b"].startswith("identical to e1/request (sha256 ")
    assert model.outputs["b"].endswith("already shown in full; covered")
    s = out.coverage
    assert s["identical_parts"] == {"e2/request": "e1/request"}
    assert "e2" not in s["missing"] and s["covered"] == 1


def test_the_parts_form_credits_an_identical_part_too(tmp_path):
    turns = [
        _call("a", "read_episode", {"episode": "e1", "parts": ["request"]}),
        _call("b", "read_episode", {"episode": "e2", "parts": ["request"]}),
    ]
    out, model = _pass(tmp_path, turns, max_calls=10)
    assert model.outputs["b"].startswith(
        "== request ==\nidentical to e1/request (sha256 ",
    )
    assert out.coverage["identical_parts"] == {"e2/request": "e1/request"}


def test_a_part_shown_only_in_part_earns_no_identical_credit(tmp_path):
    long = (
        "x" * 9000
    )  # over one view: e3's request is shown in part, so it is not "shown complete"
    EPS["e3"], EPS["e4"] = _episode("e3", request=(long,)), _episode(
        "e4",
        request=(long,),
    )
    try:
        turns = [
            _call("a", "read_episode", {"episode": "e3", "part": "request"}),
            _call("b", "read_episode", {"episode": "e4", "part": "request"}),
        ]
        out, model = _pass(tmp_path, turns, eids=("e3", "e4"), max_calls=10)
    finally:
        del EPS["e3"], EPS["e4"]
    assert not model.outputs["b"].startswith("identical to")
    assert "identical_parts" not in out.coverage

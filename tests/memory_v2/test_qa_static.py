"""The static stage-5 checks (memory v2.1): stand-in environments, cuts of truncated recordings, blob refs.

Pure: test sources are parsed, never run.
"""

import hashlib

from unify.memory_v2.episodes import Action
from unify.memory_v2.integration.adapters.dialogue import cap_text
from unify.memory_v2.integration.adapters.shell import MAX_TAIL_BYTES
from unify.memory_v2.integration.adapters.tool import TRUNCATED
from unify.memory_v2.qa_static import (
    ELIDED,
    asserted_strings,
    asserts_on_cut,
    blob_refs,
    cuts,
    on_cut,
    stand_ins,
    truncated,
)

ITEMS = {"env/phone:current_datetime": "apis"}
CLOCK = '{"date": "Tuesday, October 07, 2026", "time": "09:15 AM"}'
# test_gate.py's test kit: a hand-rolled fake that ignores positional arguments
KIT = b"""class _C:
    def __init__(s, t, n): s.t, s.n = t, n
    def __getattr__(s, m): return lambda **kw: s.t[(s.n, m, tuple(sorted(kw.items())))]
class _E:
    def __init__(s, t): s.t = t
    def __getattr__(s, n): return _C(s.t, n)
def env_from(rows): return _E({(c, m, tuple(sorted(k.items()))): r for c, m, k, r in rows})
"""
REPLAY_KIT = b"""from memlab.replay import RecordedEnv
from memlab.episodes import Action

def replayed(rows):
    return RecordedEnv([Action(0, c, m, [], k, r, "ok") for c, m, k, r in rows])
"""


def _lines(src, kit=None):
    return [line for line, _ in stand_ins(src.encode(), kit, ITEMS)]


# --- replay fidelity -------------------------------------------------------------------------------------


def test_recorded_env_is_accepted_in_every_spelling():
    direct = f"""
from memlab.replay import RecordedEnv
from memlab.episodes import Action
from env.phone import current_datetime

CALL = Action(0, "phone", "get_current_date_and_time", [], {{}}, {CLOCK}, "ok")

def test_clock():
    assert current_datetime(RecordedEnv([CALL]))["time"]
"""
    assert _lines(direct) == []
    via_name = """
import memlab.replay as replay
from env import phone

def test_clock():
    env = replay.RecordedEnv.from_jsonl("env/phone/tests/calls.jsonl")
    assert phone.current_datetime(env)
"""
    assert _lines(via_name) == []
    via_fixture = """
import pytest
from memlab.replay import RecordedEnv
from env.phone import current_datetime

@pytest.fixture
def apis():
    return RecordedEnv([])

def test_clock(apis):
    current_datetime(apis)
"""
    assert _lines(via_fixture) == []
    via_kit = """
from unify_memory_testkit import replayed
from env.phone import current_datetime

def test_clock():
    current_datetime(replayed([("phone", "get_current_date_and_time", {}, {})]))
"""
    assert _lines(via_kit, REPLAY_KIT) == []
    keyword = """
from memlab.replay import RecordedEnv
from env.phone import current_datetime

def test_clock():
    current_datetime(apis=RecordedEnv([]))
"""
    assert _lines(keyword) == []


def test_stand_ins_are_refused_with_their_lines():
    kit_fake = """from unify_memory_testkit import env_from
from env.phone import current_datetime

def test_clock():
    assert current_datetime(env_from([("phone", "get_current_date_and_time", {}, {})]))
"""
    assert _lines(kit_fake, KIT) == [5]
    local_class = """from env.phone import current_datetime

class FakePhone:
    def get_current_date_and_time(self):
        return {}

class Apis:
    phone = FakePhone()

def test_clock():
    apis = Apis()
    current_datetime(apis)
"""
    assert _lines(local_class) == [12]
    namespace = """from types import SimpleNamespace
from env.phone import current_datetime

def test_clock():
    apis = SimpleNamespace(phone=SimpleNamespace(get_current_date_and_time=lambda: {}))
    current_datetime(apis)
"""
    assert _lines(namespace) == [6]
    mock = """from unittest import mock
from env.phone import current_datetime

def test_clock():
    current_datetime(mock.MagicMock())
"""
    assert _lines(mock) == [5]
    lam = """from env.phone import current_datetime

def test_clock():
    current_datetime(lambda: None)
"""
    assert _lines(lam) == [4]
    fixture = """import pytest
from env.phone import current_datetime

class Fake:
    pass

@pytest.fixture
def apis():
    yield Fake()

def test_clock(apis):
    current_datetime(apis)
"""
    assert _lines(fixture) == [12]
    subclass = """from memlab.replay import RecordedEnv
from env.phone import current_datetime

class Lenient(RecordedEnv):
    def _serve(self, *a):
        return {}

def test_clock():
    current_datetime(Lenient([]))
"""
    assert _lines(subclass) == [
        9,
    ]  # a subclass the tests define is theirs, not the exact-call replay


def test_unknown_provenance_is_never_refused():
    helper_param = """import pytest
from env.phone import current_datetime

class Fake:
    pass

@pytest.fixture
def apis():
    return Fake()

def check(apis):
    return current_datetime(apis)

def test_clock(something_else):
    check(something_else)
"""
    assert (
        _lines(helper_param) == []
    )  # a helper's parameter is its caller's, even if a fixture shares its name
    other_item = """from env.phone import other
def test_x():
    other(lambda: None)
"""
    assert (
        _lines(other_item) == []
    )  # not an environment-taking item with recorded calls
    assert stand_ins(b"def broken(:\n", None, ITEMS) == []


def test_resolution_is_by_provenance_not_by_name():
    # the same names, rebound to the replay: accepted
    renamed = """from memlab.replay import RecordedEnv as FakeEnv
from env.phone import current_datetime as mock

def test_clock():
    mock(FakeEnv([]))
"""
    assert _lines(renamed) == []
    # a stand-in under a replay-sounding name: refused
    disguised = """from env.phone import current_datetime

class RecordedEnv:
    pass

def test_clock():
    current_datetime(RecordedEnv())
"""
    assert _lines(disguised) == [7]


# --- cuts of truncated recordings ------------------------------------------------------------------------


def _crafter_text():
    tiles = "".join(f"tile{i:03d} " for i in range(200))
    return f"You see:\n{tiles}\n\nYou face tree at your front.\n\nYour inventory:\nwood: 1\n"


def test_cut_markers_match_the_recorders():
    capped = cap_text(_crafter_text(), 400)
    assert ELIDED.search(capped)
    dialogue = Action(
        -1,
        "dialogue:crafter",
        "reply",
        ["noop"],
        {},
        capped,
        "ok",
        kind="dialogue",
    )
    (c,) = cuts(dialogue)
    assert c.where == "middle" and capped[c.start : c.end] == c.marker
    tool = Action(
        0,
        "shop",
        "list",
        [],
        {},
        {TRUNCATED: {"bytes": 9, "shape": "x", "preview": "abc"}},
        "ok",
    )
    (t,) = cuts(tool)
    assert t.where == "end" and t.start == t.end == 3
    tail = "partial first line\n" + "x" * MAX_TAIL_BYTES
    shell = Action(
        0,
        "shell:make",
        "run",
        ["make"],
        {},
        {"exit_code": 0, "tail": tail},
        "ok",
        kind="shell",
    )
    (s,) = cuts(shell)
    assert s.where == "start" and s.end == len("partial first line")
    whole = Action(
        0,
        "shell:make",
        "run",
        ["make"],
        {},
        {"exit_code": 0, "tail": "short\n"},
        "ok",
        kind="shell",
    )
    assert not truncated(whole) and not truncated(
        Action(0, "shop", "list", [], {}, {"items": []}, "ok"),
    )


def test_asserting_on_the_cut_is_found_and_on_kept_text_is_not():
    capped = cap_text(_crafter_text(), 400)
    act = Action(
        -1,
        "dialogue:crafter",
        "reply",
        ["noop"],
        {},
        capped,
        "ok",
        kind="dialogue",
    )
    (c,) = cuts(act)
    after = capped[
        c.end : c.end + 7
    ]  # the first characters past the elision: where the recorder cut
    before = capped[c.start - 7 : c.start]
    assert (
        capped.count(after) == 1 and capped.count(before) == 1
    )  # tile numbers make both unique
    assert on_cut(after, c) and on_cut(before, c)
    assert on_cut("middle elided", c)  # the marker itself
    assert not on_cut("You see:", c)  # recorded content far from the cut
    assert not on_cut("ti", c)  # too short to mean anything
    src = f"""from env.dialogue_crafter import parse_observation

def test_head():
    assert parse_observation(OBS)["first"] == "You see:"

def test_tail():
    assert parse_observation(OBS)["tail"].startswith({after!r})
""".encode()
    assert asserts_on_cut(src, [act]) == [7]
    assert (
        asserts_on_cut(
            src,
            [
                Action(
                    -1,
                    "dialogue:crafter",
                    "reply",
                    [],
                    {},
                    _crafter_text(),
                    "ok",
                    kind="dialogue",
                ),
            ],
        )
        == []
    )  # nothing truncated: nothing to learn
    tool = Action(
        0,
        "shop",
        "list",
        [],
        {},
        {
            TRUNCATED: {
                "bytes": 99,
                "shape": "x",
                "preview": '{"items": [{"sku": "A-100"}, {"sku": "B-20',
            },
        },
        "ok",
    )
    tsrc = b'def test_x():\n    assert last_sku(R) == "B-20"\n    assert "__truncated__" not in R\n'
    assert asserts_on_cut(tsrc, [tool]) == [2, 3]


def test_asserted_strings_reads_assert_statements_only():
    src = b'X = "not asserted"\ndef test_a():\n    assert f(1) == "yes" and g() in ("a", "b")\n'
    assert asserted_strings(src) == [(3, "yes"), (3, "a"), (3, "b")]
    assert asserted_strings(b"def (:") == []


# --- blob references -------------------------------------------------------------------------------------


def test_blob_refs_are_64_hex_tokens_in_first_seen_order():
    a = hashlib.sha256(b"a").hexdigest()
    b = hashlib.sha256(b"b").hexdigest()
    src = f'BLOB = "{a}"\nOTHER = ["{b}", "{a}"]\nNOT = "{a}0"\n'.encode()
    assert blob_refs([src, f'"{b}"'.encode()]) == [a, b]
    assert blob_refs([b"deadbeef"]) == []

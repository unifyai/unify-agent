"""Continual-ARC visits through the dialogue adapter (``UNIFY_MEMORY_V2_DIALOGUE=env``), keyless.

A recorded-shape ARC session (:mod:`tests.memory_v2.arc_transcript`: the benchmark's rendered messages,
delivered as record blocks, the agent's JSON action on its reply's last line) gives the expected dialogue
actions on ``env``; experience counts each observation once; the fingerprint follows the observation's
shape; a 30x30 grid pair is kept whole through the episode store; payloads are redacted in linear time; a
scripted Sol's item covering the actions passes the gate; and the episode content without dialogue actions
is what the base build (9deefbfd1) records.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import time
from collections import Counter
from types import SimpleNamespace

import pytest

from tests.memory_v2.arc_transcript import (
    ITEM,
    SOL_CELL,
    demos_feedback,
    grid,
    line,
    observation,
    render_grid,
    visit,
)
from tests.memory_v2.test_episodes import _ep
from tests.memory_v2.test_sol_pass import Script, _write
from unify.memory_v2.blobs import BlobStore
from unify.memory_v2.episodes import (
    Action,
    EpisodeWriter,
    env_channel,
    episode_dir,
    load_episode,
)
from unify.memory_v2.evidence import EvidenceStore
from unify.memory_v2.experience import (
    _observation,
    _text,
    experience_pieces,
    experience_tokens,
    transcript_replies,
)
from unify.memory_v2.fingerprint import Generations, fingerprint
from unify.memory_v2.gate import Gate
from unify.memory_v2.gitio import Repo
from unify.memory_v2.integration import consolidate
from unify.memory_v2.integration.adapters.dialogue import (
    DEFAULT_OBSERVATION_CAP,
    dialogue_actions,
)
from unify.memory_v2.integration.adapters.tool import action_fingerprints
from unify.memory_v2.integration.trajectory import assemble
from unify.memory_v2.redact import Redactor
from unify.memory_v2.sol_pass import PassConfig, SolPass
from unify.memory_v2.trigger import PassRequest

needs_bwrap = pytest.mark.skipif(
    shutil.which("bwrap") is None,
    reason="bubblewrap required",
)

EID = "20261008T120000-0a1b2c3d"
ENDED = "2026-10-08T12:30:00+00:00"


def _run(eid: str = EID) -> SimpleNamespace:
    return SimpleNamespace(
        episode_id=eid,
        started_at="2026-10-08T12:00:00+00:00",
        build="9deefbfd1",
        model="openai/gpt-6-luna",
        effort="low",
        pin="0" * 40,
        request="(the visit's first message)",
        observer=None,
        costs=None,
    )


def _episode(lines, *, dialogue: bool = True, eid: str = EID):
    acts = dialogue_actions(lines, "env") if dialogue else []
    return assemble(_run(eid), lines, "", ENDED, extra_actions=acts)


# --- the actions ---------------------------------------------------------------------------------------


def test_an_arc_visit_records_its_json_actions_on_env():
    v = visit()
    acts = dialogue_actions(v["lines"], "env")
    wrong, right = v["grids"]
    assert [(a.method, a.args, a.kwargs, a.status) for a in acts] == [
        ("request_demos", [], {}, "ok"),
        ("submit", [wrong], {}, "ok"),
        ("submit", [right], {}, "ok"),
        ("finish", [], {}, "unrecorded"),  # the session ends with no message after it
    ]
    assert all(
        a.kind == "dialogue"
        and a.channel == "env"
        and a.cell == -1
        and a.effect == "unknown"
        for a in acts
    )
    # the memory channel the offline import and the screen replay use: env/env/
    assert {env_channel(a.kind, a.channel) for a in acts} == {"env"}
    # each observation is the counterpart's message as the actor received it (a record block)
    assert [a.response for a in acts[:3]] == v["observations"]
    assert acts[3].response is None


def test_a_structured_counterpart_gives_the_offline_payloads():
    """A counterpart whose message is one JSON object is recorded parsed: the offline import's shape."""
    v = visit(structured=True, wrapped=False)
    acts = dialogue_actions(v["lines"], "env")
    assert [a.method for a in acts] == ["request_demos", "submit", "submit", "finish"]
    demos, wrong, right = (a.response for a in acts[:3])
    assert demos["type"] == "DemosFeedback" and demos["demo_requests_used"] == 1
    assert (wrong["type"], wrong["correct"]) == ("SubmitFeedback", False)
    assert (right["type"], right["correct"]) == ("SubmitFeedback", True)


def test_the_adapter_keys_on_structure_not_arc_words():
    """The same structure with other words and another counterpart name records the same actions."""
    v = visit()
    swapped = json.loads(
        json.dumps(v["lines"])
        .replace("request_demos", "fetch_more")
        .replace("submit", "propose")
        .replace("Continual-ARC", "a puzzle service")
        .replace("Demo pair", "Example"),
    )
    acts = dialogue_actions(swapped, "env")
    assert [(a.method, a.status) for a in acts] == [
        ("fetch_more", "ok"),
        ("propose", "ok"),
        ("propose", "ok"),
        ("finish", "unrecorded"),
    ]
    assert [a.args for a in acts] == [
        a.args for a in dialogue_actions(v["lines"], "env")
    ]


# --- experience E --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "structured,wrapped",
    [(False, True), (True, True), (True, False)],
    ids=["rendered", "json-in-blocks", "json-messages"],
)
def test_experience_counts_each_observation_once(structured, wrapped):
    v = visit(structured=structured, wrapped=wrapped)
    with_dialogue, _ = _episode(v["lines"])
    without, _ = _episode(v["lines"], dialogue=False)
    assert sum(a.kind == "dialogue" for a in with_dialogue.actions) == 4
    # every observation is a request message already counted: the actions add nothing
    assert experience_tokens(with_dialogue) == experience_tokens(without)
    pieces = experience_pieces(with_dialogue)
    for obs in v["observations"]:
        assert pieces.count(obs) == 1
    assert not any(
        isinstance(p, str) and p.startswith('{"observation"') for p in pieces
    )


def test_an_observation_that_is_not_a_request_message_still_counts():
    """Offline imports record observations with no request messages: those still count."""
    obs = {"type": "SubmitFeedback", "correct": True}
    ep = _ep(
        request=[],
        transcript=[],
        cells=[],
        actions=[
            Action(
                0,
                "env",
                "submit",
                [[[1]]],
                {},
                obs,
                "ok",
                "unknown",
                None,
                "dialogue",
            ),
        ],
    )
    assert json.dumps(
        obs,
        sort_keys=True,
        default=str,
        ensure_ascii=False,
    ) in experience_pieces(ep)


# --- the fingerprint -----------------------------------------------------------------------------------


def test_the_fingerprint_follows_the_observation_shape():
    g = Generations()
    first, _ = _episode(visit()["lines"])
    assert g.observe(first.fingerprints) == set()
    # other values in the same shape (another task, other cells): no drift
    same, _ = _episode(visit(task="task-0badf00d", seed=5)["lines"])
    assert g.observe(same.fingerprints) == set()
    # the counterpart's messages change shape (JSON where text was): drift on env
    changed, _ = _episode(visit(structured=True)["lines"])
    assert g.observe(changed.fingerprints) == {"env"}
    # the shapes are value-free and differ between the two forms
    assert "task-" not in json.dumps(first.fingerprints)
    assert (
        first.fingerprints["env.request_demos"]["shapes"]
        != changed.fingerprints["env.request_demos"]["shapes"]
    )
    assert first.fingerprints["env.request_demos"]["channel"] == "env"


# --- size: a 30x30 grid pair ---------------------------------------------------------------------------


def _write_and_load(tmp_path, ep, redactor):
    repo = Repo.init_bare(tmp_path / "episodes.git")
    blobs = BlobStore(tmp_path / "blobs")
    sha = EpisodeWriter(repo, blobs, redactor).write(ep)
    rel = episode_dir(ep)
    return repo, sha, rel, load_episode(repo, sha, rel, blobs)


def test_a_30x30_grid_pair_survives_intact(tmp_path):
    v = visit(size=(30, 30), pair_size=(30, 30))
    acts = dialogue_actions(v["lines"], "env")
    demos = acts[0].response
    assert demos == v["observations"][0]
    assert len(demos) > 4000  # what the 4,000-character cap used to cut
    assert render_grid(grid(30, 30, seed=1)) in demos  # the pair's input
    assert render_grid(grid(30, 30, seed=2)) in demos  # and its output
    assert render_grid(grid(30, 30, seed=0)) in demos  # the re-rendered test input
    assert acts[1].args == [v["grids"][0]] and len(acts[1].args[0]) == 30
    ep, redactor = assemble(_run(), v["lines"], "", ENDED, extra_actions=acts)
    _, _, _, back = _write_and_load(tmp_path, ep, redactor)
    got = [a for a in back.actions if a.kind == "dialogue"]
    assert [a.response for a in got] == [a.response for a in acts]
    assert [a.args for a in got] == [a.args for a in acts]


def test_a_message_over_the_writer_cap_is_stored_whole_as_a_blob(tmp_path):
    pairs = [(grid(30, 30, seed=k), grid(30, 30, seed=k + 1)) for k in range(6)]
    big = observation(
        "task-1",
        grid(30, 30),
        demos=1,
        pending=(demos_feedback(pairs, 1),),
    )
    assert 16384 < len(big) < DEFAULT_OBSERVATION_CAP
    lines = [
        line(0, "user", "go"),
        line(1, "assistant", '{"action": "request_demos"}'),
        line(2, "user", big),
    ]
    (act,) = dialogue_actions(lines, "env")
    assert act.response == big
    ep, redactor = assemble(_run(), lines, "", ENDED, extra_actions=[act])
    repo, sha, rel, back = _write_and_load(tmp_path, ep, redactor)
    assert "__capped__" in repo.show(sha, f"{rel}/actions.jsonl").decode()
    (got,) = [a for a in back.actions if a.kind == "dialogue"]
    assert got.response == big


# --- redaction -----------------------------------------------------------------------------------------


def test_dialogue_payloads_are_redacted_before_the_episode(tmp_path, monkeypatch):
    secret = "arc-proxy-secret-0123456789abcdef"  # pragma: allowlist secret
    key = "sk-or-v1-" + "ab" * 32  # pragma: allowlist secret
    monkeypatch.setenv("ARC_PROXY_TOKEN", secret)
    v = visit()
    lines = list(v["lines"])
    lines[3] = line(3, "user", v["observations"][0] + f"\nproxy {secret} {key}")
    lines[9] = line(
        9,
        "assistant",
        "Second try.\n"
        + json.dumps({"action": "submit", "grid": [[1]], "note": secret}),
    )
    acts = dialogue_actions(
        lines,
        "env",
        redactor=Redactor.from_environ({"ARC_PROXY_TOKEN": secret}),
    )
    dumped = json.dumps([[a.args, a.kwargs, a.response] for a in acts])
    assert secret not in dumped and key not in dumped
    assert "<secret:ARC_PROXY_TOKEN>" in acts[0].response
    assert "<redacted:key-shaped>" in acts[0].response
    assert acts[2].kwargs == {"grid": [[1]], "note": "<secret:ARC_PROXY_TOKEN>"}
    ep, redactor = assemble(_run(), lines, "", ENDED, extra_actions=acts)
    repo, sha, rel, _ = _write_and_load(tmp_path, ep, redactor)
    for name in repo.run("ls-tree", "-r", "--name-only", sha).split():
        data = repo.show(sha, name)
        assert secret.encode() not in data and key.encode() not in data, name


def test_redaction_and_pairing_are_linear():
    sizes: list[int] = []

    class Counting(Redactor):
        def text(self, s: str) -> str:
            sizes.append(len(s))
            return super().text(s)

    huge = "7 " * 2_000_000  # a 4 MB observation
    lines = [
        line(0, "user", "go"),
        line(1, "assistant", '{"action": "request_demos"}'),
        line(2, "user", huge),
    ]
    started = time.monotonic()
    (act,) = dialogue_actions(lines, "env", redactor=Counting())
    assert time.monotonic() - started < 30
    assert sizes.count(len(huge)) == 1  # redacted once, whole, before the cap
    assert len(act.response) <= DEFAULT_OBSERVATION_CAP
    many = [line(0, "user", "go")]
    for i in range(1, 20001, 2):
        many += [
            line(i, "assistant", '{"action": "request_demos"}'),
            line(i + 1, "user", "ok"),
        ]
    started = time.monotonic()
    acts = dialogue_actions(many, "env")
    assert time.monotonic() - started < 30
    assert len(acts) == 10000 and all(a.status == "ok" for a in acts)


# --- off: the base build's episode content -------------------------------------------------------------


def _base_experience_pieces(ep):
    """``experience_pieces`` as the base build (9deefbfd1) has it, verbatim."""
    pieces = [m for m in ep.request if isinstance(m, str)]
    requests = Counter(pieces)
    replies = list(getattr(ep, "replies", None) or []) or transcript_replies(
        ep.transcript,
    )
    pieces += replies
    for c in ep.cells:
        pieces += [c.code or "", c.output or "", c.error or ""]
    for a in ep.actions:
        obs = _observation(a)
        if (
            getattr(a, "kind", "tool") == "dialogue"
            and isinstance(obs, str)
            and requests[obs] > 0
        ):
            requests[obs] -= 1
            continue
        pieces.append(_text(obs))
    return [p for p in pieces if p]


def test_without_dialogue_actions_the_episode_is_the_base_builds():
    v = visit()
    ep, _ = _episode(v["lines"], dialogue=False)
    assert not any(a.kind == "dialogue" for a in ep.actions)
    base = fingerprint([a for a in ep.actions if a.kind != "tool"])
    base.update(action_fingerprints(ep.actions))
    assert ep.fingerprints == base
    assert experience_pieces(ep) == _base_experience_pieces(ep)
    # an offline-imported episode (observations that are no request message) counts as before
    offline = _ep(
        request=[],
        transcript=[
            {"type": "message", "message": {"role": "assistant", "content": "{}"}},
        ],
        cells=[],
        actions=[
            Action(
                0,
                "env",
                "request_demos",
                [],
                {},
                {"type": "DemosFeedback"},
                "ok",
                "unknown",
                None,
                "dialogue",
            ),
            Action(
                1,
                "env",
                "act",
                ["x"],
                {},
                "text obs",
                "ok",
                "unknown",
                None,
                "dialogue",
            ),
        ],
    )
    assert experience_pieces(offline) == _base_experience_pieces(offline)


# --- the gate: a scripted Sol's item covering the actions ----------------------------------------------


def _pass_world(tmp_path, episodes):
    """A memory repo, the evidence store with *episodes* indexed, and a gate looking their actions up."""
    by_id = {ep.episode_id: ep for ep in episodes}
    ev = EvidenceStore(tmp_path / "e.sqlite")
    for ep in episodes:
        ev.index_episode(ep, "1" * 40)
    mem = Repo.init_bare(tmp_path / "memory")

    def lookup(eid, i):
        ep = by_id.get(eid)
        return ep.actions[i] if ep is not None and 0 <= i < len(ep.actions) else None

    gate = Gate(mem, ev, BlobStore(tmp_path / "blobs"), action_lookup=lookup)
    return mem, ev, gate, by_id


@needs_bwrap
def test_a_scripted_pass_lands_an_env_item_covering_the_dialogue_actions(tmp_path):
    eps = [
        _episode(visit()["lines"])[0],
        _episode(
            visit(task="task-0badf00d", seed=5)["lines"],
            eid="20261008T121000-1b2c3d4e",
        )[0],
    ]
    mem, ev, gate, by_id = _pass_world(tmp_path, eps)
    script = Script([SOL_CELL], usd="0.0000005")
    sol = SolPass(
        mem,
        gate,
        ev,
        load=lambda eid: by_id[eid],
        model_turn=script,
        config=PassConfig(max_calls=10),
    )
    out = asyncio.run(sol.run(PassRequest("batched", None, sorted(by_id), False), "p1"))
    assert "COVERS 6 EPISODES 2" in script.outputs["c1"], script.outputs
    assert out.passed, out.reasons
    assert consolidate.reason_codes(out, None) == ["ok"]
    want = {
        (ep.episode_id, i)
        for ep in eps
        for i, a in enumerate(ep.actions)
        if a.kind == "dialogue" and a.status == "ok"
    }
    assert len(want) == 6 and ev.covered() == want
    tree = mem.run("ls-tree", "-r", "--name-only", "main").split()
    assert "env/env/__init__.py" in tree
    assert "env/env/tests/rec/observations.json" in tree
    assert ITEM.split(":")[1] in mem.show("main", "env/env/__init__.py").decode()


@needs_bwrap
def test_a_pass_with_an_empty_manifest_passes_and_changes_nothing(tmp_path):
    """What a zero-call fake Sol may write (pm6 §12): finish after an empty manifest."""
    eps = [_episode(visit()["lines"])[0]]
    mem, ev, gate, by_id = _pass_world(tmp_path, eps)
    parent = mem.head()
    script = Script(
        [
            _write(
                "/memory/.pass/manifest.json",
                json.dumps({"items": [], "summary": "nothing"}),
            ),
        ],
        usd="0.0000005",
    )
    sol = SolPass(
        mem,
        gate,
        ev,
        load=lambda eid: by_id[eid],
        model_turn=script,
        config=PassConfig(max_calls=10),
    )
    out = asyncio.run(sol.run(PassRequest("batched", None, [EID], False), "p0"))
    assert out.passed, out.reasons
    assert consolidate.reason_codes(out, None) == ["ok"]
    assert mem.head() == parent and ev.covered() == set()

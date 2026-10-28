"""A scripted actor episode that tries a captured procedure first (design I.4, I.5; decisions I.11.2 and H.10).

A recorded episode's fork staged a procedure, and the capture step stored it (``tests/memory/test_capture_end_to_end``).
The next request, of the same work with other numbers and another reference line, opens in library mode: its copy
of the library holds the capture under ``.memory/captures/``. The scripted actor never writes an import line: its
first cell opens the capture with ``memory.get("capture:<id>")`` and checks it with ``.fits(<its rows>)``, its second
calls it on the request, ``memory.get("capture:<id>")(<the request>)``. The
episode's use record, written as ``memory_use.json`` by the episode writer, then records:

* ``memory_bound: true`` (the worker bound ``memory`` in every cell);
* ``library_helper_calls`` with ``get`` called twice and ``match`` (the fit check) and ``run`` (the item's call)
  each once, all ``ok``;
* the capture's own row with its ``runtime`` (the procedure's calls inside the item's call, all returned);
* no library failure (``library_calls_failed``, ``library_name_errors`` and ``library_import_errors`` all 0).

Cells run in the real sandboxed worker, with tools called from code and without; the model is scripted (keyless).
"""

from __future__ import annotations

import asyncio
import dataclasses
import json

import pytest

from tests.actor.code_act.core_world import core_world  # noqa: F401 (fixture)
from tests.actor.code_act.sandbox_world import needs_bwrap, world  # noqa: F401 (fixture)
from tests.actor.code_act.test_in_code_tools_wiring import _act, _Collect, _text
from tests.memory.test_capture_end_to_end import (
    CAPTURE_STORE,
    capture_episode,
    recorded_episode,
    request_text,
)
from tests.memory.test_episodes import _ep
from tests.memory.test_layout import LIB
from tests.memory.test_library_helper import _commit
from tests.scripted_model import ScriptedModel, cell, reply
from unify.actor import in_code_tools
from unify.memory.blobs import BlobStore
from unify.memory.episodes import EpisodeWriter, episode_dir
from unify.memory.gitio import Repo
from unify.memory.integration import request as request_mod
from unify.memory.integration.paths import Paths
from unify.memory.integration.state import State
from unify.memory.redact import Redactor
from unify.settings import SETTINGS

# ── assumed names: the integrator adapts these to the build in this one block ─────────────────────────────────
#: An item's applicability check ``memory.get(id).fits(inputs)`` and its tracked call ``memory.get(id)(...)``
#: (``unify/memory/library_helper.py``), counted in the use record's ``library_helper_calls`` by these names, beside
#: the lookup ``memory.get``.
GET, MATCH, RUN = "get", "match", "run"
#: The fit cell. The check compares structure only, so the actor passes the rows it has read from the request.
MATCH_CALL = "item = memory.get({item!r})\nfit = item.fits({inputs!r})\nprint('MATCH', item.id, fit.ok)\n"
#: The call cell: the captured procedure is given the request's text and reads its input from it.
RUN_CALL = "result = memory.get({item!r})({request!r})\nprint('RUN', result)\n"
#: Design I.5.1: a capture's call raises ``memory.call`` with this id, so its use row is keyed by it.
CAPTURE_ITEM = "capture:{cid}"
#: Design I.2.6: the actor's copy of the capture store.
COPY_CAPTURES = ".memory/captures"
# ── end of assumed names ──────────────────────────────────────────────────────────────────────────────────────

NEW_ROWS = ((1, 2), (10, 20))
NEW_REQUEST = request_text([list(r) for r in NEW_ROWS], reference="ref-0a9d44")
WANT = {"sums": [3, 30]}


def _cells(cid: str) -> tuple[str, str]:
    rows = [list(r) for r in NEW_ROWS]
    item = CAPTURE_ITEM.format(cid=cid)
    return MATCH_CALL.format(item=item, inputs=rows), RUN_CALL.format(item=item, request=NEW_REQUEST)


def _tool_texts(model: ScriptedModel) -> str:
    return "\n".join(
        _text(m.get("content")) for m in model.calls[-1].messages if m.get("role") == "tool"
    ) if model.calls else ""


@needs_bwrap
@pytest.mark.timeout(600)
@pytest.mark.parametrize("in_code", ["on"])
def test_an_actor_episode_matches_and_runs_a_captured_procedure_with_no_import_line(
    in_code,
    core_world,  # noqa: F811
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY", "on")
    monkeypatch.setattr(in_code_tools, "SESSIONS", _Collect())
    paths = Paths.under(core_world["state"])
    # an earlier episode of the same work, captured at once (no writer pass has run)
    cid, store = capture_episode(paths.state_dir, recorded_episode())
    assert store == paths.state_dir / CAPTURE_STORE
    # this request opens in library mode over a library that holds no procedure: only the capture fits
    mem = Repo.init_bare(paths.memory)
    _commit(mem, LIB, "seed the library")
    run = request_mod.RequestRun("r", paths)
    run.state = State.load(paths.state)
    run.pin = mem.head()
    run._shape_rows = lambda consolidate, sha=None: {}  # no evidence store here
    run._open_library(None)
    monkeypatch.setattr(request_mod, "_CURRENT", run)
    assert (paths.checkout / COPY_CAPTURES / cid / "procedure.py").is_file()
    assert run.memory_bound is True

    match, try_it = _cells(cid)
    turn = (lambda c: ("execute_code", {"code": c})) if in_code == "on" else cell
    model = ScriptedModel(actor=[reply(calls=[turn(match)]), reply(calls=[turn(try_it)]), reply("done")])
    assert asyncio.run(asyncio.wait_for(_act(model, NEW_REQUEST), 420)) == "done", _tool_texts(model)
    shown = _tool_texts(model)
    assert "NameError" not in shown and "ImportError" not in shown, shown
    # the check ran on the capture and answered from structure alone: the capture was recorded on the request's text,
    # which has no structure to compare with the rows, so it says it does not fit (and the call below still runs it)
    assert f"MATCH capture:{cid} False" in shown, shown
    assert f"RUN {WANT!r}" in shown, shown  # the call returned the sums

    # the episode's use record, as the request's finish computes it, written and read back as memory_use.json
    ep = _ep(episode_id="e9", request=[NEW_REQUEST], transcript=[], actions=[])
    use = request_mod.memory_use(
        ep,
        run.item_ids,
        export_roots=run.export_roots,
        surface=run.surface,
        cell_status=dict(run.cell_status),
        runtime_calls=run.runtime_calls,
        cell_errors=dict(run.cell_errors),
        memory_bound=run.memory_bound,  # as the request's finish passes it (rev 2: always the library layout)
    )
    ep = dataclasses.replace(ep, memory_use=use)
    repo = Repo.init_bare(tmp_path / "episodes.git")
    EpisodeWriter(repo, BlobStore(tmp_path / "blobs"), Redactor()).write(ep)
    rec = json.loads(repo.show("main", f"{episode_dir(ep)}/memory_use.json"))
    assert "error" not in rec, rec
    assert rec["memory_bound"] is True
    helpers = rec["library_helper_calls"]
    assert helpers[GET] == {"ok": 2, "refused": 0, "error": 0}, helpers
    assert helpers[MATCH] == {"ok": 1, "refused": 0, "error": 0}, helpers
    assert helpers[RUN] == {"ok": 1, "refused": 0, "error": 0}, helpers
    item = CAPTURE_ITEM.format(cid=cid)
    assert item in rec["items"], sorted(rec["items"])
    runtime = rec["items"][item]["runtime"]
    assert runtime["ok"] >= 1 and runtime["refused"] == 0 and runtime["error"] == 0, runtime
    assert (
        rec["library_calls_failed"],
        rec["library_name_errors"],
        rec["library_import_errors"],
    ) == (0, 0, 0), rec
    # the request's own text never enters the record
    assert "ref-0a9d44" not in json.dumps(rec)

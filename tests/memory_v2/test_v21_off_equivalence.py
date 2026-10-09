"""UNIFY_MEMORY_V21 off (the default) keeps every P3 call site on v2's path; P3's surfaces use no benchmark words."""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

from tests.actor.code_act.sandbox_world import needs_bwrap, world  # noqa: F401
from tests.memory_v2.test_import_scope_v21 import _run
from tests.memory_v2.test_item_admission import _write
from tests.memory_v2.test_layout import _tree
from unify import sandbox
from unify.memory_v2 import (
    catalogue,
    layout,
    library_export,
    library_helper,
    library_index,
    reduction,
    shape_rows,
)
from unify.memory_v2.admission import cover_problem
from unify.memory_v2.blobs import BlobStore
from unify.memory_v2.episodes import Action
from unify.memory_v2.evidence import EvidenceStore
from unify.memory_v2.gate import Gate
from unify.memory_v2.gitio import Repo
from unify.memory_v2.integration import hooks, prompt
from unify.memory_v2.integration import request as request_mod
from unify.memory_v2.integration.paths import Paths
from unify.memory_v2.manifest import parse_manifest
from unify.memory_v2.sol_pass import PassConfig
from unify.settings import SETTINGS

WORDS = re.compile(r"\b(apis|venmo)\b", re.I)
PROMISES = ("recorded for the next consolidation", "memory.diff", "proposals/")


def test_v21_off_keeps_every_call_site_on_v2(tmp_path, monkeypatch):
    # admission: the channel rule is the default
    a = Action(0, "svc", "get", [], {}, {"x": 1}, "ok")
    for ch in ("svc", "other", None):
        assert cover_problem(a, ch, lambda s: False) == cover_problem(
            a,
            ch,
            lambda s: False,
            channel_rule=True,
        )
    # reduction: the default is v2's uses
    cand = _write(
        tmp_path / "c",
        {
            "env/venmo/__init__.py": "def me(x):\n    return 1\n\ndef pay(x):\n    return me(x)\n",
        },
    )
    man = parse_manifest(
        {
            "items": [
                {
                    "item": f"env/venmo:{n}",
                    "kind": "env_function",
                    "source_episodes": ["e1"],
                    "tests": [f"env/venmo/tests/test_{n}.py"],
                    "covers": [["e1", 0]],
                }
                for n in ("me", "pay")
            ],
        },
    )
    assert reduction.with_dependents(
        man,
        {"env/venmo:me": ["G2"]},
        cand,
    ) == reduction.with_dependents(
        man,
        {"env/venmo:me": ["G2"]},
        cand,
        v21=False,
    )
    assert reduction.with_dependents(man, {"env/venmo:me": ["G2"]}, cand)[0][
        "env/venmo:pay"
    ] == ["dependency"]
    # the gate's behaviour targets: v2's channel loop, no graph
    run = _run(
        tmp_path / "g",
        {"env/a/__init__.py": "def f(x):\n    return 1\n"},
        {"env/a/__init__.py": "def f(x):\n    return 2\n"},
        ["env/a/__init__.py"],
    )
    assert Gate._behaviour_targets(run, {}) == ["env/a:f"] and run.graph is None
    # every new flag defaults off
    gate = Gate(
        Repo.init_bare(tmp_path / "m"),
        EvidenceStore(tmp_path / "e.sqlite"),
        BlobStore(tmp_path / "b"),
    )
    assert (
        gate.v21 is None and PassConfig().v21 is False
    )  # Amendment B: no V21Config is v2
    assert request_mod.RequestRun("r", Paths.under(tmp_path / "h")).v21 is False
    # the hooks: a run without v21 binds v2's export read-write, nothing read-only
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY_V2", "on")
    paths = Paths.under(tmp_path / "home")
    paths.checkout.mkdir(parents=True)
    monkeypatch.setattr(request_mod, "_CURRENT", SimpleNamespace(index="", paths=paths))
    assert (
        hooks.worker_mounts() == [paths.checkout]
        and hooks.worker_readonly_mounts() == []
    )
    # catalogue: write_generated still writes v2's four files, byte for byte
    env = _write(
        tmp_path / "cat",
        {"env/x/__init__.py": 'def f(a):\n    """F."""\n    return a\n'},
    )
    assert catalogue.write_generated(env) == catalogue.generated(env)
    assert set(catalogue.generated(env)) == {
        "README.md",
        "memory.py",
        ".memory/catalog.json",
        ".memory/shapes.py",
    }
    # shape rows: v2's ids by default; the v2.1 tree adds none to them
    assert (
        set(shape_rows._functions(env)) == {"env/x:f"}
        and shape_rows._functions(env, v21=True) == {}
    )


@needs_bwrap
def test_wrap_argv_without_late_mounts_is_unchanged(world):  # noqa: F811
    policy = sandbox.build_policy(fresh=True)
    assert sandbox.wrap_argv(["true"], policy) == sandbox.wrap_argv(
        ["true"],
        policy,
        late_readonly=(),
    )


def test_no_benchmark_vocabulary_or_v2_promises_in_what_v21_shows(tmp_path):
    root = _tree(tmp_path / "lib")
    texts = {
        rel: data.decode("utf-8")
        for rel, data in library_export.generated_v21(root).items()
    }
    catalogue.write_files(root, library_export.generated_v21(root))
    texts["section"] = prompt.render_memory_v21(root)[0]
    texts["guide"] = prompt.GUIDE_V21
    for mod in (layout, library_index, library_helper, library_export):
        texts[mod.__name__] = Path(mod.__file__).read_text(encoding="utf-8")
    for name, text in texts.items():
        assert not WORDS.search(text), name
        for promise in PROMISES:
            assert promise not in text, (name, promise)

"""D42 (spec §4.1): covers without channel equality; D28's behaviour check and reduction by the import graph."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from tests.memory_v2.integration.test_consolidate import (
    FakeSol,
    _record,
    _settings,
    _stores,
)
from tests.memory_v2.test_item_admission import _write
from tests.memory_v2.test_layout import LIB
from unify.memory_v2 import layout, reduction
from unify.memory_v2.admission import cover_problem
from unify.memory_v2.episodes import Action
from unify.memory_v2.gate import CHECKS, Gate, GateResult, _Probe, _Run
from unify.memory_v2.gate_v21 import V21Config
from unify.memory_v2.integration import consolidate
from unify.memory_v2.integration.state import State
from unify.memory_v2.manifest import Manifest, ManifestItem

TOKENS = "memory.text.parse:tokens"


def _run(
    tmp_path: Path,
    parent: dict,
    cand: dict,
    changed: list[str],
    man: Manifest | None = None,
) -> _Run:
    run = _Run(
        GateResult(True, {c: True for c in CHECKS}),
        man or Manifest(),
        {},
        "p" * 40,
        "c" * 40,
        tmp_path,
    )
    run.p_tree, run.c_tree = _write(tmp_path / "p", parent), _write(
        tmp_path / "c",
        cand,
    )
    run.p_files = {k: ("100644", "x") for k in parent}
    run.c_files = {k: ("100644", "x") for k in cand}
    run.changed = list(changed)
    return run


def _fn_item(item: str, tests: list[str] | None = None) -> ManifestItem:
    return ManifestItem(
        item,
        "function",
        layout.item_path(item),
        ["e1"],
        tests or [],
        [],
    )


def test_a_cover_on_any_channel_is_admitted_under_v21():
    a = Action(0, "svc", "get", [], {}, {"x": 1}, "ok")
    assert cover_problem(a, "other", lambda s: False) == "a tool action on svc"
    assert cover_problem(a, "other", lambda s: False, channel_rule=False) is None
    bad = Action(0, "svc", "get", [], {}, None, "ok")
    assert cover_problem(bad, None, lambda s: False, channel_rule=False) == (
        "not a recorded successful call or rejection (tool)"
    )


def test_g2_drops_the_channel_rule_only_under_v21(tmp_path):
    action = Action(0, "svc", "get", [], {}, {"x": 1}, "ok")
    item = ManifestItem(
        "env/files:read",
        "env_function",
        "env/files/__init__.py",
        ["e1"],
        [],
        [("e1", 0)],
    )
    for v21, expect in (
        (False, ["G2: env/files:read covers (e1,0), a tool action on svc"]),
        (True, []),
    ):
        gate = Gate.__new__(Gate)
        gate.v21 = V21Config() if v21 else None  # Amendment B: a config object
        gate.blobs = SimpleNamespace(has=lambda s: False)
        run = _Run(
            GateResult(True, {c: True for c in CHECKS}),
            Manifest(items=[item]),
            {},
            "p" * 40,
            "c" * 40,
            tmp_path,
        )
        gate._g2(run, preview=True, lookup=lambda e, i: action)
        assert run.res.reasons == expect


def test_behaviour_targets_keep_v2s_channel_loop(tmp_path):
    parent = {
        "env/a/__init__.py": "def f(x):\n    return 1\n\ndef g(x):\n    return 2\n\nh = f\ni = g\n",
        "env/b/__init__.py": "def k(x):\n    return 3\n",
    }
    cand = {
        "env/a/__init__.py": "def f(x):\n    return 10\n\ndef g(x):\n    return 2\n\nh = f\ni = g\n",
        "env/b/__init__.py": "def k(x):\n    return 3\n",
    }
    run = _run(tmp_path, parent, cand, ["env/a/__init__.py"])
    recorded = {"env/a:h": {("e1", 0)}}
    assert Gate._behaviour_targets(run, recorded) == ["env/a:f", "env/a:g", "env/a:h"]
    assert run.graph is None


def test_behaviour_scope_follows_imports_transitively(tmp_path):
    cand = dict(LIB)
    cand["memory/text/parse.py"] = LIB["memory/text/parse.py"].replace(
        "return s.split()",
        "return s.split(' ')",
    )
    run = _run(tmp_path, LIB, cand, ["memory/text/parse.py"])
    assert Gate._behaviour_targets(run, {}, v21=True) == [
        "memory.text.dates:parse_date",  # imports parse by name
        "memory.text.dates:week",
        "memory.text.parse:tokens",
        "memory.text.parse:words",
        "memory.text.report:summary",  # imports dates, which imports parse
        "memory.web.fetch:get",  # imports the package, whose init imports parse (relative import)
    ]
    assert run.graph == layout.import_graph(tmp_path / "c")
    only = _run(
        tmp_path / "r",
        LIB,
        LIB,
        ["memory/text/report.py", "memory/text/tests/test_dates.py"],
    )
    assert Gate._behaviour_targets(only, {}, v21=True) == ["memory.text.report:summary"]


def test_the_owner_is_the_one_edited_function_the_item_imports(tmp_path):
    def run_with(edited, deleted=()):
        man = Manifest(items=[_fn_item(i) for i in edited], deleted=list(deleted))
        run = _run(tmp_path / str(len(edited)) / str(len(deleted)), LIB, LIB, [], man)
        run.graph = layout.import_graph(run.c_tree)
        for i in edited:
            run.p_bodies[i], run.c_bodies[i] = ("old", "def", True), (
                "new",
                "def",
                True,
            )
        return run

    pr = _Probe(item="memory.text.report:summary", strict=False)
    run = run_with([TOKENS])
    assert Gate._behaviour_owner(run, "memory.text.report:summary", pr) == TOKENS
    assert Gate._behaviour_owner(run, "memory.web.fetch:get", pr) == TOKENS
    two = run_with([TOKENS, "memory.text.dates:week"])
    assert (
        Gate._behaviour_owner(two, "memory.text.report:summary", pr) is None
    )  # which one is not known
    assert (
        Gate._behaviour_owner(two, "memory.web.fetch:get", pr) == TOKENS
    )  # fetch does not import dates
    gone = run_with([TOKENS], deleted=["memory.text.parse:old"])
    assert Gate._behaviour_owner(gone, "memory.text.report:summary", pr) is None


def test_source_channels_of_covers_stand_for_v2s_channel():
    acts = [
        ("e", 0, Action(0, "svc", "get", [], {}, None, "ok")),
        (
            "e",
            1,
            Action(0, "shell:uv", "run", [], {}, {"tail": ""}, "ok", kind="shell"),
        ),
        (
            "e",
            2,
            Action(0, "worktree:workspace", "write", [], {}, {}, "ok", kind="worktree"),
        ),
    ]
    assert Gate._source_channels(acts) == ["env/svc", "env/worktree_workspace"]


def test_reduction_drops_importers_under_v21(tmp_path):
    files = dict(LIB)
    files["memory/text/__init__.py"] = (
        '"""Text helpers."""\n'  # no re-export: the package import runs nothing
    )
    files["memory/web/fetch.py"] = (
        "from memory.text.dates import week\n\n\ndef get(u):\n    return week(u)\n"
    )
    files["memory/text/tests/test_words.py"] = "from memory.text.parse import words\n"
    cand = _write(tmp_path / "c", files)
    man = Manifest(
        items=[
            _fn_item(TOKENS),
            _fn_item("memory.text.parse:words", ["memory/text/tests/test_words.py"]),
            _fn_item("memory.text.dates:parse_date"),
            _fn_item("memory.text.report:summary"),
            _fn_item("memory.web.fetch:get"),
        ],
    )
    refused, reasons = reduction.with_dependents(man, {TOKENS: ["G2"]}, cand, v21=True)
    assert refused == {
        TOKENS: ["G2"],
        "memory.text.dates:parse_date": [
            "dependency",
        ],  # its module imports tokens by name
        "memory.text.report:summary": [
            "dependency",
        ],  # it imports dates whole, and parse_date goes
    }  # words (same module, no call, its test imports only itself) and get (imports week) stay
    assert len(reasons) == 2
    v2, _ = reduction.with_dependents(
        man,
        {TOKENS: ["G2"]},
        cand,
    )  # v2: no import rule for these ids
    assert v2 == {TOKENS: ["G2"]}
    # with LIB's package init, which re-exports tokens: an importer goes when the name it takes through the
    # package is tokens (fetch), never merely because the init runs (words' test) (Amendment A)
    lib = _write(
        tmp_path / "lib",
        {
            **LIB,
            "memory/text/tests/test_words.py": files["memory/text/tests/test_words.py"],
        },
    )
    everything, _ = reduction.with_dependents(man, {TOKENS: ["G2"]}, lib, v21=True)
    assert set(everything) == {it.item for it in man.items} - {
        "memory.text.parse:words",
    }


def test_a_package_reexport_does_not_chain_its_other_modules(tmp_path):
    """P3 Amendment A item 1: two modules re-exported by a package init; refusing an item of one refuses the
    importers that use its name, not those that use only the other module's names."""
    files = {
        "memory/kit/__init__.py": '"""Kit."""\nfrom .a import fa\nfrom .b import fb\n',
        "memory/kit/a.py": 'def fa(x):\n    """A."""\n    return x\n',
        "memory/kit/b.py": 'def fb(x):\n    """B."""\n    return x\n',
        "memory/use/x.py": 'from memory.kit import fb\n\n\ndef gx(v):\n    """X."""\n    return fb(v)\n',
        "memory/use/y.py": 'from memory.kit import fa\n\n\ndef gy(v):\n    """Y."""\n    return fa(v)\n',
    }
    cand = _write(tmp_path / "c", files)
    ids = ["memory.kit.a:fa", "memory.kit.b:fb", "memory.use.x:gx", "memory.use.y:gy"]
    man = Manifest(items=[_fn_item(i) for i in ids])
    refused, _ = reduction.with_dependents(
        man,
        {"memory.kit.a:fa": ["G3"]},
        cand,
        v21=True,
    )
    assert refused == {"memory.kit.a:fa": ["G3"], "memory.use.y:gy": ["dependency"]}


def test_the_switch_reaches_the_gate(tmp_path, monkeypatch):
    made: list = []

    class Recording(consolidate.Gate):
        def __init__(self, *a, **k):
            made.append(k.get("v21"))
            super().__init__(*a, **k)

    monkeypatch.setattr(consolidate, "Gate", Recording)
    monkeypatch.setattr(consolidate, "unillm_turn", FakeSol())
    for value, seen in (("", False), ("on", True)):
        stores = _stores(tmp_path / (value or "off"))
        sha, _ = _record(stores, "e1")
        settings = _settings()
        settings.UNIFY_MEMORY_V21 = value
        asyncio.run(
            consolidate.run_due_passes(
                stores,
                "e1",
                sha,
                State(stores.paths.state),
                effort="low",
                settings=settings,
                emit=None,
            ),
        )
        assert (
            made[-1] is not None
        ) is seen  # Amendment B: a V21Config when on, None (v2) when off
        # P4: consolidate builds it with gate_v21.config_for, the layout and the checks on
        assert made[-1] is None or (
            isinstance(made[-1], V21Config) and made[-1].layout and made[-1].checks
        )


def test_an_init_change_puts_its_packages_modules_in_scope(tmp_path):
    """RUNTIME's review B1: importing memory.text.parse runs memory/text/__init__.py first, so an __init__.py that
    rebinds a name changes the package's own modules."""
    cand = dict(LIB)
    cand["memory/text/__init__.py"] = (
        LIB["memory/text/__init__.py"] + "\nparse.tokens = lambda s: []\n"
    )
    run = _run(tmp_path, LIB, cand, ["memory/text/__init__.py"])
    assert "memory.text.parse:tokens" in Gate._behaviour_targets(run, {}, v21=True)
    assert "memory.text" in layout.import_graph(tmp_path / "c")["memory.text.parse"]

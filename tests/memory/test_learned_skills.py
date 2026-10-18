"""Learned skills: the families' gradients and fits, the stored state, the consolidation step and its fixture gate,
the actor's ``memory.learned`` call, the audit record of a fit, the export, the writer's gate check and the
consolidation after an episode (unify/memory/learned.py, unify/memory/learned_store.py).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from unify.memory import learned as L
from unify.memory import learned_store as LS
from unify.memory.catalogue import write_files
from unify.memory.gitio import Repo

needs_bwrap = pytest.mark.skipif(
    shutil.which("bwrap") is None,
    reason="bubblewrap is not installed",
)


# --- families ------------------------------------------------------------------------------------------------------


def _objective(monkeypatch, fam: L.Family, params: dict, pairs: list):
    """The (start vector, loss-and-gradient) a family's fit hands the optimiser."""
    got: dict = {}

    def fake(vec, lg, done, **kw):
        got.update(vec=vec.copy(), lg=lg)
        return vec, 0, lg(vec)[0]

    monkeypatch.setattr(L, "adam", fake)
    fam.fit(params, pairs, 10, 1.0)
    return got["vec"], got["lg"]


def _grad_check(vec0: np.ndarray, lg, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    vec = vec0 + rng.normal(scale=0.3, size=vec0.shape)
    _, g = lg(vec)
    eps = 1e-6
    for i in rng.choice(len(vec), size=min(25, len(vec)), replace=False):
        d = np.zeros_like(vec)
        d[i] = eps
        fd = (lg(vec + d)[0] - lg(vec - d)[0]) / (2 * eps)
        assert abs(fd - g[i]) < 1e-5 * max(1.0, abs(fd)), (i, fd, g[i])


def test_lookup_gradient_matches_finite_differences(monkeypatch):
    fam = L.family("lookup", {})
    pairs = [("a", "b"), ("a", "c"), (("t", 1), "b"), (3, None)]
    _grad_check(*_objective(monkeypatch, fam, fam.initial({"identity": 1.0}), pairs))


def test_affine_gradient_matches_finite_differences(monkeypatch):
    fam = L.family("affine", {"inputs": 3, "outputs": 2})
    pairs = [([1, 2, 3], [1, 0]), ([0, -1, 2], [3, 4]), ([2, 2, 2], [0.5, -1])]
    _grad_check(*_objective(monkeypatch, fam, fam.initial({}), pairs))


def test_neighbourhood_gradient_matches_finite_differences(monkeypatch):
    fam = L.family("neighbourhood", {"values": ["x", "y", "z"], "radius": 1})
    pairs = [([["x", "y"], ["z", "x"]], [["y", "y"], ["x", "z"]])]
    _grad_check(*_objective(monkeypatch, fam, fam.initial({"identity": 1.0}), pairs))


def _skill(
    family: str,
    options: dict,
    initial: dict | None = None,
    **kw,
) -> L.LearnedSkill:
    return L.LearnedSkill(
        "memory.p.m:f",
        L.initial_state(family, options, initial or {}),
        audit=False,
        **kw,
    )


def test_lookup_fits_arbitrary_categories_and_is_a_plain_callable():
    pairs = [("gold", "tier-1"), (("eu", 2), "tier-2"), (7, None), (True, "flag")]
    fitted = _skill("lookup", {}).fit(pairs)
    assert fitted.reproduced and fitted.steps > 0
    assert [fitted(x) for x, _ in pairs] == [y for _, y in pairs]
    assert callable(fitted) and type(fitted).__name__ == "function"
    assert fitted("never-seen") in [
        y for _, y in pairs
    ]  # an unseen input starts from the shared bias
    with pytest.raises(L.LearnedError):
        _skill("lookup", {}).family.predictor(_skill("lookup", {}).state["params"])(
            "x",
        )  # nothing stored


def test_lookup_bias_gives_an_unseen_input_the_most_supported_output():
    fitted = _skill("lookup", {}).fit([("a", "x"), ("b", "x"), ("c", "y"), ("d", "x")])
    assert fitted.reproduced and fitted("new") == "x"


def test_lookup_accepts_a_dict_and_elementwise_nested_lists():
    assert _skill("lookup", {}).fit({"a": 1, "b": 2})("b") == 2
    sk = _skill("lookup", {"elementwise": True}, {"identity": 2.0})
    fitted = sk.fit([([[1, 3], [3, 0]], [[1, 7], [7, 0]])])
    assert fitted.reproduced
    assert fitted([[3, 5]]) == [[7, 5]]  # 5 was never seen: the identity bonus keeps it


def test_affine_fits_a_linear_map():
    pairs = [([1, 2], [5]), ([0, 1], [2]), ([3, 0], [3]), ([1, 1], [3])]
    fitted = _skill("affine", {"inputs": 2, "outputs": 1}).fit(pairs, max_steps=3000)
    assert fitted.reproduced
    assert abs(fitted([2, 2])[0] - 6.0) < 0.2


def test_neighbourhood_fits_a_local_rule_and_refuses_non_2d_data():
    x = [[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]]
    y = [[2, 2, 2, 0], [2, 1, 2, 0], [2, 2, 2, 0]]
    sk = _skill("neighbourhood", {"classes": 3}, {"identity": 2.0})
    fitted = sk.fit([(x, y)])
    assert fitted.reproduced and fitted(x) == y
    with pytest.raises(L.LearnedError):
        sk.fit([([1, 2], [1, 2])])


def test_fit_never_changes_the_stored_state_and_bounds_steps():
    sk = _skill("affine", {"inputs": 1, "outputs": 1})
    before = L.state_text(sk.state)
    fitted = sk.fit([([1], [100]), ([2], [200])], max_steps=5)
    assert fitted.steps <= 5 and not fitted.reproduced
    assert L.state_text(sk.state) == before


def test_infer_config_and_training_pairs_hooks_shape_the_fit():
    def infer(examples, stored):
        # the most common output value is the background; the inputs with another output are the marked ones
        vals = [y for _, y in examples]
        bg = max(sorted(set(vals)), key=vals.count)
        return {"background": bg, "marked": sorted(x for x, y in examples if y != bg)}

    def pairs(x, y, config):
        return [(x, y)] if y != config["background"] else []

    def skeleton(x, model, config):
        return model(x) if x in config["marked"] else config["background"]

    sk = _skill(
        "lookup",
        {},
        infer_config=infer,
        training_pairs=pairs,
        skeleton=skeleton,
    )
    fitted = sk.fit([("a", "bg"), ("b", "bg"), ("c", "x")])
    assert fitted.config == {"background": "bg", "marked": ["c"]}
    assert fitted("c") == "x" and fitted("zzz") == "bg" and fitted.reproduced


def test_fit_without_examples_uses_the_stored_fixture_cases():
    sk = _skill(
        "lookup",
        {},
        fixtures=[{"examples": [["a", "b"]], "queries": [["a", "b"]]}],
    )
    assert sk.fit()("a") == "b"
    with pytest.raises(L.LearnedError):
        _skill("lookup", {}).fit()


# --- state, consolidation, gate ------------------------------------------------------------------------------------


def test_state_json_and_npz_round_trip(tmp_path):
    sk = _skill("lookup", {"elementwise": True}, {"table": [[3, 7]]})
    state = sk.fit([([3, 1], [7, 1])]).state
    assert json.loads(L.state_text(state)) == state
    L.save_npz(state, str(tmp_path / "s.npz"))
    assert L.load_npz(str(tmp_path / "s.npz")) == state
    assert L.compatible(state, "lookup", {"elementwise": True})
    assert not L.compatible(state, "lookup", {})


def test_consolidate_moves_the_parameters_toward_the_task_parameters():
    sk = _skill("affine", {"inputs": 1, "outputs": 1})
    task = sk.fit([([1], [2]), ([2], [4])], max_steps=2000).state
    new = L.consolidated(sk.state, task, weight=1.0, rate=0.5)
    w0, wt, w1 = (s["params"]["weights"][0][0] for s in (sk.state, task, new))
    assert abs(w1 - (w0 + 0.5 * (wt - w0))) < 1e-12
    half = L.consolidated(sk.state, task, weight=0.3, rate=0.5)
    assert abs(half["params"]["weights"][0][0] - (w0 + 0.15 * (wt - w0))) < 1e-12
    with pytest.raises(L.LearnedError):
        L.consolidated(sk.state, _skill("lookup", {}).state, 1.0, 0.5)


def test_consolidation_lowers_the_steps_to_fit_on_repeat_visits():
    sk = _skill("lookup", {"elementwise": True}, {"identity": 2.0})
    visit = [([[1, 3], [3, 0]], [[1, 7], [7, 0]])]
    first = sk.fit(visit)
    state = L.consolidated(sk.state, first.state, 1.0, 1.0)
    again = L.LearnedSkill("memory.p.m:f", state, audit=False).fit(visit)
    assert again.reproduced and again.steps < first.steps


def test_discrete_config_is_kept_by_its_outcome_weighted_record():
    base = _skill("lookup", {}).state
    a = dict(base, config={"marker": 3})
    b = dict(base, config={"marker": 5})
    s = L.consolidated(base, a, 1.0, 0.5)
    assert s["config"] == {"marker": 3}
    s = L.consolidated(s, b, 0.3, 0.5)
    assert s["config"] == {"marker": 3}
    s = L.consolidated(s, b, 1.0, 0.5)
    assert s["config"] == {"marker": 5}


def test_stored_values_carry_their_episode_and_time_and_are_labelled_hints():
    sk = _skill("lookup", {})
    first = sk.fit([("res-1", "economy"), ("res-2", "business")])
    when = {"episode": "ep-1", "at": "2026-10-18T00:00:00Z"}
    stored = L.consolidated(sk.state, first.state, 1.0, 1.0, when)
    assert stored["provenance"][L.category_key("res-1")] == when
    later = L.LearnedSkill("memory.p.m:f", stored, audit=False).fit(
        [("res-3", "economy")],
    )
    assert (
        later("res-2") == "business"
    )  # the shared bias does not override a stored row
    w = later.where("res-2")
    assert w["source"] == "stored" and w["episode"] == "ep-1" and w["at"] == when["at"]
    assert "hint" in w["note"] and "never ground truth" in w["note"]
    assert later.where("res-3") == {"source": "your examples"}
    assert later.where("res-9")["source"] == "prior"
    aff = _skill("affine", {"inputs": 1, "outputs": 1})
    task = aff.fit([([1], [2])], max_steps=500).state
    s2 = L.consolidated(aff.state, task, 1.0, 0.5, {"episode": "ep-2", "at": "t"})
    again = L.LearnedSkill("memory.p.m:f", s2, audit=False).fit(
        [([1], [2])],
        max_steps=500,
    )
    w2 = again.where([3])
    assert (
        w2["last_consolidated"] == {"episode": "ep-2", "at": "t"}
        and "hint" in w2["note"]
    )


def test_gate_rejects_a_regression_and_a_rising_loss_and_no_fixtures():
    ok = {"passed": True, "loss": 0.1, "steps": 2}
    bad = {"passed": False, "loss": 0.5, "steps": 300}
    assert L.gate_states([ok], [ok]) == (True, "ok")
    assert L.gate_states([ok], [bad]) == (False, "fixture_regressed")
    assert L.gate_states([bad], [dict(bad, loss=0.9)]) == (False, "loss_rose")
    assert L.gate_states([bad], [dict(bad, loss=0.2)]) == (True, "ok")
    assert L.gate_states([], []) == (False, "no_fixtures")


def test_options_are_bounded_before_any_list_is_built():
    import time

    t0 = time.monotonic()
    for opts in (
        {"classes": 10**12},
        {"classes": 0},
        {"classes": 2.0},
        {"classes": True},
        {"values": "x" * 10},
        {"values": []},
        {"classes": 3, "radius": 1.5},
    ):
        with pytest.raises(L.LearnedError):
            L.family("neighbourhood", opts).initial({})
    for opts in (
        {"inputs": 10**12, "outputs": 1},
        {"inputs": float("inf"), "outputs": 1},
        {"inputs": "2", "outputs": 1},
        {"inputs": 1000, "outputs": 1000},
    ):
        with pytest.raises(L.LearnedError):
            L.family("affine", opts).initial({})
    with pytest.raises(L.LearnedError):
        L.declaration(
            'LEARNED_FAMILY = "neighbourhood"\nLEARNED_OPTIONS = {"classes": 10**12}\n',
        )
    assert time.monotonic() - t0 < 5.0
    assert (
        LS.learned_modules(
            {
                "memory/p/m.py": b'LEARNED_FAMILY = "neighbourhood"\nLEARNED_OPTIONS = {"classes": 999999999999}\n',
            },
        )
        == {}
    )


def test_declaration_reads_literals_only():
    src = 'LEARNED_FAMILY = "lookup"\nLEARNED_OPTIONS = {"elementwise": True}\n\ndef f(x, model, config):\n    return model(x)\n'
    assert L.declaration(src) == {
        "family": "lookup",
        "options": {"elementwise": True},
        "initial": {},
        "skeleton": "f",
    }
    assert L.declaration("def f():\n    pass\n") is None
    with pytest.raises(L.LearnedError):
        L.declaration('LEARNED_FAMILY = "lookup"\nLEARNED_OPTIONS = dict(a=1)\n')
    with pytest.raises(L.LearnedError):
        L.declaration('LEARNED_FAMILY = "spline"\n')


# --- a library with one learned skill ------------------------------------------------------------------------------

MODULE = '''"""Map each value of a nested list through a learned table."""

LEARNED_FAMILY = "lookup"
LEARNED_OPTIONS = {"elementwise": True}
LEARNED_INITIAL = {"identity": 2.0}


def map_values(x, model, config):
    """Learned skill: memory.learned("memory.tables.recode:map_values").fit(pairs)(x)."""
    return model(x)
'''
FIXTURES = {
    "cases": [
        {
            "examples": [[["a", "b", "c"], ["A", "b", "c"]]],
            "queries": [[["a", "c"], ["A", "c"]]],
        },
    ],
}
TEST = "def test_skeleton():\n    from memory.tables.recode import map_values\n    assert map_values(1, lambda v: v, {}) == 1\n"


def _library(tmp_path: Path) -> Repo:
    mem = Repo.init_bare(tmp_path / "memory")
    with mem.temp_checkout() as wt:
        files = {
            "memory/tables/__init__.py": '"""Tables."""\n',
            "memory/tables/recode.py": MODULE,
            "memory/tables/tests/test_recode.py": TEST,
            "memory/tables/tests/data/recode.learned.json": json.dumps(FIXTURES),
        }
        for rel, text in files.items():
            (wt / rel).parent.mkdir(parents=True, exist_ok=True)
            (wt / rel).write_text(text)
        sha = mem.commit_all(wt, "a learned skill", {})
    mem.fast_forward("main", sha, expected_old=mem.head())
    return mem


def test_learned_modules_and_paths():
    mods = LS.learned_modules(
        {
            "memory/tables/recode.py": MODULE.encode(),
            "memory/tables/other.py": b"x = 1\n",
        },
    )
    assert (
        list(mods) == ["memory.tables.recode"]
        and mods["memory.tables.recode"]["skeleton"] == "map_values"
    )
    assert (
        LS.fixture_path("memory.tables.recode")
        == "memory/tables/tests/data/recode.learned.json"
    )
    assert LS.module_path("memory.tables.recode") == "memory/tables/recode.py"


def test_state_ref_commits_and_reads_back(tmp_path):
    mem = _library(tmp_path)
    main = mem.head()
    state = L.initial_state("lookup", {"elementwise": True}, {"identity": 2.0})
    first = LS.write_state(mem, "memory.tables.recode", state, "learned: one", None)
    assert LS.read_states(mem) == {"memory.tables.recode": state}
    second = LS.write_state(
        mem,
        "memory.tables.recode",
        dict(state, config={"k": 1}),
        "learned: two",
        first,
    )
    assert LS.ref_head(mem) == second and LS.read_states(mem, first) == {
        "memory.tables.recode": state,
    }
    assert mem.head() == main  # main never moves
    with pytest.raises(Exception):
        LS.write_state(
            mem,
            "memory.tables.recode",
            state,
            "stale",
            first,
        )  # the ref moved past *expected*


def _actor_copy(tmp_path: Path, mem: Repo) -> Path:
    from unify.memory import library_export as lx

    root = tmp_path / "copy"
    with mem.temp_checkout() as wt:
        shutil.copytree(wt, root, ignore=shutil.ignore_patterns(".git", "tests"))
        files = {
            "memory/tables/recode.py": (wt / "memory/tables/recode.py").read_bytes(),
        }
        files[LS.fixture_path("memory.tables.recode")] = (
            wt / LS.fixture_path("memory.tables.recode")
        ).read_bytes()
    write_files(
        root,
        {**lx.generated_library_mode(root), **LS.export_files(mem, files)},
    )
    return root


def test_export_gives_the_families_the_state_and_the_fixtures(tmp_path):
    mem = _library(tmp_path)
    files = {"memory/tables/recode.py": MODULE.encode()}
    files[LS.fixture_path("memory.tables.recode")] = json.dumps(FIXTURES).encode()
    out = LS.export_files(mem, files)
    assert out["memory/_learned.py"] == Path(L.__file__).read_bytes()
    assert ".memory/learned/memory/tables/recode.fixtures.json" in out
    assert (
        ".memory/learned/memory/tables/recode.state.json" not in out
    )  # nothing consolidated yet
    LS.write_state(
        mem,
        "memory.tables.recode",
        L.initial_state("lookup", {"elementwise": True}, {}),
        "s",
        None,
    )
    assert ".memory/learned/memory/tables/recode.state.json" in LS.export_files(
        mem,
        files,
    )
    LS.write_state(
        mem,
        "memory.tables.recode",
        L.initial_state("lookup", {}, {}),
        "s",
        LS.ref_head(mem),
    )
    assert ".memory/learned/memory/tables/recode.state.json" not in LS.export_files(
        mem,
        files,
    )  # incompatible


def test_actor_calls_memory_learned_and_the_fit_is_audited(tmp_path):
    root = _actor_copy(tmp_path, _library(tmp_path))
    code = (
        "import sys, json\n"
        f"sys.path.insert(0, {str(root)!r})\n"
        "seen = []\n"
        "sys.addaudithook(lambda e, a: seen.append(a) if e == 'memory.learned_fit' else None)\n"
        "import memory\n"
        "skill = memory.learned('memory.tables.recode:map_values')\n"
        "fitted = skill.fit([(['x', 'y'], ['X', 'y'])])\n"
        "stored = memory.learned('memory.tables.recode').fit()\n"
        "print(json.dumps([fitted(['x', 'q']), fitted.reproduced, stored(['a']), seen[0][0], seen[0][2],"
        " json.loads(seen[0][1])['family'], len(seen)]))\n"
    )
    out = subprocess.run(
        [sys.executable, "-I", "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert out.returncode == 0, out.stderr
    got = json.loads(out.stdout.strip().splitlines()[-1])
    assert got == [
        ["X", "q"],
        True,
        ["A"],
        "memory.tables.recode:map_values",
        True,
        "lookup",
        2,
    ]


def test_memory_learned_off_says_so(tmp_path):
    from unify.memory import library_export as lx

    root = tmp_path / "lib"
    (root / "memory/tables").mkdir(parents=True)
    (root / "memory/tables/recode.py").write_text(MODULE)
    write_files(root, lx.generated_library_mode(root))
    code = f"import sys\nsys.path.insert(0, {str(root)!r})\nimport memory\nmemory.learned('memory.tables.recode')\n"
    out = subprocess.run(
        [sys.executable, "-I", "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert out.returncode != 0 and "learned skills are off" in out.stderr


def test_audit_hook_keeps_fits_apart_and_bounded():
    from unify.memory.integration.adapters.audit import MAX_LEARNED_FITS, CellAudit

    audit = CellAudit([])
    audit.begin()
    audit("memory.learned_fit", ("memory.p.m:f", '{"a": 1}', True, 3))
    audit("memory.learned_fit", ("not an id", "{}", True, 3))
    audit("memory.learned_fit", ("memory.p.m:f", "{}", "yes", 3))
    for _ in range(MAX_LEARNED_FITS + 2):
        audit("memory.learned_fit", ("memory.p.m", "{}", False, 0))
    out = audit.drain()["calls"]
    assert out["learned"][0] == {
        "item": "memory.p.m:f",
        "state": '{"a": 1}',
        "reproduced": True,
        "steps": 3,
    }
    assert (
        len(out["learned"]) == MAX_LEARNED_FITS
        and out["learned_dropped"] == 3
        and out["malformed"] == 2
    )
    fits: list = []
    LS.take_fits(fits, {"calls": out})
    assert len(fits) == MAX_LEARNED_FITS and fits[0]["item"] == "memory.p.m:f"
    LS.take_fits(fits, {"calls": {"learned": [{"item": 1}]}})
    assert len(fits) == MAX_LEARNED_FITS


# --- confined: the writer's gate check and the consolidation after an episode -------------------------------------


@needs_bwrap
def test_a_key_shaped_state_is_refused_by_the_hook_and_by_the_store():
    from unify.memory.integration.adapters.audit import CellAudit

    key = "sk-or-v1-" + "a" * 64
    audit = CellAudit([])
    audit.begin()
    audit(
        "memory.learned_fit",
        ("memory.p.m:f", json.dumps({"config": {"k": key}}), True, 1),
    )
    audit("memory.learned_fit", ("memory.p.m:f", '{"ok": 1}', True, 1))
    out = audit.drain()["calls"]
    assert [r["state"] for r in out["learned"]] == ['{"ok": 1}'] and out[
        "malformed"
    ] == 1
    fits: list = []
    LS.take_fits(
        fits,
        {
            "calls": {
                "learned": [
                    {
                        "item": "memory.p.m:f",
                        "state": key,
                        "reproduced": True,
                        "steps": 1,
                    },
                    {
                        "item": "memory.p.m:f",
                        "state": "{}",
                        "reproduced": True,
                        "steps": 1,
                    },
                ],
            },
        },
    )
    assert [f["state"] for f in fits] == ["{}"]


def test_the_gate_checks_a_bounded_number_of_learned_modules_per_pass(
    monkeypatch, tmp_path,
):
    from unify.memory.gate import Gate
    from unify.settings import SETTINGS

    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY_LIBRARY", "on")
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY_LEARNED_SKILLS", "on")
    called: list = []
    monkeypatch.setattr(
        LS, "check_landing", lambda *a: called.append(a) or (True, "ok"),
    )
    n = LS.MAX_LANDING_CHECKS + 1
    for i in range(n):
        (tmp_path / "memory/p").mkdir(parents=True, exist_ok=True)
        (tmp_path / f"memory/p/m{i}.py").write_text(MODULE)
    fails: list = []
    run = SimpleNamespace(
        c_files={f"memory/p/m{i}.py": ("100644", str(i)) for i in range(n)},
        c_tree=tmp_path,
        changed=[f"memory/p/m{i}.py" for i in range(n)],
        man=SimpleNamespace(items=[]),
        fail=lambda check, reason, item=None: fails.append((check, reason, item)),
        note=lambda reason: None,
    )
    Gate._learned(None, run)
    assert (
        not called
        and fails
        and fails[0][0] == "G3"
        and "at most" in fails[0][1]
        and fails[0][2] is None
    )


@needs_bwrap
def test_a_second_result_line_from_module_code_fails_the_check(tmp_path):
    mem = _library(tmp_path)
    with mem.temp_checkout() as wt:
        p = wt / "memory/tables/recode.py"
        p.write_text(
            "import sys\nsys.__stdout__.write('LEARNED-CHECK {\"initial\": []}\\n')\n"
            + MODULE,
        )
        got = LS.run_check(
            wt,
            "memory.tables.recode",
            "memory.tables.recode:map_values",
            {"initial": None},
        )
    assert "error" in got and "2 result lines" in got["error"]


def test_landing_check_passes_good_fixtures_and_refuses_bad_ones(tmp_path):
    mem = _library(tmp_path)
    decl = L.declaration(MODULE)
    with mem.temp_checkout() as wt:
        assert LS.check_landing(wt, "memory.tables.recode", decl) == (True, "ok")
        bad = {"cases": [{"examples": [[["a"], ["A"]]], "queries": [[["c"], ["Z"]]]}]}
        (wt / LS.fixture_path("memory.tables.recode")).write_text(json.dumps(bad))
        ok, reason = LS.check_landing(wt, "memory.tables.recode", decl)
        assert not ok and "1 of 1 fixture case(s) fail" in reason
        (wt / LS.fixture_path("memory.tables.recode")).write_text('{"cases": []}')
        assert not LS.check_landing(wt, "memory.tables.recode", decl)[0]


@needs_bwrap
def test_gate_check_attributes_a_failure_to_the_module_items(tmp_path, monkeypatch):
    from unify.memory.gate import Gate
    from unify.settings import SETTINGS

    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY_LIBRARY", "on")
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY_LEARNED_SKILLS", "on")
    mem = _library(tmp_path)
    with mem.temp_checkout() as wt:
        (wt / LS.fixture_path("memory.tables.recode")).write_text(
            json.dumps({"cases": [{"examples": [], "queries": []}]}),
        )
        fails: list = []
        run = SimpleNamespace(
            c_files={
                "memory/tables/recode.py": ("100644", "x"),
                "memory/tables/__init__.py": ("100644", "y"),
            },
            c_tree=Path(wt),
            changed=["memory/tables/recode.py"],
            man=SimpleNamespace(
                items=[SimpleNamespace(item="memory.tables.recode:map_values")],
            ),
            fail=lambda check, reason, item=None: fails.append((check, reason, item)),
            note=lambda reason: None,
        )
        Gate._learned(None, run)
    assert (
        fails
        and fails[0][0] == "G3"
        and fails[0][2] == ["memory.tables.recode:map_values"]
    )


@needs_bwrap
def test_consolidation_accepts_a_helpful_update_and_rejects_a_harmful_one(tmp_path):
    mem = _library(tmp_path)
    state_dir = tmp_path / "state"
    skill = L.LearnedSkill(
        "memory.tables.recode:map_values",
        L.initial_state("lookup", {"elementwise": True}, {"identity": 2.0}),
        audit=False,
    )
    good = skill.fit([(["a", "z"], ["A", "Z"])])
    rows = LS.consolidate_episode(
        tmp_path / "memory",
        state_dir,
        "ep1",
        [
            {
                "item": "memory.tables.recode:map_values",
                "state": L.state_text(good.state),
                "reproduced": True,
                "steps": good.steps,
            },
        ],
        True,
    )
    assert (
        rows[0]["accepted"]
        and rows[0]["reason"] == "ok"
        and rows[0]["passed_after"] == 1
    )
    head = LS.ref_head(mem)
    assert head == rows[0]["ref_after"]
    stored = LS.read_states(mem)["memory.tables.recode"]
    assert stored["params"]["logits"]  # moved toward the fit
    assert stored["provenance"] and all(
        v == {"episode": "ep1", "at": rows[0]["at"]}
        for v in stored["provenance"].values()
    )
    harmful = json.loads(L.state_text(stored))
    harmful["params"]["logits"].setdefault(L.category_key("a"), {})[
        L.category_key("Q")
    ] = 1e4
    rows = LS.consolidate_episode(
        tmp_path / "memory",
        state_dir,
        "ep2",
        [
            {
                "item": "memory.tables.recode:map_values",
                "state": L.state_text(harmful),
                "reproduced": True,
                "steps": 1,
            },
        ],
        True,
    )
    assert not rows[0]["accepted"] and rows[0]["reason"] == "fixture_regressed"
    assert LS.ref_head(mem) == head  # the old state is kept
    lines = (state_dir / "learned" / "consolidations.jsonl").read_text().splitlines()
    assert [json.loads(x)["accepted"] for x in lines] == [True, False]
    assert len((state_dir / "learned" / "fits.jsonl").read_text().splitlines()) == 2


def test_consolidation_skips_fits_that_did_not_reproduce_or_a_changed_module(tmp_path):
    mem = _library(tmp_path)
    other = L.initial_state("lookup", {}, {})
    rows = LS.consolidate_episode(
        tmp_path / "memory",
        tmp_path / "state",
        "ep",
        [
            {
                "item": "memory.tables.recode:map_values",
                "state": L.state_text(other),
                "reproduced": False,
                "steps": 3,
            },
            {
                "item": "memory.tables.gone:f",
                "state": L.state_text(other),
                "reproduced": True,
                "steps": 3,
            },
        ],
        None,
    )
    assert [(r["module"], r["reason"]) for r in rows] == [
        ("memory.tables.gone", "module_changed"),
        ("memory.tables.recode", "no_reproduced_fit"),
    ]
    assert rows[0]["weight"] == LS.WEIGHTS[None] and LS.ref_head(mem) is None


def test_request_consolidates_only_when_the_switch_is_on(tmp_path, monkeypatch):
    from unify.memory.integration import request as rq
    from unify.settings import SETTINGS

    calls: list = []
    monkeypatch.setattr(LS, "consolidate_episode", lambda *a: calls.append(a) or [])
    run = rq.RequestRun(
        "r",
        SimpleNamespace(memory=tmp_path / "m", state_dir=tmp_path / "s"),
    )
    run.library_mode = True
    run.learned_fits = [
        {"item": "memory.p.m:f", "state": "{}", "reproduced": True, "steps": 1},
    ]
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY_LIBRARY", "on")
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY_LEARNED_SKILLS", "off")
    run._consolidate_learned("ep", lambda m: None)
    assert calls == []
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY_LEARNED_SKILLS", "on")
    run.solved = True
    run._consolidate_learned("ep", lambda m: None)
    assert calls and calls[0][2] == "ep" and calls[0][4] is True
    drained = {
        "calls": {
            "learned": [
                {"item": "memory.p.m:f", "state": "{}", "reproduced": True, "steps": 2},
            ],
        },
    }
    run.note_calls(drained)
    assert run.learned_fits[-1]["steps"] == 2


def test_switch_and_writer_brief(monkeypatch):
    from unify.memory import library_prompts as lp
    from unify.memory.integration import switch
    from unify.settings import SETTINGS

    assert (
        switch.parse_learned_skills("") == "off"
        and switch.parse_learned_skills("on") == "on"
    )
    with pytest.raises(ValueError):
        switch.parse_learned_skills("maybe")
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY_LIBRARY", "on")
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY_LEARNED_SKILLS", "off")
    off = lp.write_brief_now()
    assert "# Learned skills" not in off
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY_LEARNED_SKILLS", "on")
    on = lp.write_brief_now()
    assert on.count("# Learned skills") == 1 and on.index(
        "# Learned skills",
    ) < on.rindex("# Done")
    assert on.replace(lp.LEARNED_SECTION + "\n", "") == off

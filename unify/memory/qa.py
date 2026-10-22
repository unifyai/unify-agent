"""Stage 5 of the consolidation gate (library mode, under G3): static checks of the library's tests; nothing runs.

The gate runs no code (rev 2.1, :mod:`.gate_policy`), so stage 5 reads files only:

* **The test kit stays test-side** (:meth:`QAChecks.kit`, :mod:`.testkit`). The kit (``memlab``, the pin plugin, the
  blobs the tests name) is what the writer's tests import, in the writer's box and the actor's export alike. New or
  changed library code that uses it, tests importing what it does not provide, tests importing a module named only
  at run time, and tests that name the ``/inputs`` path are findings (``[qa:kit]``).
* **Static test checks** (:meth:`QAChecks.static`, each behind its :class:`QAConfig` switch): a test file over
  :data:`FIXTURE_MAX_BYTES` (``[qa:fixture-size]``), a test that passes an environment-taking function a stand-in of
  its own instead of the kit's exact-call replay (``[qa:replay]``, :func:`.qa_static.stand_ins`), and a test that
  asserts on text at a truncated recording's cut (``[qa:truncation]``, :func:`.qa_static.asserts_on_cut`).

:class:`QAConfig` also sets what the writer's box holds: with a switch on, the kit is staged for the writer's own
tests (:mod:`.writer_pass`), and recorded responses over :attr:`QAConfig.response_blob_bytes` are exported as blobs.
Reasons carry item ids, test paths, line numbers and byte sizes, never a recorded value; all are filed under G3 with
a ``[qa:<check>]`` tag.
"""

from __future__ import annotations

import ast
import dataclasses
import time
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable

from . import qa_static, testkit
from .episodes import Action
from .manifest import INPUT_KINDS, TESTKIT, TESTS_DIR, item_path
from .sandbox_run import SandboxResult, run_confined

SAMPLE_K = 8  # drawn inputs per function
MAX_MUTANTS = 8  # mutants per function
RUN_S = 120.0  # one stage-5 pytest run or probe
MUTANT_S = 60.0  # one mutant's test run
BUDGET_S = 900.0  # every dynamic stage-5 check of one gate run together
FIXTURE_MAX_BYTES = 64 * 1024
MIN_KILL = Decimal("0.5")
# library mode (from the test-quality study): the mutation kill share is gated per changed function at
# MUTATION_MIN (0.7 after the first replay). Guard coverage, the kill rate of dropped-guard (``drop_raise``) mutants,
# is measured and recorded only; GUARD_MIN_KILL is its planned gate. MAX_GUARD_MUTANTS bounds the
# extra dropped-guard mutants run to measure it.
MUTATION_MIN = Decimal("0.6")
MAX_GUARD_MUTANTS = 8
# Under the fixture-size switch, an exported response at least 1/RESPONSE_BLOB_SHARE of the fixture bound also
# becomes a blob (1024 bytes at the default 64 KiB bound): a test file then holds at least 64 inline copies of
# responses under the threshold, and the brief's "handful" (8 or so) takes at most 1/8 of the bound. Dialogue
# observations are capped at 4000 characters by the recorder, so a capped Crafter screen (about 4 KB, which the
# old fixed 4096-byte threshold just missed) is always a blob; a blob reference costs 66 bytes of JSON, so the
# 1.1 MB Crafter fixture, if it held ~275 screens of ~4 KB (inferred from its size), would take ~18 KB as
# references.
RESPONSE_BLOB_SHARE = 64
RESPONSE_BLOB_BYTES = FIXTURE_MAX_BYTES // RESPONSE_BLOB_SHARE
MAX_INPUT_FILE_BYTES = testkit.MAX_INPUT_FILE_BYTES
# --- configuration -------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class QAConfig:
    """Which stage-5 checks run, and their bounds. The default runs none of them."""

    fixtures: str = ""  # "", "on" or "strict"
    mutation: bool = False
    min_kill: Decimal = MIN_KILL
    determinism: bool = False
    replay: bool = False
    fixture_size: bool = False
    sample_k: int = SAMPLE_K
    max_mutants: int = MAX_MUTANTS
    run_s: float = RUN_S
    mutant_s: float = MUTANT_S
    budget_s: float = BUDGET_S
    fixture_max_bytes: int = FIXTURE_MAX_BYTES
    library_mode: bool = (
        False  # library mode: plain recorded-inputs files, recorded measures, guard coverage
    )
    guard_mutants: int = MAX_GUARD_MUTANTS
    clock: Callable[[], float] = field(default=time.monotonic, compare=False)
    runner: Callable[..., SandboxResult] = field(default=run_confined, compare=False)

    @property
    def on(self) -> bool:
        return bool(
            self.fixtures
            or self.mutation
            or self.determinism
            or self.replay
            or self.fixture_size,
        )

    @property
    def draws(self) -> bool:
        """Whether inputs are drawn and probed (the fixture check, or the mutants' equivalence)."""
        return bool(self.fixtures or self.mutation)

    @property
    def strict(self) -> bool:
        """``fixtures="strict"``: what the checks could not judge refuses the pass."""
        return self.fixtures == "strict"

    @property
    def response_blob_bytes(self) -> int:
        """The size from which the writer's export writes a response as a blob (relative to the fixture bound)."""
        return max(1, self.fixture_max_bytes // RESPONSE_BLOB_SHARE)


def library_config(base: QAConfig) -> QAConfig:
    """The stage-5 checks the library mode gates: drawn inputs ``strict``, mutation at MUTATION_MIN."""
    return dataclasses.replace(
        base,
        fixtures="strict",
        mutation=True,
        min_kill=MUTATION_MIN,
        library_mode=True,
    )


# --- drawn inputs --------------------------------------------------------------------------------------------


# --- the probe -----------------------------------------------------------------------------------------------

# --- library mode: judged from the tests' recorded calls ----------------------------------------------------

# --- the checks ----------------------------------------------------------------------------------------------


class QAChecks:
    """The stage-5 checks of one gate run (*run* is the gate's :class:`~.gate._Run`)."""

    def __init__(self, gate: Any, run: Any) -> None:
        self.gate, self.run, self.cfg = gate, run, gate.qa

    # -- shared -------------------------------------------------------------------------------------------
    def _fail(
        self,
        tag: str,
        reason: str,
        item: str | list[str] | None = None,
    ) -> None:
        """A G3 finding; *item* names the manifest item(s) it belongs to (None: the whole pass), for the gate's
        per-item admission (:meth:`.gate.Gate.merge`)."""
        self.run.fail("G3", f"[qa:{tag}] {reason}", item)

    def _owners(self, path: str) -> list[str]:
        """The manifest items listing test file *path* ([]: none, so a refusal of it is pass-wide)."""
        return [it.item for it in self.run.man.items if path in it.tests]

    def _tests(self, it: Any) -> list[str]:
        changed = set(self.run.changed)
        return sorted(t for t in it.tests if t in changed and t in self.run.c_files)

    def _form(self, it: Any) -> str | None:
        if self.cfg.library_mode:
            return None  # a library function takes any arguments; a drawn input's form is its fixture's
        declared = self.gate._doc_inputs(self.run).get(it.item, "")
        return it.input or (declared if declared in INPUT_KINDS else None)

    def _covers(
        self,
        item: str,
        lookup: Callable[[str, int], Action | None],
    ) -> list[tuple[str, int, Action]]:
        out = []
        for i, eid, idx in sorted(self.run.covers):
            if i == item:
                a = lookup(eid, idx)
                if a is not None:
                    out.append((eid, idx, a))
        return out

    def _kit(self) -> bytes | None:
        p = self.run.c_tree / TESTKIT
        return p.read_bytes() if TESTKIT in self.run.c_files else None

    # -- the kit stays a test-side tool ---------------------------------------------------------------------
    def kit(self) -> None:
        """Refuse new or changed library code that uses the kit, test imports the kit does not provide, test
        imports of a module named only at run time, and tests that name the ``/inputs`` path (each would work
        in one place a library's tests run, not all).

        Run wherever the kit is mounted (a switch on, or the library's tests use it); static, no test runs.
        """
        run = self.run
        for p in sorted(set(run.changed) & set(run.c_files)):
            if not p.endswith(".py"):
                continue
            source = (run.c_tree / p).read_bytes()
            if not testkit.is_test_side(p):
                for line in testkit.library_uses(source):
                    self._fail(
                        "kit",
                        f"{p} line {line} uses the test kit (memlab, the pin plugin or /inputs) outside a "
                        "test; the working model imports the library without it",
                    )
                continue
            for line, what in testkit.unresolved(source):
                what = what if len(what) <= 80 else what[:77] + "..."
                self._fail(
                    "kit",
                    f"{p} line {line} imports {what}, which the test kit (version "
                    f"{testkit.KIT_VERSION}) does not provide",
                    self._owners(p),
                )
            for line in testkit.dynamic_unnamed(source):
                self._fail(
                    "kit",
                    f"{p} line {line} imports a module whose name is not a constant string "
                    "(import_module, importorskip or __import__); name it with a constant, so the test "
                    "kit is staged wherever the library's tests run",
                    self._owners(p),
                )
            for line in testkit.names_inputs_path(source):
                self._fail(
                    "kit",
                    f"{p} line {line} names a path under /inputs; read recorded blobs through "
                    "memlab.inputs.blob(<id>), which works wherever the library's tests run",
                    self._owners(p),
                )

    # -- static: fixture size, replay fidelity, cuts ------------------------------------------------------
    def static(self, lookup: Callable[[str, int], Action | None]) -> None:
        run, cfg = self.run, self.cfg
        if cfg.fixture_size:
            support = set(run.man.support)
            for p in sorted(set(run.changed) & set(run.c_files)):
                if not (TESTS_DIR.match(p) or p == TESTKIT or p in support):
                    continue
                size = (run.c_tree / p).stat().st_size
                if size > cfg.fixture_max_bytes:
                    self._fail(
                        "fixture-size",
                        f"{p} has {size} bytes, over the {cfg.fixture_max_bytes}-byte bound for test "
                        "files: reference recorded payloads by blob id (memlab.inputs.blob) instead of "
                        "copying them",
                        self._owners(p),
                    )
        items = [it for it in run.man.items if it.kind == "env_function"]
        if cfg.replay:
            env_items: dict[str, str] = {}
            for it in items:
                covers = self._covers(it.item, lookup)
                tool_calls = any(
                    getattr(a, "kind", "tool") == "tool" and a.status == "ok"
                    for _, _, a in covers
                )
                form = self._form(it) or ("env" if tool_calls else None)
                first = _first_param(run.c_tree, it.item)
                if form == "env" and tool_calls and first is not None:
                    env_items[it.item] = first
            if env_items:
                kit = self._kit()
                for t in sorted({t for it in items for t in self._tests(it)}):
                    for line, item in qa_static.stand_ins(
                        (run.c_tree / t).read_bytes(),
                        kit,
                        env_items,
                    ):
                        self._fail(
                            "replay",
                            f"{t} line {line} passes {item} an environment the tests make (a class, "
                            "function or literal of their own); test it through memlab.replay.env_from "
                            "over its recorded calls",
                            item,
                        )
        if cfg.fixture_size:
            for it in items:
                actions = [a for _, _, a in self._covers(it.item, lookup)]
                if not any(qa_static.truncated(a) for a in actions):
                    continue
                for t in self._tests(it):
                    for line in qa_static.asserts_on_cut(
                        (run.c_tree / t).read_bytes(),
                        actions,
                    ):
                        self._fail(
                            "truncation",
                            f"{t} line {line} asserts on text at the recorder's cut of a truncated "
                            "recording; assert on what the environment returned, not on the cut",
                            it.item,
                        )


def _first_param(tree: Path, item: str) -> str | None:
    """The name of *item*'s first parameter in the candidate tree, or None."""
    path = tree / item_path(item)
    try:
        module = ast.parse(path.read_bytes())
    except (OSError, SyntaxError, ValueError, RecursionError):
        return None
    name = item.split(":", 1)[1]
    fn = None
    for node in module.body:
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == name
        ):
            fn = node
    if fn is None:
        return None
    params = fn.args.posonlyargs + fn.args.args
    return params[0].arg if params else None


# --- the consolidator's brief --------------------------------------------------------------------------------


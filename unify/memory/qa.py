"""Stage 5 of the consolidation gate (library mode, under G3): the test kit stays test-side; nothing runs.

The kit (:mod:`.testkit`: ``memlab``, the pin plugin, the blobs a test names) is what the writer's own tests import,
in the writer's box and the actor's export alike. :meth:`QAChecks.kit` reads the changed files (the gate runs no
code, rev 2.1): new or changed library code that uses the kit, tests importing what it does not provide, tests
importing a module named only at run time, and tests that name the ``/inputs`` path are findings (``[qa:kit]``).
Reasons carry item ids, test paths and line numbers, never a recorded value; all are filed under G3.
"""

from __future__ import annotations

from typing import Any

from . import testkit


class QAChecks:
    """The stage-5 checks of one gate run (*run* is the gate's :class:`~.gate._Run`)."""

    def __init__(self, gate: Any, run: Any) -> None:
        self.gate, self.run = gate, run

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


# --- the consolidator's brief --------------------------------------------------------------------------------


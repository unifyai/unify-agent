"""Stage 5 of the consolidation gate (library mode, under G3): no stored file names the ``/inputs`` path; nothing runs.

``/inputs`` holds the writer's input for one pass only, and the actor's box has none. :meth:`QAChecks.inputs_paths`
reads the changed files (the gate runs no code): new or changed library code or a test that names a path
under ``/inputs`` is a finding (``[qa:inputs]``). Reasons carry item ids, test paths and line numbers, never a
recorded value; all are filed under G3.
"""

from __future__ import annotations

import ast
from typing import Any

from .layout import classify

INPUTS_PATH = "/inputs"
#: The library layout's test-side kinds (:func:`.layout.classify`): ``memory/<package>/tests/`` files.
TEST_SIDE_KINDS = frozenset({"test", "test_helper", "fixture"})


def is_test_side(path: str) -> bool:
    """Whether *path* is test-side: a test, a test helper or a fixture under ``memory/<package>/tests/``."""
    return classify(path) in TEST_SIDE_KINDS


def names_inputs_path(source: bytes) -> list[int]:
    """Lines of string constants naming a path under ``/inputs``."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError):
        return []
    return sorted(
        {
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and (node.value == INPUTS_PATH or node.value.startswith(INPUTS_PATH + "/"))
        },
    )


class QAChecks:
    """The stage-5 checks of one gate run (*run* is the gate's :class:`~.gate._Run`)."""

    def __init__(self, gate: Any, run: Any) -> None:
        self.gate, self.run = gate, run

    def _owners(self, path: str) -> list[str]:
        """The manifest items listing test file *path* ([]: none, so a refusal of it is pass-wide)."""
        return [it.item for it in self.run.man.items if path in it.tests]

    def inputs_paths(self) -> None:
        """Label new or changed library code and tests that name a path under ``/inputs``, which only the pass
        that wrote them has. Run on every check and merge; static, no test runs."""
        run = self.run
        for p in sorted(set(run.changed) & set(run.c_files)):
            if not p.endswith(".py"):
                continue
            lines = names_inputs_path((run.c_tree / p).read_bytes())
            if not is_test_side(p):
                for line in lines:
                    run.fail(
                        "G3",
                        f"[qa:inputs] {p} line {line} names a path under /inputs outside a test; the working model "
                        "imports the library without /inputs",
                    )
                continue
            for line in lines:
                run.fail(
                    "G3",
                    f"[qa:inputs] {p} line {line} names a path under /inputs, which only this pass's box has; copy "
                    "the recorded bytes a test reads into the package with tools.fixture",
                    self._owners(p),
                )

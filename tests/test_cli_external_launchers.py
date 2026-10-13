"""The argv that external launchers build parses under ``unify``'s CLI.

Each fixture is copied literally from a launcher outside this repository that
starts ``unify``. These fixtures must change together with the launchers: a
flag the CLI drops or renames makes argparse exit 2 at every launch there, so
a change to the CLI's options that breaks a fixture needs the launcher changed
with it (and the fixture re-copied, with its source and commit).

Keyless: the CLI's own parser (``unify.cli._parse_args``, what ``main`` calls)
is called directly, and no actor is started.
"""

from __future__ import annotations

import pytest

from unify.cli import _parse_args

pytestmark = pytest.mark.no_unify_context

HOME = "/tmp/arc-ua-home/unify"
REQUEST = "Solve the task described in the attached file."

# continual-arc-baselines, src/continual_arc_baselines/systems/unify_agent.py:224
# (UnifyAgentLearner._argv), commit 99605e6:
#   [self._executable(), *self.command[1:], "act", "--persist", "--jsonl",
#    "--no-clarify", "--quiet", "--home", str(self.unify_home), request]
# with the default command ["unify"], so command[1:] is empty; the executable
# is the program name and not part of the parsed argv.
CONTINUAL_ARC_BASELINES_UNIFY_AGENT = [
    "act",
    "--persist",
    "--jsonl",
    "--no-clarify",
    "--quiet",
    "--home",
    HOME,
    REQUEST,
]

# The benchmark adapters under continual-harness-research/benchmarks/*/src
# (appworld, crafter, scienceworld, travelplanner) start unify through the
# same UnifyAgentLearner, so they build no argv of their own.


def test_the_paper_adapter_argv_parses():
    args = _parse_args(list(CONTINUAL_ARC_BASELINES_UNIFY_AGENT))
    assert args.command == "act"
    assert args.persist is True
    assert args.jsonl is True
    assert args.no_clarify is True
    assert args.quiet is True
    assert args.home == HOME
    assert args.request == REQUEST
    assert args.no_store is False
    assert args.timeout is None

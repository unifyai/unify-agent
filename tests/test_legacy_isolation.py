"""Symbolic: nothing on the actor or CLI path imports ``unify.legacy``.

``unify/legacy/`` holds legacy, unused, unsupported code (the conversation
manager), kept for reference only. Importing the CLI or the actor in a fresh
interpreter must leave every ``unify.legacy`` module out of ``sys.modules``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.no_unify_context

REPO_ROOT = Path(__file__).resolve().parent.parent

PROBE = """
import json, sys
import unify.cli
import unify.actor
import unify.actor.code_act_actor
from unify.manager_registry import ManagerRegistry
ManagerRegistry._ensure_populated()
print(json.dumps(sorted(m for m in sys.modules if m == "unify.legacy" or m.startswith("unify.legacy."))))
"""


@pytest.mark.timeout(300)
def test_the_cli_and_the_actor_import_nothing_from_legacy():
    env = dict(os.environ, UNIFY_VALIDATE_LLM_PROVIDERS="false")
    result = subprocess.run(
        [sys.executable, "-c", PROBE],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=240,
    )
    assert result.returncode == 0, result.stderr[-4000:]
    loaded = json.loads(result.stdout.strip().splitlines()[-1])
    assert loaded == [], f"the actor or CLI path imported legacy modules: {loaded}"


def test_the_probe_detects_a_legacy_import():
    """The check itself can fail: importing legacy shows up in sys.modules."""
    probe = (
        "import json, sys\n"
        "import unify.legacy\n"
        "print(json.dumps(sorted(m for m in sys.modules "
        "if m == 'unify.legacy' or m.startswith('unify.legacy.'))))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=REPO_ROOT,
        env=dict(os.environ, UNIFY_VALIDATE_LLM_PROVIDERS="false"),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr[-4000:]
    assert json.loads(result.stdout.strip().splitlines()[-1]) == ["unify.legacy"]

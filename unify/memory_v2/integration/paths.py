"""Where memory v2 keeps its harness-side state under ``<UNIFY_HOME>``.

Only ``memory-checkout/`` (the per-request export) is ever mounted into a cell; every other path stays behind
the sandbox's tmpfs over ``<UNIFY_HOME>`` (unify/sandbox.py ``build_policy``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    home: Path

    @classmethod
    def under(cls, home: Path) -> "Paths":
        return cls(Path(home))

    @property
    def memory(self) -> Path:
        """The bare memory repo; ``main`` holds only gated or hide commits."""
        return self.home / "memory"

    @property
    def checkout(self) -> Path:
        """The request's scratch export of memory ``main``, mounted read-write into the worker."""
        return self.home / "memory-checkout"

    @property
    def episodes(self) -> Path:
        return self.home / "episodes.git"

    @property
    def blobs(self) -> Path:
        return self.home / "episodes-blobs"

    @property
    def evidence(self) -> Path:
        return self.home / "memory-evidence.sqlite"

    @property
    def state_dir(self) -> Path:
        return self.home / "memory-v2"

    @property
    def lock(self) -> Path:
        return self.state_dir / "request.lock"

    @property
    def state(self) -> Path:
        return self.state_dir / "state.json"

    @property
    def errors(self) -> Path:
        return self.state_dir / "errors.jsonl"

    @property
    def worktree_git(self) -> Path:
        return self.home / "worktree.git"

    def harness_only(self) -> list[Path]:
        """Every path a cell must never see."""
        return [
            self.memory,
            self.episodes,
            self.blobs,
            self.evidence,
            self.state_dir,
            self.worktree_git,
        ]

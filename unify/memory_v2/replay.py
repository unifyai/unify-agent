"""Serve recorded environment responses in place of the environment (fixture replay)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .episodes import Action


class ReplayMiss(LookupError):
    pass


class RecordedError(RuntimeError):
    pass


def _key(channel: str, method: str, args: Any, kwargs: Any) -> str:
    return json.dumps(
        [channel, method, list(args), dict(sorted(kwargs.items()))],
        sort_keys=True,
        default=str,
    )


class _Channel:
    def __init__(self, env: "RecordedEnv", name: str) -> None:
        self._env, self._name = env, name

    def __getattr__(self, method: str):
        if method.startswith("_"):
            raise AttributeError(method)

        def call(*args: Any, **kwargs: Any) -> Any:
            return self._env._serve(self._name, method, args, kwargs)

        return call


class RecordedEnv:
    """``env.<channel>.<method>(*args, **kwargs)`` returns the recorded response of the identical call.

    Only calls recorded as ``ok`` or ``error`` are served; the first recording of a call wins. A call
    without a recording raises :class:`ReplayMiss`; a recorded failure raises :class:`RecordedError`.
    """

    def __init__(self, actions: list[Action]) -> None:
        self._table: dict[str, Action] = {}
        for a in actions:
            if a.status in ("ok", "error"):
                self._table.setdefault(_key(a.channel, a.method, a.args, a.kwargs), a)
        self.served: list[tuple[str, str]] = []

    @classmethod
    def from_jsonl(cls, path: Path) -> "RecordedEnv":
        rows = [
            json.loads(ln) for ln in Path(path).read_text().splitlines() if ln.strip()
        ]
        return cls([Action(**r) for r in rows])

    def __getattr__(self, channel: str) -> _Channel:
        if channel.startswith("_"):
            raise AttributeError(channel)
        return _Channel(self, channel)

    def _serve(self, channel: str, method: str, args: Any, kwargs: Any) -> Any:
        a = self._table.get(_key(channel, method, args, kwargs))
        if a is None:
            raise ReplayMiss(
                f"no recorded call {channel}.{method} with these arguments",
            )
        self.served.append((channel, method))
        if a.status == "error":
            raise RecordedError(a.error or "recorded error")
        return json.loads(json.dumps(a.response))

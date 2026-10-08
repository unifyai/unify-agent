"""Serve recorded environment responses in place of the environment (exact-call replay).

The harness's replay helper for library tests (``memlab.replay`` in the test kit, :mod:`.testkit`), so a test
never builds a fake environment of its own:

* :func:`env_from` (or :class:`RecordedEnv`) over recorded action rows: ``env.<channel>.<method>(*args,
  **kwargs)`` answers only the identical recorded call (channel, method, positional arguments and keyword
  arguments, keys sorted; values compared as their JSON). A call with no recording raises :class:`ReplayMiss`
  (it is also listed in ``env.misses``, in case the code under test swallows it); a call recorded as failed
  re-raises as :class:`RecordedError`.
* ``env.issued()`` lists the calls served, in order, so a test can assert that a function issued exactly the
  calls of its own recorded rows: ``assert env.issued() == calls(rows) and not env.misses``.

Rows are :class:`~.episodes.Action` objects or exported action dicts (unknown keys ignored; a dict without a
status counts as ``ok``). Only rows recorded ``ok`` or ``error`` are served; the first recording of a call
answers it.

Effects are ``read``, ``write`` or ``unknown`` (:data:`EFFECTS`), and ``unknown`` is a class of its own: the
dialogue adapter records every action, and the tool adapter every call that declares no effect, as
``unknown``. Filtering on ``read`` or ``write`` (``calls(rows, effect=...)``, ``env.issued(effect=...)``) raises
:class:`UnknownEffect` when a row in scope was recorded ``unknown``, since that row may be either and the
filtered comparison would pass without judging it. Standard library only (plus the kit's ``episodes``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .episodes import Action

SERVED_STATUSES = ("ok", "error")
EFFECTS = ("read", "write", "unknown")


class ReplayMiss(LookupError):
    pass


class RecordedError(RuntimeError):
    pass


class UnknownEffect(ValueError):
    """A ``read``/``write`` filter over calls whose recorded effect is ``unknown`` (it cannot be judged)."""


def _check_effect(effect: str | None, unknown_channels: Iterable[str]) -> None:
    if effect is not None and effect not in EFFECTS:
        raise ValueError(
            f"effect is one of {', '.join(EFFECTS)} (or None), not {effect!r}",
        )
    if effect in ("read", "write"):
        chans = sorted(set(unknown_channels))
        if chans:
            raise UnknownEffect(
                f"cannot keep only {effect} calls: channel(s) {', '.join(chans)} recorded calls with effect "
                "'unknown' (every dialogue action; tool calls that declare no effect), which may be either; "
                "compare the unfiltered calls instead: env.issued() == calls(<the function's own rows>)",
            )


def _key(channel: str, method: str, args: Any, kwargs: Any) -> str:
    return json.dumps(
        [channel, method, list(args), dict(sorted(kwargs.items()))],
        sort_keys=True,
        default=str,
    )


def _json(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


def call_of(channel: str, method: str, args: Any, kwargs: Any) -> dict:
    """One call as :meth:`RecordedEnv.issued` and :func:`calls` list it (arguments as their JSON)."""
    return {
        "channel": channel,
        "method": method,
        "args": _json(list(args)),
        "kwargs": _json(dict(sorted(dict(kwargs).items()))),
    }


def _action(row: Action | dict) -> Action:
    """An :class:`~.episodes.Action` from an Action or an exported action row (unknown keys ignored)."""
    if isinstance(row, Action):
        return row
    if not isinstance(row, dict):
        raise TypeError(
            f"a recorded action is an Action or a dict, not {type(row).__name__}",
        )
    cell = row.get("cell", 0)
    return Action(
        cell if isinstance(cell, int) and not isinstance(cell, bool) else 0,
        str(row.get("channel") or ""),
        str(row.get("method") or ""),
        list(row.get("args") or []),
        dict(row.get("kwargs") or {}),
        row.get("response"),
        str(row.get("status") or "ok"),
        str(row.get("effect") or "unknown"),
        row.get("error"),
        str(row.get("kind") or "tool"),
    )


def calls(rows: Iterable[Action | dict], *, effect: str | None = None) -> list[dict]:
    """The calls *rows* record that a replay serves (status ``ok`` or ``error``), in order, duplicates kept;
    *effect* keeps only the calls recorded with that effect (one of :data:`EFFECTS`). ``"read"`` or ``"write"``
    raises :class:`UnknownEffect` when a served row's recorded effect is ``unknown``.
    """
    served = [a for a in map(_action, rows) if a.status in SERVED_STATUSES]
    _check_effect(effect, (a.channel for a in served if a.effect == "unknown"))
    return [
        call_of(a.channel, a.method, a.args, a.kwargs)
        for a in served
        if effect is None or a.effect == effect
    ]


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
    ``served`` lists ``(channel, method)`` of each call served; :meth:`issued` the full calls.
    """

    def __init__(self, actions: Iterable[Action | dict]) -> None:
        self._table: dict[str, Action] = {}
        for a in map(_action, actions):
            if a.status in SERVED_STATUSES:
                self._table.setdefault(_key(a.channel, a.method, a.args, a.kwargs), a)
        self.served: list[tuple[str, str]] = []
        self.misses: list[dict] = []
        self._issued: list[tuple[dict, str]] = []  # (call, the recording's effect)

    @classmethod
    def from_jsonl(cls, path: Path) -> "RecordedEnv":
        rows = [
            json.loads(ln) for ln in Path(path).read_text().splitlines() if ln.strip()
        ]
        return cls(rows)

    def issued(self, *, effect: str | None = None) -> list[dict]:
        """The calls served so far, in order (recorded errors included, misses not); *effect*
        keeps only those whose recording has that effect (``"read"`` or ``"write"`` raises
        :class:`UnknownEffect` when a served call's recorded effect is ``unknown``)."""
        _check_effect(effect, (c["channel"] for c, e in self._issued if e == "unknown"))
        return [dict(c) for c, e in self._issued if effect is None or e == effect]

    def __getattr__(self, channel: str) -> _Channel:
        if channel.startswith("_"):
            raise AttributeError(channel)
        return _Channel(self, channel)

    def _serve(self, channel: str, method: str, args: Any, kwargs: Any) -> Any:
        a = self._table.get(_key(channel, method, args, kwargs))
        if a is None:
            self.misses.append(call_of(channel, method, args, kwargs))
            raise ReplayMiss(
                f"no recorded call {channel}.{method} with these arguments",
            )
        self.served.append((channel, method))
        self._issued.append((call_of(channel, method, args, kwargs), a.effect))
        if a.status == "error":
            raise RecordedError(a.error or "recorded error")
        return json.loads(json.dumps(a.response))


def env_from(rows: Iterable[Action | dict]) -> RecordedEnv:
    """A :class:`RecordedEnv` over recorded action rows (Actions or exported dicts): the harness's replay."""
    return RecordedEnv(rows)

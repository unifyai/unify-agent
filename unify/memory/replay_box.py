"""Inside a confined box: the episode's state and a stand-in for its tools that answers from the record only.

Standard library only, and no other module of this package is imported: the harness copies this file into a box
read-only (``/case/replay_box.py``) and the box's run script imports it, so a procedure under test runs against it with
no route to the host, the network or a real tool.

* :class:`State`: what a procedure of the ``state`` and ``state_env`` forms is given: ``.request`` (the first
  message), ``.messages`` (every user or environment message so far, in order), ``.last`` (the latest one) and
  ``.data`` (the JSON objects and arrays those messages contain, in order).
* :class:`ReplayEnv`: ``env.<channel>.<method>(*args, **kwargs)`` answers only an identical recorded call (channel,
  method, positional and keyword arguments, compared as their JSON), whatever its recorded effect (``read``,
  ``write`` or ``unknown``): nothing is executed, a write is recorded and compared, never made. Identical calls are
  answered in recorded order; past the last recording of a call its last one is repeated (a repeated read is an
  answer; a repeated write or unknown-effect call is one more effect than the record holds, so it counts as
  differing). A call with no identical recording raises :class:`ReplayMiss`; a call recorded as failed raises
  :class:`RecordedError` with the recorded error. Past *max_calls* calls, :class:`TooManyCalls`. A channel is
  reached by its recorded key or by its memory-channel name (``<kind>_<key>`` for a kind-qualified key,
  ``worktree:workspace`` as ``worktree_workspace``: the same rule as ``episodes.env_channel``).

:meth:`ReplayEnv.stats` compares what was issued with the record: ``matched`` (calls answered by a recording, repeated
reads included), ``differed`` (calls to a recorded channel and method whose arguments match no recording, and repeated
writes), ``unrecorded`` (calls to a channel and method the record never shows), ``missing`` (recorded calls never
issued) and ``repeats``.
"""

import copy
import json
import re

SERVED_STATUSES = ("ok", "error")
EFFECTS = ("read", "write", "unknown")
_NOT_NAME = re.compile(r"[^a-z0-9_]+")


class ReplayMiss(LookupError):
    """A call the record holds no identical recording of."""


class RecordedError(RuntimeError):
    """A call the record shows failing: raised again with the recorded error."""


class TooManyCalls(RuntimeError):
    """More calls than one replay may serve."""


def alias(kind, channel):
    """The memory-channel name of a recorded channel key (``episodes.env_channel``), or None."""
    if not isinstance(channel, str) or not channel:
        return None
    if kind == "tool" or ":" not in channel:
        return channel
    prefix, key = channel.split(":", 1)
    if prefix != kind:
        return None
    name = _NOT_NAME.sub("_", key.lower()).strip("_")
    return "%s_%s" % (kind, name) if name else None


def _json(value):
    return json.loads(json.dumps(value, default=str))


def _key(channel, method, args, kwargs):
    return json.dumps([channel, method, list(args), dict(sorted(dict(kwargs).items()))], sort_keys=True, default=str)


class State:
    """The episode's state as a procedure reads it (module docstring)."""

    def __init__(self, messages, data=None):
        self.messages = [m if isinstance(m, str) else json.dumps(m, default=str) for m in (messages or [])]
        self.request = self.messages[0] if self.messages else ""
        self.last = self.messages[-1] if self.messages else ""
        self.data = list(data or [])

    def __repr__(self):
        return "State(%d message%s, %d JSON value%s)" % (
            len(self.messages),
            "" if len(self.messages) == 1 else "s",
            len(self.data),
            "" if len(self.data) == 1 else "s",
        )


class _Channel:
    def __init__(self, env, name):
        self._env, self._name = env, name

    def __getattr__(self, method):
        if method.startswith("_"):
            raise AttributeError(method)

        def call(*args, **kwargs):
            return self._env._serve(self._name, method, args, kwargs)

        call.__name__ = method
        return call


class ReplayEnv:
    """``env.<channel>.<method>(...)`` answered from recorded rows only (module docstring)."""

    def __init__(self, rows, max_calls=2000):
        self._rows = []
        self._table = {}
        self._names = {}
        for row in rows or []:
            if not isinstance(row, dict) or row.get("status") not in SERVED_STATUSES:
                continue
            channel, method = str(row.get("channel") or ""), str(row.get("method") or "")
            effect = row.get("effect") if row.get("effect") in EFFECTS else "unknown"
            r = {
                "channel": channel,
                "method": method,
                "args": list(row.get("args") or []),
                "kwargs": dict(row.get("kwargs") or {}),
                "response": row.get("response"),
                "status": row["status"],
                "error": row.get("error"),
                "effect": effect,
            }
            i = len(self._rows)
            self._rows.append(r)
            self._table.setdefault(_key(channel, method, r["args"], r["kwargs"]), []).append(i)
            for name in {channel, alias(str(row.get("kind") or "tool"), channel)}:
                if name:
                    self._names.setdefault(name, set()).add(channel)
        self._next = {}
        self._used = set()
        self.max_calls = int(max_calls)
        self.issued = []

    def __getattr__(self, channel):
        if channel.startswith("_"):
            raise AttributeError(channel)
        return _Channel(self, channel)

    def _serve(self, name, method, args, kwargs):
        if len(self.issued) >= self.max_calls:
            raise TooManyCalls("more than %d calls in one replay" % self.max_calls)
        call = {"channel": name, "method": method, "args": _json(list(args)), "kwargs": _json(dict(kwargs))}
        channels = sorted(self._names.get(name, ()))
        for channel in channels:
            key = _key(channel, method, call["args"], call["kwargs"])
            found = self._table.get(key)
            if not found:
                continue
            n = self._next.get(key, 0)
            if n < len(found):
                i, outcome = found[n], "served"
                self._next[key] = n + 1
                self._used.add(i)
            else:
                i, outcome = found[-1], "repeat"
            row = self._rows[i]
            self.issued.append({**call, "outcome": outcome, "effect": row["effect"], "row": i})
            if row["status"] == "error":
                raise RecordedError(str(row.get("error") or "recorded error"))
            return copy.deepcopy(row["response"])
        known = any(r["method"] == method and r["channel"] in channels for r in self._rows)
        self.issued.append({**call, "outcome": "differed" if known else "unrecorded", "effect": None, "row": None})
        raise ReplayMiss("no recorded call %s.%s with these arguments" % (name, method))

    def stats(self):
        """What was issued against the record (module docstring)."""
        matched = differed = unrecorded = repeats = 0
        for c in self.issued:
            if c["outcome"] == "served":
                matched += 1
            elif c["outcome"] == "repeat":
                repeats += 1
                if c["effect"] == "read":
                    matched += 1
                else:
                    differed += 1
            elif c["outcome"] == "differed":
                differed += 1
            else:
                unrecorded += 1
        missing = len(self._rows) - len(self._used)
        return {
            "matched": matched,
            "differed": differed,
            "missing": missing,
            "unrecorded": unrecorded,
            "repeats": repeats,
            "issued": len(self.issued),
        }

    def calls(self):
        """The calls issued, in order (channel, method, arguments as JSON, how each was answered, its effect)."""
        return [dict(c) for c in self.issued]

    def recorded(self):
        """The recorded calls the replay serves, in order."""
        return [
            {"channel": r["channel"], "method": r["method"], "args": r["args"], "kwargs": r["kwargs"],
             "effect": r["effect"], "status": r["status"]}
            for r in self._rows
        ]  # fmt: skip

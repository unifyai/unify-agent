"""Environment fingerprints and generation numbers (spec §9, §J1, D14): event-driven drift, never a time limit.

A fingerprint maps a key to the shapes and error signatures recorded under it, per action kind:

* **tool** (unchanged): key ``<channel>.<method>``; response shapes of ``ok`` calls; error signatures;
* **shell**: key ``<channel>.<method>``; output shapes (:func:`.analysis.shellout.signature`: format,
  JSON tree, table delimiter and width) of recorded tails; ``exit:<n>`` for each nonzero exit code;
* **worktree**: key ``<channel>.<path family>`` (the directory, digit runs as ``#``, plus ``/*`` and the
  extension); file shapes (:func:`.analysis.shapes.signature`, without row or line counts) from the
  recorded ``shape``, or computed from the recorded blob when a blob store is given;
* **dialogue**: key ``<channel>.<method>``; observation shapes
  (:func:`.analysis.transitions.observation_shape`).

Entries of the three new kinds also carry ``"channel"`` (their keys may hold dots); tool entries keep the
v0 form, and :class:`Generations` reads an entry's channel from that field or the key's first part. A
channel's generation bumps when a key it has seen shows a new shape, or its first error after shapes.
"""

from __future__ import annotations

import posixpath
import re
from typing import Any

from .analysis import shapes as _shapes
from .analysis.shellout import output_shape, signature as _output_signature
from .analysis.transitions import observation_shape
from .episodes import Action


def shape(value: Any) -> str:
    if isinstance(value, dict):
        return (
            "{"
            + ",".join(f"{k}:{shape(value[k])}" for k in sorted(value, key=str))
            + "}"
        )
    if isinstance(value, (list, tuple)):
        return "[" + (shape(value[0]) if value else "") + "]"
    return type(value).__name__


def error_signature(text: str) -> str:
    first = (text or "").strip().splitlines()[0] if (text or "").strip() else ""
    return re.sub(r"\d+", "#", first)[:160]


def path_family(path: str) -> str:
    """``finance/ap/2026-10/invoices.csv`` -> ``finance/ap/#-#/*.csv``: directory plus extension."""
    d = posixpath.dirname(path or "")
    d = re.sub(r"\d+", "#", d)
    ext = _shapes.extension(path or "")
    return (d + "/" if d else "") + "*" + ext


def _file_shape(a: Action, blobs: Any) -> dict | None:
    r = a.response if isinstance(a.response, dict) else {}
    if isinstance(r.get("shape"), dict):
        return r["shape"]
    if blobs is None:
        return None
    sha = r.get("blob_after") if a.method == "write" else r.get("blob_before")
    sha = sha or r.get("blob_before") or r.get("blob_after")
    if not blobs.has(sha):
        return None
    path = a.args[0] if a.args and isinstance(a.args[0], str) else ""
    return _shapes.shape(path, blobs.get(sha))


def fingerprint(
    actions: list[Action],
    blobs: Any = None,
) -> dict[str, dict[str, list[str]]]:
    """The fingerprint of recorded actions (see the module docstring); *blobs* is an optional BlobStore."""
    out: dict[str, dict[str, Any]] = {}

    def slot(key: str, channel: str | None) -> dict[str, Any]:
        s = out.setdefault(key, {"shapes": set(), "errors": set()})
        if channel is not None:
            s["channel"] = channel
        return s

    for a in actions:
        kind = getattr(a, "kind", "tool")
        if kind == "tool":
            s = slot(f"{a.channel}.{a.method}", None)
            if a.status == "ok":
                s["shapes"].add(shape(a.response))
            elif a.status == "error":
                s["errors"].add(error_signature(a.error or ""))
        elif kind == "shell":
            s = slot(f"{a.channel}.{a.method}", a.channel)
            r = a.response
            if isinstance(r, dict) and isinstance(r.get("tail"), str):
                s["shapes"].add(_output_signature(output_shape(r["tail"])))
                code = r.get("exit_code")
                if isinstance(code, int) and not isinstance(code, bool) and code != 0:
                    s["errors"].add(f"exit:{code}")
            elif a.status == "error" and a.error:
                s["errors"].add(error_signature(a.error))
        elif kind == "worktree":
            if a.method not in ("read", "write"):
                continue
            path = a.args[0] if a.args and isinstance(a.args[0], str) else ""
            s = slot(f"{a.channel}.{path_family(path)}", a.channel)
            fs = _file_shape(a, blobs) if a.status == "ok" else None
            if fs is not None:
                s["shapes"].add(_shapes.signature(fs))
            elif a.status == "error":
                s["errors"].add(error_signature(a.error or ""))
        elif kind == "dialogue":
            s = slot(f"{a.channel}.{a.method}", a.channel)
            if a.status == "ok" and a.response is not None:
                s["shapes"].add(observation_shape(a.response))
            elif a.status == "error":
                s["errors"].add(error_signature(a.error or ""))
    result: dict[str, dict[str, Any]] = {}
    for k, v in out.items():
        row: dict[str, Any] = {
            "shapes": sorted(v["shapes"]),
            "errors": sorted(v["errors"]),
        }
        if "channel" in v:
            row["channel"] = v["channel"]
        result[k] = row
    return result


class Generations:
    def __init__(self) -> None:
        self._seen: dict[str, dict[str, set[str]]] = {}
        self._gen: dict[str, int] = {}

    def to_json(self) -> dict:
        """A JSON-safe, deterministic form (sorted lists), for the harness's state file."""
        return {
            "seen": {
                key: {"shapes": sorted(v["shapes"]), "errors": sorted(v["errors"])}
                for key, v in sorted(self._seen.items())
            },
            "generations": dict(sorted(self._gen.items())),
        }

    @classmethod
    def from_json(cls, data: dict) -> "Generations":
        g = cls()
        for key, v in (data.get("seen") or {}).items():
            g._seen[str(key)] = {
                "shapes": set(v.get("shapes") or []),
                "errors": set(v.get("errors") or []),
            }
        g._gen = {str(k): int(n) for k, n in (data.get("generations") or {}).items()}
        return g

    def generation(self, channel: str) -> int:
        return self._gen.get(channel, 0)

    def observe(self, fp: dict[str, dict[str, list[str]]]) -> set[str]:
        bumped: set[str] = set()
        for key, val in fp.items():
            channel = val.get("channel") or key.split(".", 1)[0]
            seen = self._seen.get(key)
            if seen is None:
                self._seen[key] = {
                    "shapes": set(val["shapes"]),
                    "errors": set(val["errors"]),
                }
                continue
            new_shape = bool(set(val["shapes"]) - seen["shapes"]) and bool(
                seen["shapes"],
            )
            new_error = (
                bool(set(val["errors"]) - seen["errors"])
                and not seen["errors"]
                and bool(seen["shapes"])
            )
            if new_shape or new_error:
                bumped.add(channel)
            seen["shapes"] |= set(val["shapes"])
            seen["errors"] |= set(val["errors"])
        for ch in bumped:
            self._gen[ch] = self._gen.get(ch, 0) + 1
        return bumped

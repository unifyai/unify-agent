"""Recorded inputs for memory-library tests (memory v2.1 stage 5); shipped into the boxes as ``memlab.inputs``.

A test that should hold for every recorded input of its function's family is parametrised over :func:`inputs`::

    import pytest
    from memlab.inputs import from_action, inputs
    from env.arc import feedback_state

    CASES = inputs("env/arc:feedback_state", [from_action(a, form="observation") for a in FIXTURE_ACTIONS])

    @pytest.mark.parametrize("x", CASES, ids=repr)
    def test_reads_the_recorded_attempts(x):
        assert feedback_state(x.value())[0] == x.response["attempts_used"]

In the consolidation sandbox and in the gate's ordinary runs :func:`inputs` returns the fixtures. In the gate's
seeded-sample run (``/qa/samples.json`` present) it appends the gate's sampled recorded inputs of that function,
and every sampled input a test case reads (:meth:`RecordedInput.value`, ``.response``, ``.kwargs``) is written to
``/qa-out/reads.jsonl`` under the running test's id (``PYTEST_CURRENT_TEST``), so the gate can tell which samples
the function's own tests actually exercised.

Recorded payloads are referenced by blob id instead of being copied into fixtures: :func:`blob` reads
``blobs/<id>`` beside the kit (a work-tree file's ``blob_before``/``blob_after``, or an exported
``response_blobs`` entry): ``/inputs/blobs`` in Sol's box and in the gate, ``<export>/.memlab/blobs`` in the
actor's export. A test reads a blob through :func:`blob` (never by its path), so it reads it the same way
wherever the library's tests run; :func:`from_blob` makes a file input of one.

Standard library and memlab only. Nothing here reads the clock or the network, and nothing is recorded outside
the gate's sample run.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable

from .episodes import Action
from .replay import RecordedEnv

_KIT = Path(__file__).resolve().parent
# the kit's root (``blobs/`` lives there): the directory holding the ``memlab`` package, else /inputs
INPUTS = _KIT.parent if _KIT.name == "memlab" else Path("/inputs")
SAMPLES = Path("/qa/samples.json")
READS = Path("/qa-out/reads.jsonl")
FORMS = ("env", "observation", "text", "path", "bytes")
DEFAULT_FORM = {"tool": "env", "worktree": "path", "dialogue": "observation"}
_BLOB_ID = re.compile(r"[0-9a-f]{64}\Z")
_NAME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9._-]{0,99}\Z")
_reads: set[tuple] = set()
_cache: dict[str, Any] = {}


def _copy(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


def _action(row: dict) -> Action:
    """An :class:`~.episodes.Action` from an exported or sampled action row (unknown keys ignored)."""
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


def blob_path(sha: str) -> str:
    """The path of a recorded blob in this box (``blobs/<id>`` beside the kit)."""
    if not isinstance(sha, str) or not _BLOB_ID.match(sha):
        raise ValueError("a blob id is 64 lowercase hex digits")
    return str(INPUTS / "blobs" / sha)


def blob(sha: str) -> bytes:
    """The bytes of a recorded blob, by id."""
    with open(blob_path(sha), "rb") as fh:
        return fh.read()


def _blob_of(action: dict) -> str | None:
    r = action.get("response") if isinstance(action.get("response"), dict) else {}
    sha = (
        r.get("blob_after") if action.get("method") == "write" else r.get("blob_before")
    )
    sha = sha or r.get("blob_before") or r.get("blob_after")
    return sha if isinstance(sha, str) and _BLOB_ID.match(sha) else None


class RecordedInput:
    """One recorded input of a library function: an exported action, a blob, or a gate sample.

    :meth:`value` gives it in an input form (:data:`FORMS`; default: the declared form, else the kind's
    convention); ``env`` is a :class:`~.replay.RecordedEnv` answering the recorded calls of its context
    (exact-call replay). ``response`` and ``kwargs`` are the recorded call's, ``status`` its status.
    """

    __slots__ = ("_row", "_item", "_sample")

    def __init__(
        self,
        row: dict,
        *,
        item: str | None = None,
        sample: int | None = None,
    ) -> None:
        if not isinstance(row, dict) or not isinstance(row.get("action"), dict):
            raise ValueError("a recorded input needs an action")
        self._row, self._item, self._sample = row, item, sample

    # -- what was recorded --------------------------------------------------------------------------------
    @property
    def kind(self) -> str:
        return str(self._row["action"].get("kind") or "tool")

    @property
    def form(self) -> str:
        form = self._row.get("form")
        return form if form in FORMS else DEFAULT_FORM.get(self.kind, "observation")

    @property
    def sampled(self) -> bool:
        """Whether the gate drew this input (not one of the test's own fixtures)."""
        return self._sample is not None

    @property
    def status(self) -> str:
        return str(self._row["action"].get("status") or "ok")

    @property
    def kwargs(self) -> dict:
        self._read()
        return _copy(self._row["action"].get("kwargs") or {})

    @property
    def response(self) -> Any:
        self._read()
        return _copy(self._row["action"].get("response"))

    def value(self, form: str | None = None) -> Any:
        """The input in *form* (default :attr:`form`)."""
        self._read()
        form = form or self.form
        a = self._row["action"]
        if form == "env":
            ctx = self._row.get("context")
            rows = ctx if isinstance(ctx, list) and ctx else [a]
            return RecordedEnv([_action(r) for r in rows if isinstance(r, dict)])
        if form == "observation":
            return _copy(a.get("response"))
        if form == "text":
            if isinstance(self._row.get("text"), str):
                return self._row["text"]
            if self.kind == "worktree":
                return self._file().read_bytes().decode("utf-8", "replace")
            if isinstance(a.get("response"), str):
                return a["response"]
            raise ValueError("this recorded input has no text form")
        if form == "path":
            return str(self._file())
        if form == "bytes":
            return self._file().read_bytes()
        raise ValueError(f"unknown input form {form!r}; one of {', '.join(FORMS)}")

    def __repr__(self) -> str:
        if self._sample is not None:
            return f"sample-{self._sample}"
        label = self._row.get("label")
        return (
            str(label) if isinstance(label, str) and label else f"{self.kind}-fixture"
        )

    # -- helpers ------------------------------------------------------------------------------------------
    def _file(self) -> Path:
        path = self._row.get("file")
        if isinstance(path, str) and path:
            return Path(path)
        sha = self._row.get("blob") or _blob_of(self._row["action"])
        if not isinstance(sha, str):
            raise ValueError("this recorded input has no file")
        name = self._row.get("name")
        if not isinstance(name, str) or not _NAME.match(name):
            args = self._row["action"].get("args") or [""]
            base = str(args[0]).rsplit("/", 1)[-1] if args else ""
            name = base if _NAME.match(base) else "file"
        dest = Path(tempfile.gettempdir()) / "memlab-inputs" / sha / name
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(blob_path(sha), dest)
        return dest

    def _read(self) -> None:
        """Note that the running test read this sampled input (gate sample run only)."""
        if self._sample is None:
            return
        test = os.environ.get("PYTEST_CURRENT_TEST")
        if not test:
            return
        node = test.rsplit(" (", 1)[0][:500]
        key = (self._item, self._sample, node)
        if key in _reads:
            return
        _reads.add(key)
        line = (
            json.dumps({"item": self._item, "sample": self._sample, "test": node})
            + "\n"
        )
        try:
            fd = os.open(
                READS,
                os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC,
                0o600,
            )
        except OSError:
            return
        try:
            os.write(fd, line.encode())
        finally:
            os.close(fd)


def from_action(
    action: dict,
    *,
    form: str | None = None,
    context: list[dict] | None = None,
    label: str | None = None,
) -> RecordedInput:
    """A fixture from an exported action row (``episodes/<id>.json``'s ``actions[i]``).

    *context*: the recorded calls an ``env`` input answers (default: this call alone).
    """
    row: dict = {"action": dict(action), "form": form, "label": label}
    if context is not None:
        row["context"] = [dict(c) for c in context]
    return RecordedInput(row)


def from_blob(
    sha: str,
    name: str = "file",
    *,
    form: str = "path",
    label: str | None = None,
) -> RecordedInput:
    """A file fixture from a recorded blob id, named *name* (its extension can matter to a reader)."""
    blob_path(sha)  # validates the id
    name = name if isinstance(name, str) and _NAME.match(name) else "file"
    action = {
        "kind": "worktree",
        "channel": "",
        "method": "read",
        "args": [name],
        "kwargs": {},
        "response": {"blob_before": sha, "blob_after": sha},
        "status": "ok",
    }
    return RecordedInput(
        {"action": action, "form": form, "blob": sha, "name": name, "label": label},
    )


def samples(item: str) -> list[RecordedInput]:
    """The gate's sampled inputs of *item* ([] outside the gate's sample run)."""
    if "samples" not in _cache:
        try:
            with open(SAMPLES, "rb") as fh:
                _cache["samples"] = json.loads(fh.read())
        except (OSError, ValueError):
            _cache["samples"] = {}
    data = _cache["samples"]
    entry = data.get("items", {}).get(item) if isinstance(data, dict) else None
    rows = entry.get("rows") if isinstance(entry, dict) else None
    out: list[RecordedInput] = []
    for row in rows if isinstance(rows, list) else []:
        if (
            isinstance(row, dict)
            and row.get("role") == "sample"
            and isinstance(row.get("id"), int)
        ):
            out.append(RecordedInput(row, item=item, sample=row["id"]))
    return out


def inputs(item: str, fixtures: Iterable = ()) -> list:
    """*fixtures*, then the gate's sampled recorded inputs of *item* when the gate draws them."""
    return list(fixtures) + samples(item)

"""``UNIFY_CONSEQUENCES``: the consequence mirror, for the actor.

Empty (off) by default and ``on`` otherwise (with ``UNIFY_TOOLS_IN_CODE=on``; refused at start-up otherwise). Off,
nothing here runs and the actor's prompt and tools are unchanged.

On, every cell result the actor reads ends with one line of measured facts about its own choices so far, and nothing
else (no advice, no world terms):
- what the current request has cost so far: its model calls and their USD, as the run's meter has them (calls whose
  cost was not reported are counted as such, never priced);
- for each draft module (``UNIFY_DRAFTS``, unify/actor/drafts.py) a cell of this request imported: the requests that
  imported it so far and its keep or skip record, from the deployment's draft record (``drafts-record.json`` under the
  harness's home, which no cell sees; :class:`DraftRecord`).

A request starts at the session's start and again after each hand-back.

The module reads the settings only when called (``unify.settings`` imports it to parse the value).
"""

from __future__ import annotations

import json
import os
import tempfile
from decimal import Decimal
from pathlib import Path
from typing import Any, Optional

SETTING = "UNIFY_CONSEQUENCES"
#: The draft record's file name under the harness's home.
RECORD = "drafts-record.json"


def enabled(settings: Any = None) -> bool:
    """Whether ``UNIFY_CONSEQUENCES=on``."""
    from unify.actor import drafts

    if settings is None:
        from unify.settings import SETTINGS as settings
    try:
        return drafts.parse(getattr(settings, SETTING, "") or "") == drafts.ON
    except ValueError:
        return False


def refusal(settings: Any = None) -> str | None:
    """Why this run may not start under the mirror (settings named, never a value), or None."""
    if settings is None:
        from unify.settings import SETTINGS as settings
    if not enabled(settings):
        return None
    if str(getattr(settings, "UNIFY_TOOLS_IN_CODE", "") or "").strip().lower() != "on":
        return f"{SETTING}=on adds a line to each cell's result, so UNIFY_TOOLS_IN_CODE must be on; nothing runs"
    return None


class DraftRecord:
    """The deployment's record of each draft module: the requests that imported it, and the names kept or skipped
    at hand-back. One JSON file in the harness's home, replaced whole at each change (written new, then renamed);
    a missing or unreadable file reads as empty."""

    def __init__(self, path: Optional[Path]) -> None:
        self.path = Path(path) if path is not None else None

    def load(self) -> dict:
        if self.path is None:
            return {}
        try:
            data = json.loads(self.path.read_text())
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def _save(self, data: dict) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".drafts-record-", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w") as fh:
                json.dump(data, fh, sort_keys=True)
            os.replace(tmp, self.path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def _entry(self, data: dict, module: str) -> dict:
        entry = data.get(module)
        if not isinstance(entry, dict):
            entry = data[module] = {}
        entry.setdefault("imports", 0)
        entry.setdefault("kept", [])
        entry.setdefault("skipped", [])
        return entry

    def wrote(self, module: str, writer: dict) -> None:
        """*module* is written by *writer* (``{"session", "arm"}``, UNIFY_ADAPTIVE_POLICY's credit to its arm)."""
        try:
            data = self.load()
            self._entry(data, module)["writer"] = dict(writer)
            self._save(data)
        except Exception:  # noqa: BLE001
            pass

    def writer(self, module: str) -> Optional[dict]:
        """Who wrote *module*, when the record knows."""
        entry = self.load().get(module)
        writer = entry.get("writer") if isinstance(entry, dict) else None
        return writer if isinstance(writer, dict) else None

    def imported(self, module: str) -> None:
        """One more request imported *module*."""
        try:
            data = self.load()
            self._entry(data, module)["imports"] += 1
            self._save(data)
        except Exception:  # noqa: BLE001 - the record never ends a request
            pass

    def decided(self, module: str, kept: list[str], skipped: list[str]) -> None:
        """A hand-back decision about *module*'s definitions."""
        try:
            data = self.load()
            entry = self._entry(data, module)
            entry["kept"] = sorted((set(entry["kept"]) - set(skipped)) | set(kept))
            entry["skipped"] = sorted((set(entry["skipped"]) - set(kept)) | set(skipped))
            self._save(data)
        except Exception:  # noqa: BLE001
            pass


def _totals(meter: Any) -> Optional[tuple[int, Decimal, int]]:
    """(model calls, known USD, calls of unreported cost) over every purpose of the run's meter; None without one."""
    if meter is None:
        return None
    from unify.common.llm_meter import PURPOSES

    calls = sum(int(v) for v in getattr(meter, "calls", {}).values())
    known = sum((meter.known_cost_usd(p) for p in PURPOSES), Decimal(0))
    unknown = sum(int(meter.unknown_cost_calls(p)) for p in PURPOSES)
    return calls, known, unknown


def _usd(value: Decimal) -> str:
    """A plain decimal string, never in exponent form, at most 6 places."""
    return format(value.quantize(Decimal("0.000001")).normalize(), "f")


class Mirror:
    """One session's consequence line: the request's cost since its start, and the imported drafts' records."""

    def __init__(self) -> None:
        self.base: tuple[int, Decimal, int] = (0, Decimal(0), 0)

    def request_ended(self, meter: Any) -> None:
        """A hand-back: the next request is measured from here."""
        totals = _totals(meter)
        if totals is not None:
            self.base = totals

    def line(self, meter: Any, imported: list[str], record: Optional[DraftRecord]) -> str:
        """The facts line for a cell result."""
        parts = []
        totals = _totals(meter)
        if totals is not None:
            calls = totals[0] - self.base[0]
            known = totals[1] - self.base[1]
            unknown = totals[2] - self.base[2]
            cost = f"{_usd(known)} USD"
            if unknown:
                cost += f" ({unknown} call{'' if unknown == 1 else 's'} with no reported cost)"
            parts.append(f"so far in this request: {calls} model call{'' if calls == 1 else 's'}, {cost}")
        data = record.load() if record is not None and imported else {}
        for module in imported:
            entry = data.get(module) if isinstance(data.get(module), dict) else {}
            n = int(entry.get("imports", 0) or 0)
            kept, skipped = list(entry.get("kept") or []), list(entry.get("skipped") or [])
            decided = (f"kept: {', '.join(kept)}" if kept else "") + (
                f"{'; ' if kept else ''}skipped: {', '.join(skipped)}" if skipped else ""
            )
            parts.append(
                f"drafts/{module}.py: imported in {n} request{'' if n == 1 else 's'}; "
                + (decided or "no keep or skip recorded"),
            )
        return f"({'; '.join(parts)})" if parts else ""

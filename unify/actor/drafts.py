"""``UNIFY_DRAFTS`` (drafts by default) and ``UNIFY_HANDBACK_DECISION`` (a decision at hand-back), for the actor.

Both are empty (off) by default and ``on`` otherwise; off, nothing here runs and the actor's prompt and tools are
unchanged.

**Drafts by default.** A function or class the actor defines in its interpreter lives as long as the interpreter: to
the end of the session (one ``act()``), or to a restart. At every hand-back (the request's answer, by
``tools.answer`` or as a plain-text reply), the harness reads the definitions the session's cells made that are new or
changed since the last hand-back, without a model call and without changing anything (a probe in a scratch copy of
the namespace: :data:`PROBE`), and writes them with the cells' top-level imports, unchanged, to one file per session
in the workspace, ``drafts/d<session start, UTC>.py`` (:class:`SessionDrafts`), which later sessions can import in one
line (the workspace is the last entry of the interpreter's import path). The file's header says only when the
harness wrote it and how to import from it: no request text (a request's text can carry a task's label). The file
persists like any workspace file. Definitions an interpreter restart took before a hand-back are not recovered.

**A decision at hand-back** (needs the drafts and ``UNIFY_TOOLS_IN_CODE=on``). Only when a hand-back drafted new or
changed definitions, the loop holds the answer and gives the actor one more turn with :data:`QUESTION`: keep the ones
worth keeping, with a one-line reason, or skip, through ``tools.decide`` (which exists only with the switch on). Keep
marks the named definitions in the draft (``# Kept: <why>``) and drops this hand-back's others from it; skip drops
this hand-back's definitions from it. Nothing requires storing, calling or checking anything; whatever the turn does,
the held answer is handed back unchanged after it. Each draft and each decision is a row of the request's
``tool-events.jsonl``.

The module reads the settings only when called (``unify.settings`` imports it to parse the values).
"""

from __future__ import annotations

import ast
import time
from typing import Any, Optional

SETTING = "UNIFY_DRAFTS"
DECISION_SETTING = "UNIFY_HANDBACK_DECISION"
ON = "on"
#: The workspace folder the drafts are written to (a namespace package: ``from drafts.<module> import <name>``).
DIR = "drafts"
HEADER = "# Drafted by the harness at {at} from definitions made in this session, unchanged."
REUSE = "# Reuse: from {dir}.{module} import <name>"
KEPT = "# Kept: {why}"
#: The notice of the decision turn; ``{names}`` are the definitions asked about, ``{path}`` the draft's path.
QUESTION = (
    "Before your answer is handed back: this request defined {names}. They are saved as a draft in {path}, which "
    'later requests can import. Keep them? Call tools.decide("keep", names=[...], why="one line") for the ones worth '
    'keeping, or tools.decide("skip"). Your answer is already recorded and goes back unchanged.'
)
PROBE_TIMEOUT_S = 60.0
#: The longest reason a decision keeps (one line).
WHY_MAX = 300

#: Run in a scratch copy of the actor's namespace (``PythonWorker.run_cell(..., scratch=True)``): every function and
#: class the session's cells defined that is still bound, with the source of its last top-level definition in the
#: cells (``<cell N>`` in linecache, as ``wrap_cell`` registers them; decorators included), and the cells' top-level
#: import lines in order. Builtins are reached through the ``builtins`` module, since a namespace's own may be reduced.
PROBE = """async def __exec_wrapper():
    __b = __import__('builtins')
    __ast = __import__('ast')
    __lc = __import__('linecache')
    __g = __b.globals()
    __cells = []
    for __k in __b.list(__lc.cache):
        if __b.isinstance(__k, __b.str) and __k.startswith('<cell ') and __k.endswith('>') and __k[6:-1].isdigit():
            __cells.append((__b.int(__k[6:-1]), __k))
    __cells.sort()
    __defs = {}
    __imports = []
    for __n, __k in __cells:
        __lines = __lc.cache[__k][2]
        try:
            __tree = __ast.parse(''.join(__lines))
        except __b.SyntaxError:
            continue
        for __node in __tree.body:
            if __b.isinstance(__node, (__ast.Import, __ast.ImportFrom)):
                __seg = ''.join(__lines[__node.lineno - 1:__node.end_lineno]).rstrip()
                if __seg not in __imports:
                    __imports.append(__seg)
            elif __b.isinstance(__node, (__ast.FunctionDef, __ast.AsyncFunctionDef, __ast.ClassDef)):
                __start = __b.min([__node.lineno] + [__d.lineno for __d in __node.decorator_list])
                __defs.pop(__node.name, None)
                __defs[__node.name] = ''.join(__lines[__start - 1:__node.end_lineno]).rstrip()
    __out = {}
    for __name, __src in __defs.items():
        __v = __g.get(__name)
        __mod = __b.getattr(__v, '__module__', None)
        if not (__b.isinstance(__mod, __b.str) and __mod.startswith('__sandbox_')):
            continue
        if not __b.isinstance(__v, __b.type):
            __u = __v
            for __i in __b.range(20):
                __w = __b.getattr(__u, '__wrapped__', None)
                if __w is None:
                    break
                __u = __w
            __code = __b.getattr(__u, '__code__', None)
            __file = __b.getattr(__code, 'co_filename', '')
            if not (__b.isinstance(__file, __b.str) and __file.startswith('<cell ')):
                continue
        __out[__name] = __src
    return {'defs': __out, 'imports': __imports}
"""


def parse(v: Any) -> str:
    """A switch's value: ``""`` for empty or ``off``, ``on``; anything else is refused."""
    value = str(v if v is not None else "").strip().lower()
    if value in ("", "off"):
        return ""
    if value == ON:
        return ON
    raise ValueError(f"the value must be empty, 'off' or 'on', not {v!r}"[:200])


def _settings(settings: Any) -> Any:
    if settings is None:
        from unify.settings import SETTINGS as settings
    return settings


def _on(settings: Any, name: str) -> bool:
    try:
        return parse(getattr(_settings(settings), name, "") or "") == ON
    except ValueError:
        return False


def enabled(settings: Any = None) -> bool:
    """Whether ``UNIFY_DRAFTS=on``."""
    return _on(settings, SETTING)


def decision_enabled(settings: Any = None) -> bool:
    """Whether ``UNIFY_HANDBACK_DECISION=on``."""
    return _on(settings, DECISION_SETTING)


def refusal(settings: Any = None) -> str | None:
    """Why this run may not start under these switches (settings named, never a value), or None."""
    settings = _settings(settings)
    if not decision_enabled(settings):
        return None
    if not enabled(settings):
        return f"{DECISION_SETTING}=on asks about the drafts, so {SETTING} must be on too; nothing runs"
    if str(getattr(settings, "UNIFY_TOOLS_IN_CODE", "") or "").strip().lower() != "on":
        return f"{DECISION_SETTING}=on is answered with tools.decide, so UNIFY_TOOLS_IN_CODE must be on; nothing runs"
    return None


def _utc(at: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(at))


class SessionDrafts:
    """One session's draft file and the decisions about it (the actor's, one per ``InCodeTools``)."""

    def __init__(self, *, root_fd: int, events: Any, started: Optional[float] = None, record: Any = None) -> None:
        self.root_fd = root_fd
        self.events = events
        #: the deployment's draft record (``unify.actor.consequences.DraftRecord``): imports and decisions; None: none
        self.record = record
        #: the draft modules a cell of the current request imported, in order (each counted once per request)
        self.imported_now: list[str] = []
        at = started if started is not None else time.time()
        self.module = "d" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(at))
        #: name -> source as drafted (the session's latest), in first-drafted order
        self.drafted: dict[str, str] = {}
        #: name -> the one-line reason it was kept
        self.kept: dict[str, str] = {}
        self.imports: list[str] = []
        #: what the open question asked about, and whether it was answered
        self.asked: list[str] = []
        self.decided = False
        #: whether the module name is claimed (a free name, taken at the first write)
        self._claimed = False

    @property
    def rel(self) -> str:
        return f"{DIR}/{self.module}.py"

    def render(self, at: float) -> str:
        """The draft file's text."""
        parts = [HEADER.format(at=_utc(at)), REUSE.format(dir=DIR, module=self.module), ""]
        if self.imports:
            parts += [*self.imports, ""]
        for name, src in self.drafted.items():
            parts.append("")
            if name in self.kept:
                parts.append(KEPT.format(why=self.kept[name]))
            parts += [src, ""]
        return "\n".join(parts).rstrip("\n") + "\n"

    def _claim(self) -> None:
        """Take a module name no draft file has yet (a session started in the same second gets ``_2``, ...)."""
        import os

        from unify.memory import safe_paths

        base, n = self.module, 1
        dfd = safe_paths.walk_dirs(self.root_fd, [DIR], create=True)
        try:
            while True:
                try:
                    os.stat(f"{self.module}.py", dir_fd=dfd, follow_symlinks=False)
                except FileNotFoundError:
                    break
                n += 1
                self.module = f"{base}_{n}"
        finally:
            os.close(dfd)
        self._claimed = True

    def _write(self) -> int:
        from unify.memory import safe_paths

        if not self.drafted:
            self._remove()
            return 0
        if not self._claimed:
            self._claim()
        data = self.render(time.time()).encode("utf-8")
        safe_paths.write_rel(self.root_fd, [DIR, f"{self.module}.py"], data)
        return len(data)

    def _remove(self) -> None:
        import os

        from unify.memory import safe_paths

        try:
            dfd = safe_paths.walk_dirs(self.root_fd, [DIR], create=False)
        except Exception:  # noqa: BLE001 - no folder: nothing to remove
            return
        try:
            os.unlink(f"{self.module}.py", dir_fd=dfd)
        except FileNotFoundError:
            pass
        finally:
            os.close(dfd)

    def _event(self, row: dict) -> None:
        try:
            self.events.append(row)
        except Exception:  # noqa: BLE001 - the record never ends the request
            pass

    def take(self, probed: Any) -> list[str]:
        """Merge a probe's result: the names new or changed since the last hand-back (the draft is rewritten when
        there are any)."""
        if not isinstance(probed, dict) or not isinstance(probed.get("defs"), dict):
            return []
        defs = {str(k): str(v) for k, v in probed["defs"].items() if isinstance(v, str)}
        changed = [name for name, src in defs.items() if self.drafted.get(name) != src]
        if not changed:
            return []
        for name in changed:
            self.drafted[name] = defs[name]
            self.kept.pop(name, None)
        for line in probed.get("imports") or []:
            if isinstance(line, str) and line not in self.imports:
                self.imports.append(line)
        size = self._write()
        self._event({"type": "draft", "file": self.rel, "names": changed, "bytes": size})
        return changed

    async def hand_back(self, interp: Any) -> list[str]:
        """At a hand-back: probe the live interpreter and draft what is new or changed (nothing when it is gone)."""
        if interp is None or not hasattr(interp, "probe"):
            return []
        try:
            probed = await interp.probe(PROBE, timeout=PROBE_TIMEOUT_S)
        except Exception:  # noqa: BLE001 - a failed probe drafts nothing and never ends the request
            return []
        return self.take(probed)

    def note_use(self, code: str) -> None:
        """A ``draft_use`` row for each draft module a cell imports from (``import drafts.<m>``, ``from drafts.<m>
        import a, b``, ``from drafts import <m>``), read from the cell's source on the host: the file, the names
        imported (empty for a module import), and whether it is this session's own draft. Nothing in the cell
        changes; an unparsable cell records nothing."""
        try:
            tree = ast.parse(code)
        except (SyntaxError, ValueError):
            return
        uses: dict[str, list[str]] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    parts = alias.name.split(".")
                    if len(parts) >= 2 and parts[0] == DIR:
                        uses.setdefault(parts[1], [])
            elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
                parts = node.module.split(".")
                if parts[0] != DIR:
                    continue
                if len(parts) >= 2:
                    uses.setdefault(parts[1], []).extend(a.name for a in node.names)
                else:
                    for alias in node.names:
                        uses.setdefault(alias.name, [])
        for module, names in uses.items():
            if module not in self.imported_now:
                self.imported_now.append(module)
                if self.record is not None:
                    self.record.imported(module)
            self._event(
                {
                    "type": "draft_use",
                    "file": f"{DIR}/{module}.py",
                    "names": sorted(set(names)),
                    # this session's own file only once it has one (an unclaimed name may be another session's)
                    "same_session": self._claimed and module == self.module,
                },
            )

    def question(self, names: list[str]) -> str:
        """The decision turn's notice about *names*, and the question opened."""
        self.asked, self.decided = list(names), False
        shown = ", ".join(f"`{n}`" for n in names)
        return QUESTION.format(names=shown, path=f"`{self.rel}`")

    def decide(self, choice: Any, names: Any, why: Any) -> tuple[Optional[str], str]:
        """``tools.decide``: (refusal code or None, the reply). Keep marks *names* and drops the question's others;
        skip drops the question's names."""
        if not self.asked or self.decided:
            return "no_question", ""
        choice = str(choice or "").strip().lower()
        if choice not in ("keep", "skip"):
            return "bad_choice", ""
        reason = " ".join(str(why or "").split())[:WHY_MAX]
        if choice == "keep":
            if not isinstance(names, (list, tuple)) or not names or not all(isinstance(n, str) for n in names):
                return "no_names", ""
            unknown = [n for n in names if n not in self.asked]
            if unknown:
                return "not_asked", ", ".join(unknown)[:300]
            keep = [n for n in self.asked if n in names]
            for name in self.asked:
                if name in keep:
                    self.kept[name] = reason or "-"
                else:
                    self.drafted.pop(name, None)
            dropped = [n for n in self.asked if n not in keep]
        else:
            keep, dropped = [], list(self.asked)
            for name in self.asked:
                self.drafted.pop(name, None)
                self.kept.pop(name, None)
        self._write()
        self.decided = True
        if self.record is not None:
            self.record.decided(self.module, keep, [n for n in self.asked if n not in keep])
        self._event({"type": "decision", "choice": choice, "names": keep, "why": reason, "asked": list(self.asked)})
        if keep:
            return None, f"recorded: kept {', '.join(keep)} in {self.rel}"
        return None, f"recorded: skipped; dropped {', '.join(dropped)}"

    def request_ended(self) -> None:
        """A hand-back: the next request's imports are counted afresh."""
        self.imported_now = []

    def after_question(self) -> None:
        """The decision turn ended: an unanswered question is recorded as ``none``."""
        if self.asked and not self.decided:
            self._event({"type": "decision", "choice": "none", "names": [], "why": "", "asked": list(self.asked)})
        self.asked, self.decided = [], False

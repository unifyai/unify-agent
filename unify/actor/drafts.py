"""``UNIFY_DRAFTS`` (drafts by default) and ``UNIFY_HANDBACK_DECISION`` (a decision at hand-back), for the actor.

Both are empty (off) by default. ``UNIFY_DRAFTS`` is ``on`` (definitions), ``cells`` (a request's working cells) or
``both``; ``UNIFY_HANDBACK_DECISION`` is ``on`` (asked once) or ``required`` (the answer waits for a decision, asked at
most twice). Off, nothing here runs and the actor's prompt and tools are unchanged.

**Drafts by default.** A function or class the actor defines in its interpreter lives as long as the interpreter: to
the end of the session (one ``act()``), or to a restart. At every hand-back (the request's answer, by
``tools.answer`` or as a plain-text reply), the harness reads the definitions the session's cells made that are new or
changed since the last hand-back, without a model call and without changing anything (a probe in a scratch copy of
the namespace: :data:`PROBE`), and writes them with the cells' top-level imports, unchanged, to one file per session
in the workspace, ``drafts/d<session start, UTC>.py`` (:class:`SessionDrafts`), which later sessions can import in one
line (the workspace is the last entry of the interpreter's import path). The file's header says only when the
harness wrote it and how to import from it: no request text (a request's text can carry a task's label). The file
persists like any workspace file. Definitions an interpreter restart took before a hand-back are not recovered.

**A request's working cells** (``cells``, or ``both`` with the definitions). Most requests do their work in top-level
code, not in functions. At every hand-back the harness writes the request's working cells, as the agent wrote them,
to ``drafts/d<session start, UTC>-r<k>.cells.py`` (the k-th hand-back of the session): the cells since the last
hand-back that ran without error, a repeated source once (its last run), reduced to the cells the last one that binds
a name at top level depends on (:func:`working_slice`: ``ast`` over the sources, nothing run). A request whose
working cells only print or query writes nothing. Each cell keeps its source under a ``# cell N`` line (the number
the agent saw). The header says only when the harness wrote it and that it is for reading: running it repeats those
steps, which need that request's variables, and its dotted name is not a module name, so no import runs it. The record
is the host's (each cell's source and outcome as it ends), so a restart in the middle of a request loses nothing. A
later cell that names such a file's path is a ``draft_use`` row.

**A decision at hand-back** (needs the drafts and ``UNIFY_TOOLS_IN_CODE=on``). Only when a hand-back drafted new or
changed definitions, the loop holds the answer and gives the actor one more turn with :data:`QUESTION`: keep the ones
worth keeping, with a one-line reason, or skip, through ``tools.decide`` (which exists only with the switch on). Keep
marks the named definitions in the draft (``# Kept: <why>``) and drops this hand-back's others from it; skip drops
this hand-back's definitions from it. When the hand-back drafted no definitions but wrote a working-cells file, the
turn asks the same about that file (:data:`CELLS_QUESTION`): keep marks it, skip removes it. Nothing requires storing, calling or checking anything; whatever the turn does,
the held answer is handed back unchanged after it.

**A required decision** (``UNIFY_HANDBACK_DECISION=required``). The same question, but the held answer goes back only
after a ``tools.decide`` call: a turn that ends without one gets the question once more, with :data:`REASK_LINE`
added. After that second turn the answer goes back unchanged whatever it did, recorded as ``none``, so the requester
never waits on the model indefinitely. Skip stays one word; nothing has to be stored or checked. Each draft and each decision is a row of the request's
``tool-events.jsonl``.

The module reads the settings only when called (``unify.settings`` imports it to parse the values).
"""

from __future__ import annotations

import ast
import re
import time
from typing import Any, Callable, Optional

SETTING = "UNIFY_DRAFTS"
DECISION_SETTING = "UNIFY_HANDBACK_DECISION"
ON = "on"
#: UNIFY_HANDBACK_DECISION's other value: the answer waits for a decision (asked at most MAX_ASKS times)
REQUIRED = "required"
CELLS = "cells"
BOTH = "both"
#: UNIFY_DRAFTS's values besides empty: definitions, a request's working cells, or both
MODES = (ON, CELLS, BOTH)
#: The workspace folder the drafts are written to (a namespace package: ``from drafts.<module> import <name>``).
DIR = "drafts"
HEADER = "# Drafted by the harness at {at} from definitions made in this session, unchanged."
REUSE = "# Reuse: from {dir}.{module} import <name>"
KEPT = "# Kept: {why}"
CELLS_HEADER = "# Drafted by the harness at {at} from cells of one request in this session that ran without error."
CELLS_REUSE = "# Reuse: read it and copy what applies; running it repeats those steps."
CELL_MARK = "# cell {n}"
#: a working-cells file's suffix: the dot before it keeps the file's name from being a module name
CELLS_SUFFIX = ".cells.py"
#: the decision turn's notice about a working-cells file; ``{path}`` is the file's path
CELLS_QUESTION = (
    "Before your answer is handed back: this request's working cells are saved as a draft in {path}, which later "
    'requests can read. Keep it? Call tools.decide("keep", why="one line") to keep it, or tools.decide("skip"). '
    "Your answer is already recorded and goes back unchanged."
)
#: what the record keeps for a working-cells file's decision (a file, not named definitions)
CELLS_RECORD = "cells"
#: the line a required decision's second ask adds to the question
REASK_LINE = "A keep or skip is needed before your answer goes back."
#: the most times a required decision is asked before the answer goes back without one
MAX_ASKS = 2
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


def parse_mode(v: Any) -> str:
    """``UNIFY_DRAFTS``'s value: ``""`` for empty or ``off``, else one of :data:`MODES`; anything else is refused."""
    value = str(v if v is not None else "").strip().lower()
    if value in ("", "off"):
        return ""
    if value in MODES:
        return value
    raise ValueError(f"the value must be empty, 'off', 'on', 'cells' or 'both', not {v!r}"[:200])


def _settings(settings: Any) -> Any:
    if settings is None:
        from unify.settings import SETTINGS as settings
    return settings


def parse_decision(v: Any) -> str:
    """``UNIFY_HANDBACK_DECISION``'s value: ``""`` for empty or ``off``, ``on`` or ``required``; anything else is
    refused."""
    value = str(v if v is not None else "").strip().lower()
    if value in ("", "off"):
        return ""
    if value in (ON, REQUIRED):
        return value
    raise ValueError(f"the value must be empty, 'off', 'on' or 'required', not {v!r}"[:200])


def decision_mode(settings: Any = None) -> str:
    """``UNIFY_HANDBACK_DECISION``'s mode: ``""`` (off), ``on`` or ``required``."""
    try:
        return parse_decision(getattr(_settings(settings), DECISION_SETTING, "") or "")
    except ValueError:
        return ""


def mode(settings: Any = None) -> str:
    """``UNIFY_DRAFTS``'s mode: ``""`` (off), ``on``, ``cells`` or ``both``."""
    try:
        return parse_mode(getattr(_settings(settings), SETTING, "") or "")
    except ValueError:
        return ""


def enabled(settings: Any = None) -> bool:
    """Whether ``UNIFY_DRAFTS`` is on in any mode."""
    return mode(settings) != ""


def decision_enabled(settings: Any = None) -> bool:
    """Whether ``UNIFY_HANDBACK_DECISION`` is on or required."""
    return decision_mode(settings) != ""


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


#: top-level statements that bind a name: a working cell's mark (a cell of only prints or queries binds nothing)
_BINDING = (
    ast.Assign,
    ast.AugAssign,
    ast.AnnAssign,
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.Import,
    ast.ImportFrom,
    ast.For,
    ast.AsyncFor,
    ast.With,
    ast.AsyncWith,
)


def _names(code: str) -> Optional[tuple[set, set]]:
    """(the names *code* binds at any depth, the names it reads), or None when it does not parse."""
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return None
    binds: set = set()
    reads: set = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            (binds if isinstance(node.ctx, (ast.Store, ast.Del)) else reads).add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            binds.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                binds.add((alias.asname or alias.name).split(".")[0])
    return binds, reads


def binds_at_top(code: str) -> bool:
    """Whether *code* binds a name at top level (:data:`_BINDING`)."""
    try:
        return any(isinstance(node, _BINDING) for node in ast.parse(code).body)
    except (SyntaxError, ValueError):
        return False


def working_slice(cells: list) -> list[tuple[int, str]]:
    """The request's working cells to draft, from ``(number, source, ran without error)`` in order: the ones that ran
    without error, a repeated source once (at its last run), then the last of them that binds a name at top level and,
    going back, every earlier one that last bound a name a kept cell reads. Empty when none binds a name."""
    seen: set = set()
    working: list[tuple[int, str]] = []
    for number, source, ok in reversed(cells):
        if ok and isinstance(source, str) and source.strip() and source not in seen:
            seen.add(source)
            working.append((int(number), source))
    working.reverse()
    roots = [i for i, (_n, source) in enumerate(working) if binds_at_top(source)]
    if not roots:
        return []
    info = [_names(source) for _n, source in working]
    root = roots[-1]
    keep = {root}
    need = set((info[root] or (set(), set()))[1])
    for i in range(root - 1, -1, -1):
        if info[i] is None:
            continue
        binds, reads = info[i]
        if binds & need:
            keep.add(i)
            need = (need - binds) | reads
    return [working[i] for i in sorted(keep)]


def render_cells(at: float, cells: list[tuple[int, str]], kept: Optional[str] = None) -> str:
    """A working-cells file's text."""
    parts = [CELLS_HEADER.format(at=_utc(at)), CELLS_REUSE]
    if kept is not None:
        parts.append(KEPT.format(why=kept))
    for number, source in cells:
        parts += ["", CELL_MARK.format(n=number), source.rstrip("\n")]
    return "\n".join(parts) + "\n"


#: a working-cells file named in a cell's string (a read of it): its stem, ``d<UTC>[_n]-r<k>.cells``
_CELLS_PATH = re.compile(r"(?:^|/)" + DIR + r"/(d[0-9]{8}T[0-9]{6}Z(?:_[0-9]+)?-r[0-9]+\.cells)\.py")


async def probe(interp: Any) -> Optional[dict]:
    """The live interpreter's definitions and imports (:data:`PROBE`); None when there is none or the probe fails."""
    if interp is None or not hasattr(interp, "probe"):
        return None
    try:
        probed = await interp.probe(PROBE, timeout=PROBE_TIMEOUT_S)
    except Exception:  # noqa: BLE001 - a failed probe drafts nothing and never ends the request
        return None
    return probed if isinstance(probed, dict) else None


class SessionDrafts:
    """One session's draft file and the decisions about it (the actor's, one per ``InCodeTools``)."""

    def __init__(
        self,
        *,
        root_fd: int,
        events: Any,
        started: Optional[float] = None,
        record: Any = None,
        writes: bool = True,
        writer: Optional[dict] = None,
        on_import: Optional[Callable[[str, bool], None]] = None,
        mode: str = ON,
    ) -> None:
        self.root_fd = root_fd
        #: UNIFY_DRAFTS's mode: definitions (on, both), a request's working cells (cells, both)
        self.definitions = mode in (ON, BOTH)
        self.cells = mode in (CELLS, BOTH)
        self.events = events
        #: False under UNIFY_ADAPTIVE_POLICY's arm without drafts: imports are still noticed, nothing is written
        self.writes = writes
        #: who writes this session's draft (``{"session", "arm"}``), kept in the draft record; None: not recorded
        self.writer = writer
        #: called with (module, same session) at a request's first import of each draft module
        self.on_import = on_import
        #: the names a question already asked about (answered or not)
        self.decided_names: set = set()
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
        #: the current request's cells as they ended: (number, source, ran without error)
        self.cells_now: list[tuple[int, str, bool]] = []
        #: the session's hand-backs so far (a working-cells file is named by its hand-back)
        self.hand_backs = 0
        #: each working-cells file written: its path -> its cells (kept for a keep's rewrite)
        self.cells_files: dict[str, list[tuple[int, str]]] = {}
        #: the working-cells file the open question asks about, if it asks about one
        self.asked_file: Optional[str] = None
        #: UNIFY_HANDBACK_DECISION=required: the answer waits for a decision (set by the session's owner)
        self.required = False
        #: how many times the open question was asked, and its text (a required decision asks it again)
        self.asks = 0
        self.question_text = ""

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
        """Take a module name no draft file has yet, as a definitions file or a working-cells file's prefix (a session
        started in the same second gets ``_2``, ...)."""
        import os

        from unify.memory import safe_paths

        base, n = self.module, 1
        dfd = safe_paths.walk_dirs(self.root_fd, [DIR], create=True)
        try:
            names = set(os.listdir(dfd))
        finally:
            os.close(dfd)
        while f"{self.module}.py" in names or any(x.startswith(f"{self.module}-r") for x in names):
            n += 1
            self.module = f"{base}_{n}"
        self._claimed = True

    def _write(self) -> int:
        from unify.memory import safe_paths

        if not self.drafted:
            self._remove()
            return 0
        if not self._claimed:
            self._claim()
            if self.record is not None and self.writer is not None:
                self.record.wrote(self.module, self.writer)
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
        if not self.definitions or not self.writes or not isinstance(probed, dict) or not isinstance(
            probed.get("defs"),
            dict,
        ):
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
        return self.take(await probe(interp) if self.definitions else None)

    def note_cell(self, number: int, source: str, ok: bool) -> None:
        """A cell ended: its number, source and whether it ran without error (the working-cells record)."""
        if self.cells and self.writes and isinstance(source, str):
            self.cells_now.append((int(number), source, bool(ok)))

    def _cells_key(self, rel: str) -> str:
        """The record's key for a working-cells file: its name without ``.py`` (so ``drafts/<key>.py`` is its path)."""
        return rel[len(DIR) + 1 : -len(".py")]

    def _write_cells(self, rel: str, kept: Optional[str] = None) -> int:
        from unify.memory import safe_paths

        data = render_cells(time.time(), self.cells_files[rel], kept).encode("utf-8")
        safe_paths.write_rel(self.root_fd, [DIR, rel[len(DIR) + 1 :]], data)
        return len(data)

    def _remove_cells(self, rel: str) -> None:
        import os

        from unify.memory import safe_paths

        try:
            dfd = safe_paths.walk_dirs(self.root_fd, [DIR], create=False)
        except Exception:  # noqa: BLE001 - no folder: nothing to remove
            return
        try:
            os.unlink(rel[len(DIR) + 1 :], dir_fd=dfd)
        except FileNotFoundError:
            pass
        finally:
            os.close(dfd)

    def take_cells(self) -> Optional[str]:
        """At a hand-back: write the request's working cells (:func:`working_slice`) to this hand-back's file; its
        path, or None when there is nothing to write. The request's record starts afresh either way."""
        self.hand_backs += 1
        cells, self.cells_now = self.cells_now, []
        if not self.cells or not self.writes:
            return None
        kept = working_slice(cells)
        if not kept:
            return None
        if not self._claimed:
            self._claim()
        rel = f"{DIR}/{self.module}-r{self.hand_backs}{CELLS_SUFFIX}"
        self.cells_files[rel] = kept
        size = self._write_cells(rel)
        if self.record is not None and self.writer is not None:
            self.record.wrote(self._cells_key(rel), self.writer)
        lines = sum(len(source.splitlines()) for _n, source in kept)
        self._event(
            {"type": "draft", "mode": CELLS, "file": rel, "cells": [n for n, _s in kept], "lines": lines, "bytes": size},
        )
        return rel

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
        # a string naming a working-cells file's path: a read of it (open, print, exec all start from its path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                for key in _CELLS_PATH.findall(node.value):
                    uses.setdefault(key, [])
        for module, names in uses.items():
            # this session's own file only once it has one (an unclaimed name may be another session's)
            same = self._claimed and (module == self.module or module.startswith(f"{self.module}-r"))
            if module not in self.imported_now:
                self.imported_now.append(module)
                if self.record is not None:
                    self.record.imported(module)
                if self.on_import is not None:
                    try:
                        self.on_import(module, same)
                    except Exception:  # noqa: BLE001 - a listener never ends the request
                        pass
            row = {"type": "draft_use", "file": f"{DIR}/{module}.py", "names": sorted(set(names)), "same_session": same}
            if module.endswith(".cells"):
                row["mode"] = CELLS
            self._event(row)

    def question(self, names: list[str]) -> str:
        """The decision turn's notice about *names*, and the question opened."""
        self.asked, self.decided, self.asked_file = list(names), False, None
        self.decided_names.update(names)
        shown = ", ".join(f"`{n}`" for n in names)
        self.asks, self.question_text = 1, QUESTION.format(names=shown, path=f"`{self.rel}`")
        return self.question_text

    def question_file(self, rel: str) -> str:
        """The decision turn's notice about the working-cells file *rel*, and the question opened."""
        self.asked, self.decided, self.asked_file = [], False, rel
        self.asks, self.question_text = 1, CELLS_QUESTION.format(path=f"`{rel}`")
        return self.question_text

    def _decide_file(self, choice: str, reason: str) -> tuple[Optional[str], str]:
        rel = self.asked_file or ""
        key = self._cells_key(rel)
        if choice == "keep":
            self._write_cells(rel, reason or "-")
        else:
            self._remove_cells(rel)
            self.cells_files.pop(rel, None)
        self.decided = True
        if self.record is not None:
            self.record.decided(key, [CELLS_RECORD] if choice == "keep" else [], [] if choice == "keep" else [CELLS_RECORD])
        self._event(self._asks({"type": "decision", "choice": choice, "names": [], "why": reason, "asked": [rel],
                                "mode": CELLS}))
        return None, (f"recorded: kept {rel}" if choice == "keep" else f"recorded: skipped; removed {rel}")

    def decide(self, choice: Any, names: Any, why: Any) -> tuple[Optional[str], str]:
        """``tools.decide``: (refusal code or None, the reply). Keep marks *names* and drops the question's others;
        skip drops the question's names."""
        if (not self.asked and self.asked_file is None) or self.decided:
            return "no_question", ""
        choice = str(choice or "").strip().lower()
        if choice not in ("keep", "skip"):
            return "bad_choice", ""
        reason = " ".join(str(why or "").split())[:WHY_MAX]
        if self.asked_file is not None:
            return self._decide_file(choice, reason)
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
        self._event(self._asks({"type": "decision", "choice": choice, "names": keep, "why": reason,
                                "asked": list(self.asked)}))
        if keep:
            return None, f"recorded: kept {', '.join(keep)} in {self.rel}"
        return None, f"recorded: skipped; dropped {', '.join(dropped)}"

    def undecided(self) -> list[str]:
        """The drafted names no question has asked about yet (a decision at every hand-back asks about these)."""
        return [name for name in self.drafted if name not in self.decided_names]

    def request_ended(self) -> None:
        """A hand-back: the next request's imports are counted afresh."""
        self.imported_now = []

    def _asks(self, row: dict) -> dict:
        """A decision row, with how many times it was asked when the decision is required."""
        if self.required:
            row["asks"] = self.asks
        return row

    def after_question(self) -> Optional[str]:
        """The decision turn ended. A required decision not made yet is asked again (the question with
        :data:`REASK_LINE`; returned) until it was asked :data:`MAX_ASKS` times; otherwise an unanswered question is
        recorded as ``none`` and the question closes (None)."""
        open_question = (self.asked_file is not None or bool(self.asked)) and not self.decided
        if open_question and self.required and self.asks < MAX_ASKS:
            self.asks += 1
            asked = [self.asked_file] if self.asked_file is not None else list(self.asked)
            self._event({"type": "decision_asked_again", "asked": asked, "asks": self.asks})
            return f"{self.question_text}\n{REASK_LINE}"
        if self.asked_file is not None and not self.decided:
            self._event(self._asks({"type": "decision", "choice": "none", "names": [], "why": "",
                                    "asked": [self.asked_file], "mode": CELLS}))
        elif self.asked and not self.decided:
            self._event(self._asks({"type": "decision", "choice": "none", "names": [], "why": "",
                                    "asked": list(self.asked)}))
        self.asked, self.decided, self.asked_file = [], False, None
        self.asks, self.question_text = 0, ""
        return None

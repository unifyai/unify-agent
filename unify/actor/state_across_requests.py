"""``UNIFY_CELL_STATE=deployment``: the actor's top-level Python names carry over from one request to the next.

Under ``persistent`` (the default) the actor's interpreter keeps its names between the cells of one request and is
gone when the request ends; only files in the workspace carry over. Under ``deployment`` the actor's names are also
kept across requests, by a save at each hand-back and a restore whenever an interpreter starts:

* **Save** (:data:`SAVE`, run in a scratch copy of the live namespace, so nothing the cells bound changes): the cells'
  top-level import lines, the source of every function and class the cells (or an earlier restore) defined that is
  still bound, and every other top-level value that pickles and unpickles back to the same type. Private names
  (``_x``), modules, the pre-bound names (``tools`` and the rest) and any class or function of the harness's own
  packages (which it binds again in every interpreter) are never saved. A value that does not round-trip
  is named with its type in the snapshot's list of names not kept. The snapshot is written to :data:`DIR` in the
  workspace (code as JSON text, values as one pickle), then ``latest.json`` points to it.
* **Restore** (:data:`RESTORE`, run in the live namespace before the first cell of an interpreter: a request's first
  cell, or the cell after a restart): first it records the names already bound (the harness's: the pre-bound names
  and what it syncs in), which this interpreter's saves leave out; then imports, then definitions (each under ``<restored N>`` in ``linecache``, so a
  later save finds its source and the drafts probe, which reads ``<cell N>`` sources only, never drafts it again),
  then values; whatever failed is tried once more after the others. What could not be kept, at the save or now, is
  returned, and the cell's output names it.

The host never unpickles anything: both sources run in the actor's confined box, which already runs this agent's own
code, and the host sees only their JSON counts. Nothing is capped; a snapshot is what the agent keeps defined (``del``
drops a name). Requires ``UNIFY_TOOLS_IN_CODE=on`` and every memory switch off (:data:`MEMORY_SWITCHES`: memory
agents read the actor's workspace), refused at start-up otherwise.
"""

from __future__ import annotations

from typing import Any, Optional

SETTING = "UNIFY_CELL_STATE"
DEPLOYMENT = "deployment"
#: The workspace folder the snapshots are written to (hidden; the agent's own files stay where it puts them).
DIR = ".python-state"
#: How long a save or a restore may run (s): it pickles or loads what the agent keeps, which has no cap.
TIMEOUT_S = 300.0
#: The switches that give other agents (the memory writer, its fork, the librarian) the actor's workspace: a snapshot
#: there must never reach a box but the actor's own, so ``deployment`` is refused while any of them is on.
MEMORY_SWITCHES = ("UNIFY_MEMORY", "UNIFY_MEMORY_LIBRARY", "UNIFY_MEMORY_LIBRARIAN", "UNIFY_MEMORY_LIBRARY_FORK")
#: What a restart note says is lost under ``deployment`` (``box_interpreter._LOST`` otherwise).
LOST = "names kept at the last hand-back are restored, later ones are gone; files remain"

_SAVE = """async def __exec_wrapper():
    __b = __import__('builtins')
    __ast = __import__('ast')
    __lc = __import__('linecache')
    __os = __import__('os')
    __json = __import__('json')
    __pickle = __import__('pickle')
    __time = __import__('time')
    __types = __import__('types')
    __g = __b.globals()
    if '__harness_names__' not in __g:
        # no restore ran in this interpreter (an earlier one failed), so the harness's names are not known: no save,
        # the last snapshot stays
        return {'snapshot': None, 'skipped': 'no record of the names bound at start'}
    __skip = __b.set(__PREBOUND__) | __b.set(__g['__harness_names__'])
    __keys = []
    for __k in __b.list(__lc.cache):
        if not (__b.isinstance(__k, __b.str) and __k.endswith('>')):
            continue
        for __rank, __head in ((0, '<restored import '), (1, '<restored '), (2, '<cell ')):
            if __k.startswith(__head) and __k[__b.len(__head):-1].isdigit():
                __keys.append((__rank, __b.int(__k[__b.len(__head):-1]), __k))
                break
    __keys.sort()
    __defs = {}
    __imports = []
    for __rank, __n, __k in __keys:
        __lines = __lc.cache[__k][2]
        try:
            __tree = __ast.parse(''.join(__lines))
        except __b.SyntaxError:
            continue
        for __node in __tree.body:
            if __b.isinstance(__node, (__ast.Import, __ast.ImportFrom)):
                __seg = ''.join(__lines[__node.lineno - 1:__node.end_lineno]).strip()
                if __seg not in __imports:
                    __imports.append(__seg)
            elif __b.isinstance(__node, (__ast.FunctionDef, __ast.AsyncFunctionDef, __ast.ClassDef)):
                __start = __b.min([__node.lineno] + [__d.lineno for __d in __node.decorator_list])
                __defs.pop(__node.name, None)
                __defs[__node.name] = ''.join(__lines[__start - 1:__node.end_lineno]).rstrip()
    def __own(__v):
        __mod = __b.getattr(__v, '__module__', None)
        if not (__b.isinstance(__mod, __b.str) and __mod.startswith('__sandbox_')):
            return False
        if __b.isinstance(__v, __b.type):
            return True
        __u = __v
        for __i in __b.range(20):
            __w = __b.getattr(__u, '__wrapped__', None)
            if __w is None:
                break
            __u = __w
        __file = __b.getattr(__b.getattr(__u, '__code__', None), 'co_filename', '')
        return __b.isinstance(__file, __b.str) and (__file.startswith('<cell ') or __file.startswith('<restored '))
    __kept_defs = [[__name, __src] for __name, __src in __defs.items() if __name in __g and __own(__g[__name])]
    __def_names = __b.set(__name for __name, __src in __kept_defs)
    __bound_by_import = __b.set()
    for __line in __imports:
        try:
            for __node in __ast.parse(__line).body:
                for __alias in __node.names:
                    __bound_by_import.add((__alias.asname or __alias.name).split('.')[0])
        except __b.SyntaxError:
            pass
    __values = {}
    __not_kept = []
    for __name, __v in __b.list(__g.items()):
        if __name.startswith('_') or __name in __skip or __name in __def_names:
            continue
        if __b.isinstance(__v, __types.ModuleType) or __name in __bound_by_import:
            continue
        if __b.isinstance(__v, __b.type) or __b.isinstance(__v, (__types.FunctionType, __types.BuiltinFunctionType)):
            __home = __b.getattr(__v, '__module__', None)
            if __b.isinstance(__home, __b.str) and __home.split('.')[0] in ('unify', 'unillm'):
                continue   # the harness's own classes and functions: it binds them again in every interpreter
        try:
            __data = __pickle.dumps(__v, protocol=5)
            __back = __pickle.loads(__data)
            if __b.type(__back) is not __b.type(__v):
                raise __b.TypeError('a different type came back')
        except __b.BaseException:
            __not_kept.append([__name, __b.type(__v).__name__])
            continue
        __values[__name] = __data
    __dir = __DIR__
    __os.makedirs(__dir, exist_ok=True)
    __sid = __time.strftime('%Y%m%dT%H%M%SZ', __time.gmtime()) + '-' + __os.urandom(4).hex()
    __blob = __pickle.dumps({'values': __values}, protocol=5)
    __manifest = {'snapshot': __sid, 'imports': __imports, 'defs': __kept_defs, 'values': __b.sorted(__values),
                  'not_kept': __not_kept, 'bytes': __b.len(__blob)}
    for __suffix, __data, __mode in (('.pkl', __blob, 'wb'), ('.json', __json.dumps(__manifest), 'w')):
        __tmp = __os.path.join(__dir, __sid + __suffix + '.part')
        with __b.open(__tmp, __mode) as __f:
            __f.write(__data)
        __os.replace(__tmp, __os.path.join(__dir, __sid + __suffix))
    __tmp = __os.path.join(__dir, 'latest.json.part')
    with __b.open(__tmp, 'w') as __f:
        __f.write(__json.dumps({'snapshot': __sid}))
    __os.replace(__tmp, __os.path.join(__dir, 'latest.json'))
    for __old in __b.sorted(__os.listdir(__dir)):
        if __old == 'latest.json' or __old.startswith(__sid) or not (__old.endswith('.pkl') or __old.endswith('.json')):
            continue
        try:
            __os.remove(__os.path.join(__dir, __old))
        except __b.OSError:
            pass
    return {'snapshot': __sid, 'imports': __b.len(__imports), 'defs': __b.len(__kept_defs), 'values': __b.len(__values),
            'not_kept': __not_kept, 'bytes': __b.len(__blob)}
"""

#: Run in the live namespace before an interpreter's first cell; returns the counts and what could not be kept.
RESTORE = """async def __exec_wrapper():
    __b = __import__('builtins')
    __lc = __import__('linecache')
    __os = __import__('os')
    __json = __import__('json')
    __pickle = __import__('pickle')
    __g = __b.globals()
    # every name bound before the first cell is the harness's (the pre-bound names and what it syncs in): the saves
    # of this interpreter leave them out, so a later restore never puts an old copy over a fresh one
    __g['__harness_names__'] = __b.frozenset(__g)
    __dir = __DIR__
    try:
        with __b.open(__os.path.join(__dir, 'latest.json')) as __f:
            __sid = __json.load(__f)['snapshot']
    except __b.FileNotFoundError:
        return {'snapshot': None}
    with __b.open(__os.path.join(__dir, __sid + '.json')) as __f:
        __m = __json.load(__f)
    with __b.open(__os.path.join(__dir, __sid + '.pkl'), 'rb') as __f:
        __values = __pickle.load(__f)['values']
    __failed = {}
    __done = {'imports': 0, 'defs': 0, 'values': 0}
    def __source(__key, __text):
        __lc.cache[__key] = (__b.len(__text), None, __text.splitlines(True), __key)
        return __b.compile(__text, __key, 'exec')
    for __i, __line in __b.enumerate(__m['imports']):
        try:
            __b.exec(__source('<restored import ' + __b.str(__i) + '>', __line + '\\n'), __g)
            __done['imports'] += 1
        except __b.BaseException as __e:
            __failed['import ' + __line.split()[1] if __b.len(__line.split()) > 1 else __line] = __b.type(__e).__name__
    def __defs(__todo):
        __left = []
        for __i, __name, __src in __todo:
            try:
                __b.exec(__source('<restored ' + __b.str(__i) + '>', __src + '\\n'), __g)
                __done['defs'] += 1
                __failed.pop(__name, None)
            except __b.BaseException as __e:
                __failed[__name] = __b.type(__e).__name__
                __left.append((__i, __name, __src))
        return __left
    def __vals(__todo):
        __left = []
        for __name in __todo:
            try:
                __g[__name] = __pickle.loads(__values[__name])
                __done['values'] += 1
                __failed.pop(__name, None)
            except __b.BaseException as __e:
                __failed[__name] = __b.type(__e).__name__
                __left.append(__name)
        return __left
    __d = __defs([(__i, __n, __s) for __i, (__n, __s) in __b.enumerate(__m['defs'])])
    __v = __vals(__b.sorted(__values))
    if __d:
        __d = __defs(__d)
    if __v:
        __vals(__v)
    return {'snapshot': __sid, **__done, 'not_kept': __m.get('not_kept') or [],
            'failed': [[__k, __t] for __k, __t in __failed.items()]}
"""
RESTORE = RESTORE.replace("__DIR__", repr(DIR))


def save_source(prebound: Any) -> str:
    """:data:`_SAVE` with the pre-bound names (never saved) and the folder filled in."""
    return _SAVE.replace("__PREBOUND__", repr(sorted(str(n) for n in prebound))).replace("__DIR__", repr(DIR))


def enabled(settings: Any = None) -> bool:
    """Whether ``UNIFY_CELL_STATE`` is ``deployment``."""
    if settings is None:
        from unify.settings import SETTINGS as settings
    return str(getattr(settings, SETTING, "") or "").strip().lower() == DEPLOYMENT


def refusal(settings: Any = None) -> Optional[str]:
    """Why this run may not start under this setting (settings named, never a value), or None."""
    if settings is None:
        from unify.settings import SETTINGS as settings
    if not enabled(settings):
        return None
    if str(getattr(settings, "UNIFY_TOOLS_IN_CODE", "") or "").strip().lower() != "on":
        return f"{SETTING}={DEPLOYMENT} keeps the actor's cell names through tools in code, so UNIFY_TOOLS_IN_CODE must be on; nothing runs"
    on = [name for name in MEMORY_SWITCHES if str(getattr(settings, name, "") or "").strip().lower() == "on"]
    if on:
        return (
            f"{SETTING}={DEPLOYMENT} keeps the actor's names in its workspace, which memory agents read, so "
            f"{', '.join(on)} must be off; nothing runs"
        )
    return None


def not_kept_note(restored: Optional[dict], error: Optional[str]) -> Optional[str]:
    """The line before the next cell's output after a restore: what could not be kept (at the save, or now), or that
    the restore itself failed; None when everything came back or there was nothing to restore."""
    if error is not None:
        return f"(the names kept from earlier requests could not be restored: {error})"
    if not restored or not restored.get("snapshot"):
        return None
    missing = [f"{name} ({kind})" for name, kind in (restored.get("not_kept") or []) + (restored.get("failed") or [])]
    if not missing:
        return None
    return "(not kept from earlier requests: " + ", ".join(missing) + ")"


def counts(result: Any) -> dict:
    """The event row's counts from a save's or a restore's JSON (names of what was not kept are left out)."""
    if not isinstance(result, dict):
        return {"error": "no result"}
    out = {k: result[k] for k in ("imports", "defs", "values", "bytes") if isinstance(result.get(k), int)}
    out["not_kept"] = len(result.get("not_kept") or []) + len(result.get("failed") or [])
    out["snapshot"] = bool(result.get("snapshot"))
    return out


__all__ = ["SETTING", "DEPLOYMENT", "DIR", "TIMEOUT_S", "MEMORY_SWITCHES", "LOST", "RESTORE", "save_source", "enabled", "refusal",
           "not_kept_note", "counts"]

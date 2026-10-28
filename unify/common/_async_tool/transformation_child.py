"""One context-compression transformation, run inside the sandbox.

Started by :func:`unify.common._async_tool.transformation_sandbox.run_transformation`
as ``python -I -c <this file's text>`` in a bubblewrap box
(:func:`unify.memory.sandbox_run.start_confined`); the harness never imports
it. It imports nothing but the standard library before the transformation
runs, and then only the modules its globals name.

stdin, one JSON object::

    {"code": str, "x": str, "builtins": [name, ...],
     "globals": {name: ["module", module] | ["attr", module, name]},
     "memory_bytes": int}

The descriptor stdout had at start receives one JSON object, ``{"x": str}``
or ``{"error": [exception type name, str(exception)]}``; anything the
transformation prints, or writes to descriptor 1, goes to stderr.
"""

import builtins
import importlib
import json
import os
import resource
import sys


def _globals(msg: dict) -> dict:
    names = msg.get("builtins") or []
    g: dict = {
        "__builtins__": {
            n: getattr(builtins, n) for n in names if hasattr(builtins, n)
        },
    }
    for name, spec in (msg.get("globals") or {}).items():
        try:
            value = importlib.import_module(spec[1])
            if spec[0] == "attr":
                value = getattr(value, spec[2])
        except Exception:  # noqa: BLE001 - absent here as it would be in use
            continue
        g[name] = value
    return g


def _transform(msg: dict) -> str:
    g = _globals(msg)
    g["x"] = msg["x"]
    code = msg["code"].strip()
    # A bare expression such as ``x.replace("old", "new")`` is wrapped as
    # ``x = <expr>``; multi-line code or code containing ``=`` runs as-is
    # and must assign ``x`` itself.
    if "\n" not in code and "=" not in code:
        try:
            compile(code, "<transformation>", "eval")
            code = f"x = {code}"
        except SyntaxError:
            pass
    exec(code, g)  # noqa: S102 - confined: this process is the sandbox
    result = g["x"]
    if not isinstance(result, str):
        result = str(result)
    return result


def main() -> None:
    msg = json.loads(sys.stdin.buffer.read())
    sys.stdin.close()
    limit = int(msg.get("memory_bytes") or 0)
    if limit > 0:
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    reply = os.fdopen(os.dup(1), "w", encoding="ascii")
    os.dup2(2, 1)
    try:
        out = {"x": _transform(msg)}
    except BaseException as exc:  # noqa: BLE001 - reported to the caller
        try:
            text = str(exc)
        except BaseException:  # noqa: BLE001
            text = ""
        out = {"error": [type(exc).__name__, text]}
    reply.write(json.dumps(out))
    reply.flush()
    # Threads or handlers the transformation left behind never delay the exit.
    os._exit(0)


if __name__ == "__main__":
    main()

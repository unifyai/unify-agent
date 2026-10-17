"""The box side of the behaviour comparison (:mod:`.behaviour_overlap`): library functions run on recorded argument
lists, and how each call ended.

Standard library only; the box runs this file by path (``python -I -B behaviour_box.py``) with the job as JSON at
/job/job.json and writes its result to /out/result.json. The library is at /memory (read-only, on the import path).
Nothing the host receives is executed or unpickled: only outcomes and digests leave the box, and the host reads the
result as untrusted input.

Job: {"functions": {item: [module, name]}, "inputs": {input_id: [[recorded_name, json_value], ...]},
      "runs": [[item, input_id], ...], "call_timeout_s": float}
Result: {"results": [[item, input_id, outcome, digest_or_null], ...], "import_errors": {item: error name}}
Outcomes: returned, refused (MemoryInputError anywhere in the error's type MRO), raised, unbindable (the argument
list does not fit the signature), timeout, opaque (returned a value with no canonical form), generator.
"""

import hashlib
import importlib
import inspect
import io
import json
import os
import signal
import sys

MAX_DEPTH = 8


class Opaque(Exception):
    pass


class CallTimeout(BaseException):
    pass


def normalise(v, depth=0):
    if depth > MAX_DEPTH:
        raise Opaque()
    if v is None or isinstance(v, (bool, int, str)):
        return v
    if isinstance(v, float):
        return v if v == v else "nan"
    if isinstance(v, (list, tuple)):
        return [normalise(x, depth + 1) for x in v]
    if isinstance(v, (set, frozenset)):
        return sorted(
            (normalise(x, depth + 1) for x in v),
            key=lambda x: json.dumps(x, sort_keys=True),
        )
    if isinstance(v, dict):
        out = {}
        for k, x in v.items():
            if not isinstance(k, (str, int)) or isinstance(k, bool):
                raise Opaque()
            out[str(k)] = normalise(x, depth + 1)
        return out
    if isinstance(v, (bytes, bytearray)):
        return {"__bytes__": hashlib.sha256(bytes(v)).hexdigest()}
    tolist = getattr(v, "tolist", None)
    if callable(tolist):
        return normalise(tolist(), depth + 1)
    raise Opaque()


def digest(v):
    return hashlib.sha256(
        json.dumps(
            normalise(v), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8"),
    ).hexdigest()


def on_alarm(signum, frame):
    raise CallTimeout()


def build_call(fn, recorded):
    """(args, kwargs) for *fn* from a recorded argument list: a value recorded under the name of one of *fn*'s
    keyword-only parameters goes by keyword, ``name.key`` (a recorded ``**kwargs`` entry) by its key, every other
    value positionally in recorded order. Raises TypeError when the list does not bind.
    """
    sig = inspect.signature(fn)
    kwonly = {
        n for n, p in sig.parameters.items() if p.kind is inspect.Parameter.KEYWORD_ONLY
    }
    has_varkw = any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    args, kwargs = [], {}
    for name, value in recorded:
        value = json.loads(
            json.dumps(value)
        )  # a fresh copy per call: a call may mutate its arguments
        if name in kwonly:
            kwargs[name] = value
        elif "." in name and has_varkw:
            kwargs[name.split(".", 1)[1]] = value
        elif "." in name:
            raise TypeError("keyword argument without a place")
        else:
            args.append(value)
    sig.bind(*args, **kwargs)
    return args, kwargs


def main():
    with open("/job/job.json", encoding="utf-8") as fh:
        job = json.load(fh)
    timeout = float(job.get("call_timeout_s", 2.0))
    sys.path.insert(0, "/memory")
    signal.signal(signal.SIGALRM, on_alarm)
    loaded, errors = {}, {}
    for item, (module, name) in sorted(job["functions"].items()):
        try:
            signal.setitimer(signal.ITIMER_REAL, timeout * 5)
            try:
                fn = getattr(importlib.import_module(module), name)
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
            if not callable(fn):
                raise TypeError("not callable")
            loaded[item] = fn
        except (
            BaseException
        ) as exc:  # noqa: BLE001 - an unimportable function makes no case
            errors[item] = type(exc).__name__
    results = []
    real_stdout, real_stderr = sys.stdout, sys.stderr
    for item, input_id in job["runs"]:
        fn = loaded.get(item)
        if fn is None:
            continue
        outcome, dg = "raised", None
        try:
            args, kwargs = build_call(fn, job["inputs"][input_id])
        except BaseException:  # noqa: BLE001
            results.append([item, input_id, "unbindable", None])
            continue
        sys.stdout, sys.stderr = io.StringIO(), io.StringIO()
        try:
            signal.setitimer(signal.ITIMER_REAL, timeout)
            try:
                value = fn(*args, **kwargs)
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
            if inspect.isgenerator(value):
                outcome = "generator"
            else:
                try:
                    dg, outcome = digest(value), "returned"
                except Opaque:
                    outcome = "opaque"
                except RecursionError:
                    outcome = "opaque"
        except CallTimeout:
            outcome = "timeout"
        except (
            BaseException
        ) as exc:  # noqa: BLE001 - every way a call ends is an outcome
            refused = any(c.__name__ == "MemoryInputError" for c in type(exc).__mro__)
            outcome = "refused" if refused else "raised"
        finally:
            sys.stdout, sys.stderr = real_stdout, real_stderr
        results.append([item, input_id, outcome, dg])
    with open("/out/result.json", "w", encoding="utf-8") as fh:
        json.dump({"results": results, "import_errors": errors}, fh)
    return 0


if __name__ == "__main__":
    os._exit(main())

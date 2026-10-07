"""Procedure memory: executable procedures with recorded cases, verified before every reuse.

A procedure is the exact code that produced a delivered answer (found by slicing, never
written from memory by the model), with the request's literals lifted into parameters.
Every recorded case holds the inputs it read, the parameters and the answer. A procedure
is only offered after (1) its recorded cases replay to the same answers in a fresh
workspace and (2) its run on the new request passes the smell tests.
"""
from __future__ import annotations

import ast
import difflib
import json
import os
import re
import tempfile
import time

from .analysis import file_smells, lift_parameters, normalized, shape, ungrounded_literals, value_smells
from .sandbox import Sandbox
from .world import COMPLETION, Gateway

# Words that never change what is being asked: articles, pronouns, copulas, question words and request verbs.
# Every other word counts, in retrieval and in binding; above all the words that set a boundary, compare,
# quantify, negate or combine (at, or, before, after, since, until, over, under, more, less, than, least, most,
# not, no, without, except, only, between, inclusive, ...), and the prepositions they come with. "At or before
# 2021" and "before 2021" are different requests, so are "with" and "without". There is deliberately no list of
# meaningful words: a word is meaningful unless it is filler.
FILLER = set("a an the is are be been was were what which please me my our us i it its this that these those "
             "give find show compute calculate tell list get".split())


def _normalize(text: str) -> str:
    """Spell negated contractions out, so "don't" carries the same "not" as "do not"."""
    return re.sub(r"n't\b", " not", text, flags=re.I)


def tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", _normalize(text).lower()) if t not in FILLER and len(t) > 1}


def similarity(a: str, b: str) -> float:
    ta, tb = tokens(a), tokens(b)
    return len(ta & tb) / len(ta | tb) if ta and tb else 0.0


def canon(value) -> str:
    def fix(v):
        if isinstance(v, float):
            return round(v, 9)
        if isinstance(v, (list, tuple)):
            return [fix(x) for x in v]
        if isinstance(v, dict):
            return {str(k): fix(x) for k, x in v.items()}
        return v
    return json.dumps(fix(value), sort_keys=True)


def replay(code: str, params: dict, files: dict[str, str], timeout: float = 10.0, recording: dict | None = None,
           upstream: str | None = None) -> dict:
    """Run a procedure in a fresh workspace holding only `files`. Returns the worker reply.
    With a `recording`, the workspace's only API endpoint is a replay gateway that answers from it and
    has no upstream: no call can reach a live world. With `upstream` (offers only), a read-only gateway
    forwards calls that do not change state and refuses the rest. For world procedures the delivered
    value is the session's effects (state-changing calls in order, and the completion answer)."""
    gw = None
    if recording is not None:
        gw = Gateway("replay", recording=recording)
    elif upstream is not None:
        gw = Gateway("readonly", upstream=upstream)
    try:
        return _replay(code, params, files, timeout, gw)
    finally:
        if gw:
            gw.close()


def _replay(code, params, files, timeout, gw) -> dict:
    with tempfile.TemporaryDirectory(prefix="cs-replay-") as d:
        for rel, text in files.items():
            path = os.path.realpath(os.path.join(d, rel))
            if not path.startswith(os.path.realpath(d) + os.sep):
                return {"ok": False, "error": f"refusing to write outside the workspace: {rel}"}
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                f.write(text)
        with Sandbox(d, timeout=timeout, world=gw) as sb:
            sb.set(**params)
            res = sb.run(code)
            if gw is not None:
                res["calls"] = gw.calls
                if gw.completion is not None:
                    res["delivered"] = {"value": gw.effects(), "files": None, "note": "completion"}
                    if gw.mode == "readonly":
                        res["response_text"] = gw.response_text()
                else:
                    res.pop("delivered", None)
            res["files"] = {}
            root = os.path.realpath(d)
            for r in res.get("reads", []):  # only files inside the replay workspace; never host paths
                path = os.path.realpath(os.path.join(root, r))
                if not os.path.isabs(r) and path.startswith(root + os.sep) and os.path.isfile(path):
                    with open(path, encoding="utf-8", errors="replace") as f:
                        res["files"][r] = f.read()
            return res


def _words(text):
    """Words, keeping inner dots (expenses.csv, 1.5) but not a sentence's final period."""
    return re.findall(r"[\w\-/$]+(?:\.[\w\-/$]+)*", text)


def bind(old_request: str, old_params: dict, new_request: str) -> tuple[dict | None, list[str]]:
    """Map each parameter's value in the old request onto the new request by aligning the
    two requests word by word. 'spend on meals in 2026-03' -> 'spend on travel in 2026-04'
    rebinds 'meals' -> 'travel' and '2026-03' -> '2026-04'.
    Returns (parameters or None if any cannot be placed, the differences that fall outside
    parameter positions). A pre-computed answer is only offered when that list is empty."""
    old, new = _words(_normalize(old_request)), _words(_normalize(new_request))
    ops = difflib.SequenceMatcher(None, [t.lower() for t in old], [t.lower() for t in new], autojunk=False).get_opcodes()
    out, covered = {}, set()
    for name, value in old_params.items():
        want = [t.lower() for t in _words(str(value))]
        starts = [i for i in range(len(old)) if [t.lower() for t in old[i:i + len(want)]] == want]
        mapped = None
        for i in starts:
            for tag, i1, i2, j1, j2 in ops:
                if i1 <= i and i + len(want) <= i2 and (tag == "equal" or (tag == "replace" and i2 - i1 == j2 - j1)):
                    if tag == "equal":  # the same words: keep the literal exactly as the code had it
                        mapped = str(value)
                    else:  # new words, written the way the code wrote the old ones
                        old_text = " ".join(old[i:i + len(want)])
                        mapped = " ".join(new[j1 + i - i1: j1 + i - i1 + len(want)])
                        if str(value) == old_text.lower():
                            mapped = mapped.lower()
                        elif str(value) == old_text.upper():
                            mapped = mapped.upper()
                    covered |= set(range(i, i + len(want)))
            if mapped:
                break
        if mapped is None:
            return None, []
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            try:
                mapped = type(value)(mapped)
            except ValueError:
                return None, []
        out[name] = mapped
    extra = []
    for tag, i1, i2, j1, j2 in ops:
        olds = [old[i] for i in range(i1, i2) if i not in covered and old[i].lower() not in FILLER]
        news = [w for w in new[j1:j2] if w.lower() not in FILLER] if not covered & set(range(i1, i2)) else []
        if tag != "equal" and (olds or news):
            extra.append(f"{' '.join(olds) or '(nothing)'} -> {' '.join(news) or '(nothing)'}")
    return out, extra


class ProcedureStore:
    def __init__(self, path: str):
        self.path = path
        self.items: list[dict] = []
        if os.path.exists(path):
            with open(path) as f:
                self.items = json.load(f)

    def save(self):
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.items, f, indent=1)
        os.replace(tmp, self.path)

    def get(self, pid: str) -> dict:
        return next(p for p in self.items if p["id"] == pid)

    # ------------------------------------------------------------ writing

    def record(self, request: str, code: str, files: dict[str, str], answer, run_id: str,
               offered: list[str] = (), timeout: float = 10.0, recording: dict | None = None) -> tuple[str | None, str]:
        """Package a delivered answer's code as a procedure case. Returns (procedure id, why).
        `recording`: the redacted API recording of the session, for world procedures."""
        lifted, params = lift_parameters(code, request)
        res = replay(lifted, params, files, timeout, recording=recording)
        got = res.get("delivered", {}).get("value") if res.get("ok") else None
        if not res.get("ok") or canon(got) != canon(answer):
            return None, f"not stored: replay gave {got!r} ({res.get('error') or 'different answer'})"
        case = {"request": request, "params": params, "files": files, "answer": answer,
                "run": run_id, "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        if recording is not None:
            case["recording"] = recording
        key = normalized(lifted)
        for p in self.items:
            if p["key"] == key and p["status"] != "retired":
                p["cases"].append(case)
                if len({c["request"] for c in p["cases"]}) >= 2 and p["status"] == "candidate":
                    p["status"] = "trusted"
                self.save()
                return p["id"], "added a case to an existing procedure"
        same_shape = [p["id"] for p in self.items if p["status"] != "retired" and shape(p["code"]) == shape(lifted)]
        # a new version when the model rejected an offer and solved differently, or when the code is an
        # existing procedure with other values lifted (a paraphrased return of the same job)
        parent = offered[0] if offered else (same_shape[-1] if same_shape else None)
        lineage = self.get(parent)["lineage"] if parent else f"L{len({p['lineage'] for p in self.items}) + 1}"
        version = 1 + max((p["version"] for p in self.items if p["lineage"] == lineage), default=0)
        pid = f"{lineage}v{version}"
        changes = None if recording is None else any(c["changes_state"] for c in recording["calls"]
                                                     if c["answer"].get("status") == "ok"
                                                     and not c.get("issues_credentials")
                                                     and (c["app"], c["api"]) != COMPLETION)
        self.items.append({"id": pid, "lineage": lineage, "version": version, "supersedes": parent,
                           "world": recording is not None, "changes_state": changes,
                           "status": "candidate", "key": key, "code": lifted, "cases": [case],
                           "negative": [], "stats": {"offered": 0, "accepted": 0, "replay_failures": 0}})
        self.save()
        why = "same shape as " + parent if parent and parent in same_shape else "new"
        return pid, f"new procedure (version {version} of lineage {lineage}; {why})"

    def accepted(self, pid: str, request: str, params: dict, files: dict, answer, run_id: str):
        p = self.get(pid)
        p["stats"]["accepted"] += 1
        p["cases"].append({"request": request, "params": params, "files": files, "answer": answer,
                           "run": run_id, "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        if len({c["request"] for c in p["cases"]}) >= 2 and p["status"] == "candidate":
            p["status"] = "trusted"
        self.save()

    def rejected(self, pid: str, request: str):
        """The model saw this procedure's offer and delivered something else."""
        self.get(pid)["negative"].append(request)
        self.save()

    # ------------------------------------------------------------ reading

    def candidates(self, request: str, k: int = 2, threshold: float = 0.3) -> list[dict]:
        """Lexical retrieval with a negative check: a procedure is skipped when the request is
        at least as close to a request it was wrongly offered for as to its own cases."""
        scored = []
        for p in self.items:
            if p["status"] in ("retired", "stale"):
                continue
            pos = max(similarity(request, c["request"]) for c in p["cases"])
            neg = max((similarity(request, r) for r in p["negative"]), default=0.0)
            if pos >= threshold and pos > neg:
                scored.append((pos, p["version"], p))
        scored.sort(key=lambda t: (-t[0], -t[1]))
        return [p for _, _, p in scored[:k]]

    def verify(self, p: dict, max_cases: int = 2, timeout: float = 10.0) -> bool:
        """Verify before reuse: the newest recorded cases must replay to their answers."""
        for case in p["cases"][-max_cases:]:
            res = replay(p["code"], case["params"], case["files"], timeout, recording=case.get("recording"))
            if not res.get("ok") or canon(res.get("delivered", {}).get("value")) != canon(case["answer"]):
                p["status"] = "stale"
                p["stats"]["replay_failures"] += 1
                self.save()
                return False
        return True

    def try_on(self, p: dict, request: str, workdir_files: dict[str, str], timeout: float = 10.0,
               upstream: str | None = None) -> dict:
        """Bind parameters for the new request and run the procedure on today's files. A world
        procedure that changes state is never pre-run: it is offered as a cell for the model to run.
        A read-only world procedure is pre-run through a read-only gateway (its completion call is
        captured, not sent)."""
        case = max(p["cases"], key=lambda c: similarity(request, c["request"]))
        params, extra = bind(case["request"], case["params"], request)
        if params is None:
            return {"ok": False, "reference": True, "why": "could not map its parameters onto this request"}
        if extra:
            return {"ok": False, "reference": True,
                    "why": f"this request differs from its recorded case beyond the parameters: {'; '.join(extra)}"}
        code = _bound_code(p["code"], params)
        if p.get("world"):
            last = p["cases"][-1]["recording"]["calls"]
            effects = [f"{c['app']}.{c['api']}" for c in last if c["changes_state"] and not c.get("issues_credentials")]
            if p.get("changes_state") or upstream is None:
                p["stats"]["offered"] += 1
                self.save()
                return {"ok": True, "cell": True, "value": None, "params": params, "smells": [], "inputs": {},
                        "outputs": None, "code": code,
                        "preview": "not pre-run: it changes the world; recorded state-changing calls: "
                                   + ", ".join(effects)}
        res = replay(p["code"], params, workdir_files, timeout, upstream=upstream if p.get("world") else None)
        if not res.get("ok") or "delivered" not in res:
            return {"ok": False, "why": (res.get("error") or "delivered nothing").strip().splitlines()[-1]}
        value, files = res["delivered"]["value"], res["delivered"].get("files")
        if p.get("world"):
            answer = value.get("__answer__")
            smells = (value_smells(answer) if answer is not None else []) + \
                ungrounded_literals(code, {"API responses": res.get("response_text", "")})
            p["stats"]["offered"] += 1
            self.save()
            return {"ok": True, "cell": True, "value": value, "params": params, "smells": smells, "inputs": {},
                    "outputs": None, "code": code, "preview": f"pre-run read-only; it would answer {answer!r}"}
        smells = (file_smells(files) if files is not None else value_smells(value)) + \
            ungrounded_literals(_bound_code(p["code"], params), res["files"])
        p["stats"]["offered"] += 1
        self.save()
        return {"ok": True, "value": value, "params": params, "smells": smells, "inputs": res["files"],
                "outputs": files, "code": _bound_code(p["code"], params)}


def _bound_code(code: str, params: dict) -> str:
    """Substitute parameter values back as literals (so the grounding check can see them)."""
    class Sub(ast.NodeTransformer):
        def visit_Name(self, node):
            return ast.copy_location(ast.Constant(params[node.id]), node) if node.id in params else node
    return ast.unparse(Sub().visit(ast.parse(code)))

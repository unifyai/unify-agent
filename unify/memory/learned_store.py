"""Learned skills, harness side: their stored states, their export, their fixture check and the consolidation after
each episode (``UNIFY_MEMORY_LEARNED_SKILLS=on``; :mod:`.learned` holds the families and the skill itself).

* **Where the state lives.** A learned skill's module, its skeleton and its fixtures are ordinary library files on
  ``main`` (``memory/<package>/<module>.py``, ``memory/<package>/tests/data/<module>.learned.json``), landed by the
  writer through the gate. Its fitted state lives on its own ref of the same repository, :data:`STATE_REF`: a flat
  tree with one readable JSON file per skill (``memory.<package>.<module>.json``) and one commit per accepted
  consolidation, so an old state is a commit away, as code is. Only :func:`consolidate_episode` writes the ref;
  the writer's passes never touch it and nothing on ``main`` moves.
* **Export.** The actor's copy gets ``memory/_learned.py`` (:mod:`.learned` byte for byte) and, per learned module
  of the pinned commit, ``.memory/learned/<module path>.state.json`` (the ref's state when it fits the module's
  declaration) and ``.fixtures.json`` (the module's fixture cases; the copy holds no tests).
* **The fixture check** runs model-written code (the module's hooks), so it runs confined
  (:func:`.sandbox_run.run_confined`): the library tree read-only at ``/memory``, :mod:`.learned` and the job at
  ``/learned``, no network, bounded time.
* **Consolidation** (after the episode is recorded, before the next request): for each learned skill the actor
  fitted, its last fit that reproduced its examples is the task state; the stored state takes the Reptile step
  ``p + rate * w(outcome) * (p_task - p)`` (:func:`.learned.consolidated`), then the fixture gate
  (:func:`.learned.gate_states`) accepts it or keeps the old state. Every decision is one row of
  ``<state dir>/learned/consolidations.jsonl``; the fits themselves (the task states) are kept in
  ``<state dir>/learned/fits.jsonl``. Both are harness-only. No model call.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping

from . import learned as L
from .gitio import GitError, Repo
from .redact import KEY_SHAPED

STATE_REF = "refs/learned/state"
#: Where the export puts each learned module's state and fixtures (under the generated ``.memory/``).
EXPORT_DIR = ".memory/learned"
FIXTURE_SUFFIX = ".learned.json"
#: The weight of an episode's fit by its outcome (solved, not solved, unknown) and the step's rate.
WEIGHTS = {True: 1.0, False: 0.3, None: 0.5}
RATE = 0.5
#: The most learned skills one episode consolidates, the most fits a request keeps, and one check's seconds.
MAX_SKILLS_PER_EPISODE = 5
MAX_FITS_PER_REQUEST = 20
CHECK_TIMEOUT_S = 120.0
#: The most learned modules the writer's gate checks in one pass (each check is one confined run).
MAX_LANDING_CHECKS = 10
LEARNED_SOURCE = Path(__file__).with_name("learned.py")


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def module_of(item_id: str) -> str:
    """``memory.<package>.<module>`` of an item id (``...:<name>``) or of a module id."""
    return item_id.split(":", 1)[0]


def module_path(module: str) -> str:
    return module.replace(".", "/") + ".py"


def fixture_path(module: str) -> str:
    parts = module.split(".")
    return "/".join(parts[:-1]) + "/tests/data/" + parts[-1] + FIXTURE_SUFFIX


def state_name(module: str) -> str:
    """The skill's file in the state ref's flat tree."""
    return module + ".json"


def learned_modules(files: Mapping[str, bytes]) -> dict[str, dict]:
    """``module id -> declaration`` for every library module of *files* (path -> bytes) that declares a learned
    skill; a module whose declaration does not parse or build is left out."""
    out: dict[str, dict] = {}
    for path, data in sorted(files.items()):
        parts = path.split("/")
        if (
            len(parts) != 3
            or parts[0] != "memory"
            or not path.endswith(".py")
            or parts[2].startswith("_")
        ):
            continue
        try:
            text = data.decode("utf-8")
            if "LEARNED_FAMILY" not in text:
                continue
            decl = L.declaration(text)
        except (UnicodeDecodeError, SyntaxError, ValueError, L.LearnedError):
            continue
        if decl is not None:
            out[path[:-3].replace("/", ".")] = decl
    return out


# --- the state ref -------------------------------------------------------------------------------------------------


def ref_head(repo: Repo) -> str | None:
    try:
        return repo.run("rev-parse", "--verify", "--quiet", STATE_REF).strip() or None
    except GitError:
        return None


def read_states(repo: Repo, at: str | None = None) -> dict[str, dict]:
    """``module id -> state`` on the state ref (at commit *at*, default its head); {} when there is none."""
    head = at or ref_head(repo)
    if head is None:
        return {}
    out: dict[str, dict] = {}
    for line in repo.run("ls-tree", head).splitlines():
        meta, _, name = line.partition("\t")
        if not name.endswith(".json"):
            continue
        try:
            out[name[: -len(".json")]] = json.loads(
                repo.run("cat-file", "blob", meta.split()[2]),
            )
        except (GitError, ValueError, IndexError):
            continue
    return out


def write_state(
    repo: Repo,
    module: str,
    state: dict,
    message: str,
    expected: str | None,
) -> str:
    """Commit *state* as the skill's file on the state ref (one commit; the ref moves only from *expected*)."""
    entries: dict[str, str] = {}
    if expected is not None:
        for line in repo.run("ls-tree", expected).splitlines():
            meta, _, name = line.partition("\t")
            entries[name] = meta.split()[2]
    entries[state_name(module)] = repo.run(
        "hash-object",
        "-w",
        "--stdin",
        input=L.state_text(state),
    ).strip()
    tree = repo.run(
        "mktree",
        input="".join(
            f"100644 blob {sha}\t{name}\n" for name, sha in sorted(entries.items())
        ),
    ).strip()
    args = ["-c", "commit.gpgsign=false", "commit-tree", tree]
    if expected is not None:
        args += ["-p", expected]
    sha = repo.run(*args, "-F", "-", input=message.strip() + "\n").strip()
    repo.run("update-ref", STATE_REF, sha, expected or "0" * 40)
    return sha


# --- the export ----------------------------------------------------------------------------------------------------


def export_files(repo: Repo, files: Mapping[str, bytes]) -> dict[str, bytes]:
    """The generated files of the actor's copy for the learned skills of *files* (the pinned commit's path -> bytes):
    the families module and, per learned module, its state (when the ref holds a compatible one) and fixtures.
    """
    out: dict[str, bytes] = {"memory/_learned.py": LEARNED_SOURCE.read_bytes()}
    modules = learned_modules(files)
    if not modules:
        return out
    states = read_states(repo)
    for module, decl in modules.items():
        base = f"{EXPORT_DIR}/{module_path(module)}"
        state = states.get(module)
        if L.compatible(state, decl["family"], decl["options"]):
            out[base[:-3] + ".state.json"] = L.state_text(state).encode("utf-8")
        fx = files.get(fixture_path(module))
        if fx is not None:
            out[base[:-3] + ".fixtures.json"] = fx
    return out


# --- the confined fixture check ------------------------------------------------------------------------------------


def run_check(
    tree: Path,
    module: str,
    item: str,
    states: Mapping[str, dict | None],
) -> dict:
    """Each state's per-case results on the module's fixtures (:func:`.learned.check_cases`), from a confined run on
    the library *tree*; ``{"error": ...}`` when the run fails or gives no result line.
    """
    from .sandbox_run import PYTHON, run_confined

    stage = Path(tempfile.mkdtemp(prefix="memory-learned-"))
    try:
        shutil.copyfile(LEARNED_SOURCE, stage / "learned.py")
        job = {
            "module": module,
            "item": item,
            "source_path": f"/memory/{module_path(module)}",
            "fixtures_path": f"/memory/{fixture_path(module)}",
            "states": dict(states),
        }
        (stage / "job.json").write_text(json.dumps(job), encoding="utf-8")
        r = run_confined(
            [str(PYTHON), "/learned/learned.py", "check", "/learned/job.json"],
            ro={Path(tree): "/memory", stage: "/learned"},
            cwd="/memory",
            timeout_s=CHECK_TIMEOUT_S,
            env={"PYTHONPATH": "/memory", "PYTHONHASHSEED": "0"},
        )
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    if r.timed_out:
        return {"error": "timeout"}
    lines = [ln for ln in (r.stdout or "").splitlines() if ln.startswith(L.CHECK_MARK)]
    if r.returncode != 0 or len(lines) != 1:
        # exactly one result line: a second one could only come from the module's own code
        return {
            "error": f"exit {r.returncode}, {len(lines)} result lines: {(r.stderr or '')[-300:]}",
        }
    try:
        return json.loads(lines[-1][len(L.CHECK_MARK) :])
    except ValueError:
        return {"error": "malformed result line"}


def check_landing(tree: Path, module: str, decl: dict) -> tuple[bool, str]:
    """The writer's gate for one learned module of a candidate *tree*: its fixture file holds at least one case, and
    from its initial state every case's examples and queries are reproduced after the fit.
    """
    fx = Path(tree) / fixture_path(module)
    try:
        cases = json.loads(fx.read_text(encoding="utf-8")).get("cases")
    except (OSError, ValueError, AttributeError):
        return (
            False,
            f"{fixture_path(module)} is missing or not JSON with a 'cases' list",
        )
    if not isinstance(cases, list) or not cases:
        return False, f"{fixture_path(module)} holds no case"
    item = module + (":" + decl["skeleton"] if decl.get("skeleton") else "")
    got = run_check(tree, module, item, {"initial": None})
    if "error" in got:
        return False, f"the fixture check did not run: {got['error']}"[:300]
    rows = got.get("initial") or []
    bad = [i for i, r in enumerate(rows) if not r.get("passed")]
    if bad:
        errs = "; ".join(
            f"case {i}: {rows[i].get('error') or 'not reproduced after the fit'}"
            for i in bad[:3]
        )
        return False, f"{len(bad)} of {len(rows)} fixture case(s) fail: {errs}"[:300]
    return True, "ok"


# --- the fits a request made, and the consolidation after it -------------------------------------------------------


def take_fits(acc: list, drained: Any) -> None:
    """Add one cell's drained fit records (``calls.learned`` of the audit drain) to *acc*, at most
    :data:`MAX_FITS_PER_REQUEST`."""
    calls = drained.get("calls") if isinstance(drained, dict) else None
    rows = calls.get("learned") if isinstance(calls, dict) else None
    for row in rows if isinstance(rows, list) else []:
        if len(acc) >= MAX_FITS_PER_REQUEST:
            return
        if (
            isinstance(row, dict)
            and isinstance(row.get("item"), str)
            and isinstance(row.get("state"), str)
            and isinstance(row.get("reproduced"), bool)
            and isinstance(row.get("steps"), int)
            and KEY_SHAPED.search(row["state"])
            is None  # never a key-shaped string (the audit hook refuses them too)
        ):
            acc.append(
                {k: row[k] for k in ("item", "state", "reproduced", "steps")},
            )


def _append(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")


def _case_sums(rows: list[dict]) -> tuple[int, float | None]:
    passed = sum(1 for r in rows if r.get("passed"))
    losses = [r.get("loss") for r in rows]
    return passed, (
        round(sum(losses), 6) if losses and all(x is not None for x in losses) else None
    )


def consolidate_episode(
    memory_dir: Path,
    state_dir: Path,
    episode_id: str,
    fits: list[dict],
    solved: bool | None,
) -> list[dict]:
    """Record the episode's fits and consolidate each learned skill it fitted (module docstring). Returns the
    decision rows (also appended to ``consolidations.jsonl``)."""
    at = _now()
    learned_dir = Path(state_dir) / "learned"
    _append(
        learned_dir / "fits.jsonl",
        [
            {
                "episode_id": episode_id,
                "at": at,
                "item": f["item"],
                "reproduced": f["reproduced"],
                "steps": f["steps"],
                "state_sha256": hashlib.sha256(f["state"].encode("utf-8")).hexdigest(),
                "state": f["state"],
            }
            for f in fits
        ],
    )
    last: dict[str, dict] = {}
    for f in fits:  # per skill: its last fit that reproduced its examples
        mod = module_of(f["item"])
        if f["reproduced"] and f["state"]:
            last[mod] = f
        else:
            last.setdefault(mod, None)  # type: ignore[arg-type]
    weight = WEIGHTS.get(solved if isinstance(solved, bool) else None, WEIGHTS[None])
    repo = Repo(Path(memory_dir))
    rows: list[dict] = []
    for module in sorted(last)[:MAX_SKILLS_PER_EPISODE]:
        f = last[module]
        row: dict[str, Any] = {
            "episode_id": episode_id,
            "at": at,
            "module": module,
            "solved": solved if isinstance(solved, bool) else None,
            "weight": weight,
            "rate": RATE,
            "accepted": False,
        }
        rows.append(row)
        if f is None:
            row["reason"] = "no_reproduced_fit"
            continue
        try:
            task = json.loads(f["state"])
            with repo.temp_checkout("main") as tree:
                path = Path(tree) / module_path(module)
                decl = (
                    L.declaration(path.read_text(encoding="utf-8"))
                    if path.is_file()
                    else None
                )
                if decl is None or not L.compatible(
                    task,
                    decl["family"],
                    decl["options"],
                ):
                    row["reason"] = "module_changed"
                    continue
                expected = ref_head(repo)
                stored = read_states(repo, expected).get(module)
                old = (
                    stored
                    if L.compatible(stored, decl["family"], decl["options"])
                    else L.initial_state(
                        decl["family"],
                        decl["options"],
                        decl["initial"],
                    )
                )
                new = L.consolidated(
                    old,
                    task,
                    weight,
                    RATE,
                    {"episode": episode_id, "at": at},
                )
                got = run_check(tree, module, f["item"], {"old": old, "new": new})
                if "error" in got:
                    row["reason"] = "check_error"
                    row["error"] = str(got["error"])[:300]
                    continue
                ok, reason = L.gate_states(got.get("old") or [], got.get("new") or [])
                row["cases"] = len(got.get("old") or [])
                row["passed_before"], row["loss_before"] = _case_sums(
                    got.get("old") or [],
                )
                row["passed_after"], row["loss_after"] = _case_sums(
                    got.get("new") or [],
                )
                row["steps_before"] = [r.get("steps") for r in got.get("old") or []]
                row["steps_after"] = [r.get("steps") for r in got.get("new") or []]
                row["reason"] = reason
                row["ref_before"] = expected
                if ok:
                    row["ref_after"] = write_state(
                        repo,
                        module,
                        new,
                        f"learned: {module} consolidated from episode {episode_id}\n\n"
                        f"weight {weight}, rate {RATE}; fixture cases passed {row['passed_before']} -> "
                        f"{row['passed_after']}, loss {row['loss_before']} -> {row['loss_after']}",
                        expected,
                    )
                    row["accepted"] = True
        except (
            Exception
        ) as exc:  # noqa: BLE001 - a consolidation never stops a request; the row says why
            row["reason"] = "error"
            row["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    _append(learned_dir / "consolidations.jsonl", rows)
    return rows

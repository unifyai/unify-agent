"""No world's vocabulary in what any agent reads: the guard grep over every text the harness gives a model.

The general harness names no world (no benchmark, no environment, no kind of task it was tuned on). This test
runs one case-insensitive whole-word grep (:data:`GUARD`) over every text an agent can read:

* the exported ``memory`` package, file by file, whole (the actor reads its help and can read its source):
  ``memory/__init__.py`` (with ``memory.ask``), its typed stub, its contract runtime, the shape matcher, and the
  v2 helper;
* every file of the test kit ``memlab``, whole, as both stagings write it (the writer's box and the library's
  tests);
* every string of the modules that hold the tool specs and docstrings (``tool_registry``) and every role's
  prompt (:data:`GUARD_MODULES`): docstrings, prompt texts, tool descriptions, refusal texts;
* the prompts and tool texts as rendered for each role and switch combination (:func:`_rendered`), so text
  composed at run time is read too.

A hit passes only where a line of the reviewed allow-list (:data:`ALLOW_FILE`) covers it: a phrase holding the
word, for a scope of sources, with the reason it is generic English. Every line must still cover a hit, so the
list cannot grow stale or wide. A module that gains model-facing text joins :data:`GUARD_MODULES`.
"""

from __future__ import annotations

import io
import itertools
import re
import tokenize
from dataclasses import dataclass
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
#: The guard grep (the redesign plan's C1 test 1), case-insensitive, whole words.
GUARD = re.compile(
    r"\b(grid|grids|puzzle|demo|demos|task|tasks|airline|booking|crafter|arc|poker|hand|game|colleague|office|"
    r"stencil|colour|color)\b",
    re.IGNORECASE,
)
#: The reviewed allow-list: ``scope<TAB>phrase<TAB>reason`` per line (``#`` starts a comment).
ALLOW_FILE = Path(__file__).with_name("no_world_vocabulary_allowlist.tsv")
#: The modules whose every string is model-facing text (tool specs, docstrings, role prompts and briefs).
GUARD_MODULES = (
    "unify/common/tool_registry.py",
    "unify/memory/library_prompts.py",
    "unify/memory/writer_pass.py",
    "unify/memory/curate.py",
    "unify/memory/qa.py",
    "unify/memory/helpers.py",
    "unify/memory/librarian.py",
    "unify/memory/library_ask.py",
    "unify/memory/agent_profiles.py",
    "unify/memory/messaging.py",
    "unify/memory/integration/fork.py",
    "unify/memory/integration/prompt.py",
    "unify/memory/integration/hooks.py",
    "unify/actor/prompt_builders.py",
    "unify/actor/library_switch.py",
)


@dataclass(frozen=True)
class Hit:
    source: str
    word: str
    start: int
    end: int
    context: str


@dataclass(frozen=True)
class Allowed:
    scope: str
    phrase: str
    reason: str

    def covers(self, source: str, text: str, start: int, end: int) -> bool:
        if self.scope != "*" and not source.startswith(self.scope):
            return False
        lo = max(0, start - len(self.phrase))
        for m in re.finditer(re.escape(self.phrase), text[lo : end + len(self.phrase)]):
            if lo + m.start() <= start and end <= lo + m.end():
                return True
        return False


def allow_list() -> list[Allowed]:
    out = []
    for n, line in enumerate(ALLOW_FILE.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split("\t")
        assert len(parts) == 3 and all(
            p.strip() for p in parts
        ), f"{ALLOW_FILE.name}:{n}: scope, phrase, reason"
        scope, phrase, reason = parts
        assert GUARD.search(
            phrase
        ), f"{ALLOW_FILE.name}:{n}: the phrase holds no guarded word"
        out.append(Allowed(scope, phrase, reason))
    return out


def _strings(source: str) -> str:
    """Every string literal of a module (f-string parts included), one per line, as written."""
    middle = getattr(tokenize, "FSTRING_MIDDLE", None)
    return "\n".join(
        tok.string
        for tok in tokenize.generate_tokens(io.StringIO(source).readline)
        if tok.type == tokenize.STRING or (middle is not None and tok.type == middle)
    )


def _package() -> dict[str, str]:
    from unify.memory import library_export

    return {
        "package:memory/__init__.py": library_export.package_source(
            librarian=True
        ).decode("utf-8"),
        "package:memory/__init__.py (no librarian)": library_export.package_source().decode(
            "utf-8"
        ),
        "package:memory/__init__.pyi": library_export.stub_source(
            librarian=True
        ).decode("utf-8"),
        "package:memory/_contract.py": library_export.CONTRACT_SOURCE.read_text(
            encoding="utf-8"
        ),
        "package:matcher": library_export.matcher_source().decode("utf-8"),
        "package:memory_helper.py": (ROOT / "unify/memory/memory_helper.py").read_text(
            encoding="utf-8"
        ),
    }


def _memlab(tmp: Path) -> dict[str, str]:
    from unify.memory import testkit, writer_pass

    testkit.stage(
        tmp / "kit",
        [],
        has_blob=lambda _sha: False,
        read_blob=lambda _sha: b"",
        blob_size=lambda _sha: 0,
    )
    writer_pass._stage_memlab(tmp / "box")
    out = {}
    for root in (tmp / "kit", tmp / "box"):
        for p in sorted(root.rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts:
                out[f"memlab:{root.name}/{p.relative_to(root)}"] = p.read_text(
                    encoding="utf-8"
                )
    assert any(k.endswith("memlab/replay.py") for k in out) and any(
        k.endswith("box/replay.py") for k in out
    )
    return out


def _modules() -> dict[str, str]:
    return {
        f"module:{m}": _strings((ROOT / m).read_text(encoding="utf-8"))
        for m in GUARD_MODULES
    }


def _rendered(monkeypatch) -> dict[str, str]:
    import json

    from unify.common import in_code_switch
    from unify.common import tool_registry as R
    from unify.memory import curate, helpers, librarian
    from unify.memory import library_prompts as lp
    from unify.memory import writer_pass
    from unify.memory.integration import fork

    out: dict[str, str] = {}
    flags = sorted(
        {s.requires for s in R.registry().values() if s.requires}
        | {"pool", "similar", "helper"}
    )
    for role in sorted(R.OLD_ROLES):
        for n in range(len(flags) + 1):
            for chosen in itertools.combinations(flags, n):
                f = frozenset(chosen)
                key = f"rendered:tools:{role}:{','.join(chosen)}"
                out[key] = (
                    json.dumps(R.tool_json(role, f)) + "\n" + R.tools_section(role, f)
                )
    every = frozenset(flags)
    for agent in R.AGENTS:
        out[f"rendered:preamble:{agent}"] = R.preamble(
            agent, {"environment": "Your box.", "flags": every}
        )
        for name in R.allow_list(agent, every):
            out[f"rendered:help:{agent}:{name}"] = R.help_text(name, agent)
    out["rendered:refusals"] = json.dumps(R.refusal_codes()) + json.dumps(
        R.GENERAL_REFUSALS
    )
    if hasattr(R, "COMMON_SECTION"):
        out["rendered:common_section"] = R.COMMON_SECTION
    for in_code in (False, True):
        monkeypatch.setattr(in_code_switch, "tools_in_code", lambda v=in_code: v)
        tag = "code" if in_code else "json"
        names = (
            "staging",
            "pool",
            "helper",
            "similar",
            "compact",
            "finish_capped",
            "edit_tools",
        )
        for values in itertools.product((False, True), repeat=len(names)):
            kw = dict(zip(names, values))
            out[f"rendered:write:{tag}:{values}"] = lp.write_brief_now(**kw)
        for values in itertools.product((False, True), repeat=3):
            kw = dict(zip(("compact", "finish_capped", "edit_tools"), values))
            out[f"rendered:curate:{tag}:{values}"] = lp.curate_brief_now(**kw)
        for similar in (False, True):
            out[f"rendered:helper:{tag}:{similar}"] = helpers.helper_brief(similar)
        out[f"rendered:fork:{tag}"] = fork.fork_message("`request`: (a request)")
        out[f"rendered:guide:{tag}"] = (
            lp.guide_library_mode()
            + "\n"
            + lp.guide_pointer()
            + "\n"
            + lp.guide_library()
        )
    for values in itertools.product((False, True), repeat=3):
        kw = dict(zip(("docstrings", "catalogue", "soft_budget"), values))
        out[f"rendered:writer_system:{values}"] = writer_pass.writer_system(**kw)
    out["rendered:librarian"] = librarian.system_prompt() + "\n" + lp.GUIDE_LIBRARIAN
    out["rendered:curate_system"] = curate.curate_system()
    out["rendered:guide"] = lp.GUIDE_LIBRARY
    return out


def texts(tmp: Path, monkeypatch) -> dict[str, str]:
    return {**_package(), **_memlab(tmp), **_modules(), **_rendered(monkeypatch)}


def hits(texts_by_source: dict[str, str]) -> list[Hit]:
    out = []
    for source, text in texts_by_source.items():
        for m in GUARD.finditer(text):
            lo, hi = max(0, m.start() - 40), min(len(text), m.end() + 40)
            out.append(Hit(source, m.group(0), m.start(), m.end(), text[lo:hi]))
    return out


def test_no_text_an_agent_reads_names_a_world(tmp_path, monkeypatch):
    allowed = allow_list()
    by_source = texts(tmp_path, monkeypatch)
    assert len(by_source) > 300  # every source above was read
    used: set[Allowed] = set()
    leaks = []
    for h in hits(by_source):
        cover = {
            a
            for a in allowed
            if a.covers(h.source, by_source[h.source], h.start, h.end)
        }
        if not cover:
            leaks.append(f"{h.source}: {h.word!r} in {h.context!r}")
        used |= cover
    assert not leaks, "world vocabulary outside the allow-list:\n" + "\n".join(
        leaks[:200]
    )
    stale = [f"{a.scope}\t{a.phrase}" for a in allowed if a not in used]
    assert (
        not stale
    ), "allow-list lines that cover nothing (remove them):\n" + "\n".join(stale)


def test_the_replaced_examples_name_no_world():
    """The examples the actor sees in the package's errors and help (``_GET_EXAMPLE``, the NotFound fallbacks and
    the docstrings of ``get``, ``items`` and ``show``) and the test kit's usage example are neutral.
    """
    from unify.memory import inputs, library_helper

    assert library_helper._GET_EXAMPLE == "memory.get('notes/reports/summary.md')"
    for fn in (library_helper.get, library_helper.items, library_helper.show):
        assert not GUARD.search(fn.__doc__ or ""), fn.__name__
    assert not GUARD.search(inputs.__doc__ or "")
    assert not GUARD.search(library_helper.wanted_key.__doc__ or "")


def test_the_allow_list_is_well_formed():
    allowed = allow_list()
    assert len(allowed) == len({(a.scope, a.phrase) for a in allowed})
    for a in allowed:
        assert a.scope == "*" or a.scope.split(":", 1)[0] in (
            "package",
            "memlab",
            "module",
            "rendered",
        ), a


@pytest.mark.parametrize(
    "text, words",
    [
        ("Read notes/grid/stencil.md first.", ["grid", "stencil"]),
        ("the ARC demo pairs", ["ARC", "demo"]),
        ("a task id", ["task"]),
        ("arcs, tasked, handled, colours and griddle", []),
    ],
)
def test_the_guard_matches_whole_words_in_any_case(text, words):
    assert [m.group(0) for m in GUARD.finditer(text)] == words


# On a keyless host: print every hit and every unused allow-list line, to review the allow-list.
if __name__ == "__main__":
    import sys
    import tempfile

    class _Patch:
        def setattr(self, obj, name, value):
            setattr(obj, name, value)

    with tempfile.TemporaryDirectory() as d:
        found = texts(Path(d), _Patch())
        allowed = allow_list() if ALLOW_FILE.exists() else []
        seen: set[Allowed] = set()
        for h in hits(found):
            cover = {
                a
                for a in allowed
                if a.covers(h.source, found[h.source], h.start, h.end)
            }
            seen |= cover
            state = "ok" if cover else "LEAK"
            sys.stdout.write(f"{state}\t{h.source}\t{h.word}\t{h.context!r}\n")
        for a in allowed:
            if a not in seen:
                sys.stdout.write(f"STALE\t{a.scope}\t{a.phrase}\n")
        sys.stdout.write(f"SOURCES\t{len(found)}\n")

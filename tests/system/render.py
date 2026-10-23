"""Every static text an agent of a memory run is shown, rendered from the tree's own functions, per role and mode.

The keys are ``<role>/<mode>/<what>``: the actor and a teammate in ``off``, ``on`` and ``in_episode`` (their flags
are the ones :func:`unify.actor.tool_handlers.actor_flags` gives each mode), and the memory roles (writer, writer's
helper, librarian, fork, kinship judge) in ``on``. ``what`` is ``tools_list``, ``help/<tool>``,
``refusal/<tool>/<form>`` (a template filled with ``<field>`` markers), ``common_section``, a brief, a
guide line or a card template. Rendering runs nothing of a model and opens no store.

Used by the static guards: the rendered goldens (J2), the claims registry (J1) and the shared-constant lint (J3).
"""

from __future__ import annotations

import contextlib
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterator

ACTOR_ROLES = ("actor", "teammate")
MODES = ("off", "on", "in_episode")
MEMORY_ROLES = ("writer", "helper", "librarian", "fork", "judge")


class _Marks(dict):
    def __missing__(self, key: str) -> str:
        return f"<{key}>"


@contextlib.contextmanager
def memory_mode(mode: str) -> Iterator[None]:
    """``UNIFY_MEMORY`` set to *mode* for the block (the settings object only; nothing starts)."""
    from unify.settings import SETTINGS

    old = SETTINGS.UNIFY_MEMORY
    SETTINGS.UNIFY_MEMORY = mode
    try:
        yield
    finally:
        SETTINGS.UNIFY_MEMORY = old


def _template(text: str) -> str:
    try:
        return text.format_map(_Marks())
    except (ValueError, IndexError):
        return text


def _tools(out: dict, prefix: str, agent: str, flags: frozenset, memory: bool) -> None:
    from unify.common import tool_registry as R

    names = R.allow_list(agent, flags)
    out[f"{prefix}/tools_list"] = R.list_text(names, agent, memory=memory)
    reg = R.registry()
    for nm in names:
        out[f"{prefix}/help/{nm}"] = R.help_text(nm, agent, memory=memory)
        if nm not in reg:
            continue
        # the refusals this role is given: its own texts where the tool has them for it, else the shared ones
        own, texts = R._own_text(reg[nm], "refusals", agent, memory)
        table = texts if own and isinstance(texts, dict) else reg[nm].refusals
        for form, text in sorted(table.items()):
            out[f"{prefix}/refusal/{nm}/{form}"] = _template(text)
    for form, text in sorted(R.GENERAL_REFUSALS.items()):
        out[f"{prefix}/refusal/general/{form}"] = _template(text)


def _memory_help(librarian: bool = True) -> str:
    """``help(memory)`` and the help of its calls, as a cell shows them (the generated package, in a child)."""
    from unify.memory import library_export as LE

    with tempfile.TemporaryDirectory() as d:
        pkg = Path(d) / "memory"
        pkg.mkdir()
        (pkg / "__init__.py").write_bytes(LE.package_source(librarian=librarian))
        (pkg / "_contract.py").write_bytes(LE.CONTRACT_SOURCE.read_bytes())
        (pkg / "__init__.pyi").write_bytes(LE.stub_source(librarian=librarian))
        code = (
            f"import sys\nsys.path.insert(0, {d!r})\n"
            "import pydoc, memory\n"
            "print(pydoc.plain(pydoc.render_doc(memory)))\n"
            "for n in ('get', 'items', 'request', 'ask', 'Item', 'Items'):\n"
            "    o = getattr(memory, n, None)\n"
            "    print('\\n######## help(memory.%s) ########' % n)\n"
            "    print(pydoc.plain(pydoc.render_doc(o)) if o is not None else 'ABSENT')\n"
        )
        r = subprocess.run([sys.executable, "-I", "-c", code], cwd=d, capture_output=True, text=True, timeout=120)
        assert r.returncode == 0, r.stderr[-2000:]
        return r.stdout.replace(d, "<pkg>")


def render_all() -> dict[str, str]:
    from unify.actor import tool_handlers as TH
    from unify.common import tool_registry as R
    from unify.memory import helpers as HP
    from unify.memory import kinship as KN
    from unify.memory import librarian as LB
    from unify.memory import library_helper as LH
    from unify.memory import library_prompts as LP
    from unify.memory.integration import fork as FK
    from unify.memory.integration import hooks as HK
    from unify.memory.integration import prompt as IP

    out: dict[str, str] = {}
    for mode in MODES:
        with memory_mode(mode):
            memory = mode != "off"
            for agent in ACTOR_ROLES:
                flags = TH.actor_flags(agent)
                prefix = f"{agent}/{mode}"
                _tools(out, prefix, agent, flags, memory)
                out[f"{prefix}/common_section"] = R.common_section(memory)
            if memory:
                out[f"actor/{mode}/guide_librarian"] = LP.GUIDE_LIBRARIAN
                out[f"actor/{mode}/library_line"] = HK.LIBRARY_LINE
                out[f"actor/{mode}/start_head"] = LB.START_HEAD
                out[f"actor/{mode}/memory_section"] = IP.render_memory_section(Path("/memory-library"))[0]
    with memory_mode("on"):
        _tools(out, "writer/on", "writer", frozenset({"helper", "similar"}), True)
        out["writer/on/brief"] = LP.write_brief_now(helper=True, similar=True)
        _tools(out, "helper/on", "helper", HP.helper_flags(True), True)
        out["helper/on/brief"] = HP.helper_brief(similar=True)
        _tools(out, "librarian/on", "librarian", frozenset(), True)
        out["librarian/on/system_prompt"] = LB.system_prompt(evidence=True)
        out["librarian/on/question"] = LB.question_message("<question>", rows=3)
        out["librarian/on/evidence_question"] = LB.evidence_question_message("<question>", "<last message>", rows=3)
        out["fork/on/instruction"] = FK.fork_instruction()
        out["fork/on/segment_instruction"] = FK.segment_instruction(1, 4)
        out["judge/on/brief"] = KN.BRIEF
        out["cards/on/by_hand"] = LH.BY_HAND.format(cid="capture:<id>")
        out["memory_help/on/package"] = _memory_help(librarian=True)
    return out


def sentences(text: str) -> list[str]:
    """*text* split into sentences: at a line break, and after ``.``, ``!`` or ``?`` followed by a space; whitespace
    collapsed, empty pieces dropped."""
    import re

    out = []
    for line in text.splitlines():
        for piece in re.split(r"(?<=[.!?])\s+(?=[A-Z`(\"'*])", line):
            piece = " ".join(piece.split()).strip("-* ")
            if piece:
                out.append(piece)
    return out


#: The claims registry's scope (guard J1): sentences about the gate, library and input paths, cards and recorded
#: procedures, the librarian and its answer; and, in a memory-off actor's or teammate's texts, any sentence naming memory.
SCOPE = {
    "gate": r"\bgate\b",
    "paths": r"(?<![\w.])/(memory-library|memory|inputs|workspace|outputs)\b",
    "cards": r"\bcards?\b|recorded procedure|\bcapture",
    "ask": r"memory\.ask|\blibrarian\b|fallback|best fit",
}
OFF_MEMORY = r"memor|librar|writer|fork|captur|/inputs|attention"


def scoped(texts: dict[str, str]) -> dict[str, dict]:
    """Every in-scope sentence of *texts* (keys as :func:`render_all` gives them): ``{sentence: {"keys", "topics"}}``."""
    import re

    out: dict[str, dict] = {}
    for key, text in texts.items():
        role, mode = key.split("/")[:2]
        for s in sentences(text):
            topics = [t for t, rx in SCOPE.items() if re.search(rx, s)]
            if not topics and mode == "off" and role in ACTOR_ROLES and re.search(OFF_MEMORY, s, re.I):
                topics = ["off"]
            if topics:
                row = out.setdefault(s, {"keys": [], "topics": []})
                row["keys"].append(key)
                row["topics"] = sorted(set(row["topics"]) | set(topics))
    return out


if __name__ == "__main__":  # dump: python -m tests.system.render OUT.json
    import json

    Path(sys.argv[1]).write_text(json.dumps(render_all(), indent=1, sort_keys=True))

"""The calls a prompt shows, taken from its text and made ready to run as written in an agent's own sandbox.

Used by the prompt-to-runtime contract test (``test_prompt_runtime_contract_from_spec.py``). Nothing here imports the
harness:

* :func:`spans` takes every inline code span and fenced block of a prompt, with the sentence around it;
* :func:`prepare` fills each placeholder by one fixed rule (a table the test owns), decides whether the span is a
  call (a call or an attribute read on a free name, an import, or a bare name the sentence says is bound), and
  turns ``tools.<name>(...)`` into that tool's first registry example;
* :func:`contract_cell` runs the prepared spans one by one in the cell, each in its own ``try``, and prints one
  ``SPAN {...}`` line per span; :func:`results` reads those lines back;
* :func:`failures` says which results break the contract.
"""

from __future__ import annotations

import ast
import builtins
import json
import re
import textwrap
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

FENCE = re.compile(r"(`{3,})[^\n]*\n(.*?)\n?\1", re.S)
INLINE = re.compile(r"`([^`\n]+)`")
ANGLE = re.compile(r"<([A-Za-z][A-Za-z0-9_ -]*)>")
#: The sentence words that say a name is in every cell already (a bare name span next to one is run as written).
BINDING_WORDS = (
    "importable",
    "already available",
    "already bound",
    "already imported",
    "no import needed",
    "is bound",
    "are bound",
)
#: The namespaces and helpers the harness binds (design B.2); a free name among these is never a placeholder.
HARNESS_NAMES = frozenset(
    {
        "tools",
        "memory",
        "help",
        "display",
        "query_llm",
        "list_llms",
        "unillm",
        "run_coro_sync",
        "request_clarification",
        "ModelRequestError",
        "SteerableToolHandle",
        "primitives",
    },
)
#: The calls that end the agent's work: run last, in a cell of their own.
END_TOOLS = frozenset({"finish", "done"})
_SIGNATURE = re.compile(
    r"unexpected keyword argument|missing \d+ required|takes \d+ positional|takes from \d+ to \d+ positional"
    r"|positional arguments? but|required (keyword|positional)",
)
SPAN_MARK = "SPAN "
NAMESPACE_MARK = "NAMESPACE "

#: The first cell of every contract run: the names bound before any cell (globals and the cell's builtins), as one
#: line. Only names every sandbox's builtins keep are used (the actor's worker has no ``globals`` or ``exec``).
NAMESPACE_CELL = (
    "_v = dict(vars())\n"
    "_bi = _v.get('__builtins__')\n"
    "_names = set(k for k in _v if not k.startswith('_')) | set(_bi if isinstance(_bi, dict) else dir(_bi))\n"
    "import json as _j\n"
    "print('NAMESPACE ' + _j.dumps(sorted(_names)))\n"
)


@dataclass
class Span:
    text: str
    sentence: str
    fenced: bool = False


@dataclass
class Prepared:
    k: int
    span: str
    code: str
    end: bool = False


@dataclass
class Plan:
    """What a prompt's spans come to: the calls to run, the spans that do not parse (for the reviewed list) and the
    contract's failures found before anything runs (a placeholder with no rule)."""

    calls: list[Prepared] = field(default_factory=list)
    unparsed: list[str] = field(default_factory=list)
    no_rule: list[str] = field(default_factory=list)


def _sentence(text: str, start: int, end: int) -> str:
    left = max(text.rfind(". ", 0, start), text.rfind("\n", 0, start))
    rights = [i for i in (text.find(". ", end), text.find("\n", end)) if i != -1]
    right = min(rights) if rights else len(text)
    return text[left + 1 : right]


def spans(text: str) -> list[Span]:
    """Every fenced block and inline code span of *text*, in order, each with its sentence."""
    out: list[Span] = []
    rest = text
    for m in FENCE.finditer(text):
        out.append(Span(m.group(2), "", fenced=True))
    rest = FENCE.sub(lambda m: " " * len(m.group(0)), text)
    for m in INLINE.finditer(rest):
        out.append(Span(m.group(1).strip(), _sentence(rest, m.start(), m.end())))
    return out


def _in_quotes(code: str, pos: int) -> bool:
    line_start = code.rfind("\n", 0, pos) + 1
    before = code[line_start:pos]
    return (before.count('"') - before.count('\\"')) % 2 == 1 or (before.count("'") - before.count("\\'")) % 2 == 1


def fill(code: str, angle_rules: dict[str, str]) -> tuple[str, list[str]]:
    """*code* with each ``<placeholder>`` replaced by its rule's text (inside quotes) or its quoted text (outside),
    and the placeholders that have no rule (each replaced by an identifier so the span still parses)."""
    missing: list[str] = []
    out, last = [], 0
    for m in ANGLE.finditer(code):
        key = m.group(1).strip()
        out.append(code[last : m.start()])
        if key in angle_rules:
            value = angle_rules[key]
            out.append(value if _in_quotes(code, m.start()) else repr(value))
        else:
            missing.append(f"<{key}>")
            out.append("placeholder_without_rule")
        last = m.end()
    out.append(code[last:])
    return "".join(out), missing


def _names(tree: ast.AST) -> tuple[set[str], set[str]]:
    """(names read, names bound) in *tree*."""
    read, bound = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            (read if isinstance(node.ctx, ast.Load) else bound).add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, ast.arg):
            bound.add(node.arg)
        elif isinstance(node, ast.alias):
            bound.add((node.asname or node.name).split(".")[0])
    return read, bound


def _root(node: ast.AST) -> Optional[str]:
    while isinstance(node, (ast.Attribute, ast.Call, ast.Subscript)):
        node = node.func if isinstance(node, ast.Call) else node.value
    return node.id if isinstance(node, ast.Name) else None


def _is_call(tree: ast.Module, free: set[str], sentence: str) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.Call)):
            return True
        if isinstance(node, ast.Attribute) and _root(node) in free:
            return True
    body = tree.body
    if len(body) == 1 and isinstance(body[0], ast.Expr) and isinstance(body[0].value, (ast.Name, ast.Attribute)):
        return any(w in sentence.lower() for w in BINDING_WORDS)
    return False


def _not_code(tree: ast.Module) -> bool:
    """A span that parses only by accident: a path (``notes/<topic>/x.md`` reads as a division) or a name with a
    colon (``module:name(...)`` reads as an annotation)."""
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and node.value is None:
            return True
    body = tree.body
    return len(body) == 1 and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.BinOp) and isinstance(
        body[0].value.op, ast.Div,
    )


def _tool_of(call: ast.Call) -> Optional[str]:
    f = call.func
    if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id == "tools":
        return f.attr
    return None


def _ellipsis_only(call: ast.Call) -> bool:
    """Whether the call's arguments are shown with ``...`` (alone, or beside others: ``f(..., append=True)``)."""
    values = [*call.args, *(k.value for k in call.keywords)]
    return any(isinstance(v, ast.Constant) and v.value is Ellipsis for v in values)


def prepare(
    text: str,
    *,
    angle_rules: dict[str, str],
    bare_rules: dict[str, str],
    namespace: Iterable[str],
    example: Callable[[str], Optional[str]],
    skip: Iterable[str] = (),
) -> Plan:
    """The calls *text* shows, ready to run in a sandbox whose names before the first cell are *namespace*.

    * ``<placeholder>``: filled from *angle_rules*; one with no rule in a call is a contract failure.
    * A free name that is not bound in the sandbox, not a harness name and not bound by an earlier span is a
      placeholder: *bare_rules* gives the Python expression it stands for (an assignment before the span), a key
      ``tools.<tool>:<name>`` first for a call of that tool; one with no rule is a contract failure.
    * ``tools.<name>(...)`` (``...`` alone or beside other arguments): replaced by *example(name)*, the tool's
      first registry example (None: a failure). ``f(...)`` on any other name only resolves ``f`` (the shape of its
      arguments is not shown).
    * A call of an end tool (``tools.finish``, ``tools.done``) runs last.
    * A span in *skip* (the reviewed spans a prompt shows as something not to run) is neither run nor listed.
    """
    plan = Plan()
    bound_before = set(namespace) | set(dir(builtins))
    seen: set[str] = set()
    skipped = set(skip)
    for sp in spans(text):
        if sp.text in skipped:
            continue
        filled, missing = fill(sp.text, angle_rules)
        try:
            tree = ast.parse(filled)
        except SyntaxError:
            plan.unparsed.append(sp.text)
            continue
        if _not_code(tree):
            plan.unparsed.append(sp.text)
            continue
        read, bound = _names(tree)
        free = (read - bound) & (HARNESS_NAMES | bound_before)
        if not _is_call(tree, free, sp.sentence):
            continue
        if sp.text in seen:
            continue
        seen.add(sp.text)
        if missing:
            plan.no_rule.append(f"{sp.text!r}: {', '.join(missing)}")
            continue
        code = filled
        end = False
        body = tree.body
        if len(body) == 1 and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Call):
            call = body[0].value
            tool = _tool_of(call)
            if tool in END_TOOLS:
                end = True
            if _ellipsis_only(call):
                if tool is not None:
                    shown = example(tool)
                    if shown is None:
                        plan.no_rule.append(f"{sp.text!r}: no registry example for tools.{tool}")
                        continue
                    code = shown
                else:
                    code = ast.unparse(call.func)
        try:
            final = ast.parse(code)
        except SyntaxError:
            plan.no_rule.append(f"{sp.text!r}: its registry example does not parse")
            continue
        if any(isinstance(n, ast.Constant) and n.value is Ellipsis for n in ast.walk(final)):
            plan.no_rule.append(f"{sp.text!r}: `...` with no rule")
            continue
        read, bound = _names(final)
        placeholders = sorted((read - bound) - bound_before - HARNESS_NAMES)
        # a rule for one tool's argument (``tools.help:name``) wins over the general one (``name``)
        tool = None
        if len(final.body) == 1 and isinstance(final.body[0], ast.Expr) and isinstance(final.body[0].value, ast.Call):
            tool = _tool_of(final.body[0].value)
        rules = {p: bare_rules.get(f"tools.{tool}:{p}", bare_rules.get(p)) for p in placeholders}
        lacking = [p for p, rule in rules.items() if rule is None]
        if lacking:
            plan.no_rule.append(f"{sp.text!r}: {', '.join(lacking)}")
            continue
        prefix = "".join(f"{p} = {rules[p]}\n" for p in placeholders)
        plan.calls.append(Prepared(len(plan.calls), sp.text, prefix + code, end))
        bound_before |= bound
    return plan


def contract_cell(calls: list[Prepared]) -> str:
    """One cell that runs each prepared span as written, in order, inline (never through ``exec``: the actor's
    worker has none), its printing kept out of the cell's output, and prints one ``SPAN {...}`` line per span:
    ``s`` ok or raised, with the exception's class ``e``, its refusal form ``form`` (``tools.Refused``) and the
    start of its text ``m``."""
    lines = ["import contextlib as _cl, io as _io, json as _js"]
    for p in calls:
        lines += [
            "try:",
            "    with _cl.redirect_stdout(_io.StringIO()):",
            textwrap.indent(p.code.rstrip("\n") or "pass", " " * 8),
            f"    print('SPAN ' + _js.dumps({{'k': {p.k}, 's': 'ok'}}))",
            "except Exception as _e:",
            f"    print('SPAN ' + _js.dumps({{'k': {p.k}, 's': 'raised', 'e': type(_e).__name__, "
            "'form': getattr(_e, 'form', None), 'm': str(_e)[:160]}))",
        ]
    return "\n".join(lines) + "\n"


def results(text: str) -> dict[int, dict]:
    out = {}
    for line in (text or "").splitlines():
        line = line.strip()
        if line.startswith(SPAN_MARK):
            try:
                row = json.loads(line[len(SPAN_MARK) :])
            except ValueError:
                continue
            out[row["k"]] = row
    return out


def namespace(text: str) -> Optional[list[str]]:
    for line in (text or "").splitlines():
        if line.strip().startswith(NAMESPACE_MARK):
            return json.loads(line.strip()[len(NAMESPACE_MARK) :])
    return None


def failures(calls: list[Prepared], got: dict[int, dict], *, ended: bool = False) -> list[str]:
    """The spans whose run breaks the contract: a name, import or attribute error on a bound name, a signature
    mismatch, an ``unknown_tool`` or "not available" refusal, or no result at all (an end call that ended the
    work leaves none: *ended*)."""
    bad = []
    for p in calls:
        row = got.get(p.k)
        if row is None:
            # an end call ends the cell (its box is stopped), so it leaves no line of its own
            if not p.end:
                bad.append(f"{p.span!r}: no result (the cell did not reach it)")
            continue
        if row["s"] == "ok":
            continue
        e, m, form = row.get("e") or "", row.get("m") or "", row.get("form")
        if e in ("NameError", "ImportError", "ModuleNotFoundError"):
            bad.append(f"{p.span!r}: {e}: {m}")
        elif e == "AttributeError" and (any(n in m for n in HARNESS_NAMES) or "module" in m):
            bad.append(f"{p.span!r}: {e}: {m}")
        elif e == "TypeError" and _SIGNATURE.search(m):
            bad.append(f"{p.span!r}: {e} (signature): {m}")
        elif form == "unknown_tool" or "not available" in m:
            bad.append(f"{p.span!r}: refused as unavailable: {m}")
    return bad


def opening_names(text: str, sentence_end: str) -> list[str]:
    """The names the prompt's names sentence (the line holding *sentence_end*) says are already available."""
    for line in text.splitlines():
        if sentence_end in line:
            head = line.split(sentence_end, 1)[0]
            return re.findall(r"`([A-Za-z_][A-Za-z0-9_]*)`", head)
    return []


def harness_calls(text: str, tool_names: Iterable[str]) -> list[str]:
    """The spans of *text* that call the harness: a call on ``tools`` or ``memory``, or of a tool by its bare name
    (*tool_names*). Calls of functions a span defines or describes (``read_input(text)``) are not harness calls."""
    names = set(tool_names)
    out = []
    for sp in spans(text):
        filled, _ = fill(sp.text, {})
        try:
            tree = ast.parse(filled)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            f = node.func
            if _root(f) in ("tools", "memory") or (isinstance(f, ast.Name) and f.id in names):
                out.append(sp.text)
                break
    return out

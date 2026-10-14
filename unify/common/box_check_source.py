"""The source text of ``check_examples(candidate, examples)``, the box-local example check the in-box ``tools`` module
embeds.

:data:`SOURCE` is standard-library-only Python, self-contained apart from two names the ``tools`` module binds before
it runs it: ``_difference``, the one comparison of an expected value with an actual one
(:func:`unify.memory.value_diff.difference`, whose module source the ``tools`` module embeds verbatim and runs in its
own namespace, so the box and the gate compare with the same function), and ``_contract``, the module's error
renderer (``unify.common.contract_runtime``'s ``render`` and ``call_text``), so a malformed call teaches as every
other ``tools`` call does. It defines ``CheckReport`` and ``check_examples`` and private names starting ``_ck_``.

It runs inside the agent's interpreter, where the agent's cells run: the candidate is a function the agent defined
in its own session, and it is called there, with the same mounts, filter and limits as the cell that calls the
check. It never touches the tool channel or the host. Nothing bounds how many examples it runs: the cell's own time
limit bounds it, as it bounds any code.
"""

from __future__ import annotations

#: The longest rendering of one value in the report's text, in characters; a cut is marked with what was left out.
#: The whole values are in the report's ``.results``.
SHOWN_CHARS = 600

SOURCE = r'''
# --- check_examples(): run a candidate on each example's input in this interpreter and compare --------------------
import copy as _ck_copy
import inspect as _ck_inspect
import traceback as _ck_traceback

_CK_SHOWN_CHARS = __SHOWN_CHARS__
_CK_SIGNATURE = "check_examples(candidate, examples)"
_CK_EXAMPLE = 'tools.check_examples(solve, [{"input": [2, 3], "expected": 5}, ([1, 1], 2)])'
_CK_FORMS = (
    'a list of examples, each {"input": ..., "expected": ...} (or "output" for "expected") or a pair '
    "(input, expected)"
)


class CheckReport(str):
    """The report as text: a first line ``passed n of m``, then one block per example. ``.passed``: how many
    passed; ``.total``: how many examples ran; ``.ok``: at least one ran and all passed; ``.results``: one dict per
    example, in order, with its values whole: ``input``, ``expected``, ``actual`` (None when it raised),
    ``error`` (``"Type: message"`` when it raised, else None), ``passed`` and ``difference`` (the difference report,
    None when it passed or raised)."""

    passed = 0
    total = 0
    ok = False
    results = ()

    def __new__(cls, text, results):
        self = str.__new__(cls, text)
        self.results = results
        self.total = len(results)
        self.passed = sum(1 for r in results if r["passed"])
        self.ok = self.total > 0 and self.passed == self.total
        return self


def _ck_problem(args, problem):
    """A malformed call's text: the call as written, what is wrong, and a call that works."""
    c = _contract()
    return c["render"](
        c["call_text"]("tools.check_examples", args),
        problem,
        expected="tools." + _CK_SIGNATURE + " with " + _CK_FORMS,
        example=_CK_EXAMPLE,
    )


def _ck_cut(text):
    """*text* cut at _CK_SHOWN_CHARS, the cut marked with how many characters were left out."""
    if len(text) <= _CK_SHOWN_CHARS:
        return text
    return text[:_CK_SHOWN_CHARS] + "… [%d more characters not shown]" % (len(text) - _CK_SHOWN_CHARS)


def _ck_shown(value):
    """*value* as the report's text shows it: its repr, cut (_ck_cut)."""
    try:
        text = repr(value)
    except Exception:  # a repr that fails still shows the type
        text = "<%s>" % type(value).__name__
    return _ck_cut(text)


def _ck_pairs(candidate, examples):
    """The examples as (input, expected) pairs; a malformed one raises with the forms that are taken."""
    if isinstance(examples, dict):
        examples = [examples]  # one example given alone
    if isinstance(examples, (str, bytes)) or not hasattr(examples, "__iter__"):
        raise TypeError(
            _ck_problem((candidate, examples), "examples is %s, not a list of examples" % type(examples).__name__)
        )
    pairs = []
    for n, example in enumerate(examples, 1):
        if isinstance(example, dict):
            keys = set(example)
            shown = keys & {"expected", "output"}
            if "input" in keys and len(shown) == 1 and not keys - {"input", "expected", "output"}:
                pairs.append((example["input"], example[shown.pop()]))
                continue
            why = "example %d has the keys %s" % (n, sorted(map(str, keys)))
        elif isinstance(example, (tuple, list)) and len(example) == 2:
            pairs.append((example[0], example[1]))
            continue
        else:
            why = "example %d is %s, not a dict or a pair" % (n, type(example).__name__)
        raise ValueError(_ck_problem((candidate, examples), why))
    return pairs


def _ck_spreads(candidate):
    """Whether the candidate needs two or more positional arguments: then an input that is a tuple or a list is
    passed as those arguments; otherwise every input is passed as the one argument."""
    try:
        parameters = _ck_inspect.signature(candidate).parameters.values()
    except (TypeError, ValueError):
        return False
    positional = (_ck_inspect.Parameter.POSITIONAL_ONLY, _ck_inspect.Parameter.POSITIONAL_OR_KEYWORD)
    required = [p for p in parameters if p.kind in positional and p.default is _ck_inspect.Parameter.empty]
    return len(required) >= 2


def _ck_raised(exc):
    """``Type: message``, with the line of the innermost frame it was raised at."""
    text = "%s: %s" % (type(exc).__name__, exc)
    frames = _ck_traceback.extract_tb(exc.__traceback__)
    if frames:
        text += " (line %s, in %s)" % (frames[-1].lineno, frames[-1].name)
    return text


def _ck_text(results):
    lines = ["passed %d of %d" % (sum(1 for r in results if r["passed"]), len(results))]
    if not results:
        lines[0] += ": no examples were given"
    for n, r in enumerate(results, 1):
        if r["error"] is not None:
            lines.append("example %d: raised %s" % (n, _ck_cut(r["error"])))
        else:
            lines.append("example %d: %s" % (n, "passed" if r["passed"] else "failed"))
        lines.append("  input: " + _ck_shown(r["input"]))
        if r["passed"]:
            continue
        lines.append("  expected: " + _ck_shown(r["expected"]))
        if r["error"] is None:
            lines.append("  actual: " + _ck_shown(r["actual"]))
            lines.append("  difference:")
            lines.extend("    " + line for line in str(r["difference"]).splitlines())
    return "\n".join(lines)


def check_examples(candidate, examples):
    """Run *candidate* on each example's input here, in your session, and compare what it returns with the
    example's expected output (tools.help("check_examples") has the whole text)."""
    if not callable(candidate):
        raise TypeError(
            _ck_problem((candidate, examples), "candidate is %s, not a function" % type(candidate).__name__)
        )
    pairs = _ck_pairs(candidate, examples)
    spread = _ck_spreads(candidate)
    results = []
    for given, expected in pairs:
        row = {"input": given, "expected": expected, "actual": None, "error": None, "passed": False}
        row["difference"] = None
        try:
            # a copy, so a candidate that changes its input in place changes neither the example nor the report
            value = _ck_copy.deepcopy(given)
        except Exception:
            value = given
        args = tuple(value) if spread and isinstance(value, (tuple, list)) else (value,)
        try:
            actual = candidate(*args)
        except Exception as exc:
            row["error"] = _ck_raised(exc)
        else:
            row["actual"] = actual
            row["difference"] = _difference(expected, actual)
            row["passed"] = row["difference"] is None
        results.append(row)
    return CheckReport(_ck_text(results), results)
'''.replace("__SHOWN_CHARS__", str(SHOWN_CHARS))

#: The names :data:`SOURCE` defines that the ``tools`` module exports.
NAMES = ("check_examples", "CheckReport")


def source() -> str:
    """:data:`SOURCE`, the text the ``tools`` module embeds."""
    return SOURCE

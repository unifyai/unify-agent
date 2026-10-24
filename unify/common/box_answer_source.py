"""The source text of ``answer_from(fn, input=None)``, the box-local answer from a function the in-box ``tools`` module
embeds.

:data:`SOURCE` is standard-library-only Python, self-contained apart from the names the ``tools`` module binds before it
runs it: ``_call`` (the channel client's call), ``_answer`` (the module's ``answer``, so the answer takes the same path
and the same response-format check as ``tools.answer``) and ``_contract`` (the module's error renderer). It defines
``answer_from`` and private names starting ``_af_``.

It runs inside the agent's interpreter, where the agent's cells run: ``fn`` is called there, once, on the request's
input, with the same mounts, filter and limits as the cell that calls it. A stored item of the library passed as ``fn``
is called through its own tracked entry (a use on the real input); nothing else is run. The request's input is the
value the world declared beside the request (the channel's ``declared_input``), or the ``input=`` given; with
neither, the call raises and answers nothing. What it answered from is recorded on the channel (``answered_from``): the
item called, when ``fn`` is one, where the input came from, and the input and output (their digests, and the values
while they are small).
"""

from __future__ import annotations

#: The longest JSON text of an input or output kept whole in the record of an answer; a longer one is kept as its
#: digest only.
KEPT_CHARS = 65536

SOURCE = r'''
# --- answer_from(): answer with a function's output on the request's input ----------------------------------------
import hashlib as _af_hashlib
import json as _af_json

_AF_KEPT_CHARS = __KEPT_CHARS__
_AF_SIGNATURE = "answer_from(fn, input=None)"
_AF_EXAMPLE = "tools.answer_from(solve)"
_AF_MISSING = object()


class WrongCall(TypeError):
    """A call of answer_from that cannot be made as written; the message says what to pass."""


def _af_problem(args, problem):
    """A malformed call's text: the call as written, what is wrong, and a call that works."""
    c = _contract()
    return c["render"](
        c["call_text"]("tools.answer_from", args),
        problem,
        expected="tools." + _AF_SIGNATURE,
        example=_AF_EXAMPLE,
    )


def _af_canonical(value):
    return _af_json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


def _af_kept(value):
    """(digest of the value's canonical JSON, the value when its JSON is short enough, whether it was kept)."""
    try:
        text = _af_canonical(value)
    except Exception:
        text = _af_canonical(repr(value))
        value = text
    digest = _af_hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()[:16]
    if len(text) <= _AF_KEPT_CHARS:
        return digest, _af_json.loads(text), True
    return digest, None, False


def _af_declared():
    """The request's declared input: (True, value), or (False, None) when the request declares none."""
    got = _af_json.loads(str(_call("declared_input", {})))
    if isinstance(got, dict) and got.get("declared") is True:
        return True, got.get("value")
    return False, None


def answer_from(fn, input=_AF_MISSING):
    """Answer the request with what *fn* returns on the request's input (tools.help("answer_from") has the whole
    text)."""
    if not callable(fn):
        raise TypeError(_af_problem((fn,), "fn is %s, not a function" % type(fn).__name__))
    if input is _AF_MISSING or input is None:
        declared, value = _af_declared()
        if not declared:
            raise WrongCall(
                _af_problem((fn,), "this request declares no input: pass input=, the value fn should answer on")
            )
        source = "declared"
    else:
        value, source = input, "given"
    item = getattr(fn, "__memory_item__", None)
    result = fn(value)
    in_digest, in_kept, in_whole = _af_kept(value)
    out_digest, out_kept, out_whole = _af_kept(result)
    row = {
        "item": item if isinstance(item, str) else None,
        "source": source,
        "input_digest": in_digest,
        "output_digest": out_digest,
        "input": in_kept,
        "output": out_kept,
        "kept": bool(in_whole and out_whole),
    }
    try:
        reply = _answer(result)
    except Exception:
        _call("answered_from", dict(row, answered=False))
        raise
    _call("answered_from", dict(row, answered=True))
    return reply
'''.replace("__KEPT_CHARS__", str(KEPT_CHARS))

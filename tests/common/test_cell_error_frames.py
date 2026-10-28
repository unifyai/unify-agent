"""Symbolic: the traceback a model reads leaves out the harness's own frames, and an output cut keeps its end.

The cell runner formats a failing cell's traceback twice: in full, for the harness's logs, and without the frames of
the harness's own namespaces in the box (the tools stub and its channel client, the library package's module and
contract runtime) for the model. A stored library function's frames stay. A head cut of a cell's output never drops
the traceback's last lines.
"""

from __future__ import annotations

import sys
import types

from unify.actor.execution import worker_child as wc
from unify.common.box_interpreter import (
    LOG_HEAD_CHARS,
    LOG_TAIL_CHARS,
    TRACEBACK_END_CHARS,
    bounded_for_log,
    keep_traceback_end,
)


def _raise_through(files: list[str]) -> BaseException:
    """An exception raised through one function per file in *files*, outermost first (a cell, then what it calls)."""
    source = ""
    for i, name in enumerate(files):
        call = f"f{i + 1}()" if i + 1 < len(files) else "raise LookupError('no item named x')"
        source = f"def f{i}():\n    {call}\n"
        exec(compile(source, name, "exec"), globals())  # noqa: S102 - each function is compiled under its file name
    try:
        f0()  # noqa: F821 - defined above
    except BaseException as exc:  # noqa: BLE001
        return exc
    raise AssertionError("nothing raised")


def test_the_model_reads_the_traceback_without_the_harness_frames(monkeypatch):
    library = types.ModuleType("memory")
    library.__file__ = "/memory/memory/__init__.py"
    contract = types.ModuleType("memory._contract")
    contract.__file__ = "/memory/memory/_contract.py"
    monkeypatch.setitem(sys.modules, "memory", library)
    monkeypatch.setitem(sys.modules, "memory._contract", contract)
    exc = _raise_through(
        [
            "<cell 1>",
            "/memory/memory/grid/stencil.py",  # a stored function: the model's own code
            "/memory/memory/__init__.py",
            "/memory/memory/_contract.py",
            "<tools>",
            "<tools: channel client>",
        ],
    )
    full = wc._format_cell_error(exc)
    shown = wc._format_cell_error(exc, wc._harness_files())
    for kept in ("<cell 1>", "/memory/memory/grid/stencil.py"):
        assert kept in full and kept in shown
    for hidden in ("/memory/memory/__init__.py", "/memory/memory/_contract.py", '"<tools>"', "<tools: channel client>"):
        assert hidden in full and hidden not in shown
    assert shown.rstrip().endswith("LookupError: no item named x")


def test_a_cut_keeps_the_end_of_the_traceback():
    error = "Traceback (most recent call last):\n  File \"<cell 1>\", line 3, in <module>\nKeyError: 'width'"
    head = "x" * 50 + "\n[… 90000 bytes in all; head shown]"
    shown = keep_traceback_end(head, error)
    assert shown.endswith("[the traceback ends:]\n" + "\n".join(error.splitlines()[-3:]))
    # nothing is added when the end is already there, or when the cell did not fail
    assert keep_traceback_end(error, error) == error
    assert keep_traceback_end(head, None) == head


def test_one_long_line_cannot_carry_the_cut_into_the_next_prompt():
    """A traceback whose last line is 1 MB (a model can raise or print one) adds at most TRACEBACK_END_CHARS, the
    end kept, past an 8,000-byte head cut."""
    error = "Traceback (most recent call last):\nValueError: " + "x" * 1_000_000
    head = "y" * 8000 + "\n[… 1000050 bytes in all; head shown]"
    shown = keep_traceback_end(head, error)
    assert len(shown) <= len(head) + TRACEBACK_END_CHARS + 100
    assert shown.endswith("x" * 100) and "chars cut]" in shown


def test_a_host_log_line_keeps_the_head_and_tail_of_box_text():
    text = "h" * 50_000 + "t" * 50_000
    logged = bounded_for_log(text)
    assert len(logged) <= LOG_HEAD_CHARS + LOG_TAIL_CHARS + 40
    assert logged.startswith("h" * LOG_HEAD_CHARS) and logged.endswith("t" * LOG_TAIL_CHARS)
    assert bounded_for_log("short") == "short"

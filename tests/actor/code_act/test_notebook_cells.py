"""Symbolic: ``UNIFY_CODE_PROJECTION=notebook`` maps a cell's magics onto ``execute_code``.

As shipped the model fills ``thought``, ``state_mode``, ``session_id``,
``session_name`` (and ``language`` in a sandboxed workspace) on every call;
a cheap model fills every field it is offered. With the switch on the tool
takes one field, ``code``, and where a cell runs is written in the cell as
Jupyter magics on its first lines. These tests pin the mapping from each
magic to the arguments the legacy projection would have sent, the refusals
of magics the session lacks or that are malformed, the hidden channels
passed through untouched, the notebook rendering of results, and the tool
schema the model is sent. The end-to-end behaviour of a session (prompt,
cells in the real worker, product paths) is in
``test_notebook_cells_session.py``.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from unify.actor import notebook_cells as nb
from unify.actor.execution.types import ExecutionResult, ImagePart, TextPart
from unify.common.llm_helpers import method_to_schema
from unify.common.tool_errors import ToolInputError
from unify.common.tool_spec import ToolSpec
from unify.settings import ProductionSettings, SETTINGS

ALL = nb.Capabilities(bash=True)
PYTHON_ONLY = nb.Capabilities(bash=False)


# ── the setting ──────────────────────────────────────────────────────────────


def test_the_switch_is_off_by_default_and_takes_legacy_or_notebook():
    assert ProductionSettings().UNIFY_CODE_PROJECTION == ""
    assert (
        ProductionSettings(UNIFY_CODE_PROJECTION="legacy").UNIFY_CODE_PROJECTION == ""
    )
    assert (
        ProductionSettings(UNIFY_CODE_PROJECTION="Notebook").UNIFY_CODE_PROJECTION
        == "notebook"
    )
    with pytest.raises(ValueError):
        ProductionSettings(UNIFY_CODE_PROJECTION="cells")


def test_enabled_reads_the_setting(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_CODE_PROJECTION", "")
    assert not nb.enabled()
    monkeypatch.setattr(SETTINGS, "UNIFY_CODE_PROJECTION", "notebook")
    assert nb.enabled()


# ── the parser: every magic onto the legacy arguments ────────────────────────


def _args(cell: nb.Cell) -> dict:
    return {
        "language": cell.language,
        "state_mode": cell.state_mode,
        "session_id": cell.session_id,
        "session_name": cell.session_name,
        "packages": cell.packages,
        "list_sessions": cell.list_sessions,
    }


DEFAULT = {
    "language": "python",
    "state_mode": "stateful",
    "session_id": None,
    "session_name": None,
    "packages": (),
    "list_sessions": False,
}


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("x = 1\nx", {}),
        ("%%bash\nls -la", {"language": "bash"}),
        ("%%scratch\nx = 1", {"state_mode": "stateless"}),
        ("%%what_if\nx = 0", {"state_mode": "read_only", "session_id": 0}),
        ("%%session work\nx = 1", {"session_name": "work"}),
        (
            "%%what_if\n%%session work\nx = 0",
            {"state_mode": "read_only", "session_name": "work"},
        ),
        (
            "%%session work\n%%what_if\nx = 0",
            {"state_mode": "read_only", "session_name": "work"},
        ),
        ("%%session work\n%%bash\nls", {"language": "bash", "session_name": "work"}),
        ("%%bash\n%%scratch\nls", {"language": "bash", "state_mode": "stateless"}),
        ("%pip install pandas\nimport pandas", {"packages": ("pandas",)}),
        (
            '%pip install pandas "numpy>=2,<3"\n%pip install rich',
            {"packages": ("pandas", "numpy>=2,<3", "rich")},
        ),
        ("%sessions", {"list_sessions": True}),
        ("\n\n%%scratch\nx = 1", {"state_mode": "stateless"}),
        ("  %%bash\nls", {"language": "bash"}),
    ],
)
def test_each_magic_maps_onto_the_legacy_arguments(code, expected):
    assert _args(nb.parse_cell(code, ALL)) == {**DEFAULT, **expected}


def test_magic_lines_are_blanked_so_line_numbers_are_the_cells_own():
    cell = nb.parse_cell(
        "%%scratch\n%pip install rich\nx = 1\nraise ValueError(x)",
        ALL,
    )
    assert cell.code == "\n\nx = 1\nraise ValueError(x)"
    assert cell.code.split("\n")[3] == "raise ValueError(x)"
    assert cell.magics == ("%%scratch", "%pip install rich")


@pytest.mark.parametrize(
    "code",
    [
        'print("%%bash")',
        "s = '''\n%%scratch\n'''\nprint(s)",
        "# %%what_if is a magic\nx = 1",
        "x = (10\n% 3)\nx",
        "x = 10 % 3",
        "print('!ls')",
    ],
)
def test_magic_like_text_inside_code_is_code(code):
    cell = nb.parse_cell(code, ALL)
    assert _args(cell) == DEFAULT
    assert cell.code == code
    assert cell.magics == ()


# ── the thought: first comment, else first code line ─────────────────────────


@pytest.mark.parametrize(
    ("code", "thought"),
    [
        ("# Totalling March invoices\ntotal = sum(rows)", "Totalling March invoices"),
        ("total = sum(rows)\n# later", "total = sum(rows)"),
        ("%%bash\n# list the workspace\nls", "list the workspace"),
        ("%%bash\n#!/bin/bash\nls -la", "ls -la"),
        ("%%scratch\n\n  value = 3", "value = 3"),
        ("%pip install rich", "%pip install rich"),
        ("%sessions", "%sessions"),
        ("x = '" + "a" * 300 + "'", ("x = '" + "a" * 300 + "'")[: nb.CAPTION_CHARS]),
    ],
)
def test_the_thought_is_the_first_comment_or_the_first_code_line(code, thought):
    assert nb.parse_cell(code, ALL).thought == thought


# ── refusals: clear, actionable, never a silent fallback ─────────────────────


@pytest.mark.parametrize(
    ("code", "caps", "says", "suggests"),
    [
        (
            "%matplotlib inline\nplot()",
            ALL,
            "`%matplotlib` is not a magic",
            "%%scratch",
        ),
        ("%%time\nx = 1", ALL, "`%%time` is not a magic", "%%what_if"),
        ("%%bash\nls", PYTHON_ONLY, "`%%bash` is not available", "subprocess"),
        ("%%bash -x\nls", ALL, "takes no arguments", "%%bash"),
        ("%%what_if\n%%bash\nrm x", ALL, "cannot run as a what-if", "%%what_if"),
        ("%%scratch\n%%what_if\nx", ALL, "cannot be combined", "one of them"),
        ("%%scratch\n%%session a\nx", ALL, "cannot be combined", "one of them"),
        ("%%session\nx = 1", ALL, "needs one name", "%%session experiment"),
        ("%%session a b\nx = 1", ALL, "needs one name", "%%session experiment"),
        ("%%session a/b\nx = 1", ALL, "needs one name", "%%session experiment"),
        ("%%scratch now\nx = 1", ALL, "takes no arguments", "%%scratch"),
        ("%%scratch\n%%scratch\nx", ALL, "appears twice", "once"),
        ("%pip list", ALL, "Only `%pip install`", "%pip install PKG"),
        ("%pip install", ALL, "at least one package", "%pip install pandas"),
        ("%pip install -q pandas", ALL, "options", "%pip install pandas"),
        ("%pip install 'pandas", ALL, "could not read", "%pip install"),
        (
            "%pip install pandas",
            nb.Capabilities(install=False),
            "not available",
            "already installed",
        ),
        ("%sessions\nx = 1", ALL, "cell of its own", "%sessions"),
        ("%sessions all", ALL, "takes no arguments", "%sessions"),
        (
            "%%scratch\nx = 1",
            nb.Capabilities(install=False, sessions=False),
            "not available here",
            "no magics",
        ),
        ("!ls -la", ALL, "Shell lines", "%%bash"),
        ("!ls -la", PYTHON_ONLY, "Shell lines", "subprocess"),
        ("x = 1\n%pip install rich\nimport rich", ALL, "Line 2", "top of the cell"),
        ("# setup\n%%bash\nls", ALL, "Line 2", "top of the cell"),
        ("x = 1\n!ls", ALL, "Line 2 is a shell line", "%%bash"),
    ],
)
def test_a_magic_the_session_cannot_run_is_refused_with_what_to_write(
    code,
    caps,
    says,
    suggests,
):
    with pytest.raises(ToolInputError) as exc:
        nb.parse_cell(code, caps)
    text = exc.value.as_tool_result()
    assert says in exc.value.message, text
    assert exc.value.suggestion and suggests in exc.value.suggestion, text


def test_a_python_syntax_error_unrelated_to_magics_is_left_to_the_runtime():
    # The runtime reports it as today: a traceback in the cell's result.
    cell = nb.parse_cell("x = (1,\nprint(x)", ALL)
    assert cell.code == "x = (1,\nprint(x)"


# ── the projection: the tool the model sees, onto the unchanged runtime ──────


class Runtime:
    """Stands in for the session's execute_code; records every call."""

    def __init__(self, *, bash: bool = True, result: Any = None):
        self.calls: list[dict] = []
        self.result = result
        if bash:

            async def execute_code(
                thought: str,
                code: str | None = None,
                *,
                language: str = "python",
                state_mode: str | None = None,
                session_id: int | None = None,
                session_name: str | None = None,
                _notification_up_q=None,
                _clarification_up_q=None,
                _clarification_down_q=None,
                _interject_queue=None,
                _pause_event=None,
                _parent_chat_context=None,
            ):
                return self._record(locals())

        else:

            async def execute_code(  # type: ignore[misc]
                thought: str,
                code: str | None = None,
                *,
                state_mode: str | None = None,
                session_id: int | None = None,
                session_name: str | None = None,
                _notification_up_q=None,
                _clarification_up_q=None,
                _clarification_down_q=None,
                _interject_queue=None,
                _pause_event=None,
                _parent_chat_context=None,
                _language: str = "python",
            ):
                return self._record(locals())

        self.fn = execute_code

    def _record(self, args: dict) -> Any:
        self.calls.append(args)
        if self.result is not None:
            return self.result
        return ExecutionResult(
            stdout=[TextPart(text="ran\n")],
            state_mode=args.get("state_mode"),
            session_id=args.get("session_id"),
            duration_ms=3,
        )


SESSION_TOOLS = {
    name: ToolSpec(fn=lambda **_: None)
    for name in (
        "list_sessions",
        "inspect_state",
        "close_session",
        "close_all_sessions",
    )
}


def _project(runtime: Runtime, *, parent_context=False, caps=ALL, names=None, **kw):
    tools = {
        "execute_code": ToolSpec(fn=runtime.fn, display_label="Running code"),
        "other": ToolSpec(fn=lambda: None),
        **SESSION_TOOLS,
    }
    return nb.project_tools(
        tools,
        caps=caps,
        steering=kw.pop("steering", True),
        structured=kw.pop("structured", False),
        parent_context=parent_context,
        resolve_session_name=(names or {}).get,
        session_tools=kw.pop("session_tools", SESSION_TOOLS),
        **kw,
    )


def _schema(tool: ToolSpec, **kw) -> dict:
    return method_to_schema(tool.fn, tool_name="execute_code", **kw)["function"]


def test_the_model_sees_one_field_and_no_session_tools():
    tools = _project(Runtime())
    assert sorted(tools) == ["execute_code", "other"]
    assert tools["execute_code"].display_label == "Running code"
    params = _schema(tools["execute_code"], expose_context_control=True)["parameters"]
    assert list(params["properties"]) == ["code"]
    assert params["required"] == ["code"]


def test_parent_chat_context_stays_offered_only_where_sub_agents_read_it():
    with_ctx = _project(Runtime(), parent_context=True)["execute_code"]
    params = _schema(with_ctx, expose_context_control=True)["parameters"]
    assert list(params["properties"]) == ["code", "include_parent_chat_context"]
    assert params["required"] == ["code"]


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        (
            "# total\nx = 1",
            {"thought": "total", "code": "# total\nx = 1", "state_mode": "stateful"},
        ),
        (
            "%%scratch\nx = 1",
            {"thought": "x = 1", "code": "\nx = 1", "state_mode": "stateless"},
        ),
        (
            "%%what_if\nx = 0",
            {"code": "\nx = 0", "state_mode": "read_only", "session_id": 0},
        ),
        (
            "%%session work\nx = 1",
            {"code": "\nx = 1", "state_mode": "stateful", "session_name": "work"},
        ),
        ("%%bash\nls", {"code": "\nls", "language": "bash"}),
    ],
)
def test_a_cell_calls_the_runtime_with_the_legacy_arguments(code, expected):
    runtime = Runtime()
    tool = _project(runtime, names={"work": 1})["execute_code"]
    asyncio.run(tool.fn(code=code))
    (call,) = runtime.calls
    base = {
        "language": "python",
        "session_id": None,
        "session_name": None,
    }
    for key, value in {**base, **expected}.items():
        assert call[key] == value, (key, call)


def test_bash_goes_to_the_runtimes_private_language_argument_where_it_has_one():
    runtime = Runtime(bash=False)
    tool = _project(runtime, caps=PYTHON_ONLY)["execute_code"]
    asyncio.run(tool.fn(code="x = 1"))
    assert runtime.calls[0]["_language"] == "python"
    # Bash is refused: this runtime has no language argument to send it by.
    with pytest.raises(ToolInputError, match="not available"):
        asyncio.run(_project(Runtime(bash=False))["execute_code"].fn(code="%%bash\nls"))


def test_hidden_channels_reach_the_runtime_untouched():
    runtime = Runtime()
    tool = _project(runtime, parent_context=True)["execute_code"]
    channels = {
        "_notification_up_q": asyncio.Queue(),
        "_clarification_up_q": asyncio.Queue(),
        "_clarification_down_q": asyncio.Queue(),
        "_interject_queue": asyncio.Queue(),
        "_pause_event": asyncio.Event(),
        "_parent_chat_context": [{"role": "user", "content": "hi"}],
    }

    async def run():
        await tool.fn(code="%%session work\nx = 1", **channels)

    asyncio.run(run())
    (call,) = runtime.calls
    for name, value in channels.items():
        assert call[name] is value, name


def test_a_refused_cell_never_reaches_the_runtime():
    runtime = Runtime()
    tool = _project(runtime)["execute_code"]
    for code in ("%matplotlib inline", "%%what_if\n%%bash\nls", "x = 1\n!ls"):
        with pytest.raises(ToolInputError):
            asyncio.run(tool.fn(code=code))
    assert runtime.calls == []


def test_a_what_if_on_a_notebook_that_does_not_exist_is_refused():
    runtime = Runtime()
    tool = _project(runtime, names={"work": 1})["execute_code"]
    with pytest.raises(ToolInputError, match="no notebook called 'other'") as exc:
        asyncio.run(tool.fn(code="%%what_if\n%%session other\nx = 0"))
    assert "%sessions" in exc.value.suggestion
    assert runtime.calls == []
    asyncio.run(tool.fn(code="%%what_if\n%%session work\nx = 0"))
    assert runtime.calls[0]["session_name"] == "work"


def test_sessions_lists_every_notebook_and_its_variables():
    async def list_sessions(detail: str = "summary"):
        return {
            "sessions": [
                {"session_id": 0, "session_name": None},
                {"session_id": 1, "session_name": "work"},
                {"session_id": 2, "session_name": None},
            ],
        }

    variables = {0: ["df", "total"], 1: ["x"], 2: []}

    async def inspect_state(session_name=None, session_id=None, detail="summary"):
        assert detail == "names"
        return {"state": {"variables": variables[session_id]}}

    runtime = Runtime()
    tool = _project(
        runtime,
        session_tools={
            "list_sessions": ToolSpec(fn=list_sessions),
            "inspect_state": ToolSpec(fn=inspect_state),
        },
    )["execute_code"]
    out = asyncio.run(tool.fn(code="%sessions"))
    assert runtime.calls == []
    assert out.to_llm_content() == [
        {
            "type": "text",
            "text": "this notebook (the default): df, total\n"
            "%%session work: x\n"
            "session 2 (unnamed): (empty)",
        },
    ]


def test_pip_install_runs_harness_side_then_the_rest_of_the_cell(monkeypatch):
    from unify import environment

    installs: list[list[str]] = []

    def install(specs):
        installs.append(list(specs))
        return {"success": True, "stdout": "", "stderr": "Installed 1 package"}

    monkeypatch.setattr(environment, "install", install)
    runtime = Runtime()
    tool = _project(runtime)["execute_code"]
    out = asyncio.run(tool.fn(code='%pip install "rich>=13"\nimport rich'))
    assert installs == [["rich>=13"]]
    (call,) = runtime.calls
    assert call["code"] == "\nimport rich" and call["state_mode"] == "stateful"
    text = out.to_llm_content()[0]["text"]
    assert text.startswith("[pip] installed rich>=13\nInstalled 1 package\nran")


def test_a_failed_install_stops_the_cell(monkeypatch):
    from unify import environment

    monkeypatch.setattr(
        environment,
        "install",
        lambda specs: {"success": False, "stdout": "", "stderr": "No solution"},
    )
    runtime = Runtime()
    out = asyncio.run(_project(runtime)["execute_code"].fn(code="%pip install nope\nx"))
    assert runtime.calls == []
    assert out.error.startswith("%pip install failed")
    assert "No solution" in out.to_llm_content()[0]["text"]


def test_an_install_alone_does_not_call_the_runtime(monkeypatch):
    from unify import environment

    monkeypatch.setattr(
        environment,
        "install",
        lambda specs: {"success": True, "stdout": "", "stderr": ""},
    )
    runtime = Runtime()
    out = asyncio.run(_project(runtime)["execute_code"].fn(code="%pip install rich"))
    assert runtime.calls == []
    assert out.to_llm_content() == [{"type": "text", "text": "[pip] installed rich"}]


# ── the description ──────────────────────────────────────────────────────────


def test_the_description_names_only_the_magics_the_session_has():
    full = nb.describe(ALL, steering=True, structured=False, network="no network")
    for magic in (
        "%%bash",
        "%pip install",
        "%%scratch",
        "%%what_if",
        "%%session NAME",
        "%sessions",
    ):
        assert magic in full, magic
    assert "workspace sandbox" in full and "no network" in full
    assert '`steer(call_id=<id>, action="stop")`' in full
    assert "steering.messages" not in full
    assert full.endswith(
        "A cell is for computing. You answer, and take any action the requester "
        "defines, by replying.",
    )
    python_only = nb.describe(PYTHON_ONLY, steering=False, structured=True)
    assert "%%bash" not in python_only and "workspace sandbox" not in python_only
    assert "steer(" not in python_only
    assert python_only.endswith("You answer by calling `final_response`.")
    for text in (full, python_only):
        for field in (
            "state_mode",
            "session_id",
            "session_name",
            "thought",
            "language",
        ):
            assert field not in text, field


# ── what the model reads back ────────────────────────────────────────────────


def test_a_result_reads_as_a_notebook_cell():
    result = nb.NotebookCellResult(
        stdout=[TextPart(text="hello\n")],
        stderr=[TextPart(text="careful\n")],
        result={"n": 2},
        state_mode="stateful",
        session_id=0,
        session_created=False,
        duration_ms=12,
    )
    assert result.to_llm_content() == [
        {"type": "text", "text": "hello\n[stderr]\ncareful\nOut: {'n': 2}"},
    ]


def test_a_failed_cell_shows_its_traceback_and_a_note_first():
    result = nb.NotebookCellResult(
        error="Traceback (most recent call last):\nValueError: bad",
        note="placeholder arguments",
        steering={"steps": 0, "retries": 0, "replayed": 0, "executed": 0},
    )
    assert result.to_llm_content() == [
        {
            "type": "text",
            "text": "[note] placeholder arguments\n"
            "Traceback (most recent call last):\nValueError: bad",
        },
    ]


def test_steering_shows_only_when_something_steered_the_cell():
    result = nb.NotebookCellResult(
        steering={"steps": 2, "interjections_received": 1},
    )
    assert result.to_llm_content() == [
        {
            "type": "text",
            "text": '[steering] {"steps": 2, "interjections_received": 1}',
        },
    ]


def test_an_empty_cell_reads_no_output_and_images_keep_their_place():
    assert nb.NotebookCellResult().to_llm_content() == [
        {"type": "text", "text": "(no output)"},
    ]
    png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    result = nb.NotebookCellResult(
        stdout=[TextPart(text="before\n"), ImagePart(data=png), TextPart(text="after")],
    )
    kinds = [block["type"] for block in result.to_llm_content()]
    assert kinds[0] == "text" and kinds[-1] == "text" and "image_url" in kinds


def test_a_result_holding_a_handle_keeps_the_shipped_rendering_for_adoption():
    from unify.common._async_tool.tools_data import _HANDLE_SENTINEL

    fields = dict(result=_HANDLE_SENTINEL, state_mode="stateful", session_id=0)
    assert (
        nb.NotebookCellResult(**fields).to_llm_content()
        == ExecutionResult(**fields).to_llm_content()
    )


def test_the_product_keeps_the_runtimes_result_object():
    runtime = Runtime()
    out = asyncio.run(_project(runtime)["execute_code"].fn(code="%%scratch\nx"))
    assert isinstance(out, ExecutionResult)
    assert (out.state_mode, out.duration_ms) == ("stateless", 3)
    # The runtime's empty-cell envelope is a dict; it reads as a cell too.
    empty = Runtime(result={"stdout": "", "stderr": "", "result": None, "error": None})
    out = asyncio.run(_project(empty)["execute_code"].fn(code="%%scratch"))
    assert isinstance(out, ExecutionResult)
    assert out.to_llm_content() == [{"type": "text", "text": "(no output)"}]


# ── readers of the transcript ────────────────────────────────────────────────


def _calls(*codes: str) -> list[dict]:
    import json

    messages: list[dict] = []
    for i, code in enumerate(codes):
        messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": f"c{i}",
                        "type": "function",
                        "function": {
                            "name": "execute_code",
                            "arguments": json.dumps({"code": code}),
                        },
                    },
                ],
            },
        )
        messages.append(
            {
                "role": "tool",
                "tool_call_id": f"c{i}",
                "name": "execute_code",
                "content": "ok",
            },
        )
    return messages


def test_the_inspection_digest_reads_the_cells_caption(monkeypatch):
    from unify.common._async_tool.transcript_ops import _tool_call_meta

    messages = _calls("# Totalling March invoices\nx = 1", "%%bash\nls -la")
    monkeypatch.setattr(SETTINGS, "UNIFY_CODE_PROJECTION", "")
    assert [m["thought"] for m in _tool_call_meta(messages).values()] == [None, None]
    monkeypatch.setattr(SETTINGS, "UNIFY_CODE_PROJECTION", "notebook")
    assert [m["thought"] for m in _tool_call_meta(messages).values()] == [
        "Totalling March invoices",
        "ls -la",
    ]


def test_origin_capture_reads_a_bash_cell_as_bash(monkeypatch):
    from unify.function_manager import origin_capture

    messages = _calls("%%bash\necho hi", "x = 1")
    monkeypatch.setattr(SETTINGS, "UNIFY_CODE_PROJECTION", "")
    assert [c[2] for c in origin_capture._code_cells(messages)] == ["python", "python"]
    monkeypatch.setattr(SETTINGS, "UNIFY_CODE_PROJECTION", "notebook")
    cells = origin_capture._code_cells(messages)
    assert [(c[1], c[2]) for c in cells] == [("echo hi", "bash"), ("x = 1", "python")]

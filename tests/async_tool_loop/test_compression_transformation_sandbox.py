"""Compression's transformations run in a confined box, never in the harness.

The compression model edits transcript entries with Python
(``update(index, transformation)``); the transcript holds untrusted content.
Each transformation runs in a fresh bubblewrap box
(``transformation_sandbox.run_transformation``): no credentials or
environment, no network, none of the host's files, writes only to the box's
own temporary space, a time limit with a verified kill, and without
bubblewrap it is refused rather than run here. Its results are what
in-process ``exec`` gave.
"""

from __future__ import annotations

import builtins
import json
import os
import shutil
import socket
import sys
import threading
import time
import uuid
from pathlib import Path

import pytest

import unify.common._async_tool.context_compression as _cc_mod
import unify.common._async_tool.transformation_sandbox as _ts_mod
from unify import sandbox
from unify.common._async_tool import transformation_sandbox
from unify.common._async_tool.context_compression import (
    _eval_transformation,
    _get_sandbox_globals,
    _make_update_tool,
    compress_messages,
)
from unify.memory import sandbox_run
from tests.helpers import _handle_project
from tests.scripted_model import reply

needs_bwrap = pytest.mark.skipif(
    shutil.which("bwrap") is None or not sys.platform.startswith("linux"),
    reason="bubblewrap (bwrap) is not installed; the box cannot run here",
)

_ENDPOINT = "openai/gpt-4o@openrouter"
FAKE_KEY = "sk-fake-compression-canary-6b1d"  # pragma: allowlist secret
SETTING_CANARY = "harness-only-setting-41f0"

# What the box's environment may hold: what the box sets, and what the
# interpreter itself adds (locale coercion).
_BOX_ENV = {"PATH", "HOME", "PYTHONDONTWRITEBYTECODE", "PWD", "LC_CTYPE"}


def _in_process(transformation_str: str, content: str) -> str:
    """``_eval_transformation`` as it was before the box (the reference)."""
    namespace = dict(_get_sandbox_globals())
    namespace["x"] = content
    code = transformation_str.strip()
    if "\n" not in code and "=" not in code:
        try:
            compile(code, "<transformation>", "eval")
            code = f"x = {code}"
        except SyntaxError:
            pass
    exec(code, namespace)  # noqa: S102 - the test's own, known transformations
    result = namespace["x"]
    if not isinstance(result, str):
        result = str(result)
    return result


def _outcome(fn, code: str, content: str) -> str:
    """What the ``update`` tool shows for *code*: the result, or its error line."""
    try:
        return "ok: " + fn(code, content)
    except Exception as exc:  # noqa: BLE001
        return f"Error applying transformation: {type(exc).__name__}: {exc}"


@pytest.fixture
def no_exec_here(monkeypatch):
    """Record any exec/eval/compile the compression modules make in this process."""
    seen: list[tuple[str, object]] = []

    def spy(name):
        real = getattr(builtins, name)

        def wrapper(source, *args, **kwargs):
            seen.append((name, source))
            return real(source, *args, **kwargs)

        return wrapper

    for module in (_cc_mod, _ts_mod):
        for name in ("exec", "eval", "compile"):
            monkeypatch.setattr(module, name, spy(name), raising=False)
    return seen


# ── equivalence: the same results as in-process exec ──────────────────────

_TOOL_MSG = json.dumps(
    {
        "role": "tool",
        "content": "line1\nERROR line2\nTraceback (most recent call last):\n"
        '  File "a.py", line 3\nValueError: bad\nline3 ' + "x" * 600,
    },
)
_UNICODE = json.dumps({"role": "user", "content": "naïve café — 東京 🎉 \u00a0end"})
_UNICODE_RAW = "naïve café — 東京 🎉 \u00a0end"

EQUIVALENT = [
    ('x = "compressed"', _TOOL_MSG),
    ('x = x.replace("ERROR line2", "error")', _TOOL_MSG),
    ('x.replace("line1", "first")', _TOOL_MSG),
    ('x = x.split("\\n")[0]', "first line\nsecond line"),
    ("x = x[:100]", _TOOL_MSG),
    ("x = x[-50:]", _TOOL_MSG),
    ("x[:200] + '... [truncated]'", _TOOL_MSG),
    (r'x = re.sub(r"Traceback[\s\S]*", "traceback omitted", x)', _TOOL_MSG),
    (
        "msg = json.loads(x)\n"
        "lines = msg['content'].split('\\n')\n"
        "msg['content'] = '\\n'.join(l for l in lines if 'ERROR' not in l)\n"
        "x = json.dumps(msg)",
        _TOOL_MSG,
    ),
    (
        "msg = json.loads(x)\nmsg['content'] = msg['content'][:40]\n"
        "x = json.dumps(msg, ensure_ascii=False)",
        _UNICODE,
    ),
    ('x = x.replace("東京", "Tokyo")', _UNICODE_RAW),
    ("x = x.upper()[::-1]", _UNICODE_RAW),
    ("x = len(x)", "hello"),
    ("x = None", "hello"),
    ("x = [1, 2]", "hello"),
    ("x = str(collections.Counter(x).most_common(2))", "aabbbc"),
    ("x = str(statistics.mean([1, 2, 4]))", ""),
    ("x = x.strip()", "   padded   "),
    ('x = ""', "something"),
    ("x = '\\n'.join(sorted(set(x.split())))", "b a c a b"),
    # errors: the same type name and message
    ("x = 1 // 0", "content"),
    ('x = {}["missing"]', "content"),
    ("def def def", "content"),
    ("x = undefined_name", "content"),
    ("del x", "content"),
    ("x = int('nope')", "content"),
    ("x = x[", "content"),
    ("raise RuntimeError('custom message')", "content"),
    ("class Mine(Exception): pass\nraise Mine('own type')", "content"),
]


@needs_bwrap
@pytest.mark.parametrize("code,content", EQUIVALENT)
def test_a_transformation_gives_what_in_process_exec_gave(code, content):
    assert _outcome(_eval_transformation, code, content) == _outcome(
        _in_process,
        code,
        content,
    )


@needs_bwrap
def test_the_error_keeps_its_builtin_type():
    with pytest.raises(ZeroDivisionError, match="integer division or modulo by zero"):
        _eval_transformation("x = 1 // 0", "content")


@needs_bwrap
def test_the_box_has_the_same_globals():
    code = "x = json.dumps(sorted(dir()))"
    names = json.loads(_eval_transformation(code, ""))
    assert names == json.loads(_in_process(code, ""))
    assert {"re", "json", "typing", "collections", "pydantic", "x"} <= set(names)
    builtins_code = "x = json.dumps(sorted(__builtins__))"
    assert _eval_transformation(builtins_code, "") == _in_process(builtins_code, "")


@needs_bwrap
def test_the_update_tool_reports_and_applies_as_before():
    entries = {
        0: json.dumps({"role": "user", "content": "hello world"}),
        1: json.dumps({"role": "tool", "content": "keep"}),
    }
    update = _make_update_tool(entries, _ENDPOINT)
    assert update(0, 'x.replace("hello", "hi")').startswith(
        json.dumps({"role": "user", "content": "hi world"}),
    )
    before = entries[1]
    assert update(1, "x = 1 // 0") == (
        "Error applying transformation: ZeroDivisionError: "
        "integer division or modulo by zero"
    )
    assert entries[1] == before
    assert update(1, 'x = ""').startswith("(empty)\n")


# ── confinement ─────────────────────────────────────────────────────────────


@needs_bwrap
def test_the_transformation_sees_no_credentials_or_environment(
    monkeypatch,
    no_exec_here,
):
    monkeypatch.setenv("OPENAI_API_KEY", FAKE_KEY)
    monkeypatch.setenv("UNIFY_COMPRESSION_CANARY", SETTING_CANARY)
    seen = _eval_transformation("import os\nx = json.dumps(dict(os.environ))", "")
    assert FAKE_KEY not in seen
    assert SETTING_CANARY not in seen
    env = json.loads(seen)
    assert set(env) <= _BOX_ENV
    assert not any(sandbox.is_secret_name(k) for k in env)
    # Nor through the process's own records.
    proc_env = _eval_transformation(
        "x = open('/proc/self/environ', 'rb').read().decode('ascii', 'replace')",
        "",
    )
    assert FAKE_KEY not in proc_env
    assert SETTING_CANARY not in proc_env
    assert no_exec_here == []


@needs_bwrap
def test_the_transformation_cannot_read_host_files(tmp_path, no_exec_here):
    secret = tmp_path / "harness-only.txt"
    secret.write_text("harness file contents 9a7e")
    entries = {0: "entry"}
    update = _make_update_tool(entries, _ENDPOINT)
    result = update(0, f"x = open({str(secret)!r}).read()")
    assert result.startswith("Error applying transformation: FileNotFoundError")
    assert "9a7e" not in result
    assert entries[0] == "entry"
    # Nor the checkout the harness runs from.
    here = Path(__file__).resolve()
    assert _outcome(
        _eval_transformation,
        f"x = open({str(here)!r}).read()",
        "",
    ).startswith("Error applying transformation: FileNotFoundError")
    # Nor the Unify state directory (the transcripts, the store).
    home = Path(os.environ["UNIFY_HOME"])
    (home / "transcripts").mkdir(parents=True, exist_ok=True)
    (home / "transcripts" / "t.jsonl").write_text("transcript line 51c2")
    listing = _outcome(
        _eval_transformation,
        f"import os\nx = str(os.listdir({str(home)!r}))",
        "",
    )
    assert "transcripts" not in listing
    assert "51c2" not in listing
    assert no_exec_here == []


@needs_bwrap
def test_the_transformation_writes_only_to_its_own_temporary_space(
    tmp_path,
    no_exec_here,
):
    tag = uuid.uuid4().hex[:8]
    # A host directory that exists: it is not in the box.
    host_target = tmp_path / f"written-{tag}.txt"
    assert _outcome(
        _eval_transformation,
        f"open({str(host_target)!r}, 'w').write('from the box')\nx = 'wrote'",
        "",
    ).startswith("Error applying transformation: FileNotFoundError")
    assert not host_target.exists()
    # Beside this test file, in the checkout.
    checkout_target = Path(__file__).resolve().with_name(f"written-{tag}.txt")
    assert _outcome(
        _eval_transformation,
        f"open({str(checkout_target)!r}, 'w').write('from the box')\nx = 'wrote'",
        "",
    ).startswith("Error applying transformation:")
    assert not checkout_target.exists()
    # The read-only root.
    readonly = _outcome(
        _eval_transformation,
        f"open('/usr/written-{tag}.txt', 'w').write('from the box')\nx = 'wrote'",
        "",
    )
    assert readonly.startswith("Error applying transformation: OSError")
    assert "Read-only file system" in readonly
    assert not Path(f"/usr/written-{tag}.txt").exists()
    # Its own /tmp is writable, and is the box's alone: nothing reaches the
    # host's /tmp, and the next box starts with an empty one.
    inside = f"/tmp/written-{tag}.txt"
    code = f"open({inside!r}, 'w').write('scratch')\nx = open({inside!r}).read()"
    assert _eval_transformation(code, "") == "scratch"
    assert not Path(inside).exists()
    assert _eval_transformation(
        f"import os\nx = str(os.path.exists({inside!r}))",
        "",
    ) == "False"
    assert no_exec_here == []


@needs_bwrap
def test_the_transformation_has_no_network(no_exec_here):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]
    try:
        code = (
            "import socket\n"
            f"s = socket.create_connection(('127.0.0.1', {port}), timeout=2)\n"
            "x = 'connected'"
        )
        assert _outcome(_eval_transformation, code, "").startswith(
            "Error applying transformation: ConnectionRefusedError",
        )
        external = (
            "import socket\n"
            "s = socket.create_connection(('1.1.1.1', 53), timeout=2)\n"
            "x = 'connected'"
        )
        assert _outcome(_eval_transformation, external, "").startswith(
            "Error applying transformation: OSError",
        )
    finally:
        server.close()
    assert no_exec_here == []


@needs_bwrap
def test_the_harness_process_is_untouched(no_exec_here):
    marker = f"UNIFY_TRANSFORMATION_PROBE_{uuid.uuid4().hex[:8]}"
    code = (
        "import builtins, os, sys\n"
        f"setattr(builtins, {marker!r}, 1)\n"
        f"os.environ[{marker!r}] = '1'\n"
        f"sys.modules['json'].{marker} = 1\n"
        "x = 'done'"
    )
    assert _eval_transformation(code, "") == "done"
    assert not hasattr(builtins, marker)
    assert marker not in os.environ
    assert not hasattr(json, marker)
    assert no_exec_here == []


def test_without_bubblewrap_the_transformation_is_refused(monkeypatch, no_exec_here):
    real_which = shutil.which

    def which(name, *args, **kwargs):
        return None if name == "bwrap" else real_which(name, *args, **kwargs)

    monkeypatch.setattr(sandbox_run.shutil, "which", which)
    marker = f"UNIFY_NO_BWRAP_PROBE_{uuid.uuid4().hex[:8]}"
    entries = {0: "entry"}
    update = _make_update_tool(entries, _ENDPOINT)
    result = update(
        0,
        f"import builtins\nsetattr(builtins, {marker!r}, 1)\nx = 'ran'",
    )
    assert result == "Error applying transformation: RuntimeError: bubblewrap is required"
    assert entries[0] == "entry"
    assert not hasattr(builtins, marker)
    assert no_exec_here == []


# ── bounds ──────────────────────────────────────────────────────────────────


def _alive_with(argument: str) -> list[int]:
    pids = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            argv = (proc / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        if argument.encode() in argv:
            pids.append(int(proc.name))
    return pids


def _sleep_argument() -> str:
    """A unique ``sleep`` duration (60-90 s) that names its process."""
    return f"{60 + uuid.uuid4().int % 30}.{uuid.uuid4().int % 10**6:06d}"


def _hanging(argument: str) -> str:
    # A child process of its own, then a loop that never ends.
    return (
        "import subprocess\n"
        f"subprocess.Popen(['/bin/sleep', {argument!r}])\n"
        "while True:\n"
        "    pass"
    )


def _in_thread(fn, timeout: float):
    box: dict = {}

    def run() -> None:
        box["result"] = fn()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(timeout)
    assert not thread.is_alive(), f"still running after {timeout} s"
    return box["result"]


def _gone(argument: str, within: float = 10.0) -> bool:
    deadline = time.monotonic() + within
    while _alive_with(argument):
        if time.monotonic() > deadline:
            return False
        time.sleep(0.1)
    return True


@needs_bwrap
def test_a_hanging_transformation_is_killed_and_compression_continues(monkeypatch):
    monkeypatch.setattr(transformation_sandbox, "TIMEOUT_S", 2.0)
    argument = _sleep_argument()
    entries = {0: "first entry", 1: "second entry"}
    update = _make_update_tool(entries, _ENDPOINT)
    started = time.monotonic()
    result = _in_thread(lambda: update(0, _hanging(argument)), timeout=30.0)
    elapsed = time.monotonic() - started
    assert result == (
        "Error applying transformation: TimeoutError: the transformation ran "
        "longer than 2 s and was stopped"
    )
    assert elapsed < 15.0
    assert entries[0] == "first entry"
    # Every process of the box is gone, the transformation's own child too
    # (the kill is verified before update returns).
    assert _alive_with(argument) == []
    # The next transformation runs as usual.
    assert update(1, 'x = "short"').startswith("short\n")
    assert entries[1] == "short"


@needs_bwrap
def test_a_finished_transformation_leaves_no_process_behind():
    argument = _sleep_argument()
    code = (
        "import subprocess\n"
        f"subprocess.Popen(['/bin/sleep', {argument!r}])\n"
        "x = 'returned'"
    )
    assert _eval_transformation(code, "") == "returned"
    assert _alive_with(argument) == []


@needs_bwrap
def test_an_oversized_result_is_discarded(monkeypatch):
    monkeypatch.setattr(transformation_sandbox, "MAX_REPLY_BYTES", 1000)
    # The bound is the larger of MAX_REPLY_BYTES and twice the request.
    with pytest.raises(RuntimeError, match=r"larger than \d+ bytes and was discarded"):
        _eval_transformation("x = 'y' * 100000", "small")
    assert _eval_transformation("x = 'y' * 500", "small") == "y" * 500


@needs_bwrap
def test_printing_does_not_corrupt_the_result():
    code = (
        "import os, sys\nprint('noise')\nsys.stdout.flush()\n"
        "os.write(1, b'{\"x\": \"forged\"}')\nx = 'real'"
    )
    assert _eval_transformation(code, "") == "real"


@needs_bwrap
@pytest.mark.asyncio
@_handle_project
async def test_compress_messages_continues_past_a_hanging_transformation(
    llm_config,
    scripted_model,
    monkeypatch,
):
    monkeypatch.setattr(transformation_sandbox, "TIMEOUT_S", 2.0)
    argument = _sleep_argument()
    messages = [
        {"role": "user", "content": "do the task"},
        {"role": "tool", "content": "very long tool output " * 20},
    ]
    model = scripted_model(
        compactor=[
            reply(
                calls=[("update", {"index": 1, "transformation": _hanging(argument)})],
            ),
            reply(
                calls=[
                    (
                        "update",
                        {
                            "index": 1,
                            "transformation": (
                                "msg = json.loads(x)\nmsg['content'] = 'tool output'\n"
                                "x = json.dumps(msg)"
                            ),
                        },
                    ),
                ],
            ),
            reply("compressed"),
        ],
    )
    result = await compress_messages(messages, llm_config["model"])
    model.assert_used_up()
    tool_results = [
        m["content"]
        for m in model.of("compactor")[1].messages
        if m.get("role") == "tool"
    ]
    assert any(
        "Error applying transformation: TimeoutError" in str(c) for c in tool_results
    )
    assert [m.content for m in result.messages] == [
        json.dumps(messages[0]),
        json.dumps({"role": "tool", "content": "tool output"}),
    ]
    assert _gone(argument)

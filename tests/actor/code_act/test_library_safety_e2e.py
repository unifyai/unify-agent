"""Symbolic, end to end: stored code's dependencies, confinement and guidance under the core surface.

A stored function that imports a package declares it as a dependency; the
harness installs it into the workspace venv before the function first runs
in a fresh home, also when the function is a helper of the entry point that
was run, and an unreachable package index ends the call with a clear error
rather than a hang. Stored code runs only in the sandboxed worker: without
the harness's credentials or network, without the harness's ``/tmp``,
unable to write the store, and with the harness objects' underscored
attributes refused; what another session recorded in the store carries no
credential it was passed; an environment's globals reach it as proxies; a
sub-agent cannot be granted store writes its caller lacks. Guidance stored
from code is found from code and listed in the task's first message per the
shortlist rules, the gated list naming the entry point and the request it
was stored for.

Cells run in the real sandboxed worker, packages are installed from a local
wheel with the installer offline, the model (where there is one) is a
scripted transport, and search ranks with a deterministic embedder; nothing
reaches a model or the package index. Skipped where bubblewrap is missing.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import socket
import tempfile
import time
import zipfile
from pathlib import Path

import pytest

from tests import cache_discipline_helpers as h
from tests.actor.code_act.core_world import core_world, world  # noqa: F401
from tests.actor.code_act.library_world import (
    GUIDANCE_CONTENT,
    GUIDANCE_TITLE,
    HIERARCHY,
    SUMMARY,
    TEXT,
    Cells,
    install_fake_embed,
    new_actor,
    run_in_another_process,
)
from tests.actor.code_act.sandbox_world import (
    SSH_SECRET,
    TOKEN_VALUE,
    needs_bwrap,
    serve,
)
from tests.helpers import _handle_project
from unify import db, environment, sandbox
from unify.settings import SETTINGS


@pytest.fixture
def library(core_world, monkeypatch):  # noqa: F811
    install_fake_embed(monkeypatch.setattr)
    monkeypatch.setattr(SETTINGS, "UNIFY_FUNCTION_CASES", True)
    monkeypatch.setattr(SETTINGS, "UNIFY_STORE_TRUST", "ramp")
    monkeypatch.setattr(SETTINGS, "UNIFY_BUILTIN_GUIDANCE", False)
    return core_world


# ── 4: dependencies ─────────────────────────────────────────────────────────

PACKAGE = "e2etinypkg"


def _wheel(directory: Path) -> Path:
    """A pure-Python wheel of ``e2etinypkg`` 1.0, built without the network."""
    files = {
        f"{PACKAGE}/__init__.py": (
            "VALUE = 'tiny-ok'\n\ndef shout(text):\n    return text.upper() + '!'\n"
        ),
        f"{PACKAGE}-1.0.dist-info/METADATA": (
            f"Metadata-Version: 2.1\nName: {PACKAGE}\nVersion: 1.0\n"
        ),
        f"{PACKAGE}-1.0.dist-info/WHEEL": (
            "Wheel-Version: 1.0\nGenerator: tests\nRoot-Is-Purelib: true\n"
            "Tag: py3-none-any\n"
        ),
    }
    record = []
    for name, text in files.items():
        data = text.encode()
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=")
        record.append(f"{name},sha256={digest.decode()},{len(data)}")
    record_name = f"{PACKAGE}-1.0.dist-info/RECORD"
    record.append(f"{record_name},,")
    files[record_name] = "\n".join(record) + "\n"
    path = directory / f"{PACKAGE}-1.0-py3-none-any.whl"
    with zipfile.ZipFile(path, "w") as zf:
        for name, text in files.items():
            zf.writestr(name, text)
    return path


SHOUT = (
    "def shout_text(text: str) -> str:\n"
    '    """Upper-case text with an exclamation mark, via e2etinypkg."""\n'
    "    import e2etinypkg\n"
    "    return e2etinypkg.shout(text)\n"
)
GREET = (
    "def greet(name: str) -> str:\n"
    '    """Greet someone loudly."""\n'
    "    return shout_text('hello ' + name)\n"
)


@pytest.fixture
def offline_wheel(library, monkeypatch):
    """The package as a local wheel, the installer offline."""
    directory = library["home"] / "wheels"
    directory.mkdir()
    wheel = _wheel(directory)
    monkeypatch.setenv("UV_OFFLINE", "1")
    return f"{PACKAGE} @ {wheel.as_uri()}"


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(300)
@_handle_project
@pytest.mark.parametrize("run_helper_directly", [True, False])
async def test_a_declared_dependency_is_installed_on_first_run_in_a_fresh_home(
    library,
    offline_wheel,
    run_helper_directly,
):
    """The helper declares the package; the entry point calling it declares
    nothing. Either one, run first in a home without a venv, works."""
    from unify.function_manager.function_manager import FunctionManager

    assert not environment.environment_dir().exists()
    fm = FunctionManager(include_primitives=False)
    fm.add_functions(implementations=[SHOUT], dependencies=[offline_wheel])
    fm.add_functions(implementations=[GREET])
    assert fm._get_function_data_by_name(name="greet")["dependencies"] == []
    cells = Cells(new_actor(can_store=False))
    try:
        code = (
            "await functions.run('shout_text', text='hi')"
            if run_helper_directly
            else "await functions.run('greet', name='ada')"
        )
        out = await cells(code)
        assert out.error is None, out.error
        assert out.result == ("HI!" if run_helper_directly else "HELLO ADA!")
        # Installed into the workspace venv, importable in later cells too.
        assert environment.missing([offline_wheel]) == []
        out = await cells("import e2etinypkg\ne2etinypkg.VALUE")
        assert out.result == "tiny-ok", out.error
    finally:
        await cells.close()


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(240)
@_handle_project
async def test_an_unreachable_package_index_is_a_clear_error_not_a_hang(
    library,
    monkeypatch,
):
    from unify.function_manager.function_manager import FunctionManager

    # A listener that accepts and never answers, and a closed port.
    silent = socket.socket()
    silent.bind(("127.0.0.1", 0))
    silent.listen(8)
    closed = socket.socket()
    closed.bind(("127.0.0.1", 0))
    closed_port = closed.getsockname()[1]
    closed.close()
    monkeypatch.setenv("UV_HTTP_TIMEOUT", "3")
    monkeypatch.delenv("UV_OFFLINE", raising=False)
    fm = FunctionManager(include_primitives=False)
    fm.add_functions(
        implementations=[SHOUT.replace("e2etinypkg", "e2eabsentpkg")],
        dependencies=["e2eabsentpkg>=1"],
    )
    cells = Cells(new_actor(can_store=False))
    try:
        for index in (
            f"http://127.0.0.1:{closed_port}/simple",
            f"http://127.0.0.1:{silent.getsockname()[1]}/simple",
        ):
            monkeypatch.setenv("UV_INDEX_URL", index)
            monkeypatch.setenv("UV_DEFAULT_INDEX", index)
            started = time.monotonic()
            out = await cells("await functions.run('shout_text', text='hi')")
            took = time.monotonic() - started
            assert out.result is None
            assert "RuntimeError" in out.error, out.error
            assert "Failed to install ['e2eabsentpkg>=1']" in out.error, out.error
            assert took < 120, took
        # The failure is the function's own, held against it.
        rows = db.query("SELECT failures FROM function_trust")
        assert [int(r["failures"]) for r in rows] == [2]
        out = await cells("1 + 1")
        assert out.result == 2, out.error
    finally:
        silent.close()
        await cells.close()


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(240)
@_handle_project
async def test_installing_adds_no_mount_or_network_to_the_sandbox(
    library,
    offline_wheel,
):
    from unify.function_manager.function_manager import FunctionManager

    fm = FunctionManager(include_primitives=False)
    fm.add_functions(implementations=[SHOUT], dependencies=[offline_wheel])
    srv, port = serve(b"from-the-host\n")
    cells = Cells(new_actor(can_store=False))
    try:
        out = await cells("await functions.run('shout_text', text='x')")
        assert out.result == "X!", out.error
        policy = sandbox.build_policy(fresh=True)
        argv = sandbox.wrap_argv(["true"], policy, cwd=str(policy.workspace))
        assert "--unshare-all" in argv and "--share-net" not in argv
        venv = str(environment.environment_dir().resolve())
        binds = [argv[i + 1] for i, a in enumerate(argv) if a == "--bind"]
        assert venv not in binds and binds == [str(policy.workspace)]
        out = await cells(
            "import socket\n"
            "try:\n"
            f"    socket.create_connection(('127.0.0.1', {port}), timeout=3)\n"
            "    reached = True\n"
            "except OSError:\n"
            "    reached = False\n"
            "reached",
        )
        assert out.result is False, out.error
        out = await cells(
            f"open({str(Path(venv) / 'pwned')!r}, 'w')",
        )
        assert "Read-only file system" in out.error, out.error
    finally:
        srv.close()
        await cells.close()


# ── 7: safety ───────────────────────────────────────────────────────────────

PROBE = (
    "def probe(paths: list) -> dict:\n"
    '    """What this code can see of the machine it runs on."""\n'
    "    import os, pathlib\n"
    "    seen = {}\n"
    "    for path in paths:\n"
    "        try:\n"
    "            seen[path] = pathlib.Path(path).read_text()\n"
    "        except OSError as exc:\n"
    "            seen[path] = type(exc).__name__\n"
    "    return {\n"
    "        'process': f\"{os.getpid()}@{os.readlink('/proc/self/ns/pid')}\",\n"
    "        'env': sorted(k for k in os.environ if 'TOKEN' in k or 'PASSWORD' in k),\n"
    "        'files': seen,\n"
    "    }\n"
)


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(240)
@_handle_project
async def test_stored_code_runs_only_in_the_sandboxed_worker(library):
    from unify.function_manager.function_manager import FunctionManager

    harness = f"{os.getpid()}@{os.readlink('/proc/self/ns/pid')}"
    verifier = Path(tempfile.mkdtemp(prefix="unify-verifier-", dir="/tmp"))
    (verifier / "answers.json").write_text('{"task-1": 42}\n')
    fm = FunctionManager(include_primitives=False)
    # The store-time check refuses ``open`` by name; it is not the boundary
    # (``pathlib`` reads the same files), the sandbox is.
    refused = fm.add_functions(
        implementations=["def peek(path):\n    return open(path).read()\n"],
        raise_on_error=False,
    )
    assert "Dangerous built-in 'open'" in refused["peek"]
    fm.add_functions(implementations=[PROBE])
    paths = [
        str(library["home"] / ".ssh" / "id_rsa"),
        str(verifier / "answers.json"),
        str(library["state"] / "logs" / "unify.log"),
    ]
    cells = Cells(new_actor(can_store=False))
    try:
        out = await cells(f"await functions.run('probe', paths={paths!r})")
        assert out.error is None, out.error
        ran = out.result
        await cells("await functions.get('probe')")
        out = await cells(f"probe({paths!r})")
        assert out.error is None, out.error
        for seen in (ran, out.result):
            assert seen["process"] != harness
            assert seen["env"] == []
            files = seen["files"]
            assert SSH_SECRET not in json.dumps(files)
            assert files[paths[1]] == "FileNotFoundError"
            assert files[paths[2]] in ("FileNotFoundError", "PermissionError")
        # The store is readable (as shipped) but never writable.
        store = db.store_path()
        out = await cells(
            "import sqlite3\n"
            f"con = sqlite3.connect('file:{store}?mode=ro', uri=True)\n"
            "names = [r[0] for r in con.execute('select name from functions')]\n"
            "try:\n"
            f"    sqlite3.connect({store!r}).execute('delete from functions')\n"
            "    wrote = True\n"
            "except sqlite3.OperationalError:\n"
            "    wrote = False\n"
            "(names, wrote)",
        )
        assert out.result == (["probe"], False), out.error
        # Underscored attributes of the harness objects are refused.
        for expression in (
            "functions._fm",
            "functions._actor",
            "guidance._gm",
            "await functions._end(token=1)",
            "functions.search.__self__",
        ):
            out = await cells(expression)
            assert out.error is not None and (
                "do not cross the worker boundary" in out.error
            ), (expression, out.error)
    finally:
        await cells.close()


LOGIN = (
    "def fetch_profile(access_token: str) -> dict:\n"
    '    """The profile an access token opens."""\n'
    "    return {'user': 'ada', 'token_length': len(access_token)}\n"
)
OTHER_SECRET = "tok-other-session-9f8e7d6c5b4a"  # pragma: allowlist secret
DUMP = (
    "def dump_store(path: str) -> str:\n"
    '    """Every text value of every table of a SQLite file."""\n'
    "    import sqlite3\n"
    "    con = sqlite3.connect(f'file:{path}?mode=ro', uri=True)\n"
    "    out = []\n"
    "    tables = [r[0] for r in con.execute(\"select name from sqlite_master where type='table'\")]\n"
    "    for table in tables:\n"
    "        for row in con.execute(f'select * from {table}'):\n"
    "            out.append(repr(row))\n"
    "    return '\\n'.join(out)\n"
)


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(300)
@_handle_project
async def test_a_stored_function_cannot_read_another_sessions_credentials(library):
    """Another process stores and calls a function with a credential; what
    this session's stored code can read of the store never holds it."""
    first = run_in_another_process(
        [
            f"await functions.add({LOGIN!r})",
            f"await functions.run('fetch_profile', access_token={OTHER_SECRET!r})",
            f"await functions.add({DUMP!r})",
        ],
    )
    assert [c["error"] for c in first] == [None, None, None], first
    assert first[1]["result"] == {"user": "ada", "token_length": len(OTHER_SECRET)}
    db.reset_store()
    assert db.query("SELECT count(*) AS n FROM function_cases")[0]["n"] >= 1
    cells = Cells(new_actor(can_store=False))
    try:
        out = await cells(
            f"await functions.run('dump_store', path={db.store_path()!r})",
        )
        assert out.error is None, out.error
        assert "fetch_profile" in out.result and "<redacted:" in out.result
        assert OTHER_SECRET not in out.result
        out = await cells("await functions.search('profile access token', n=5)")
        assert out.error is None and OTHER_SECRET not in json.dumps(
            out.result,
            default=str,
        )
        out = await cells("import os\nsorted(os.environ)")
        assert "FAKE_SERVICE_TOKEN" not in out.result
        assert TOKEN_VALUE not in json.dumps(out.result)
    finally:
        await cells.close()


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(240)
@_handle_project
async def test_a_stored_function_reaches_an_environment_global_by_proxy(
    library,
    monkeypatch,
):
    from tests.actor.code_act import fake_relay_env
    from unify.function_manager.function_manager import FunctionManager
    from unify.function_manager.primitives import (
        EnvironmentMethod,
        EnvironmentNamespace,
        EnvironmentSurface,
        register_environment,
    )
    from unify.function_manager.primitives.environment import (
        clear_environment_namespaces,
    )

    root = Path(tempfile.mkdtemp(prefix="unify-relay-", dir="/tmp"))
    (root / "relay.sock").write_text("relay-token-e2e\n")
    client = fake_relay_env.RelayClient(str(root / "relay.sock"))
    monkeypatch.setattr(fake_relay_env, "relay", client)
    clear_environment_namespaces()
    register_environment(
        EnvironmentSurface(
            namespaces=(
                EnvironmentNamespace(
                    name="music",
                    methods=(
                        EnvironmentMethod(
                            name="show",
                            call=lambda item: client.music.show(item),
                            effect="read",
                            signature="(item: str)",
                        ),
                    ),
                ),
            ),
            globals={"relay": client},
        ),
        source="tests:relay",
    )
    FunctionManager(include_primitives=False).add_functions(
        implementations=[
            "def relay_word(word: str) -> dict:\n"
            '    """Ping the relay."""\n'
            "    return relay.ping(word)\n",
            "def read_relay_file(path: str) -> str:\n"
            '    """Read the relay\'s file directly."""\n'
            "    import pathlib\n"
            "    return pathlib.Path(path).read_text()\n",
        ],
    )
    harness = fake_relay_env.process_id()
    cells = Cells(new_actor(can_store=False))
    try:
        out = await cells("await functions.run('relay_word', word='hi')")
        assert out.error is None, out.error
        assert out.result == {
            "word": "hi",
            "token": "relay-token-e2e",
            "ran_in": harness,
        }
        out = await cells(
            f"await functions.run('read_relay_file', path={str(root / 'relay.sock')!r})",
        )
        assert "FileNotFoundError" in out.error, out.error
        out = await cells("relay._client")
        assert "do not cross the worker boundary" in (out.error or ""), out
    finally:
        await cells.close()
        clear_environment_namespaces()


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(240)
@_handle_project
async def test_a_sub_agent_cannot_be_granted_store_writes_its_caller_lacks(
    library,
    monkeypatch,
):
    from unify.actor import code_act_actor as caa
    from unify.actor.environments.actor import ActorEnvironment

    built: list = []

    class _Inner:
        def __init__(self, *args, **kwargs):
            built.append(kwargs)

        async def act(self, request, **kwargs):
            from unify.actor.simulated import _StaticAnswerHandle

            return _StaticAnswerHandle("child answered")

        async def close(self):
            return None

    real = caa.CodeActActor

    def _factory(*args, **kwargs):
        if "prompt_caching" in kwargs and kwargs.get("timeout") is not None:
            return _Inner(*args, **kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(caa, "CodeActActor", _factory)
    replies = [
        lambda: h.completion(
            calls=[
                (
                    "execute_code",
                    {
                        "thought": "Delegate.",
                        "code": "await primitives.actor.act(request='store it', "
                        "can_store=True)",
                    },
                ),
            ],
        ),
        lambda: h.completion(
            calls=[
                (
                    "execute_code",
                    {
                        "thought": "Store it myself.",
                        "code": "await functions.add('def f():\\n    return 1\\n')",
                    },
                ),
            ],
        ),
        *[lambda: h.completion(content="done")] * 4,
    ]
    parent = real(environments=[ActorEnvironment()], can_store=False)
    try:
        with h.scripted(replies) as provider:
            handle = await parent.act("Store a function.", persist=False)
            assert await asyncio.wait_for(handle.result(), 120) == "done"
    finally:
        await parent.close()
    results = [
        json.dumps(m["content"], default=str)
        for m in provider.requests[-1]["messages"]
        if m.get("role") == "tool"
    ]
    assert "GrantEscalationError" in results[0] and "can_store=True" in results[0]
    assert "PermissionError" in results[1] and "can_store is off" in results[1]
    assert built == []
    assert db.query("SELECT count(*) AS n FROM functions")[0]["n"] == 0


# ── 8: guidance and the shortlist ───────────────────────────────────────────

REQUEST = "Summarise the pairs in order-7f3e9a21: a=1, b=2, a=3. Give per-key totals."


def _first_user(request: dict) -> str:
    return next(m["content"] for m in request["messages"] if m["role"] == "user")


async def _act_once(request: str, replies=None):
    from unify.function_manager import task_origin

    actor = new_actor(can_store=False)
    try:
        with h.scripted(replies or [lambda: h.completion(content="done")] * 4) as p:
            handle = await actor.act(request, persist=False)
            await asyncio.wait_for(handle.result(), 120)
    finally:
        await actor.close()
        task_origin.leave(None)
    return h.session_requests(p.requests)


def _store_in_another_process(**env) -> list:
    return run_in_another_process(
        [
            f"await functions.add({json.dumps(list(HIERARCHY))})",
            "entry = await functions.get('summarize_pairs')\n"
            f"await guidance.add(title={GUIDANCE_TITLE!r}, "
            f"content={GUIDANCE_CONTENT!r}, function_ids=[entry['function_id']])",
        ],
        request=REQUEST,
        env=env,
    )


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(300)
@_handle_project
async def test_guidance_stored_from_code_round_trips_from_code(library):
    first = _store_in_another_process()
    assert [c["error"] for c in first] == [None, None], first
    gid = first[1]["result"]["details"]["guidance_id"]
    db.reset_store()
    cells = Cells(new_actor())
    try:
        out = await cells(
            "[(g.guidance_id, g.title) for g in "
            "await guidance.search('how do I total key value pairs')]",
        )
        assert out.result[0] == (gid, GUIDANCE_TITLE), out
        out = await cells(f"(await guidance.get({gid})).function_ids")
        entry_id = db.query(
            "SELECT function_id FROM functions WHERE name = 'summarize_pairs'",
        )[0]["function_id"]
        assert out.result == [entry_id], out
        out = await cells(
            f"await guidance.update({gid}, content={GUIDANCE_CONTENT + ' Totals are ints.'!r})",
        )
        assert out.error is None, out.error
        out = await cells(f"(await guidance.get({gid})).content")
        assert out.result.endswith("Totals are ints."), out
        # The function knows the guidance that cites it.
        out = await cells(
            "(await functions.get('summarize_pairs')).get('guidance_ids')",
        )
        assert out.result in (None, [gid]), out
        out = await cells(f"await guidance.delete({gid})")
        assert out.error is None, out.error
        out = await cells("await guidance.filter()")
        assert out.result == [], out
    finally:
        await cells.close()


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(300)
@_handle_project
async def test_the_shortlist_lists_the_entry_point_and_its_guidance(
    library,
    monkeypatch,
):
    from unify.actor import library_shortlist as ls

    assert _store_in_another_process()[0]["error"] is None
    db.reset_store()
    monkeypatch.setattr(SETTINGS, "UNIFY_LIBRARY_SHORTLIST", True)
    monkeypatch.setattr(SETTINGS, "UNIFY_LIBRARY_SNAPSHOT", True)
    requests = await _act_once(
        REQUEST,
        [
            lambda: h.completion(
                calls=[
                    (
                        "execute_code",
                        {
                            "thought": "Use the stored entry point.",
                            "code": f"await functions.run('summarize_pairs', text={TEXT!r})",
                        },
                    ),
                ],
            ),
            *[lambda: h.completion(content="done")] * 3,
        ],
    )
    first = _first_user(requests[0])
    assert ls._HEADER in first, first
    block = first[first.index(ls._HEADER) :].split("\n\n", 1)[0]
    assert "- function `summarize_pairs(text: str, sep: str = '; ') -> str`" in block
    assert "- guidance " in block and GUIDANCE_TITLE in block, block
    tool = [m for m in requests[-1]["messages"] if m.get("role") == "tool"]
    assert SUMMARY in json.dumps(tool[0]["content"], default=str), tool


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(300)
@_handle_project
@pytest.mark.parametrize("provenance", [False, True])
async def test_the_gated_shortlist_names_the_entry_point_stored_for_this_request(
    library,
    monkeypatch,
    provenance,
):
    from unify.actor import library_shortlist as ls

    stored = _store_in_another_process(UNIFY_TASK_ORIGIN="true")
    assert [c["error"] for c in stored] == [None, None], stored
    db.reset_store()
    monkeypatch.setattr(SETTINGS, "UNIFY_LIBRARY_SHORTLIST", True)
    monkeypatch.setattr(SETTINGS, "UNIFY_TASK_ORIGIN", True)
    monkeypatch.setattr(SETTINGS, "UNIFY_SHORTLIST_GATE", "similar_request:0.5")
    monkeypatch.setattr(SETTINGS, "UNIFY_ORIGIN_PROVENANCE", provenance)
    requests = await _act_once(REQUEST)
    first = _first_user(requests[0])
    assert ls._GATED_HEADER in first, first
    block = first[first.index(ls._GATED_HEADER) :].split("\n\n", 1)[0]
    lines = block.splitlines()[1:]
    entry = [l for l in lines if l.startswith("- function `summarize_pairs(")]
    assert len(entry) == 1, block
    assert "similar_request 1.00" in entry[0], entry
    # Guidance records no request, so the gated list never names it.
    assert GUIDANCE_TITLE not in block and "- guidance" not in block
    if provenance:
        assert "same request" in entry[0], entry
    else:
        assert "same request" not in entry[0], entry
    # Another request gets no list.
    requests = await _act_once("Plan a week of vegetarian dinners for two.")
    assert ls._GATED_HEADER not in _first_user(requests[0])

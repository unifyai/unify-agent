"""The workspace environment: one persistent venv under ``UNIFY_HOME``.

A stored function records its third-party requirements as ``dependencies``
and runs in-process once they are present; the actor's install tool puts a
package into the environment and every later cell can import it.
"""

from __future__ import annotations

import importlib
import sys

import pytest

from tests.helpers import _handle_project
from unify import environment
from unify.actor.code_act_actor import CodeActActor
from unify.actor.execution import parts_to_text
from unify.function_manager.function_manager import FunctionManager

# Absent from the runtime's own environment, so an install has to happen.
_ABSENT_PACKAGE = "humanize"

_USES_RUNTIME_DEPENDENCY = (
    "def parse_version(text: str) -> str:\n"
    "    from packaging.version import Version\n"
    "    return str(Version(text))\n"
)


def _forget_absent_package() -> None:
    sys.modules.pop(_ABSENT_PACKAGE, None)
    importlib.invalidate_caches()


@pytest.fixture
def workspace_home(unify_home, monkeypatch):
    """The test's own ``UNIFY_HOME``, whose environment does not exist yet.

    ``sys.path`` is restored on exit, so an environment activated by one
    test is not still importable in the next.
    """
    monkeypatch.setattr(sys, "path", list(sys.path))
    _forget_absent_package()
    yield unify_home
    _forget_absent_package()


@pytest.fixture
def in_process(monkeypatch):
    """Cells in this process (``UNIFY_WORKSPACE_PYTHON`` empty): the harness
    then imports from the environment and runs stored functions itself. With
    Python in the sandboxed worker it does neither
    (test_bind_load_confinement.py)."""
    monkeypatch.setattr("unify.actor.execution.worker.enabled", lambda: False)


# ---------------------------------------------------------------------------
# Recording dependencies
# ---------------------------------------------------------------------------


@_handle_project
def test_third_party_import_requires_dependencies():
    fm = FunctionManager()
    with pytest.raises(ValueError, match="dependencies"):
        fm.add_functions(implementations=_USES_RUNTIME_DEPENDENCY)
    assert "parse_version" not in fm.list_functions()


@_handle_project
def test_dependencies_must_be_requirement_strings():
    fm = FunctionManager()
    with pytest.raises(ValueError, match="not a valid requirement"):
        fm.add_functions(
            implementations=_USES_RUNTIME_DEPENDENCY,
            dependencies=["packaging >>= 1"],
        )


@_handle_project
def test_dependencies_are_recorded_on_the_function():
    fm = FunctionManager()
    fm.add_functions(
        implementations=_USES_RUNTIME_DEPENDENCY,
        dependencies=["packaging>=20"],
    )
    row = fm.list_functions()["parse_version"]
    assert row["dependencies"] == ["packaging>=20"]
    assert row["third_party_imports"] == ["packaging"]


@_handle_project
@pytest.mark.asyncio
async def test_execute_function_runs_a_function_with_satisfied_dependencies(
    in_process,
):
    """A dependency the runtime already provides needs no install."""
    fm = FunctionManager()
    fm.add_functions(
        implementations=_USES_RUNTIME_DEPENDENCY,
        dependencies=["packaging>=20"],
    )
    result = await fm.execute_function(
        function_name="parse_version",
        call_kwargs={"text": "2.0"},
    )
    assert result["error"] is None
    assert result["result"] == "2.0"


# ---------------------------------------------------------------------------
# The environment itself
# ---------------------------------------------------------------------------


def test_missing_reports_only_unsatisfied_specifiers(workspace_home):
    assert environment.missing(["packaging", "packaging>=20"]) == []
    assert environment.missing(["packaging>=999"]) == ["packaging>=999"]
    assert environment.missing([_ABSENT_PACKAGE]) == [_ABSENT_PACKAGE]


def test_activate_is_a_no_op_until_the_environment_exists(workspace_home, in_process):
    assert environment.activate() is None
    assert not environment.environment_dir().exists()


@pytest.fixture
def installed(tmp_path, monkeypatch):
    """An environment holding one package, with its dist-info."""
    packages = tmp_path / "venv-site-packages"
    package = packages / "bindprobe"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("LOADED_IN = 'somewhere'\n")
    info = packages / "bindprobe-1.2.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: bindprobe\nVersion: 1.2\n",
    )
    monkeypatch.setattr(environment, "site_packages", lambda: packages)
    yield packages
    if str(packages) in sys.path:
        sys.path.remove(str(packages))
    sys.modules.pop("bindprobe", None)


def test_with_python_in_the_worker_the_harness_reads_the_environment_by_path(
    workspace_home,
    installed,
    monkeypatch,
):
    """The harness holds the provider credentials, so it never imports what
    an install put in the environment: only the worker does."""
    from unify.function_manager import store_check

    assert environment.missing(["bindprobe>=1", "bindprobe>=2"]) == ["bindprobe>=2"]
    assert environment.missing(["packaging>=20"]) == []
    environment.ensure(["bindprobe==1.2"])
    with pytest.raises(RuntimeError, match="workspace environment"):
        environment.activate()
    # The store check still sees what the worker can import from there.
    assert store_check._in_workspace_environment("bindprobe")
    assert not store_check._in_workspace_environment("bindprobe_absent")
    assert str(installed) not in sys.path
    with pytest.raises(ImportError):
        importlib.import_module("bindprobe")


def test_in_process_the_environment_joins_sys_path(
    workspace_home,
    installed,
    in_process,
):
    assert environment.missing(["bindprobe==1.2"]) == []
    assert str(installed) in sys.path
    assert importlib.import_module("bindprobe").LOADED_IN == "somewhere"


@pytest.mark.timeout(180)
def test_install_makes_a_package_importable_in_process(workspace_home, in_process):
    outcome = environment.install([_ABSENT_PACKAGE])
    assert outcome["success"], outcome["stderr"]
    assert outcome["packages"] == [_ABSENT_PACKAGE]

    packages = environment.site_packages()
    assert environment.environment_dir() == workspace_home / "venv"
    assert str(packages) in sys.path
    module = importlib.import_module(_ABSENT_PACKAGE)
    assert module.__file__.startswith(str(packages))
    assert environment.missing([_ABSENT_PACKAGE]) == []


@_handle_project
@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_execute_function_installs_missing_dependencies(
    workspace_home,
    monkeypatch,
):
    """A stored function's dependencies are ensured before it runs."""
    fm = FunctionManager()
    # Stored with Python in the worker, whose store check installs nothing
    # (in process it loads the function, installing its dependencies).
    fm.add_functions(
        implementations=(
            "def humanise(n: int) -> str:\n"
            "    import humanize\n"
            "    return humanize.intcomma(n)\n"
        ),
        dependencies=[_ABSENT_PACKAGE],
    )
    assert environment.missing([_ABSENT_PACKAGE]) == [_ABSENT_PACKAGE]
    monkeypatch.setattr("unify.actor.execution.worker.enabled", lambda: False)

    result = await fm.execute_function(
        function_name="humanise",
        call_kwargs={"n": 1234567},
    )
    assert result["error"] is None
    assert result["result"] == "1,234,567"
    assert environment.missing([_ABSENT_PACKAGE]) == []


@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_install_tool_makes_a_package_importable_in_execute_code(
    workspace_home,
):
    actor = CodeActActor(environments=[])
    try:
        tools = actor._build_tools()
        install = tools["install_python_packages"]
        execute_code = tools["execute_code"]

        outcome = await install(packages=[_ABSENT_PACKAGE])
        assert outcome["success"], outcome["stderr"]

        out = await execute_code(
            thought="Use the package that was just installed.",
            code="import humanize\nprint(humanize.naturalsize(1000))",
        )
        assert out.error is None
        assert "1.0 kB" in parts_to_text(out.stdout)
    finally:
        await actor.close()

"""The workspace environment: one persistent venv under ``UNIFY_HOME``.

An install puts a package into the environment and every later cell can import
it; the harness itself never imports from it.

The installs are real (``uv`` under bubblewrap) but never reach the network:
the package is a wheel the test builds, served from a package index on disk
(``local_index``).
"""

from __future__ import annotations

import base64
import hashlib
import importlib
import sys
import zipfile
from pathlib import Path

import pytest

from tests.actor.code_act.sandbox_world import actor_interpreter, needs_bwrap
from unify import environment

# Absent from the runtime's own environment, so an install has to happen.
_ABSENT_PACKAGE = "group_digits"
_ABSENT_SOURCE = "def group(n: int) -> str:\n    return f'{n:,}'\n"

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


def _wheel(directory: Path, name: str, version: str, source: str) -> Path:
    """A pure-Python wheel of the module *name* with *source*, in *directory*."""
    info = f"{name}-{version}.dist-info"
    files = {
        f"{name}/__init__.py": source,
        f"{info}/METADATA": (
            f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
        ),
        f"{info}/WHEEL": (
            "Wheel-Version: 1.0\nGenerator: tests\n"
            "Root-Is-Purelib: true\nTag: py3-none-any\n"
        ),
    }
    rows = []
    for path, text in files.items():
        data = text.encode()
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest())
        rows.append(f"{path},sha256={digest.rstrip(b'=').decode()},{len(data)}")
    rows.append(f"{info}/RECORD,,")
    wheel = directory / f"{name}-{version}-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        for path, text in files.items():
            archive.writestr(path, text)
        archive.writestr(f"{info}/RECORD", "\n".join(rows) + "\n")
    return wheel


@pytest.fixture
def local_index(workspace_home, monkeypatch):
    """A package index on disk serving one wheel of :data:`_ABSENT_PACKAGE`,
    configured as the operator's index (``UV_DEFAULT_INDEX``).

    It lives in the installer's own cache, the one directory beside the
    environment that the installer's sandbox mounts; the Unify home and the
    host's ``/tmp`` are hidden from it.
    """
    root = environment.installer_cache() / "local-index"
    project = root / _ABSENT_PACKAGE.replace("_", "-")
    project.mkdir(parents=True)
    wheel = _wheel(project, _ABSENT_PACKAGE, "1.0", _ABSENT_SOURCE)
    # the simple repository pages (PEP 503): the projects, then each one's files
    page = '<!DOCTYPE html><html><body><a href="{0}">{0}</a></body></html>\n'
    (project / "index.html").write_text(page.format(wheel.name))
    (root / "index.html").write_text(page.format(f"{project.name}/"))
    for name in ("UV_INDEX", "UV_INDEX_URL", "UV_EXTRA_INDEX_URL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("UV_DEFAULT_INDEX", root.as_uri())
    return root


@needs_bwrap
@pytest.mark.timeout(180)
def test_an_install_lands_in_the_environment_and_the_harness_never_imports_it(
    workspace_home,
    local_index,
):
    outcome = environment.install([_ABSENT_PACKAGE])
    assert outcome["success"], outcome["stderr"]
    assert outcome["packages"] == [_ABSENT_PACKAGE]
    packages = environment.site_packages()
    assert environment.environment_dir() == workspace_home / "venv"
    assert (packages / _ABSENT_PACKAGE / "__init__.py").exists()
    assert str(packages) not in sys.path
    with pytest.raises(ImportError):
        importlib.import_module(_ABSENT_PACKAGE)


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(240)
async def test_a_cell_imports_what_an_install_put_in_the_environment(
    workspace_home,
    local_index,
):
    outcome = environment.install([_ABSENT_PACKAGE])
    assert outcome["success"], outcome["stderr"]
    interp = await actor_interpreter()
    try:
        res = await interp.run_cell("import group_digits\nprint(group_digits.group(1000))")
        assert res.error is None, res.text()
        assert "1,000" in res.output
    finally:
        await interp.stop()

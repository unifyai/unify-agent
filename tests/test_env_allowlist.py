"""Every ``UNIFY_*`` environment variable the package reads is documented in CONFIG.md, and every one CONFIG.md lists
is read: the configuration is exactly what CONFIG.md says, nothing hidden and nothing stale.

A variable is read when a settings class declares it as a field (pydantic reads fields from the environment) or when
code reads it from ``os.environ`` / ``os.getenv`` by a literal name. The test-only variables (``UNIFY_TEST_*`` and the
ones CONFIG.md names as test-only) are left out on both sides. Nothing is imported or run: the sources are parsed.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "unify"
CONFIG = ROOT / "CONFIG.md"
NAME = re.compile(r"UNIFY_[A-Z0-9_]*[A-Z0-9]")
#: read only by the test suite (CONFIG.md's "Test-only" section)
TEST_ONLY = re.compile(r"UNIFY_TEST_\w+|UNIFY_RECORD_GOLDEN")


def _settings_fields(tree: ast.Module) -> set[str]:
    """The ``UNIFY_*`` fields of every class whose bases name a settings class (``BaseSettings`` or a class of this
    module that derives from one)."""
    settings_classes: set[str] = {"BaseSettings"}
    out: set[str] = set()
    changed = True
    classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
    while changed:
        changed = False
        for c in classes:
            bases = {b.id if isinstance(b, ast.Name) else getattr(b, "attr", "") for b in c.bases}
            if c.name not in settings_classes and bases & settings_classes:
                settings_classes.add(c.name)
                changed = True
    for c in classes:
        if c.name in settings_classes:
            for b in c.body:
                if isinstance(b, ast.AnnAssign) and isinstance(b.target, ast.Name) and NAME.fullmatch(b.target.id):
                    out.add(b.target.id)
    return out


def _is_environ(node: ast.AST) -> bool:
    # os.environ, environ
    return (isinstance(node, ast.Attribute) and node.attr == "environ") or (
        isinstance(node, ast.Name) and node.id == "environ"
    )


def _literal(node: ast.AST | None) -> str | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _constants(tree: ast.Module) -> dict[str, str]:
    """The module's top-level ``NAME = "..."`` string constants."""
    out: dict[str, str] = {}
    for b in tree.body:
        if isinstance(b, ast.Assign) and len(b.targets) == 1 and isinstance(b.targets[0], ast.Name):
            value = _literal(b.value)
            if value is not None:
                out[b.targets[0].id] = value
        elif isinstance(b, ast.AnnAssign) and isinstance(b.target, ast.Name):
            value = _literal(b.value)
            if value is not None:
                out[b.target.id] = value
    return out


def _env_reads(tree: ast.Module) -> set[str]:
    """``UNIFY_*`` names read from the environment by a literal name, or by a module constant that holds one:
    ``os.environ.get/pop/setdefault(name)``, ``os.environ[name]``, ``os.getenv(name)``, ``name in os.environ``, and a
    scan of the environment for every case variant of a name (``[k for k in environ if k.upper() == name]``)."""
    consts = _constants(tree)

    def named(node: ast.AST | None) -> str | None:
        if isinstance(node, ast.Name):
            return consts.get(node.id)
        return _literal(node)

    out: set[str] = set()
    for n in ast.walk(tree):
        names: list[str | None] = []
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.args:
            if n.func.attr in ("get", "pop", "setdefault") and _is_environ(n.func.value):
                names.append(named(n.args[0]))
            elif n.func.attr == "getenv":
                names.append(named(n.args[0]))
        elif isinstance(n, ast.Subscript) and _is_environ(n.value) and isinstance(n.ctx, ast.Load):
            names.append(named(n.slice))
        elif isinstance(n, ast.Compare) and len(n.ops) == 1 and isinstance(n.ops[0], (ast.In, ast.NotIn)):
            if _is_environ(n.comparators[0]):
                names.append(named(n.left))
        elif isinstance(n, ast.comprehension) and _is_environ(n.iter):
            for cond in n.ifs:
                if isinstance(cond, ast.Compare) and len(cond.ops) == 1 and isinstance(cond.ops[0], ast.Eq):
                    names.append(named(cond.comparators[0]))
        out.update(name for name in names if name and NAME.fullmatch(name))
    return out


def package_reads() -> dict[str, list[str]]:
    """Every ``UNIFY_*`` variable the package reads, with the files that read it."""
    found: dict[str, list[str]] = {}
    for path in sorted(PACKAGE.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for name in _settings_fields(tree) | _env_reads(tree):
            found.setdefault(name, []).append(str(path.relative_to(ROOT)))
    return {k: v for k, v in found.items() if not TEST_ONLY.fullmatch(k)}


def documented() -> set[str]:
    """Every ``UNIFY_*`` name CONFIG.md lists in code spans, outside its test-only section."""
    text = CONFIG.read_text(encoding="utf-8")
    text = text.split("## Test-only", 1)[0]
    return {m for span in re.findall(r"`([^`]+)`", text) for m in NAME.findall(span)}


def test_config_lists_every_variable_the_package_reads():
    reads = package_reads()
    undocumented = {k: v for k, v in reads.items() if k not in documented()}
    assert not undocumented, f"read by the package but not in CONFIG.md: {undocumented}"


def test_config_lists_no_variable_the_package_does_not_read():
    stale = documented() - set(package_reads())
    assert not stale, f"in CONFIG.md but read nowhere in the package: {sorted(stale)}"


def test_the_embedding_lanes_token_is_read_and_documented():
    """``UNIFY_EMBED_TOKEN_FD`` is read by a case-insensitive scan of the environment (every variant of the name is
    then taken out of it), not by one literal lookup: the scan above counts it, and CONFIG.md lists it."""
    assert package_reads().get("UNIFY_EMBED_TOKEN_FD") == ["unify/lanes/embed_token.py"]
    assert "UNIFY_EMBED_TOKEN_FD" in documented()


def test_memory_has_exactly_one_switch():
    """Of the memory variables, only UNIFY_MEMORY changes what the harness does; the rest are the writer's model, its
    spend cap and the lanes a launcher sets."""
    memory = {k for k in package_reads() if k.startswith("UNIFY_MEMORY")}
    assert memory == {
        "UNIFY_MEMORY",
        "UNIFY_MEMORY_WRITER_MODEL",
        "UNIFY_MEMORY_CAP_USD",
        "UNIFY_MEMORY_WRITER_BASE_URL",
        "UNIFY_MEMORY_WRITER_TOKEN_FD",
        "UNIFY_MEMORY_WRITER_JOURNAL",
        "UNIFY_MEMORY_WRITER_PRICES_USD_PER_MILLION",
        "UNIFY_MEMORY_LIBRARY_FORK_BASE_URL",
        "UNIFY_MEMORY_LIBRARY_FORK_TOKEN_FD",
        "UNIFY_MEMORY_LIBRARY_HELPER_BASE_URL",
        "UNIFY_MEMORY_LIBRARY_HELPER_TOKEN_FD",
    }


@pytest.mark.parametrize("value", ["off", "on", "in_episode", "", "ON", " In_Episode "])
def test_the_switch_takes_its_three_values(value):
    from unify.memory.integration import switch

    assert switch.parse_memory(value) == (value.strip().lower() or "on")


@pytest.mark.parametrize("value", ["yes", "1", "repin", "paper", "true", "on,in_episode"])
def test_any_other_value_is_refused_with_the_three_values(value):
    from unify.memory.integration import switch

    with pytest.raises(ValueError) as exc:
        switch.parse_memory(value)
    assert "off, on, in_episode" in str(exc.value)


def test_the_default_is_memory_between_episodes(monkeypatch):
    from types import SimpleNamespace

    from unify.memory.integration import switch
    from unify.settings import ProductionSettings

    monkeypatch.delenv("UNIFY_MEMORY", raising=False)
    assert ProductionSettings.model_fields["UNIFY_MEMORY"].default == "on"
    assert switch.memory_mode(SimpleNamespace()) == "on"
    for value, on, ep in (("off", False, False), ("on", True, False), ("in_episode", True, True)):
        s = SimpleNamespace(UNIFY_MEMORY=value)
        assert (switch.memory_mode(s), switch.memory_on(s), switch.in_episode(s)) == (value, on, ep)


def test_the_settings_refuse_another_value(monkeypatch):
    from unify.settings import ProductionSettings

    monkeypatch.setenv("UNIFY_MEMORY", "repin")
    with pytest.raises(ValueError, match="off, on, in_episode"):
        ProductionSettings()


def test_the_writer_runs_on_its_own_model_or_the_runs():
    from types import SimpleNamespace

    from unify.memory.integration import switch

    run = SimpleNamespace(UNIFY_MODEL="vendor/actor@openrouter", UNIFY_MEMORY_WRITER_MODEL="")
    assert switch.writer_model(run) == "vendor/actor@openrouter"
    run.UNIFY_MEMORY_WRITER_MODEL = " vendor/writer "
    assert switch.writer_model(run) == "vendor/writer"


@pytest.mark.parametrize("value", ["20.00", "0.01", "7.5"])
def test_the_cap_is_a_plain_decimal(value):
    from decimal import Decimal
    from types import SimpleNamespace

    from unify.memory.integration import switch

    assert switch.memory_cap_usd(SimpleNamespace(UNIFY_MEMORY_CAP_USD=value)) == Decimal(value)


@pytest.mark.parametrize("value", ["0", "1e1", "-1", "two", "$5"])
def test_any_other_cap_is_refused(value):
    from types import SimpleNamespace

    from unify.memory.integration import switch

    with pytest.raises(ValueError):
        switch.memory_cap_usd(SimpleNamespace(UNIFY_MEMORY_CAP_USD=value))

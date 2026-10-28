#!/usr/bin/env python3
import re
import sys
import tomllib
from pathlib import Path

# This repository's distribution and its first-party dependency.
FIRST_PARTY = {
    "unillm",
    "unify-agent",
}
INERT_VERSION = "0.0.0"


def _is_first_party(name: str | None) -> bool:
    return bool(name and name in FIRST_PARTY)


def _check_pyproject(root: Path, failures: list[str]) -> None:
    path = root / "pyproject.toml"
    if not path.exists():
        return

    data = tomllib.loads(path.read_text())
    project = data.get("project") or {}
    poetry = data.get("tool", {}).get("poetry") or {}
    name = project.get("name") or poetry.get("name")
    version = project.get("version") or poetry.get("version")
    if _is_first_party(name) and version != INERT_VERSION:
        failures.append(
            f"{path}: first-party package version must be {INERT_VERSION!r}",
        )

    text = path.read_text()
    if re.search(r"\b(?:tag|rev)\s*=", text):
        failures.append(
            f"{path}: first-party dependencies must use branch refs, not tags/revs",
        )
    if re.search(r"github\.com/unifyai/[^ \]\"'}]+\.git@v?\d", text):
        failures.append(
            f"{path}: first-party Git dependencies must not use version-like refs",
        )


def main() -> int:
    root = Path.cwd()
    failures: list[str] = []
    _check_pyproject(root, failures)
    if failures:
        print("\n".join(f"FAIL: {failure}" for failure in failures))
        return 1
    print("OK: no first-party release versioning found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Start the fork's relay (:mod:`unify.agents.fork_relay`) with no route to a provider key: run by path
(``python -I <this file>``), never as ``-m unify...``, because importing the ``unify`` package runs its ``__init__``,
which imports the settings, the logger and the model client, and those read ``.env`` files and fetch provider keys.

Standard library only. It drops the script's folder from the path, refuses every import of a module that reads
provider settings or keys (:data:`FORBIDDEN`: a refused import raises, so a code path that would need one fails
instead of reading anything), puts this tree first on the path, stands in for the ``unify`` and ``unify.common``
packages (their submodules load; their ``__init__`` files never run), and runs the relay. ``--modules`` instead
imports what the relay runs and prints the modules loaded, one per line (tests).
"""

import importlib.abc
import os
import sys
import types

#: Modules the relay never imports: the model client, the settings and the logger (which reads them), dotenv, and
#: every provider or cloud SDK. Prefixes: a submodule of one is refused too.
FORBIDDEN = (
    "unillm",
    "unify.settings",
    "unify.logger",
    "unify.common.llm_client",
    "unify.common.async_tool_loop",
    "dotenv",
    "litellm",
    "openai",
    "anthropic",
    "google",
    "boto3",
    "botocore",
    "httpx",
    "requests",
    "pydantic_settings",
)


class Refuse(importlib.abc.MetaPathFinder):
    """Refuses :data:`FORBIDDEN` imports, and notes each on stderr (the relay's log)."""

    def find_spec(self, name, path=None, target=None):
        if any(name == f or name.startswith(f + ".") for f in FORBIDDEN):
            print(f"fork relay: refused import {name}", file=sys.stderr, flush=True)
            raise ImportError(f"the fork relay never imports {name}")
        return None


def _package(name: str, folder: str) -> None:
    mod = types.ModuleType(name)
    mod.__path__ = [folder]
    mod.__package__ = name
    sys.modules[name] = mod


def boot() -> str:
    """Set up the import guard and the stand-in packages; returns this tree's root."""
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(os.path.dirname(here))
    sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != here]
    sys.path.insert(0, root)
    sys.meta_path.insert(0, Refuse())
    _package("unify", os.path.join(root, "unify"))
    _package("unify.common", os.path.join(root, "unify", "common"))
    return root


def main() -> int:
    boot()
    if sys.argv[1:] == ["--modules"]:
        from unify.agents import fork_relay

        fork_relay.load_all()
        print("\n".join(sorted(sys.modules)))
        return 0
    from unify.agents import fork_relay

    return fork_relay.main()


if __name__ == "__main__":
    raise SystemExit(main())

"""No real provider key may reach a memory v2 test (incident 8 Oct 2026: a laptop run built request bodies with one).

At import, before any test here runs:

* record whether a real provider key was loadable (set in the environment, or fetchable through unillm's Secret
  Manager source because its service-account key file exists), comparing values only, never printing them;
* replace every provider key, in the environment and in unillm's loaded settings, with a placeholder, so request
  bodies a test builds can only ever hold the placeholder;
* do the same check for memory v2's Sol route (``UNIFY_MEMORY_V2_SOL_TOKEN`` / ``_BASE_URL``) and drop it from the
  environment.

``test_no_real_provider_key_is_loadable`` then fails the suite on a host where a real key was loadable: these tests
run only on keyless hosts.
"""

from __future__ import annotations

import os
import sys

PLACEHOLDER = "test-placeholder-not-a-key"  # pragma: allowlist secret

try:
    from unillm import settings as _unillm_settings

    PROVIDER_KEYS = tuple(_unillm_settings.PROVIDER_KEYS)
    _KEY_FILE = _unillm_settings.SERVICE_ACCOUNT_KEY
except Exception:  # noqa: BLE001 - unillm absent: nothing can load a key through it
    _unillm_settings = None
    PROVIDER_KEYS = (
        "OPENROUTER_API_KEY",
        "OPENROUTER_MANAGEMENT_API_KEY",
        "TOGETHER_API_KEY",
        "ANTHROPIC_API_KEY",
    )
    _KEY_FILE = None


def _secret_value(value: object) -> str:
    getter = getattr(value, "get_secret_value", None)
    return str(getter() if callable(getter) else (value or ""))


def _real_key_seen() -> list[str]:
    """Names of provider keys that held a real value (never the values)."""
    seen = [
        n for n in PROVIDER_KEYS if os.environ.get(n) not in (None, "", PLACEHOLDER)
    ]
    loaded = getattr(_unillm_settings, "SETTINGS", None) if _unillm_settings else None
    for n in PROVIDER_KEYS:
        if loaded is not None and _secret_value(getattr(loaded, n, "")) not in (
            "",
            PLACEHOLDER,
        ):
            seen.append(f"{n} (loaded)")
    if _KEY_FILE is not None and _KEY_FILE.is_file():
        seen.append(
            "service-account key file present (Secret Manager source can fetch keys)",
        )
    return sorted(set(seen))


#: Sol's route (unify/memory_v2/integration/switch.py): its token is a credential as well, and a test never runs
#: with a real route. The controller's settings may already hold it (unify.settings removes it from the
#: environment once read), so both places are looked at; names only, never values.
SOL_ROUTE = (
    "UNIFY_MEMORY_V2_SOL_TOKEN",
    "UNIFY_MEMORY_V2_SOL_BASE_URL",
    "UNIFY_MEMORY_V2_SOL_TOKEN_FD",
)


def _sol_token_seen() -> list[str]:
    name = SOL_ROUTE[0]
    seen = [
        n for n in os.environ if n.upper() == name and (os.environ[n] or "").strip()
    ]
    loaded = getattr(sys.modules.get("unify.settings"), "SETTINGS", None)
    if loaded is not None and _secret_value(getattr(loaded, name, "")).strip():
        seen.append(f"{name} (loaded)")
    switch = sys.modules.get("unify.memory_v2.integration.switch")
    if getattr(switch, "_FD_TOKEN", None) is not None:  # read from its descriptor
        seen.append(f"{SOL_ROUTE[2]} (read)")
    return seen


REAL_KEY_SEEN = sorted(set(_real_key_seen() + _sol_token_seen()))

for _name in [n for n in os.environ if n.upper() in SOL_ROUTE]:  # any letter case
    os.environ.pop(_name, None)

for _name in PROVIDER_KEYS:
    os.environ[_name] = PLACEHOLDER
    _loaded = getattr(_unillm_settings, "SETTINGS", None) if _unillm_settings else None
    if _loaded is not None and hasattr(_loaded, _name):
        try:
            current = getattr(_loaded, _name)
            replacement = (
                type(current)(PLACEHOLDER)
                if hasattr(current, "get_secret_value")
                else PLACEHOLDER
            )
            object.__setattr__(_loaded, _name, replacement)
        except Exception:  # noqa: BLE001 - the guard test below still fails the suite
            pass

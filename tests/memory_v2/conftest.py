"""No real provider key may reach a memory v2 test (incident 8 Oct 2026: a laptop run built request bodies with one).

At import, before any test here runs:

* record whether a real provider key was loadable (set in the environment, or fetchable through unillm's Secret
  Manager source because its service-account key file exists), comparing values only, never printing them;
* replace every provider key, in the environment and in unillm's loaded settings, with a placeholder, so request
  bodies a test builds can only ever hold the placeholder.

``test_no_real_provider_key_is_loadable`` then fails the suite on a host where a real key was loadable: these tests
run only on keyless hosts.
"""

from __future__ import annotations

import os

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


REAL_KEY_SEEN = _real_key_seen()

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

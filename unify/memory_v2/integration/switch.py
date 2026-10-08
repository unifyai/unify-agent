"""Value parsing for the UNIFY_MEMORY_V2 settings (unify/settings.py calls these validators).

The contract (online build, spec §F1 and D23):

- ``UNIFY_MEMORY_V2``: ``on``, ``off`` or empty (empty and ``off`` mean off).
- ``UNIFY_MEMORY_V2_E``: the experience budget E in tokens, a positive int; empty means 150000. A pass
  becomes due once the evidence since the last pass reaches E, and its USD cap is E times the allowance.
- ``UNIFY_MEMORY_V2_SOL_MODEL``: the model that runs consolidation passes; empty means
  ``openai/gpt-6-sol``. Sol's reasoning effort has no switch: it is the actor's effort for the run.
- ``UNIFY_MEMORY_V2_SOL_ALLOWANCE_USD_PER_TOKENS``: USD Sol may spend per token of experience, a positive
  plain decimal string; empty means ``0.00000073``.
- ``UNIFY_MEMORY_V2_SOL_RUN_GUARD_USD``: a non-negative plain decimal string, or empty for no guard. When
  set, no further pass starts once the run's committed Sol USD plus the next pass's cap would exceed it.
- ``UNIFY_MEMORY_V2_SOL_BASE_URL`` and ``UNIFY_MEMORY_V2_SOL_TOKEN``: Sol's own route, both or neither. Set,
  Sol's model calls go to that OpenAI-compatible base URL with that token (a proxy listener of Sol's own,
  so the actor's route never carries Sol's model); empty, they go as shipped. The URL is http(s) with a
  host and no user, password, query or fragment. The token is a secret: printable ASCII without spaces,
  held as a ``SecretStr``. Settings only normalise the two (a settings error would print its input), and
  :func:`sol_route` checks them when a pass is about to start; its errors never quote either value.

Money stays a decimal string as written (never a float), and exponent forms are refused, so a value is
read the same way by every consumer.
"""

from __future__ import annotations

import re
from decimal import Decimal
from typing import Any
from urllib.parse import urlsplit

from pydantic import SecretStr

SWITCH = "UNIFY_MEMORY_V2"
EXPERIENCE_BUDGET = "UNIFY_MEMORY_V2_E"
SOL_MODEL = "UNIFY_MEMORY_V2_SOL_MODEL"
SOL_ALLOWANCE = "UNIFY_MEMORY_V2_SOL_ALLOWANCE_USD_PER_TOKENS"
SOL_RUN_GUARD = "UNIFY_MEMORY_V2_SOL_RUN_GUARD_USD"
SOL_BASE_URL = "UNIFY_MEMORY_V2_SOL_BASE_URL"
SOL_TOKEN = "UNIFY_MEMORY_V2_SOL_TOKEN"

EXPERIENCE_BUDGET_DEFAULT = 150000
SOL_MODEL_DEFAULT = "openai/gpt-6-sol"
SOL_ALLOWANCE_DEFAULT = "0.00000073"

#: A plain decimal: digits, optionally a point and more digits. No sign, exponent, separator or name.
_PLAIN_DECIMAL = re.compile(r"[0-9]+(?:\.[0-9]+)?")
_DIGITS = re.compile(r"[0-9]+")


def _stripped(v: Any) -> str:
    return str(v if v is not None else "").strip()


def parse_switch(v: Any) -> str:
    """``on``, or ``""`` for off (empty or ``off``)."""
    value = _stripped(v).lower()
    if value in ("", "off"):
        return ""
    if value != "on":
        raise ValueError(f"{SWITCH} must be empty, 'off' or 'on', not {v!r}")
    return value


def parse_experience_budget(v: Any) -> int:
    """A positive token count, written as digits only."""
    refusal = (
        f"{EXPERIENCE_BUDGET} must be a positive whole number of tokens, not {v!r}"
    )
    if isinstance(v, bool):
        raise ValueError(refusal)
    if isinstance(v, int):
        value = v
    else:
        text = _stripped(v)
        if text == "":
            return EXPERIENCE_BUDGET_DEFAULT
        if not _DIGITS.fullmatch(text):
            raise ValueError(refusal)
        value = int(text)
    if value <= 0:
        raise ValueError(refusal)
    return value


def parse_sol_model(v: Any) -> str:
    value = _stripped(v) or SOL_MODEL_DEFAULT
    if any(c.isspace() for c in value):
        raise ValueError(f"{SOL_MODEL} must be a model id without spaces, not {v!r}")
    return value


def _plain_decimal(name: str, text: str, v: Any, *, positive: bool) -> str:
    if not _PLAIN_DECIMAL.fullmatch(text):
        raise ValueError(
            f"{name} must be a plain decimal USD amount (digits and at most one point, "
            f"no sign or exponent), not {v!r}",
        )
    if positive and Decimal(text) <= 0:
        raise ValueError(f"{name} must be above zero, not {v!r}")
    return text


def parse_sol_allowance(v: Any) -> str:
    """USD per token of experience: a positive plain decimal string."""
    text = _stripped(v) or SOL_ALLOWANCE_DEFAULT
    return _plain_decimal(SOL_ALLOWANCE, text, v, positive=True)


def parse_sol_run_guard(v: Any) -> str:
    """The run's Sol USD guard: a non-negative plain decimal string, or ``""`` for none."""
    text = _stripped(v)
    if text == "":
        return ""
    return _plain_decimal(SOL_RUN_GUARD, text, v, positive=False)


def sol_base_url_setting(v: Any) -> str:
    """``UNIFY_MEMORY_V2_SOL_BASE_URL`` as settings hold it: stripped, never refused (see :func:`sol_route`)."""
    return _stripped(v)


def sol_token_setting(v: Any) -> SecretStr:
    """``UNIFY_MEMORY_V2_SOL_TOKEN`` as settings hold it: a stripped ``SecretStr``, never refused."""
    getter = getattr(v, "get_secret_value", None)
    return SecretStr(_stripped(getter() if callable(getter) else v))


def parse_sol_base_url(v: Any) -> str:
    """An http(s) base URL with a host and no user, password, query or fragment; ``""`` when empty.

    Errors name the rule broken, never the value (a URL with a password in it is a credential).
    """
    text = _stripped(v)
    if text == "":
        return ""
    refusal = f"{SOL_BASE_URL} must be an http(s) URL with a host and no user, password, query or fragment"
    if any(c.isspace() or not c.isprintable() for c in text):
        raise ValueError(f"{refusal}: it holds a space or control character")
    try:
        parts = urlsplit(text)
        port = parts.port  # a malformed port raises here
    except ValueError:
        raise ValueError(f"{refusal}: it does not parse") from None
    if parts.scheme.lower() not in ("http", "https"):
        raise ValueError(f"{refusal}: its scheme is not http or https")
    if "@" in parts.netloc or parts.username is not None or parts.password is not None:
        raise ValueError(f"{refusal}: it carries a user or password")
    if not parts.hostname:
        raise ValueError(f"{refusal}: it has no host")
    if port == 0:
        raise ValueError(f"{refusal}: its port is 0")
    if parts.query or "?" in text:
        raise ValueError(f"{refusal}: it has a query")
    if parts.fragment or "#" in text:
        raise ValueError(f"{refusal}: it has a fragment")
    return text.rstrip("/")


def parse_sol_token(v: Any) -> SecretStr:
    """The token as a ``SecretStr``: printable ASCII without spaces (it goes in a header); empty when unset.

    Errors never quote the value.
    """
    value = sol_token_setting(v).get_secret_value()
    if value and not all("!" <= c <= "~" for c in value):
        raise ValueError(
            f"{SOL_TOKEN} must be printable ASCII without spaces or control characters",
        )
    return SecretStr(value)


def sol_route(base_url: Any, token: Any) -> tuple[str, SecretStr] | None:
    """Sol's route, ``(base_url, token)``, or ``None`` when neither is set (the shipped route).

    Exactly one set is refused (fail closed: no pass starts, so no call is made), as is either value in a
    wrong form. No error quotes a value.
    """
    base = parse_sol_base_url(base_url)
    secret = parse_sol_token(token)
    if not base and not secret.get_secret_value():
        return None
    if not base or not secret.get_secret_value():
        missing = SOL_BASE_URL if not base else SOL_TOKEN
        raise ValueError(
            f"{SOL_BASE_URL} and {SOL_TOKEN} are set together or not at all; "
            f"{missing} is empty, so no consolidation pass starts",
        )
    return base, secret


#: The validator for each setting (unify/settings.py ``parse_memory_v2``).
PARSERS = {
    SWITCH: parse_switch,
    EXPERIENCE_BUDGET: parse_experience_budget,
    SOL_MODEL: parse_sol_model,
    SOL_ALLOWANCE: parse_sol_allowance,
    SOL_RUN_GUARD: parse_sol_run_guard,
    # normalised only; checked by sol_route when a pass starts
    SOL_BASE_URL: sol_base_url_setting,
    SOL_TOKEN: sol_token_setting,
}

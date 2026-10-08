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
  host and no user, password, query or fragment; plain http only to exactly ``127.0.0.1`` with an explicit
  port (the launcher's loopback bridge; the token travels in a header). The token is a secret: at least 16
  URL-safe characters (RFC 3986 unreserved: ``A-Z a-z 0-9 . _ ~ -``, as ``secrets.token_urlsafe`` makes),
  held as a ``SecretStr`` and registered with every value-based redactor
  (:func:`unify.process_secrets.register_secret`). Settings only normalise the two (a settings error would
  print its input), and :func:`sol_route` checks them when a pass is about to start; its errors never quote
  either value. Both are read from the controller's process environment only: a value that reaches
  ``os.environ`` after settings were built (from ``.env``, say) refuses every pass
  (:func:`settle_sol_route_env`). Every refusal of the route is a :class:`SolRouteRefused`.
- ``UNIFY_MEMORY_V2_SOL_TOKEN_FD``: instead of ``UNIFY_MEMORY_V2_SOL_TOKEN`` (exactly one of the two), the number
  (above 2) of a descriptor the launcher passes the controller, holding the token (one trailing newline
  allowed). Settings only normalise it. When settings are first settled it is parsed and the descriptor read
  once (at most :data:`TOKEN_FD_MAX_BYTES` bytes and :data:`TOKEN_FD_TIMEOUT_S` seconds), then closed, so no
  later child inherits it; the token never enters ``os.environ`` (which carries only the number) and is
  registered with the redactors. A bad number, or a descriptor not open, not inherited, not a pipe or file,
  empty, oversize, slow or not holding a bearer token: every pass is refused, naming the rule, never a value.

Money stays a decimal string as written (never a float), and exponent forms are refused, so a value is
read the same way by every consumer.
"""

from __future__ import annotations

import os
import re
import select
import stat
import time
from decimal import Decimal
from typing import Any, MutableMapping
from urllib.parse import urlsplit

from pydantic import SecretStr

from unify.process_secrets import register_secret

SWITCH = "UNIFY_MEMORY_V2"
EXPERIENCE_BUDGET = "UNIFY_MEMORY_V2_E"
SOL_MODEL = "UNIFY_MEMORY_V2_SOL_MODEL"
SOL_ALLOWANCE = "UNIFY_MEMORY_V2_SOL_ALLOWANCE_USD_PER_TOKENS"
SOL_RUN_GUARD = "UNIFY_MEMORY_V2_SOL_RUN_GUARD_USD"
SOL_BASE_URL = "UNIFY_MEMORY_V2_SOL_BASE_URL"
SOL_TOKEN = "UNIFY_MEMORY_V2_SOL_TOKEN"
SOL_TOKEN_FD = "UNIFY_MEMORY_V2_SOL_TOKEN_FD"
SOL_EFFORT = "UNIFY_MEMORY_V2_SOL_EFFORT"

EXPERIENCE_BUDGET_DEFAULT = 150000
SOL_MODEL_DEFAULT = "openai/gpt-6-sol"
SOL_ALLOWANCE_DEFAULT = "0.00000073"
#: Sol's reasoning effort. ``actor`` (the default) matches the actor's effort for the run (the lead, 8 Oct ~15:4xZ:
#: "match the agent doing the task's effort with the agent that writes the stored memory"); ``low``, ``medium`` or
#: ``high`` fixes it, for a declared mismatch ablation only (e.g. a HIGH actor with a LOW storer).
SOL_EFFORT_DEFAULT = "actor"
SOL_EFFORTS = ("actor", "low", "medium", "high")

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


def sol_token_fd_setting(v: Any) -> str:
    """``UNIFY_MEMORY_V2_SOL_TOKEN_FD`` as settings hold it: stripped, never refused (a settings error would
    print its input, a token set there by mistake included); :func:`settle_sol_route_env` parses it with
    :func:`parse_sol_token_fd`.
    """
    return _stripped(v)


def parse_sol_token_fd(v: Any) -> int | None:
    """An inherited descriptor's number above 2 (not stdin, stdout or stderr), or ``None`` when empty.

    Its error never quotes the value (it would print a token set here by mistake).
    """
    refusal = f"{SOL_TOKEN_FD} must be empty or a file descriptor number above 2"
    if isinstance(v, bool):
        raise ValueError(refusal)
    if isinstance(v, int):
        value = v
    else:
        text = _stripped(v)
        if text == "":
            return None
        if not _DIGITS.fullmatch(text):
            raise ValueError(refusal)
        value = int(text)
    if value <= 2:
        raise ValueError(refusal)
    return value


class SolRouteRefused(ValueError):
    """Sol's route is set but not in effect (malformed, half set, or not what the settings hold): no
    consolidation pass starts. Its text names settings and rules, never a value.
    """


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
    if parts.scheme.lower() == "http" and parts.hostname != HTTP_HOST:
        raise ValueError(f"{refusal}: plain http is allowed only to {HTTP_HOST}")
    if parts.scheme.lower() == "http" and port is None:
        raise ValueError(f"{refusal}: plain http to {HTTP_HOST} needs an explicit port")
    return text.rstrip("/")


#: The one host plain http may name: the launcher's loopback bridge to Sol's proxy listener (the token travels
#: in a header, in clear). Exactly this IPv4 literal, compared as text: no other name, address or DNS lookup;
#: always with an explicit port, as the launcher sends it (``http://127.0.0.1:<port>/sol/v1``).
HTTP_HOST = "127.0.0.1"


def parse_sol_token(v: Any) -> SecretStr:
    """The token as a ``SecretStr``: URL-safe characters only (it goes in a header); empty when unset.

    Errors never quote the value.
    """
    value = sol_token_setting(v).get_secret_value()
    if value and not _BEARER_TOKEN.fullmatch(value):
        raise ValueError(
            f"{SOL_TOKEN} must be a bearer token: at least {_MIN_TOKEN_CHARS} characters from "
            "A-Z a-z 0-9 . _ ~ -",
        )
    return SecretStr(value)


#: At least 16 RFC 3986 unreserved characters (``secrets.token_urlsafe`` output fits): it goes in a header,
#: and no percent-encoding, JSON or other escaping form differs from it, so redaction by value always finds it
#: (``+``, ``/`` and ``=`` would turn into ``%2B``, ``\/`` and the like).
_MIN_TOKEN_CHARS = 16
_BEARER_TOKEN = re.compile(r"[A-Za-z0-9._~-]{%d,}" % _MIN_TOKEN_CHARS)


def sol_route(base_url: Any, token: Any) -> tuple[str, SecretStr] | None:
    """Sol's route, ``(base_url, token)``, or ``None`` when neither is set (the shipped route).

    Exactly one set is refused (fail closed: no pass starts, so no call is made), as is either value in a
    wrong form. No error quotes a value.
    """
    if _ENV_REFUSAL is not None:
        raise SolRouteRefused(_ENV_REFUSAL)
    raw = token.get_secret_value() if isinstance(token, SecretStr) else token
    register_secret(SOL_TOKEN, raw)  # kept redacted by value even if refused below
    try:
        base = parse_sol_base_url(base_url)
        secret = parse_sol_token(token)
    except ValueError as exc:  # the parsers' text names the rule, never the value
        raise SolRouteRefused(str(exc)) from None
    if not base and not secret.get_secret_value():
        return None
    if not base or not secret.get_secret_value():
        missing = SOL_BASE_URL if not base else SOL_TOKEN
        raise SolRouteRefused(
            f"{SOL_BASE_URL} and {SOL_TOKEN} are set together or not at all; "
            f"{missing} is empty, so no consolidation pass starts",
        )
    return base, secret


#: Why every pass is refused, once :func:`settle_sol_route_env` found the environment and the settings
#: disagreeing about Sol's route; sticky for the life of the process.
_ENV_REFUSAL: str | None = None


#: The most bytes Sol's token may take on its descriptor, and how long reading it may take.
TOKEN_FD_MAX_BYTES = 4096
TOKEN_FD_TIMEOUT_S = 5.0

#: Sol's token as read from ``UNIFY_MEMORY_V2_SOL_TOKEN_FD`` (once per process; never in ``os.environ``).
_FD_TOKEN: SecretStr | None = None
_FD_TRIED = False


def _read_token_fd(fd: int) -> str:
    """Sol's token from inherited descriptor *fd*, which is then closed; refusals name the rule, never a value."""
    where = f"{SOL_TOKEN_FD} names descriptor {fd}, which"
    try:
        mode = os.fstat(fd).st_mode
        inherited = os.get_inheritable(fd)
    except OSError:
        raise SolRouteRefused(f"{where} is not open") from None
    # opened by this process (PEP 446), not passed by the launcher: left alone
    if not inherited:
        raise SolRouteRefused(f"{where} was not inherited by this process")
    data = b""
    try:
        if not (stat.S_ISFIFO(mode) or stat.S_ISREG(mode)):
            raise SolRouteRefused(f"{where} is not a pipe or a file")
        poller = select.poll()
        poller.register(fd, select.POLLIN)
        deadline = time.monotonic() + TOKEN_FD_TIMEOUT_S
        while True:
            left = deadline - time.monotonic()
            if left <= 0 or not poller.poll(max(1, int(left * 1000))):
                raise SolRouteRefused(
                    f"{where} did not reach its end within {TOKEN_FD_TIMEOUT_S:g} s",
                )
            chunk = os.read(fd, TOKEN_FD_MAX_BYTES + 1 - len(data))
            if not chunk:
                break
            data += chunk
            if len(data) > TOKEN_FD_MAX_BYTES:
                raise SolRouteRefused(
                    f"{where} holds more than {TOKEN_FD_MAX_BYTES} bytes",
                )
    except OSError:
        raise SolRouteRefused(f"{where} could not be read") from None
    finally:
        register_secret(SOL_TOKEN, data.decode("latin-1"))
        try:
            os.close(fd)  # whatever happened: no later child inherits it
        except OSError:
            pass
    text = data.decode("latin-1")
    text = text[:-1] if text.endswith("\n") else text
    register_secret(SOL_TOKEN, text)
    if not text:
        raise SolRouteRefused(f"{where} is empty")
    if not _BEARER_TOKEN.fullmatch(text):
        raise SolRouteRefused(
            f"{where} does not hold a bearer token: at least {_MIN_TOKEN_CHARS} characters from "
            "A-Z a-z 0-9 . _ ~ -",
        )
    return text


def sol_token(settings: Any) -> Any:
    """The token Sol's route uses: the one read from ``UNIFY_MEMORY_V2_SOL_TOKEN_FD`` when that is set (empty
    if it could not be read; :func:`sol_route` then refuses), else ``UNIFY_MEMORY_V2_SOL_TOKEN``.
    """
    if _stripped(getattr(settings, SOL_TOKEN_FD, "")):
        return _FD_TOKEN if _FD_TOKEN is not None else SecretStr("")
    return getattr(settings, SOL_TOKEN, "")


def settle_sol_route_env(
    environ: MutableMapping[str, str],
    settings: Any,
) -> str | None:
    """Keep Sol's token out of *environ* and refuse every pass if *environ* and *settings* disagree.

    Called once the settings are built (``unify/settings.py``), again after the CLI loads ``.env`` and whenever
    a pass is about to start. The first call with ``UNIFY_MEMORY_V2_SOL_TOKEN_FD`` set reads the token from that
    descriptor and closes it (:func:`_read_token_fd`); a failed read, or both that and
    ``UNIFY_MEMORY_V2_SOL_TOKEN`` set, refuses every pass. Every
    case variant of ``UNIFY_MEMORY_V2_SOL_TOKEN`` is removed from *environ* and its value registered for
    redaction (pydantic-settings matches names in any case). A token or base URL found in *environ* that the
    settings do not hold (one that arrived after they were built, from ``.env`` say, or two case variants
    with different values) would otherwise silently leave Sol on the actor's route, so from then on
    :func:`sol_route` refuses every pass with the returned message. The message names settings, never
    values. Returns ``None`` when they agree.
    """
    global _ENV_REFUSAL, _FD_TOKEN, _FD_TRIED
    raw = getattr(settings, SOL_TOKEN, "")
    held_token = _stripped(
        raw.get_secret_value() if isinstance(raw, SecretStr) else raw,
    )
    register_secret(SOL_TOKEN, held_token)
    held_base = _stripped(getattr(settings, SOL_BASE_URL, ""))
    held_fd = _stripped(getattr(settings, SOL_TOKEN_FD, ""))
    fd_refusal = None
    if held_fd and not _FD_TRIED:
        _FD_TRIED = True
        try:  # a SolRouteRefused is a ValueError; neither quotes a value
            fd = parse_sol_token_fd(held_fd)
            _FD_TOKEN = SecretStr(_read_token_fd(int(fd or 0)))
        except ValueError as exc:
            register_secret(SOL_TOKEN_FD, held_fd)  # a token set there by mistake
            fd_refusal = f"{exc}; no consolidation pass starts"
    if held_fd and held_token:
        fd_refusal = fd_refusal or (
            f"{SOL_TOKEN} and {SOL_TOKEN_FD} are both set (set exactly one); no consolidation pass starts"
        )
    stray: list[str] = []
    for name in [k for k in environ if k.upper() == SOL_TOKEN]:
        value = environ.pop(name, "")
        register_secret(SOL_TOKEN, value)
        if _stripped(value) != held_token:
            stray.append(SOL_TOKEN)
    for name in [k for k in environ if k.upper() == SOL_BASE_URL]:
        if _stripped(environ.get(name)) != held_base:
            stray.append(SOL_BASE_URL)
    for name in [k for k in environ if k.upper() == SOL_TOKEN_FD]:
        if _stripped(environ.get(name)) != held_fd:
            stray.append(SOL_TOKEN_FD)
    if fd_refusal and _ENV_REFUSAL is None:
        _ENV_REFUSAL = fd_refusal
    if stray and _ENV_REFUSAL is None:
        _ENV_REFUSAL = (
            f"{' and '.join(sorted(set(stray)))} reached the environment after settings were read "
            "(from .env, or in another letter case), so Sol's route is not what the settings hold; no "
            "consolidation pass starts. Set both in the controller's process environment, not .env"
        )
    return _ENV_REFUSAL


def parse_sol_effort(v: Any) -> str:
    """Sol's reasoning effort: ``actor`` (the default: the actor's effort for the run), or a fixed ``low``,
    ``medium`` or ``high`` for a declared mismatch ablation."""
    value = _stripped(v).lower() or SOL_EFFORT_DEFAULT
    if value not in SOL_EFFORTS:
        raise ValueError(
            f"{SOL_EFFORT} must be one of {', '.join(SOL_EFFORTS)}, not {v!r}",
        )
    return value


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
    SOL_TOKEN_FD: sol_token_fd_setting,
    SOL_EFFORT: parse_sol_effort,
}

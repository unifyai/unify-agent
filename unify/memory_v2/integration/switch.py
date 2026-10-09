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
- ``UNIFY_MEMORY_V2_SOL_EFFORT_SCALE`` and ``UNIFY_MEMORY_V2_SOL_MAX_CALLS``: per-pass limits by the pass's Sol
  effort, maps ``low:1,medium:2,high:5`` (positive plain decimals multiplying the allowance cap) and
  ``low:40,medium:80,high:80`` (positive whole numbers of calls); empty means those defaults.
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
  later child inherits it; the token never enters ``os.environ``, the number leaves it (every letter case;
  held internally for the stray check) so no later child sees the name, and the token is registered with
  the redactors. A bad number, or a descriptor not open, not inherited, not a pipe or file,
  empty, oversize, slow or not holding a bearer token: every pass is refused, naming the rule, never a value.
- ``UNIFY_MEMORY_V2_SOL_USAGE``: ``on``, ``off`` or empty (empty and ``off`` mean off). When on, each pass's
  first message ends with the table of how requests used each library function (``usage.usage_table``);
  off, that message is as before. The use record itself is kept either way.
- ``UNIFY_MEMORY_V2_DIALOGUE``: ``off`` (or empty; the default) or ``env``. With ``env``, each request's
  episode also records the dialogue actions of its transcript (each turn-ending reply's action, paired with
  the counterpart's next message; :mod:`.adapters.dialogue`) on the channel ``env``, which is the memory
  channel ``env`` (``env/env/``). Off, the episode is recorded exactly as without the setting.

The v2.1 surfacing switches (lane S1). Each default restores the behaviour of the v2 screen build
(``9deefbfd1``) exactly, so a paired v2 vs v2.1 comparison runs on one build:

- ``UNIFY_MEMORY_V2_SURFACING``: ``index`` (default; empty means it): the system prompt ends with the v2
  per-function index and the export line, the export holds only the commit's files, no input shapes are
  recorded or frozen, and Sol's first message carries the index. ``catalogue``: a constant guide paragraph
  in the prompt (the same bytes for the whole run, from the first request with a non-empty library; the
  channels, counts and suspect flags are what ``memory.catalog()`` prints in the cell); the generated
  ``README.md``, ``.memory/catalog.json``, ``.memory/shapes.py`` and ``memory.py`` in every export; input
  shapes recorded at each merge and frozen per commit; Sol's first message carries the README and its brief
  says never to write those files.
- ``UNIFY_MEMORY_V2_DOCSTRINGS``: ``off`` (default; empty means it) or ``on``: the gate's lean docstring
  standard (G1) and examples run (G3), with the standard and the check list in Sol's brief. The online
  driver and the offline replay (``memory_v2_offline/cadence_replay.py``) both build the gate from this
  switch (:func:`surfacing_options`), so a replay enforces exactly what Sol is told.
- ``UNIFY_MEMORY_V2_SOFT_BUDGET``: ``off`` (default; empty means it): G4 refuses an index over 4,000
  estimated tokens, as in v2. ``on``: G4 only notes that hygiene is due past the budget, measured on the
  library's surface (the README and channel lines under ``catalogue``, the index under ``index``).

Stage-5 test checks in the consolidation gate (memory v2.1; :mod:`unify.memory_v2.qa`), each off by default
and read only while ``UNIFY_MEMORY_V2`` is on:

- ``UNIFY_MEMORY_V2_QA_FIXTURES``: ``on``, ``strict``, ``off`` or empty. Seeded random draws of recorded inputs
  of each new or changed function's family; ``strict`` also refuses a drawn input its own tests never read.
- ``UNIFY_MEMORY_V2_QA_MUTATION``: ``on``, ``off`` or empty. Mutation testing of each new or changed function.
- ``UNIFY_MEMORY_V2_QA_MUTATION_MIN_KILL``: the minimum share of output-changing mutants the tests must kill, a
  plain decimal string from 0 to 1; empty means ``0.5``.
- ``UNIFY_MEMORY_V2_QA_DETERMINISM``: ``on``, ``off`` or empty. Pinned clock and randomness; each new or changed
  test file runs twice.
- ``UNIFY_MEMORY_V2_QA_REPLAY``: ``on``, ``off`` or empty. Environment functions are tested through the
  recorded replay, never a stand-in.
- ``UNIFY_MEMORY_V2_QA_FIXTURE_SIZE``: ``on``, ``off`` or empty. A bound on test files, blob references and
  truncation markers.

Not switched (a declared safety fix in every mode): the gate refuses bytecode, native code, start-up hooks
and root entries other than ``env/``, ``workflows/`` and the test kit before extraction
(:func:`unify.memory_v2.manifest.unsafe_path`), and changes to the paths the harness reserves for its
generated catalogue (:func:`unify.memory_v2.catalogue.reserved`).

Money stays a decimal string as written (never a float), and exponent forms are refused, so a value is
read the same way by every consumer.
"""

from __future__ import annotations

import os
import re
import select
import stat
import time
from dataclasses import dataclass
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
SOL_EFFORT_SCALE = "UNIFY_MEMORY_V2_SOL_EFFORT_SCALE"
SOL_MAX_CALLS = "UNIFY_MEMORY_V2_SOL_MAX_CALLS"
SURFACING = "UNIFY_MEMORY_V2_SURFACING"
DOCSTRINGS = "UNIFY_MEMORY_V2_DOCSTRINGS"
SOFT_BUDGET = "UNIFY_MEMORY_V2_SOFT_BUDGET"
SOL_USAGE = "UNIFY_MEMORY_V2_SOL_USAGE"
# memory v2.1 (spec v2.1; P1 onwards): the writer's batch map, views and coverage; off by default
V21 = "UNIFY_MEMORY_V21"
# memory v2.1 P9: the bed's runner posts the verdicts the actor sees as structured checker lines; off by default
CHECKER_VISIBLE = "UNIFY_MEMORY_V21_CHECKER_VISIBLE"
DIALOGUE = "UNIFY_MEMORY_V2_DIALOGUE"

QA_FIXTURES = "UNIFY_MEMORY_V2_QA_FIXTURES"
QA_MUTATION = "UNIFY_MEMORY_V2_QA_MUTATION"
QA_MIN_KILL = "UNIFY_MEMORY_V2_QA_MUTATION_MIN_KILL"
QA_DETERMINISM = "UNIFY_MEMORY_V2_QA_DETERMINISM"
QA_REPLAY = "UNIFY_MEMORY_V2_QA_REPLAY"
QA_FIXTURE_SIZE = "UNIFY_MEMORY_V2_QA_FIXTURE_SIZE"

EXPERIENCE_BUDGET_DEFAULT = 150000
SOL_MODEL_DEFAULT = "openai/gpt-6-sol"
SOL_ALLOWANCE_DEFAULT = "0.00000073"
#: Sol's reasoning effort. ``actor`` (the default) matches the actor's effort for the run (the lead, 8 Oct ~15:4xZ:
#: "match the agent doing the task's effort with the agent that writes the stored memory"); ``low``, ``medium`` or
#: ``high`` fixes it, for a declared mismatch ablation only (e.g. a HIGH actor with a LOW storer).
SOL_EFFORT_DEFAULT = "actor"
SOL_EFFORTS = ("actor", "low", "medium", "high")
#: Per-pass limits by the pass's Sol effort (one rule for every bed; MAIN, 8 Oct): the allowance cap (E x a_tok)
#: is multiplied by the effort's scale, and the pass makes at most the effort's calls. The offline diagnostic
#: measured USD per pass 0.095 (ARC MEDIUM), 0.178 (ARC HIGH), 0.116 (office MEDIUM), 0.30 (office HIGH) and
#: 10/14/13/23 calls per pass; at scale 1 and 40 calls a HIGH pass merged nothing.
SOL_EFFORT_SCALE_DEFAULT = "low:1,medium:2,high:5"
SOL_MAX_CALLS_DEFAULT = "low:40,medium:80,high:80"
SURFACING_VALUES = ("index", "catalogue")
SURFACING_DEFAULT = "index"
ON_OFF = ("off", "on")
QA_MIN_KILL_DEFAULT = "0.5"

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
#: The descriptor number the first settle read (settings' value), kept once it has left ``os.environ``.
_FD_HELD = ""


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
    ``UNIFY_MEMORY_V2_SOL_TOKEN`` set, refuses every pass. Every case variant of
    ``UNIFY_MEMORY_V2_SOL_TOKEN_FD`` then leaves *environ* (the number is held here for the stray check, and
    no later call reads a descriptor again), so no child spawned afterwards sees the name. Every
    case variant of ``UNIFY_MEMORY_V2_SOL_TOKEN`` is removed from *environ* and its value registered for
    redaction (pydantic-settings matches names in any case). A token or base URL found in *environ* that the
    settings do not hold (one that arrived after they were built, from ``.env`` say, or two case variants
    with different values) would otherwise silently leave Sol on the actor's route, so from then on
    :func:`sol_route` refuses every pass with the returned message. The message names settings, never
    values. Returns ``None`` when they agree.
    """
    global _ENV_REFUSAL, _FD_TOKEN, _FD_TRIED, _FD_HELD
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
        _FD_HELD = held_fd
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
        # removed once read, so no later child sees the name; compared with the number held
        if _stripped(environ.pop(name, "")) != (held_fd or _FD_HELD):
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


def _effort_map(name: str, v: Any, default: str, value_ok: Any) -> dict[str, str]:
    """``low:<v>,medium:<v>,high:<v>``: each fixed effort exactly once, in any order, each value checked."""
    text = _stripped(v).lower() or default
    refusal = f"{name} must map each of low, medium and high exactly once, as in {default!r}, not {v!r}"[
        :300
    ]
    got: dict[str, str] = {}
    for part in text.split(","):
        key, sep, value = part.partition(":")
        key, value = key.strip(), value.strip()
        if not sep or key not in SOL_EFFORTS[1:] or key in got or not value_ok(value):
            raise ValueError(refusal)
        got[key] = value
    if len(got) != len(SOL_EFFORTS) - 1:
        raise ValueError(refusal)
    return {k: got[k] for k in SOL_EFFORTS[1:]}


def _positive_decimal(text: str) -> bool:
    return bool(_PLAIN_DECIMAL.fullmatch(text)) and Decimal(text) > 0


def _positive_int(text: str) -> bool:
    return bool(_DIGITS.fullmatch(text)) and int(text) > 0


def sol_effort_scale_map(v: Any) -> dict[str, Decimal]:
    """The allowance-cap multiplier per Sol effort, from ``UNIFY_MEMORY_V2_SOL_EFFORT_SCALE``'s value."""
    m = _effort_map(SOL_EFFORT_SCALE, v, SOL_EFFORT_SCALE_DEFAULT, _positive_decimal)
    return {k: Decimal(x) for k, x in m.items()}


def sol_max_calls_map(v: Any) -> dict[str, int]:
    """The per-pass call limit per Sol effort, from ``UNIFY_MEMORY_V2_SOL_MAX_CALLS``'s value."""
    m = _effort_map(SOL_MAX_CALLS, v, SOL_MAX_CALLS_DEFAULT, _positive_int)
    return {k: int(x) for k, x in m.items()}


def parse_sol_effort_scale(v: Any) -> str:
    """Positive plain decimals per effort; normalised to ``low:..,medium:..,high:..``."""
    m = _effort_map(SOL_EFFORT_SCALE, v, SOL_EFFORT_SCALE_DEFAULT, _positive_decimal)
    return ",".join(f"{k}:{x}" for k, x in m.items())


def parse_sol_max_calls(v: Any) -> str:
    """Positive whole numbers per effort; normalised to ``low:..,medium:..,high:..``."""
    m = _effort_map(SOL_MAX_CALLS, v, SOL_MAX_CALLS_DEFAULT, _positive_int)
    return ",".join(f"{k}:{int(x)}" for k, x in m.items())


def parse_surfacing(v: Any) -> str:
    """``index`` (also for empty) or ``catalogue``."""
    value = _stripped(v).lower() or SURFACING_DEFAULT
    if value not in SURFACING_VALUES:
        raise ValueError(
            f"{SURFACING} must be empty, 'index' or 'catalogue', not {v!r}"[:200],
        )
    return value


def _on_off(name: str, v: Any) -> str:
    value = _stripped(v).lower() or "off"
    if value not in ON_OFF:
        raise ValueError(f"{name} must be empty, 'off' or 'on', not {v!r}"[:200])
    return value


def parse_docstrings(v: Any) -> str:
    """``off`` (also for empty) or ``on``."""
    return _on_off(DOCSTRINGS, v)


def parse_soft_budget(v: Any) -> str:
    """``off`` (also for empty) or ``on``."""
    return _on_off(SOFT_BUDGET, v)


def parse_memory_v21(v: Any) -> str:
    """``UNIFY_MEMORY_V21``: ``off`` (also for empty) or ``on``."""
    return _on_off(V21, v)


parse_v21 = parse_memory_v21


def v21_enabled(settings: Any) -> bool:
    """Whether memory v2.1 is on in *settings* (``unify.settings.SETTINGS`` or any object; missing is off)."""
    return parse_v21(getattr(settings, V21, "") or "") == "on"


def parse_checker_visible(v: Any) -> str:
    """``UNIFY_MEMORY_V21_CHECKER_VISIBLE``: ``off`` (also for empty) or ``on``. On (with ``UNIFY_MEMORY_V21``),
    the CLI takes ``{"checker": {"label": "pass"|"fail"}}`` lines: the verdicts the bed shows the actor, recorded
    as agent-visible checker signals (spec v2.1 §5, P9)."""
    return _on_off(CHECKER_VISIBLE, v)


#: v2 settings whose value contradicts the v2.1 prompts or gate (spec §12.4): the v2 docstring check (fixtures
#: under env/<channel>/tests/), the catalogue guide, memlab replay in tests, and the fixture-size cap (P5 forbids
#: caps on data).
V21_CONFLICTS: tuple[tuple[str, str], ...] = (
    ("UNIFY_MEMORY_V2_DOCSTRINGS", "on"),
    ("UNIFY_MEMORY_V2_SURFACING", "catalogue"),
    ("UNIFY_MEMORY_V2_QA_REPLAY", "on"),
    ("UNIFY_MEMORY_V2_QA_FIXTURE_SIZE", "on"),
)


def v21_conflicts(settings: Any) -> list[str]:
    """``NAME=value`` for each setting in :data:`V21_CONFLICTS` at its contradicting value, while v2.1 is on."""
    if not v21_enabled(settings):
        return []
    out = []
    for name, bad in V21_CONFLICTS:
        if str(getattr(settings, name, "") or "").strip().lower() == bad:
            out.append(f"{name}={bad}")
    return out


V21_E = "UNIFY_MEMORY_V21_E"
#: The v2.1 writer's USD per recorded token (spec §15, F9): v2's rate until the offline replay sizes it for coverage.
V21_SOL_USD_PER_TOKEN = "UNIFY_MEMORY_V21_SOL_USD_PER_TOKEN"
V21_PASS_WALL_S = "UNIFY_MEMORY_V21_PASS_WALL_S"
SOL_JOURNAL = "UNIFY_MEMORY_V2_SOL_JOURNAL"
#: The pass's wall-clock bound in seconds (spec §6, §9.2, §15): calibrated with the budget by the offline replay.
V21_PASS_WALL_S_DEFAULT = 2700


def _positive_whole(name: str, default: int, v: Any) -> int:
    refusal = f"{name} must be a positive whole number, not {v!r}"
    if isinstance(v, bool):
        raise ValueError(refusal)
    if isinstance(v, int):
        value = v
    else:
        text = _stripped(v)
        if text == "":
            return default
        if not _DIGITS.fullmatch(text):
            raise ValueError(refusal)
        value = int(text)
    if value <= 0:
        raise ValueError(refusal)
    return value


def parse_v21_e(v: Any) -> int:
    """``UNIFY_MEMORY_V21_E``: E under v2.1, positive digits (D43; empty: 100,000)."""
    from ..trigger import EXPERIENCE_BUDGET_V21

    return _positive_whole(V21_E, EXPERIENCE_BUDGET_V21, v)


def parse_v21_pass_wall_s(v: Any) -> int:
    """``UNIFY_MEMORY_V21_PASS_WALL_S``: one pass's wall-clock bound in seconds (empty: 2700)."""
    return _positive_whole(V21_PASS_WALL_S, V21_PASS_WALL_S_DEFAULT, v)


def parse_sol_journal(v: Any) -> str:
    """Empty, or the absolute path of the Sol route proxy's journal (JSON lines with ``request_attempt_id``,
    ``generation_id`` and ``account_charge``), read only to reconcile cancelled calls (spec §6).
    """
    text = _stripped(v)
    if text and not text.startswith("/"):
        raise ValueError(f"{SOL_JOURNAL} must be empty or an absolute path")
    return text


def parse_v21_sol_usd_per_token(v: Any) -> str:
    """``UNIFY_MEMORY_V21_SOL_USD_PER_TOKEN``: a positive plain decimal string (empty: v2's rate, 0.00000073)."""
    return _plain_decimal(
        V21_SOL_USD_PER_TOKEN,
        _stripped(v) or SOL_ALLOWANCE_DEFAULT,
        v,
        positive=True,
    )


def v21_sol_usd_per_token(settings: Any) -> Decimal:
    """The USD per recorded token of a v2.1 pass's cap (MAIN, 9 Oct: the replay can raise it without code)."""
    return Decimal(
        parse_v21_sol_usd_per_token(getattr(settings, V21_SOL_USD_PER_TOKEN, "")),
    )


def v21_experience_budget(settings: Any) -> int:
    return parse_v21_e(getattr(settings, V21_E, ""))


def v21_pass_wall_s(settings: Any) -> int:
    return parse_v21_pass_wall_s(getattr(settings, V21_PASS_WALL_S, ""))


def sol_journal(settings: Any) -> str | None:
    return parse_sol_journal(getattr(settings, SOL_JOURNAL, "")) or None


def checker_visible(settings: Any) -> bool:
    """The bed's declaration ``checker_visible_to_actor`` (spec v2.1 §5): whether the checker's verdict is one
    the actor itself sees. Only then does the item lifecycle count checker signals. Missing is off. The one
    switch P9 reads too."""
    return parse_checker_visible(getattr(settings, CHECKER_VISIBLE, "") or "") == "on"


@dataclass(frozen=True)
class SurfacingOptions:
    """The three v2.1 surfacing switches, parsed; the defaults are the v2 screen build's behaviour."""

    surfacing: str = SURFACING_DEFAULT
    docstrings: bool = False
    soft_budget: bool = False

    @property
    def catalogue(self) -> bool:
        return self.surfacing == "catalogue"

    def gate_kwargs(self) -> dict[str, Any]:
        """The :class:`unify.memory_v2.gate.Gate` keyword arguments these switches set (Sol's brief follows
        the gate it is given, so this is the one place both are configured)."""
        return {
            "surfacing": self.surfacing,
            "docstring_standard": self.docstrings,
            "soft_budget": self.soft_budget,
        }

    def as_dict(self) -> dict[str, str]:
        """The settings as written (for a run's provenance)."""
        return {
            SURFACING: self.surfacing,
            DOCSTRINGS: "on" if self.docstrings else "off",
            SOFT_BUDGET: "on" if self.soft_budget else "off",
        }


def surfacing_options(settings: Any) -> SurfacingOptions:
    """The switches from *settings* (``unify.settings.SETTINGS``, or any object; a missing one is its default),
    each through its validator, so an invalid value is refused here as at settings load.
    """
    return SurfacingOptions(
        surfacing=parse_surfacing(getattr(settings, SURFACING, "")),
        docstrings=parse_docstrings(getattr(settings, DOCSTRINGS, "")) == "on",
        soft_budget=parse_soft_budget(getattr(settings, SOFT_BUDGET, "")) == "on",
    )


def parse_sol_usage(v: Any) -> str:
    """``on``, or ``""`` for off (empty or ``off``)."""
    value = _stripped(v).lower()
    if value in ("", "off"):
        return ""
    if value != "on":
        raise ValueError(f"{SOL_USAGE} must be empty, 'off' or 'on', not {v!r}")
    return value


def _on_or_empty(name: str):
    """A parser for a stage-5 switch: ``on``, or ``""`` for off (empty or ``off``). (Named apart from the
    surfacing switches' :func:`_on_off`, which keeps ``off`` as written.)"""

    def parse(v: Any) -> str:
        """``on``, or ``""`` for off (empty or ``off``)."""
        value = _stripped(v).lower()
        if value in ("", "off"):
            return ""
        if value != "on":
            raise ValueError(f"{name} must be empty, 'off' or 'on', not {v!r}")
        return value

    return parse


parse_qa_mutation = _on_or_empty(QA_MUTATION)
parse_qa_determinism = _on_or_empty(QA_DETERMINISM)
parse_qa_replay = _on_or_empty(QA_REPLAY)
parse_qa_fixture_size = _on_or_empty(QA_FIXTURE_SIZE)


def parse_qa_fixtures(v: Any) -> str:
    """``on``, ``strict``, or ``""`` for off (empty or ``off``)."""
    value = _stripped(v).lower()
    if value in ("", "off"):
        return ""
    if value not in ("on", "strict"):
        raise ValueError(
            f"{QA_FIXTURES} must be empty, 'off', 'on' or 'strict', not {v!r}",
        )
    return value


def parse_qa_min_kill(v: Any) -> str:
    """A share from 0 to 1: a plain decimal string."""
    text = _stripped(v) or QA_MIN_KILL_DEFAULT
    if not _PLAIN_DECIMAL.fullmatch(text) or Decimal(text) > 1:
        raise ValueError(
            f"{QA_MIN_KILL} must be a plain decimal from 0 to 1 (digits and at most one point), not {v!r}",
        )
    return text


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
    SOL_EFFORT_SCALE: parse_sol_effort_scale,
    SOL_MAX_CALLS: parse_sol_max_calls,
    SURFACING: parse_surfacing,
    DOCSTRINGS: parse_docstrings,
    SOFT_BUDGET: parse_soft_budget,
    SOL_USAGE: parse_sol_usage,
    V21: parse_memory_v21,
    CHECKER_VISIBLE: parse_checker_visible,
    QA_FIXTURES: parse_qa_fixtures,
    QA_MUTATION: parse_qa_mutation,
    QA_MIN_KILL: parse_qa_min_kill,
    QA_DETERMINISM: parse_qa_determinism,
    QA_REPLAY: parse_qa_replay,
    QA_FIXTURE_SIZE: parse_qa_fixture_size,
    V21_E: parse_v21_e,
    V21_SOL_USD_PER_TOKEN: parse_v21_sol_usd_per_token,
    V21_PASS_WALL_S: parse_v21_pass_wall_s,
    SOL_JOURNAL: parse_sol_journal,
}


#: The counterparts ``UNIFY_MEMORY_V2_DIALOGUE`` may name: the channel key of the recorded dialogue
#: actions. ``env`` is the counterpart a benchmark runner serves (the offline imports' name).
DIALOGUE_COUNTERPARTS = ("env",)


def parse_dialogue(v: Any) -> str:
    """The dialogue counterpart, or ``""`` for off (empty or ``off``)."""
    value = _stripped(v).lower()
    if value in ("", "off"):
        return ""
    if value not in DIALOGUE_COUNTERPARTS:
        raise ValueError(f"{DIALOGUE} must be empty, 'off' or 'env', not {v!r}")
    return value


PARSERS[DIALOGUE] = parse_dialogue

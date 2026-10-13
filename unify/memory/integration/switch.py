"""Value parsing for the UNIFY_MEMORY settings (unify/settings.py calls these validators).

The contract (online build):

- ``UNIFY_MEMORY``: ``on``, ``off`` or empty (empty and ``off`` mean off).
- ``UNIFY_MEMORY_E``: the experience budget E in tokens, a positive int; empty means 150000. A pass
  becomes due once the evidence since the last pass reaches E, and its USD cap is E times the allowance.
- ``UNIFY_MEMORY_WRITER_MODEL``: the model that runs consolidation passes; empty means
  the setting's default model id. The writer's reasoning effort has no switch: it is the actor's effort for the run.
- ``UNIFY_MEMORY_WRITER_ALLOWANCE_USD_PER_TOKENS``: USD the writer may spend per token of experience, a positive
  plain decimal string; empty means ``0.00000073``.
- ``UNIFY_MEMORY_WRITER_RUN_GUARD_USD``: a non-negative plain decimal string, or empty for no guard. When
  set, no further pass starts once the run's committed writer USD plus the next pass's cap would exceed it.
- ``UNIFY_MEMORY_WRITER_EFFORT_SCALE`` and ``UNIFY_MEMORY_WRITER_MAX_CALLS``: per-pass limits by the pass's writer
  effort, maps ``low:1,medium:2,high:5`` (positive plain decimals multiplying the allowance cap) and
  ``low:40,medium:80,high:80`` (positive whole numbers of calls); empty means those defaults.
- ``UNIFY_MEMORY_WRITER_BASE_URL`` and ``UNIFY_MEMORY_WRITER_TOKEN``: the writer's route, both or neither. Set,
  the writer's model calls go to that OpenAI-compatible base URL with that token (a proxy listener of the writer's own,
  so the actor's route never carries the writer's model); empty, they go as shipped. The URL is http(s) with a
  host and no user, password, query or fragment; plain http only to exactly ``127.0.0.1`` with an explicit
  port (the launcher's loopback bridge; the token travels in a header). The token is a secret: at least 16
  URL-safe characters (RFC 3986 unreserved: ``A-Z a-z 0-9 . _ ~ -``, as ``secrets.token_urlsafe`` makes),
  held as a ``SecretStr`` and registered with every value-based redactor
  (:func:`unify.process_secrets.register_secret`). Settings only normalise the two (a settings error would
  print its input), and :func:`writer_route` checks them when a pass is about to start; its errors never quote
  either value. Both are read from the controller's process environment only: a value that reaches
  ``os.environ`` after settings were built (from ``.env``, say) refuses every pass
  (:func:`settle_writer_route_env`). Every refusal of the route is a :class:`WriterRouteRefused`.
- ``UNIFY_MEMORY_WRITER_TOKEN_FD``: instead of ``UNIFY_MEMORY_WRITER_TOKEN`` (exactly one of the two), the number
  (above 2) of a descriptor the launcher passes the controller, holding the token (one trailing newline
  allowed). Settings only normalise it. When settings are first settled it is parsed and the descriptor read
  once (at most :data:`TOKEN_FD_MAX_BYTES` bytes and :data:`TOKEN_FD_TIMEOUT_S` seconds), then closed, so no
  later child inherits it; the token never enters ``os.environ``, the number leaves it (every letter case;
  held internally for the stray check) so no later child sees the name, and the token is registered with
  the redactors. A bad number, or a descriptor not open, not inherited, not a pipe or file,
  empty, oversize, slow or not holding a bearer token: every pass is refused, naming the rule, never a value.
- ``UNIFY_MEMORY_WRITER_USAGE``: ``on``, ``off`` or empty (empty and ``off`` mean off). When on, each pass's
  first message ends with the table of how requests used each library function (``usage.usage_table``);
  off, that message is as before. The use record itself is kept either way.
- ``UNIFY_MEMORY_DIALOGUE``: ``off`` (or empty; the default) or ``env``. With ``env``, each request's
  episode also records the dialogue actions of its transcript (each turn-ending reply's action, paired with
  the counterpart's next message; :mod:`.adapters.dialogue`) on the channel ``env``, which is the memory
  channel ``env`` (``env/env/``). Off, the episode is recorded exactly as without the setting.

The library mode's surfacing switches. Each default restores the behaviour of the v2 screen build
(``9deefbfd1``) exactly, so a paired v2 vs library-mode comparison runs on one build:

- ``UNIFY_MEMORY_SURFACING``: ``index`` (default; empty means it): the system prompt ends with the v2
  per-function index and the export line, the export holds only the commit's files, no input shapes are
  recorded or frozen, and the writer's first message carries the index. ``catalogue``: a constant guide paragraph
  in the prompt (the same bytes for the whole run, from the first request with a non-empty library; the
  channels, counts and suspect flags are what ``memory.catalog()`` prints in the cell); the generated
  ``README.md``, ``.memory/catalog.json``, ``.memory/shapes.py`` and ``memory.py`` in every export; input
  shapes recorded at each merge and frozen per commit; the writer's first message carries the README and its brief
  says never to write those files.
- ``UNIFY_MEMORY_DOCSTRINGS``: ``off`` (default; empty means it) or ``on``: the gate's lean docstring
  standard (G1) and examples run (G3), with the standard and the check list in the writer's brief. The online
  driver and the offline replay (``memory_offline/cadence_replay.py``) both build the gate from this
  switch (:func:`surfacing_options`), so a replay enforces exactly what the writer is told.
- ``UNIFY_MEMORY_SOFT_BUDGET``: ``off`` (default; empty means it): G4 refuses an index over 4,000
  estimated tokens, as in v2. ``on``: G4 only notes that hygiene is due past the budget, measured on the
  library's surface (the README and channel lines under ``catalogue``, the index under ``index``).

Stage-5 test checks in the consolidation gate (the library mode; :mod:`unify.memory.qa`), each off by default
and read only while ``UNIFY_MEMORY`` is on:

- ``UNIFY_MEMORY_QA_FIXTURES``: ``on``, ``strict``, ``off`` or empty. Seeded random draws of recorded inputs
  of each new or changed function's family; ``strict`` also refuses a drawn input its own tests never read.
- ``UNIFY_MEMORY_QA_MUTATION``: ``on``, ``off`` or empty. Mutation testing of each new or changed function.
- ``UNIFY_MEMORY_QA_MUTATION_MIN_KILL``: the minimum share of output-changing mutants the tests must kill, a
  plain decimal string from 0 to 1; empty means ``0.5``.
- ``UNIFY_MEMORY_QA_DETERMINISM``: ``on``, ``off`` or empty. Pinned clock and randomness; each new or changed
  test file runs twice.
- ``UNIFY_MEMORY_QA_REPLAY``: ``on``, ``off`` or empty. Environment functions are tested through the
  recorded replay, never a stand-in.
- ``UNIFY_MEMORY_QA_FIXTURE_SIZE``: ``on``, ``off`` or empty. A bound on test files, blob references and
  truncation markers.

Not switched (a declared safety fix in every mode): the gate refuses bytecode, native code, start-up hooks
and root entries other than ``env/``, ``workflows/`` and the test kit before extraction
(:func:`unify.memory.manifest.unsafe_path`), and changes to the paths the harness reserves for its
generated catalogue (:func:`unify.memory.catalogue.reserved`).

Money stays a decimal string as written (never a float), and exponent forms are refused, so a value is
read the same way by every consumer.
"""

from __future__ import annotations

import json
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

SWITCH = "UNIFY_MEMORY"
EXPERIENCE_BUDGET = "UNIFY_MEMORY_E"
WRITER_MODEL = "UNIFY_MEMORY_WRITER_MODEL"
WRITER_ALLOWANCE = "UNIFY_MEMORY_WRITER_ALLOWANCE_USD_PER_TOKENS"
WRITER_RUN_GUARD = "UNIFY_MEMORY_WRITER_RUN_GUARD_USD"
WRITER_BASE_URL = "UNIFY_MEMORY_WRITER_BASE_URL"
WRITER_TOKEN = "UNIFY_MEMORY_WRITER_TOKEN"
WRITER_TOKEN_FD = "UNIFY_MEMORY_WRITER_TOKEN_FD"
WRITER_EFFORT = "UNIFY_MEMORY_WRITER_EFFORT"
WRITER_EFFORT_SCALE = "UNIFY_MEMORY_WRITER_EFFORT_SCALE"
WRITER_MAX_CALLS = "UNIFY_MEMORY_WRITER_MAX_CALLS"
SURFACING = "UNIFY_MEMORY_SURFACING"
DOCSTRINGS = "UNIFY_MEMORY_DOCSTRINGS"
SOFT_BUDGET = "UNIFY_MEMORY_SOFT_BUDGET"
WRITER_USAGE = "UNIFY_MEMORY_WRITER_USAGE"
# library mode: the writer's batch map, views and coverage; off by default
LIBRARY_MODE = "UNIFY_MEMORY_LIBRARY"
# library mode: the bed's runner posts the verdicts the actor sees as structured checker lines; off by default
CHECKER_VISIBLE = "UNIFY_MEMORY_LIBRARY_CHECKER_VISIBLE"
DIALOGUE = "UNIFY_MEMORY_DIALOGUE"

QA_FIXTURES = "UNIFY_MEMORY_QA_FIXTURES"
QA_MUTATION = "UNIFY_MEMORY_QA_MUTATION"
QA_MIN_KILL = "UNIFY_MEMORY_QA_MUTATION_MIN_KILL"
QA_DETERMINISM = "UNIFY_MEMORY_QA_DETERMINISM"
QA_REPLAY = "UNIFY_MEMORY_QA_REPLAY"
QA_FIXTURE_SIZE = "UNIFY_MEMORY_QA_FIXTURE_SIZE"

EXPERIENCE_BUDGET_DEFAULT = 150000
WRITER_MODEL_DEFAULT = "openai/gpt-6-sol"
WRITER_ALLOWANCE_DEFAULT = "0.00000073"
#: The writer's reasoning effort. ``actor`` (the default) matches the actor's effort for the run (the design rule:
#: "match the agent doing the task's effort with the agent that writes the stored memory"); ``low``, ``medium`` or
#: ``high`` fixes it, for a declared mismatch ablation only (e.g. a HIGH actor with a LOW storer).
WRITER_EFFORT_DEFAULT = "actor"
WRITER_EFFORTS = ("actor", "low", "medium", "high")
#: Per-pass limits by the pass's writer effort (one rule for every bed): the allowance cap (E x a_tok)
#: is multiplied by the effort's scale, and the pass makes at most the effort's calls. The offline diagnostic
#: measured USD per pass 0.095 (ARC MEDIUM), 0.178 (ARC HIGH), 0.116 (office MEDIUM), 0.30 (office HIGH) and
#: 10/14/13/23 calls per pass; at scale 1 and 40 calls a HIGH pass merged nothing.
WRITER_EFFORT_SCALE_DEFAULT = "low:1,medium:2,high:5"
WRITER_MAX_CALLS_DEFAULT = "low:40,medium:80,high:80"
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


def parse_writer_model(v: Any) -> str:
    value = _stripped(v) or WRITER_MODEL_DEFAULT
    if any(c.isspace() for c in value):
        raise ValueError(f"{WRITER_MODEL} must be a model id without spaces, not {v!r}")
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


def parse_writer_allowance(v: Any) -> str:
    """USD per token of experience: a positive plain decimal string."""
    text = _stripped(v) or WRITER_ALLOWANCE_DEFAULT
    return _plain_decimal(WRITER_ALLOWANCE, text, v, positive=True)


def parse_writer_run_guard(v: Any) -> str:
    """The run's writer USD guard: a non-negative plain decimal string, or ``""`` for none."""
    text = _stripped(v)
    if text == "":
        return ""
    return _plain_decimal(WRITER_RUN_GUARD, text, v, positive=False)


def writer_base_url_setting(v: Any) -> str:
    """``UNIFY_MEMORY_WRITER_BASE_URL`` as settings hold it: stripped, never refused (see :func:`writer_route`)."""
    return _stripped(v)


def writer_token_setting(v: Any) -> SecretStr:
    """``UNIFY_MEMORY_WRITER_TOKEN`` as settings hold it: a stripped ``SecretStr``, never refused."""
    getter = getattr(v, "get_secret_value", None)
    return SecretStr(_stripped(getter() if callable(getter) else v))


def writer_token_fd_setting(v: Any) -> str:
    """``UNIFY_MEMORY_WRITER_TOKEN_FD`` as settings hold it: stripped, never refused (a settings error would
    print its input, a token set there by mistake included); :func:`settle_writer_route_env` parses it with
    :func:`parse_writer_token_fd`.
    """
    return _stripped(v)


def parse_writer_token_fd(v: Any) -> int | None:
    """An inherited descriptor's number above 2 (not stdin, stdout or stderr), or ``None`` when empty.

    Its error never quotes the value (it would print a token set here by mistake).
    """
    refusal = f"{WRITER_TOKEN_FD} must be empty or a file descriptor number above 2"
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


class WriterRouteRefused(ValueError):
    """The writer's route is set but not in effect (malformed, half set, or not what the settings hold): no
    consolidation pass starts. Its text names settings and rules, never a value.
    """


def parse_writer_base_url(v: Any) -> str:
    """An http(s) base URL with a host and no user, password, query or fragment; ``""`` when empty.

    Errors name the rule broken, never the value (a URL with a password in it is a credential).
    """
    text = _stripped(v)
    if text == "":
        return ""
    refusal = f"{WRITER_BASE_URL} must be an http(s) URL with a host and no user, password, query or fragment"
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


#: The one host plain http may name: the launcher's loopback bridge to the writer's proxy listener (the token travels
#: in a header, in clear). Exactly this IPv4 literal, compared as text: no other name, address or DNS lookup;
#: always with an explicit port, as the launcher sends it (``http://127.0.0.1:<port>/<route>/v1``).
HTTP_HOST = "127.0.0.1"


def parse_writer_token(v: Any) -> SecretStr:
    """The token as a ``SecretStr``: URL-safe characters only (it goes in a header); empty when unset.

    Errors never quote the value.
    """
    value = writer_token_setting(v).get_secret_value()
    if value and not _BEARER_TOKEN.fullmatch(value):
        raise ValueError(
            f"{WRITER_TOKEN} must be a bearer token: at least {_MIN_TOKEN_CHARS} characters from "
            "A-Z a-z 0-9 . _ ~ -",
        )
    return SecretStr(value)


#: At least 16 RFC 3986 unreserved characters (``secrets.token_urlsafe`` output fits): it goes in a header,
#: and no percent-encoding, JSON or other escaping form differs from it, so redaction by value always finds it
#: (``+``, ``/`` and ``=`` would turn into ``%2B``, ``\/`` and the like).
_MIN_TOKEN_CHARS = 16
_BEARER_TOKEN = re.compile(r"[A-Za-z0-9._~-]{%d,}" % _MIN_TOKEN_CHARS)


def writer_route(base_url: Any, token: Any) -> tuple[str, SecretStr] | None:
    """The writer's route, ``(base_url, token)``, or ``None`` when neither is set (the shipped route).

    Exactly one set is refused (fail closed: no pass starts, so no call is made), as is either value in a
    wrong form. No error quotes a value.
    """
    if _ENV_REFUSAL is not None:
        raise WriterRouteRefused(_ENV_REFUSAL)
    raw = token.get_secret_value() if isinstance(token, SecretStr) else token
    register_secret(WRITER_TOKEN, raw)  # kept redacted by value even if refused below
    try:
        base = parse_writer_base_url(base_url)
        secret = parse_writer_token(token)
    except ValueError as exc:  # the parsers' text names the rule, never the value
        raise WriterRouteRefused(str(exc)) from None
    if not base and not secret.get_secret_value():
        return None
    if not base or not secret.get_secret_value():
        missing = WRITER_BASE_URL if not base else WRITER_TOKEN
        raise WriterRouteRefused(
            f"{WRITER_BASE_URL} and {WRITER_TOKEN} are set together or not at all; "
            f"{missing} is empty, so no consolidation pass starts",
        )
    return base, secret


#: Why every pass is refused, once :func:`settle_writer_route_env` found the environment and the settings
#: disagreeing about the writer's route; sticky for the life of the process.
_ENV_REFUSAL: str | None = None


#: The most bytes the writer's token may take on its descriptor, and how long reading it may take.
TOKEN_FD_MAX_BYTES = 4096
TOKEN_FD_TIMEOUT_S = 5.0

#: The writer's token as read from ``UNIFY_MEMORY_WRITER_TOKEN_FD`` (once per process; never in ``os.environ``).
_FD_TOKEN: SecretStr | None = None
_FD_TRIED = False
#: The descriptor number the first settle read (settings' value), kept once it has left ``os.environ``.
_FD_HELD = ""


def _read_token_fd(
    fd: int,
    *,
    fd_name: str = WRITER_TOKEN_FD,
    secret_name: str = WRITER_TOKEN,
) -> str:
    """A route token (the writer's, or the fork lane's with its own names) from inherited descriptor *fd*, which is then
    closed; refusals name the rule, never a value."""
    where = f"{fd_name} names descriptor {fd}, which"
    try:
        mode = os.fstat(fd).st_mode
        inherited = os.get_inheritable(fd)
    except OSError:
        raise WriterRouteRefused(f"{where} is not open") from None
    # opened by this process (PEP 446), not passed by the launcher: left alone
    if not inherited:
        raise WriterRouteRefused(f"{where} was not inherited by this process")
    data = b""
    try:
        if not (stat.S_ISFIFO(mode) or stat.S_ISREG(mode)):
            raise WriterRouteRefused(f"{where} is not a pipe or a file")
        poller = select.poll()
        poller.register(fd, select.POLLIN)
        deadline = time.monotonic() + TOKEN_FD_TIMEOUT_S
        while True:
            left = deadline - time.monotonic()
            if left <= 0 or not poller.poll(max(1, int(left * 1000))):
                raise WriterRouteRefused(
                    f"{where} did not reach its end within {TOKEN_FD_TIMEOUT_S:g} s",
                )
            chunk = os.read(fd, TOKEN_FD_MAX_BYTES + 1 - len(data))
            if not chunk:
                break
            data += chunk
            if len(data) > TOKEN_FD_MAX_BYTES:
                raise WriterRouteRefused(
                    f"{where} holds more than {TOKEN_FD_MAX_BYTES} bytes",
                )
    except OSError:
        raise WriterRouteRefused(f"{where} could not be read") from None
    finally:
        register_secret(secret_name, data.decode("latin-1"))
        try:
            os.close(fd)  # whatever happened: no later child inherits it
        except OSError:
            pass
    text = data.decode("latin-1")
    text = text[:-1] if text.endswith("\n") else text
    register_secret(secret_name, text)
    if not text:
        raise WriterRouteRefused(f"{where} is empty")
    if not _BEARER_TOKEN.fullmatch(text):
        raise WriterRouteRefused(
            f"{where} does not hold a bearer token: at least {_MIN_TOKEN_CHARS} characters from "
            "A-Z a-z 0-9 . _ ~ -",
        )
    return text


def writer_token(settings: Any) -> Any:
    """The token the writer's route uses: the one read from ``UNIFY_MEMORY_WRITER_TOKEN_FD`` when that is set (empty
    if it could not be read; :func:`writer_route` then refuses), else ``UNIFY_MEMORY_WRITER_TOKEN``.
    """
    if _stripped(getattr(settings, WRITER_TOKEN_FD, "")):
        return _FD_TOKEN if _FD_TOKEN is not None else SecretStr("")
    return getattr(settings, WRITER_TOKEN, "")


def settle_writer_route_env(
    environ: MutableMapping[str, str],
    settings: Any,
) -> str | None:
    """Keep the writer's token out of *environ* and refuse every pass if *environ* and *settings* disagree.

    Called once the settings are built (``unify/settings.py``), again after the CLI loads ``.env`` and whenever
    a pass is about to start. The first call with ``UNIFY_MEMORY_WRITER_TOKEN_FD`` set reads the token from that
    descriptor and closes it (:func:`_read_token_fd`); a failed read, or both that and
    ``UNIFY_MEMORY_WRITER_TOKEN`` set, refuses every pass. Every case variant of
    ``UNIFY_MEMORY_WRITER_TOKEN_FD`` then leaves *environ* (the number is held here for the stray check, and
    no later call reads a descriptor again), so no child spawned afterwards sees the name. Every
    case variant of ``UNIFY_MEMORY_WRITER_TOKEN`` is removed from *environ* and its value registered for
    redaction (pydantic-settings matches names in any case). A token or base URL found in *environ* that the
    settings do not hold (one that arrived after they were built, from ``.env`` say, or two case variants
    with different values) would otherwise silently leave the writer on the actor's route, so from then on
    :func:`writer_route` refuses every pass with the returned message. The message names settings, never
    values. Returns ``None`` when they agree.
    """
    global _ENV_REFUSAL, _FD_TOKEN, _FD_TRIED, _FD_HELD
    raw = getattr(settings, WRITER_TOKEN, "")
    held_token = _stripped(
        raw.get_secret_value() if isinstance(raw, SecretStr) else raw,
    )
    register_secret(WRITER_TOKEN, held_token)
    held_base = _stripped(getattr(settings, WRITER_BASE_URL, ""))
    held_fd = _stripped(getattr(settings, WRITER_TOKEN_FD, ""))
    fd_refusal = None
    if held_fd and not _FD_TRIED:
        _FD_TRIED = True
        _FD_HELD = held_fd
        try:  # a WriterRouteRefused is a ValueError; neither quotes a value
            fd = parse_writer_token_fd(held_fd)
            _FD_TOKEN = SecretStr(_read_token_fd(int(fd or 0)))
        except ValueError as exc:
            register_secret(WRITER_TOKEN_FD, held_fd)  # a token set there by mistake
            fd_refusal = f"{exc}; no consolidation pass starts"
    if held_fd and held_token:
        fd_refusal = fd_refusal or (
            f"{WRITER_TOKEN} and {WRITER_TOKEN_FD} are both set (set exactly one); no consolidation pass starts"
        )
    stray: list[str] = []
    for name in [k for k in environ if k.upper() == WRITER_TOKEN]:
        value = environ.pop(name, "")
        register_secret(WRITER_TOKEN, value)
        if _stripped(value) != held_token:
            stray.append(WRITER_TOKEN)
    for name in [k for k in environ if k.upper() == WRITER_BASE_URL]:
        if _stripped(environ.get(name)) != held_base:
            stray.append(WRITER_BASE_URL)
    for name in [k for k in environ if k.upper() == WRITER_TOKEN_FD]:
        # removed once read, so no later child sees the name; compared with the number held
        if _stripped(environ.pop(name, "")) != (held_fd or _FD_HELD):
            stray.append(WRITER_TOKEN_FD)
    if fd_refusal and _ENV_REFUSAL is None:
        _ENV_REFUSAL = fd_refusal
    if stray and _ENV_REFUSAL is None:
        _ENV_REFUSAL = (
            f"{' and '.join(sorted(set(stray)))} reached the environment after settings were read "
            "(from .env, or in another letter case), so the writer's route is not what the settings hold; no "
            "consolidation pass starts. Set both in the controller's process environment, not .env"
        )
    return _ENV_REFUSAL


def parse_writer_effort(v: Any) -> str:
    """The writer's reasoning effort: ``actor`` (the default: the actor's effort for the run), or a fixed ``low``,
    ``medium`` or ``high`` for a declared mismatch ablation."""
    value = _stripped(v).lower() or WRITER_EFFORT_DEFAULT
    if value not in WRITER_EFFORTS:
        raise ValueError(
            f"{WRITER_EFFORT} must be one of {', '.join(WRITER_EFFORTS)}, not {v!r}",
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
        if (
            not sep
            or key not in WRITER_EFFORTS[1:]
            or key in got
            or not value_ok(value)
        ):
            raise ValueError(refusal)
        got[key] = value
    if len(got) != len(WRITER_EFFORTS) - 1:
        raise ValueError(refusal)
    return {k: got[k] for k in WRITER_EFFORTS[1:]}


def _positive_decimal(text: str) -> bool:
    return bool(_PLAIN_DECIMAL.fullmatch(text)) and Decimal(text) > 0


def _positive_int(text: str) -> bool:
    return bool(_DIGITS.fullmatch(text)) and int(text) > 0


def writer_effort_scale_map(v: Any) -> dict[str, Decimal]:
    """The allowance-cap multiplier per writer effort, from ``UNIFY_MEMORY_WRITER_EFFORT_SCALE``'s value."""
    m = _effort_map(
        WRITER_EFFORT_SCALE,
        v,
        WRITER_EFFORT_SCALE_DEFAULT,
        _positive_decimal,
    )
    return {k: Decimal(x) for k, x in m.items()}


def writer_max_calls_map(v: Any) -> dict[str, int]:
    """The per-pass call limit per writer effort, from ``UNIFY_MEMORY_WRITER_MAX_CALLS``'s value."""
    m = _effort_map(WRITER_MAX_CALLS, v, WRITER_MAX_CALLS_DEFAULT, _positive_int)
    return {k: int(x) for k, x in m.items()}


def parse_writer_effort_scale(v: Any) -> str:
    """Positive plain decimals per effort; normalised to ``low:..,medium:..,high:..``."""
    m = _effort_map(
        WRITER_EFFORT_SCALE,
        v,
        WRITER_EFFORT_SCALE_DEFAULT,
        _positive_decimal,
    )
    return ",".join(f"{k}:{x}" for k, x in m.items())


def parse_writer_max_calls(v: Any) -> str:
    """Positive whole numbers per effort; normalised to ``low:..,medium:..,high:..``."""
    m = _effort_map(WRITER_MAX_CALLS, v, WRITER_MAX_CALLS_DEFAULT, _positive_int)
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


def parse_memory_library_mode(v: Any) -> str:
    """``UNIFY_MEMORY_LIBRARY``: ``off`` (also for empty) or ``on``."""
    return _on_off(LIBRARY_MODE, v)


parse_library_mode = parse_memory_library_mode


def library_enabled(settings: Any) -> bool:
    """Whether the library mode is on in *settings* (``unify.settings.SETTINGS`` or any object; missing is off)."""
    return parse_library_mode(getattr(settings, LIBRARY_MODE, "") or "") == "on"


def parse_checker_visible(v: Any) -> str:
    """``UNIFY_MEMORY_LIBRARY_CHECKER_VISIBLE``: ``off`` (also for empty) or ``on``. On (with ``UNIFY_MEMORY_LIBRARY``),
    the CLI takes ``{"checker": {"label": "pass"|"fail"}}`` lines: the verdicts the bed shows the actor, recorded
    as agent-visible checker signals."""
    return _on_off(CHECKER_VISIBLE, v)


#: v2 settings whose value contradicts the library mode's prompts or gate: the v2 docstring check (fixtures
#: under env/<channel>/tests/), the catalogue guide, memlab replay in tests, and the fixture-size cap (no
#: caps on data).
LIBRARY_CONFLICTS: tuple[tuple[str, str], ...] = (
    ("UNIFY_MEMORY_DOCSTRINGS", "on"),
    ("UNIFY_MEMORY_SURFACING", "catalogue"),
    ("UNIFY_MEMORY_QA_REPLAY", "on"),
    ("UNIFY_MEMORY_QA_FIXTURE_SIZE", "on"),
)


def library_conflicts(settings: Any) -> list[str]:
    """``NAME=value`` for each setting in :data:`LIBRARY_CONFLICTS` at its contradicting value, in library mode."""
    if not library_enabled(settings):
        return []
    out = []
    for name, bad in LIBRARY_CONFLICTS:
        if str(getattr(settings, name, "") or "").strip().lower() == bad:
            out.append(f"{name}={bad}")
    return out


LIBRARY_E = "UNIFY_MEMORY_LIBRARY_E"
#: The library-mode writer's reader calls per pass (PassConfig.max_reads), settable for the replay and the reruns.
LIBRARY_MAX_READS = "UNIFY_MEMORY_LIBRARY_MAX_READS"
#: Opt-in: the CLI waits for the pass slot it spawned before it exits, for hosts whose sandbox ends
#: every process of the controller with it (a PID namespace); off: the request returns at once, as built.
LIBRARY_WAIT_SLOT = "UNIFY_MEMORY_LIBRARY_WAIT_SLOT"
#: At each episode's end the actor's own conversation is forked once, in a confined
#: process that writes only to the episode's staging dir (integration/fork.py); off: nothing happens, as built.
LIBRARY_FORK = "UNIFY_MEMORY_LIBRARY_FORK"
#: Retired: helpers are launched by the writer itself with the ``delegate`` tool (helpers.py), never by the harness.
#: Empty or ``off`` is accepted as a no-op; any other value is refused, so a configuration that still asks for
#: harness-launched helpers fails loudly instead of running without them.
LIBRARY_ANALYSTS = "UNIFY_MEMORY_LIBRARY_ANALYSTS"
#: The model of writer-launched helpers (helpers.py). Empty: the model the episode-end fork uses, which is the
#: actor's own model (``UNIFY_MODEL``). Never a model name in code: this setting or that one.
LIBRARY_HELPER_MODEL = "UNIFY_MEMORY_LIBRARY_HELPER_MODEL"
#: The candidate pool (pool.py): ``on`` keeps every proposal (staging, drafts, deferrals) open across passes on the
#: memory repo's refs/heads/pool until the writer closes it; ``off``: none.
LIBRARY_POOL = "UNIFY_MEMORY_LIBRARY_POOL"
#: ``on`` gives the library writer and the curator the ``write`` and ``edit`` file tools on /memory
#: (edit_tools.py); ``off``: neither tool exists and every prompt is as before.
LIBRARY_EDIT_TOOLS = "UNIFY_MEMORY_LIBRARY_EDIT_TOOLS"
#: On by default (also when empty): the memory agents (writer, curator, helpers) compact their conversations as the
#: agent loop compacts its own (writer_pass.compact_by_summary: the same trigger, summary and restart, over each
#: agent's own route); ``off``: the conversation is never changed.
LIBRARY_COMPACT = "UNIFY_MEMORY_LIBRARY_COMPACT"
#: The former reading rule. Finish never waits on reading: empty and ``optional`` are accepted and change nothing;
#: ``required`` (the removed rule) is refused, so a configuration that still asks for it fails loudly at start-up.
LIBRARY_COVERAGE = "UNIFY_MEMORY_LIBRARY_COVERAGE"
#: How the actor's prompt reaches the library index: ``inline`` (default) ends the prompt with the index view, as
#: built; ``pointer`` ends it with a constant guide that points to ``memory.index()``, ``memory.show()`` and
#: ``memory.find()`` in a cell, so the prompt never changes as the library grows.
LIBRARY_ACTOR_INDEX = "UNIFY_MEMORY_LIBRARY_ACTOR_INDEX"
LIBRARY_ACTOR_INDEX_VALUES = ("inline", "pointer")
LIBRARY_COVERAGE_VALUES = ("optional",)
LIBRARY_MAX_READS_DEFAULT = 400
#: The library-mode writer's USD per recorded token: v2's rate until the offline replay sizes it for coverage.
LIBRARY_WRITER_USD_PER_TOKEN = "UNIFY_MEMORY_LIBRARY_WRITER_USD_PER_TOKEN"
LIBRARY_PASS_WALL_S = "UNIFY_MEMORY_LIBRARY_PASS_WALL_S"
WRITER_JOURNAL = "UNIFY_MEMORY_WRITER_JOURNAL"
#: The writer model's prices, USD per million tokens, as the run pins them: a JSON object with ``input`` and
#: ``output`` and optionally ``cache_write``. Empty: a writer call with no charge stays unpriced.
WRITER_PRICES = "UNIFY_MEMORY_WRITER_PRICES_USD_PER_MILLION"
WRITER_PRICE_KEYS = ("input", "output")
WRITER_OPTIONAL_PRICE_KEYS = ("cache_write",)
#: The pass's wall-clock bound in seconds: calibrated with the budget by the offline replay.
LIBRARY_PASS_WALL_S_DEFAULT = 2700


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


def parse_library_e(v: Any) -> int:
    """``UNIFY_MEMORY_LIBRARY_E``: E under the library mode, positive digits (empty: 100,000)."""
    from ..trigger import EXPERIENCE_BUDGET_LIBRARY

    return _positive_whole(LIBRARY_E, EXPERIENCE_BUDGET_LIBRARY, v)


def parse_library_pass_wall_s(v: Any) -> int:
    """``UNIFY_MEMORY_LIBRARY_PASS_WALL_S``: one pass's wall-clock bound in seconds (empty: 2700)."""
    return _positive_whole(LIBRARY_PASS_WALL_S, LIBRARY_PASS_WALL_S_DEFAULT, v)


def parse_writer_journal(v: Any) -> str:
    """Empty, or the absolute path of the writer route proxy's journal (JSON lines with ``request_attempt_id``,
    ``generation_id`` and ``account_charge``), read only to reconcile cancelled calls.
    """
    text = _stripped(v)
    if text and not text.startswith("/"):
        raise ValueError(f"{WRITER_JOURNAL} must be empty or an absolute path")
    return text


def parse_library_writer_usd_per_token(v: Any) -> str:
    """``UNIFY_MEMORY_LIBRARY_WRITER_USD_PER_TOKEN``: a positive plain decimal string (empty: v2's rate, 0.00000073)."""
    return _plain_decimal(
        LIBRARY_WRITER_USD_PER_TOKEN,
        _stripped(v) or WRITER_ALLOWANCE_DEFAULT,
        v,
        positive=True,
    )


def parse_writer_prices(v: Any) -> str:
    """``UNIFY_MEMORY_WRITER_PRICES_USD_PER_MILLION``: empty, or a JSON object of non-negative decimal prices with
    ``input`` and ``output`` (and optionally ``cache_write``) and no other key; returned in one canonical form.
    """
    text = _stripped(v)
    if text == "":
        return ""
    refusal = (
        f"{WRITER_PRICES} must be a JSON object with input and output (and optionally cache_write) USD per million "
        "tokens"
    )
    try:
        raw = json.loads(text)
    except ValueError:
        raise ValueError(refusal) from None
    allowed = set(WRITER_PRICE_KEYS) | set(WRITER_OPTIONAL_PRICE_KEYS)
    if (
        not isinstance(raw, dict)
        or not set(WRITER_PRICE_KEYS) <= set(raw)
        or set(raw) - allowed
    ):
        raise ValueError(refusal)
    out = {}
    for key, value in raw.items():
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            raise ValueError(refusal)
        out[key] = _plain_decimal(
            f"{WRITER_PRICES} {key}",
            str(value).strip(),
            v,
            positive=False,
        )
    return json.dumps(out, sort_keys=True)


def writer_prices(settings: Any) -> dict[str, Decimal] | None:
    """The pinned writer prices per million tokens, or None when none are set."""
    text = parse_writer_prices(getattr(settings, WRITER_PRICES, ""))
    return {k: Decimal(v) for k, v in json.loads(text).items()} if text else None


def library_writer_usd_per_token(settings: Any) -> Decimal:
    """The USD per recorded token of a library-mode pass's cap (a setting, so the replay can raise it without code)."""
    return Decimal(
        parse_library_writer_usd_per_token(
            getattr(settings, LIBRARY_WRITER_USD_PER_TOKEN, ""),
        ),
    )


def parse_library_max_reads(v: Any) -> int:
    """``UNIFY_MEMORY_LIBRARY_MAX_READS``: a pass's reader calls in library mode, positive digits (empty: 400)."""
    return _positive_whole(LIBRARY_MAX_READS, LIBRARY_MAX_READS_DEFAULT, v)


def parse_library_wait_slot(v: Any) -> str:
    """``UNIFY_MEMORY_LIBRARY_WAIT_SLOT``: ``off`` (also for empty) or ``on``."""
    return _on_off(LIBRARY_WAIT_SLOT, v)


def library_wait_slot(settings: Any) -> bool:
    return (
        parse_library_wait_slot(getattr(settings, LIBRARY_WAIT_SLOT, "") or "") == "on"
    )


def parse_library_fork(v: Any) -> str:
    """``UNIFY_MEMORY_LIBRARY_FORK``: ``off`` (also for empty) or ``on``."""
    return _on_off(LIBRARY_FORK, v)


def library_fork(settings: Any) -> bool:
    return parse_library_fork(getattr(settings, LIBRARY_FORK, "") or "") == "on"


# --- the fork lane's route (docs/design/fork-lane-design.md) ----------------------------------------------------
#: The key proxy's fork route: its base URL and the descriptor carrying its token, handed by the runner's learner to
#: this controller's spawn alone. Read once (:func:`settle_fork_route_env`), then both names leave the environment;
#: neither ever reaches an actor box (``HARNESS_ONLY_ENV``); the token lives in this process and goes to the fork worker
#: on its stdin only.
LIBRARY_FORK_BASE_URL = "UNIFY_MEMORY_LIBRARY_FORK_BASE_URL"
LIBRARY_FORK_TOKEN_FD = "UNIFY_MEMORY_LIBRARY_FORK_TOKEN_FD"
LIBRARY_FORK_TOKEN = "UNIFY_MEMORY_LIBRARY_FORK_TOKEN"  # the redaction name of the token's value (never a setting)
_FORK_BASE_FORM = re.compile(r"http://127\.0\.0\.1:([0-9]{1,5})/fork/v1")
_FORK_TOKEN: str | None = None
_FORK_BASE: str = ""
_FORK_TRIED = False
_FORK_REFUSAL: str | None = None


def parse_library_fork_base_url(v: Any) -> str:
    """``UNIFY_MEMORY_LIBRARY_FORK_BASE_URL``: empty, or ``http://127.0.0.1:<port>/fork/v1`` exactly."""
    s = _stripped(v)
    if not s:
        return ""
    m = _FORK_BASE_FORM.fullmatch(s)
    if m is None or not 1 <= int(m.group(1)) <= 65535:
        raise ValueError(
            f"{LIBRARY_FORK_BASE_URL} must be http://127.0.0.1:<port>/fork/v1 or empty",
        )
    return s


def parse_library_fork_token_fd(v: Any) -> str:
    """``UNIFY_MEMORY_LIBRARY_FORK_TOKEN_FD``: empty, or a descriptor number above 2."""
    s = _stripped(v)
    if not s:
        return ""
    if not s.isdigit() or int(s) <= 2:
        raise ValueError(
            f"{LIBRARY_FORK_TOKEN_FD} must be a descriptor number above 2 or empty",
        )
    return s


def settle_fork_route_env(
    environ: MutableMapping[str, str],
    settings: Any,
) -> str | None:
    """The first call with ``UNIFY_MEMORY_LIBRARY_FORK_TOKEN_FD`` set reads the fork token from that inherited
    descriptor once (:func:`_read_token_fd`: inherited, a pipe or file, size-capped, timed, closed; registered for
    redaction by value) and keeps it in this process. Every call removes every case variant of both route names from
    *environ*, so no child spawned afterwards sees them. The base URL and the descriptor are set together or not at
    all. Returns why the fork lane is refused (sticky), or None."""
    global _FORK_TOKEN, _FORK_BASE, _FORK_TRIED, _FORK_REFUSAL
    try:
        base = parse_library_fork_base_url(getattr(settings, LIBRARY_FORK_BASE_URL, ""))
        fd = parse_library_fork_token_fd(getattr(settings, LIBRARY_FORK_TOKEN_FD, ""))
    except ValueError as exc:
        base, fd = "", ""
        _FORK_REFUSAL = _FORK_REFUSAL or str(exc)
    if fd and not _FORK_TRIED:
        _FORK_TRIED = True
        try:
            _FORK_TOKEN = _read_token_fd(
                int(fd),
                fd_name=LIBRARY_FORK_TOKEN_FD,
                secret_name=LIBRARY_FORK_TOKEN,
            )
        except WriterRouteRefused as exc:
            _FORK_REFUSAL = _FORK_REFUSAL or str(exc)
    if base:
        _FORK_BASE = base
    if bool(base) != bool(fd) and _FORK_REFUSAL is None:
        _FORK_REFUSAL = f"{LIBRARY_FORK_BASE_URL} and {LIBRARY_FORK_TOKEN_FD} are set together or not at all"
    for name in list(environ):
        if name.upper() in (LIBRARY_FORK_BASE_URL, LIBRARY_FORK_TOKEN_FD):
            environ.pop(name, None)
    return _FORK_REFUSAL


def fork_lane_route() -> tuple[tuple[str, str] | None, str | None]:
    """``((chat completions URL, token), None)`` on the key proxy's fork lane, or ``(None, why)``: a refusal from
    :func:`settle_fork_route_env`, or ``no fork lane``. Never any other route (no fallback to the actor's).
    """
    if _FORK_REFUSAL is not None:
        return None, _FORK_REFUSAL
    if not _FORK_BASE or not _FORK_TOKEN:
        return None, "no fork lane"
    return (_FORK_BASE + "/chat/completions", _FORK_TOKEN), None


def parse_library_pool(v: Any) -> str:
    """``UNIFY_MEMORY_LIBRARY_POOL``: ``off`` (also for empty) or ``on``."""
    return _on_off(LIBRARY_POOL, v)


def library_pool(settings: Any) -> bool:
    return parse_library_pool(getattr(settings, LIBRARY_POOL, "") or "") == "on"


def parse_library_edit_tools(v: Any) -> str:
    """``UNIFY_MEMORY_LIBRARY_EDIT_TOOLS``: ``off`` (also for empty) or ``on``."""
    return _on_off(LIBRARY_EDIT_TOOLS, v)


def library_edit_tools(settings: Any) -> bool:
    return (
        parse_library_edit_tools(getattr(settings, LIBRARY_EDIT_TOOLS, "") or "")
        == "on"
    )


def parse_library_actor_index(v: Any) -> str:
    """``UNIFY_MEMORY_LIBRARY_ACTOR_INDEX``: ``inline`` (also for empty) or ``pointer``."""
    value = _stripped(v).lower() or "inline"
    if value not in LIBRARY_ACTOR_INDEX_VALUES:
        raise ValueError(
            f"{LIBRARY_ACTOR_INDEX} must be empty, 'inline' or 'pointer', not {v!r}"[
                :200
            ],
        )
    return value


def library_actor_index(settings: Any) -> str:
    return parse_library_actor_index(getattr(settings, LIBRARY_ACTOR_INDEX, "") or "")


def parse_library_compact(v: Any) -> str:
    """``UNIFY_MEMORY_LIBRARY_COMPACT``: ``on`` (also for empty: every agent compacts under the one shared rule) or
    ``off``."""
    return _on_off(LIBRARY_COMPACT, _stripped(v) or "on")


def library_compact(settings: Any) -> bool:
    return parse_library_compact(getattr(settings, LIBRARY_COMPACT, "") or "") == "on"


def parse_library_coverage(v: Any) -> str:
    """``UNIFY_MEMORY_LIBRARY_COVERAGE``: empty or ``optional`` (a no-op: finish never waits on reading). The removed
    ``required`` rule, and anything else, is refused."""
    value = _stripped(v).lower() or "optional"
    if value == "required":
        raise ValueError(
            f"{LIBRARY_COVERAGE}=required is no longer supported: the reading rule was removed and finish never "
            "waits on reading; leave the setting empty or set it to 'optional'",
        )
    if value not in LIBRARY_COVERAGE_VALUES:
        raise ValueError(
            f"{LIBRARY_COVERAGE} must be empty or 'optional', not {v!r}"[:200],
        )
    return value


def library_coverage(settings: Any) -> str:
    return parse_library_coverage(getattr(settings, LIBRARY_COVERAGE, "") or "")


def parse_library_analysts(v: Any) -> str:
    """``UNIFY_MEMORY_LIBRARY_ANALYSTS`` (retired): empty or ``off`` only; anything else is refused."""
    s = str(v or "").strip().lower() or "off"
    if s != "off":
        raise ValueError(
            f"{LIBRARY_ANALYSTS} is retired (helpers are launched by the writer with the delegate tool); "
            f"it must be empty or off, got {v!r}"[:300],
        )
    return s


def parse_library_helper_model(v: Any) -> str:
    """``UNIFY_MEMORY_LIBRARY_HELPER_MODEL``: empty (the fork's model, the actor's ``UNIFY_MODEL``) or a model id
    without spaces."""
    s = _stripped(v)
    if s and (any(ch.isspace() for ch in s) or len(s) > 200):
        raise ValueError(
            f"{LIBRARY_HELPER_MODEL} must be empty or a model id without spaces",
        )
    return s


def library_helper_model(settings: Any) -> str:
    """The helpers' model: :data:`LIBRARY_HELPER_MODEL`, else the model the fork uses (the actor's ``UNIFY_MODEL``)."""
    own = parse_library_helper_model(getattr(settings, LIBRARY_HELPER_MODEL, "") or "")
    return own or _stripped(getattr(settings, "UNIFY_MODEL", "") or "")


# --- the helper lane's route (same form and rules as the fork lane's) -------------------------------------------
#: The key proxy's helper lane: its own base URL and token, so helper spend is selected by the token (never by a
#: header or the model, which may equal the actor's) and booked on the lane's own ledger. The token comes from an
#: inherited descriptor only, is read once (:func:`settle_helper_route_env`), and stays in this process: helper
#: model calls are made host-side by the writer's process, never inside a helper's box.
LIBRARY_HELPER_BASE_URL = "UNIFY_MEMORY_LIBRARY_HELPER_BASE_URL"
LIBRARY_HELPER_TOKEN_FD = "UNIFY_MEMORY_LIBRARY_HELPER_TOKEN_FD"
LIBRARY_HELPER_TOKEN = "UNIFY_MEMORY_LIBRARY_HELPER_TOKEN"  # the redaction name of the token's value (never a setting)
_HELPER_BASE_FORM = re.compile(r"http://127\.0\.0\.1:([0-9]{1,5})/helper/v1")
_HELPER_TOKEN: str | None = None
_HELPER_BASE: str = ""
_HELPER_TRIED = False
_HELPER_REFUSAL: str | None = None


def parse_library_helper_base_url(v: Any) -> str:
    """``UNIFY_MEMORY_LIBRARY_HELPER_BASE_URL``: empty, or ``http://127.0.0.1:<port>/helper/v1`` exactly."""
    s = _stripped(v)
    if not s:
        return ""
    m = _HELPER_BASE_FORM.fullmatch(s)
    if m is None or not 1 <= int(m.group(1)) <= 65535:
        raise ValueError(
            f"{LIBRARY_HELPER_BASE_URL} must be http://127.0.0.1:<port>/helper/v1 or empty",
        )
    return s


def parse_library_helper_token_fd(v: Any) -> str:
    """``UNIFY_MEMORY_LIBRARY_HELPER_TOKEN_FD``: empty, or a descriptor number above 2."""
    s = _stripped(v)
    if not s:
        return ""
    if not s.isdigit() or int(s) <= 2:
        raise ValueError(
            f"{LIBRARY_HELPER_TOKEN_FD} must be a descriptor number above 2 or empty",
        )
    return s


def settle_helper_route_env(
    environ: MutableMapping[str, str],
    settings: Any,
) -> str | None:
    """As :func:`settle_fork_route_env`, for the helper lane: the token is read once from its inherited descriptor
    (then closed, registered for redaction), both names leave *environ* so no child sees them, and the base URL and
    descriptor are set together or not at all. Returns why the helper lane is refused (sticky), or None.
    """
    global _HELPER_TOKEN, _HELPER_BASE, _HELPER_TRIED, _HELPER_REFUSAL
    try:
        base = parse_library_helper_base_url(
            getattr(settings, LIBRARY_HELPER_BASE_URL, ""),
        )
        fd = parse_library_helper_token_fd(
            getattr(settings, LIBRARY_HELPER_TOKEN_FD, ""),
        )
    except ValueError as exc:
        base, fd = "", ""
        _HELPER_REFUSAL = _HELPER_REFUSAL or str(exc)
    if fd and not _HELPER_TRIED:
        _HELPER_TRIED = True
        try:
            _HELPER_TOKEN = _read_token_fd(
                int(fd),
                fd_name=LIBRARY_HELPER_TOKEN_FD,
                secret_name=LIBRARY_HELPER_TOKEN,
            )
        except WriterRouteRefused as exc:
            _HELPER_REFUSAL = _HELPER_REFUSAL or str(exc)
    if base:
        _HELPER_BASE = base
    if bool(base) != bool(fd) and _HELPER_REFUSAL is None:
        _HELPER_REFUSAL = f"{LIBRARY_HELPER_BASE_URL} and {LIBRARY_HELPER_TOKEN_FD} are set together or not at all"
    for name in list(environ):
        if name.upper() in (LIBRARY_HELPER_BASE_URL, LIBRARY_HELPER_TOKEN_FD):
            environ.pop(name, None)
    return _HELPER_REFUSAL


def helper_lane_route() -> tuple[tuple[str, SecretStr] | None, str | None]:
    """``((base URL, token), None)`` on the key proxy's helper lane, or ``(None, why)``: a refusal from
    :func:`settle_helper_route_env`, or ``no helper lane``. Never another lane's route (no fallback to the writer's
    or the actor's)."""
    if _HELPER_REFUSAL is not None:
        return None, _HELPER_REFUSAL
    if not _HELPER_BASE or not _HELPER_TOKEN:
        return None, "no helper lane"
    return (_HELPER_BASE, SecretStr(_HELPER_TOKEN)), None


def library_max_reads(settings: Any) -> int:
    return parse_library_max_reads(getattr(settings, LIBRARY_MAX_READS, ""))


def library_experience_budget(settings: Any) -> int:
    return parse_library_e(getattr(settings, LIBRARY_E, ""))


def library_pass_wall_s(settings: Any) -> int:
    return parse_library_pass_wall_s(getattr(settings, LIBRARY_PASS_WALL_S, ""))


def writer_journal(settings: Any) -> str | None:
    return parse_writer_journal(getattr(settings, WRITER_JOURNAL, "")) or None


def checker_visible(settings: Any) -> bool:
    """The bed's declaration ``checker_visible_to_actor``: whether the checker's verdict is one
    the actor itself sees. Only then does the item lifecycle count checker signals. Missing is off. The one
    switch the checker-line reader reads too."""
    return parse_checker_visible(getattr(settings, CHECKER_VISIBLE, "") or "") == "on"


@dataclass(frozen=True)
class SurfacingOptions:
    """The three library-mode surfacing switches, parsed; the defaults are the v2 screen build's behaviour."""

    surfacing: str = SURFACING_DEFAULT
    docstrings: bool = False
    soft_budget: bool = False

    @property
    def catalogue(self) -> bool:
        return self.surfacing == "catalogue"

    def gate_kwargs(self) -> dict[str, Any]:
        """The :class:`unify.memory.gate.Gate` keyword arguments these switches set (the writer's brief follows
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


def parse_writer_usage(v: Any) -> str:
    """``on``, or ``""`` for off (empty or ``off``)."""
    value = _stripped(v).lower()
    if value in ("", "off"):
        return ""
    if value != "on":
        raise ValueError(f"{WRITER_USAGE} must be empty, 'off' or 'on', not {v!r}")
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


#: The validator for each setting (unify/settings.py ``parse_memory``).
PARSERS = {
    SWITCH: parse_switch,
    EXPERIENCE_BUDGET: parse_experience_budget,
    WRITER_MODEL: parse_writer_model,
    WRITER_ALLOWANCE: parse_writer_allowance,
    WRITER_RUN_GUARD: parse_writer_run_guard,
    # normalised only; checked by writer_route when a pass starts
    WRITER_BASE_URL: writer_base_url_setting,
    WRITER_TOKEN: writer_token_setting,
    WRITER_TOKEN_FD: writer_token_fd_setting,
    WRITER_EFFORT: parse_writer_effort,
    WRITER_EFFORT_SCALE: parse_writer_effort_scale,
    WRITER_MAX_CALLS: parse_writer_max_calls,
    SURFACING: parse_surfacing,
    DOCSTRINGS: parse_docstrings,
    SOFT_BUDGET: parse_soft_budget,
    WRITER_USAGE: parse_writer_usage,
    LIBRARY_MODE: parse_memory_library_mode,
    CHECKER_VISIBLE: parse_checker_visible,
    QA_FIXTURES: parse_qa_fixtures,
    QA_MUTATION: parse_qa_mutation,
    QA_MIN_KILL: parse_qa_min_kill,
    QA_DETERMINISM: parse_qa_determinism,
    QA_REPLAY: parse_qa_replay,
    QA_FIXTURE_SIZE: parse_qa_fixture_size,
    LIBRARY_E: parse_library_e,
    LIBRARY_MAX_READS: parse_library_max_reads,
    LIBRARY_WAIT_SLOT: parse_library_wait_slot,
    LIBRARY_FORK: parse_library_fork,
    LIBRARY_FORK_BASE_URL: parse_library_fork_base_url,
    LIBRARY_FORK_TOKEN_FD: parse_library_fork_token_fd,
    LIBRARY_ANALYSTS: parse_library_analysts,
    LIBRARY_HELPER_MODEL: parse_library_helper_model,
    LIBRARY_HELPER_BASE_URL: parse_library_helper_base_url,
    LIBRARY_HELPER_TOKEN_FD: parse_library_helper_token_fd,
    LIBRARY_POOL: parse_library_pool,
    LIBRARY_EDIT_TOOLS: parse_library_edit_tools,
    LIBRARY_COMPACT: parse_library_compact,
    LIBRARY_ACTOR_INDEX: parse_library_actor_index,
    LIBRARY_COVERAGE: parse_library_coverage,
    LIBRARY_WRITER_USD_PER_TOKEN: parse_library_writer_usd_per_token,
    WRITER_PRICES: parse_writer_prices,
    LIBRARY_PASS_WALL_S: parse_library_pass_wall_s,
    WRITER_JOURNAL: parse_writer_journal,
}


#: The counterparts ``UNIFY_MEMORY_DIALOGUE`` may name: the channel key of the recorded dialogue
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

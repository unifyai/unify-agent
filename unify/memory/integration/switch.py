"""Memory's settings: the one memory switch and its spend cap, and the deployment configuration memory reads.

The behaviour settings: ``UNIFY_MEMORY`` and ``UNIFY_MEMORY_CAP_USD``
======================================================================

``UNIFY_MEMORY`` chooses what memory does, and ``UNIFY_MEMORY_CAP_USD`` how much it may spend. They are the only
memory behaviour settings; every other setting memory reads is deployment configuration (below). ``UNIFY_MEMORY``
takes exactly two values:

``on`` (the default; also when unset or empty)
    Memory between episodes. Each episode is recorded in the memory home (``<UNIFY_HOME>/memory``). At its end the
    actor's own conversation is forked once, and the fork stages a capture of what the episode did. The attention
    ledger hands what needs reading to the writer, a detached service per memory home that keeps the library. The
    next episode is served from what the library holds when it starts.

``off``
    No memory: no memory home, no writer, no fork and no memory tools. This is the no-memory baseline.

Any other value is refused when the settings are read, and the error lists the two values. The switch is read in
one place, :func:`memory_mode`; every other reader asks :func:`memory_on`.

``UNIFY_MEMORY_CAP_USD`` is memory's spend cap for one run (one ``unify act``, with the writer service it ran), in
USD as a plain decimal string (default :data:`MEMORY_CAP_DEFAULT`). It covers the writer's and its helpers' calls, as
recorded in the home's ``spend.jsonl``, counted from the controller's first memory request and across any restart of
the service within the run; a lane relay holds the episode-end forks' calls to
the same cap. It fails closed:

- a pass starts only when the room left under the cap holds one call at its worst case;
- an unpriced call is booked at its worst case;
- a value that is not a plain decimal above zero refuses the settings.

When the cap is reached, the library stops changing for the rest of the run and the actor is still served from it.

Deployment configuration (not behaviour switches)
=================================================

Each setting below is deployment configuration, not a behaviour switch: a launcher or the host sets it to say which
model runs, where its calls go, what they cost and how texts are embedded, and nothing sets one to choose what memory
does. Where a missing value leaves a part of memory out, the entry says so.

* ``UNIFY_MEMORY_WRITER_MODEL``: the model the writer, its helpers and the kinship judge run on. Empty means the
  actor's ``UNIFY_MODEL``. The model must be one the pinned price catalogue bounds (:mod:`..call_bounds`), because
  the writer books every call at its worst case until it is priced; otherwise the writer does not start.
* ``UNIFY_MODEL`` and ``UNIFY_REASONING_EFFORT`` (the actor's model and effort, :mod:`unify.settings`): memory reaches
  them as the writer's model when ``UNIFY_MEMORY_WRITER_MODEL`` is empty, as the librarian's model and effort
  (``librarian.actor_turn``), as the effort of the writer, its helpers and the kinship judge, and as the episode-end
  fork's model (it continues the actor's own conversation).
* Writer lane: ``UNIFY_MEMORY_WRITER_BASE_URL`` and ``UNIFY_MEMORY_WRITER_TOKEN_FD``, the lane's journal
  ``UNIFY_MEMORY_WRITER_JOURNAL`` (where the relay records each call) and its pinned prices
  ``UNIFY_MEMORY_WRITER_PRICES_USD_PER_MILLION`` (which price a writer call the relay reports no charge for). Without
  the lane, the writer's calls go on the controller's own provider route; without prices, such a call stays unpriced
  and is booked at its worst case.
* Fork lane: ``UNIFY_MEMORY_LIBRARY_FORK_BASE_URL`` and ``UNIFY_MEMORY_LIBRARY_FORK_TOKEN_FD``, the route of the
  episode-end forks. Without it no fork runs (``refused: no fork lane``), so no capture is made.
* Helper lane: ``UNIFY_MEMORY_LIBRARY_HELPER_BASE_URL`` and ``UNIFY_MEMORY_LIBRARY_HELPER_TOKEN_FD``, the route of
  the writer's helpers. Without it the writer is not offered ``delegate``, and its brief and tools leave helpers out.
* Embeddings (:mod:`unify.common.embeddings`): ``UNIFY_EMBED_URL`` with ``UNIFY_EMBED_TOKEN_FD`` (the embedding lane,
  :mod:`unify.lanes.embed_token`), or ``UNIFY_LOCAL_EMBEDDINGS`` (a small model in process). They say how the
  library's "When to use" texts are embedded for the librarian's ``tools.search`` and the writer's ``tools.find``.
  Without either, a text not yet embedded is refused and those tools list the library's rows unranked, saying why.
* The ``similar`` tool's index (``PassConfig.similar_index``, :mod:`..similar`): not a setting; the writer service
  always points it at the home's index folder.

A lane is a loopback relay that holds the provider key, with its own token and spend cap. A launcher hands each lane
to the controller as a base URL plus an inherited descriptor holding the lane's token. The descriptor is read once,
closed and removed from the environment, and the token is registered for redaction. No token ever reaches an actor
box. Each lane's base URL and descriptor are set together or not at all. Every refusal names the rule broken, never a
value, and is sticky for the life of the process.

Numbers
=======

The numbers memory uses are module constants, each beside the code it bounds (this module's are below). Money stays a
decimal string as written, never a float, and exponent forms are refused, so a value is read the same way by every
consumer.
"""

from __future__ import annotations

import json
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

# --- the one switch ---------------------------------------------------------------------------------------------

SWITCH = "UNIFY_MEMORY"
OFF = "off"
ON = "on"
#: The values :data:`SWITCH` takes, in the order the refusal lists them.
MEMORY_MODES = (OFF, ON)
#: An unset or empty ``UNIFY_MEMORY``: memory between episodes.
MEMORY_DEFAULT = ON


def _stripped(v: Any) -> str:
    return str(v if v is not None else "").strip()


def parse_memory(v: Any) -> str:
    """``UNIFY_MEMORY``: ``off`` or ``on``; unset or empty is ``on``. Anything else is refused, and the refusal lists
    the two values."""
    value = _stripped(v).lower() or MEMORY_DEFAULT
    if value not in MEMORY_MODES:
        raise ValueError(
            f"{SWITCH} must be one of {', '.join(MEMORY_MODES)} (empty means {MEMORY_DEFAULT}), not {v!r}"[:300],
        )
    return value


def memory_mode(settings: Any) -> str:
    """The run's memory mode: ``off`` or ``on`` (the one read of ``UNIFY_MEMORY``)."""
    return parse_memory(getattr(settings, SWITCH, ""))


def memory_on(settings: Any) -> bool:
    """Whether the run has memory (``on``)."""
    return memory_mode(settings) != OFF


# --- the writer's model and memory's spend cap ---------------------------------------------------------------------------

WRITER_MODEL = "UNIFY_MEMORY_WRITER_MODEL"
MEMORY_CAP = "UNIFY_MEMORY_CAP_USD"
#: Memory's spend cap for one run, in USD, when ``UNIFY_MEMORY_CAP_USD`` is empty: the only USD bound on the writer
#: and its helpers (a pass has no cap of its own).
MEMORY_CAP_DEFAULT = "20.00"

#: A plain decimal: digits, optionally a point and more digits. No sign, exponent, separator or name.
_PLAIN_DECIMAL = re.compile(r"[0-9]+(?:\.[0-9]+)?")
_DIGITS = re.compile(r"[0-9]+")


def parse_writer_model(v: Any) -> str:
    """``UNIFY_MEMORY_WRITER_MODEL``: empty (the actor's ``UNIFY_MODEL``) or a model id without spaces."""
    value = _stripped(v)
    if any(c.isspace() for c in value) or len(value) > 200:
        raise ValueError(f"{WRITER_MODEL} must be empty or a model id without spaces")
    return value


def writer_model(settings: Any) -> str:
    """The model the writer and its helpers run on: :data:`WRITER_MODEL`, else the actor's ``UNIFY_MODEL``."""
    own = parse_writer_model(getattr(settings, WRITER_MODEL, "") or "")
    return own or _stripped(getattr(settings, "UNIFY_MODEL", "") or "")


def _plain_decimal(name: str, text: str, v: Any, *, positive: bool) -> str:
    if not _PLAIN_DECIMAL.fullmatch(text):
        raise ValueError(
            f"{name} must be a plain decimal USD amount (digits and at most one point, "
            f"no sign or exponent), not {v!r}"[:300],
        )
    if positive and Decimal(text) <= 0:
        raise ValueError(f"{name} must be above zero, not {v!r}"[:300])
    return text


def parse_memory_cap(v: Any) -> str:
    """``UNIFY_MEMORY_CAP_USD``: a plain decimal string above 0; empty means :data:`MEMORY_CAP_DEFAULT`."""
    return _plain_decimal(MEMORY_CAP, _stripped(v) or MEMORY_CAP_DEFAULT, v, positive=True)


def memory_cap_usd(settings: Any) -> Decimal:
    """Memory's spend cap for the memory home."""
    return Decimal(parse_memory_cap(getattr(settings, MEMORY_CAP, "") or ""))


# --- fixed values -----------------------------------------------------------------------------------------------

#: One pass's wall-clock bound, in seconds.
PASS_WALL_S = 2700


# --- the writer's lane --------------------------------------------------------------------------------------------

WRITER_BASE_URL = "UNIFY_MEMORY_WRITER_BASE_URL"
WRITER_TOKEN_FD = "UNIFY_MEMORY_WRITER_TOKEN_FD"
#: The redaction name of the writer lane token's value (never a setting).
WRITER_TOKEN = "UNIFY_MEMORY_WRITER_TOKEN"
WRITER_JOURNAL = "UNIFY_MEMORY_WRITER_JOURNAL"
#: The writer model's prices, USD per million tokens, as the launcher pins them: a JSON object with ``input`` and
#: ``output`` and optionally ``cache_write``. Empty: a writer call with no charge stays unpriced.
WRITER_PRICES = "UNIFY_MEMORY_WRITER_PRICES_USD_PER_MILLION"
WRITER_PRICE_KEYS = ("input", "output")
WRITER_OPTIONAL_PRICE_KEYS = ("cache_write",)


def writer_base_url_setting(v: Any) -> str:
    """``UNIFY_MEMORY_WRITER_BASE_URL`` as settings hold it: stripped, never refused (see :func:`writer_route`)."""
    return _stripped(v)


def writer_token_fd_setting(v: Any) -> str:
    """``UNIFY_MEMORY_WRITER_TOKEN_FD`` as settings hold it: stripped, never refused (a settings error would print its
    input, a token set there by mistake included); :func:`settle_writer_route_env` parses it with
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
    """The writer's lane is set but not in effect (malformed, half set, or not what the settings hold): no pass
    starts. Its text names settings and rules, never a value.
    """


#: The one host plain http may name: the launcher's loopback relay (the token travels in a header, in clear). Exactly
#: this IPv4 literal, compared as text: no other name, address or DNS lookup; always with an explicit port.
HTTP_HOST = "127.0.0.1"


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


#: At least 16 RFC 3986 unreserved characters (``secrets.token_urlsafe`` output fits): it goes in a header, and no
#: percent-encoding, JSON or other escaping form differs from it, so redaction by value always finds it.
_MIN_TOKEN_CHARS = 16
_BEARER_TOKEN = re.compile(r"[A-Za-z0-9._~-]{%d,}" % _MIN_TOKEN_CHARS)

#: Why every pass is refused, once :func:`settle_writer_route_env` found the environment and the settings disagreeing
#: about the writer's lane, or its descriptor unreadable; sticky for the life of the process.
_ENV_REFUSAL: str | None = None

#: The most bytes a lane token may take on its descriptor, and how long reading it may take.
TOKEN_FD_MAX_BYTES = 4096
TOKEN_FD_TIMEOUT_S = 5.0

#: The writer's token as read from ``UNIFY_MEMORY_WRITER_TOKEN_FD`` (once per process; never in ``os.environ``).
_FD_TOKEN: SecretStr | None = None
_FD_TRIED = False
#: The descriptor number the first settle read (settings' value), kept once it has left ``os.environ``.
_FD_HELD = ""


def writer_route(base_url: Any, token: Any) -> tuple[str, SecretStr] | None:
    """The writer's lane, ``(base_url, token)``, or ``None`` when neither is set (the controller's own route).

    Exactly one set is refused (fail closed: no pass starts, so no call is made), as is either value in a wrong form.
    No error quotes a value.
    """
    if _ENV_REFUSAL is not None:
        raise WriterRouteRefused(_ENV_REFUSAL)
    raw = token.get_secret_value() if isinstance(token, SecretStr) else _stripped(token)
    register_secret(WRITER_TOKEN, raw)  # kept redacted by value even if refused below
    try:
        base = parse_writer_base_url(base_url)
    except ValueError as exc:  # the parser's text names the rule, never the value
        raise WriterRouteRefused(str(exc)) from None
    if raw and not _BEARER_TOKEN.fullmatch(raw):
        raise WriterRouteRefused(
            f"the writer lane's token must be a bearer token: at least {_MIN_TOKEN_CHARS} characters from "
            "A-Z a-z 0-9 . _ ~ -",
        )
    if not base and not raw:
        return None
    if not base or not raw:
        missing = WRITER_BASE_URL if not base else WRITER_TOKEN_FD
        raise WriterRouteRefused(
            f"{WRITER_BASE_URL} and {WRITER_TOKEN_FD} are set together or not at all; "
            f"{missing} is empty, so no pass starts",
        )
    return base, SecretStr(raw)


def _read_token_fd(
    fd: int,
    *,
    fd_name: str = WRITER_TOKEN_FD,
    secret_name: str = WRITER_TOKEN,
) -> str:
    """A lane token from inherited descriptor *fd*, which is then closed; refusals name the rule, never a value."""
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


def writer_token(settings: Any) -> SecretStr:
    """The writer lane's token as this process holds it (read once from its descriptor), or an empty one: without a
    descriptor, or when it could not be read (:func:`writer_route` then refuses)."""
    if _stripped(getattr(settings, WRITER_TOKEN_FD, "")):
        return _FD_TOKEN if _FD_TOKEN is not None else SecretStr("")
    return SecretStr("")


def settle_writer_route_env(
    environ: MutableMapping[str, str],
    settings: Any,
) -> str | None:
    """Read the writer lane's token once, keep it out of *environ*, and refuse every pass if *environ* and *settings*
    disagree.

    Called once the settings are built (``unify/settings.py``), again after the CLI loads ``.env`` and whenever a pass
    is about to start. The first call with ``UNIFY_MEMORY_WRITER_TOKEN_FD`` set reads the token from that descriptor
    and closes it (:func:`_read_token_fd`); a failed read refuses every pass. Every case variant of
    ``UNIFY_MEMORY_WRITER_TOKEN_FD`` then leaves *environ* (the number is held here for the stray check, and no later
    call reads a descriptor again), so no child spawned afterwards sees the name. A base URL or descriptor found in
    *environ* that the settings do not hold (one that arrived after they were built, from ``.env`` say, or two case
    variants with different values) would otherwise silently leave the writer on another route, so from then on
    :func:`writer_route` refuses every pass with the returned message. The message names settings, never values.
    Returns ``None`` when they agree.
    """
    global _ENV_REFUSAL, _FD_TOKEN, _FD_TRIED, _FD_HELD
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
            fd_refusal = f"{exc}; no pass starts"
    stray: list[str] = []
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
            "(from .env, or in another letter case), so the writer's lane is not what the settings hold; no "
            "pass starts. Set both in the controller's process environment, not .env"
        )
    return _ENV_REFUSAL


def parse_writer_journal(v: Any) -> str:
    """Empty, or the absolute path of the writer lane's journal (JSON lines with ``request_attempt_id``,
    ``generation_id`` and ``account_charge``), read only to price the writer's calls after a pass.
    """
    text = _stripped(v)
    if text and not text.startswith("/"):
        raise ValueError(f"{WRITER_JOURNAL} must be empty or an absolute path")
    return text


def writer_journal(settings: Any) -> str | None:
    return parse_writer_journal(getattr(settings, WRITER_JOURNAL, "")) or None


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
    if not isinstance(raw, dict) or not set(WRITER_PRICE_KEYS) <= set(raw) or set(raw) - allowed:
        raise ValueError(refusal)
    out = {}
    for key, value in raw.items():
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            raise ValueError(refusal)
        out[key] = _plain_decimal(f"{WRITER_PRICES} {key}", str(value).strip(), v, positive=False)
    return json.dumps(out, sort_keys=True)


def writer_prices(settings: Any) -> dict[str, Decimal] | None:
    """The pinned writer prices per million tokens, or None when none are set."""
    text = parse_writer_prices(getattr(settings, WRITER_PRICES, ""))
    return {k: Decimal(v) for k, v in json.loads(text).items()} if text else None


# --- the fork's and the helpers' lanes ----------------------------------------------------------------------------


class _Lane:
    """A lane whose token comes from an inherited descriptor only: its base URL must be
    ``http://127.0.0.1:<port>/<path>/v1`` exactly. The token is read once (:meth:`settle`) and stays in this process;
    both names leave the environment so no child sees them, and the base URL and descriptor are set together or not
    at all. Any refusal is sticky."""

    def __init__(self, base_name: str, fd_name: str, secret_name: str, path: str, absent: str) -> None:
        self.base_name, self.fd_name, self.secret_name = base_name, fd_name, secret_name
        self.path, self.absent = path, absent
        self.form = re.compile(r"http://127\.0\.0\.1:([0-9]{1,5})/%s/v1" % re.escape(path))
        self.token: str | None = None
        self.base = ""
        self.tried = False
        self.refusal: str | None = None

    def parse_base_url(self, v: Any) -> str:
        s = _stripped(v)
        if not s:
            return ""
        m = self.form.fullmatch(s)
        if m is None or not 1 <= int(m.group(1)) <= 65535:
            raise ValueError(f"{self.base_name} must be http://127.0.0.1:<port>/{self.path}/v1 or empty")
        return s

    def parse_token_fd(self, v: Any) -> str:
        s = _stripped(v)
        if not s:
            return ""
        if not s.isdigit() or int(s) <= 2:
            raise ValueError(f"{self.fd_name} must be a descriptor number above 2 or empty")
        return s

    def settle(self, environ: MutableMapping[str, str], settings: Any) -> str | None:
        try:
            base = self.parse_base_url(getattr(settings, self.base_name, ""))
            fd = self.parse_token_fd(getattr(settings, self.fd_name, ""))
        except ValueError as exc:
            base, fd = "", ""
            self.refusal = self.refusal or str(exc)
        if fd and not self.tried:
            self.tried = True
            try:
                self.token = _read_token_fd(int(fd), fd_name=self.fd_name, secret_name=self.secret_name)
            except WriterRouteRefused as exc:
                self.refusal = self.refusal or str(exc)
        if base:
            self.base = base
        if bool(base) != bool(fd) and self.refusal is None:
            self.refusal = f"{self.base_name} and {self.fd_name} are set together or not at all"
        for name in list(environ):
            if name.upper() in (self.base_name, self.fd_name):
                environ.pop(name, None)
        return self.refusal

    def route(self) -> tuple[tuple[str, str] | None, str | None]:
        if self.refusal is not None:
            return None, self.refusal
        if not self.base or not self.token:
            return None, self.absent
        return (self.base, self.token), None


LIBRARY_FORK_BASE_URL = "UNIFY_MEMORY_LIBRARY_FORK_BASE_URL"
LIBRARY_FORK_TOKEN_FD = "UNIFY_MEMORY_LIBRARY_FORK_TOKEN_FD"
LIBRARY_FORK_TOKEN = "UNIFY_MEMORY_LIBRARY_FORK_TOKEN"  # the redaction name of the token's value (never a setting)
LIBRARY_HELPER_BASE_URL = "UNIFY_MEMORY_LIBRARY_HELPER_BASE_URL"
LIBRARY_HELPER_TOKEN_FD = "UNIFY_MEMORY_LIBRARY_HELPER_TOKEN_FD"
LIBRARY_HELPER_TOKEN = "UNIFY_MEMORY_LIBRARY_HELPER_TOKEN"  # the redaction name of the token's value (never a setting)

_FORK = _Lane(LIBRARY_FORK_BASE_URL, LIBRARY_FORK_TOKEN_FD, LIBRARY_FORK_TOKEN, "fork", "no fork lane")
_HELPER = _Lane(LIBRARY_HELPER_BASE_URL, LIBRARY_HELPER_TOKEN_FD, LIBRARY_HELPER_TOKEN, "helper", "no helper lane")
parse_library_fork_base_url = _FORK.parse_base_url
parse_library_fork_token_fd = _FORK.parse_token_fd
parse_library_helper_base_url = _HELPER.parse_base_url
parse_library_helper_token_fd = _HELPER.parse_token_fd


def settle_fork_route_env(environ: MutableMapping[str, str], settings: Any) -> str | None:
    """The fork lane's settle (:class:`_Lane`): returns why the fork lane is refused (sticky), or None."""
    return _FORK.settle(environ, settings)


def fork_lane_route() -> tuple[tuple[str, str] | None, str | None]:
    """``((chat completions URL, token), None)`` on the fork lane, or ``(None, why)``: a refusal from
    :func:`settle_fork_route_env`, or ``no fork lane``. Never any other route (no fallback to the actor's)."""
    pair, why = _FORK.route()
    if pair is None:
        return None, why
    return (pair[0] + "/chat/completions", pair[1]), None


def settle_helper_route_env(environ: MutableMapping[str, str], settings: Any) -> str | None:
    """The helper lane's settle (:class:`_Lane`): returns why the helper lane is refused (sticky), or None."""
    return _HELPER.settle(environ, settings)


def helper_lane_route() -> tuple[tuple[str, SecretStr] | None, str | None]:
    """``((base URL, token), None)`` on the helper lane, or ``(None, why)``: a refusal from
    :func:`settle_helper_route_env`, or ``no helper lane``. Never another lane's route (no fallback to the writer's or
    the actor's)."""
    pair, why = _HELPER.route()
    if pair is None:
        return None, why
    return (pair[0], SecretStr(pair[1])), None


# --- settings validation ------------------------------------------------------------------------------------------

#: The validator of each memory setting (``unify/settings.py``).
PARSERS = {
    SWITCH: parse_memory,
    WRITER_MODEL: parse_writer_model,
    MEMORY_CAP: parse_memory_cap,
    # normalised only; checked when a pass starts
    WRITER_BASE_URL: writer_base_url_setting,
    WRITER_TOKEN_FD: writer_token_fd_setting,
    WRITER_JOURNAL: parse_writer_journal,
    WRITER_PRICES: parse_writer_prices,
    LIBRARY_FORK_BASE_URL: parse_library_fork_base_url,
    LIBRARY_FORK_TOKEN_FD: parse_library_fork_token_fd,
    LIBRARY_HELPER_BASE_URL: parse_library_helper_base_url,
    LIBRARY_HELPER_TOKEN_FD: parse_library_helper_token_fd,
}


def writer_refusal(settings: Any) -> str | None:
    """Why the writer may not start with these settings (the settings named, never a value), or None: its model must
    be one the pinned price catalogue bounds, since every call is booked at its worst case until priced."""
    from ..call_bounds import NoBound, call_worst_case_usd

    model = writer_model(settings)
    if not model:
        return f"{WRITER_MODEL} and UNIFY_MODEL are both empty, so the writer has no model; no pass starts"
    try:
        call_worst_case_usd(model)
    except NoBound:
        name = WRITER_MODEL if _stripped(getattr(settings, WRITER_MODEL, "") or "") else "UNIFY_MODEL"
        return (
            f"the writer's model ({name}) is not one the pinned price catalogue bounds, and the writer books every "
            "call at its model's worst case; no pass starts"
        )
    return None

"""The local lane relay: one process that holds the provider key and serves the memory lanes on loopback.

Run by path with the standard library only (``<python> -I -S <this file>``): no site packages, no ``PYTHON*``
variables, not the script's folder on the path, so nothing of the harness or a provider SDK is ever imported here.
:mod:`unify.lanes.relay` starts it; nothing imports it.

**Start-up.** The first line on stdin is the configuration, one JSON object (:func:`read_config`): the provider key,
the upstream base URL, each lane's token and models, the spend cap, every model's worst case per call and any pinned
prices, and the journal folder. Then:

* the key lives in this process's memory only. It never reaches argv, the environment, a file, a log line or a reply.
  The process cannot be dumped or traced by another process of the same user (``PR_SET_DUMPABLE`` 0), and writes no
  core file;
* the relay listens on ``127.0.0.1`` on a port the kernel picks, prints one ready line on stdout
  (``{"ready": ..., "port": ..., "pid": ...}``), and then points stdout at ``/dev/null``. A start-up failure prints
  ``{"failed": <fixed reason>}`` and exits 3;
* **stdin stays open as the lifeline.** When it reaches its end (the starting process closed it, or died, which closes
  it too), the relay stops. SIGTERM stops it the same way. SIGINT is ignored: the relay has its own session, and its
  owner decides when it ends.

**A call.** Only ``POST /<lane>/v1/chat/completions`` is served, for ``lane`` in ``writer``, ``helper`` and ``fork``,
with ``Host: 127.0.0.1:<port>``:

1. exactly one ``Authorization: Bearer <token>`` header equal (constant time) to that lane's token. Another lane's
   token is refused too (``other_lane_token``): each token is valid on its own lane only;
2. a JSON object body of at most :data:`MAX_REQUEST_BYTES`, without duplicate keys, OpenRouter's fallback ``models``
   field, its ``plugins``, or ``n`` other than 1 or ``best_of`` (each costs beyond a model's worst case per call, which
   bounds one completion), whose ``model`` is exactly one of the lane's models; anything else is ``off_model_refused``
   before anything goes upstream;
3. room under the spend cap (below) and a free slot among :data:`MAX_IN_FLIGHT` calls;
4. the ``X-Unify-*`` headers are taken off (journaled as ``unify_meta`` when well formed) and the call goes upstream
   with the provider key, ``usage: {"include": true}`` added when the body does not ask for usage, so the provider
   reports the charge. Connection failures before anything was sent are tried again (:data:`CONNECT_ATTEMPTS`); a call
   that may have been sent never is;
5. the reply is relayed with every secret (the key, each lane token) replaced by value: a JSON reply whole, a
   server-sent event stream line by line as it arrives. A stream's redaction cannot catch a secret a provider split
   across two events. That is accepted, because no secret is ever in a request body (the key and the token travel only
   in headers this relay writes or reads, and of a caller's headers only ``Accept`` is passed on), so a provider has
   none to echo there.

Refusals are ``{"error": {"code", "type", "message", "reason"}}`` with fixed texts that never echo a header, a token or
the key (``unify/memory/writer_pass.py`` reads ``reason``): ``<lane>_route_refused`` (path, method, host, token),
``off_model_refused``, ``<lane>_budget_exhausted`` with reason ``<lane>_cap_reached`` (the cap), and
``<lane>_route_busy`` with reason ``<lane>_in_flight`` (no slot within :data:`BUSY_WAIT_S`). Each refusal is a row of
``refusals.jsonl`` (lane, reason, the model id asked for; never content).

**The journals** (``<lane>.jsonl`` in the journal folder, append-only, mode 0600; one per lane, so a reader that
selects rows by model never books another lane's calls). Each call has a ``request_started`` row before it goes
upstream and one terminal row (``completed``, ``failed`` or ``cancelled``) under the same ``request_attempt_id``,
written before the caller can have the whole reply (so its next call is admitted against what this one cost; a reply
that then does not reach it adds one more row, ``cancelled``, with the same charge), in the shape
``unify/memory/reconcile.py`` and ``unify/memory/metering.py`` read: ``account_charge`` (the provider's charge, a
decimal string, or null), ``tokens``, ``generation_id``, ``status_code`` and a fixed ``error_type``, with what the run
booked for it (``booked_usd``) and how (``charge_class``, :data:`CHARGE_CLASSES`). A call that failed before anything
was sent, and one the provider refused before generating (a :data:`REJECTED_STATUSES` reply without usage), have the
charge ``"0"``.

**Spend.** One cap over all lanes, for this run: this relay, which lives as long as one ``unify act`` and the writer
service it ran. Each call is booked in this process's memory from its admission, at its model's worst case, to its
end: then at its charge; else at its tokens times the model's pinned prices; else still at its worst case. A call is
admitted only while the booked amount plus its own model's worst case stays within the cap, and each terminal row
records what was booked (``booked_usd``). Money is a decimal string everywhere.
"""

from __future__ import annotations

import ctypes
import hmac
import http.client
import json
import os
import re
import resource
import select
import signal
import socket
import ssl
import stat
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

FORMAT = "unify-lane-relay/1"
LANES = ("writer", "helper", "fork")
LOOPBACK = "127.0.0.1"
#: The client path of each lane's one route, and the upstream path under the upstream base URL.
ROUTE = re.compile(r"/(writer|helper|fork)/v1/chat/completions")
UPSTREAM_PATH = "/chat/completions"

#: A lane token: what ``secrets.token_urlsafe(32)`` gives (the harness's own rule is 16 or more of these characters).
TOKEN_FORM = re.compile(r"[A-Za-z0-9._~-]{32,512}")
#: A model id as OpenRouter names it (``vendor/model``, ``vendor/model:variant``, ``~vendor/model-latest``).
MODEL_FORM = re.compile(r"[A-Za-z0-9~][A-Za-z0-9._:/@+~-]{0,199}")
PLAIN_DECIMAL = re.compile(r"[0-9]{1,12}(\.[0-9]{1,18})?")
PRICE_KEYS = ("input", "output", "cache_write")

#: Bounds: one request body and one reply, one line of an event stream, the configuration line.
MAX_REQUEST_BYTES = 64 << 20
MAX_RESPONSE_BYTES = 64 << 20
MAX_EVENT_LINE = 16 << 20
MAX_CONFIG_BYTES = 4 << 20
#: The piece a refused request's body is read and dropped in.
DISCARD_CHUNK = 64 << 10
#: Calls in flight upstream at once, over all lanes, and how long a call waits for a slot before ``*_in_flight``.
MAX_IN_FLIGHT = 8
BUSY_WAIT_S = 30.0
#: Connections open at once (each has a thread, and at most :data:`CLIENT_TIMEOUT_S` to send its request); one more is
#: closed at once, unread.
MAX_CONNECTIONS = 64
#: Seconds: reading the configuration; reading a request from a client; connecting upstream (each attempt); each read
#: from upstream (the fork worker's own timeout); one call's whole time; the drain at a stop.
CONFIG_TIMEOUT_S = 10.0
CLIENT_TIMEOUT_S = 60.0
CONNECT_TIMEOUT_S = 30.0
READ_TIMEOUT_S = 900.0
CALL_WALL_S = 1800.0
DRAIN_S = 20.0
#: Connection attempts before anything was sent, and the wait before the next one (doubled each time).
CONNECT_ATTEMPTS = 3
CONNECT_BACKOFF_S = 1.0

#: Statuses at which a JSON reply without usage is the provider refusing the call before it generated anything: such a
#: call is booked at nothing (``rejected``). A timeout (408), a server error (5xx), a caller gone, and any streamed or
#: partial output keep the worst case: the call may have been billed.
REJECTED_STATUSES = frozenset({400, 401, 403, 404, 409, 422, 429})
#: How a call's booking was decided, as its terminal row's ``charge_class``: the provider's charge, its tokens at the
#: pinned prices, the worst case, refused before generating, or never sent.
CHARGE_CLASSES = ("charge", "estimate", "worst_case", "rejected", "not_sent")
REDACTED = b"[redacted]"
META_PREFIX = "x-unify-"
#: ``X-Unify-*`` header -> journal field, and the form each value must have to be journaled (else it is dropped and
#: flagged by name; a value is never echoed).
META_FIELDS = {
    "x-unify-session": "session",
    "x-unify-parent": "parent",
    "x-unify-request": "request",
    "x-unify-call-kind": "call_kind",
    "x-unify-msg-count": "msg_count",
    "x-unify-pass-id": "pass_id",
    "x-unify-entries": "entries",
    "x-unify-episode": "episode",
    "x-unify-segment": "segment",
}
META_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}")
META_COUNT = re.compile(r"[0-9]{1,9}")
META_PASS = re.compile(r"[0-9a-f]{16}")
META_ENTRIES = re.compile(r"[0-9a-f]{16}(?:\.[0-9a-f]{16}){0,63}")


class ConfigError(ValueError):
    """A configuration this relay refuses; its text is fixed and never quotes a value."""


def _now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _money(d: Decimal) -> str:
    return "0" if d.is_zero() else format(d, "f")


def _decimal(value, *, positive: bool, what: str) -> Decimal:
    if not isinstance(value, str) or not PLAIN_DECIMAL.fullmatch(value):
        raise ConfigError(f"{what} must be a plain decimal string")
    d = Decimal(value)
    if positive and d <= 0:
        raise ConfigError(f"{what} must be above 0")
    return d


def _charge(value) -> Decimal | None:
    """A provider-reported charge: a finite, non-negative number or decimal string; else None."""
    if value is None or isinstance(value, bool):
        return None
    try:
        d = Decimal(str(value).strip())
    except (InvalidOperation, ValueError):
        return None
    return d if d.is_finite() and d >= 0 else None


def _count(value) -> int | None:
    return (
        value
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0
        else None
    )


# --- the configuration ------------------------------------------------------------------------------------------------


def read_config(fd: int = 0, timeout_s: float = CONFIG_TIMEOUT_S) -> dict:
    """The first line on *fd* (bounded in size and time), parsed; the descriptor stays open (the lifeline)."""
    data = b""
    deadline = time.monotonic() + timeout_s
    while b"\n" not in data:
        left = deadline - time.monotonic()
        if left <= 0 or not select.select([fd], [], [], left)[0]:
            raise ConfigError("no configuration line in time")
        chunk = os.read(fd, 65536)
        if not chunk:
            raise ConfigError("the configuration line ended early")
        data += chunk
        if len(data) > MAX_CONFIG_BYTES:
            raise ConfigError("the configuration line is too long")
    line = data.split(b"\n", 1)[0]
    try:
        doc = json.loads(line.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        raise ConfigError("the configuration line is not JSON") from None
    if not isinstance(doc, dict):
        raise ConfigError("the configuration is not an object")
    return doc


def _upstream(value) -> tuple[str, str, int, str]:
    """``(scheme, host, port, base path)``: https to any host, or plain http to 127.0.0.1 with a port (a local
    stand-in); never a user, password, query or fragment."""
    if not isinstance(value, str) or any(c.isspace() for c in value):
        raise ConfigError("the upstream must be a URL without spaces")
    try:
        parts = urlsplit(value)
        port = parts.port
    except ValueError:
        raise ConfigError("the upstream does not parse") from None
    if (
        parts.username
        or parts.password
        or parts.query
        or parts.fragment
        or not parts.hostname
    ):
        raise ConfigError(
            "the upstream must have a host and no user, password, query or fragment"
        )
    if parts.scheme == "https":
        return "https", parts.hostname, port or 443, parts.path.rstrip("/")
    if parts.scheme == "http" and parts.hostname == LOOPBACK and port:
        return "http", parts.hostname, port, parts.path.rstrip("/")
    raise ConfigError(
        "the upstream must be https, or plain http to 127.0.0.1 with a port"
    )


class Config:
    """The checked configuration. Refusals name the rule, never a value."""

    def __init__(self, doc: dict):
        if doc.get("format") != FORMAT:
            raise ConfigError("the configuration's format differs")
        key = doc.get("key")
        if (
            not isinstance(key, str)
            or not 8 <= len(key) <= 4096
            or not key.isprintable()
            or any(c.isspace() for c in key)
        ):
            raise ConfigError("the provider key is missing or malformed")
        self.key = key
        self.scheme, self.host, self.port, self.base_path = _upstream(
            doc.get("upstream")
        )
        cafile = doc.get("cafile")
        if cafile is not None and not (
            isinstance(cafile, str) and os.path.isabs(cafile) and os.path.isfile(cafile)
        ):
            raise ConfigError("the CA file must be an absolute path to a file")
        self.cafile = cafile
        folder = doc.get("journal_dir")
        if not isinstance(folder, str) or not os.path.isabs(folder):
            raise ConfigError("the journal folder must be an absolute path")
        try:
            st = os.stat(folder, follow_symlinks=False)
        except OSError:
            raise ConfigError("the journal folder does not exist") from None
        if (
            not stat.S_ISDIR(st.st_mode)
            or st.st_uid != os.getuid()
            or st.st_mode & 0o022
        ):
            raise ConfigError(
                "the journal folder must be a folder of this user that no one else can write"
            )
        self.journal_dir = folder
        self.cap = _decimal(doc.get("cap_usd"), positive=True, what="the spend cap")
        worst = doc.get("worst_case_usd")
        if not isinstance(worst, dict) or not worst:
            raise ConfigError("the worst cases per call are missing")
        self.worst: dict[str, Decimal] = {}
        for model, value in worst.items():
            if not isinstance(model, str) or not MODEL_FORM.fullmatch(model):
                raise ConfigError("a worst case names a malformed model")
            self.worst[model] = _decimal(
                value, positive=True, what="a worst case per call"
            )
        prices = doc.get("prices") or {}
        if not isinstance(prices, dict):
            raise ConfigError("the prices must be an object")
        self.prices: dict[str, dict[str, Decimal]] = {}
        for model, table in prices.items():
            if (
                not isinstance(model, str)
                or not MODEL_FORM.fullmatch(model)
                or not isinstance(table, dict)
            ):
                raise ConfigError("a price names a malformed model")
            if not {"input", "output"} <= set(table) <= set(PRICE_KEYS):
                raise ConfigError(
                    "a model's prices are input and output, optionally cache_write"
                )
            self.prices[model] = {
                k: _decimal(v, positive=False, what="a price") for k, v in table.items()
            }
        lanes = doc.get("lanes")
        if not isinstance(lanes, dict) or not lanes or not set(lanes) <= set(LANES):
            raise ConfigError("the lanes must be some of writer, helper and fork")
        self.tokens: dict[str, bytes] = {}
        self.models: dict[str, frozenset[str]] = {}
        for lane, spec in lanes.items():
            if not isinstance(spec, dict) or set(spec) != {"token", "models"}:
                raise ConfigError("each lane has a token and its models")
            token, models = spec["token"], spec["models"]
            if not isinstance(token, str) or not TOKEN_FORM.fullmatch(token):
                raise ConfigError("a lane token is malformed")
            if (
                not isinstance(models, list)
                or not 1 <= len(models) <= 16
                or len(set(models)) != len(models)
            ):
                raise ConfigError("each lane lists 1 to 16 distinct models")
            for model in models:
                if not isinstance(model, str) or not MODEL_FORM.fullmatch(model):
                    raise ConfigError("a lane names a malformed model")
                if model not in self.worst:
                    raise ConfigError("every lane model needs a worst case per call")
            self.tokens[lane] = token.encode("ascii")
            self.models[lane] = frozenset(models)
        if (
            len(set(self.tokens.values())) != len(self.tokens)
            or key.encode() in self.tokens.values()
        ):
            raise ConfigError(
                "every lane token must be distinct, and none the provider key"
            )

    def secrets(self) -> list[bytes]:
        """Every value no reply may carry: the key and the lane tokens."""
        return [self.key.encode("utf-8"), *self.tokens.values()]


# --- the journals and the ledger --------------------------------------------------------------------------------------


def _open_append(path: str) -> int:
    fd = os.open(
        path,
        os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC,
        0o600,
    )
    st = os.fstat(fd)
    if not stat.S_ISREG(st.st_mode) or st.st_uid != os.getuid():
        os.close(fd)
        raise ConfigError("a journal is not a regular file of this user")
    os.fchmod(fd, 0o600)
    return fd


class Journal:
    """One lane's append-only JSONL journal: one ``write`` per row, so a row is never interleaved with another."""

    def __init__(self, path: str):
        self.path = path
        self.fd = _open_append(path)
        self.lock = threading.Lock()

    def write(self, row: dict) -> None:
        data = (
            json.dumps(row, sort_keys=True, separators=(",", ":"), default=str) + "\n"
        ).encode("utf-8")
        with self.lock:
            view = memoryview(data)
            while view:
                n = os.write(self.fd, view)
                view = view[n:]


class Ledger:
    """What this run's calls have booked against the cap (module docstring): one amount per call, in this process's
    memory, from the call's admission (its model's worst case) to its end (what it cost, :meth:`price`).
    """

    def __init__(self, config: Config):
        self.config = config
        self.booked: dict[str, Decimal] = {}
        self.total = Decimal(0)
        self.classes: dict[str, str] = {}
        self.lock = threading.Lock()

    def _book(self, rid: str, amount: Decimal) -> None:
        self.total += amount - self.booked.get(rid, Decimal(0))
        self.booked[rid] = amount

    def admit(self, rid: str, model: str, start) -> bool:
        """If the booked amount plus *model*'s worst case stays within the cap, call *start* (which writes the call's
        ``request_started`` row), book the call at that worst case and return True; else False.
        """
        with self.lock:
            worst = self.config.worst[model]
            if self.total + worst > self.config.cap:
                return False
            start()
            self._book(rid, worst)
            return True

    def price(self, model: str, charge: Decimal | None, tokens) -> tuple[Decimal, str]:
        """What an ended call books, and the class of that booking: its charge; else its tokens at the model's pinned
        prices; else its model's worst case."""
        if charge is not None:
            return charge, "charge"
        estimate = self._estimate(model, tokens)
        if estimate is not None:
            return estimate, "estimate"
        return self.config.worst[model], "worst_case"

    def _estimate(self, model: str, tokens) -> Decimal | None:
        """As ``unify/memory/reconcile.estimate``: input, output and cache-write tokens at the pinned prices."""
        prices = self.config.prices.get(model)
        if not prices or not isinstance(tokens, dict):
            return None
        if tokens.get("input") is None and tokens.get("output") is None:
            return None
        counts = {
            k: _count(tokens.get(k)) or 0 for k in ("input", "output", "cache_write")
        }
        usd = prices["input"] * counts["input"] + prices["output"] * counts["output"]
        if "cache_write" in prices:
            usd += prices["cache_write"] * counts["cache_write"]
        return usd / Decimal(1000000)

    def settle(self, rid: str, amount: Decimal, kind: str) -> None:
        with self.lock:
            self._book(rid, amount)
            self.classes[rid] = kind

    def by_class(self) -> str:
        """How many ended calls were booked in each class (for the relay's log)."""
        with self.lock:
            counts = {k: list(self.classes.values()).count(k) for k in CHARGE_CLASSES}
        return ", ".join(f"{k} {n}" for k, n in counts.items() if n) or "none"

    def state(self) -> dict:
        with self.lock:
            return {
                "booked_usd": _money(self.total),
                "cap_usd": _money(self.config.cap),
                "calls": len(self.booked),
            }


# --- request metadata, bodies and replies -----------------------------------------------------------------------------


def request_metadata(headers) -> dict | None:
    """The ``X-Unify-*`` headers' well-formed values as journal fields (or None when there are none); a malformed,
    repeated or unknown one is dropped and flagged by name only."""
    seen: dict[str, list[str]] = {}
    for name, value in headers.items():
        low = str(name).lower()
        if low.startswith(META_PREFIX):
            seen.setdefault(low, []).append(str(value))
    if not seen:
        return None
    meta: dict = {}
    flags: list[str] = []
    for low, values in sorted(seen.items()):
        field = META_FIELDS.get(low)
        if field is None:
            flags.append("unknown header")
            continue
        if len(values) != 1:
            flags.append(f"{field}: repeated")
            continue
        text = values[0].strip()
        if field in ("request", "msg_count", "segment"):
            ok = META_COUNT.fullmatch(text) is not None
            value = int(text) if ok else None
        elif field == "pass_id":
            ok, value = META_PASS.fullmatch(text) is not None, text
        elif field == "entries":
            ok, value = META_ENTRIES.fullmatch(text) is not None, text.split(".")
        else:
            ok, value = META_ID.fullmatch(text) is not None, text
        if ok:
            meta[field] = value
        else:
            flags.append(f"{field}: malformed")
    meta["flags"] = flags[:16]
    return meta


class _DuplicateKey(ValueError):
    pass


def _unique(pairs):
    out = {}
    for k, v in pairs:
        if k in out:
            raise _DuplicateKey
        out[k] = v
    return out


def check_body(
    body: bytes, allowed: frozenset[str]
) -> tuple[dict | None, str | None, str | None]:
    """``(parsed body, model, None)`` when the call may go upstream, else ``(None, model or None, reason)``."""
    try:
        parsed = json.loads(body, object_pairs_hook=_unique)
    except _DuplicateKey:
        return None, None, "duplicate_keys"
    except (ValueError, RecursionError):
        return None, None, "unparseable_body"
    if not isinstance(parsed, dict):
        return None, None, "not_an_object"
    model = parsed.get("model")
    if not isinstance(model, str):
        return None, None, "missing_model" if model is None else "model_not_a_string"
    shown = model if MODEL_FORM.fullmatch(model) else None
    if "models" in parsed:
        return None, shown, "fallback_models_field"
    if "plugins" in parsed or "web_search_options" in parsed:
        # OpenRouter's plugins (web search and the like) cost beyond a model's worst case per call
        return None, shown, "plugins_field"
    n = parsed.get("n", 1)
    if "best_of" in parsed or isinstance(n, bool) or n != 1:
        # the worst case per call bounds one completion: several per call (n, best_of) cost a multiple of it
        return None, shown, "several_completions"
    if "stream" in parsed and not isinstance(parsed["stream"], bool):
        return None, shown, "stream_not_a_boolean"
    if model not in allowed:
        return None, shown, "not_allowed"
    return parsed, model, None


def bill(doc) -> dict:
    """What one reply (or one stream event) says of the call's cost: charge, tokens, generation id, finish reason."""
    out: dict = {}
    if not isinstance(doc, dict):
        return out
    gid = doc.get("id")
    if isinstance(gid, str) and 0 < len(gid) <= 200:
        out["generation_id"] = gid
    choices = doc.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        reason = choices[0].get("finish_reason")
        if isinstance(reason, str) and len(reason) <= 64:
            out["finish_reason"] = reason
    if isinstance(doc.get("error"), dict):
        out["error"] = True
    usage = doc.get("usage")
    if isinstance(usage, dict):
        pt = (
            usage.get("prompt_tokens_details")
            if isinstance(usage.get("prompt_tokens_details"), dict)
            else {}
        )
        ct = (
            usage.get("completion_tokens_details")
            if isinstance(usage.get("completion_tokens_details"), dict)
            else {}
        )
        out["tokens"] = {
            "input": _count(usage.get("prompt_tokens")),
            "output": _count(usage.get("completion_tokens")),
            "cache_read": _count(pt.get("cached_tokens")),
            "cache_write": _count(pt.get("cache_write_tokens")),
            "reasoning": _count(ct.get("reasoning_tokens")),
        }
        charge = _charge(usage.get("cost"))
        if charge is not None:
            out["charge"] = charge
    return out


def redact(data: bytes, secrets: list[bytes]) -> bytes:
    for secret in secrets:
        if secret and secret in data:
            data = data.replace(secret, REDACTED)
    return data


# --- the relay ----------------------------------------------------------------------------------------------------------


class Refusal(Exception):
    def __init__(
        self,
        status: int,
        kind: str,
        message: str,
        reason: str,
        *,
        lane: str | None,
        model=None,
    ):
        super().__init__(kind)
        self.status, self.kind, self.message, self.reason, self.lane, self.model = (
            status,
            kind,
            message,
            reason,
            lane,
            model,
        )


class _Cut(Exception):
    """The relay is stopping and cut this call."""


#: The token counts of a journal row.
TOKEN_FIELDS = ("input", "output", "cache_read", "cache_write", "reasoning")


class _Ending:
    """One call's terminal journal row, written once (:meth:`end`), in the shape of its ``request_started`` row, with
    what the run's ledger books for the call (``booked_usd``)."""

    def __init__(self, journal: Journal, ledger: Ledger, base: dict, began: float):
        self.journal, self.ledger, self.base, self.began = journal, ledger, base, began
        self.row: dict | None = None

    def past(self, seconds: float) -> bool:
        return time.monotonic() - self.began > seconds

    def end(self, end: dict, info: dict | None = None) -> None:
        """Journal how the call ended and book what it cost (only the first time; later calls are no-ops)."""
        if self.row is not None:
            return
        info = info or {}
        charge = info.get("charge")
        row = {
            **self.base,
            "origin": "response",
            "at": _now(),
            "latency_ms": round((time.monotonic() - self.began) * 1000, 1),
            "status_code": None,
            "account_charge": _money(charge) if charge is not None else None,
            "measurement": "provider_reported" if charge is not None else "unknown",
            "tokens": info.get("tokens") or dict.fromkeys(TOKEN_FIELDS),
            "generation_id": info.get("generation_id"),
            "finish_reason": info.get("finish_reason"),
        }
        row.update(end)
        model, rid = self.base["requested_model"], self.base["request_attempt_id"]
        booked, kind = self.ledger.price(
            model, _charge(row["account_charge"]), row["tokens"]
        )
        kind = {"not_sent": "not_sent", "rejected_before_generation": "rejected"}.get(
            row["measurement"], kind
        )
        row["booked_usd"], row["charge_class"] = _money(booked), kind
        self.row = row
        self.journal.write(row)
        self.ledger.settle(rid, booked, kind)

    def undelivered(self) -> None:
        """The reply did not reach the caller after the call was journaled: one more row, ``cancelled``, with the same
        charge and booking (the last row is the one readers take)."""
        if self.row is not None:
            self.journal.write(
                {
                    **self.row,
                    "at": _now(),
                    "status": "cancelled",
                    "error_type": "client_gone",
                }
            )


class Relay:
    def __init__(self, config: Config):
        self.config = config
        self.secrets = config.secrets()
        self.journals = {
            lane: Journal(os.path.join(config.journal_dir, f"{lane}.jsonl"))
            for lane in config.tokens
        }
        self.ledger = Ledger(config)
        self.refusals = Journal(os.path.join(config.journal_dir, "refusals.jsonl"))
        self.slots = threading.BoundedSemaphore(MAX_IN_FLIGHT)
        self.stopping = threading.Event()
        self.active: dict[str, http.client.HTTPConnection] = {}
        #: The calls :meth:`drain` cut (their upstream connection shut): they end ``cancelled``, never ``completed``.
        self.cut: set[str] = set()
        self.active_lock = threading.Lock()
        self.handlers = 0
        self.handlers_idle = threading.Condition()
        self.instance = uuid.uuid4().hex[:12]
        self.port = 0
        if config.scheme == "https":
            self.tls = ssl.create_default_context(cafile=config.cafile)
        else:
            self.tls = None

    # -- the request ------------------------------------------------------------------------------------------------

    def handle(self, handler: "Handler", method: str) -> None:
        with self.handlers_idle:
            self.handlers += 1
        try:
            self._handle(handler, method)
        except Refusal as r:
            self._refuse(handler, r)
        except (ConnectionError, socket.timeout, OSError):
            handler.close_connection = True
        finally:
            with self.handlers_idle:
                self.handlers -= 1
                self.handlers_idle.notify_all()

    def _handle(self, handler: "Handler", method: str) -> None:
        handler.close_connection = True
        path = handler.path
        m = ROUTE.fullmatch(path)
        lane = (
            m.group(1)
            if m
            else next((n for n in LANES if path.startswith(f"/{n}/")), None)
        )
        hosts = handler.headers.get_all("Host") or []
        if len(hosts) != 1 or hosts[0].strip() != f"{LOOPBACK}:{self.port}":
            raise Refusal(
                403,
                f"{lane or 'lane'}_route_refused",
                "the lane relay serves 127.0.0.1 only",
                "host",
                lane=lane,
            )
        if m is None:
            raise Refusal(
                404,
                f"{lane or 'lane'}_route_refused",
                "the lane relay serves POST /<lane>/v1/chat/completions only",
                "path",
                lane=lane,
            )
        if method != "POST":
            raise Refusal(
                405,
                f"{lane}_route_refused",
                "the lane relay serves POST only",
                "method",
                lane=lane,
            )
        if self.stopping.is_set():
            raise Refusal(
                403,
                f"{lane}_route_refused",
                "the lane relay is stopping",
                "stopping",
                lane=lane,
            )
        token = self.config.tokens.get(lane)
        if token is None:
            raise Refusal(
                403,
                f"{lane}_route_refused",
                "this lane is not served in this run",
                "lane_off",
                lane=lane,
            )
        auth = handler.headers.get_all("Authorization") or []
        given = auth[0].strip().encode("latin-1", "replace") if len(auth) == 1 else b""
        if not (len(auth) == 1 and hmac.compare_digest(given, b"Bearer " + token)):
            crossing = any(
                hmac.compare_digest(given, b"Bearer " + other)
                for name, other in self.config.tokens.items()
                if name != lane
            )
            raise Refusal(
                403,
                f"{lane}_route_refused",
                "this lane needs its own token",
                "other_lane_token" if crossing else "no_lane_token",
                lane=lane,
            )
        if handler.headers.get_all("Transfer-Encoding"):
            raise Refusal(
                411,
                f"{lane}_route_refused",
                "the request needs a Content-Length",
                "length",
                lane=lane,
            )
        lengths = handler.headers.get_all("Content-Length") or []
        if len(lengths) != 1 or not lengths[0].strip().isdigit():
            raise Refusal(
                411,
                f"{lane}_route_refused",
                "the request needs a Content-Length",
                "length",
                lane=lane,
            )
        size = int(lengths[0].strip())
        if size > MAX_REQUEST_BYTES:
            raise Refusal(
                413,
                f"{lane}_route_refused",
                "the request body is over the relay's bound",
                "size",
                lane=lane,
            )
        body = handler.rfile.read(size)
        handler.body_read = True
        if len(body) != size:
            return  # the client went away mid-body
        parsed, model, why = check_body(body, self.config.models[lane])
        if why is not None:
            allowed = ", ".join(sorted(self.config.models[lane]))
            raise Refusal(
                403,
                "off_model_refused",
                f"this lane may call only {allowed}",
                why,
                lane=lane,
                model=model,
            )
        if not self.slots.acquire(timeout=BUSY_WAIT_S):
            raise Refusal(
                403,
                f"{lane}_route_busy",
                "every slot of the lane relay is in flight",
                f"{lane}_in_flight",
                lane=lane,
                model=model,
            )
        try:
            self._call(handler, lane, model, parsed, body)
        finally:
            self.slots.release()

    def _call(
        self, handler: "Handler", lane: str, model: str, parsed: dict, body: bytes
    ) -> None:
        rid = uuid.uuid4().hex
        base = {
            "request_attempt_id": rid,
            "call_id": rid,
            "lane": lane,
            "requested_model": model,
            "model": model,
            "path": "/v1/chat/completions",
            "provider": "openrouter",
            "currency": "USD",
            "unit": "openrouter_credits",
            "stream": parsed.get("stream") is True,
            "relay": self.instance,
        }
        meta = request_metadata(handler.headers)
        if meta is not None:
            base["unify_meta"] = meta
        started = {
            **base,
            "origin": "request_started",
            "status": "started",
            "at": _now(),
            "account_charge": None,
            "measurement": "unknown",
            "tokens": dict.fromkeys(TOKEN_FIELDS),
        }
        if not self.ledger.admit(
            rid, model, lambda: self.journals[lane].write(started)
        ):
            raise Refusal(
                403,
                f"{lane}_budget_exhausted",
                "the memory spend cap is reached: the booked spend plus this call's worst case is over it",
                f"{lane}_cap_reached",
                lane=lane,
                model=model,
            )
        if "usage" not in parsed:
            parsed["usage"] = {"include": True}
            body = json.dumps(parsed, separators=(",", ":")).encode("utf-8")
        call = _Ending(self.journals[lane], self.ledger, base, time.monotonic())
        try:
            self._upstream(handler, lane, rid, body, call)
        except _Cut:
            call.end({"status": "cancelled", "error_type": "stopping"})
        except (
            Exception
        ) as exc:  # noqa: BLE001 - a fixed category; the call may have been billed
            call.end({"status": "failed", "error_type": _category(exc)})
        finally:
            with self.active_lock:
                self.active.pop(rid, None)
            call.end({"status": "error", "error_type": "relay_error"})

    def _connect(self, rid: str) -> http.client.HTTPConnection:
        """A connection upstream, tried again on failure up to :data:`CONNECT_ATTEMPTS` times (nothing sent yet)."""
        wait = CONNECT_BACKOFF_S
        for attempt in range(CONNECT_ATTEMPTS):
            if self.stopping.is_set():
                raise _NotSent("stopping")
            if self.config.scheme == "https":
                conn = http.client.HTTPSConnection(
                    self.config.host,
                    self.config.port,
                    timeout=CONNECT_TIMEOUT_S,
                    context=self.tls,
                )
            else:
                conn = http.client.HTTPConnection(
                    self.config.host, self.config.port, timeout=CONNECT_TIMEOUT_S
                )
            try:
                conn.connect()
                conn.sock.settimeout(READ_TIMEOUT_S)
                with self.active_lock:
                    self.active[rid] = conn
                return conn
            except OSError as exc:
                conn.close()
                if attempt + 1 >= CONNECT_ATTEMPTS:
                    raise _NotSent(_category(exc)) from None
                if self.stopping.wait(wait):
                    raise _NotSent("stopping") from None
                wait *= 2
        raise _NotSent("connection")

    def _upstream(
        self, handler: "Handler", lane: str, rid: str, body: bytes, call: "_Ending"
    ) -> None:
        """The call upstream and its reply relayed. The call's terminal row is journaled before the caller can have
        the whole reply, so a next call it makes is admitted against what this one cost, not its worst case.
        """
        try:
            conn = self._connect(rid)
        except _NotSent as ns:
            call.end(
                {
                    "status": "failed",
                    "error_type": ns.category,
                    "account_charge": "0",
                    "measurement": "not_sent",
                }
            )
            self._send_json(
                handler,
                503,
                {
                    "error": {
                        "code": 503,
                        "type": f"{lane}_upstream_unreachable",
                        "message": "the lane relay could not reach the provider; nothing was sent",
                    }
                },
            )
            return
        try:
            conn.request(
                "POST",
                self.config.base_path + UPSTREAM_PATH,
                body=body,
                headers={
                    "Authorization": f"Bearer {self.config.key}",
                    "Content-Type": "application/json",
                    "Accept": handler.headers.get("Accept") or "application/json",
                },
            )
            resp = conn.getresponse()
            ctype = resp.getheader("Content-Type") or "application/json"
            ctype = (
                ctype
                if re.fullmatch(r"[A-Za-z0-9!#$%&'*+.^_`|~/=; -]{1,128}", ctype)
                else "application/json"
            )
            if "text/event-stream" in ctype.lower():
                self._relay_stream(handler, resp, ctype, rid, call)
            else:
                self._relay_whole(handler, resp, ctype, rid, call)
        except _Cut:
            raise
        except (
            Exception
        ) as exc:  # noqa: BLE001 - the call may have been billed: never tried again
            if self.stopping.is_set():
                raise _Cut from None
            call.end({"status": "failed", "error_type": _category(exc)})
            self._fail(handler, lane)
        finally:
            conn.close()

    def _fail(self, handler: "Handler", lane: str) -> None:
        if not handler.sent_headers:
            self._send_json(
                handler,
                502,
                {
                    "error": {
                        "code": 502,
                        "type": f"{lane}_upstream_failed",
                        "message": "the call upstream failed",
                    }
                },
            )

    def _was_cut(self, rid: str) -> None:
        with self.active_lock:
            if rid in self.cut:
                raise _Cut

    @staticmethod
    def _outcome(status: int, error: bool) -> dict:
        ok = 200 <= status < 300 and not error
        out = {"status": "completed" if ok else "failed", "status_code": status}
        if not ok:
            out["error_type"] = (
                f"http_{status}" if not 200 <= status < 300 else "provider_error"
            )
        return out

    def _relay_whole(self, handler, resp, ctype, rid, call: "_Ending") -> None:
        chunks: list[bytes] = []
        size = 0
        while True:
            if call.past(CALL_WALL_S):
                raise TimeoutError
            chunk = resp.read(65536)
            if not chunk:
                break
            size += len(chunk)
            if size > MAX_RESPONSE_BYTES:
                raise _TooLarge
            chunks.append(chunk)
        self._was_cut(rid)
        data = b"".join(chunks)
        try:
            info = bill(json.loads(data, parse_float=Decimal))
        except (ValueError, RecursionError):
            info = {}
        outcome = self._outcome(resp.status, bool(info.pop("error", False)))
        if (
            resp.status in REJECTED_STATUSES
            and "tokens" not in info
            and "charge" not in info
        ):
            # refused before generating: nothing was billed
            outcome.update(account_charge="0", measurement="rejected_before_generation")
        call.end(outcome, info)
        out = redact(data, self.secrets)
        try:
            handler.send_response_only(resp.status)
            handler.send_header("Content-Type", ctype)
            handler.send_header("Content-Length", str(len(out)))
            handler.send_header("Connection", "close")
            handler.end_headers()
            handler.sent_headers = True
            handler.wfile.write(out)
            handler.wfile.flush()
        except OSError:
            call.undelivered()

    def _relay_stream(self, handler, resp, ctype, rid, call: "_Ending") -> None:
        handler.send_response_only(resp.status)
        handler.send_header("Content-Type", ctype)
        handler.send_header("Transfer-Encoding", "chunked")
        handler.send_header("Connection", "close")
        handler.end_headers()
        handler.sent_headers = True
        info: dict = {}
        error = False
        while True:
            if call.past(CALL_WALL_S):
                raise TimeoutError
            line = resp.readline(MAX_EVENT_LINE + 1)
            if not line:
                self._was_cut(rid)
                break
            if len(line) > MAX_EVENT_LINE:
                raise _TooLarge
            stripped = line.strip()
            if stripped.startswith(b"data:"):
                payload = stripped[5:].strip()
                if payload and payload != b"[DONE]":
                    try:
                        event = bill(json.loads(payload, parse_float=Decimal))
                    except (ValueError, RecursionError):
                        event = {}
                    error = error or bool(event.pop("error", False))
                    info.update(event)
            out = redact(line, self.secrets)
            try:
                handler.wfile.write(b"%x\r\n%s\r\n" % (len(out), out))
            except OSError:
                call.end(
                    {
                        "status": "cancelled",
                        "error_type": "client_gone",
                        "status_code": resp.status,
                    },
                    info,
                )
                return
        call.end(self._outcome(resp.status, error), info)
        try:
            handler.wfile.write(b"0\r\n\r\n")  # the stream's end, after its row
        except OSError:
            call.undelivered()

    # -- refusals and replies ---------------------------------------------------------------------------------------

    def _refuse(self, handler: "Handler", r: Refusal) -> None:
        try:
            self.refusals.write(
                {
                    "at": _now(),
                    "lane": r.lane,
                    "type": r.kind,
                    "reason": r.reason,
                    "requested_model": r.model,
                    "relay": self.instance,
                }
            )
        except OSError:
            pass
        if handler.sent_headers:
            return
        self._discard_body(handler)
        body = {
            "error": {
                "code": r.status,
                "type": r.kind,
                "message": r.message,
                "reason": r.reason,
            }
        }
        self._send_json(handler, r.status, body)

    @staticmethod
    def _discard_body(handler: "Handler") -> None:
        """Read and drop a refused request's body (within the request bound) before answering, so closing the
        connection never resets it before the caller has read the answer. It is read in pieces of
        :data:`DISCARD_CHUNK` bytes, so a refused body never takes more memory than one piece.
        """
        if (
            handler.body_read
            or handler.headers is None
            or handler.headers.get_all("Transfer-Encoding")
        ):
            return
        handler.body_read = True
        lengths = handler.headers.get_all("Content-Length") or []
        if not (len(lengths) == 1 and lengths[0].strip().isdigit()):
            return
        left = int(lengths[0].strip())
        if left > MAX_REQUEST_BYTES:
            return
        try:
            while left > 0:
                piece = handler.rfile.read(min(DISCARD_CHUNK, left))
                if not piece:
                    return
                left -= len(piece)
        except OSError:
            pass

    @staticmethod
    def _send_json(handler: "Handler", status: int, doc: dict) -> None:
        data = json.dumps(doc).encode("utf-8")
        try:
            handler.send_response_only(status)
            handler.send_header("Content-Type", "application/json")
            handler.send_header("Content-Length", str(len(data)))
            handler.send_header("Connection", "close")
            handler.end_headers()
            handler.sent_headers = True
            if handler.command != "HEAD":
                handler.wfile.write(data)
        except OSError:
            pass
        handler.close_connection = True

    # -- stopping ---------------------------------------------------------------------------------------------------

    def drain(self, server: ThreadingHTTPServer) -> int:
        """Stop: refuse new calls, stop listening, let the calls in flight end within :data:`DRAIN_S`, then cut the
        rest (each journals ``cancelled``). Returns how many were cut."""
        self.stopping.set()
        server.shutdown()
        server.server_close()
        deadline = time.monotonic() + DRAIN_S
        with self.handlers_idle:
            while self.handlers and time.monotonic() < deadline:
                self.handlers_idle.wait(0.2)
        with self.active_lock:
            cut = list(self.active.values())
            self.cut.update(self.active)
        for conn in cut:
            try:
                if conn.sock is not None:
                    conn.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        deadline = time.monotonic() + 3.0
        with self.handlers_idle:
            while self.handlers and time.monotonic() < deadline:
                self.handlers_idle.wait(0.2)
        return len(cut)


class _NotSent(Exception):
    def __init__(self, category: str):
        super().__init__(category)
        self.category = category


class _TooLarge(Exception):
    pass


def _category(exc: BaseException) -> str:
    """A fixed category for a failure: never its text."""
    if isinstance(exc, _TooLarge):
        return "too_large"
    if isinstance(exc, (socket.timeout, TimeoutError)):
        return "timeout"
    if isinstance(exc, ssl.SSLError):
        return "tls"
    if isinstance(exc, socket.gaierror):
        return "dns"
    if isinstance(exc, (ConnectionError, OSError, http.client.HTTPException)):
        return "connection"
    return "relay_error"


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "unify-lane-relay"
    sys_version = ""
    timeout = CLIENT_TIMEOUT_S

    def setup(self) -> None:
        super().setup()
        self.sent_headers = False
        self.body_read = False
        self.headers = None

    def log_message(
        self, format, *args
    ) -> None:  # noqa: A002 - no request is ever logged (a path may hold anything)
        return

    def send_error(self, code, message=None, explain=None) -> None:
        """A malformed request line or header block: a fixed JSON refusal that echoes nothing of the request."""
        self.close_connection = True
        Relay._send_json(
            self,
            code if isinstance(code, int) and 400 <= code < 600 else 400,
            {
                "error": {
                    "code": code,
                    "type": "lane_route_refused",
                    "message": "malformed request",
                    "reason": "malformed",
                }
            },
        )

    def _any(self) -> None:
        self.server.relay.handle(self, self.command)

    do_POST = do_GET = do_PUT = do_DELETE = do_PATCH = do_OPTIONS = do_HEAD = _any


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False
    request_queue_size = 64

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.connections = threading.BoundedSemaphore(MAX_CONNECTIONS)

    def process_request(self, request, client_address) -> None:
        """A thread per connection, at most :data:`MAX_CONNECTIONS` at once; one more is closed unread."""
        if not self.connections.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self.connections.release()
            raise

    def process_request_thread(self, request, client_address) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.connections.release()

    def handle_error(
        self, request, client_address
    ) -> None:  # no traceback on stderr (it may quote a request)
        return


def bind(config: Config) -> Server:
    """The relay for *config*, listening on 127.0.0.1 on a port the kernel picks (``server.relay.port``); serving
    starts with ``server.serve_forever``."""
    relay = Relay(config)
    server = Server((LOOPBACK, 0), Handler)
    server.relay = relay
    relay.port = server.server_address[1]
    return server


# --- the process --------------------------------------------------------------------------------------------------------


def harden() -> bool:
    """No core file, and not dumpable (no other process of this user may read this one's memory or trace it).
    Returns whether the second took."""
    try:
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except (ValueError, OSError):
        pass
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        return libc.prctl(4, 0, 0, 0, 0) == 0  # PR_SET_DUMPABLE, 0
    except (OSError, AttributeError):
        return False


def _say(stream, doc: dict) -> None:
    stream.write(json.dumps(doc) + "\n")
    stream.flush()


def _log(text: str) -> None:
    try:
        sys.stderr.write(f"{_now()} lane relay: {text}\n")
        sys.stderr.flush()
    except (OSError, ValueError):
        pass


def main() -> int:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    stop = threading.Event()
    why: list[str] = []

    def on_term(_sig, _frame):
        why.append("terminated")
        stop.set()

    signal.signal(signal.SIGTERM, on_term)
    undumpable = harden()
    try:
        config = Config(read_config(0))
        server = bind(config)
    except ConfigError as exc:
        _say(sys.stdout, {"failed": str(exc)})
        return 3
    except (OSError, ssl.SSLError) as exc:
        _say(sys.stdout, {"failed": f"start-up failed: {type(exc).__name__}"})
        return 3
    relay = server.relay
    serving = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.5}, daemon=True
    )
    serving.start()
    _say(
        sys.stdout,
        {
            "ready": FORMAT,
            "port": relay.port,
            "pid": os.getpid(),
            "lanes": sorted(config.tokens),
            "undumpable": undumpable,
        },
    )
    null = os.open(os.devnull, os.O_WRONLY)
    os.dup2(null, 1)
    os.close(null)

    def lifeline() -> None:
        while True:
            try:
                if not os.read(0, 65536):
                    break
            except InterruptedError:
                continue
            except OSError:
                break
        why.append("lifeline closed")
        stop.set()

    threading.Thread(target=lifeline, daemon=True).start()
    _log(f"serving {len(config.tokens)} lane(s) on port {relay.port}")
    while not stop.wait(0.5):
        pass
    _log(f"stopping ({why[0] if why else 'stopped'})")
    cut = relay.drain(server)
    _log(
        f"stopped; {cut} call(s) cut; calls by charge class: {relay.ledger.by_class()}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

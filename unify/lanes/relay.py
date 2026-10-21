"""Start and stop the local lane relay (:mod:`.relay_server`), the process that holds the provider key for the memory
lanes. Standard library only.

:meth:`LaneRelay.start` runs ``<python> -I -S relay_server.py`` in its own session with an environment of ``PATH``
alone, writes the configuration (the key included) as one line on its stdin, and reads its ready line. The key goes
nowhere else: not argv, not the environment, not a file. The pipe to the relay's stdin then stays open as its lifeline:
this process holds the only write end (it is not inheritable, and every child is started with ``close_fds``), so the
relay ends when :meth:`LaneRelay.stop` closes it or when this process dies for any reason, ``kill -9`` included.

Each lane gets a fresh random token (``secrets.token_urlsafe(32)``) for this run; :meth:`LaneRelay.token` hands it to
whoever passes it on (``unify/lanes/launch.py``: an inherited descriptor, as the lane contract says).

:meth:`LaneRelay.stop` closes the lifeline (the relay lets its calls in flight end, up to its drain bound, and exits),
then SIGTERM and SIGKILL if it has not ended in time, and verifies the end: the relay is reaped and no process is left
in its process group (it starts none).
"""

from __future__ import annotations

import atexit
import json
import os
import secrets
import select
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Mapping, Sequence

SERVER = Path(__file__).with_name("relay_server.py")
FORMAT = "unify-lane-relay/1"
LANES = ("writer", "helper", "fork")
#: OpenRouter's API: the only upstream the lanes have (their bodies are OpenRouter-shaped). Tests pass a local stand-in.
DEFAULT_UPSTREAM = "https://openrouter.ai/api/v1"
#: The relay's environment: nothing of this process's (no key, no token, no ``PYTHON*``).
RELAY_ENV = {"PATH": "/usr/bin:/bin"}
#: Seconds: for the ready line; for the relay to end after its lifeline closes (its own drain bound plus a margin);
#: after SIGTERM; after SIGKILL.
START_TIMEOUT_S = 15.0
STOP_WAIT_S = 30.0
TERM_WAIT_S = 5.0
KILL_WAIT_S = 5.0


class LaneRelayError(RuntimeError):
    """The relay did not start. The text is fixed (or the relay's own fixed refusal); it never quotes a value."""


@dataclass(frozen=True)
class Stopped:
    """How :meth:`LaneRelay.stop` ended the relay: ``how`` is ``lifeline`` (it ended on its own once the lifeline
    closed), ``terminated``, ``killed`` or ``already_ended``; ``verified`` when it is reaped and nothing is left in its
    process group."""

    verified: bool
    how: str
    returncode: int | None


def mint_token() -> str:
    return secrets.token_urlsafe(32)


def _group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _money(value: Decimal | str) -> str:
    d = Decimal(str(value))
    if not d.is_finite() or d < 0:
        raise ValueError("an amount must be a finite, non-negative decimal")
    return format(abs(d) if d.is_zero() else d, "f")


def _private_folder(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    os.chmod(path, 0o700)
    return path


class LaneRelay:
    """A running relay: its lanes' base URLs and tokens, and :meth:`stop`."""

    def __init__(
        self,
        proc: subprocess.Popen,
        port: int,
        tokens: dict[str, str],
        journal_dir: Path,
        undumpable: bool = True,
    ):
        self._proc = proc
        self.port = port
        #: Whether the relay made itself undumpable (no other process of this user may read its memory or trace it).
        self.undumpable = undumpable
        self._tokens = tokens
        self.journal_dir = journal_dir
        self._stopped: Stopped | None = None
        self._lock = threading.Lock()

    @classmethod
    def start(
        cls,
        *,
        key: str,
        lanes: Mapping[str, Sequence[str]],
        cap_usd: Decimal | str,
        worst_case_usd: Mapping[str, Decimal | str],
        journal_dir: Path | str,
        prices: Mapping[str, Mapping[str, Decimal | str]] | None = None,
        upstream: str = DEFAULT_UPSTREAM,
        cafile: str | None = None,
        python: str | None = None,
        log_path: Path | str | None = None,
        start_timeout_s: float = START_TIMEOUT_S,
    ) -> "LaneRelay":
        """Start the relay for *lanes* (lane -> the models it serves), all held to *cap_usd* on one ledger for the
        relay's life (one run), each model's unpriced call booked at *worst_case_usd*; the journals in *journal_dir*
        (made mode 0700); the relay's own log (fixed lines only) at *log_path* (else nowhere).
        """
        if not lanes or not set(lanes) <= set(LANES):
            raise LaneRelayError("the lanes must be some of writer, helper and fork")
        folder = _private_folder(Path(journal_dir).resolve())
        tokens = {lane: mint_token() for lane in lanes}
        config = {
            "format": FORMAT,
            "key": key,
            "upstream": upstream,
            "cafile": cafile,
            "journal_dir": str(folder),
            "cap_usd": _money(cap_usd),
            "worst_case_usd": {m: _money(v) for m, v in worst_case_usd.items()},
            "prices": {
                m: {k: _money(v) for k, v in t.items()}
                for m, t in (prices or {}).items()
            },
            "lanes": {
                lane: {"token": tokens[lane], "models": list(models)}
                for lane, models in lanes.items()
            },
        }
        line = (json.dumps(config) + "\n").encode("utf-8")
        del config
        log = None
        if log_path is not None:
            log = os.open(
                str(log_path),
                os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC,
                0o600,
            )
        try:
            proc = subprocess.Popen(
                [python or sys.executable, "-I", "-S", str(SERVER)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=log if log is not None else subprocess.DEVNULL,
                env=dict(RELAY_ENV),
                cwd=str(folder),
                close_fds=True,
                start_new_session=True,
            )
        finally:
            if log is not None:
                os.close(log)
        try:
            proc.stdin.write(line)
            proc.stdin.flush()
            del line
            port, undumpable = cls._ready(proc, start_timeout_s)
        except BaseException as exc:
            cls._kill(proc)
            if isinstance(exc, LaneRelayError):
                raise
            raise LaneRelayError(
                f"the lane relay did not start ({type(exc).__name__})"
            ) from None
        relay = cls(proc, port, tokens, folder, undumpable)
        atexit.register(relay.stop)
        return relay

    @staticmethod
    def _ready(proc: subprocess.Popen, timeout_s: float) -> tuple[int, bool]:
        fd = proc.stdout.fileno()
        data = b""
        deadline = time.monotonic() + timeout_s
        while b"\n" not in data:
            left = deadline - time.monotonic()
            if left <= 0 or not select.select([fd], [], [], left)[0]:
                raise LaneRelayError("the lane relay did not report ready in time")
            chunk = os.read(fd, 65536)
            if not chunk:
                raise LaneRelayError("the lane relay ended before it was ready")
            data += chunk
            if len(data) > 65536:
                raise LaneRelayError("the lane relay's ready line is too long")
        proc.stdout.close()
        try:
            doc = json.loads(data.split(b"\n", 1)[0])
        except ValueError:
            raise LaneRelayError("the lane relay's ready line is not JSON") from None
        if not isinstance(doc, dict):
            raise LaneRelayError("the lane relay's ready line is not an object")
        if "failed" in doc:
            reason = doc["failed"] if isinstance(doc["failed"], str) else "no reason"
            raise LaneRelayError(f"the lane relay refused to start: {reason[:200]}")
        port = doc.get("port")
        if (
            doc.get("ready") != FORMAT
            or doc.get("pid") != proc.pid
            or not isinstance(port, int)
            or not 0 < port < 65536
        ):
            raise LaneRelayError(
                "the lane relay's ready line differs from its contract"
            )
        return port, doc.get("undumpable") is True

    @staticmethod
    def _kill(proc: subprocess.Popen) -> None:
        try:
            proc.kill()
        except OSError:
            pass
        try:
            proc.wait(KILL_WAIT_S)
        except subprocess.TimeoutExpired:
            pass
        for stream in (proc.stdin, proc.stdout):
            try:
                if stream is not None:
                    stream.close()
            except OSError:
                pass

    # -- what the lanes need ----------------------------------------------------------------------------------------

    @property
    def pid(self) -> int:
        return self._proc.pid

    @property
    def lanes(self) -> tuple[str, ...]:
        return tuple(self._tokens)

    def base_url(self, lane: str) -> str:
        """The lane's base URL, as the lane contract has it: ``http://127.0.0.1:<port>/<lane>/v1``."""
        if lane not in self._tokens:
            raise KeyError(lane)
        return f"http://127.0.0.1:{self.port}/{lane}/v1"

    def token(self, lane: str) -> str:
        return self._tokens[lane]

    def journal(self, lane: str) -> Path:
        return self.journal_dir / f"{lane}.jsonl"

    def alive(self) -> bool:
        return self._proc.poll() is None

    # -- stopping ---------------------------------------------------------------------------------------------------

    def stop(self, wait_s: float = STOP_WAIT_S) -> Stopped:
        """Close the lifeline, wait for the relay to end (*wait_s*), then SIGTERM, then SIGKILL; verified when it is
        reaped and its process group is empty. Idempotent."""
        with self._lock:
            if self._stopped is not None:
                return self._stopped
            proc = self._proc
            how = "already_ended" if proc.poll() is not None else "lifeline"
            try:
                proc.stdin.close()
            except OSError:
                pass
            if proc.returncode is None:
                for sig, wait in (
                    (None, wait_s),
                    (signal.SIGTERM, TERM_WAIT_S),
                    (signal.SIGKILL, KILL_WAIT_S),
                ):
                    if sig is not None:
                        how = "terminated" if sig == signal.SIGTERM else "killed"
                        try:
                            proc.send_signal(sig)
                        except OSError:
                            pass
                    try:
                        proc.wait(wait)
                        break
                    except subprocess.TimeoutExpired:
                        continue
            verified = proc.returncode is not None and not _group_alive(proc.pid)
            self._stopped = Stopped(verified, how, proc.returncode)
            self._tokens = {}
            try:
                atexit.unregister(self.stop)
            except Exception:  # noqa: BLE001
                pass
            return self._stopped

    def __enter__(self) -> "LaneRelay":
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

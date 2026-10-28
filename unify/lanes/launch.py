"""The ``unify`` command's side of the memory lanes: with memory on and no lane given by the environment, run the
controller as a child of this process, its lanes served by a local relay (:mod:`.relay`) that this process owns.

**Why a child.** The controller reads each lane's token from an inherited descriptor once, when its settings are built
(``unify/memory/integration/switch.py``: ``settle_writer_route_env`` and the fork and helper lanes' likewise), which is
at import, before any code of the command runs. So the relay has to exist first: this process starts it, mints one
token per lane, and starts the controller (``python -m unify <the same arguments>``) with, per lane, the base URL and
the number of a fresh pipe holding the token (``UNIFY_MEMORY_*_BASE_URL`` and ``*_TOKEN_FD``, the lane contract's
names), and ``UNIFY_MEMORY_WRITER_JOURNAL`` naming the writer lane's journal. The pipes' write ends are closed before
the controller starts; their read ends are closed here as soon as it has started. The controller inherits stdin,
stdout and stderr as they are, so ``--jsonl`` drivers and terminals see no difference.

**What the relay gets.** The provider key (``OPENROUTER_API_KEY``), on its stdin only; each lane's model (the writer's
for the writer lane and the helper lane (the helpers run on the writer's model), the actor's for the fork, which continues the actor's
conversation); the memory spend cap (``UNIFY_MEMORY_CAP_USD``, or its default), which holds this run (the relay's life) on one ledger for all three lanes; the
worst case per call of every model the pinned catalogue bounds (``unify/memory/call_bounds.py``); the writer's pinned
prices when set. A lane that cannot be served
(a model that is not an OpenRouter one, or that the catalogue does not bound) is left out with a note; with no key, a cap
that is not a plain decimal, or no lane left, the command runs as before, in this process, with a one-line note on stderr (the fork then
records ``refused: no fork lane``).

**The writer service** (one per home, detached) holds the tokens of the run whose controller started it, so it must not
outlive that run's relay. When the controller ends, the service is stopped as a stage end (``drain(end="suspend")``):
its pass in flight gets at most :data:`EXIT_WAIT_S` (5 minutes) to finish, and is then suspended (cut through its
abort path, its turns kept); the entries still owed, the cut pass's included, stay owed and are handed again next run,
never dropped. Then the relay is stopped and its end verified. A Ctrl-C during that wait stops waiting. Every run holds a shared lock on ``runs.lock`` for its
whole life; a run that finds no other run on the home (an exclusive lock was free) and a service still running, or a
stop marker left, stops it the same way before it starts its own (a run that was killed, or whose stop was cut short,
is tidied up by the next one). If this process dies, the relay's lifeline closes and it ends too.

**Signals.** The controller stays in this process's process group, so a terminal's Ctrl-C (SIGINT, SIGQUIT) reaches
it directly; here those are only noted (after the controller has ended, they cut the wait for the writer service).
SIGTERM and SIGHUP sent to this process are passed on to the controller. The handlers are Python's, so the controller,
once started, has every signal at its default again.
"""

from __future__ import annotations

import fcntl
import json
import os
import signal
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Mapping, MutableMapping, Sequence

from .relay import DEFAULT_UPSTREAM, LaneRelay, LaneRelayError

#: The provider key the relay holds. The only variable this module reads for itself.
KEY_ENV = "OPENROUTER_API_KEY"
#: Each lane's base URL and token descriptor, as the controller reads them (the lane contract).
LANE_ENV = {
    "writer": ("UNIFY_MEMORY_WRITER_BASE_URL", "UNIFY_MEMORY_WRITER_TOKEN_FD"),
    "helper": (
        "UNIFY_MEMORY_LIBRARY_HELPER_BASE_URL",
        "UNIFY_MEMORY_LIBRARY_HELPER_TOKEN_FD",
    ),
    "fork": (
        "UNIFY_MEMORY_LIBRARY_FORK_BASE_URL",
        "UNIFY_MEMORY_LIBRARY_FORK_TOKEN_FD",
    ),
}
JOURNAL_ENV = "UNIFY_MEMORY_WRITER_JOURNAL"
#: The embedding lane, as the embedding client reads it (``unify/common/embeddings.py``, :mod:`.embed_token`):
#: the URL it posts to and the descriptor holding the lane's token. Given only when the user set neither.
EMBED_URL_ENV = "UNIFY_EMBED_URL"
EMBED_TOKEN_FD_ENV = "UNIFY_EMBED_TOKEN_FD"
EMBED_MODEL = "openai/text-embedding-3-small"
#: One embedding request's worst case: the relay bounds a request to 2 MiB of input, at most 2,097,152 tokens, at the
#: model's listed USD 0.02 per million input tokens (the pinned call-bounds table has no embedding model).
EMBED_WORST_CASE_USD = Decimal("0.05")
#: Its listed prices, USD per million tokens: a call that reports its tokens but no charge is estimated from them.
EMBED_PRICES = {"input": Decimal("0.02"), "output": Decimal("0")}
#: Names that mean the lanes were given from outside (a research launcher, or this module in the parent process): the
#: command then runs as it is, and no relay starts.
GIVEN_ENV = (
    *(name for pair in LANE_ENV.values() for name in pair),
    "UNIFY_MEMORY_WRITER_TOKEN",
)
#: The suffix of an OpenRouter endpoint (``vendor/model@openrouter``); a model without a suffix is an OpenRouter id.
OPENROUTER = "@openrouter"
#: Signals passed on to the controller, and signals it gets from the terminal itself (only noted here).
FORWARDED = (signal.SIGTERM, signal.SIGHUP)
NOTED = (signal.SIGINT, signal.SIGQUIT)
POLL_S = 0.2
#: The most a writer pass in flight gets to finish when a run ends (or when a run tidies what an ended one left)
#: before it is suspended.
EXIT_WAIT_S = 300.0


@dataclass(frozen=True)
class Plan:
    """What the relay serves: each lane's models, the cap, each model's worst case per call, pinned prices."""

    lanes: dict[str, tuple[str, ...]]
    cap: Decimal
    worst: dict[str, Decimal]
    prices: dict[str, dict[str, Decimal]] = field(default_factory=dict)


def openrouter_id(endpoint: str) -> str | None:
    """The OpenRouter model id of *endpoint* (``vendor/model`` or ``vendor/model@openrouter``), else None."""
    text = str(endpoint or "").strip()
    if text.endswith(OPENROUTER):
        text = text[: -len(OPENROUTER)]
    return text if text and "@" not in text else None


def lanes_given(settings: Any, environ: Mapping[str, str]) -> bool:
    """Whether the controller's lanes come from outside: any lane setting in *settings* (the controller's settles
    take the names out of the environment once read) or in *environ* (in any letter case).
    """
    names = {n.upper() for n in environ}
    for name in GIVEN_ENV:
        value = getattr(settings, name, "")
        value = (
            value.get_secret_value() if hasattr(value, "get_secret_value") else value
        )
        if str(value or "").strip() or name in names:
            return True
    return False


def _writer_model(settings: Any) -> str:
    """The model the writer and its helpers run on (:func:`..switch.writer_model`: its own, else the actor's)."""
    from unify.memory.integration import switch

    return switch.writer_model(settings)


def _bounds() -> dict[str, Decimal]:
    """Every model the pinned catalogue bounds, at its worst case per call (a model it lists at no bound is left out:
    the relay serves no lane on it, and books an old call of it at the table's largest bound).
    """
    from unify.memory import call_bounds

    table = (
        json.loads(call_bounds.TABLE.read_text(encoding="utf-8")).get("bounds") or {}
    )
    out = {}
    for model in table:
        try:
            out[model] = call_bounds.call_worst_case_usd(model)
        except call_bounds.NoBound:
            continue
    return out


def plan(settings: Any, environ: Mapping[str, str]) -> tuple[Plan | None, str | None]:
    """``(plan, None)`` when the lanes should be served here; ``(None, None)`` when memory wants none or they are
    given; ``(None, why)`` when memory wants them and they cannot be served (*why* names a rule, never a value).
    """
    from unify.memory.integration import switch

    if lanes_given(settings, environ):
        return None, None
    if not switch.memory_on(settings):
        return None, None
    if not str(environ.get(KEY_ENV, "") or "").strip():
        return None, f"memory has no model lanes: {KEY_ENV} is not set"
    try:
        cap = switch.memory_cap_usd(settings)  # empty: the default cap
    except ValueError:
        return (
            None,
            f"memory has no model lanes: {switch.MEMORY_CAP} (the memory spend cap) is not a plain decimal above zero",
        )
    try:
        bounds = _bounds()
    except (
        Exception
    ) as exc:  # noqa: BLE001 - a table that cannot be read bounds nothing: no lane
        return (
            None,
            f"memory has no model lanes: the worst-case table could not be read ({type(exc).__name__})",
        )
    writer = openrouter_id(_writer_model(settings))
    helper = writer  # the helpers run on the writer's model
    actor = openrouter_id(str(getattr(settings, "UNIFY_MODEL", "") or ""))
    lanes: dict[str, tuple[str, ...]] = {}
    skipped: list[str] = []
    for lane, model in (("writer", writer), ("helper", helper), ("fork", actor)):
        if model is not None and model in bounds:
            lanes[lane] = (model,)
        else:
            skipped.append(lane)
    if not lanes:
        return (
            None,
            "memory has no model lanes: no lane's model is an OpenRouter model the worst-case table bounds",
        )
    if not _embedding_given(settings, environ):
        lanes["embed"] = (EMBED_MODEL,)
        bounds = {**bounds, EMBED_MODEL: EMBED_WORST_CASE_USD}
    prices = {EMBED_MODEL: EMBED_PRICES} if "embed" in lanes else {}
    pinned = switch.writer_prices(settings)
    if pinned and writer is not None:
        prices[writer] = pinned
    why = f"memory: no {', '.join(skipped)} lane (its model is not an OpenRouter model the worst-case table bounds)"
    return Plan(lanes, cap, bounds, prices), (why if skipped else None)


def _embedding_given(settings: Any, environ: Mapping[str, str]) -> bool:
    """Whether embeddings need no lane here: the user set an embedding route (or its token), or local embeddings."""
    names = {n.upper() for n in environ}
    return (
        EMBED_URL_ENV in names
        or EMBED_TOKEN_FD_ENV in names
        or bool(str(getattr(settings, EMBED_URL_ENV, "") or "").strip())
        or bool(getattr(settings, "UNIFY_LOCAL_EMBEDDINGS", False))
    )


def handover(
    relay: LaneRelay, environ: Mapping[str, str]
) -> tuple[dict[str, str], list[int]]:
    """The controller's environment (*environ* with each lane's base URL and token descriptor, and the writer's
    journal) and the descriptors it inherits: one fresh pipe per lane, its token written and its write end closed.
    """
    env = {
        k: v
        for k, v in environ.items()
        if k.upper() not in GIVEN_ENV and k.upper() != JOURNAL_ENV
    }
    fds: list[int] = []
    try:
        for lane in relay.lanes:
            base_name, fd_name = LANE_ENV.get(lane, (EMBED_URL_ENV, EMBED_TOKEN_FD_ENV))
            r, w = os.pipe()
            fds.append(r)
            try:
                os.write(w, (relay.token(lane) + "\n").encode("ascii"))
            finally:
                os.close(w)
            os.set_inheritable(r, True)
            env[base_name] = relay.base_url(lane) + (
                "/embeddings" if lane == "embed" else ""
            )
            env[fd_name] = str(r)
        if "writer" in relay.lanes:
            env[JOURNAL_ENV] = str(relay.journal("writer"))
    except BaseException:
        for r in fds:
            os.close(r)
        raise
    return env, fds


def _cafile() -> str | None:
    """The CA bundle the harness's own HTTP clients use, when it is installed; else the system's (None)."""
    try:
        import certifi

        return certifi.where()
    except Exception:  # noqa: BLE001
        return None


def _drain_writer(home: Path) -> Any:
    from unify.memory.integration.async_pass import drain
    from unify.memory.integration.paths import Paths

    return drain(Paths.under(home), end="suspend", service_wait_s=EXIT_WAIT_S)


def _left_behind(home: Path) -> bool:
    """Whether a writer service an earlier run started may still hold that run's lanes, or its stop marker is left:
    its state not ``stopped``, or a marker still there (a stop that was cut short)."""
    from unify.memory.writer_service import read_state, stopping_marker

    doc = read_state(home)
    return stopping_marker(home) is not None or (
        isinstance(doc, dict) and doc.get("state") not in (None, "stopped")
    )


def _exit_code(returncode: int) -> int:
    return 128 - returncode if returncode < 0 else returncode


def _note(text: str) -> None:
    print(text, file=sys.stderr, flush=True)


class _Signals:
    """While installed: SIGTERM and SIGHUP are passed on to the controller (once it runs), SIGINT and SIGQUIT only
    noted. Every signal is noted in :attr:`seen`."""

    def __init__(self) -> None:
        self.child: subprocess.Popen | None = None
        self.seen: list[int] = []
        self.before: dict = {}

    def _on(self, signum, _frame) -> None:
        self.seen.append(signum)
        if signum in FORWARDED and self.child is not None and self.child.poll() is None:
            try:
                self.child.send_signal(signum)
            except OSError:
                pass

    def __enter__(self) -> "_Signals":
        self.before = {
            sig: signal.signal(sig, self._on) for sig in (*FORWARDED, *NOTED)
        }
        return self

    def __exit__(self, *exc) -> None:
        for sig, handler in self.before.items():
            signal.signal(sig, handler)


def _run_lock(folder: Path) -> tuple[int, bool]:
    """A shared lock on ``runs.lock`` held for the whole run (closed with this process, however it ends), and whether
    this run is the only one on the home: then a writer service still running was left by a run that has ended.
    """
    fd = os.open(
        folder / "runs.lock",
        os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC,
        0o600,
    )
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        alone = True
    except BlockingIOError:
        alone = False
    fcntl.flock(fd, fcntl.LOCK_SH)
    return fd, alone


def run_controller(
    argv: Sequence[str],
    home: Path | str,
    *,
    settings: Any = None,
    environ: MutableMapping[str, str] | None = None,
    controller: Sequence[str] | None = None,
    upstream: str = DEFAULT_UPSTREAM,
    drain: Callable[[Path], Any] | None = None,
    note: Callable[[str], None] = _note,
) -> int | None:
    """Run the controller as a child with the relay's lanes and return its exit code; or None when this process
    should run the command itself (no lane wanted, the lanes given from outside, or they cannot be served: a note
    says why). *controller* is the child's command before *argv* (default ``<python> -m unify``).
    """
    if settings is None:
        from unify.settings import SETTINGS as settings
    environ = os.environ if environ is None else environ
    home = Path(home)
    drain = drain or _drain_writer
    the_plan, why = plan(settings, environ)
    if why is not None:
        note(why)
    if the_plan is None:
        return None
    from unify.memory.integration.paths import Paths

    folder = Paths.under(home).state_dir / "lanes"
    folder.mkdir(parents=True, exist_ok=True)
    os.chmod(folder, 0o700)
    lock, alone = _run_lock(folder)
    try:
        with _Signals() as signals:
            if alone and "writer" in the_plan.lanes and _left_behind(home):
                # a service an ended run left holds that run's dead lanes, or its stop was cut short: stop it now, so
                # this run's controller starts one with its own
                _stop_writer(home, drain, signals, note, at_start=True)
                signals.seen.clear()
            return _with_relay(
                argv,
                home,
                the_plan,
                folder,
                environ,
                controller,
                upstream,
                drain,
                note,
                signals,
            )
    finally:
        os.close(lock)


def _with_relay(
    argv, home, the_plan, folder, environ, controller, upstream, drain, note, signals
) -> int | None:
    try:
        relay = LaneRelay.start(
            key=environ[KEY_ENV].strip(),
            lanes=the_plan.lanes,
            cap_usd=the_plan.cap,
            worst_case_usd=the_plan.worst,
            prices=the_plan.prices,
            journal_dir=folder,
            upstream=upstream,
            cafile=_cafile(),
            log_path=folder / "relay.log",
        )
    except (LaneRelayError, OSError) as exc:
        note(
            f"memory has no model lanes: {exc if isinstance(exc, LaneRelayError) else type(exc).__name__}"
        )
        return None
    if not relay.undumpable:
        note(
            "memory: the lane relay could not make itself undumpable; it runs, its key readable by this user's processes"
        )
    try:
        env, fds = handover(relay, environ)
        command = [*(controller or [sys.executable, "-m", "unify"]), *argv]
        try:
            child = subprocess.Popen(
                command, env=env, pass_fds=tuple(fds), close_fds=True
            )
        finally:
            for r in fds:
                os.close(r)
        signals.child = child
        for sig in [s for s in signals.seen if s in FORWARDED]:
            child.send_signal(
                sig
            )  # one that came while the controller was being started
        child.wait()
        signals.seen.clear()
        if "writer" in relay.lanes:
            _stop_writer(home, drain, signals, note, at_start=False)
        return _exit_code(child.returncode)
    finally:
        stopped = relay.stop()
        if not stopped.verified:
            note(
                f"memory: the lane relay (pid {relay.pid}) did not end verifiably ({stopped.how})"
            )


def _stop_writer(
    home: Path,
    drain: Callable[[Path], Any],
    signals: _Signals,
    note: Callable[[str], None],
    *,
    at_start: bool,
) -> None:
    """Stop the writer service as a stage end (its pass in flight finishes; owed entries stay owed). An interrupt
    stops the wait: the service still stops after its pass, and the next run that is alone on the home tidies up.
    """
    outcome: dict = {}

    def run() -> None:
        try:
            outcome["report"] = drain(home)
        except Exception as exc:  # noqa: BLE001 - reported, never raised
            outcome["error"] = type(exc).__name__

    worker = threading.Thread(target=run, name="lanes-drain", daemon=True)
    worker.start()
    worker.join(2.0)
    if worker.is_alive() and not signals.seen:
        note(
            "memory: an earlier run's writer is finishing its pass before this run starts, at most 5 minutes "
            "(Ctrl-C starts it now, without the writer)"
            if at_start
            else "memory: letting the writer's pass in flight finish, at most 5 minutes; then it is suspended and "
            "resumes next run (Ctrl-C stops waiting)"
        )
    while worker.is_alive():
        if signals.seen:
            note(
                "memory: not waiting; this run's writer starts once the earlier one has stopped"
                if at_start
                else "memory: not waiting; the writer stops after its pass, and the next run tidies up"
            )
            return
        worker.join(POLL_S)
    if "error" in outcome:
        note(f"memory: the writer service could not be stopped ({outcome['error']})")

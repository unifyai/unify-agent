"""cleanslate-v1 as a system for the benchmark runners that Unify's cells use (continual-arc-baselines'
SystemLearner): the runner renders every message, so the request text the harness receives is,
by construction, the same text Unify receives. The harness never sees a task id beyond what the
message itself says, and adds no prompt of its own about examples or checking.

    --learner cleanslate_systems:CleanslateAppWorld   (Continual-AppWorld, relay route)
    --learner cleanslate_systems:CleanslateARC        (Continual-ARC)

AppWorld: the runner provides APPWORLD_RELAY_SOCKET (its relay to the task's world). The harness
puts a recording gateway in front of it (cleanslate.world), so the workspace reaches the world only
through that one socket. The task finishes with apis.supervisor.complete_task(...), which the
harness gates once for smells before it reaches the world.
ARC: the harness delivers a value. A list of lists becomes {"action": "submit", "grid": ...}; a dict
with an "action" is passed on as is. The reply's last line is that JSON, as the ARC protocol asks.

Model calls go to the runner's tracking proxy (ARC_LLM_BASE_URL, key in ARC_LLM_API_KEY), so the
runner's own journal records the spend exactly as for Unify's cells.

Not verified offline: that bwrap and the user scope work inside the runner's outer sandbox
(nested user namespaces), and the runner's behaviour around this class. Tests use a stand-in base.
"""
from __future__ import annotations

import json
import shutil
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cleanslate import Agent, Caps, ChatClient, CostLedger, ProcedureStore  # noqa: E402
from cleanslate import Limits  # noqa: E402
from cleanslate import limits as L  # noqa: E402
from cleanslate.llm import OPENROUTER  # noqa: E402
from cleanslate.sandbox import ConfinementError, Sandbox  # noqa: E402

try:  # the runner's base class, present in the benchmark runtime
    from continual_arc_baselines.systems.base import SystemLearner
    from continual_arc_baselines.systems.base import SystemUnavailable as Unavailable
except ImportError:  # offline tests: the same lifecycle surface, nothing else
    class Unavailable(RuntimeError):
        pass

    class SystemLearner:  # type: ignore[no-redef]
        name = "system"

        def __init__(self, **_: Any):
            self.workdir, self.environ, self.stats = Path(".workbench") / "system", {}, {"sessions": 0, "wipes": 0}

        def attach(self, workdir, environ, tracker=None):
            self.workdir = Path(workdir)
            self.workdir.mkdir(parents=True, exist_ok=True)
            self.environ = dict(environ)


class _Cleanslate(SystemLearner):
    name = "cleanslate"
    world = False

    def __init__(self, *, memory: bool = True, model: str = "openai/gpt-6-luna", reasoning_effort: str | None = "low",
                 max_task_usd: str = "0.50", max_task_calls: int = 200, max_wall_s: float = 900.0,
                 cell_timeout: float = 60.0, model_factory=None, limits_profile: str = "worker",
                 dedicated_user: str | None = None, **kwargs: Any):
        super().__init__(**kwargs)
        # the runner validates `model` and `reasoning_effort` against its pinned catalogue (run.py); `stateless`
        # (the wiped arm) arrives through **kwargs to the base class
        self.memory, self.model_name, self.effort = memory, model, reasoning_effort
        self.caps = Caps(max_usd=Decimal(max_task_usd), max_calls=max_task_calls, max_wall_s=max_wall_s)
        self.cell_timeout, self.model_factory = cell_timeout, model_factory
        self.limits_profile, self.dedicated_user = limits_profile, dedicated_user
        self.messages: list[str] = []   # this instance's messages from the runner
        self.replies: list[str] = []
        self.last_result = None

    # -- paths and parts
    @property
    def home(self) -> Path:
        return Path(self.workdir) / "cleanslate"

    def _model(self):
        if self.model_factory:
            return self.model_factory()
        return ChatClient(model=self.model_name, effort=self.effort,
                          base_url=self.environ.get("ARC_LLM_BASE_URL") or OPENROUTER,
                          key_env="ARC_LLM_API_KEY", environ=self.environ)

    # -- lifecycle (SystemLearner)
    def start(self) -> None:
        """Set the resource-limit profile (worker by default: memory, pids and cpu verified, or a dedicated
        user) and prove it with one confined workspace before the first message. Refuses otherwise."""
        L.set_default(Limits.worker(self.dedicated_user) if self.limits_profile == "worker" else Limits.local())
        self.home.mkdir(parents=True, exist_ok=True)
        probe = self.home / "preflight"
        probe.mkdir(exist_ok=True)
        try:
            with Sandbox(str(probe)) as sb:
                if sb.run("1 + 1").get("last") != "2":
                    raise ConfinementError("the preflight workspace did not run code")
                self.limits_report = sb.limits_report
        except (ConfinementError, L.LimitError) as exc:
            raise Unavailable(f"cleanslate cannot start confined here: {exc}") from None
        finally:
            shutil.rmtree(probe, ignore_errors=True)
        self.ledger = CostLedger(path=str(self.home / "cost-record.jsonl"))

    def stop(self) -> None:
        pass

    def new_session(self) -> None:
        self.messages, self.replies = [], []

    def wipe_state(self) -> None:
        shutil.rmtree(self.home, ignore_errors=True)

    def describe(self) -> dict[str, Any]:
        return {"system": self.name, "memory": self.memory, "model": self.model_name, "effort": self.effort,
                "caps": {k: str(v) for k, v in vars(self.caps).items()}, "limits_profile": self.limits_profile,
                "limits": getattr(self, "limits_report", None)}

    # -- one message
    def ask(self, message: str, *, reprompt: bool = False) -> str:
        if not getattr(self, "ledger", None):
            self.start()
        self.messages.append(message)
        request = message if len(self.messages) == 1 else self._conversation()
        ws = self.home / "work"
        if len(self.messages) == 1:  # a new instance: a fresh working folder; the procedure store persists
            shutil.rmtree(ws, ignore_errors=True)
        ws.mkdir(parents=True, exist_ok=True)
        store = ProcedureStore(str(self.home / "procedures.json")) if self.memory else None
        agent = Agent(self._model(), store, ledger=self.ledger, caps=self.caps, cell_timeout=self.cell_timeout,
                      world_upstream=self.environ.get("APPWORLD_RELAY_SOCKET") if self.world else None)
        res = self.last_result = agent.solve(request, str(ws), label=self.name)
        reply = self.reply(res)
        self.replies.append(reply)
        return reply

    def _conversation(self) -> str:
        """Later messages of the same instance: the runner's messages and this system's replies so far,
        in order, as one request (the conversation is the request)."""
        parts = []
        for i, m in enumerate(self.messages):
            parts.append(m)
            if i < len(self.replies):
                parts.append(f"[your reply]\n{self.replies[i]}")
        return "\n\n".join(parts)

    def reply(self, res) -> str:
        raise NotImplementedError


class CleanslateAppWorld(_Cleanslate):
    name = "cleanslate-appworld"
    world = True

    def reply(self, res) -> str:
        if res.delivered:
            return "Done: the task was completed with apis.supervisor.complete_task."
        return "Not finished: " + (res.events[-1] if res.events else "no completion call was made")


class CleanslateARC(_Cleanslate):
    name = "cleanslate-arc"

    def reply(self, res) -> str:
        value = res.answer if res.delivered else None
        if isinstance(value, list) and value and all(isinstance(r, list) for r in value):
            action = {"action": "submit", "grid": value}
        elif isinstance(value, dict) and "action" in value:
            action = value
        else:
            action = {"action": "finish"}  # nothing delivered: the runner refuses finish and asks again
        return json.dumps(action)

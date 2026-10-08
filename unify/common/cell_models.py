"""Which LLM endpoints model-written code may call: one choke point.

A cell reaches the harness's key-routed LLM calls through three globals:
``query_llm(..., model=...)``, ``list_llms()`` and ``unillm``. In the
sandboxed worker each is a proxy served by the harness, so a call a cell makes
with ``model="anthropic/claude-opus-4.6@openrouter"`` runs here, with the
harness's keys, whatever model the session was configured with.
``UNIFY_CELL_LLM_MODELS`` (unify/settings.py) is the deployment policy for
those calls, and :func:`check_model` is the one place it is enforced:

* empty (the default): only the session's configured model;
* ``any``: every endpoint (a labelled best-setup track);
* a comma list of ``model@provider`` endpoints: those, besides the session's.

The session's configured model is the current act LLM profile's model when it
names one, and what ``model=None`` resolves to (``resolve_default_model()``:
the assistant's default model, else ``UNIFY_MODEL``). Both are allowed: the
first drives the act, the second is what an omitted ``model=`` uses.

Only the cell's globals use this module. ``unify.common.reasoning.query_llm``
and ``list_llms``, ``new_llm_client`` and ``unillm`` itself are unchanged, so
harness-side calls (the actor's loop, reviews, compression, memory passes that
name a different model by design) are never restricted by it.

Endpoints are compared as unillm resolves them: ``<id>@openrouter`` by its
OpenRouter id (``openai/gpt-6-luna@openrouter`` and
``openrouter/openai/gpt-6-luna@openrouter`` are one model), and any other
provider's ``model@provider`` by the alias its registry maps it to. Nothing
here makes a network call.
"""

from __future__ import annotations

import functools
import inspect
from typing import Any, Callable, FrozenSet, Iterable, List, Optional

from unify.common.tool_errors import ToolInputError

__all__ = [
    "CellModelRefused",
    "CellAttributeRefused",
    "allowed_models",
    "check_model",
    "configured_models",
    "normalize_endpoint",
    "cell_query_llm",
    "cell_list_llms",
    "CellUnillm",
    "cell_unillm",
]

_SETTING = "UNIFY_CELL_LLM_MODELS"
_OPENROUTER = "openrouter"


class CellModelRefused(ToolInputError, PermissionError):
    """A cell named an LLM endpoint ``UNIFY_CELL_LLM_MODELS`` does not allow."""


class CellAttributeRefused(ToolInputError, AttributeError):
    """A ``unillm`` attribute the cell facade does not expose.

    Also an ``AttributeError``, so a probe with a default
    (``getattr(unillm, name, None)``) reads it as missing.
    """


# -- policy -------------------------------------------------------------------


def _policy() -> str:
    from unify.settings import SETTINGS

    return str(getattr(SETTINGS, _SETTING, "") or "").strip()


def _allows_any(policy: Optional[str] = None) -> bool:
    return (_policy() if policy is None else policy).lower() == "any"


def normalize_endpoint(endpoint: str) -> str:
    """*endpoint* in the form unillm resolves it to, for comparison only.

    ``<id>@openrouter`` becomes ``openrouter/<id>@openrouter`` (unillm's
    ``openrouter_model``, which leaves an id already so prefixed alone); any
    other ``model@provider`` becomes ``<registry alias>@provider`` when the
    registry knows it. Anything else (no ``@``, an unknown endpoint) is
    compared as written, so it matches only itself.
    """
    text = str(endpoint).strip()
    model, sep, provider = text.rpartition("@")
    if not sep or not model or not provider:
        return text
    try:
        from unillm.endpoints.utils import (
            _MODEL_ALIAS_MAP,
            ensure_endpoints_imported,
            openrouter_model,
        )
    except ImportError:  # pragma: no cover - unillm is a dependency
        return text
    if provider == _OPENROUTER:
        return f"{openrouter_model(model)}@{provider}"
    # The registry lookup ``get_model_alias`` does for a non-OpenRouter
    # provider, without its error for an endpoint the registry lacks.
    ensure_endpoints_imported()
    alias = _MODEL_ALIAS_MAP.get(text)
    return f"{alias}@{provider}" if alias else text


def configured_models() -> List[str]:
    """The session's configured model(s), as written: the act profile's model
    (when it names one) and what ``model=None`` resolves to."""
    from unify.common.act_llm_profiles import CURRENT_ACT_LLM_PROFILE
    from unify.common.llm_client import resolve_default_model

    models: List[str] = []
    try:
        profile_model = CURRENT_ACT_LLM_PROFILE.get().model
    except LookupError:
        profile_model = None
    if profile_model:
        models.append(profile_model)
    default_model, _effort = resolve_default_model()
    if default_model and default_model not in models:
        models.append(default_model)
    return models


def allowed_models() -> Optional[List[str]]:
    """The endpoints a cell may name, as written; ``None`` means any."""
    policy = _policy()
    if _allows_any(policy):
        return None
    models = configured_models()
    for endpoint in policy.split(","):
        endpoint = endpoint.strip()
        if endpoint and endpoint not in models:
            models.append(endpoint)
    return models


def _allowed_keys(models: Iterable[str]) -> FrozenSet[str]:
    return frozenset(normalize_endpoint(m) for m in models)


def is_allowed(model: str) -> bool:
    models = allowed_models()
    if models is None:
        return True
    return normalize_endpoint(model) in _allowed_keys(models)


def check_model(
    model: Any,
    *,
    where: str = "query_llm",
    hint: str = "omit model= to use it",
) -> None:
    """Refuse *model* unless ``UNIFY_CELL_LLM_MODELS`` allows a cell to use it.

    ``None`` is always allowed: it is the session's configured model.
    """
    if model is None or _allows_any():
        return
    if not isinstance(model, str):
        raise CellModelRefused(
            f"{where} takes a model as a 'model@provider' string, not "
            f"{type(model).__name__}",
            received={"model": model},
        )
    if is_allowed(model):
        return
    session = configured_models()
    shown = session[0] if session else "the configured model"
    extra = [m for m in (allowed_models() or []) if m not in session]
    also = f" (or {', '.join(extra)})" if extra else ""
    raise CellModelRefused(
        f"{where} can only use the session's model {shown}{also} here "
        f"({_SETTING}); {hint}",
        suggestion=hint,
        received={"model": model},
    )


# -- the cell's query_llm and list_llms -------------------------------------

#: Keyword names under which a model could be chosen past ``model=``.
_MODEL_KEYS = ("model", "endpoint")


def _check_kwargs(kwargs: Optional[dict], where: str) -> None:
    if not isinstance(kwargs, dict):
        return
    for key in _MODEL_KEYS:
        if key in kwargs:
            check_model(kwargs[key], where=where)


def _cell_query_llm() -> Callable[..., Any]:
    from unify.common import reasoning

    signature = inspect.signature(reasoning.query_llm)

    @functools.wraps(reasoning.query_llm)
    async def query_llm_for_cells(*args: Any, **kwargs: Any) -> Any:
        bound = signature.bind_partial(*args, **kwargs)
        check_model(bound.arguments.get("model"), where="query_llm")
        _check_kwargs(bound.arguments.get("client_kwargs"), "query_llm")
        _check_kwargs(bound.arguments.get("generate_kwargs"), "query_llm")
        # Looked up per call, as the cell's global was bound per sandbox.
        return await reasoning.query_llm(*args, **kwargs)

    return query_llm_for_cells


def _cell_list_llms() -> Callable[..., Any]:
    from unify.common import reasoning

    signature = inspect.signature(reasoning.list_llms)

    @functools.wraps(reasoning.list_llms)
    def list_llms_for_cells(*args: Any, **kwargs: Any) -> List[str]:
        endpoints = reasoning.list_llms(*args, **kwargs)
        models = allowed_models()
        if models is None:
            return endpoints
        keys = _allowed_keys(models)
        listed = [e for e in endpoints if normalize_endpoint(e) in keys]
        # An allowed endpoint the registry does not list is still routable
        # (any OpenRouter id is), so it is listed too, under its provider.
        try:
            provider = signature.bind_partial(*args, **kwargs).arguments.get(
                "provider",
            )
        except TypeError:
            provider = None
        seen = {normalize_endpoint(e) for e in listed}
        for model in models:
            if normalize_endpoint(model) in seen:
                continue
            if provider and not model.endswith(f"@{provider}"):
                continue
            listed.append(model)
            seen.add(normalize_endpoint(model))
        return listed

    return list_llms_for_cells


cell_query_llm = _cell_query_llm()
cell_list_llms = _cell_list_llms()


# -- the cell's unillm ---------------------------------------------------------

#: Client methods that change the endpoint a client calls.
_ENDPOINT_SETTERS = ("set_endpoint",)


class CellLLMClient:
    """A unillm client a cell built, held so it keeps its allowed endpoint.

    Every public attribute of the client is reachable except a change of
    endpoint to one ``UNIFY_CELL_LLM_MODELS`` does not allow; a call that
    returns a client (``copy()``, ``to_sync_client()``, a setter's ``self``)
    returns it held the same way.
    """

    __slots__ = ("_client",)

    def __init__(self, client: Any) -> None:
        object.__setattr__(self, "_client", client)

    def __repr__(self) -> str:
        return repr(object.__getattribute__(self, "_client"))

    def __dir__(self) -> List[str]:
        client = object.__getattribute__(self, "_client")
        return [n for n in dir(client) if not n.startswith("_")]

    def __setattr__(self, name: str, value: Any) -> None:
        raise CellAttributeRefused(
            f"unillm client attribute {name!r} cannot be set from a cell; use "
            "its set_* methods",
        )

    def _rehold(self, out: Any) -> Any:
        """A setter's ``self`` back as this holder; another client held anew."""
        if out is object.__getattribute__(self, "_client"):
            return self
        return _hold(out)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        client = object.__getattribute__(self, "_client")
        value = getattr(client, name)
        if not callable(value):
            return _hold(value)
        if name in _ENDPOINT_SETTERS:

            @functools.wraps(value)
            def set_endpoint(endpoint: Any, *args: Any, **kwargs: Any) -> Any:
                check_model(
                    endpoint,
                    where=f"unillm client .{name}()",
                    hint="keep the client's endpoint",
                )
                return self._rehold(value(endpoint, *args, **kwargs))

            return set_endpoint

        @functools.wraps(value)
        def call(*args: Any, **kwargs: Any) -> Any:
            _check_kwargs(kwargs, f"unillm client .{name}()")
            out = value(*args, **kwargs)
            if inspect.isawaitable(out):
                return _held_awaitable(out)
            return self._rehold(out)

        return call


async def _held_awaitable(awaitable: Any) -> Any:
    return _hold(await awaitable)


def _is_client(value: Any) -> bool:
    try:
        from unillm.clients.uni_llm import _UniClient
    except ImportError:  # pragma: no cover - unillm is a dependency
        return False
    return isinstance(value, _UniClient)


def _hold(value: Any) -> Any:
    if _is_client(value):
        check_model(getattr(value, "endpoint", None), where="unillm client")
        return CellLLMClient(value)
    return value


class CellUnillm:
    """The ``unillm`` a cell sees: client construction for allowed endpoints only.

    ``unillm.AsyncUnify(endpoint, ...)`` and ``unillm.Unify(endpoint, ...)``
    build the module's own clients, refused for an endpoint
    ``UNIFY_CELL_LLM_MODELS`` does not allow; nothing else of the module is
    exposed (its settings, keys, caches and event hooks stay the harness's).
    """

    __slots__ = ()

    #: The public names a cell may read, each a client constructor.
    EXPOSED = ("AsyncUnify", "Unify")

    def __repr__(self) -> str:
        return "<unillm: client constructors for the session's model>"

    def __dir__(self) -> List[str]:
        return list(self.EXPOSED)

    @staticmethod
    def _build(kind: str, endpoint: Any, kwargs: dict) -> CellLLMClient:
        import unillm

        if endpoint is None:
            session = configured_models()
            raise CellModelRefused(
                f"unillm.{kind}() needs an endpoint; use the session's model "
                f"{session[0] if session else ''}".rstrip(),
            )
        check_model(
            endpoint,
            where=f"unillm.{kind}()",
            hint="pass that endpoint, or use query_llm(...) without model=",
        )
        return CellLLMClient(getattr(unillm, kind)(endpoint, **kwargs))

    def AsyncUnify(  # noqa: N802
        self,
        endpoint: Optional[str] = None,
        **kwargs: Any,
    ) -> CellLLMClient:
        """Build ``unillm.AsyncUnify(endpoint, ...)`` for an allowed endpoint."""
        return self._build("AsyncUnify", endpoint, kwargs)

    def Unify(  # noqa: N802
        self,
        endpoint: Optional[str] = None,
        **kwargs: Any,
    ) -> CellLLMClient:
        """Build ``unillm.Unify(endpoint, ...)`` for an allowed endpoint."""
        return self._build("Unify", endpoint, kwargs)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        raise CellAttributeRefused(
            f"unillm.{name} is not available to cell code: a cell's unillm "
            f"exposes only {', '.join(self.EXPOSED)} (for the session's model, "
            f"{_SETTING}); use query_llm(...) for one-shot calls",
        )

    def __setattr__(self, name: str, value: Any) -> None:
        raise CellAttributeRefused("unillm attributes cannot be set from a cell")


#: The ``unillm`` global of every cell (unify/function_manager/execution_env.py).
cell_unillm = CellUnillm()

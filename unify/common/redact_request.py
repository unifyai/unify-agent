"""Remove credentials from a copy of an LLM request before anything publishes it.

unillm hands every listener the request it sent (``LLMEvent.request``),
including the transport's ``api_key`` and any authorization header, and a
listener that publishes or persists that dict would carry the provider key
with it. :func:`redact_llm_request` returns a copy with:

- the value of every credential-named field replaced, at any depth:
  ``api_key`` in any case or spelling (``x-api-key``, ``apiKey``,
  ``openai_api_key``, any ``*_api_key``), ``Authorization``,
  ``Proxy-Authorization``, ``Cookie``, ``Set-Cookie`` and a few transport
  credentials LiteLLM accepts (``aws_secret_access_key``, ``client_secret``...);
- the user information of every URL in a string (a user name and password
  before the host's ``@``), leaving ``https://[REDACTED]@host``;
- every provider-key-shaped token in a string (``sk-...``, ``sk-or-...``,
  ``sk-ant-...``, ``AIza...``), whatever field holds it;
- every credential value this process holds (the values of environment
  variables whose name contains KEY, TOKEN, SECRET or PASSWORD, and unillm's
  provider keys), as the session transcripts replace them.

The request passed in is never changed: dicts, lists and tuples are rebuilt and
only immutable leaves are shared, so the request that was sent, which shares
its nested headers and messages with the event's request, stays as it was.
"""

from __future__ import annotations

import re
from typing import Any

__all__ = ["REDACTED", "redact_llm_request", "redact_value"]

REDACTED = "[REDACTED]"

# Field names, lower-cased with every run of non-alphanumerics as "_".
_SECRET_FIELDS = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "proxy_authorization",
        "cookie",
        "set_cookie",
        "aws_secret_access_key",
        "aws_session_token",
        "azure_ad_token",
        "client_secret",
        "vertex_credentials",
    },
)
_SECRET_SUFFIXES = ("_api_key", "_apikey")
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_SEPARATORS = re.compile(r"[^a-z0-9]+")

# scheme://userinfo@ (a password, or a token as the user name).
_URL_USERINFO = re.compile(r"([A-Za-z][A-Za-z0-9+.-]*://)[^/\s@]+@")
# Provider keys by shape: OpenAI (sk-, sk-proj-), OpenRouter (sk-or-v1-),
# Anthropic (sk-ant-...), Google (AIza...).
_KEY_SHAPED = re.compile(
    r"(?<![A-Za-z0-9_-])(?:sk-[A-Za-z0-9_-]{16,}|AIza[0-9A-Za-z_-]{30,})",
)

# A request is JSON-like and shallow; deeper than this is not data to keep.
_MAX_DEPTH = 64


def _field_is_secret(name: Any) -> bool:
    if not isinstance(name, str):
        return False
    norm = _SEPARATORS.sub("_", _CAMEL.sub("_", name).lower()).strip("_")
    return norm in _SECRET_FIELDS or norm.endswith(_SECRET_SUFFIXES)


def _held_secrets() -> list[str]:
    """The credential values this process holds, longest first."""
    try:
        from unify.transcripts import _secret_values

        return [value for _, value in _secret_values()]
    except Exception:
        return []


def _redact_string(text: str, held: list[str]) -> str:
    for value in held:
        if value in text:
            text = text.replace(value, REDACTED)
    text = _URL_USERINFO.sub(lambda m: f"{m.group(1)}{REDACTED}@", text)
    return _KEY_SHAPED.sub(REDACTED, text)


def _redact(value: Any, held: list[str], depth: int) -> Any:
    if depth > _MAX_DEPTH:
        return REDACTED
    if isinstance(value, str):
        return _redact_string(value, held)
    if isinstance(value, dict):
        return {
            key: (
                REDACTED
                if _field_is_secret(key) and item is not None
                else _redact(item, held, depth + 1)
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        items = [_redact(item, held, depth + 1) for item in value]
        return items if isinstance(value, list) else tuple(items)
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump) and not isinstance(value, type):
        # A Pydantic object inside the request: publish its redacted data.
        try:
            return _redact(model_dump(mode="python"), held, depth + 1)
        except Exception:
            return REDACTED
    return value


def redact_value(value: Any) -> Any:
    """*value* with every credential removed (see the module docstring); *value* is unchanged."""
    return _redact(value, _held_secrets(), 0)


def redact_llm_request(request: Any) -> dict[str, Any]:
    """A copy of an LLM *request* dict that carries no credential; *request* is unchanged.

    Anything that is not a dict gives an empty dict.
    """
    if not isinstance(request, dict):
        return {}
    return redact_value(request)

"""Chat client for an OpenAI-compatible endpoint (OpenRouter by default), standard library only.

The route matches the one Unify's everyday office cells use: OpenRouter, `openai/gpt-6-luna`,
reasoning effort `low` (sent as OpenRouter's `reasoning: {"effort": ...}`, as the AppWorld
adapter's ReAct baseline sends it). Usage accounting is requested so the provider reports each
request's charge (`usage.cost`), which the cost record and the lab journal keep as the charge.

The key:
  * is read from this process's environment (`key_env`, default OPENROUTER_API_KEY) at the moment
    of each request, on the worker only; it is never an argument, never stored on the object,
    never logged, and never passed to the workspace (the workspace is started with a cleared
    environment; tests prove it cannot be seen there);
  * error messages are built from the HTTP status and the provider's error text with the key
    removed, never from request headers.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

OPENROUTER = "https://openrouter.ai/api/v1"


class ModelError(RuntimeError):
    pass


class ChatClient:
    takes_timeout = True

    def __init__(self, model: str = "openai/gpt-6-luna", effort: str | None = "low", base_url: str = OPENROUTER,
                 key_env: str = "OPENROUTER_API_KEY", max_tokens: int = 16000, timeout_s: float = 180.0,
                 environ=None):
        self.model, self.effort, self.base_url = model, effort, base_url.rstrip("/")
        self.key_env, self.max_tokens, self.timeout_s = key_env, max_tokens, timeout_s
        self.last_usage: dict = {}
        # a benchmark runner's environment for its system (proxy URL and key), else this process's; kept inside a
        # closure so neither the object's attributes nor their repr ever show the mapping or the key
        source = environ if environ is not None else os.environ

        def getenv(name: str, default: str = "") -> str:
            return source.get(name, default)
        self._getenv = getenv

    def _key(self) -> str:
        key = self._getenv(self.key_env, "")
        if not key:
            raise ModelError(f"{self.key_env} is not set in the harness process")
        return key

    def body(self, messages: list[dict]) -> dict:
        body = {"model": self.model, "messages": messages, "max_tokens": self.max_tokens, "usage": {"include": True}}
        if self.effort not in (None, "", "none"):
            body["reasoning"] = {"effort": self.effort}
        return body

    def complete(self, messages: list[dict], timeout: float | None = None) -> str:
        self.last_usage = {"model": self.model}  # if the request fails, its charge stays unknown
        key = self._key()
        req = urllib.request.Request(f"{self.base_url}/chat/completions", data=json.dumps(self.body(messages)).encode(),
                                     headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                                     method="POST")
        try:
            with urllib.request.urlopen(req, timeout=min(timeout or self.timeout_s, self.timeout_s)) as r:
                data = json.loads(r.read())
        except urllib.error.HTTPError as exc:
            text = exc.read()[:500].decode("utf-8", "replace").replace(key, "[key]")
            raise ModelError(f"HTTP {exc.code}: {text}") from None
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise ModelError(f"transport: {type(exc).__name__}") from None
        finally:
            del key
        u = data.get("usage") or {}
        self.last_usage = {
            "id": data.get("id"), "model": self.model, "served_model": data.get("model"),
            "prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens"),
            "cached_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
            "provider_cost": u.get("cost"),
            "upstream_inference_cost": (u.get("cost_details") or {}).get("upstream_inference_cost")}
        if "error" in data:
            raise ModelError(f"provider error: {str(data['error'])[:300]}")
        try:
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            raise ModelError("response without a message") from None

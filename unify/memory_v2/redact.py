"""Credential redaction applied before anything reaches git or the blob store."""

from __future__ import annotations

import re
from typing import Any, Mapping

# Credentials this process holds outside its environment (memory v2's Sol token).
try:
    from unify.process_secrets import registered_secrets
except ImportError:  # copied into Sol's box as part of memlab, where unify is absent: it holds no secrets there

    def registered_secrets() -> tuple:  # type: ignore[misc]
        return ()


KEY_SHAPED = re.compile(
    r"sk-or-v1-[0-9a-f]{64}"  # OpenRouter
    r"|sk-(?:proj-|ant-)?[A-Za-z0-9_-]{32,}"  # OpenAI / Anthropic style
    r"|AKIA[0-9A-Z]{16}"  # AWS access key id
    # a PEM private-key block (RSA, EC, OPENSSH, ENCRYPTED, plain PKCS#8), whole, or to the end of the text
    # when its END line was cut off: the armour is the format, so the block goes however it is labelled
    r"|-----BEGIN [A-Z0-9 ]{0,40}PRIVATE KEY-----(?s:.*?)(?:-----END [A-Z0-9 ]{0,40}PRIVATE KEY-----|\Z)",
)
_SECRET_NAME_PARTS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")
_MIN_SECRET_LEN = 8

# A credential in a structure an error message may carry: an Authorization header (any scheme), a bearer token,
# or an api key / access token field, in header, kwargs, JSON or repr form. The name stays; the value goes.
CREDENTIAL_STRUCTURE = re.compile(
    r"(?i)(\bauthorization[\"']?\s*[:=]\s*[\"']?)(?:(?:bearer|basic|token)\s+)?[^\s\"',;}]+"
    r"|(\bbearer\s+)[A-Za-z0-9._~+/=-]+"
    r"|(\b(?:x-)?api[_-]?key[\"']?\s*[:=]\s*[\"']?|\b(?:access|auth)[_-]?token[\"']?\s*[:=]\s*[\"']?)[^\s\"',;}&]+",
)


def redact_error(text: str) -> str:
    """Error text as it may be recorded (pass notes, ``errors.jsonl``, logs): registered credentials by value,
    credential structures (:data:`CREDENTIAL_STRUCTURE`) and key-shaped strings (:data:`KEY_SHAPED`) removed.
    """
    for label, value in registered_secrets():
        if len(value) >= _MIN_SECRET_LEN and value in text:
            text = text.replace(value, f"<secret:{label}>")
    text = CREDENTIAL_STRUCTURE.sub(
        lambda m: f"{m.group(1) or m.group(2) or m.group(3)}<redacted>",
        text,
    )
    return KEY_SHAPED.sub("<redacted:key-shaped>", text)


class Redactor:
    def __init__(self, secrets: Mapping[str, str] | None = None) -> None:
        pairs = [
            (k, v)
            for k, v in (secrets or {}).items()
            if v and len(v) >= _MIN_SECRET_LEN
        ]
        self._secrets = sorted(pairs, key=lambda kv: -len(kv[1]))  # longest first
        self.hits = 0

    @classmethod
    def from_environ(cls, environ: Mapping[str, str]) -> "Redactor":
        """The environment's credentials by name, plus every :func:`unify.process_secrets.register_secret` value."""
        secrets = {
            k: v
            for k, v in environ.items()
            if any(p in k.upper() for p in _SECRET_NAME_PARTS)
        }
        for i, (label, value) in enumerate(registered_secrets()):
            secrets[label if label not in secrets else f"{label}.{i}"] = value
        return cls(secrets)

    def text(self, s: str) -> str:
        for label, value in self._secrets:
            if value in s:
                self.hits += s.count(value)
                s = s.replace(value, f"<secret:{label}>")
        s, n = KEY_SHAPED.subn("<redacted:key-shaped>", s)
        self.hits += n
        return s

    def obj(self, o: Any) -> Any:
        if isinstance(o, str):
            return self.text(o)
        if isinstance(o, dict):
            return {
                self.text(k) if isinstance(k, str) else k: self.obj(v)
                for k, v in o.items()
            }
        if isinstance(o, (list, tuple)):
            return [self.obj(v) for v in o]
        return o

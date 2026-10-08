"""Credential redaction applied before anything reaches git or the blob store."""

from __future__ import annotations

import re
from typing import Any, Mapping

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
        return cls(
            {
                k: v
                for k, v in environ.items()
                if any(p in k.upper() for p in _SECRET_NAME_PARTS)
            },
        )

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

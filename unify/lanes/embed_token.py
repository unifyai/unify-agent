"""The embedding lane's token (``UNIFY_EMBED_TOKEN_FD``): read once from an inherited descriptor, as the other lanes'
tokens are, and sent by the embedding client in place of the provider key.

:func:`settle` runs when the settings are built (``unify/settings.py``): it reads the token from the descriptor
named by ``UNIFY_EMBED_TOKEN_FD`` (at most :data:`MAX_BYTES`, within :data:`TIMEOUT_S`, the write end closed by the
launcher), closes it, registers the token for redaction and takes the name out of the environment, so no child
inherits it. A descriptor that is set but unusable refuses every embedding (:func:`token` raises): the client never
falls back to the key. Standard library only.
"""

from __future__ import annotations

import os
import re
import select
import time
from typing import MutableMapping

EMBED_TOKEN_FD = "UNIFY_EMBED_TOKEN_FD"
MAX_BYTES = 4096
TIMEOUT_S = 5.0
_FORM = re.compile(r"[A-Za-z0-9._~-]{16,}")

_TOKEN: str | None = None
_REFUSAL: str | None = None
_TRIED = False


class EmbedTokenRefused(RuntimeError):
    """``UNIFY_EMBED_TOKEN_FD`` was set and could not be read; its text names the rule, never a value."""


def _read(fd: int) -> str:
    data = b""
    try:
        if not os.get_inheritable(fd):
            raise EmbedTokenRefused(
                f"{EMBED_TOKEN_FD} names a descriptor this process did not inherit"
            )
        poller = select.poll()
        poller.register(fd, select.POLLIN)
        deadline = time.monotonic() + TIMEOUT_S
        while True:
            left = deadline - time.monotonic()
            if left <= 0 or not poller.poll(max(1, int(left * 1000))):
                raise EmbedTokenRefused(
                    f"{EMBED_TOKEN_FD}'s descriptor did not reach its end in time"
                )
            chunk = os.read(fd, MAX_BYTES + 1 - len(data))
            if not chunk:
                break
            data += chunk
            if len(data) > MAX_BYTES:
                raise EmbedTokenRefused(
                    f"{EMBED_TOKEN_FD}'s descriptor holds more than {MAX_BYTES} bytes"
                )
    except OSError:
        raise EmbedTokenRefused(
            f"{EMBED_TOKEN_FD}'s descriptor could not be read"
        ) from None
    finally:
        try:
            os.close(fd)
        except OSError:
            pass
    text = data.decode("latin-1")
    text = text[:-1] if text.endswith("\n") else text
    if not _FORM.fullmatch(text):
        raise EmbedTokenRefused(
            f"{EMBED_TOKEN_FD}'s descriptor does not hold a bearer token"
        )
    return text


def settle(environ: MutableMapping[str, str]) -> None:
    """Read the token once (the first call that finds the name), then take every case variant of the name out of
    *environ*."""
    global _TOKEN, _REFUSAL, _TRIED
    names = [k for k in environ if k.upper() == EMBED_TOKEN_FD]
    if names and not _TRIED:
        _TRIED = True
        value = environ[names[0]].strip()
        try:
            if not value.isdigit() or int(value) <= 2:
                raise EmbedTokenRefused(
                    f"{EMBED_TOKEN_FD} must be a descriptor number above 2"
                )
            _TOKEN = _read(int(value))
            from unify.process_secrets import register_secret

            register_secret("UNIFY_EMBED_TOKEN", _TOKEN)
        except EmbedTokenRefused as exc:
            _REFUSAL = str(exc)
    for name in names:
        environ.pop(name, None)


def token() -> str | None:
    """The embedding lane's token, or None when no descriptor was given; raises :class:`EmbedTokenRefused` when one was
    given and could not be used."""
    if _REFUSAL is not None:
        raise EmbedTokenRefused(_REFUSAL)
    return _TOKEN

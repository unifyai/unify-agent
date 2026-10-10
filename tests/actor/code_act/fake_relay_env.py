"""A stand-in for a benchmark adapter's environment global (AppWorld's ``apis``).

The client reaches its environment through a resource under the harness's
``/tmp`` (AppWorld's relay socket); the workspace sandbox gives the worker a
private ``/tmp``, so a copy of the client made inside the worker cannot reach
it. ``relay`` is a module attribute, as ``appworld_client.apis`` is, so the
worker could import it by name.
"""

from __future__ import annotations

import os
from typing import Optional


def process_id() -> str:
    return f"{os.getpid()}@{os.readlink('/proc/self/ns/pid')}"


class _App:
    def __init__(self, client: "RelayClient", name: str) -> None:
        self._client = client
        self.name = name

    def show(self, item: str) -> dict:
        return {"app": self.name, "item": item, **self._client.ping("show")}


class RelayClient:
    def __init__(self, path: str) -> None:
        self.path = path
        self.music = _App(self, "music")

    def ping(self, word: str) -> dict:
        with open(self.path) as fh:  # the "socket": only the harness's /tmp has it
            token = fh.read().strip()
        return {"word": word, "token": token, "ran_in": process_id()}

    def __repr__(self) -> str:
        return "<relay client>"


relay: Optional[RelayClient] = None

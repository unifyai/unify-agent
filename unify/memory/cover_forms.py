"""What a recorded cover can be given as, and the recorded file a work-tree action shows. Reads records only.

:func:`unfit_reason` says why a cover's recorded input cannot be given in an item's declared ``input`` form (the
form each kind of cover can give: :data:`FORMS`), for G2's and :meth:`.gate.Gate.preview`'s structural check.
:func:`blob_sha` names the blob a work-tree action recorded. Nothing here runs an item: the gate runs no code (rev
2.1, :mod:`.gate_policy`).
"""

from __future__ import annotations

import json
from typing import Any

from .episodes import Action

# The input forms each kind of cover can give (:data:`.manifest.INPUT_KINDS`). Shell covers are not checked at all.
FORMS = {
    "tool": ("env", "observation"),
    "worktree": ("path", "text", "bytes"),
    "dialogue": ("observation", "text"),
}


def blob_sha(a: Action) -> str | None:
    """The recorded file a work-tree action shows: a write's after-blob, else its before-blob."""
    r = a.response if isinstance(a.response, dict) else {}
    sha = r.get("blob_after") if a.method == "write" else r.get("blob_before")
    sha = sha or r.get("blob_before") or r.get("blob_after")
    return sha if isinstance(sha, str) else None


def _json_observation(obs: Any) -> bool:
    """Whether *obs* is a JSON observation: a dict or list, or a string holding one."""
    if isinstance(obs, str):
        try:
            obs = json.loads(obs)
        except ValueError:
            return False
    return isinstance(obs, (dict, list))


def unfit_reason(a: Action, input_kind: str | None) -> str | None:
    """Why *a*'s covered input cannot be given in the declared form, or None (also when none is declared)."""
    if input_kind is None:
        return None
    kind = getattr(a, "kind", "tool")
    if input_kind not in FORMS.get(kind, ()):
        return f"declares input {input_kind}, which a {kind} cover cannot give"
    if kind == "dialogue" and input_kind == "text" and not isinstance(a.response, str):
        return "declares input text, which a dialogue cover with a non-text observation cannot give"
    if kind == "tool" and input_kind == "observation" and not _json_observation(a.response):
        return "declares input observation, which a tool cover without a JSON response cannot give"
    return None

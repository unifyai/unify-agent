"""The structural shape of a recorded dialogue observation, for fingerprints (:func:`observation_shape`).

Parsed values (and text that parses as JSON) give their key tree; other text gives the sequence of its line
labels. Values, numbers and free text never appear. Imports only the standard library.
"""

from __future__ import annotations

import json
import re
from typing import Any

_LABEL = re.compile(r"^\s*([^:\n]{1,40}?)\s*:")


def _canon(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str, ensure_ascii=False)


def observation_shape(obs: Any) -> str:
    """A structural summary of one observation, for fingerprints.

    Parsed values (and text that parses as JSON) give their key tree; other text gives the sequence of
    its line labels (the text before a ``:`` within the first 40 characters, else ``_``), with
    consecutive repeats collapsed. Values, numbers and free text never appear.
    """
    from .shapes import tree

    if obs is None:
        return "none"
    if not isinstance(obs, str):
        return "tree:" + _canon(tree(obs))
    s = obs.strip()
    if s[:1] in ("{", "["):
        try:
            return "tree:" + _canon(tree(json.loads(s)))
        except (ValueError, RecursionError):
            pass
    labels: list[str] = []
    for line in s.splitlines():
        if not line.strip():
            continue
        m = _LABEL.match(line)
        lab = (m.group(1).strip() + ":") if m else "_"
        if not labels or labels[-1] != lab:
            labels.append(lab)
    return "text:" + "|".join(labels[:32])

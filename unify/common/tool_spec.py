from dataclasses import dataclass
from typing import Callable, Dict, Union


@dataclass(slots=True)
class ToolSpec:
    """One tool of a loop: the callable the loop runs for it."""

    fn: Callable


def normalise_tools(
    raw: Dict[str, Union[Callable, "ToolSpec"]],
) -> Dict[str, "ToolSpec"]:
    """Return a uniform ``dict[name → ToolSpec]`` from callables or ToolSpec values."""
    return {n: v if isinstance(v, ToolSpec) else ToolSpec(fn=v) for n, v in raw.items()}

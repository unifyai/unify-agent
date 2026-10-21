from dataclasses import dataclass
from typing import Any, Callable, Optional, Dict, Union


# ─────────────────────────────────────────────────────────────────────────────
# 0.  metadata wrapper - lets us attach `max_concurrent` to a tool
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class ToolSpec:
    """Wrap the callable with optional metadata (e.g., max_concurrent)."""

    fn: Callable
    # «None» ⇒ unlimited. Schema visibility is unconditional regardless of
    # this limit (a stable schema beats a minimal one) — a call attempted
    # while saturated is refused at execution time instead.
    max_concurrent: Optional[int] = None
    # Hidden per-loop quota: when set, the tool will only be callable
    # `max_total_calls` times within a single async tool-use loop. The tool
    # stays visible in the schema once exhausted (same rationale as
    # max_concurrent above); an over-quota call is pruned/rejected at
    # execution time instead of hiding the tool.
    max_total_calls: Optional[int] = None
    read_only: Optional[bool] = None
    manager_tool: bool = False
    display_label: Optional[Union[str, Callable[[Dict[str, Any]], str]]] = None

    # Let a ToolSpec be invoked like the underlying callable (nice for tests)
    def __call__(self, *a, **kw):  # pragma: no cover
        return self.fn(*a, **kw)


def normalise_tools(
    raw: Dict[str, Union[Callable, "ToolSpec"]],
) -> Dict[str, "ToolSpec"]:
    """Return a uniform ``dict[name → ToolSpec]`` from callables or ToolSpec values."""
    out: Dict[str, ToolSpec] = {}
    for n, v in raw.items():
        if isinstance(v, ToolSpec):
            if v.read_only is None or not v.manager_tool:
                fn_ro = getattr(v.fn, "_tool_spec_read_only", None)
                fn_mt = getattr(v.fn, "_tool_spec_manager_tool", False)
                if (v.read_only is None and fn_ro is not None) or (
                    not v.manager_tool and fn_mt
                ):
                    v = ToolSpec(
                        fn=v.fn,
                        max_concurrent=v.max_concurrent,
                        max_total_calls=v.max_total_calls,
                        read_only=v.read_only if v.read_only is not None else fn_ro,
                        manager_tool=v.manager_tool or fn_mt,
                        display_label=v.display_label,
                    )
            out[n] = v
        else:
            out[n] = ToolSpec(
                fn=v,
                read_only=getattr(v, "_tool_spec_read_only", None),
                manager_tool=getattr(v, "_tool_spec_manager_tool", None),
            )
    return out


def read_only(fn: Callable) -> Callable:
    """Mark a tool as read-only; eligible for semantic-cache re-execution."""
    setattr(fn, "_tool_spec_read_only", True)
    return fn


def manager_tool(fn: Callable) -> Callable:
    """Mark a tool as a manager tool; used to call other tools."""
    setattr(fn, "_tool_spec_manager_tool", True)
    return fn



from typing import TYPE_CHECKING
from importlib import import_module

__all__ = ["CodeActActor"]


def __getattr__(name: str):
    if name == "CodeActActor":
        return import_module("unify.actor.code_act_actor").CodeActActor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + __all__)


if TYPE_CHECKING:
    from .code_act_actor import CodeActActor

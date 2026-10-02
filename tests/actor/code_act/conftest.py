import asyncio

from unittest.mock import MagicMock

from unify.function_manager.function_manager import FunctionManager

_FM_METHOD_NAMES = (
    "search_functions",
    "filter_functions",
    "list_functions",
    "add_functions",
    "delete_function",
    "reconcile_dependencies",
    # added to the actor's tools only under UNIFY_FUNCTION_PATCH /
    # UNIFY_FUNCTION_CASES, so a run with those switches on can build them
    "patch_function",
    "retire_case",
)


def make_fm_mock() -> MagicMock:
    """Create a FunctionManager MagicMock compatible with ``methods_to_tool_dict``.

    ``methods_to_tool_dict`` derives tool keys from ``fn.__self__.__class__``
    and ``fn.__name__``, which plain ``MagicMock`` methods lack. Using
    ``spec=FunctionManager`` gives the correct MRO so the canonical class name
    resolves to ``FunctionManager``, and we set ``__name__`` / ``__self__`` on
    each method so they look like real bound methods.
    """
    fm = MagicMock(spec=FunctionManager)
    for name in _FM_METHOD_NAMES:
        method = getattr(fm, name)
        method.__name__ = name
        method.__self__ = fm
    return fm


async def wait_for_turn_completion(task, initial_history_len, timeout=30):
    """
    Wait for the agent to process an interjection and enter an idle state.

    An idle state is detected when the last message is from the assistant and contains
    no tool calls, indicating it's waiting for the next command.
    """
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout

    while loop.time() < deadline:
        if len(task.get_history()) > initial_history_len:
            last_message = task.get_history()[-1]
            if last_message.get("role") == "assistant" and not last_message.get(
                "tool_calls",
            ):
                return
        await asyncio.sleep(0.5)

    raise AssertionError("Timed out waiting for turn completion")

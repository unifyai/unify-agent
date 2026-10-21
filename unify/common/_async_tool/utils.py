import copy
import inspect
import json
import textwrap

from unillm.logger import _expand_string_newlines


async def maybe_await(obj):
    """Return *obj* if it is a value, or `await` and return its result if it is
    an awaitable."""
    if inspect.isawaitable(obj):
        return await obj
    return obj


def try_parse_json(value):
    """Return JSON-parsed value when `value` is a JSON string; otherwise return value unchanged."""
    try:
        if isinstance(value, str):
            return json.loads(value)
    except Exception:
        pass
    return value


def _add_code_delimiters(args: dict) -> None:
    """Add Python markdown fences around the code field."""
    code = args.get("code", "")
    if not code:
        return
    args["code"] = f"\n```python\n{textwrap.dedent(code).strip()}\n```"


def format_json_for_log(body: dict) -> str:
    """Human-readable JSON for terminal logging, with escaped newlines in
    string values expanded so prompts, markdown and code render naturally."""
    return _expand_string_newlines(
        json.dumps(body, indent=4, default=str, ensure_ascii=False),
    )


def format_llm_response_for_log(msg: dict) -> str:
    """An assistant message for terminal logging: stringified tool-call
    arguments are parsed for pretty-printing and execute_code blocks get
    markdown fences."""
    msg = copy.deepcopy(msg)
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function", {})
        fn["arguments"] = try_parse_json(fn.get("arguments"))
        if fn.get("name") == "execute_code" and isinstance(fn.get("arguments"), dict):
            _add_code_delimiters(fn["arguments"])
    return format_json_for_log(msg)

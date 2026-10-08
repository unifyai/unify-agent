"""The ``memory`` helper's addition under ``UNIFY_MEMORY_V2_OBSERVATIONS=on``, as source text.

:data:`SOURCE` is appended to the export's ``memory.py`` (:mod:`.memory_helper`) only while the switch is on
(:func:`.catalogue.helper_bytes`), so with it off the helper is the frozen file byte for byte. It adds
``observations()`` and ``observation()``, the catalogue line pointing at them and, for a function whose input
form is ``observation``, a line in its description. It runs inside the helper's module (its names: ``os``,
``json``, ``Any``, ``_HERE``, ``_Text``, ``_entry``, ``_forms_line``, ``describe``); standard library only.
"""

SOURCE = r'''
# --- UNIFY_MEMORY_V2_OBSERVATIONS=on (appended by the harness) ------------------------------------------

__doc__ = (__doc__ or "") + """
This library's harness also keeps the counterpart's messages so far in this request
(``.memory/observations.json``, rewritten before each cell): the request first, then each message the
counterpart sent back (an environment's observation, a person's follow-up), redacted and capped exactly as the
recorder keeps an observation. ``memory.observation()`` returns the latest and ``memory.observations()`` all of
them, so a function whose input form is ``observation`` takes the current one as it was built on recorded ones,
without the cell pasting text.
"""

OBSERVATIONS_FILE = ".memory/observations.json"
_OBSERVATIONS_LINE = (
    "`memory.observation()` is the latest message the counterpart sent in this request (the request itself "
    "before any reply), as an `observation` input takes it; `memory.observations()` lists them all, oldest "
    "first.\n"
)


def observations() -> list:
    """The counterpart's messages so far in this request, oldest first: the request, then each message the
    counterpart sent back, each as an ``observation`` input takes it (parsed JSON or text); ``[]`` before the
    harness has written any."""
    path = os.path.join(_HERE, OBSERVATIONS_FILE)
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        return []
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            f"memory: the observations {path} cannot be read ({type(exc).__name__})",
        ) from exc
    got = data.get("messages") if isinstance(data, dict) else None
    return list(got) if isinstance(got, list) else []


def observation() -> Any:
    """The counterpart's latest message in this request (the request itself before any reply), as an
    ``observation`` input takes it; ``None`` before the harness has written any."""
    got = observations()
    return got[-1] if got else None


_frozen_forms_line = _forms_line
_frozen_describe = describe


def _forms_line(cat: dict) -> str:  # noqa: F811 - extends the helper's own
    line = _frozen_forms_line(cat)
    return line + _OBSERVATIONS_LINE if line else line


def describe(fn_or_name: Any) -> str:  # noqa: F811 - extends the helper's own
    text = _frozen_describe(fn_or_name)
    if _entry(fn_or_name).get("input") == "observation":
        text = _Text(text + "The current one: memory.observation().\n")
    return text
'''

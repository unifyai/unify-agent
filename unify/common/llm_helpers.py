from __future__ import annotations
import base64
import binascii
import json
import inspect
import random
import string
from enum import Enum
from pydantic import BaseModel
from typing import (
    List,
    Dict,
    Union,
    Optional,
    Any,
    get_type_hints,
    get_origin,
    get_args,
)


def short_id(length=4):
    alphabet = string.ascii_lowercase + string.digits  # base36
    return "".join(random.choice(alphabet) for _ in range(length))


TYPE_MAP = {str: "string", int: "integer", float: "number", bool: "boolean"}

# ─────────────────────────────────────────────────────────────────────────────
# Image-handling helpers
# ─────────────────────────────────────────────────────────────────────────────


def _is_b64_image(value: str) -> bool:
    """True when *value* is base64 whose decoded head carries a PNG or JPEG magic.

    Payloads use "image" keys for plenty of non-image strings (asset URNs,
    URLs, schema fragments); promoting those into image_url content blocks
    ships garbage that providers reject as invalid base64. Magic bytes are
    the only reliable discriminator, so anything undecodable or unrecognized
    stays in the textual payload instead.
    """
    head = value[:44]
    if len(head) < 8:
        return False
    try:
        raw = base64.b64decode(head, validate=True)
    except (binascii.Error, ValueError):
        return False
    return raw[:2] == b"\xff\xd8" or raw[:8] == b"\x89PNG\r\n\x1a\n"


# Recursively collect *every* base-64 image that lives under an "image" key
def _collect_images(obj, acc: list[str]) -> None:
    if isinstance(obj, dict):
        if (
            "image" in obj
            and isinstance(obj["image"], str)
            and _is_b64_image(obj["image"])
        ):
            acc.append(obj["image"])
        for v in obj.values():
            _collect_images(v, acc)
    elif isinstance(obj, list):
        for v in obj:
            _collect_images(v, acc)


# Deep-copy *obj* **without** any "image" keys so we can still present the
# textual part of a tool result next to the promoted image blocks.
def _strip_image_keys(obj):
    """
    Recursively deep-copies an object, removing any base64 image data.
    This handles images in `{"image": "b64..."}` format and also the
    more complex `[{"type": "image_url", ...}]` format in tool content.
    """
    if isinstance(obj, dict):
        # Special handling for tool messages with list content
        if obj.get("role") == "tool" and isinstance(obj.get("content"), list):
            new_content = [
                item
                for item in obj["content"]
                if not (isinstance(item, dict) and item.get("type") == "image_url")
            ]
            # If only text remains, flatten content to a simple string for clarity
            if len(new_content) == 1 and new_content[0].get("type") == "text":
                obj_copy = {k: v for k, v in obj.items() if k != "content"}
                obj_copy["content"] = new_content[0].get("text", "")
                return obj_copy
            else:  # Return a new dict with the filtered content
                obj_copy = {k: v for k, v in obj.items() if k != "content"}
                obj_copy["content"] = new_content
                return obj_copy

        # Standard recursive dictionary copy; only actual base64 images are
        # stripped — non-image "image" values stay in the textual view since
        # _collect_images never promotes them to blocks.
        return {
            k: _strip_image_keys(v)
            for k, v in obj.items()
            if not (k == "image" and isinstance(v, str) and _is_b64_image(v))
        }
    elif isinstance(obj, list):
        return [_strip_image_keys(v) for v in obj]
    else:
        return obj


def _dumps(
    obj: Any,
    idx: List[Union[str, int]] = None,
    indent: int = None,
    context: dict | None = None,
) -> Any:
    base = False
    if idx is None:
        base = True
        idx = list()
    if isinstance(obj, BaseModel):
        # Thread optional serialization context (ignored by models that don't use it)
        ret = obj.model_dump(mode="json", context=context)
    elif inspect.isclass(obj) and issubclass(obj, BaseModel):
        ret = obj.model_json_schema()
    elif isinstance(obj, dict):
        ret = {k: _dumps(v, idx + ["k"], context=context) for k, v in obj.items()}
    elif isinstance(obj, list):
        ret = [_dumps(v, idx + [i], context=context) for i, v in enumerate(obj)]
    elif isinstance(obj, set):
        # Convert sets to a sorted list for deterministic, JSON-serialisable output
        try:
            ret = sorted(
                _dumps(v, idx + [i], context=context) for i, v in enumerate(sorted(obj))
            )
        except Exception:
            # Fallback: best-effort conversion preserving insertion order where possible
            ret = [
                _dumps(v, idx + [i], context=context) for i, v in enumerate(list(obj))
            ]
    elif isinstance(obj, tuple):
        ret = tuple(_dumps(v, idx + [i], context=context) for i, v in enumerate(obj))
    else:
        ret = obj
    return json.dumps(ret, indent=indent) if base else ret


import re
from textwrap import dedent
from typing import Iterable

# recognised section headings (case-insensitive, colon optional)
_PARAM_SECTIONS = {"args", "arguments", "parameters", "other parameters"}
_OTHER_SECTIONS = {
    "returns",
    "yields",
    "raises",
    "notes",
    "examples",
    "references",
    "see also",
}

# ––– parameter-line pattern that also accepts the "a / b / c : …" variant –––
_PARAM_LINE_RX = re.compile(
    r"""
    ^(?P<indent>\s*)                    # leading spaces
    (?P<names>[^:]+?)                   # everything until " :", *if any*
    (?:\s*:\s*(?P<type>.+))?            # " : type"  ← now OPTIONAL
    $                                   # EOL
    """,
    re.VERBOSE,
)

# dash-only underline used by the NumPy style ("----------")
_DASH_UNDERLINE_RX = re.compile(r"^\s*-{3,}\s*$")


def _strip_hidden_params_from_doc(
    doc: str,
    hidden: set[str] | Iterable[str],
) -> str:
    """Remove parameter blocks for any names in `hidden` from a docstring."""
    hidden = set(hidden)
    if not doc or not hidden:
        return doc

    lines = dedent(doc).splitlines()
    out: list[str] = []

    i = 0
    in_params = False  # are we _currently_ inside a param section?
    skip = False  # are we skipping the current block?
    base_indent = 0  # indent of the "name : type" line we skip

    while i < len(lines):
        ln, stripped = lines[i], lines[i].lstrip()
        lower = stripped.rstrip(":").lower()

        # ───────────────────────────────────────────────────────────────── #
        # 1.  Detect the *start* of a Parameters/Args section
        # ───────────────────────────────────────────────────────────────── #
        if not in_params and lower in _PARAM_SECTIONS:
            in_params = True
            out.append(ln)  # keep the heading
            # keep the NumPy underline if present
            j = i + 1
            if j < len(lines) and _DASH_UNDERLINE_RX.match(lines[j]):
                out.append(lines[j])
                i += 1
            i += 1
            continue

        # ───────────────────────────────────────────────────────────────── #
        # 2.  While inside the section, decide whether to keep or skip
        # ───────────────────────────────────────────────────────────────── #
        if in_params:
            # Heading of some *other* section → we are done with "Parameters"
            if lower in _OTHER_SECTIONS:
                in_params = False
                out.append(ln)
                i += 1
                continue

            # Dash underline that belongs to the *next* section
            if _DASH_UNDERLINE_RX.match(stripped):
                in_params = False
                out.append(ln)
                i += 1
                continue

            # When skipping a hidden param block, check indent continuation
            # BEFORE trying to match a new param line. Description lines are
            # always more indented than the param heading in NumPy style; only
            # when the indent drops back do we leave skip mode and try to
            # match the line as a new parameter.
            if skip:
                indent = len(ln) - len(stripped)
                if indent > base_indent:
                    i += 1  # continuation of hidden param description
                    continue
                skip = False  # indent dropped → may be a new param

            # Parameter definition line
            m = _PARAM_LINE_RX.match(ln)
            if m:
                # the spec allows either "a, b" or "a / b" to list synonyms
                names = {
                    n.strip()
                    for part in m.group("names").split("/")
                    for n in part.split(",")
                }
                if names & hidden:
                    skip = True
                    base_indent = len(m.group("indent"))
                    # do *not* append this very line
                    i += 1
                    continue
                else:
                    skip = False  # keep this parameter

            if not skip:
                out.append(ln)  # normal, unskipped content
            i += 1
            continue

        # ───────────────────────────────────────────────────────────────── #
        # 3.  Normal line outside any param section
        # ───────────────────────────────────────────────────────────────── #
        out.append(ln)
        i += 1

    # ───────────────────────────────────────────────────────────────────── #
    # Second pass: strip lines in Returns / Raises / other sections that
    # reference hidden param names (e.g. "When ``_return_callable=False``").
    # Also strip more-indented continuation lines that follow them.
    # ───────────────────────────────────────────────────────────────────── #
    if hidden:
        out2: list[str] = []
        j = 0
        ref_skip_indent = -1
        while j < len(out):
            ln2 = out[j]
            stripped2 = ln2.lstrip()
            indent2 = len(ln2) - len(stripped2)

            if ref_skip_indent >= 0:
                if stripped2 and indent2 > ref_skip_indent:
                    j += 1
                    continue
                ref_skip_indent = -1

            if any(h in ln2 for h in hidden):
                ref_skip_indent = indent2
                j += 1
                continue

            out2.append(ln2)
            j += 1
        out = out2

    # Collapse runs of >2 blank lines that the removals may have created
    doc_clean = re.sub(r"\n{3,}", "\n\n", "\n".join(out)).rstrip()
    return doc_clean


def _resolve_doc_with_mro_fallback(bound_method) -> str:
    """
    Return a docstring for `bound_method`, falling back to the first ancestor
    in the MRO that defines a docstring for a method with the same name.
    """
    import inspect as _inspect  # local import to avoid polluting global scope

    # 1) Prefer the callable as exposed to the LLM. Runtime wrappers may append
    # routing guidance to the visible docstring while preserving __wrapped__.
    try:
        doc = _inspect.getdoc(bound_method)
        if isinstance(doc, str) and doc.strip():
            return doc.strip()
    except Exception:
        pass

    # 2) Then inspect the underlying implementation.
    try:
        unwrapped = _inspect.unwrap(bound_method)
    except Exception:
        unwrapped = bound_method
    try:
        doc = _inspect.getdoc(unwrapped)
        if isinstance(doc, str) and doc.strip():
            return doc.strip()
    except Exception:
        pass

    # 3) MRO fallback for bound methods: find an ancestor method with a docstring
    try:
        name = getattr(unwrapped, "__name__", None) or getattr(
            bound_method,
            "__name__",
            "",
        )
        owner = getattr(getattr(bound_method, "__self__", None), "__class__", None)
        if not name or owner is None:
            return ""
        for base in getattr(owner, "__mro__", ())[1:]:
            try:
                cand = getattr(base, name, None)
            except Exception:
                cand = None
            if cand is None:
                continue
            fn_obj = getattr(cand, "__func__", cand)
            base_doc = _inspect.getdoc(fn_obj)
            if isinstance(base_doc, str) and base_doc.strip():
                return base_doc.strip()
    except Exception:
        pass

    # 3) No doc found
    return ""


def annotation_to_schema(ann: Any) -> Dict[str, Any]:
    """Convert a Python annotation into a JSON Schema fragment (supports Pydantic).

    A string entry in ``typing.Annotated`` metadata is surfaced as the JSON
    Schema ``description`` for the field, so tools can attach a property-level
    description directly on a parameter, e.g.::

        thought: Annotated[str, "One-sentence rationale shown to the user."]

    This is a stronger, more reliable signal to the model than describing the
    argument only in the function docstring.
    """
    description: Optional[str] = None
    origin = get_origin(ann)
    if origin is not None and getattr(origin, "__name__", "") == "Annotated":
        args = get_args(ann)
        ann = args[0]
        for meta in args[1:]:
            if isinstance(meta, str):
                description = meta
                break

    schema = _annotation_to_schema(ann)
    if description and isinstance(schema, dict) and "description" not in schema:
        schema = {**schema, "description": description}
    return schema


def _strip_model_schema_noise(schema: Any) -> Any:
    """Drop pydantic's auto-generated ``title`` and ``default`` annotations.

    Titles restate the property name, and defaults restate optionality that
    the ``required`` list already encodes (worse: a ``default`` on a
    discriminator ``const`` suggests the discriminator may be omitted, which
    the runtime validator rejects). Neither constrains validation, and both
    multiply across dereferenced unions. The walk follows JSON-schema
    structure so property *names* called ``title``/``default`` survive.
    """
    if isinstance(schema, list):
        return [_strip_model_schema_noise(item) for item in schema]
    if not isinstance(schema, dict):
        return schema
    out: Dict[str, Any] = {}
    for key, value in schema.items():
        if key in ("title", "default"):
            continue
        if key == "properties" and isinstance(value, dict):
            out[key] = {
                name: _strip_model_schema_noise(prop) for name, prop in value.items()
            }
        elif key in (
            "items",
            "prefixItems",
            "additionalProperties",
            "anyOf",
            "oneOf",
            "allOf",
            "not",
            "$defs",
            "definitions",
        ):
            out[key] = _strip_model_schema_noise(value)
        else:
            out[key] = value
    return out


def _annotation_to_schema(ann: Any) -> Dict[str, Any]:
    """Inner type→schema conversion. Annotated-description handling lives in the
    public :func:`annotation_to_schema` wrapper."""

    # Unwrap typing.Annotated (defensive — the public wrapper already does this
    # and captures any string description).
    origin = get_origin(ann)
    if origin is not None and getattr(origin, "__name__", "") == "Annotated":
        ann = get_args(ann)[0]
        origin = get_origin(ann)

    # NoneType
    if ann is type(None):
        return {"type": "null"}

    # Primitive scalars
    if ann in TYPE_MAP:
        return {"type": TYPE_MAP[ann]}

    # Builtin container types without typing metadata
    # Treat builtin dict/list/set/tuple as generic JSON containers rather than strings
    # so tools like `images: dict | None` surface to the LLM with the correct types.
    if ann is dict:
        # Unknown key/value types → allow arbitrary properties
        return {"type": "object", "additionalProperties": True}
    if ann is list:
        # Unknown item type → generic array
        return {"type": "array"}
    if ann is set:
        # Sets map naturally to arrays with unique items
        return {"type": "array", "uniqueItems": True}
    if ann is tuple:
        # Tuples as heterogeneous arrays; without metadata fall back to generic array
        return {"type": "array"}

    # Enum
    if isinstance(ann, type) and issubclass(ann, Enum):
        return {"type": "string", "enum": [member.value for member in ann]}

    # Pydantic model — dereference $refs so the schema stays valid when nested
    # as a single tool-parameter property (providers like Parasail compile the
    # whole parameters object as one JSON grammar; nested `#/$defs/...` refs
    # otherwise resolve against the wrong document root).
    if isinstance(ann, type) and issubclass(ann, BaseModel):
        import copy

        import jsonref

        schema = ann.model_json_schema()
        dereferenced = jsonref.loads(json.dumps(schema))
        plain = copy.deepcopy(dereferenced)
        if isinstance(plain, dict):
            plain.pop("$defs", None)
            plain.pop("definitions", None)
        return _strip_model_schema_noise(plain)

    # Dict[K, V]
    if origin is dict or origin is Dict:
        args = get_args(ann)
        if len(args) < 2:
            return {"type": "object", "additionalProperties": True}
        _, value_type = args
        return {
            "type": "object",
            "additionalProperties": annotation_to_schema(value_type),
        }

    # List[T]
    if origin in (list, List):
        (item_type,) = get_args(ann)
        return {"type": "array", "items": annotation_to_schema(item_type)}

    # Union / Optional
    try:
        import types as _types  # local import

        is_union = origin is Union or origin is _types.UnionType
    except Exception:
        is_union = origin is Union

    if is_union:
        sub_schemas = [annotation_to_schema(a) for a in get_args(ann)]
        null_schema = {"type": "null"}
        # An optional parameter has to stay *expressible as absent*. Collapsing
        # `int | None` to a bare integer leaves a model with no way to say "no
        # value", so it supplies a sentinel instead — and a sentinel that means
        # nothing to the receiver (session 0) reads as a real request.
        if len(sub_schemas) == 2 and null_schema in sub_schemas:
            inner = next(s for s in sub_schemas if s != null_schema)
            inner_type = inner.get("type")
            # Widening the existing `type` keeps the schema a single shape, so
            # the description, `items` and `additionalProperties` all survive
            # and no second branch appears for the model to mistake for a
            # serialized alternative.
            if isinstance(inner_type, str):
                return {**inner, "type": [inner_type, "null"]}
            return inner
        return {"anyOf": sub_schemas}

    # `object` and `typing.Any`: any JSON value is valid (empty schema = no
    # constraints). Both must land here rather than on the string fallback below.
    # A `Dict[str, Any]` parameter otherwise compiles to
    # `additionalProperties: {"type": "string"}`, which tells the model every
    # value must be a string — contradicting the callee's real signature and
    # leaving no valid token to emit for a numeric argument.
    if ann is object or ann is Any:
        return {}

    # Fallback
    return {"type": "string"}


def method_to_schema(bound_method, tool_name: str) -> dict:
    """Convert a callable into an OpenAI-compatible function-tool schema named *tool_name*.

    Parameters whose names start with ``_`` are internal plumbing and never
    reach the model; ``*args`` and ``**kwargs`` are not expressible as fixed
    JSON fields (a ``**kwargs`` callable allows extra keys). The schema is
    non-strict: the tool loop keeps Python defaults, open mappings and
    argument normalisation available at execution time.
    """

    sig = inspect.signature(bound_method)
    # Be robust to unresolved forward references or missing symbols in
    # annotations. If type-hint evaluation fails, fall back to an empty
    # mapping and infer JSON schema types from defaults.
    try:
        # include_extras keeps typing.Annotated metadata intact so that a
        # string description on a parameter (Annotated[str, "..."]) reaches
        # annotation_to_schema and lands on the property schema.
        hints = get_type_hints(bound_method, include_extras=True)
    except Exception:
        hints = {}

    props, required, hidden = {}, [], set()

    # Detect whether the callable accepts **kwargs so we can permit extra keys
    has_var_keyword = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )

    for name, param in sig.parameters.items():
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        if name.startswith("_"):
            hidden.add(name)
            continue  # do NOT surface to the model

        ann = hints.get(name, str)
        props[name] = annotation_to_schema(ann)
        if param.default is inspect._empty:
            required.append(name)

    # ── resolve docstring with MRO fallback, then scrub hidden args ───────────
    raw_doc = _resolve_doc_with_mro_fallback(bound_method) or ""
    cleaned_doc = _strip_hidden_params_from_doc(raw_doc, hidden)

    schema: dict = {
        "type": "function",
        "strict": False,
        "function": {
            "name": tool_name,
            "description": cleaned_doc.strip(),
            "parameters": {
                "type": "object",
                "properties": props,
                "required": required,
            },
        },
    }
    # Allow arbitrary extra keys when the function accepts **kwargs
    if has_var_keyword:
        schema["function"]["parameters"]["additionalProperties"] = True
    return schema



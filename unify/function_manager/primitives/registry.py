"""
ToolSurfaceRegistry: Single source of truth for manager/primitive configuration.

This module centralizes all manager definitions, primitive method discovery,
prompt context generation, and primitive row collection for FunctionManager indexing.

No other module should define manager lists or primitive configurations.
"""

from __future__ import annotations

import hashlib
import inspect
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type, TYPE_CHECKING

from unify.function_manager.hash_utils import stable_hash_for_rows
from unify.function_manager.primitives.scope import PrimitiveScope

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# =============================================================================
# ManagerSpec - Per-manager configuration
# =============================================================================


@dataclass(frozen=True, slots=True)
class ManagerSpec:
    """
    Configuration for a single primitive namespace in the tool surface.

    This is the authoritative specification for each namespace, containing:
    - Identity (alias, class path)
    - Method exclusions
    - Prompt metadata (domain, description)
    - Sandbox namespace root (the top-level key in the sandbox's global_state)
    """

    manager_alias: str
    primitive_class_path: str
    # The top-level key under which this manager's methods appear in a
    # CodeActActor sandbox's global_state.  All managers live under
    # ``"primitives"`` (accessed as ``primitives.<alias>.<method>``).
    # Used by construct_sandbox_root() to map dotted depends_on entries
    # back to the class that provides them.
    sandbox_root: str = "primitives"
    excluded_methods: frozenset[str] = field(default_factory=frozenset)
    domain: str = ""
    description: str = ""


# =============================================================================
# Common excluded methods (applies to every primitive namespace)
# =============================================================================

_COMMON_EXCLUDED_METHODS: frozenset[str] = frozenset(
    {
        # State management / lifecycle
        "clear",
        # Internal helpers
        "add_tools",
        "get_tools",
    },
)


# =============================================================================
# Canonical Primitive Registry (SINGLE SOURCE OF TRUTH)
# =============================================================================

_MANAGER_SPECS: tuple[ManagerSpec, ...] = (
    ManagerSpec(
        manager_alias="actor",
        primitive_class_path="unify.actor.environments.actor._ActorRunner",
        excluded_methods=frozenset(),
        domain="Actor Delegation",
        description="Spawn focused sub-actors for isolated multi-step sub-tasks",
    ),
)

# Build lookup dicts for fast access
_MANAGER_BY_ALIAS: Dict[str, ManagerSpec] = {
    spec.manager_alias: spec for spec in _MANAGER_SPECS
}

# Reverse mapping: primitive_class_path -> manager_alias (for deriving alias from stored primitive_class)
_CLASS_PATH_TO_ALIAS: Dict[str, str] = {
    spec.primitive_class_path: spec.manager_alias for spec in _MANAGER_SPECS
}

# sandbox_root -> [specs]: maps each sandbox namespace root to its manager specs.
# Used by construct_sandbox_root() to materialise environment namespaces at
# runtime.  The keys here must stay in sync with the env_namespaces frozenset
# in FunctionManager.add_functions (the storage-time AST detection side).
_SANDBOX_ROOTS: Dict[str, list[ManagerSpec]] = {}
for _spec in _MANAGER_SPECS:
    _SANDBOX_ROOTS.setdefault(_spec.sandbox_root, []).append(_spec)


def _environment_specs(
    primitive_scope: Optional[PrimitiveScope] = None,
) -> List[ManagerSpec]:
    """Specs for the namespaces the environment registered, in scope.

    They live under the same ``primitives`` sandbox root as ``actor``, so
    ``construct_sandbox_root`` and ``depends_on`` need nothing more; their
    ``primitive_class_path`` is ``environment:<name>``, which no importable
    class can have.
    """
    from unify.function_manager.primitives.environment import (
        environment_namespaces,
    )

    return [
        ManagerSpec(
            manager_alias=name,
            primitive_class_path=namespace.class_path,
            domain="Environment",
            description=namespace.description,
        )
        for name, namespace in sorted(environment_namespaces().items())
        if primitive_scope is None or name in primitive_scope.scoped_managers
    ]


def _environment_rows(
    primitive_scope: Optional[PrimitiveScope] = None,
) -> Dict[str, Dict[str, Any]]:
    """Primitive rows for the registered namespaces' methods (see collect_primitives)."""
    from unify.function_manager.primitives.environment import (
        environment_namespaces,
        method_docstring,
    )

    rows: Dict[str, Dict[str, Any]] = {}
    for name, namespace in sorted(environment_namespaces().items()):
        if primitive_scope is not None and name not in primitive_scope.scoped_managers:
            continue
        for method in sorted(namespace.methods, key=lambda m: m.name):
            qualified_name = f"primitives.{name}.{method.name}"
            rows[qualified_name] = {
                "name": qualified_name,
                "function_id": _get_stable_id(namespace.class_path, method.name),
                "argspec": method.signature,
                "docstring": method_docstring(name, method),
                "implementation": None,
                "is_primitive": True,
                "depends_on": [],
                "precondition": None,
                "guidance_ids": [],
                "primitive_class": namespace.class_path,
                "primitive_method": method.name,
                "metadata": {"effect": method.effect, "environment": name},
            }
    return rows


def construct_sandbox_root(
    root_name: str,
    *,
    primitive_scope: Optional[PrimitiveScope] = None,
) -> Optional[Any]:
    """Construct a fresh root namespace object for a sandbox namespace key.

    This is the factory used by ``FunctionManager._inject_dependencies``
    to satisfy *dotted* entries in a stored function's ``depends_on`` list.
    When a function declares a dependency like ``"primitives.actor.act"``,
    ``_inject_dependencies`` extracts the root segment (``"primitives"``)
    and calls this function to obtain a live ``Primitives`` instance that
    provides those methods.

    The returned object is **stateless** — it does not require any ambient
    ContextVars or parent actor state.  This is essential because stored
    functions may be executed outside of a live ``CodeActActor`` sandbox
    (e.g. by the storage-check loop's ``FunctionManager_add_functions``
    tool or by a future caller that only has a FunctionManager).

    Returns ``None`` when *root_name* does not match any known sandbox root.
    """
    specs = _SANDBOX_ROOTS.get(root_name)
    if not specs:
        return None

    from unify.function_manager.primitives.runtime import Primitives

    return Primitives(primitive_scope=primitive_scope)


# =============================================================================
# Stable ID Generation (Matching Old Format)
# =============================================================================


def _get_stable_id(class_name: str, method_name: str) -> int:
    """
    Generate a stable integer ID from class.method name.

    Uses a hash-based approach so IDs are:
    - Deterministic (same name → same ID)
    - Stable across code changes (no positional dependencies)
    - Unique (collision-resistant within practical limits)

    Args:
        class_name: Short class name (e.g., "_ActorRunner")
        method_name: Method name (e.g., "act")

    Returns:
        A stable non-negative integer ID within signed 32-bit range.
    """
    key = f"{class_name}.{method_name}"
    # Use first 4 bytes of SHA256, masked to signed int32 range (0x7FFFFFFF)
    # to avoid PostgreSQL integer overflow when sorting by function_id
    digest = hashlib.sha256(key.encode()).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


# =============================================================================
# NumPy-style docstring helpers
# =============================================================================


def _get_line_indent(line: str) -> int:
    """Count the number of leading spaces in a line."""
    return len(line) - len(line.lstrip())


def _is_param_declaration_at_level(line: str, base_indent: int) -> bool:
    """Detect whether a line is a parameter declaration at the given indent level.

    A declaration is at ``base_indent`` and matches one of:
    - ``name : type``  (standard single-param)
    - ``name1, name2``  (grouped params sharing a description)
    - ``name1 / name2 : type``  (slash-separated grouped params)
    - ``name``  (bare name, no type annotation)
    """
    stripped = line.strip()
    if not stripped:
        return False
    indent = _get_line_indent(line)
    if indent != base_indent:
        return False
    # Standard "name : type" format
    if " : " in stripped:
        return True
    # Grouped comma-separated names: "a, b, c"
    if "," in stripped:
        parts = [p.strip().rstrip(",") for p in stripped.split(",") if p.strip()]
        if all(re.match(r"^[a-zA-Z_]\w*$", p) for p in parts):
            return True
    # Bare single name
    if re.match(r"^[a-zA-Z_]\w*$", stripped):
        return True
    return False


def _extract_param_names(declaration_line: str) -> list[str]:
    """Extract parameter names from a NumPy-style declaration line.

    Handles: ``name : type``, ``name1, name2``, ``name1 / name2 : type``.
    """
    stripped = declaration_line.strip()
    name_part = stripped.split(" : ")[0] if " : " in stripped else stripped
    # Handle "/" separator (e.g., "_clarification_up_q / _clarification_down_q")
    if " / " in name_part:
        names = [n.strip() for n in name_part.split("/") if n.strip()]
    else:
        names = [n.strip().rstrip(",") for n in name_part.split(",") if n.strip()]
    return names


# =============================================================================
# ToolSurfaceRegistry - Main API
# =============================================================================


class ToolSurfaceRegistry:
    """
    Central registry for tool surface configuration.

    This class provides the API for:
    - Getting manager specs filtered by scope
    - Discovering primitive methods for a manager
    - Collecting primitive rows for FunctionManager indexing
    - Building row filters for scoped queries
    """

    # Class-level reference to canonical data
    MANAGERS = _MANAGER_SPECS

    def __init__(self) -> None:
        """Initialize the registry."""
        # Cache for dynamically loaded classes (avoids repeated imports)
        self._class_cache: Dict[str, Type] = {}

    def manager_specs(self, primitive_scope: PrimitiveScope) -> List[ManagerSpec]:
        """
        Get manager specs for a given scope, in registry order.

        Args:
            primitive_scope: The scope defining which managers are exposed.

        Returns:
            List of ManagerSpec for exposed managers.
        """
        specs = [
            spec
            for spec in _MANAGER_SPECS
            if spec.manager_alias in primitive_scope.scoped_managers
        ]
        if primitive_scope.scoped_managers - _MANAGER_BY_ALIAS.keys():
            specs += _environment_specs(primitive_scope)
        return specs

    def get_manager_spec(self, manager_alias: str) -> Optional[ManagerSpec]:
        """Get a single manager spec by alias."""
        spec = _MANAGER_BY_ALIAS.get(manager_alias)
        if spec is None:
            spec = next(
                (s for s in _environment_specs() if s.manager_alias == manager_alias),
                None,
            )
        return spec

    def get_function_id(self, manager_alias: str, method_name: str) -> int:
        """Compute the stable function_id for a primitive method.

        This returns the same ID that ``collect_primitives`` produces for rows
        stored in the builtins primitives catalogue, without requiring a DB
        round-trip.

        Args:
            manager_alias: Canonical manager alias (e.g., ``"actor"``).
            method_name: Method name (e.g., ``"act"``).

        Returns:
            Stable non-negative integer ID (deterministic, hash-based).

        Raises:
            ValueError: If *manager_alias* is not a known manager.
        """
        spec = _MANAGER_BY_ALIAS.get(manager_alias)
        if spec is None:
            from unify.function_manager.primitives.environment import (
                environment_namespace,
            )

            env_namespace = environment_namespace(manager_alias)
            if env_namespace is None:
                raise ValueError(f"Unknown manager alias: {manager_alias!r}")
            return _get_stable_id(env_namespace.class_path, method_name)
        class_name = spec.primitive_class_path.rsplit(".", 1)[-1]
        return _get_stable_id(class_name, method_name)

    def _load_manager_class(self, class_path: str) -> Optional[Type]:
        """Dynamically load a manager class for introspection."""
        if class_path in self._class_cache:
            return self._class_cache[class_path]

        try:
            module_path, class_name = class_path.rsplit(".", 1)
            import importlib

            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)
            self._class_cache[class_path] = cls
            return cls
        except Exception as e:
            logger.warning(f"Failed to load class {class_path}: {e}")
            return None

    def primitive_methods(self, *, manager_alias: str) -> List[str]:
        """
        Get the list of primitive methods for a manager.

        Discovers methods from @abstractmethod definitions on Base* classes,
        minus common exclusions and per-manager exclusions. A class that
        declares ``_PRIMITIVE_METHODS`` uses that constant instead.

        Args:
            manager_alias: The canonical manager alias.

        Returns:
            Sorted list of method names exposed as primitives.
        """
        spec = _MANAGER_BY_ALIAS.get(manager_alias)
        if not spec:
            from unify.function_manager.primitives.environment import (
                environment_namespace,
            )

            env_namespace = environment_namespace(manager_alias)
            if env_namespace is not None:
                return env_namespace.method_names()
            logger.warning(f"Unknown manager alias: {manager_alias}")
            return []

        cls = self._load_manager_class(spec.primitive_class_path)
        if cls is None:
            return []

        # Combine common exclusions with per-manager exclusions
        exclude = _COMMON_EXCLUDED_METHODS | spec.excluded_methods

        # Classes that declare _PRIMITIVE_METHODS explicitly (e.g. because
        # methods are dynamically created or not discoverable via @abstractmethod)
        # use that constant as the authoritative method list.
        if hasattr(cls, "_PRIMITIVE_METHODS"):
            return sorted([m for m in cls._PRIMITIVE_METHODS if m not in exclude])

        methods = []

        # Standard case: find @abstractmethod definitions in Base* classes
        for base in cls.__mro__:
            base_name = base.__name__
            # Look for Base* classes (e.g., BaseFunctionManager)
            if not base_name.startswith("Base"):
                continue
            # Skip the root BaseStateManager - we want the specific manager's base
            if base_name == "BaseStateManager":
                continue

            for name, method in vars(base).items():
                if name.startswith("_"):
                    continue
                if name in exclude:
                    continue
                if getattr(method, "__isabstractmethod__", False):
                    if name not in methods:
                        methods.append(name)

        return sorted(methods)

    def tool_names(self, primitive_scope: PrimitiveScope) -> List[str]:
        """
        Get fully-qualified tool names for a scope.

        Args:
            primitive_scope: The scope defining which managers are exposed.

        Returns:
            List of tool names like "primitives.actor.act".
        """
        names = []
        for alias in sorted(primitive_scope.scoped_managers):
            for method in self.primitive_methods(manager_alias=alias):
                names.append(f"primitives.{alias}.{method}")
        return names

    @staticmethod
    def _resolve_base_method(cls: Optional[Type], method_name: str):
        """Find the canonical method object from the Base* class in the MRO.

        Returns the method object or None.
        """
        if cls is None:
            return None
        for base in cls.__mro__:
            if base.__name__.startswith("Base") and method_name in vars(base):
                return vars(base)[method_name]
        return getattr(cls, method_name, None)

    @staticmethod
    def _format_method_signature(cls: Optional[Type], method_name: str) -> str:
        """Extract a compact signature string for a method.

        Strips ``self`` and any private ``_``-prefixed parameters (internal
        wiring like ``_parent_chat_context`` or ``_clarification_up_q``).

        Falls back to ``...`` when the class is unavailable or introspection
        fails for any reason.
        """
        method = ToolSurfaceRegistry._resolve_base_method(cls, method_name)
        if method is None:
            return "..."
        try:
            sig = inspect.signature(method)
            params = [
                p
                for name, p in sig.parameters.items()
                if name != "self" and not name.startswith("_")
            ]
            return str(sig.replace(parameters=params))
        except (ValueError, TypeError):
            return "..."

    @staticmethod
    def _format_method_heading(
        cls: Optional[Type],
        method_name: str,
        qualified_name: str,
    ) -> str:
        """*qualified_name* with the method's compact signature, as a prompt shows it.

        A coroutine method reads ``async def``, so the call is awaited rather
        than its coroutine taken for the return value.
        """
        method = ToolSurfaceRegistry._resolve_base_method(cls, method_name)
        prefix = "async def " if inspect.iscoroutinefunction(method) else ""
        signature = ToolSurfaceRegistry._format_method_signature(cls, method_name)
        return f"{prefix}{qualified_name}{signature}"

    @staticmethod
    def _extract_method_docstring(cls: Optional[Type], method_name: str) -> str:
        """Extract the full docstring for a method from the Base* class.

        Unwraps ``functools.wraps`` / ``__wrapped__`` to reach the original
        docstring.  Returns an empty string when unavailable.
        """
        method = ToolSurfaceRegistry._resolve_base_method(cls, method_name)
        if method is None:
            return ""
        fn = method
        while hasattr(fn, "__wrapped__"):
            fn = fn.__wrapped__
        return inspect.getdoc(fn) or ""

    @staticmethod
    def _extract_section_lines(
        lines: list[str],
        *,
        start_index: int,
        section_name: str,
    ) -> list[str]:
        """Return body lines of a NumPy-style section, or empty if missing."""
        section_lines: list[str] = []
        in_section = False
        for j in range(start_index, len(lines)):
            stripped = lines[j].strip()
            if stripped == section_name:
                in_section = True
                continue
            if in_section and stripped.startswith("---"):
                continue
            if in_section:
                if (
                    stripped
                    and not stripped[0].isspace()
                    and not stripped.startswith("-")
                ):
                    if j + 1 < len(lines) and lines[j + 1].strip().startswith(
                        "---",
                    ):
                        break
                    if re.match(r"^[A-Z][a-zA-Z\s]+$", stripped) and len(stripped) < 30:
                        break
                section_lines.append(lines[j])
        return section_lines

    @staticmethod
    def _extract_summary_and_params(docstring: str) -> str:
        """Extract summary, Parameters, and Anti-patterns from a docstring.

        Returns a compact version containing the summary (up to the first
        blank line), the NumPy-style ``Parameters`` section, and
        ``Anti-patterns`` when present. Omits Returns, Raises, Examples,
        Notes, and internal ``_``-prefixed parameters.
        """
        if not docstring:
            return ""

        lines = docstring.splitlines()

        # First paragraph (up to first blank line after content)
        summary_lines: list[str] = []
        i = 0
        for i, line in enumerate(lines):
            if not line.strip():
                if summary_lines:
                    break
            else:
                summary_lines.append(line)

        params_lines = ToolSurfaceRegistry._extract_section_lines(
            lines,
            start_index=i,
            section_name="Parameters",
        )
        anti_lines = ToolSurfaceRegistry._extract_section_lines(
            lines,
            start_index=i,
            section_name="Anti-patterns",
        )

        # Filter out internal _-prefixed parameter entries.
        # Detect the base indentation level from the first non-empty line.
        base_indent = 0
        for line in params_lines:
            if line.strip():
                base_indent = _get_line_indent(line)
                break

        filtered_params: list[str] = []
        skip_entry = False
        for line in params_lines:
            if _is_param_declaration_at_level(line, base_indent):
                names = _extract_param_names(line)
                skip_entry = bool(names) and all(n.startswith("_") for n in names)
            if not skip_entry:
                filtered_params.append(line)

        result = "\n".join(summary_lines)
        params = "\n".join(filtered_params).rstrip()
        if params:
            result += "\n\nParameters\n----------\n" + params
        anti = "\n".join(anti_lines).rstrip()
        if anti:
            result += "\n\nAnti-patterns\n-------------\n" + anti
        return result

    @staticmethod
    def _filter_internal_params_from_docstring(docstring: str) -> str:
        """Remove ``_``-prefixed parameter entries from a full docstring.

        Scans the Parameters section and drops any entry whose parameter
        name starts with ``_`` (including its indented description lines).
        All other docstring content is preserved unchanged.
        """
        if not docstring or "Parameters" not in docstring:
            return docstring

        lines = docstring.splitlines()
        result_lines: list[str] = []
        in_params = False
        skip_entry = False
        param_base_indent: int = 0

        i = 0
        while i < len(lines):
            stripped = lines[i].strip()

            # Detect start of Parameters section
            if not in_params and stripped == "Parameters":
                # Check for the dashes line that follows
                if i + 1 < len(lines) and lines[i + 1].strip().startswith("---"):
                    in_params = True
                    result_lines.append(lines[i])
                    result_lines.append(lines[i + 1])
                    # Detect base indent from the first param declaration
                    for k in range(i + 2, len(lines)):
                        if lines[k].strip():
                            param_base_indent = _get_line_indent(lines[k])
                            break
                    i += 2
                    continue
                else:
                    result_lines.append(lines[i])
                    i += 1
                    continue

            if in_params:
                # Detect next section header (end of Parameters)
                if (
                    stripped
                    and _get_line_indent(lines[i]) <= param_base_indent
                    and not stripped.startswith("-")
                ):
                    is_section = False
                    if i + 1 < len(lines) and lines[i + 1].strip().startswith("---"):
                        is_section = True
                    if re.match(r"^[A-Z][a-zA-Z\s]+$", stripped) and len(stripped) < 30:
                        is_section = True

                    if is_section:
                        in_params = False
                        skip_entry = False
                        result_lines.append(lines[i])
                        i += 1
                        continue

                if _is_param_declaration_at_level(lines[i], param_base_indent):
                    names = _extract_param_names(lines[i])
                    skip_entry = bool(names) and all(n.startswith("_") for n in names)

                if not skip_entry:
                    result_lines.append(lines[i])
            else:
                result_lines.append(lines[i])

            i += 1

        return "\n".join(result_lines)

    def _get_method_metadata(
        self,
        cls: Type,
        method_name: str,
        class_name: str,
        manager_alias: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Extract metadata (signature, docstring) from a class method.

        Handles functools.wraps by looking for __wrapped__ attribute.

        Args:
            cls: The class containing the method.
            method_name: Name of the method to introspect.
            class_name: Short class name (used for stable ID generation).
            manager_alias: The canonical alias (e.g. "actor") for
                building the qualified name ``primitives.{alias}.{method}``.

        Returns:
            Primitive metadata dict, or None if method not found.
        """
        method = getattr(cls, method_name, None)

        if method is None:
            return None

        # Unwrap functools.wraps to get original function metadata
        fn = method
        while hasattr(fn, "__wrapped__"):
            fn = fn.__wrapped__

        qualified_name = f"primitives.{manager_alias}.{method_name}"

        try:
            signature = str(inspect.signature(fn))
        except (ValueError, TypeError):
            signature = "(...)"

        docstring = inspect.getdoc(fn) or ""

        return {
            "name": qualified_name,
            "function_id": _get_stable_id(class_name, method_name),
            "argspec": signature,
            "docstring": docstring,
            "implementation": None,
            "is_primitive": True,
            "depends_on": [],
            "precondition": None,
            "guidance_ids": [],
            "primitive_class": cls.__module__ + "." + cls.__name__,
            "primitive_method": method_name,
        }

    def collect_primitives(
        self,
        primitive_scope: Optional[PrimitiveScope] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Introspect primitive classes and return their metadata.

        Auto-discovers primitive methods from each class by inspecting @abstractmethod
        definitions on base classes (minus per-class exclusions), then extracts
        signature and docstring information.

        Each primitive receives a stable `function_id` derived from a hash of its
        fully-qualified name (e.g., "_ActorRunner.act"). This ensures IDs are:
        - Deterministic across runs
        - Stable when methods are added/removed (no positional dependencies)
        - Consistent across every store

        Args:
            primitive_scope: Optional scope to filter managers. If None, collects all
                           primitives.

        Returns:
            Dict mapping qualified_name (e.g. "primitives.actor.act") to primitive
            metadata in the shape of a ``primitives`` table row.
        """
        primitives: Dict[str, Dict[str, Any]] = {}

        # Determine which specs to process
        if primitive_scope is None:
            specs_to_process = _MANAGER_SPECS
        else:
            specs_to_process = [
                spec
                for spec in _MANAGER_SPECS
                if spec.manager_alias in primitive_scope.scoped_managers
            ]

        for spec in specs_to_process:
            cls = self._load_manager_class(spec.primitive_class_path)
            if cls is None:
                continue

            class_name = cls.__name__
            method_names = self.primitive_methods(manager_alias=spec.manager_alias)

            for method_name in method_names:
                metadata = self._get_method_metadata(
                    cls,
                    method_name,
                    class_name,
                    spec.manager_alias,
                )
                if metadata is not None:
                    primitives[metadata["name"]] = metadata

        primitives.update(_environment_rows(primitive_scope))

        logger.debug(f"Collected {len(primitives)} primitives")
        return primitives

    def compute_primitives_hash(
        self,
        primitives: Optional[Dict[str, Dict[str, Any]]] = None,
        primitive_scope: Optional[PrimitiveScope] = None,
    ) -> str:
        """
        Compute a stable hash of all primitive signatures.

        Used to detect when primitives have changed (docstrings updated, methods
        added/removed) and a sync is needed.

        Args:
            primitives: Optional pre-collected primitives dict. If provided, uses these
                       directly without re-collecting.
            primitive_scope: Optional scope to filter managers. Only used if primitives
                           is not provided.

        Returns:
            16-character hex hash string.
        """
        if primitives is None:
            primitives = self.collect_primitives(primitive_scope)

        return stable_hash_for_rows(
            primitives.values(),
            fields=("name", "argspec", "docstring"),
            digest_chars=16,
            projection="delimited",
        )

    def compute_hash_for_manager(self, manager_alias: str) -> str:
        """
        Compute hash for a single manager's primitives.

        Args:
            manager_alias: The manager to hash.

        Returns:
            16-character hex hash string.
        """
        scope = PrimitiveScope(scoped_managers=frozenset({manager_alias}))
        return self.compute_primitives_hash(primitive_scope=scope)

    def primitive_row_filter(self, primitive_scope: PrimitiveScope) -> str:
        """
        Build the SQL clause selecting the primitives of the scoped managers.

        Matches on ``primitive_class`` (already stored on every primitive row),
        so no separate manager column is needed.

        Args:
            primitive_scope: The scope defining which managers to include.

        Returns:
            A ``WHERE`` clause over ``primitive_class``; ``"0"`` for an empty scope.
        """
        # Collect class paths for scoped managers
        class_paths = []
        for alias in primitive_scope.scoped_managers:
            spec = _MANAGER_BY_ALIAS.get(alias)
            if spec:
                class_paths.append(spec.primitive_class_path)
        if primitive_scope.scoped_managers - _MANAGER_BY_ALIAS.keys():
            class_paths += [
                spec.primitive_class_path
                for spec in _environment_specs(primitive_scope)
            ]

        if not class_paths:
            return "0"
        class_list = ", ".join(f"'{cp}'" for cp in sorted(class_paths))
        return f"primitive_class IN ({class_list})"


# =============================================================================
# Module-level convenience functions (matching old API)
# =============================================================================


def collect_primitives() -> Dict[str, Dict[str, Any]]:
    """
    Introspect all registered primitive classes and return their metadata.

    This is a convenience function matching the old module-level API.
    Delegates to ToolSurfaceRegistry.collect_primitives().

    Returns:
        Dict mapping qualified_name (e.g. "primitives.actor.act") to primitive
        metadata in the shape of a ``primitives`` table row.
    """
    return get_registry().collect_primitives()


def compute_primitives_hash(primitives: Dict[str, Dict[str, Any]]) -> str:
    """
    Compute a stable hash of all primitive signatures.

    This is a convenience function matching the old module-level API.
    Delegates to ToolSurfaceRegistry.compute_primitives_hash().

    Args:
        primitives: Dict from collect_primitives().

    Returns:
        16-character hex hash string.
    """
    return get_registry().compute_primitives_hash(primitives=primitives)


def get_primitive_sources() -> List[tuple[Type, List[str]]]:
    """
    Get all primitive classes with their discovered method names.

    Returns a list of (class, method_names) tuples. This provides access to
    the auto-discovered primitives for code that needs to iterate over them.

    Returns:
        List of (class_object, [method_names]) tuples.
    """
    registry = get_registry()
    result = []
    for spec in _MANAGER_SPECS:
        cls = registry._load_manager_class(spec.primitive_class_path)
        if cls is not None:
            methods = registry.primitive_methods(manager_alias=spec.manager_alias)
            result.append((cls, methods))
    return result


# Module-level singleton for convenience
_registry_instance: Optional[ToolSurfaceRegistry] = None


def get_registry() -> ToolSurfaceRegistry:
    """Get the singleton registry instance."""
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = ToolSurfaceRegistry()
    return _registry_instance

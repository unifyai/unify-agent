# FunctionManager

The `FunctionManager` maintains a catalogue of executable Python functions, split into two categories:

1. **Primitives** – System action methods (the `primitives.*` namespaces) exposed for direct invocation
2. **Compositional Functions** – User-specific functions stored with their source code

## Architecture

### Two Tables, One View

The two categories live in two tables of the store (`unify/db.py`), read together through one view:

| Table | Purpose | ID Assignment |
|-------|---------|---------------|
| `primitives` | System primitives | Explicit stable IDs (hashed from the code) |
| `functions` | User-specific functions | Auto-incrementing |
| `all_functions` (view) | Both populations, with `is_primitive` set to 1 or 0 | |

This separation guarantees:
- Primitive IDs are consistent across every store
- Compositional function IDs are never affected by primitive changes
- No ID collisions between the two tables

### Primitive ID Stability

Primitives receive stable IDs derived from a hash of their fully-qualified name (e.g., "_ActorRunner.act" → deterministic integer). This means:
- IDs are consistent across every store
- Adding/removing methods doesn't affect other primitives' IDs
- No manual ID management required

Primitive methods are auto-discovered from `@abstractmethod` definitions on base classes, or from an explicit `_PRIMITIVE_METHODS` constant, minus an explicit exclusion list for non-primitive methods like `clear()`.

---

## The `primitives` Object

The `primitives` object provides lazy access to the primitive namespaces. Instantiation only happens when accessed:

```python
async def delegate(topic: str):
    # The actor runner is constructed on first access
    handle = await primitives.actor.act(request=f"Research {topic}")
    return await handle.result()
```

### Available Properties

| Property | Class | Methods |
|----------|-------|---------|
| `primitives.actor` | `_ActorRunner` | `act` |

Guidance is **not** a primitive. It is a typed catalogue exposed as top-level Actor JSON tools (`GuidanceManager_*`).

---

## Primitive Catalogue Seeding

The `primitives` table is made equal to the primitive registry when a `FunctionManager` is constructed, once per process per store: every row is deleted and re-inserted with its stable ID in one transaction. Reads go through the `all_functions` view, so primitives and stored functions answer to the same SQL. `db.clear()` leaves the table alone.

---

## Filtering

`filter_functions(filter=...)` takes a SQL `WHERE` clause (without the keyword) over the `all_functions` columns, for example `name LIKE 'get_%'` or `docstring LIKE '%csv%' AND is_primitive = 0`. JSON list columns such as `depends_on` are queried with `EXISTS (SELECT 1 FROM json_each(depends_on) WHERE value = 'helper')`. The clause is combined with the manager's `filter_scope` and runs read-only; a clause SQLite rejects comes back as an `invalid_filter` tool error listing the columns it may use.

---

## Writing Compositional Functions

Compositional functions are stored with their full source code. They may be created by:
- The Actor generating and saving a function during execution
- Pre-provisioning functions for a specific user/client
- Direct API calls to `add_functions()`

### Format Requirements

Each implementation string must contain **exactly one function definition** starting at column 0:

```python
# ✓ Correct
async def my_function(path: str):
    import csv
    with open(path) as fh:
        return sum(1 for _ in csv.reader(fh))

# ✗ Wrong - indented
    async def my_function():
        pass

# ✗ Wrong - multiple functions
def helper():
    pass

def main():
    helper()
```

### Imports: Optional (Pre-Injected Globals Available)

Functions are executed in a sandboxed environment with pre-injected globals. Common modules are already available, but **imports do work** if needed:

```python
# ✓ Works - using pre-injected globals (preferred for common modules)
async def example_no_import():
    data = json.dumps({"key": "value"})
    return data

# ✓ Also works - explicit import (useful for non-standard packages)
async def example_with_import():
    import numpy as np
    x = np.array([1, 2, 3])
    return x.sum()
```

The sandbox includes `__import__`, so any package installed in the environment can be imported.

### Best Practice: Domain Types & Type Hints (Avoid Surprise `NameError`s)

Compositional functions are often used by the Actor/CodeActActor by **retrieving a callable** (e.g. via `search_functions(..., return_callable=True)`) and executing it in a fresh sandbox namespace.

Here’s the simple rule:
- If you reference a symbol in the **function body**, **import/define it** in the function (don’t assume it exists in globals).
- If you reference a symbol only in **annotations** (including forward-ref strings), that’s usually fine.

```python
# ✅ OK: "User" only appears in the annotation (as a forward-ref string).
# The function body doesn't need the User symbol at runtime.
async def greet(user: "User") -> str:
    return f"Hello {user.name}"


# ⚠️ NOT OK: Role is used at runtime, so it MUST exist (import/define it).
async def is_admin(role: "Role") -> bool:
    return role == Role.ADMIN


# ✅ Preferred: import the runtime type inside the function.
async def is_admin(role: "Role") -> bool:
    from my_app.types import Role
    return role == Role.ADMIN
```

Note: if your code resolves type hints at runtime (e.g. `typing.get_type_hints(...)` or Pydantic model building),
then all referenced names must be resolvable. `FunctionManager` makes annotation-resolution more robust, but it cannot
guess the *real* domain objects for runtime logic—imports/definitions are the reliable solution.

#### Pre-Injected by Sandbox

These are always available (from `create_execution_globals()`):

| Category | Available Names |
|----------|-----------------|
| **Builtins** | `print`, `len`, `str`, `int`, `float`, `list`, `dict`, `set`, `range`, `isinstance`, `issubclass`, `hasattr`, `getattr`, `enumerate`, `zip`, `sorted`, `min`, `max`, `sum`, `any`, `all`, etc. |
| **Modules** | `asyncio`, `re`, `json`, `datetime`, `collections`, `statistics`, `functools` |
| **Typing** | `typing`, `Any`, `Callable`, `Dict`, `List`, `Optional`, `Tuple`, `Set`, `Union`, `Literal` |
| **Pydantic** | `pydantic`, `BaseModel`, `Field` |
| **Primitives** | `primitives` – lazy access to the primitive namespaces |
| **Steerable** | `SteerableToolHandle` |

#### Injected by Actor at Runtime

When functions are executed via an Actor (`CodeActActor`), additional objects are injected:

| Name | Description |
|------|-------------|
| `request_clarification` | Ask the user for clarification during execution |

---

## Example: Complete Function

```python
async def summarize_csv(path: str, question: str) -> str:
    """
    Answer a question about a CSV file with a focused LLM call.

    Args:
        path: Absolute path of the CSV file in the workspace.
        question: What to answer about the rows.

    Returns:
        The model's answer, grounded in the file's first rows.
    """
    import csv

    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))[:200]

    return await query_llm(
        f"Rows:\n{json.dumps(rows)}\n\nQuestion: {question}",
        temperature=0.0,
    )
```

---

## Steerable Functions

Compositional functions can optionally return a **steerable handle** instead of a final result. This allows the calling layer (e.g., `CodeActActor`) to forward steering operations (interject, pause, stop) into the running function.

### What is a Steerable Function?

A steerable function is one that:
1. Starts a background task (e.g., an async tool loop, an actor)
2. Returns a `SteerableToolHandle` immediately (before the task completes)
3. Allows the caller to interact with the running task via the handle

### Runtime Detection

Steerability is detected at **runtime** via `isinstance(result, SteerableToolHandle)`:

```python
from unify.common.async_tool_loop import SteerableToolHandle

result = await my_function()

if isinstance(result, SteerableToolHandle):
    # Function returned a steerable handle - can forward steering operations
    await result.interject("Please also check for errors")
    final_result = await result.result()
else:
    # Function returned a plain value - no steering possible
    final_result = result
```

### Writing a Steerable Function

Return a handle from a nested actor:

```python
async def research(topic: str) -> SteerableToolHandle:
    """
    Start a research sub-task and return its handle.

    The caller can interject, pause, or stop the research while it runs.
    """
    return await primitives.actor.act(request=f"Research {topic}")
```

### Available Infrastructure

These are injected into the execution globals by `create_execution_globals()`:

| Name | Purpose |
|------|---------|
| `SteerableToolHandle` | Base ABC for steerable handles and runtime `isinstance` checks |

### Handle Methods

Steerable handles provide these methods:

| Method | Description |
|--------|-------------|
| `await handle.result()` | Wait for and return the final result |
| `await handle.interject(message)` | Inject a message into the running task |
| `await handle.pause()` | Pause the task (in-flight operations continue) |
| `await handle.resume()` | Resume a paused task |
| `handle.stop(reason)` | Cancel the task immediately |
| `await handle.ask(question)` | Query the task's status without modifying it |

### Example: Delegated Actor Workflow

```python
async def delegated_research(topic: str) -> SteerableToolHandle:
    """
    Delegate research to an actor and return the running handle.

    The caller can interject to refine the search, or stop early
    if enough information has been gathered.
    """
    return await primitives.actor.act(
        request=f"Research the following topic thoroughly: {topic}",
        prompt_functions=["summarize_csv"],
        can_store=False,
        timeout=300,
    )
```

### Non-Steerable Functions

Regular functions that return plain values are **not** steerable:

```python
async def word_count(path: str) -> int:
    """A simple function - returns a plain value, not steerable."""
    with open(path) as fh:
        return len(fh.read().split())  # Plain int, not a handle
```

The execution layer will detect this via `isinstance` and handle it normally.

---

## API Summary

### Primitives

```python
fm = FunctionManager()

# List all primitives
fm.list_primitives()

# Search includes primitives by default
fm.search_functions(query="delegate a sub-task", include_primitives=True)
```

### Compositional Functions

```python
# Add functions
fm.add_functions(implementations=["async def foo(): pass"])

# List functions
fm.list_functions(include_implementations=False)

# Search by similarity
fm.search_functions(query="summarise a csv report", n=5)

# Delete
fm.delete_function(function_id=1)
```

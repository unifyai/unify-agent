import asyncio

import pytest
from pydantic import TypeAdapter

from unify.actor.code_act_actor import CodeActActor
from unify.actor.execution import (
    PythonExecutionSession,
    ExecutionResult,
    _CURRENT_SANDBOX,
    parts_to_text,
    parts_to_llm_content,
    TextPart,
    ImagePart,
    OutputPart,
)
from unify.actor.execution.capture import _encode_image_for_llm
from unify.common._async_tool.formatting import (
    serialize_tool_content,
    FormattedToolResult,
)


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_sandbox_stateful_variable_execution():
    """Tests that the sandbox maintains simple variable state between calls."""
    sandbox = PythonExecutionSession()

    result1 = await sandbox.execute("x = 100")
    assert result1["error"] is None
    # Read back through the session: with Python in the sandboxed worker
    # (the default) the cell's names live there, not in ``global_state``.
    assert (await sandbox.execute("x"))["result"] == 100

    result2 = await sandbox.execute("y = x * 2\nprint(y)")
    assert result2["error"] is None
    assert (await sandbox.execute("y"))["result"] == 200
    assert parts_to_text(result2["stdout"]) == "200\n"


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_sandbox_stateful_import_execution():
    """Tests that the sandbox maintains imported modules between calls."""
    sandbox = PythonExecutionSession()

    result1 = await sandbox.execute("import json")
    assert result1["error"] is None
    assert "json" in sandbox.global_state

    result2 = await sandbox.execute(
        "my_dict = {'key': 'value'}\nprint(json.dumps(my_dict))",
    )
    assert result2["error"] is None
    assert parts_to_text(result2["stdout"]).strip() == '{"key": "value"}'


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_sandbox_stateful_function_definition():
    """Tests that the sandbox maintains function definitions between calls."""
    sandbox = PythonExecutionSession()

    func_def_code = "def my_adder(a, b):\n    return a + b"
    result1 = await sandbox.execute(func_def_code)
    assert result1["error"] is None

    func_call_code = "result = my_adder(10, 5)\nprint(result)"
    result2 = await sandbox.execute(func_call_code)
    assert result2["error"] is None
    assert parts_to_text(result2["stdout"]).strip() == "15"
    assert (await sandbox.execute("result"))["result"] == 15


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_sandbox_stateful_class_definition():
    """Tests that the sandbox maintains class definitions between calls."""
    sandbox = PythonExecutionSession()

    class_def_code = (
        "class Greeter:\n"
        "    def __init__(self, name):\n"
        "        self.name = name\n"
        "    def greet(self):\n"
        "        return f'Hello, {self.name}!'\n"
    )
    result1 = await sandbox.execute(class_def_code)
    assert result1["error"] is None

    class_use_code = "g = Greeter('World')\nprint(g.greet())"
    result2 = await sandbox.execute(class_use_code)
    assert result2["error"] is None
    assert parts_to_text(result2["stdout"]).strip() == "Hello, World!"


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_sandbox_pydantic_model_with_typing_annotations():
    """Pydantic models using typing constructs (List, Optional, Dict) must work.

    Pydantic v2 resolves type annotations lazily via sys.modules[cls.__module__].
    If the sandbox doesn't register itself as a proper module, cls.__module__
    points to 'builtins' (because __name__ is missing from the exec globals),
    and typing constructs can't be resolved — only builtin types (int, str) work.
    """
    sandbox = PythonExecutionSession()

    code = """
from pydantic import BaseModel
from typing import List, Optional

class RecordItem(BaseModel):
    record_id: int
    first_name: str
    surname: str
    phone_number: str = ""
    email_address: str = ""

class RecordListResponse(BaseModel):
    records: List[RecordItem]
    total_count: int

result = RecordListResponse(
    records=[RecordItem(record_id=1, first_name="Alice", surname="Smith")],
    total_count=1,
)
print(result.model_dump_json())
"""
    result = await sandbox.execute(code)
    assert (
        result["error"] is None
    ), f"Pydantic model with List[RecordItem] failed in sandbox:\n{result['error']}"
    stdout = parts_to_text(result["stdout"]).strip()
    assert '"total_count": 1' in stdout or '"total_count":1' in stdout


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_sandbox_pydantic_model_with_optional_nested():
    """Optional[List[Model]] annotations must resolve in the sandbox."""
    sandbox = PythonExecutionSession()

    code = """
from pydantic import BaseModel
from typing import Dict, List, Optional

class Tag(BaseModel):
    key: str
    value: str

class Record(BaseModel):
    name: str
    tags: Optional[List[Tag]] = None
    metadata: Optional[Dict[str, str]] = None

r = Record(name="test", tags=[Tag(key="a", value="b")], metadata={"x": "y"})
print(r.model_dump_json())
"""
    result = await sandbox.execute(code)
    assert (
        result["error"] is None
    ), f"Pydantic model with Optional[List[Tag]] failed in sandbox:\n{result['error']}"
    stdout = parts_to_text(result["stdout"]).strip()
    assert '"name": "test"' in stdout or '"name":"test"' in stdout


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_sandbox_exposes_query_llm_helper_for_generated_code():
    """Generated Python sessions should expose the LLM query helper."""
    sandbox = PythonExecutionSession()

    assert "query_llm" in sandbox.global_state
    assert "reason" not in sandbox.global_state

    async def fake_query_llm(prompt: str, **kwargs):
        assert "reply" in prompt
        return "yes"

    sandbox.global_state["query_llm"] = fake_query_llm
    result = await sandbox.execute(
        """
decision = await query_llm("Does this email need a reply?")
print(decision)
decision
""",
    )

    assert result["error"] is None
    assert parts_to_text(result["stdout"]).strip() == "yes"
    assert result["result"] == "yes"


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_sandbox_error_handling():
    """Tests that the sandbox correctly captures and reports exceptions."""
    sandbox = PythonExecutionSession()
    result = await sandbox.execute("x = 1 / 0")

    assert parts_to_text(result["stdout"]) == ""
    assert parts_to_text(result["stderr"]) == ""
    assert result["result"] is None
    assert result["error"] is not None
    assert "ZeroDivisionError" in result["error"]


# ---------------------------------------------------------------------------
# Tests for Pydantic OutputPart types and order-preserving serialization
# ---------------------------------------------------------------------------


def test_output_part_pydantic_discriminated_union():
    """Tests that OutputPart is a proper Pydantic discriminated union."""
    # TextPart
    text = TextPart(text="hello")
    assert text.type == "text"
    assert text.text == "hello"

    # ImagePart
    image = ImagePart(data="base64data", mime="image/png")
    assert image.type == "image"
    assert image.data == "base64data"
    assert image.mime == "image/png"

    # Discriminated union parsing via TypeAdapter
    adapter = TypeAdapter(OutputPart)

    parsed_text = adapter.validate_python({"type": "text", "text": "world"})
    assert isinstance(parsed_text, TextPart)
    assert parsed_text.text == "world"

    parsed_image = adapter.validate_python(
        {"type": "image", "data": "abc123", "mime": "image/jpeg"},
    )
    assert isinstance(parsed_image, ImagePart)
    assert parsed_image.data == "abc123"


def test_parts_to_llm_content_preserves_order():
    """Tests that parts_to_llm_content preserves text/image interleaving."""
    parts = [
        TextPart(text="before image\n"),
        ImagePart(data="img1_base64", mime="image/png"),
        TextPart(text="between images\n"),
        ImagePart(data="img2_base64", mime="image/jpeg"),
        TextPart(text="after images\n"),
    ]

    llm_content = parts_to_llm_content(parts)

    # Should produce 5 blocks in exact order
    assert len(llm_content) == 5

    assert llm_content[0] == {"type": "text", "text": "before image\n"}
    assert llm_content[1] == {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,img1_base64"},
    }
    assert llm_content[2] == {"type": "text", "text": "between images\n"}
    assert llm_content[3] == {
        "type": "image_url",
        "image_url": {"url": "data:image/jpeg;base64,img2_base64"},
    }
    assert llm_content[4] == {"type": "text", "text": "after images\n"}


def test_parts_to_llm_content_merges_adjacent_text():
    """Tests that adjacent TextParts are merged into a single text block."""
    parts = [
        TextPart(text="line1\n"),
        TextPart(text="line2\n"),
        ImagePart(data="img_base64", mime="image/png"),
        TextPart(text="line3\n"),
        TextPart(text="line4\n"),
    ]

    llm_content = parts_to_llm_content(parts)

    # Adjacent text parts should be merged: 2 text blocks + 1 image
    assert len(llm_content) == 3

    assert llm_content[0] == {"type": "text", "text": "line1\nline2\n"}
    assert llm_content[1]["type"] == "image_url"
    assert llm_content[2] == {"type": "text", "text": "line3\nline4\n"}


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_sandbox_display_produces_image_parts():
    """Tests that display() produces ImagePart for PIL images."""
    sandbox = PythonExecutionSession()

    code = """
from PIL import Image
img = Image.new("RGB", (10, 10), color="red")
print("before")
display(img)
print("after")
"""
    result = await sandbox.execute(code)
    assert result["error"] is None

    stdout = result["stdout"]

    # Should have TextPart, ImagePart, TextPart in order
    assert len(stdout) == 3
    assert isinstance(stdout[0], TextPart)
    assert isinstance(stdout[1], ImagePart)
    assert isinstance(stdout[2], TextPart)

    # Verify text content
    assert "before" in stdout[0].text
    assert "after" in stdout[2].text

    # Verify image part has valid base64 data
    assert stdout[1].mime == "image/png"
    assert len(stdout[1].data) > 0

    # Verify LLM content preserves order
    llm_content = parts_to_llm_content(stdout)
    assert llm_content[0]["type"] == "text"
    assert llm_content[1]["type"] == "image_url"
    assert llm_content[2]["type"] == "text"


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_display_compresses_oversized_image():
    """display() must keep images within Anthropic's 5 MB base64 limit.

    A 4032x3024 random-noise image is huge as a lossless PNG at native
    resolution. ``display()`` first fits to the observation space, then
    encodes as PNG, falling back to JPEG with progressive quality
    reduction only when PNG still exceeds the limit.
    """
    sandbox = PythonExecutionSession()

    code = """
import numpy as np
from PIL import Image
img = Image.fromarray(np.random.randint(0, 255, (3024, 4032, 3), dtype=np.uint8))
display(img)
"""
    result = await sandbox.execute(code)
    assert result["error"] is None, f"display() raised: {result['error']}"

    stdout = result["stdout"]
    images = [p for p in stdout if isinstance(p, ImagePart)]
    assert len(images) == 1, f"Expected exactly 1 image, got {len(images)}"

    img_part = images[0]
    assert len(img_part.data) <= 5_242_880, (
        f"display() produced a {len(img_part.data):,} byte base64 image "
        f"(limit is 5,242,880). It should compress to fit."
    )
    assert img_part.mime in {
        "image/png",
        "image/jpeg",
    }, f"Expected PNG or JPEG for oversized image, got {img_part.mime}"


def test_encode_image_falls_back_to_jpeg_when_png_exceeds_limit(monkeypatch):
    """JPEG fallback still triggers when observation fitting is a no-op."""
    import numpy as np
    from PIL import Image

    monkeypatch.setattr(
        "unify.common.observation_scaling.fit_image_to_observation_space",
        lambda img: img,
    )
    img = Image.fromarray(
        np.random.randint(0, 255, (3024, 4032, 3), dtype=np.uint8),
    )
    b64_data, mime = _encode_image_for_llm(img)
    assert mime == "image/jpeg"
    assert len(b64_data) <= 5_242_880


def test_execution_result_implements_formatted_tool_result():
    """Tests that ExecutionResult implements the FormattedToolResult protocol."""
    # ExecutionResult should be recognized as a FormattedToolResult
    result = ExecutionResult(
        stdout=[TextPart(text="hello\n")],
        stderr=[],
        result=42,
        error=None,
    )

    # Protocol check
    assert isinstance(result, FormattedToolResult)

    # to_llm_content should return list of dicts
    llm_content = result.to_llm_content()
    assert isinstance(llm_content, list)
    assert all(isinstance(block, dict) for block in llm_content)
    assert all("type" in block for block in llm_content)


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_serialize_tool_content_delegates_to_execution_result():
    """Tests that serialize_tool_content delegates to ExecutionResult.to_llm_content().

    This verifies the protocol-based design: the tool result (ExecutionResult)
    controls its own formatting, not serialize_tool_content.
    """
    sandbox = PythonExecutionSession()

    code = """
from PIL import Image
img = Image.new("RGB", (10, 10), color="blue")
print("text before image")
display(img)
print("text after image")
"""
    result_dict = await sandbox.execute(code)
    assert result_dict["error"] is None

    # Wrap in ExecutionResult (simulating what execute_code tool does)
    result = ExecutionResult(**result_dict)

    # Verify result is an ExecutionResult implementing the protocol
    assert isinstance(result, ExecutionResult)
    assert isinstance(result, FormattedToolResult)

    # Pass through serialize_tool_content (simulating what happens in tool loop)
    serialized = serialize_tool_content(
        tool_name="execute_code",
        payload=result,
        is_final=True,
    )

    # Should be a list of content blocks
    assert isinstance(serialized, list)

    # Extract block types
    block_types = [b.get("type") for b in serialized]

    # Find image and text indices
    image_indices = [i for i, t in enumerate(block_types) if t == "image_url"]
    text_indices = [i for i, t in enumerate(block_types) if t == "text"]

    # There should be at least one image
    assert len(image_indices) >= 1, "Expected at least one image_url block"

    # Critical: image should NOT be at the very end if there's text after it
    # The last text block should come AFTER the image (preserving print order)
    last_image_idx = max(image_indices)
    last_text_idx = max(text_indices)

    assert last_text_idx > last_image_idx, (
        f"Image at index {last_image_idx} should not be after all text. "
        f"Text indices: {text_indices}, Image indices: {image_indices}. "
        "This indicates images are being collected at the end instead of "
        "preserving their original interleaved positions."
    )


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_sandbox_builtin_open_available():
    """Python builtins like open() and FileNotFoundError must be available.

    The sandbox restricts the global namespace, but standard builtins that
    are essential for basic file I/O should not be stripped. The cell writes
    its own file: with Python in the sandboxed worker (the default) the
    worker's /tmp is private, so a file the test wrote on the host's /tmp is
    not visible to it (confinement, not a missing builtin).
    """
    sandbox = PythonExecutionSession()

    code = (
        "import tempfile, os\n"
        "path = os.path.join(tempfile.mkdtemp(), 'hello.txt')\n"
        "with open(path, 'w') as f:\n"
        "    f.write('hello world')\n"
        "with open(path, 'r') as f:\n"
        "    print(f.read())\n"
        "try:\n"
        "    open(path + '.missing')\n"
        "except FileNotFoundError:\n"
        "    print('missing raises FileNotFoundError')"
    )
    result = await sandbox.execute(code)
    assert (
        result["error"] is None
    ), f"open() should be available in the sandbox but got: {result['error']}"
    out = parts_to_text(result["stdout"])
    assert "hello world" in out
    assert "missing raises FileNotFoundError" in out


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_sandbox_concurrent_cells_serialize_in_arrival_order():
    """Cells submitted concurrently to one session run one at a time, in order.

    Cells share the session's module dict, display hook, and primitives
    binding, so a cell that awaits mid-body must not interleave with a
    later-submitted cell, and each cell's captured output stays its own.
    """
    sandbox = PythonExecutionSession()
    seed = await sandbox.execute("order = []")
    assert seed["error"] is None

    task_a = asyncio.create_task(
        sandbox.execute(
            "import asyncio\n"
            "await asyncio.sleep(0.2)\n"
            "print('from-A')\n"
            "order.append('A')",
        ),
    )
    # Let cell A start (and hit its await) before submitting cell B.
    await asyncio.sleep(0.05)
    task_b = asyncio.create_task(
        sandbox.execute("print('from-B')\norder.append('B')"),
    )
    result_a, result_b = await asyncio.gather(task_a, task_b)

    assert result_a["error"] is None
    assert result_b["error"] is None
    # A finishes before B despite sleeping mid-cell: execution is exclusive
    # and arrival-ordered, not interleaved at await points.
    final = await sandbox.execute("order")
    assert final["result"] == ["A", "B"]
    # Output capture is per-cell, not crosstalked through the shared session.
    assert parts_to_text(result_a["stdout"]) == "from-A\n"
    assert parts_to_text(result_b["stdout"]) == "from-B\n"


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_sandbox_distinct_sessions_do_not_contend():
    """A cell in one session runs while another session's cell is in flight."""
    session_a = PythonExecutionSession()
    session_b = PythonExecutionSession()

    # Cell A blocks on an event only the test can set, so it is guaranteed
    # to still be executing while session B's cell runs to completion.
    gate = asyncio.Event()
    session_a.global_state["gate"] = gate
    task_a = asyncio.create_task(session_a.execute("await gate.wait()"))
    # Let session A's cell start before running session B's cell.
    await asyncio.sleep(0.05)
    result_b = await session_b.execute("value = 41 + 1\nvalue")

    assert result_b["error"] is None
    assert result_b["result"] == 42
    # B completed while A was still executing: no cross-session contention.
    assert not task_a.done()
    gate.set()
    result_a = await task_a
    assert result_a["error"] is None


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_execute_code_python_cell_defaults_to_persistent_session_zero():
    """A python cell with no state_mode runs stateful in the bound sandbox.

    Variables defined in one default cell are readable in the next, and an
    explicitly stateless cell neither sees nor pollutes session-0 names.
    """
    actor = CodeActActor(environments=[])
    sandbox = PythonExecutionSession()
    token = _CURRENT_SANDBOX.set(sandbox)
    try:
        execute_code = actor._build_tools()["execute_code"]

        first = await execute_code(
            thought="define a variable in a default cell",
            code="answer = 41",
        )
        assert first.error is None
        assert first.state_mode == "stateful"
        assert first.session_id == 0

        second = await execute_code(
            thought="read the variable back in a later default cell",
            code="answer + 1",
        )
        assert second.error is None
        assert second.result == 42

        isolated = await execute_code(
            thought="run an isolated cell",
            state_mode="stateless",
            code=(
                "try:\n"
                "    print(answer)\n"
                "except NameError:\n"
                "    print('NOT_FOUND')\n"
                "leak = True"
            ),
        )
        assert isolated.error is None
        assert "NOT_FOUND" in parts_to_text(isolated.stdout)

        # Read back through session 0 (with Python in the sandboxed worker its
        # names live there, not in ``global_state``).
        after = await execute_code(
            thought="check session 0 after the isolated cell",
            code=(
                "try:\n"
                "    leak\n"
                "    print('LEAKED')\n"
                "except NameError:\n"
                "    print('NOT_FOUND')\n"
                "answer"
            ),
        )
        assert after.error is None
        assert "NOT_FOUND" in parts_to_text(after.stdout)
        assert after.result == 41
    finally:
        _CURRENT_SANDBOX.reset(token)
        try:
            await actor.close()
        except Exception:
            pass


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_execute_code_default_session_is_scoped_to_the_bound_sandbox():
    """Each act binds its own sandbox, so default cells never share state.

    Each binding — the shape of nested or successive act() calls — resolves
    its default cells to its own session 0, never another act's namespace.
    """
    actor = CodeActActor(environments=[])
    try:
        execute_code = actor._build_tools()["execute_code"]

        async def run_default_cell(code: str):
            sandbox = PythonExecutionSession()
            token = _CURRENT_SANDBOX.set(sandbox)
            try:
                return await execute_code(
                    thought="run a default cell in this act's sandbox",
                    code=code,
                )
            finally:
                _CURRENT_SANDBOX.reset(token)

        first = await run_default_cell("own_name = 'mine'")
        assert first.error is None
        assert first.session_id == 0

        probe = await run_default_cell(
            "try:\n"
            "    print(own_name)\n"
            "except NameError:\n"
            "    print('NOT_FOUND')",
        )
        assert probe.error is None
        assert "NOT_FOUND" in parts_to_text(probe.stdout)
    finally:
        try:
            await actor.close()
        except Exception:
            pass

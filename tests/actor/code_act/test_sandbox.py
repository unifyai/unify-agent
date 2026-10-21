"""Symbolic: the actor's interpreter keeps state between cells, and a cell's output keeps its order.

Nothing here reaches a model. Cells run in the actor's real interpreter (the
sandboxed worker, ``sandbox_world.actor_interpreter``); tests that need
bubblewrap are skipped, saying so, where it is missing.
"""

import asyncio

import pytest
from pydantic import TypeAdapter

from tests.actor.code_act.sandbox_world import actor_interpreter, needs_bwrap
from unify.actor.execution import (
    ExecutionResult,
    ImagePart,
    OutputPart,
    TextPart,
    parts_to_llm_content,
)
from unify.common._async_tool.formatting import (
    FormattedToolResult,
    serialize_tool_content,
)


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_cells_keep_variables_imports_functions_and_classes():
    interp = await actor_interpreter()
    try:
        assert (await interp.run_cell("x = 100")).error is None
        assert (await interp.run_cell("x")).result == 100
        res = await interp.run_cell("y = x * 2\nprint(y)")
        assert res.error is None and res.output == "200\n"
        assert (await interp.run_cell("import json")).error is None
        res = await interp.run_cell("print(json.dumps({'key': 'value'}))")
        assert res.output.strip() == '{"key": "value"}'
        await interp.run_cell("def my_adder(a, b):\n    return a + b")
        res = await interp.run_cell("result = my_adder(10, 5)\nprint(result)")
        assert res.output.strip() == "15" and (await interp.run_cell("result")).result == 15
        await interp.run_cell(
            "class Greeter:\n"
            "    def __init__(self, name):\n"
            "        self.name = name\n"
            "    def greet(self):\n"
            "        return f'Hello, {self.name}!'\n",
        )
        res = await interp.run_cell("print(Greeter('World').greet())")
        assert res.output.strip() == "Hello, World!"
    finally:
        await interp.stop()


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_pydantic_models_with_typing_annotations_resolve():
    """Pydantic resolves annotations through ``sys.modules[cls.__module__]``: a cell's namespace is a module."""
    interp = await actor_interpreter()
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

class Listing(BaseModel):
    records: List[Record]
    total_count: int

print(Listing(records=[Record(name="test", tags=[Tag(key="a", value="b")])], total_count=1).model_dump_json())
"""
    try:
        res = await interp.run_cell(code)
        assert res.error is None, res.text()
        assert '"total_count":1' in res.output and '"name":"test"' in res.output
    finally:
        await interp.stop()


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_an_exception_is_the_cells_error():
    interp = await actor_interpreter()
    try:
        res = await interp.run_cell("x = 1 / 0")
        assert res.output == "" and res.result is None
        assert res.error is not None and "ZeroDivisionError" in res.error
    finally:
        await interp.stop()


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_builtin_open_works_in_the_cells_private_tmp():
    interp = await actor_interpreter()
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
    try:
        res = await interp.run_cell(code)
        assert res.error is None, res.text()
        assert "hello world" in res.output
        assert "missing raises FileNotFoundError" in res.output
    finally:
        await interp.stop()


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_concurrent_cells_run_one_at_a_time_in_arrival_order():
    """A cell that awaits mid-body does not interleave with a later one; each keeps its own output."""
    interp = await actor_interpreter()
    try:
        await interp.run_cell("order = []")
        task_a = asyncio.create_task(
            interp.run_cell(
                "import asyncio\nawait asyncio.sleep(0.2)\nprint('from-A')\norder.append('A')",
            ),
        )
        await asyncio.sleep(0.05)
        task_b = asyncio.create_task(interp.run_cell("print('from-B')\norder.append('B')"))
        res_a, res_b = await asyncio.gather(task_a, task_b)
        assert res_a.error is None and res_b.error is None
        assert (await interp.run_cell("order")).result == ["A", "B"]
        assert res_a.output == "from-A\n" and res_b.output == "from-B\n"
    finally:
        await interp.stop()


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_display_keeps_images_between_the_text_around_them():
    interp = await actor_interpreter()
    code = """
from PIL import Image
img = Image.new("RGB", (10, 10), color="red")
print("before")
display(img)
print("after")
"""
    try:
        res = await interp.run_cell(code)
        assert res.error is None, res.text()
        parts = res.parts
        assert [type(p) for p in parts] == [TextPart, ImagePart, TextPart]
        assert "before" in parts[0].text and "after" in parts[2].text
        assert parts[1].mime == "image/png" and parts[1].data
        # The result the model reads keeps the order (serialize_tool_content defers to it).
        result = ExecutionResult(stdout=parts, stderr=[], result=None, error=None)
        assert isinstance(result, FormattedToolResult)
        blocks = serialize_tool_content(tool_name="execute_code", payload=result, is_final=True)
        kinds = [b.get("type") for b in blocks]
        assert kinds.index("image_url") < max(i for i, k in enumerate(kinds) if k == "text")
    finally:
        await interp.stop()


def test_output_part_is_a_discriminated_union():
    text = TextPart(text="hello")
    image = ImagePart(data="base64data", mime="image/png")
    assert (text.type, image.type) == ("text", "image")
    adapter = TypeAdapter(OutputPart)
    assert isinstance(adapter.validate_python({"type": "text", "text": "world"}), TextPart)
    parsed = adapter.validate_python({"type": "image", "data": "abc123", "mime": "image/jpeg"})
    assert isinstance(parsed, ImagePart) and parsed.data == "abc123"


def test_parts_to_llm_content_preserves_order_and_merges_adjacent_text():
    parts = [
        TextPart(text="before image\n"),
        ImagePart(data="img1_base64", mime="image/png"),
        TextPart(text="between\n"),
        TextPart(text="images\n"),
        ImagePart(data="img2_base64", mime="image/jpeg"),
        TextPart(text="after images\n"),
    ]
    assert parts_to_llm_content(parts) == [
        {"type": "text", "text": "before image\n"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,img1_base64"}},
        {"type": "text", "text": "between\nimages\n"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,img2_base64"}},
        {"type": "text", "text": "after images\n"},
    ]


def test_execution_result_is_a_formatted_tool_result():
    result = ExecutionResult(stdout=[TextPart(text="hello\n")], stderr=[], result=42, error=None)
    assert isinstance(result, FormattedToolResult)
    blocks = result.to_llm_content()
    assert blocks and all(isinstance(b, dict) and "type" in b for b in blocks)

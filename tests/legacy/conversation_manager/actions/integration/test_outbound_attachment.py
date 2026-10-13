"""
Outbound file attachment: CM → CodeActActor → generate image → send with attachment.

Validates the full production path:
1. User asks the actor to generate an image (red square)
2. Actor creates a .png file via execute_code
3. CM brain sends the result back with attachment_filepath
4. The attached file decodes to a red square on white

The slow brain and the actor's model are scripted (``scripted_brain``); the
actor's cell runs for real and writes the PNG with the standard library.
The image is checked by decoding its pixels instead of asking a vision
model what it shows.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import pytest

from tests.helpers import _handle_project
from tests.legacy.conversation_manager.actions.integration.helpers import (
    assert_no_errors,
    get_actor_started_event,
    inject_actor_result,
    run_cm_until_wait,
    scripted_brain_and_actor,
    wait_for_actor_completion,
    wait_until_done,
)
from tests.legacy.conversation_manager.actions.scripted_brain import (
    act,
    say,
    turn,
    wait,
)
from tests.scripted_model import cell, reply
from unify.legacy.conversation_manager.events import (
    UnifyMessageReceived,
    UnifyMessageSent,
)
from unify.workspace import get_local_root

pytestmark = [pytest.mark.integration]

IMAGE = "Outputs/red_square.png"

# The actor's cell: a 200x200 RGB PNG, white, with a red square in the middle.
DRAW_RED_SQUARE = """\
import os, struct, zlib
w = h = 200
red, white = b"\\xff\\x00\\x00", b"\\xff\\xff\\xff"
rows = b"".join(
    b"\\x00" + b"".join(red if 50 <= x < 150 and 50 <= y < 150 else white for x in range(w))
    for y in range(h)
)
def chunk(kind, data):
    return (
        struct.pack(">I", len(data)) + kind + data
        + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    )
png = (
    b"\\x89PNG\\r\\n\\x1a\\n"
    + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
    + chunk(b"IDAT", zlib.compress(rows))
    + chunk(b"IEND", b"")
)
os.makedirs("Outputs", exist_ok=True)
with open("Outputs/red_square.png", "wb") as f:
    f.write(png)
print("wrote", len(png), "bytes")
"""


def _pixels(path: Path) -> tuple[int, int, list[bytes]]:
    """Decode an 8-bit RGB PNG whose rows use no filter: (width, height, rows)."""
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n", "not a PNG"
    pos, idat, width, height = 8, b"", 0, 0
    while pos < len(data):
        (length,) = struct.unpack(">I", data[pos : pos + 4])
        kind = data[pos + 4 : pos + 8]
        body = data[pos + 8 : pos + 8 + length]
        if kind == b"IHDR":
            width, height, depth, colour = struct.unpack(">IIBB", body[:10])
            assert (depth, colour) == (8, 2), "expected 8-bit RGB"
        elif kind == b"IDAT":
            idat += body
        pos += 12 + length
    raw = zlib.decompress(idat)
    stride = 1 + 3 * width
    rows = [raw[y * stride : (y + 1) * stride] for y in range(height)]
    assert all(row[0] == 0 for row in rows), "expected unfiltered rows"
    return width, height, [row[1:] for row in rows]


def _at(rows: list[bytes], x: int, y: int) -> bytes:
    return rows[y][3 * x : 3 * x + 3]


@pytest.mark.asyncio
@pytest.mark.timeout(300)
@_handle_project
async def test_generate_image_and_send_as_attachment(
    initialized_cm_codeact,
    scripted_brain,
):
    """Generate a red square image via the actor and send it back as an attachment.

    Asserts:
    - The actor creates a .png file inside the workspace.
    - The brain is shown the action's result, and its send_unify_message
      with attachment_filepath goes out with the file attached.
    - The attached file decodes to a red square on a white background.
    """
    cm = initialized_cm_codeact
    local_root = Path(get_local_root())
    final_answer = f"Saved a 200x200 PNG of a red square on white to {IMAGE}."
    model = scripted_brain_and_actor(
        scripted_brain,
        reply(calls=[cell(DRAW_RED_SQUARE)]),
        reply(final_answer),
    )

    # ------------------------------------------------------------------
    # Step 1: Ask the actor to generate a red square image
    # ------------------------------------------------------------------
    model.next(
        turn(
            act(
                "Generate a 200x200 PNG of a solid red square on a white "
                "background and save it to the Outputs folder.",
            ),
            say("Making the image now."),
            wait(),
        ),
    )
    result = await cm.step_until_wait(
        UnifyMessageReceived(
            content=(
                "Please generate a simple PNG image of a solid red square on a white "
                "background (200x200 pixels) and save it to the Outputs folder. "
                "Send me the image file when done."
            ),
        ),
    )

    actor_event = get_actor_started_event(result)
    handle_id = actor_event.handle_id
    handle = cm.cm.in_flight_actions[handle_id]["handle"]
    final = await wait_for_actor_completion(cm, handle_id, timeout=120)
    assert final == final_answer
    assert_no_errors(result)

    # ------------------------------------------------------------------
    # Step 2: Verify the .png file was created inside the workspace
    # ------------------------------------------------------------------
    generated_image_path = local_root / IMAGE
    assert generated_image_path.is_file(), (
        f"Expected {IMAGE} in the workspace after the actor completed. "
        f"Actor result: {final}"
    )

    # ------------------------------------------------------------------
    # Step 3: Inject actor result and run CM brain; the outbound message
    #         carries the generated file as a local attachment
    # ------------------------------------------------------------------
    await inject_actor_result(
        cm,
        handle_id=handle_id,
        result=final,
        success=True,
    )
    model.next(
        turn(say("Here's your red square.", attachment_filepath=IMAGE), wait()),
    )
    followup_events = await run_cm_until_wait(cm, max_steps=6)

    # The brain was shown the action's result, naming the file.
    assert f"<result>{final_answer}</result>" in model.last_state()

    msg_events = [e for e in followup_events if isinstance(e, UnifyMessageSent)]
    attachment_events = [e for e in msg_events if e.attachments]
    assert len(attachment_events) == 1, (
        f"Expected one UnifyMessageSent with the image attached. Got "
        f"{[(e.content, e.attachments) for e in msg_events]}"
    )

    # A sent attachment resolves to a readable local file.
    (attachment,) = attachment_events[0].attachments
    assert attachment == IMAGE
    attachment_path = Path(attachment)
    if not attachment_path.is_absolute():
        attachment_path = local_root / attachment_path
    assert attachment_path.is_file(), f"Attachment is not a file: {attachment}"
    assert attachment_path.stat().st_size > 0, f"Attachment is empty: {attachment}"

    # ------------------------------------------------------------------
    # Step 4: The attached image is a red square on white
    # ------------------------------------------------------------------
    width, height, rows = _pixels(attachment_path)
    assert (width, height) == (200, 200)
    assert _at(rows, 100, 100) == b"\xff\x00\x00", "centre should be red"
    assert _at(rows, 60, 140) == b"\xff\x00\x00", "inside the square should be red"
    assert _at(rows, 5, 5) == b"\xff\xff\xff", "corner should be white"
    assert _at(rows, 195, 100) == b"\xff\xff\xff", "edge should be white"

    await wait_until_done(handle)

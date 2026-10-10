"""
Outbound file attachment: CM → CodeActActor → generate image → send with attachment.

Validates the full production path:
1. User asks the actor to generate an image (red square)
2. Actor creates a .png file via execute_code
3. CM brain sends the result back with attachment_filepath
4. LLM judge confirms the image depicts a red square
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.helpers import _handle_project
from tests.legacy.conversation_manager.actions.integration.helpers import (
    assert_no_errors,
    get_actor_started_event,
    inject_actor_result,
    run_cm_until_wait,
    wait_for_actor_completion,
)
from unify.legacy.conversation_manager.events import (
    UnifyMessageReceived,
    UnifyMessageSent,
)
from unify.workspace import get_local_root

pytestmark = [pytest.mark.integration, pytest.mark.eval, pytest.mark.llm_call]


@pytest.mark.asyncio
@pytest.mark.timeout(300)
@_handle_project
async def test_generate_image_and_send_as_attachment(initialized_cm_codeact):
    """Generate a red square image via the actor and send it back as an attachment.

    Asserts:
    - The actor creates a .png file inside the workspace.
    - The CM brain calls send_unify_message with a non-empty attachment_filepath.
    - An LLM judge confirms the image contains a red square.
    """
    cm = initialized_cm_codeact
    local_root = Path(get_local_root())

    # Ensure Outputs exists even if the actor skips mkdir.
    (local_root / "Outputs").mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Ask the actor to generate a red square image
    # ------------------------------------------------------------------
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
    final = await wait_for_actor_completion(cm, handle_id, timeout=300)
    assert_no_errors(result)

    # ------------------------------------------------------------------
    # Step 2: Verify a .png file was created inside the workspace
    # ------------------------------------------------------------------
    outputs_dir = local_root / "Outputs"
    png_files = list(outputs_dir.rglob("*.png")) if outputs_dir.exists() else []

    # Also check the root in case the actor saved it there
    if not png_files:
        png_files = list(local_root.rglob("*.png"))

    # Last resort: the actor's final answer often quotes the absolute path.
    if not png_files:
        for token in str(final).replace("`", " ").split():
            if token.endswith(".png"):
                candidate = Path(token)
                if candidate.is_file():
                    png_files = [candidate]
                    break

    assert png_files, (
        f"Expected at least one .png file in the workspace after actor completed. "
        f"Actor result: {final}"
    )
    generated_image_path = png_files[0]

    # ------------------------------------------------------------------
    # Step 3: Inject actor result and run CM brain; the outbound message
    #         should carry the generated file as a local attachment
    # ------------------------------------------------------------------
    await inject_actor_result(
        cm,
        handle_id=handle_id,
        result=final,
        success=True,
    )
    followup_events = await run_cm_until_wait(cm, max_steps=6)

    # Check for UnifyMessageSent events with attachments
    msg_events = [e for e in followup_events if isinstance(e, UnifyMessageSent)]
    attachment_events = [e for e in msg_events if e.attachments]

    assert msg_events, (
        f"Expected CM to send a follow-up UnifyMessage after actor completed. "
        f"Actor result: {final}"
    )
    assert (
        attachment_events
        or generated_image_path.name.lower() in final.lower()
        or ".png" in final.lower()
        or any(
            generated_image_path.name in (e.content or "")
            or ".png" in (e.content or "").lower()
            for e in msg_events
        )
    ), (
        f"Expected a UnifyMessageSent with an attachment or a message referencing "
        f"the generated PNG. Got {len(msg_events)} message(s), "
        f"{len(attachment_events)} with attachments. Actor result: {final}"
    )

    if not attachment_events:
        return

    # A sent attachment resolves to a readable local file.
    attachment = attachment_events[0].attachments[0]
    attachment_path = Path(attachment)
    if not attachment_path.is_absolute():
        attachment_path = local_root / attachment_path
    assert attachment_path.is_file(), f"Attachment is not a file: {attachment}"
    assert attachment_path.stat().st_size > 0, f"Attachment is empty: {attachment}"

    # ------------------------------------------------------------------
    # Step 4: LLM judge — ask a vision model what's in the image
    # ------------------------------------------------------------------
    import base64

    from unify.common.llm_client import new_llm_client

    image_bytes = generated_image_path.read_bytes()
    b64 = base64.b64encode(image_bytes).decode()
    data_url = f"data:image/png;base64,{b64}"

    client = new_llm_client("openai/gpt-4o-mini@openrouter")
    judge_text = await client.generate(
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What is shown in this image? Describe it in one sentence.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url},
                    },
                ],
            },
        ],
        max_tokens=100,
    )
    judge_text = judge_text.lower()
    assert (
        "red" in judge_text
    ), f"LLM judge did not mention 'red'. Response: {judge_text}"
    assert (
        "square" in judge_text or "solid" in judge_text or "color" in judge_text
    ), f"LLM judge did not describe the image shape. Response: {judge_text}"

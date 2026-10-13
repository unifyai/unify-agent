from __future__ import annotations

import base64
from pathlib import Path

from PIL import Image

from unify.common.image_content import sniff_image_mime, to_image_content_block

ASSET_JPEG = Path(__file__).parent.parent / "images" / "gcp_homepage.jpg"


def test_sniff_image_mime_detects_jpeg_and_png(tmp_path):
    assert sniff_image_mime(ASSET_JPEG.read_bytes()) == "image/jpeg"

    png_path = tmp_path / "sniff.png"
    Image.new("RGB", (4, 4), color=(255, 0, 0)).save(png_path, format="PNG")
    assert sniff_image_mime(png_path.read_bytes()) == "image/png"


def test_to_image_content_block_from_local_jpeg_path():
    block = to_image_content_block(str(ASSET_JPEG))

    assert block["type"] == "image_url"
    url = block["image_url"]["url"]
    assert url.startswith("data:image/jpeg;base64,")
    assert base64.b64decode(url.split(",", 1)[1]) == ASSET_JPEG.read_bytes()


def test_to_image_content_block_passes_through_http_url():
    url = "https://example.com/screenshot.png"
    block = to_image_content_block(url)

    assert block == {"type": "image_url", "image_url": {"url": url}}

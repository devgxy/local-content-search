"""deskew 轻量测试。"""

from __future__ import annotations

from PIL import Image, ImageDraw

from app.parsing.deskew import deskew_image, estimate_skew


def test_deskew_rotated_text_lines() -> None:
    image = Image.new("RGB", (800, 600), "white")
    draw = ImageDraw.Draw(image)
    for i, y in enumerate(range(100, 500, 40)):
        draw.text((100, y), f"line {i} text for deskew", fill="black")
    rotated = image.rotate(8, expand=False, fillcolor="white")
    angle = estimate_skew(rotated)
    assert abs(angle) >= 5
    corrected, applied = deskew_image(rotated)
    assert abs(applied) >= 5
    assert corrected.size == rotated.size

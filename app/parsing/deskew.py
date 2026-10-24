"""轻量 deskew：用投影方差估计文本行倾角并校正页面。"""

from __future__ import annotations

import numpy as np
from PIL import Image


def _projection_score(image: Image.Image) -> float:
    gray = np.asarray(image.convert("L").resize((800, max(1, int(800 * image.height / image.width)))))
    ink = 1.0 - gray.astype(np.float32) / 255.0
    rows = ink.sum(axis=1)
    return float(np.var(rows))


def estimate_skew(image: Image.Image, *, max_angle: float = 15.0, step: float = 1.0) -> float:
    """返回使水平投影最锐利的校正角；输出角度用于 rotate。"""
    if image.width < 40 or image.height < 40:
        return 0.0
    gray = np.asarray(image.convert("L").resize((800, max(1, int(800 * image.height / image.width)))))
    ink = 1.0 - gray.astype(np.float32) / 255.0
    rows = ink.sum(axis=1)
    # 统计“连续墨迹行段”而不是墨迹像素行；文字行太少时投影不可靠。
    threshold = max(float(rows.max()) * 0.15, image.width * 0.01)
    runs = 0
    active = False
    for value in rows:
        if value >= threshold and not active:
            runs += 1
            active = True
        elif value < threshold:
            active = False
    if runs < 4:
        return 0.0
    zero_score = _projection_score(image)
    best_angle = 0.0
    best_score = -1.0
    angle = -max_angle
    while angle <= max_angle + 1e-9:
        rotated = image.rotate(angle, resample=Image.Resampling.NEAREST, expand=False, fillcolor="white")
        score = _projection_score(rotated)
        if score > best_score:
            best_score = score
            best_angle = angle
        angle += step
    # 没有明显改善时不做 deskew，避免把正常页面转坏。
    if best_score < zero_score * 1.05:
        return 0.0
    return best_angle


def deskew_image(
    image: Image.Image,
    *,
    max_angle: float = 15.0,
    step: float = 1.0,
) -> tuple[Image.Image, float]:
    """校正倾斜页面；返回 (图像, 应用角度)。"""
    angle = estimate_skew(image, max_angle=max_angle, step=step)
    if abs(angle) < 0.5:
        return image, 0.0
    return (
        image.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False, fillcolor=255),
        angle,
    )

"""图片一致性比较工具。"""

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


DEFAULT_PIXEL_TOLERANCE = 10
DEFAULT_MIN_MATCH_RATIO = 0.95
DEFAULT_MAX_MEAN_ABSOLUTE_ERROR = 3.0


def compare_images(
    reference_path: str | Path,
    candidate_path: str | Path,
    diff_output_path: str | Path | None = None,
    pixel_tolerance: int = DEFAULT_PIXEL_TOLERANCE,
    min_match_ratio: float = DEFAULT_MIN_MATCH_RATIO,
    max_mean_absolute_error: float = DEFAULT_MAX_MEAN_ABSOLUTE_ERROR,
) -> dict[str, Any]:
    """尺寸完全一致时统一为 RGB，并执行带容差的逐像素比较。"""
    reference_file = Path(reference_path)
    candidate_file = Path(candidate_path)
    if not reference_file.is_file():
        raise FileNotFoundError(f"参考图片不存在: {reference_file}")
    if not candidate_file.is_file():
        raise FileNotFoundError(f"待比较图片不存在: {candidate_file}")
    if not 0 <= pixel_tolerance <= 255:
        raise ValueError("pixel_tolerance 必须在 0 到 255 之间")
    if not 0 <= min_match_ratio <= 1:
        raise ValueError("min_match_ratio 必须在 0 到 1 之间")
    if max_mean_absolute_error < 0:
        raise ValueError("max_mean_absolute_error 不能小于 0")

    with Image.open(reference_file) as reference_source, Image.open(candidate_file) as candidate_source:
        reference_size = reference_source.size
        candidate_size = candidate_source.size
        reference_mode = reference_source.mode
        candidate_mode = candidate_source.mode
        dimension_match = reference_size == candidate_size

        if not dimension_match:
            return {
                "is_consistent": False,
                "dimension_match": False,
                "reference_path": str(reference_file),
                "candidate_path": str(candidate_file),
                "reference_size": list(reference_size),
                "candidate_size": list(candidate_size),
                "reference_mode": reference_mode,
                "candidate_mode": candidate_mode,
                "comparison_size": None,
                "pixel_tolerance": pixel_tolerance,
                "min_match_ratio": min_match_ratio,
                "max_mean_absolute_error": max_mean_absolute_error,
                "matched_pixel_ratio": None,
                "similarity_percent": None,
                "mean_absolute_error": None,
                "max_absolute_error": None,
                "diff_path": None,
                "failure_reason": "图片尺寸不一致，未执行像素比较",
            }

        reference_image = reference_source.convert("RGB")
        candidate_image = candidate_source.convert("RGB")

        reference_pixels = np.asarray(reference_image, dtype=np.int16)
        candidate_pixels = np.asarray(candidate_image, dtype=np.int16)

    absolute_difference = np.abs(reference_pixels - candidate_pixels)
    per_pixel_max_difference = absolute_difference.max(axis=2)
    matched_pixel_ratio = float(np.mean(per_pixel_max_difference <= pixel_tolerance))
    mean_absolute_error = float(np.mean(absolute_difference))
    max_absolute_error = int(absolute_difference.max())
    similarity_percent = matched_pixel_ratio * 100.0
    is_consistent = (
        matched_pixel_ratio >= min_match_ratio
        and mean_absolute_error <= max_mean_absolute_error
    )

    saved_diff_path = None
    if diff_output_path is not None:
        diff_file = Path(diff_output_path)
        diff_file.parent.mkdir(parents=True, exist_ok=True)
        # 放大细微差异，便于在 Allure 报告中定位问题区域。
        visible_difference = np.clip(absolute_difference * 4, 0, 255).astype(np.uint8)
        Image.fromarray(visible_difference).save(diff_file, format="PNG")
        saved_diff_path = str(diff_file)

    return {
        "is_consistent": is_consistent,
        "dimension_match": True,
        "reference_path": str(reference_file),
        "candidate_path": str(candidate_file),
        "reference_size": list(reference_size),
        "candidate_size": list(candidate_size),
        "reference_mode": reference_mode,
        "candidate_mode": candidate_mode,
        "comparison_size": list(candidate_size),
        "pixel_tolerance": pixel_tolerance,
        "min_match_ratio": min_match_ratio,
        "max_mean_absolute_error": max_mean_absolute_error,
        "matched_pixel_ratio": matched_pixel_ratio,
        "similarity_percent": similarity_percent,
        "mean_absolute_error": mean_absolute_error,
        "max_absolute_error": max_absolute_error,
        "diff_path": saved_diff_path,
        "failure_reason": None if is_consistent else "图片像素差异超过阈值",
    }

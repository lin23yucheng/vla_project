"""
图像对比功能单元测试。

测试范围：
- 尺寸不一致时返回不匹配
- 尺寸和内容完全一致时通过
- 内容不同（像素值差异）时返回不匹配
"""

from pathlib import Path

from PIL import Image

from common.image_compare import compare_images


def test_compare_images_rejects_dimension_mismatch(tmp_path: Path):
    """当参考图与候选图尺寸不一致时，应判定为不一致，且不生成差异图。"""
    reference_path = tmp_path / "mcap.png"
    candidate_path = tmp_path / "parquet.png"
    diff_path = tmp_path / "diff.png"

    reference = Image.new("RGB", (64, 48), (25, 100, 180))
    reference.save(reference_path)
    reference.resize((48, 64), Image.Resampling.LANCZOS).save(candidate_path)

    result = compare_images(reference_path, candidate_path, diff_path)

    assert result["is_consistent"] is False
    assert result["dimension_match"] is False
    assert result["matched_pixel_ratio"] is None
    assert not diff_path.exists()


def test_compare_images_accepts_same_dimensions_and_content(tmp_path: Path):
    """当参考图与候选图尺寸和内容完全一致时，应判定为一致。"""
    reference_path = tmp_path / "mcap.png"
    candidate_path = tmp_path / "parquet.png"
    image = Image.new("RGB", (64, 48), (25, 100, 180))
    image.save(reference_path)
    image.save(candidate_path)

    result = compare_images(reference_path, candidate_path)

    assert result["is_consistent"] is True
    assert result["dimension_match"] is True
    assert result["matched_pixel_ratio"] == 1.0


def test_compare_images_rejects_different_content(tmp_path: Path):
    """当参考图与候选图尺寸相同但内容完全不同（黑 vs 白）时，应判定为不一致。"""
    reference_path = tmp_path / "mcap.png"
    candidate_path = tmp_path / "parquet.png"
    Image.new("RGB", (32, 32), (0, 0, 0)).save(reference_path)
    Image.new("RGB", (32, 32), (255, 255, 255)).save(candidate_path)

    result = compare_images(reference_path, candidate_path)

    assert result["is_consistent"] is False
    assert result["matched_pixel_ratio"] == 0.0
    assert result["mean_absolute_error"] == 255.0

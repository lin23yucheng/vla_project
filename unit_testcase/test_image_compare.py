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
    assert result["reference_resized"] is False
    assert result["gaussian_blur_applied"] is False
    assert result["matched_pixel_ratio"] is None
    assert not diff_path.exists()


def test_compare_images_can_resize_reference_when_explicitly_enabled(tmp_path: Path):
    """特殊构型可显式缩放参考图，但默认严格尺寸行为保持不变。"""
    reference_path = tmp_path / "mcap.png"
    candidate_path = tmp_path / "parquet.png"
    Image.new("RGB", (96, 74), (25, 100, 180)).save(reference_path)
    Image.new("RGB", (64, 48), (25, 100, 180)).save(candidate_path)

    result = compare_images(
        reference_path,
        candidate_path,
        resize_reference_to_candidate=True,
    )

    assert result["is_consistent"] is True
    assert result["dimension_match"] is True
    assert result["source_dimension_match"] is False
    assert result["reference_resized"] is True
    assert result["reference_resize_resampling"] == "BICUBIC"
    assert result["reference_size"] == [96, 74]
    assert result["comparison_size"] == [64, 48]


def test_compare_images_can_apply_gaussian_blur_when_explicitly_enabled(tmp_path: Path):
    """高斯平滑必须显式启用，并在比较结果中记录预处理参数。"""
    reference_path = tmp_path / "mcap.png"
    candidate_path = tmp_path / "parquet.png"
    Image.new("RGB", (64, 48), (25, 100, 180)).save(reference_path)
    Image.new("RGB", (64, 48), (25, 100, 180)).save(candidate_path)

    result = compare_images(
        reference_path,
        candidate_path,
        gaussian_blur_radius=1.0,
    )

    assert result["is_consistent"] is True
    assert result["gaussian_blur_applied"] is True
    assert result["gaussian_blur_radius"] == 1.0


def test_compare_images_accepts_same_dimensions_and_content(tmp_path: Path):
    """当参考图与候选图尺寸和内容完全一致时，应判定为一致。"""
    reference_path = tmp_path / "mcap.png"
    candidate_path = tmp_path / "parquet.png"
    diff_path = tmp_path / "diff.png"
    image = Image.new("RGB", (64, 48), (25, 100, 180))
    image.save(reference_path)
    image.save(candidate_path)

    result = compare_images(reference_path, candidate_path, diff_path)

    assert result["is_consistent"] is True
    assert result["dimension_match"] is True
    assert result["matched_pixel_ratio"] == 1.0
    assert result["diff_path"] is None
    assert not diff_path.exists()


def test_compare_images_rejects_different_content(tmp_path: Path):
    """当参考图与候选图尺寸相同但内容完全不同（黑 vs 白）时，应判定为不一致。"""
    reference_path = tmp_path / "mcap.png"
    candidate_path = tmp_path / "parquet.png"
    diff_path = tmp_path / "diff.png"
    Image.new("RGB", (32, 32), (0, 0, 0)).save(reference_path)
    Image.new("RGB", (32, 32), (255, 255, 255)).save(candidate_path)

    result = compare_images(reference_path, candidate_path, diff_path)

    assert result["is_consistent"] is False
    assert result["matched_pixel_ratio"] == 0.0
    assert result["mean_absolute_error"] == 255.0
    assert result["diff_path"] == str(diff_path)
    assert diff_path.is_file()

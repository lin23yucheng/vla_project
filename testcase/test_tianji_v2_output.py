"""天机 V2 校验流程的输出文案与跨数据源时间对齐回归测试。"""

from typing import Any

import pytest

import testcase.test_tianji_v2_verify as tianji


def test_image_comparison_summary_explains_dimension_mismatch() -> None:
    """未启用天机缩放时，尺寸错误应包含两侧的实际分辨率。"""
    summary = tianji.format_image_comparison_summary(
        {
            "dimension_match": False,
            "reference_size": [960, 744],
            "candidate_size": [640, 480],
        }
    )

    assert summary == (
        "尺寸不一致：MCAP=960x744，Parquet=640x480；"
        "尺寸不同，未执行像素比较"
    )


def test_image_comparison_summary_explains_tianji_resize() -> None:
    """启用天机缩放后，摘要应说明缩放方式及完整像素指标。"""
    summary = tianji.format_image_comparison_summary(
        {
            "dimension_match": True,
            "reference_resized": True,
            "reference_resize_resampling": "BICUBIC",
            "gaussian_blur_applied": True,
            "gaussian_blur_radius": 1.0,
            "reference_size": [960, 744],
            "comparison_size": [640, 480],
            "similarity_percent": 98.5,
            "mean_absolute_error": 1.25,
            "max_absolute_error": 15,
            "min_match_ratio": 0.85,
            "max_mean_absolute_error": 4.5,
        }
    )

    assert summary == (
        "MCAP已从960x744缩放为640x480（BICUBIC）；"
        "两侧已进行高斯平滑（半径=1.0）；"
        "像素匹配率=98.500%（要求>=85.000%），"
        "平均像素误差=1.250000（要求<=4.5），最大像素误差=15"
    )


def test_robot_vector_validation_aligns_mcap_to_parquet_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MCAP 向量必须对齐 Parquet 实际帧，不能再次按标注时间取相邻帧。"""
    annotation_target_ns = 100
    parquet_matched_ns = 120
    captured_mcap_targets: list[int] = []
    vectors = [
        {"section": "action", "group": "left"}
        for _ in range(4)
    ]

    def fake_extract_parquet(**_kwargs: Any) -> dict[str, Any]:
        return {
            "target_ns": annotation_target_ns,
            "matched_timestamp_ns": parquet_matched_ns,
            "vectors": vectors,
        }

    monkeypatch.setattr(
        tianji,
        "extract_parquet_robot_vectors_at_time",
        fake_extract_parquet,
    )

    def fake_extract_mcap(**kwargs: Any) -> dict[str, Any]:
        mcap_target_ns = int(kwargs["target_ns"])
        captured_mcap_targets.append(mcap_target_ns)
        return {
            "target_ns": mcap_target_ns,
            "main_frame_ns": mcap_target_ns,
            "vectors": vectors,
        }

    def fake_compare_vectors(**_kwargs: Any) -> dict[str, Any]:
        return {
            "is_consistent": True,
            "absolute_tolerance": 1e-5,
            "comparisons": [],
        }

    monkeypatch.setattr(tianji, "extract_robot_vectors_at_time", fake_extract_mcap)
    monkeypatch.setattr(tianji, "compare_robot_vectors", fake_compare_vectors)

    verifier = tianji.TestTianjiV2Verify()
    verifier.parquet_sources = [object()]
    verifier.robot_config_path = "robot.json"
    verifier.mcap_sources = [object()]
    verifier.mcap_source_label = "test"
    # noinspection PyProtectedMember
    result = verifier._verify_one_robot_vector_time(
        segment_key="l1_01",
        time_key="start",
        target_ns=annotation_target_ns,
    )

    assert captured_mcap_targets == [parquet_matched_ns]
    assert result["mcap_result"]["annotation_target_ns"] == annotation_target_ns
    assert result["mcap_result"]["main_frame_ns"] == parquet_matched_ns

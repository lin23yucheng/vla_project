"""
Parquet 字段提取与对比功能单元测试。

测试范围：
- 对比两组完全相同的机器人向量数据，验证一致性判断
- 按 display 字段解析左右各七维的完整布局
- 明确拒绝转换后缺少夹爪 angle 的不完整数据
"""

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from common.extract_parquet_fields import (
    VECTOR_LABELS,
    compare_robot_vectors,
    extract_parquet_robot_vectors_at_time,
)


POSE_LABELS = VECTOR_LABELS[:-1]


def _write_robot_parquet(path, include_angle: bool, gripper_field: str = "angle") -> None:
    display_fields = []
    for group in ("left", "right"):
        display_fields.extend(f"{group}_joint_states.{label}" for label in POSE_LABELS)
        if include_angle:
            display_fields.append(f"{group}_gripper.{gripper_field}")
    values = [float(index) for index in range(1, len(display_fields) + 1)]
    display_type = pa.struct([pa.field(name, pa.float32()) for name in display_fields])
    display_value = dict(zip(display_fields, values))
    table = pa.table(
        {
            "original_timestamp_ns": pa.array([100], type=pa.int64()),
            "actions": pa.array([values], type=pa.list_(pa.float32())),
            "state": pa.array([values], type=pa.list_(pa.float32())),
            "ac_display": pa.array([display_value], type=display_type),
            "st_display": pa.array([display_value], type=display_type),
        }
    )
    pq.write_table(table, path)


def test_compare_robot_vectors_accepts_equal_values():
    """当两组机器人向量数据完全相同时，应判定为一致。"""
    vectors = [
        {"section": section, "group": group, "vector_labels": ["x"], "vector": [1.0]}
        for section in ("action", "observation_state")
        for group in ("left", "right")
    ]

    result = compare_robot_vectors({"vectors": vectors}, {"vectors": vectors})

    assert result["is_consistent"] is True
    assert result["absolute_tolerance"] == 5e-04


def test_extract_parquet_robot_vectors_supports_complete_layout(tmp_path):
    parquet_path = tmp_path / "robot.parquet"
    _write_robot_parquet(parquet_path, include_angle=True)

    result = extract_parquet_robot_vectors_at_time(parquet_path, target_ns=101)

    assert len(result["vectors"]) == 4
    for item in result["vectors"]:
        assert item["vector_labels"] == VECTOR_LABELS
        assert item["missing_vector_labels"] == []
        assert len(item["vector"]) == len(VECTOR_LABELS)


def test_extract_parquet_robot_vectors_maps_tianji_data_index_to_angle(tmp_path):
    parquet_path = tmp_path / "tianji_robot.parquet"
    _write_robot_parquet(parquet_path, include_angle=True, gripper_field="data[0]")

    result = extract_parquet_robot_vectors_at_time(parquet_path, target_ns=101)

    assert len(result["vectors"]) == 4
    for item in result["vectors"]:
        assert item["vector_labels"] == VECTOR_LABELS
        assert item["display_fields"][-1].endswith("_gripper.data[0]")
        assert item["values_by_label"]["angle"] == item["vector"][-1]


def test_extract_parquet_robot_vectors_rejects_missing_angle(tmp_path):
    parquet_path = tmp_path / "robot.parquet"
    _write_robot_parquet(parquet_path, include_angle=False)

    with pytest.raises(ValueError, match="left缺少=.*angle.*right缺少=.*angle"):
        extract_parquet_robot_vectors_at_time(parquet_path, target_ns=101)


def test_compare_robot_vectors_rejects_missing_parquet_angle():
    mcap_vectors = [
        {
            "section": section,
            "group": group,
            "vector_labels": VECTOR_LABELS,
            "vector": [1.0] * 7,
        }
        for section in ("action", "observation_state")
        for group in ("left", "right")
    ]
    parquet_vectors = [
        {
            "section": section,
            "group": group,
            "vector_labels": POSE_LABELS,
            "vector": [1.0] * 6,
        }
        for section in ("action", "observation_state")
        for group in ("left", "right")
    ]

    with pytest.raises(ValueError, match="Parquet缺少=.*angle"):
        compare_robot_vectors(
            {"vectors": mcap_vectors},
            {"vectors": parquet_vectors},
        )


def test_compare_robot_vectors_rejects_missing_pose_dimension():
    vectors = [
        {
            "section": section,
            "group": group,
            "vector_labels": POSE_LABELS,
            "vector": [1.0] * 6,
        }
        for section in ("action", "observation_state")
        for group in ("left", "right")
    ]
    incomplete_vectors = [dict(item, vector_labels=POSE_LABELS[:-1], vector=[1.0] * 5) for item in vectors]

    with pytest.raises(ValueError, match="Parquet缺少"):
        compare_robot_vectors({"vectors": vectors}, {"vectors": incomplete_vectors})

"""
Parquet 字段提取与对比功能单元测试。

测试范围：
- 对比两组完全相同的机器人向量数据，验证一致性判断
"""

from common.extract_parquet_fields import compare_robot_vectors


def test_compare_robot_vectors_accepts_equal_values():
    """当两组机器人向量数据完全相同时，应判定为一致。"""
    vectors = [
        {"section": section, "group": group, "vector_labels": ["x"], "vector": [1.0]}
        for section in ("action", "observation_state")
        for group in ("left", "right")
    ]

    result = compare_robot_vectors({"vectors": vectors}, {"vectors": vectors})

    assert result["is_consistent"] is True

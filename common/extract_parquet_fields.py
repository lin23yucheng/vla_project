"""从 Parquet 最近时间行提取 action/state 七维向量。"""

from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


VECTOR_LABELS = ["x", "y", "z", "roll", "pitch", "yaw", "angle"]
PARQUET_VECTOR_MAPPING = {
    ("action", "left"): ("actions", 0),
    ("action", "right"): ("actions", 7),
    ("observation_state", "left"): ("state", 0),
    ("observation_state", "right"): ("state", 7),
}


def extract_parquet_robot_vectors_at_time(
    parquet_path: str | Path,
    target_ns: int,
) -> dict[str, Any]:
    """选择最近时间行，按左右各 7 维切分 actions 和 state。"""
    parquet_file_path = Path(parquet_path)
    if not parquet_file_path.is_file():
        raise FileNotFoundError(f"parquet 文件不存在: {parquet_file_path}")

    required_columns = ["original_timestamp_ns", "actions", "state", "ac_display", "st_display"]
    parquet_file = pq.ParquetFile(parquet_file_path)
    missing_columns = [column for column in required_columns if column not in parquet_file.schema_arrow.names]
    if missing_columns:
        raise ValueError(f"parquet 缺少机器人向量列: {missing_columns}")

    rows = parquet_file.read(columns=required_columns).to_pylist()
    target_ns = int(target_ns)
    usable_rows = [row for row in rows if isinstance(row.get("original_timestamp_ns"), int)]
    if not usable_rows:
        raise ValueError("parquet 中没有有效 original_timestamp_ns 数据")
    nearest_row = min(usable_rows, key=lambda row: abs(row["original_timestamp_ns"] - target_ns))
    matched_timestamp_ns = nearest_row["original_timestamp_ns"]

    vectors = []
    for (section, group), (column, start_index) in PARQUET_VECTOR_MAPPING.items():
        source_vector = nearest_row.get(column)
        if not isinstance(source_vector, list) or len(source_vector) < start_index + 7:
            raise ValueError(f"parquet {column} 长度不足，无法提取 {section}/{group} 七维向量")
        vector = [float(value) for value in source_vector[start_index:start_index + 7]]
        display_column = "ac_display" if section == "action" else "st_display"
        display_values = nearest_row.get(display_column)
        vectors.append(
            {
                "section": section,
                "group": group,
                "parquet_column": column,
                "slice": [start_index, start_index + 7],
                "vector_labels": VECTOR_LABELS,
                "vector": vector,
                "values_by_label": dict(zip(VECTOR_LABELS, vector)),
                "display_column": display_column,
                "display_values": display_values,
                "target_ns": target_ns,
                "matched_timestamp_ns": matched_timestamp_ns,
                "diff_ns": abs(matched_timestamp_ns - target_ns),
            }
        )

    return {
        "parquet_file": str(parquet_file_path),
        "target_ns": target_ns,
        "matched_timestamp_ns": matched_timestamp_ns,
        "diff_ns": abs(matched_timestamp_ns - target_ns),
        "vector_labels": VECTOR_LABELS,
        "vectors": vectors,
    }


def compare_robot_vectors(
    mcap_result: dict[str, Any],
    parquet_result: dict[str, Any],
    absolute_tolerance: float = 1e-5,
) -> dict[str, Any]:
    """按 section/group 和七维下标比较 MCAP 转换值与 Parquet 原始值。"""
    mcap_vectors = {
        (item["section"], item["group"]): item
        for item in mcap_result.get("vectors", [])
    }
    parquet_vectors = {
        (item["section"], item["group"]): item
        for item in parquet_result.get("vectors", [])
    }
    keys = [
        ("action", "left"),
        ("action", "right"),
        ("observation_state", "left"),
        ("observation_state", "right"),
    ]
    comparisons = []
    all_consistent = True
    for key in keys:
        if key not in mcap_vectors or key not in parquet_vectors:
            raise ValueError(f"缺少向量，无法比较: section={key[0]}, group={key[1]}")
        mcap_item = mcap_vectors[key]
        parquet_item = parquet_vectors[key]
        labels = mcap_item["vector_labels"]
        if labels != parquet_item["vector_labels"]:
            raise ValueError(f"向量标签不一致: {labels} != {parquet_item['vector_labels']}")

        dimensions = []
        for label, mcap_value, parquet_value in zip(labels, mcap_item["vector"], parquet_item["vector"]):
            difference = abs(float(mcap_value) - float(parquet_value))
            is_consistent = difference <= absolute_tolerance
            all_consistent = all_consistent and is_consistent
            dimensions.append(
                {
                    "label": label,
                    "mcap_value": float(mcap_value),
                    "parquet_value": float(parquet_value),
                    "absolute_difference": difference,
                    "is_consistent": is_consistent,
                }
            )
        comparisons.append(
            {
                "section": key[0],
                "group": key[1],
                "is_consistent": all(item["is_consistent"] for item in dimensions),
                "dimensions": dimensions,
            }
        )

    return {
        "is_consistent": all_consistent,
        "absolute_tolerance": absolute_tolerance,
        "vector_labels": VECTOR_LABELS,
        "comparisons": comparisons,
    }

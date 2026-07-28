"""从 Parquet 最近时间行提取 action/state 机器人向量。"""

from typing import Any

from common.parquet_source import (
    find_nearest_parquet_row,
    read_matched_parquet_row,
)


VECTOR_LABELS = ["x", "y", "z", "roll", "pitch", "yaw", "angle"]
PARQUET_SECTION_COLUMNS = {
    "action": ("actions", "ac_display"),
    "observation_state": ("state", "st_display"),
}


def _infer_display_group(field_name: str) -> str | None:
    normalized = field_name.lower().replace("-", "_")
    tokens = normalized.replace(".", "_").split("_")
    if "left" in tokens or normalized.startswith("left"):
        return "left"
    if "right" in tokens or normalized.startswith("right"):
        return "right"
    return None


def _infer_vector_label(field_name: str) -> str | None:
    """将转换后的展示字段归一化为标准七维标签。"""
    normalized = field_name.lower().replace("-", "_")
    label = normalized.rsplit(".", 1)[-1]
    if label in VECTOR_LABELS:
        return label
    if label == "data[0]" and "gripper" in normalized.replace(".", "_").split("_"):
        return "angle"
    return None


def _build_group_layout(
    source_vector: list[Any],
    display_values: dict[str, Any],
    section: str,
    column: str,
    display_column: str,
) -> dict[str, dict[str, Any]]:
    display_fields = list(display_values)
    if len(display_fields) != len(source_vector):
        raise ValueError(
            f"parquet {column} 与 {display_column} 长度不一致: "
            f"{len(source_vector)} != {len(display_fields)}"
        )

    grouped_fields: dict[str, dict[str, tuple[int, str]]] = {"left": {}, "right": {}}
    for index, field_name in enumerate(display_fields):
        group = _infer_display_group(field_name)
        label = _infer_vector_label(field_name)
        if group is None or label is None:
            raise ValueError(f"无法识别 parquet {display_column} 字段: {field_name}")
        if label in grouped_fields[group]:
            raise ValueError(f"parquet {display_column} 中 {group}/{label} 字段重复")
        grouped_fields[group][label] = (index, field_name)

    missing_by_group = {
        group: [label for label in VECTOR_LABELS if label not in fields_by_label]
        for group, fields_by_label in grouped_fields.items()
    }
    if any(missing_by_group.values()):
        missing_summary = ", ".join(
            f"{group}缺少={missing}"
            for group, missing in missing_by_group.items()
            if missing
        )
        raise ValueError(
            f"parquet 转换数据缺少 {section} 七维字段: {missing_summary}; "
            f"{column}长度={len(source_vector)}, "
            f"{display_column}字段={display_fields}"
        )

    layouts = {}
    for group, fields_by_label in grouped_fields.items():
        labels = list(VECTOR_LABELS)
        indices = [fields_by_label[label][0] for label in labels]
        field_names = [fields_by_label[label][1] for label in labels]
        is_contiguous = indices == list(range(min(indices), max(indices) + 1))
        layouts[group] = {
            "labels": labels,
            "indices": indices,
            "field_names": field_names,
            "slice": [min(indices), max(indices) + 1] if is_contiguous else None,
        }
    return layouts


def extract_parquet_robot_vectors_at_time(
    parquet_path: Any,
    target_ns: int,
) -> dict[str, Any]:
    """跨本地或远端 parquet 源选择最近时间行并解析左右臂向量。"""
    required_columns = ["timestamp", "original_timestamp_ns", "actions", "state", "ac_display", "st_display"]
    target_ns = int(target_ns)
    match = find_nearest_parquet_row(parquet_path, target_ns)
    nearest_row, _ = read_matched_parquet_row(match, required_columns)
    matched_timestamp_ns = nearest_row["original_timestamp_ns"]
    output_timestamp = float(nearest_row["timestamp"])

    vectors = []
    for section, (column, display_column) in PARQUET_SECTION_COLUMNS.items():
        source_vector = nearest_row.get(column)
        display_values = nearest_row.get(display_column)
        if not isinstance(source_vector, list):
            raise ValueError(f"parquet {column} 不是有效向量")
        if not isinstance(display_values, dict):
            raise ValueError(f"parquet {display_column} 不是有效字段映射")

        layouts = _build_group_layout(
            source_vector,
            display_values,
            section,
            column,
            display_column,
        )
        for group in ("left", "right"):
            layout = layouts[group]
            labels = layout["labels"]
            vector = [float(source_vector[index]) for index in layout["indices"]]
            vectors.append(
                {
                    "section": section,
                    "group": group,
                    "parquet_column": column,
                    "slice": layout["slice"],
                    "source_indices": layout["indices"],
                    "vector_labels": labels,
                    "missing_vector_labels": [label for label in VECTOR_LABELS if label not in labels],
                    "vector": vector,
                    "values_by_label": dict(zip(labels, vector)),
                    "display_column": display_column,
                    "display_fields": layout["field_names"],
                    "display_values": display_values,
                    "target_ns": target_ns,
                    "matched_timestamp_ns": matched_timestamp_ns,
                    "output_timestamp": output_timestamp,
                    "diff_ns": abs(matched_timestamp_ns - target_ns),
                }
            )

    available_label_sets = {tuple(item["vector_labels"]) for item in vectors}
    return {
        "parquet_file": match["source_name"],
        "matched_row_group_index": match["row_group_index"],
        "matched_row_index": match["row_index"],
        "target_ns": target_ns,
        "matched_timestamp_ns": matched_timestamp_ns,
        "output_timestamp": output_timestamp,
        "diff_ns": abs(matched_timestamp_ns - target_ns),
        "vector_labels": list(next(iter(available_label_sets))) if len(available_label_sets) == 1 else None,
        "expected_vector_labels": VECTOR_LABELS,
        "vectors": vectors,
    }


def compare_robot_vectors(
    mcap_result: dict[str, Any],
    parquet_result: dict[str, Any],
    absolute_tolerance: float = 5e-04,
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
        mcap_labels = mcap_item["vector_labels"]
        parquet_labels = parquet_item["vector_labels"]
        mcap_values = dict(zip(mcap_labels, mcap_item["vector"]))
        parquet_values = dict(zip(parquet_labels, parquet_item["vector"]))
        if len(mcap_values) != len(mcap_labels) or len(parquet_values) != len(parquet_labels):
            raise ValueError(f"向量标签重复: {mcap_labels} / {parquet_labels}")
        if len(mcap_labels) != len(mcap_item["vector"]) or len(parquet_labels) != len(parquet_item["vector"]):
            raise ValueError("向量标签与数值长度不一致")

        missing_in_parquet = [label for label in mcap_labels if label not in parquet_values]
        extra_in_parquet = [label for label in parquet_labels if label not in mcap_values]
        if missing_in_parquet or extra_in_parquet:
            raise ValueError(
                f"向量标签不一致: MCAP={mcap_labels}, Parquet={parquet_labels}, "
                f"Parquet缺少={missing_in_parquet}, Parquet多出={extra_in_parquet}"
            )

        dimensions = []
        for label in mcap_labels:
            mcap_value = mcap_values[label]
            parquet_value = parquet_values[label]
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

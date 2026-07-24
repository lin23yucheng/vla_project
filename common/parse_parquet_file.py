#!/usr/bin/env python3
"""解析 parquet 文件并输出基础信息。"""

from __future__ import annotations

import argparse
import io
import json
import re
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from common.parquet_source import (
    find_nearest_parquet_rows,
    normalize_parquet_sources,
    open_parquet_file,
    parquet_source_name,
)


def _find_latest_parquet(parquet_dir: Path) -> Path:
    parquet_files = sorted(
        [p for p in parquet_dir.glob("*.parquet") if p.is_file()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not parquet_files:
        raise FileNotFoundError(f"目录中未找到 parquet 文件: {parquet_dir}")
    return parquet_files[0]


def _preview_rows(parquet_file: pq.ParquetFile, rows: int) -> list[dict[str, Any]]:
    table = parquet_file.read().slice(0, max(rows, 0))
    if table.num_rows == 0:
        return []

    columns = table.column_names
    pydict = table.to_pydict()
    preview: list[dict[str, Any]] = []
    for idx in range(table.num_rows):
        row = {col: pydict[col][idx] for col in columns}
        preview.append(row)
    return preview


def parse_parquet_file(parquet_path: str | Path, preview_rows: int = 5) -> dict[str, Any]:
    """解析 parquet，返回结构化结果，供外部调用。"""
    parquet_file_path = Path(parquet_path)
    if not parquet_file_path.exists():
        raise FileNotFoundError(f"parquet 文件不存在: {parquet_file_path}")

    parquet_file = pq.ParquetFile(parquet_file_path)
    schema = parquet_file.schema_arrow

    result = {
        "file": str(parquet_file_path),
        "num_rows": parquet_file.metadata.num_rows,
        "num_columns": len(schema.names),
        "num_row_groups": parquet_file.metadata.num_row_groups,
        "columns": [
            {
                "name": field.name,
                "type": str(field.type),
                "nullable": field.nullable,
            }
            for field in schema
        ],
        "preview_rows": _preview_rows(parquet_file, preview_rows),
    }
    return result


ANNOTATION_COLUMNS = (
    "original_timestamp_ns",
    "episode_index",
    "frame_index",
    "index",
    "episode_id",
    "task",
    "subtask",
    "subtask_id",
    "subtask_progress",
    "annotation.level_1_id",
    "annotation.level_1_name",
    "annotation.leaf_id",
    "annotation.leaf_name",
    "annotation.path",
    "annotation.hierarchy_json",
    "done",
)
PLAYBACK_DURATION_TOLERANCE_SECONDS = 0.04


def _annotation_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _annotation_int(value: Any) -> int | None:
    try:
        return int(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _annotation_float(value: Any) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _format_duration_seconds(duration_ns: int | None) -> str:
    """将纳秒时长格式化为不丢失精度的秒数字符串。"""
    if duration_ns is None:
        return "无效"
    seconds, nanoseconds = divmod(duration_ns, 1_000_000_000)
    if not nanoseconds:
        return str(seconds)
    return f"{seconds}.{nanoseconds:09d}".rstrip("0")


def _hierarchy_value(item: dict[str, Any], snake_name: str, camel_name: str) -> Any:
    value = item.get(snake_name)
    return item.get(camel_name) if value in (None, "") else value


def _extract_parquet_annotation_summary_single(parquet_source: Any) -> dict[str, Any]:
    """独立校验单个 episode parquet，保留其局部行号语义。"""
    source_name = parquet_source_name(parquet_source)
    with open_parquet_file(parquet_source) as parquet_file:
        available_columns = set(parquet_file.schema_arrow.names)
        missing_columns = [name for name in ANNOTATION_COLUMNS if name not in available_columns]
        read_columns = [name for name in ANNOTATION_COLUMNS if name in available_columns]
        if "timestamp" in available_columns:
            read_columns.append("timestamp")
        rows = parquet_file.read(columns=read_columns, use_threads=False).to_pylist()

    validation_errors: list[str] = []
    validation_error_count = 0

    def add_error(message: str) -> None:
        nonlocal validation_error_count
        validation_error_count += 1
        if len(validation_errors) < 100:
            validation_errors.append(message)

    for column in missing_columns:
        add_error(f"缺少标注校验必需列: {column}")

    annotations: dict[tuple[Any, ...], dict[str, Any]] = {}
    definitions_by_id: dict[tuple[Any, ...], set[tuple[Any, ...]]] = defaultdict(set)
    previous_timestamp_ns: int | None = None
    previous_progress_by_subtask: dict[str, float] = {}
    episodes: dict[str, dict[str, list[int]]] = defaultdict(lambda: {"rows": [], "done_rows": []})

    for row_index, row in enumerate(rows):
        timestamp_ns = _annotation_int(row.get("original_timestamp_ns"))
        playback_timestamp = _annotation_float(row.get("timestamp"))
        if timestamp_ns is None:
            add_error(f"第{row_index}行 original_timestamp_ns 不是有效整数")
        elif previous_timestamp_ns is not None and timestamp_ns <= previous_timestamp_ns:
            add_error(
                f"第{row_index}行 original_timestamp_ns 未严格递增: "
                f"{timestamp_ns} <= {previous_timestamp_ns}"
            )
        if timestamp_ns is not None:
            previous_timestamp_ns = timestamp_ns

        frame_index = _annotation_int(row.get("frame_index"))
        index_value = _annotation_int(row.get("index"))
        if frame_index != row_index:
            add_error(f"第{row_index}行 frame_index={frame_index!r}，预期为 {row_index}")
        if index_value != row_index:
            add_error(f"第{row_index}行 index={index_value!r}，预期为 {row_index}")

        episode_key = _annotation_text(row.get("episode_id")) or _annotation_text(row.get("episode_index"))
        if not episode_key:
            add_error(f"第{row_index}行 episode_id 和 episode_index 均为空")
            episode_key = "<missing>"
        episodes[episode_key]["rows"].append(row_index)
        done_value = row.get("done")
        if done_value is True:
            episodes[episode_key]["done_rows"].append(row_index)
        elif done_value is not False:
            add_error(f"第{row_index}行 done={done_value!r}，预期为布尔值")

        subtask_id = _annotation_text(row.get("subtask_id"))
        progress = row.get("subtask_progress")
        if progress is None:
            add_error(f"第{row_index}行 subtask_progress 为空")
        else:
            try:
                progress_value = float(progress)
            except (TypeError, ValueError):
                add_error(f"第{row_index}行 subtask_progress={progress!r} 不是有效数字")
            else:
                if not 0.0 <= progress_value <= 1.0:
                    add_error(f"第{row_index}行 subtask_progress={progress_value} 超出 [0, 1]")
                previous_progress = previous_progress_by_subtask.get(subtask_id)
                if previous_progress is not None and progress_value < previous_progress:
                    add_error(
                        f"第{row_index}行 subtask_progress 在 subtask_id={subtask_id!r} 内倒退: "
                        f"{progress_value} < {previous_progress}"
                    )
                previous_progress_by_subtask[subtask_id] = progress_value

        raw_hierarchy = row.get("annotation.hierarchy_json")
        try:
            hierarchy = json.loads(raw_hierarchy) if isinstance(raw_hierarchy, str) else raw_hierarchy
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            add_error(f"第{row_index}行 annotation.hierarchy_json 解析失败: {exc}")
            continue
        if not isinstance(hierarchy, list) or not hierarchy:
            add_error(f"第{row_index}行 annotation.hierarchy_json 必须是非空数组")
            continue

        normalized_hierarchy: list[dict[str, Any]] = []
        for hierarchy_index, item in enumerate(hierarchy):
            if not isinstance(item, dict):
                add_error(f"第{row_index}行层级索引{hierarchy_index}不是对象: {item!r}")
                continue
            level = _annotation_int(item.get("level"))
            layer_id = _annotation_text(_hierarchy_value(item, "layer_id", "layerId"))
            segment_id = _annotation_text(item.get("id") or item.get("segmentId"))
            name = _annotation_text(item.get("name") or item.get("description") or item.get("prompt"))
            start_ns = _annotation_int(_hierarchy_value(item, "start_timestamp_ns", "startTimeNs"))
            end_ns = _annotation_int(_hierarchy_value(item, "end_timestamp_ns", "endTimeNs"))
            normalized = {
                "level": level,
                "layer_id": layer_id,
                "id": segment_id,
                "name": name,
                "start_timestamp_ns": start_ns,
                "end_timestamp_ns": end_ns,
            }
            normalized_hierarchy.append(normalized)

            location = f"第{row_index}行/{layer_id or f'层级{level}'}"
            if level is None or level < 1:
                add_error(f"{location} level 不是有效正整数: {item.get('level')!r}")
            elif layer_id != f"l{level}":
                add_error(f"{location} layer_id 与 level 不一致，预期 l{level}")
            if not segment_id:
                add_error(f"{location} id 为空")
            if not name:
                add_error(f"{location} name 为空")
            if start_ns is None or end_ns is None:
                add_error(f"{location} 开始或结束时间不是有效纳秒整数")
            elif start_ns >= end_ns:
                add_error(f"{location} 时间区间无效: {start_ns} >= {end_ns}")
            elif timestamp_ns is not None and not start_ns <= timestamp_ns <= end_ns:
                add_error(
                    f"{location} 当前帧时间 {timestamp_ns} 不在标注区间 "
                    f"[{start_ns}, {end_ns}]"
                )

            key = (level, layer_id, segment_id, name, start_ns, end_ns)
            summary = annotations.setdefault(
                key,
                {
                    **normalized,
                    "row_count": 0,
                    "first_row_index": row_index,
                    "last_row_index": row_index,
                    "first_timestamp_ns": timestamp_ns,
                    "last_timestamp_ns": timestamp_ns,
                    "first_playback_timestamp": playback_timestamp,
                    "last_playback_timestamp": playback_timestamp,
                },
            )
            summary["row_count"] += 1
            summary["last_row_index"] = row_index
            summary["last_timestamp_ns"] = timestamp_ns
            if playback_timestamp is not None:
                first_playback = summary["first_playback_timestamp"]
                last_playback = summary["last_playback_timestamp"]
                summary["first_playback_timestamp"] = (
                    playback_timestamp
                    if first_playback is None
                    else min(first_playback, playback_timestamp)
                )
                summary["last_playback_timestamp"] = (
                    playback_timestamp
                    if last_playback is None
                    else max(last_playback, playback_timestamp)
                )
            definitions_by_id[(level, layer_id, segment_id)].add((name, start_ns, end_ns))

        if not normalized_hierarchy:
            add_error(f"第{row_index}行 annotation.hierarchy_json 不包含有效层级对象")
            continue

        levels = [item["level"] for item in normalized_hierarchy]
        # L3 标注不要求完全落在某个 L2 标注区间内，因此 hierarchy 可以省略
        # 当前帧未激活的中间层；只要求已出现的层级按业务顺序递增且不重复。
        if any(
            current is None or following is None or current >= following
            for current, following in zip(levels, levels[1:])
        ):
            add_error(f"第{row_index}行标注层级顺序错误或重复: {levels}")

        first = normalized_hierarchy[0]
        leaf = normalized_hierarchy[-1]
        flattened_checks = (
            ("task", first["name"]),
            ("annotation.level_1_id", first["id"]),
            ("annotation.level_1_name", first["name"]),
            ("subtask", leaf["name"]),
            ("subtask_id", leaf["id"]),
            ("annotation.leaf_id", leaf["id"]),
            ("annotation.leaf_name", leaf["name"]),
        )
        for column, expected_value in flattened_checks:
            if column in available_columns and _annotation_text(row.get(column)) != expected_value:
                add_error(
                    f"第{row_index}行 {column}={row.get(column)!r}，"
                    f"与 hierarchy_json 中的 {expected_value!r} 不一致"
                )

        path_value = _annotation_text(row.get("annotation.path"))
        path_position = 0
        for item in normalized_hierarchy:
            name = item["name"]
            next_position = path_value.find(name, path_position) if name else -1
            if next_position < 0:
                add_error(
                    f"第{row_index}行 annotation.path={path_value!r} "
                    f"未按层级包含名称 {name!r}"
                )
                break
            path_position = next_position + len(name)

    for episode_key, episode in episodes.items():
        expected_done_rows = [episode["rows"][-1]] if episode["rows"] else []
        if episode["done_rows"] != expected_done_rows:
            add_error(
                f"episode={episode_key!r} 的 done=True 行为 {episode['done_rows']}，"
                f"预期仅最后一行 {expected_done_rows}"
            )

    for identity, definitions in definitions_by_id.items():
        if identity[-1] and len(definitions) > 1:
            add_error(f"同一标注 ID 存在多个名称或时间定义: {identity} -> {list(definitions)}")

    annotation_list = sorted(
        annotations.values(),
        key=lambda item: (
            item["level"] if item["level"] is not None else 999,
            item["start_timestamp_ns"] if item["start_timestamp_ns"] is not None else -1,
            item["id"],
        ),
    )
    level_counts = {
        f"L{level}": sum(1 for item in annotation_list if item["level"] == level)
        for level in (1, 2, 3)
    }
    for level in sorted({item["level"] for item in annotation_list if item["level"] not in (None, 1, 2, 3)}):
        level_counts[f"L{level}"] = sum(1 for item in annotation_list if item["level"] == level)

    if validation_error_count > len(validation_errors):
        validation_errors.append(
            f"另有 {validation_error_count - len(validation_errors)} 条校验错误未展示"
        )
    return {
        "file": source_name,
        "num_rows": len(rows),
        "annotation_columns": read_columns,
        "missing_columns": missing_columns,
        "level_counts": level_counts,
        "annotations": annotation_list,
        "episode_count": len(episodes),
        "validation_error_count": validation_error_count,
        "validation_errors": validation_errors,
    }


def extract_parquet_annotation_summary(parquet_path: Any) -> dict[str, Any]:
    """逐个校验所有 episode parquet，并汇总唯一的 L1/L2/L3 分段。"""
    file_summaries = [
        _extract_parquet_annotation_summary_single(source)
        for source in normalize_parquet_sources(parquet_path)
    ]

    merged_annotations: dict[tuple[Any, ...], dict[str, Any]] = {}
    validation_errors: list[str] = []
    annotation_columns: set[str] = set()
    missing_columns: set[str] = set()
    for summary in file_summaries:
        source_name = summary["file"]
        annotation_columns.update(summary["annotation_columns"])
        missing_columns.update(summary["missing_columns"])
        validation_errors.extend(
            f"[{source_name}] {message}"
            for message in summary["validation_errors"]
        )
        for item in summary["annotations"]:
            key = (
                item["level"],
                item["layer_id"],
                item["id"],
                item["name"],
                item["start_timestamp_ns"],
                item["end_timestamp_ns"],
            )
            existing = merged_annotations.get(key)
            if existing is None:
                existing = {**item, "source_files": [source_name]}
                merged_annotations[key] = existing
                continue
            existing["row_count"] += item["row_count"]
            existing["source_files"].append(source_name)
            first_timestamps = [
                value
                for value in (
                    existing.get("first_timestamp_ns"),
                    item.get("first_timestamp_ns"),
                )
                if value is not None
            ]
            last_timestamps = [
                value
                for value in (
                    existing.get("last_timestamp_ns"),
                    item.get("last_timestamp_ns"),
                )
                if value is not None
            ]
            existing["first_timestamp_ns"] = min(first_timestamps) if first_timestamps else None
            existing["last_timestamp_ns"] = max(last_timestamps) if last_timestamps else None
            first_playback_timestamps = [
                value
                for value in (
                    existing.get("first_playback_timestamp"),
                    item.get("first_playback_timestamp"),
                )
                if value is not None
            ]
            last_playback_timestamps = [
                value
                for value in (
                    existing.get("last_playback_timestamp"),
                    item.get("last_playback_timestamp"),
                )
                if value is not None
            ]
            existing["first_playback_timestamp"] = (
                min(first_playback_timestamps) if first_playback_timestamps else None
            )
            existing["last_playback_timestamp"] = (
                max(last_playback_timestamps) if last_playback_timestamps else None
            )

    annotation_list = sorted(
        merged_annotations.values(),
        key=lambda item: (
            item["level"] if item["level"] is not None else 999,
            item["start_timestamp_ns"] if item["start_timestamp_ns"] is not None else -1,
            item["id"],
        ),
    )
    levels = {item["level"] for item in annotation_list if item["level"] is not None}
    level_counts = {
        f"L{level}": sum(1 for item in annotation_list if item["level"] == level)
        for level in sorted({1, 2, 3, *levels})
    }
    return {
        "file": [summary["file"] for summary in file_summaries],
        "file_count": len(file_summaries),
        "file_summaries": file_summaries,
        "num_rows": sum(summary["num_rows"] for summary in file_summaries),
        "annotation_columns": sorted(annotation_columns),
        "missing_columns": sorted(missing_columns),
        "level_counts": level_counts,
        "annotations": annotation_list,
        "episode_count": sum(summary["episode_count"] for summary in file_summaries),
        "validation_error_count": sum(
            summary["validation_error_count"] for summary in file_summaries
        ),
        "validation_errors": validation_errors,
    }


def _normalize_expected_annotations(expected_layers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    expected_annotations: list[dict[str, Any]] = []
    for layer_position, layer in enumerate(expected_layers, start=1):
        layer_id = _annotation_text(layer.get("layerId") or layer.get("layer_id"))
        match = re.fullmatch(r"l(\d+)", layer_id.lower())
        level = int(match.group(1)) if match else layer_position
        for segment in layer.get("segments") or []:
            expected_annotations.append(
                {
                    "level": level,
                    "layer_id": layer_id or f"l{level}",
                    "id": _annotation_text(segment.get("segmentId") or segment.get("id")),
                    "name": _annotation_text(
                        segment.get("description") or segment.get("prompt") or segment.get("name")
                    ),
                    "start_timestamp_ns": _annotation_int(
                        segment.get("startTimeNs") or segment.get("startTimestampNs")
                    ),
                    "end_timestamp_ns": _annotation_int(
                        segment.get("endTimeNs") or segment.get("endTimestampNs")
                    ),
                }
            )
    return sorted(
        expected_annotations,
        key=lambda item: (
            item["level"],
            item["start_timestamp_ns"] if item["start_timestamp_ns"] is not None else -1,
            item["id"],
        ),
    )


def compare_parquet_annotations(
    parquet_path: Any,
    expected_layers: list[dict[str, Any]],
    validate_l1_playback_duration: bool = False,
) -> dict[str, Any]:
    """将代码中提交的 layers/segments 与 parquet 标注元数据逐层比较。"""
    summary = extract_parquet_annotation_summary(parquet_path)
    expected = _normalize_expected_annotations(expected_layers)
    actual = summary["annotations"]
    failures = list(summary["validation_errors"])
    segment_comparisons: list[dict[str, Any]] = []
    l1_playback_duration_comparisons: list[dict[str, Any]] = []

    levels = sorted({1, 2, 3, *(item["level"] for item in expected), *(item["level"] for item in actual)})
    level_comparisons = []
    for level in levels:
        expected_count = sum(1 for item in expected if item["level"] == level)
        actual_count = sum(1 for item in actual if item["level"] == level)
        is_consistent = expected_count == actual_count
        level_comparisons.append(
            {
                "level": f"L{level}",
                "expected_count": expected_count,
                "actual_count": actual_count,
                "is_consistent": is_consistent,
            }
        )
        if not is_consistent:
            failures.append(f"L{level} 标注数量不一致: 期望 {expected_count}，实际 {actual_count}")

    unused_actual = set(range(len(actual)))
    fields = ("level", "layer_id", "id", "name", "start_timestamp_ns", "end_timestamp_ns")
    field_labels = {
        "level": "层级",
        "layer_id": "层标签",
        "id": "标注ID",
        "name": "描述",
        "start_timestamp_ns": "开始时间",
        "end_timestamp_ns": "结束时间",
    }
    for expected_index, expected_item in enumerate(expected):
        candidates = [index for index in unused_actual if actual[index]["level"] == expected_item["level"]]
        if expected_item["id"]:
            id_matches = [index for index in candidates if actual[index]["id"] == expected_item["id"]]
            if id_matches:
                candidates = id_matches
        if candidates:
            exact_shape_matches = [
                index
                for index in candidates
                if actual[index]["name"] == expected_item["name"]
                and actual[index]["start_timestamp_ns"] == expected_item["start_timestamp_ns"]
                and actual[index]["end_timestamp_ns"] == expected_item["end_timestamp_ns"]
            ]
            if exact_shape_matches:
                candidates = exact_shape_matches

        actual_index = candidates[0] if candidates else None
        actual_item = actual[actual_index] if actual_index is not None else None
        differences = []
        if actual_item is None:
            failures.append(f"未找到期望标注: {expected_item}")
        else:
            unused_actual.remove(actual_index)
            for field in fields:
                if field == "id" and not expected_item[field]:
                    continue
                if expected_item[field] != actual_item[field]:
                    difference = {
                        "field": field,
                        "expected": expected_item[field],
                        "actual": actual_item[field],
                    }
                    differences.append(difference)
                    failures.append(
                        f"L{expected_item['level']}第{expected_index + 1}条{field_labels[field]}不一致: "
                        f"期望 {expected_item[field]!r}，实际 {actual_item[field]!r}"
                    )

            if validate_l1_playback_duration and expected_item["level"] == 1:
                expected_duration_ns = (
                    expected_item["end_timestamp_ns"] - expected_item["start_timestamp_ns"]
                    if expected_item["start_timestamp_ns"] is not None
                    and expected_item["end_timestamp_ns"] is not None
                    else None
                )
                playback_start_seconds = actual_item.get("first_playback_timestamp")
                playback_end_seconds = actual_item.get("last_playback_timestamp")
                actual_duration_seconds = (
                    playback_end_seconds - playback_start_seconds
                    if playback_start_seconds is not None and playback_end_seconds is not None
                    else None
                )
                expected_duration_seconds = (
                    expected_duration_ns / 1_000_000_000
                    if expected_duration_ns is not None
                    else None
                )
                duration_error_seconds = (
                    abs(expected_duration_seconds - actual_duration_seconds)
                    if expected_duration_seconds is not None
                    and actual_duration_seconds is not None
                    else None
                )
                duration_is_consistent = (
                    duration_error_seconds is not None
                    and duration_error_seconds
                    <= PLAYBACK_DURATION_TOLERANCE_SECONDS
                )
                l1_playback_duration_comparisons.append(
                    {
                        "expected_duration_ns": expected_duration_ns,
                        "expected_duration_seconds": expected_duration_seconds,
                        "parquet_playback_start_seconds": playback_start_seconds,
                        "parquet_playback_end_seconds": playback_end_seconds,
                        "parquet_playback_duration_seconds": actual_duration_seconds,
                        "duration_error_seconds": duration_error_seconds,
                        "is_consistent": duration_is_consistent,
                    }
                )
                if not duration_is_consistent:
                    failures.append(
                        f"L1第{expected_index + 1}条播放时长不一致: "
                        f"parquet timestamp {playback_start_seconds!r} -> {playback_end_seconds!r}，"
                        f"时长 {actual_duration_seconds!r} 秒；"
                        f"代码标注时长 {_format_duration_seconds(expected_duration_ns)} 秒"
                        f"（{expected_duration_ns!r} ns），"
                        f"实际误差 {duration_error_seconds!r} 秒，"
                        f"允许误差 {PLAYBACK_DURATION_TOLERANCE_SECONDS} 秒"
                    )
        segment_comparisons.append(
            {
                "expected": expected_item,
                "actual": actual_item,
                "differences": differences,
                "is_consistent": actual_item is not None and not differences,
            }
        )

    unexpected_annotations = [actual[index] for index in sorted(unused_actual)]
    for item in unexpected_annotations:
        failures.append(f"parquet 中存在代码未提交的额外标注: {item}")

    return {
        "is_consistent": not failures,
        "expected_annotations": expected,
        "actual_summary": summary,
        "level_comparisons": level_comparisons,
        "segment_comparisons": segment_comparisons,
        "l1_playback_duration_comparisons": l1_playback_duration_comparisons,
        "unexpected_annotations": unexpected_annotations,
        "failures": failures,
    }


def _discover_image_columns(schema: pa.Schema) -> list[str]:
    """查找形如 struct<bytes: binary, path: string> 的图片列。"""
    image_columns: list[str] = []
    for field in schema:
        if not pa.types.is_struct(field.type):
            continue
        child_names = {child.name for child in field.type}
        if {"bytes", "path"}.issubset(child_names):
            image_columns.append(field.name)
    return image_columns


def _sanitize_filename_component(value: Any) -> str:
    text = str(value).strip()
    sanitized = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff_-]+", "_", text)
    return sanitized.strip("_") or "unknown"


def _infer_side_label(name: str) -> str:
    lower_name = name.lower()
    if "left" in lower_name or "_l" in lower_name:
        return "left"
    if "right" in lower_name or "_r" in lower_name:
        return "right"
    return _sanitize_filename_component(lower_name)


def _to_int_ns(raw_value: Any) -> int | None:
    if raw_value is None:
        return None
    try:
        return int(raw_value)
    except (TypeError, ValueError):
        return None


def _save_preview_image(image_bytes: bytes, output_png_path: Path) -> tuple[Path, str]:
    """优先将 bytes 解析为可展示 PNG；失败则退化为 bin。"""
    try:
        from PIL import Image

        with Image.open(io.BytesIO(image_bytes)) as img:
            output_png_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(output_png_path, format="PNG")
        return output_png_path, "direct_image_bytes"
    except Exception:
        pass

    try:
        from common.extract_mcap_image import decode_image_for_preview, parse_ros2_image

        image_info = parse_ros2_image(image_bytes)
        if image_info:
            output_png_path.parent.mkdir(parents=True, exist_ok=True)
            decode_image_for_preview(image_info).save(output_png_path, format="PNG")
            return output_png_path, "ros2_image_message"
    except Exception:
        pass

    bin_path = output_png_path.with_suffix(".bin")
    bin_path.parent.mkdir(parents=True, exist_ok=True)
    bin_path.write_bytes(image_bytes)
    return bin_path, "raw_bin_fallback"


def _build_parquet_image_output_path(
    output_root: Path,
    column_name: str,
    target_ns: int,
    matched_timestamp_ns: int,
    name_prefix: str | None,
    extra_name_parts: list[str] | None,
) -> Path:
    name_parts = []
    if name_prefix:
        name_parts.append(_sanitize_filename_component(name_prefix))
    name_parts.extend(
        [
            "parquet",
            _infer_side_label(column_name),
            _sanitize_filename_component(column_name),
            f"target_{target_ns}",
            f"matched_{matched_timestamp_ns}",
        ]
    )
    for extra in extra_name_parts or []:
        if extra:
            name_parts.append(_sanitize_filename_component(extra))
    return output_root / f"{'_'.join(name_parts)}.png"


def _s3_video_object_name(source: Any, column_name: str) -> str:
    object_name = str(getattr(source, "object_name", ""))
    match = re.fullmatch(
        r"(.+)/data/(chunk-\d+)/(episode_[^/]+)\.parquet",
        object_name,
    )
    if match is None:
        raise ValueError(f"无法从 parquet 路径推导视频路径: {object_name}")
    dataset_prefix, chunk_name, episode_name = match.groups()
    return f"{dataset_prefix}/videos/{column_name}/{chunk_name}/{episode_name}.mp4"


def _extract_s3_video_frames_for_source(
    source_batch: dict[str, Any],
    selected_image_columns: list[str],
    output_root: Path,
    name_prefix: str | None,
) -> tuple[dict[str, dict[str, Any]], int]:
    """从 LeRobot S3 视频中一次解码多个目标帧，避开巨大的 Parquet bytes 列。"""
    try:
        import av
    except ImportError as exc:
        raise RuntimeError("天机视频抽帧需要安装 requirements.txt 中的 av") from exc

    source = source_batch["source"]
    client = getattr(source, "client", None)
    bucket = str(getattr(source, "bucket", ""))
    if client is None or not bucket:
        raise TypeError("S3 视频抽帧要求 parquet source 提供 client 和 bucket")

    grouped_requests = [
        pair
        for row_group_requests in source_batch["row_groups"].values()
        for pair in row_group_requests
    ]
    results: dict[str, dict[str, Any]] = {}
    requests_by_frame: dict[int, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for request, match in grouped_requests:
        frame_index = int(match["row_index"])
        requests_by_frame[frame_index].append((request, match))
        results[request["key"]] = {
            "parquet_file": match["source_name"],
            "matched_row_group_index": match["row_group_index"],
            "target_ns": request["target_ns"],
            "matched_row_index": frame_index,
            "matched_timestamp_ns": int(match["timestamp_ns"]),
            "diff_ns": int(match["diff_ns"]),
            "used_nearest": int(match["diff_ns"]) != 0,
            "image_columns": selected_image_columns,
            "output_dir": str(output_root),
            "batch_row_group_target_count": len(
                source_batch["row_groups"][int(match["row_group_index"])]
            ),
            "image_source_mode": "s3_video",
            "saved_files": [],
        }

    target_frames = set(requests_by_frame)
    video_files_read = 0
    with tempfile.TemporaryDirectory(prefix="tianji_video_") as temporary_dir:
        temporary_root = Path(temporary_dir)
        for column_name in selected_image_columns:
            video_object_name = _s3_video_object_name(source, column_name)
            local_video = temporary_root / f"{_sanitize_filename_component(column_name)}.mp4"
            client.fget_object(bucket, video_object_name, str(local_video))
            video_files_read += 1

            found_frames: set[int] = set()
            with av.open(str(local_video)) as container:
                stream = container.streams.video[0]
                for frame_index, frame in enumerate(container.decode(stream)):
                    if frame_index not in target_frames:
                        continue
                    for request, match in requests_by_frame[frame_index]:
                        output_png = _build_parquet_image_output_path(
                            output_root=output_root,
                            column_name=column_name,
                            target_ns=request["target_ns"],
                            matched_timestamp_ns=int(match["timestamp_ns"]),
                            name_prefix=name_prefix,
                            extra_name_parts=request["extra_name_parts"],
                        )
                        output_png.parent.mkdir(parents=True, exist_ok=True)
                        frame.to_image().save(output_png, format="PNG")
                        results[request["key"]]["saved_files"].append(
                            {
                                "column": column_name,
                                "side": _infer_side_label(column_name),
                                "row_path": f"frame_{frame_index:06d}.png",
                                "saved_path": str(output_png),
                                "decode_mode": "s3_video_frame",
                                "bytes_length": output_png.stat().st_size,
                                "video_object": f"s3://{bucket}/{video_object_name}",
                                "video_frame_index": frame_index,
                            }
                        )
                    found_frames.add(frame_index)
                    if found_frames == target_frames:
                        break

            missing_frames = sorted(target_frames - found_frames)
            if missing_frames:
                raise ValueError(
                    f"视频 {video_object_name} 缺少目标帧: {missing_frames}"
                )

    expected_count = len(selected_image_columns)
    for request_key, result in results.items():
        if len(result["saved_files"]) != expected_count:
            raise ValueError(
                f"{request_key} 视频抽帧数量不完整: "
                f"{len(result['saved_files'])} != {expected_count}"
            )
    return results, video_files_read


def _build_parquet_image_result(
    match: dict[str, Any],
    nearest_row: dict[str, Any],
    target_ns: int,
    selected_image_columns: list[str],
    output_root: Path,
    name_prefix: str | None,
    extra_name_parts: list[str] | None,
    batch_row_group_target_count: int,
) -> dict[str, Any]:
    nearest_index = int(match["row_index"])
    nearest_timestamp_ns = int(match["timestamp_ns"])
    nearest_diff_ns = int(match["diff_ns"])
    saved_files: list[dict[str, Any]] = []
    for column_name in selected_image_columns:
        image_struct = nearest_row.get(column_name)
        if not isinstance(image_struct, dict):
            continue

        image_bytes = image_struct.get("bytes")
        image_path_in_row = image_struct.get("path")
        if isinstance(image_bytes, memoryview):
            image_bytes = image_bytes.tobytes()
        elif isinstance(image_bytes, bytearray):
            image_bytes = bytes(image_bytes)

        if not isinstance(image_bytes, (bytes, bytearray)):
            continue

        side_label = _infer_side_label(column_name)
        output_png = _build_parquet_image_output_path(
            output_root=output_root,
            column_name=column_name,
            target_ns=target_ns,
            matched_timestamp_ns=nearest_timestamp_ns,
            name_prefix=name_prefix,
            extra_name_parts=extra_name_parts,
        )
        saved_path, decode_mode = _save_preview_image(bytes(image_bytes), output_png)
        saved_files.append(
            {
                "column": column_name,
                "side": side_label,
                "row_path": image_path_in_row,
                "saved_path": str(saved_path),
                "decode_mode": decode_mode,
                "bytes_length": len(image_bytes),
            }
        )

    if not saved_files:
        raise ValueError("最近时间行未提取到有效图片 bytes")

    return {
        "parquet_file": match["source_name"],
        "matched_row_group_index": match["row_group_index"],
        "target_ns": target_ns,
        "matched_row_index": nearest_index,
        "matched_timestamp_ns": nearest_timestamp_ns,
        "diff_ns": nearest_diff_ns,
        "used_nearest": nearest_diff_ns != 0,
        "image_columns": selected_image_columns,
        "output_dir": str(output_root),
        "batch_row_group_target_count": batch_row_group_target_count,
        "saved_files": saved_files,
    }


def extract_nearest_parquet_images_batch(
    parquet_path: Any,
    target_requests: Sequence[dict[str, Any]],
    output_dir: str | Path,
    image_columns: list[str] | None = None,
    name_prefix: str | None = None,
    prefer_s3_videos: bool = False,
) -> dict[str, Any]:
    """批量匹配多个时间点，并以小批次流式扫描图片 row group。"""
    requests = []
    request_keys = set()
    for request in target_requests:
        key = request.get("key")
        if not isinstance(key, str) or not key:
            raise ValueError("每个 target_request 必须包含非空字符串 key")
        if key in request_keys:
            raise ValueError(f"target_request key 重复: {key}")
        request_keys.add(key)
        requests.append(
            {
                "key": key,
                "target_ns": int(request["target_ns"]),
                "extra_name_parts": list(request.get("extra_name_parts") or []),
            }
        )
    if not requests:
        raise ValueError("target_requests 不能为空")

    matches = find_nearest_parquet_rows(
        parquet_path,
        [request["target_ns"] for request in requests],
    )
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    source_batches: dict[str, dict[str, Any]] = {}
    for request, match in zip(requests, matches):
        source_name = match["source_name"]
        source_batch = source_batches.setdefault(
            source_name,
            {"source": match["source"], "row_groups": defaultdict(list)},
        )
        source_batch["row_groups"][int(match["row_group_index"])].append((request, match))

    results: dict[str, dict[str, Any]] = {}
    row_groups_read = 0
    video_files_read = 0
    for source_batch in source_batches.values():
        with open_parquet_file(source_batch["source"]) as parquet_file:
            schema = parquet_file.schema_arrow
            discovered_image_columns = _discover_image_columns(schema)
            if image_columns is None:
                selected_image_columns = discovered_image_columns
            else:
                selected_image_columns = [
                    name for name in image_columns if name in discovered_image_columns
                ]
            if not selected_image_columns:
                raise ValueError("parquet 中未找到可用图片列（期望 struct<bytes, path>）")

            if prefer_s3_videos:
                source_results, source_video_files_read = _extract_s3_video_frames_for_source(
                    source_batch=source_batch,
                    selected_image_columns=selected_image_columns,
                    output_root=output_root,
                    name_prefix=name_prefix,
                )
                results.update(source_results)
                video_files_read += source_video_files_read
                continue

            for row_group_index, grouped_requests in source_batch["row_groups"].items():
                requests_by_row: dict[int, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
                for request, match in grouped_requests:
                    requests_by_row[int(match["row_in_group"])].append((request, match))

                target_rows = set(requests_by_row)
                matched_rows: dict[int, dict[str, Any]] = {}
                row_offset = 0
                for batch in parquet_file.iter_batches(
                    batch_size=8,
                    row_groups=[row_group_index],
                    columns=selected_image_columns,
                    use_threads=False,
                ):
                    batch_end = row_offset + batch.num_rows
                    rows_in_batch = sorted(
                        row_index
                        for row_index in target_rows
                        if row_offset <= row_index < batch_end
                    )
                    for row_index in rows_in_batch:
                        matched_rows[row_index] = batch.slice(row_index - row_offset, 1).to_pylist()[0]
                    row_offset = batch_end
                    if len(matched_rows) == len(target_rows):
                        break
                row_groups_read += 1

                for request, match in grouped_requests:
                    row_in_group = int(match["row_in_group"])
                    nearest_row = matched_rows.get(row_in_group)
                    if nearest_row is None:
                        raise IndexError(
                            f"parquet row group 未读取到目标行: row_group={row_group_index}, "
                            f"row={row_in_group}, scanned_rows={row_offset}"
                        )
                    results[request["key"]] = _build_parquet_image_result(
                        match=match,
                        nearest_row=nearest_row,
                        target_ns=request["target_ns"],
                        selected_image_columns=selected_image_columns,
                        output_root=output_root,
                        name_prefix=name_prefix,
                        extra_name_parts=request["extra_name_parts"],
                        batch_row_group_target_count=len(grouped_requests),
                    )

    return {
        "target_count": len(requests),
        "source_count": len(source_batches),
        "row_groups_read": row_groups_read,
        "video_files_read": video_files_read,
        "image_source_mode": "s3_video" if prefer_s3_videos else "parquet_bytes",
        "output_dir": str(output_root),
        "results": {request["key"]: results[request["key"]] for request in requests},
    }


def extract_nearest_parquet_images(
    parquet_path: Any,
    target_ns: int,
    output_dir: str | Path,
    image_columns: list[str] | None = None,
    name_prefix: str | None = None,
    extra_name_parts: list[str] | None = None,
    prefer_s3_videos: bool = False,
) -> dict[str, Any]:
    """跨本地或远端 parquet 源寻找最近帧并导出图片。"""
    batch_result = extract_nearest_parquet_images_batch(
        parquet_path=parquet_path,
        target_requests=[
            {
                "key": "single",
                "target_ns": int(target_ns),
                "extra_name_parts": extra_name_parts or [],
            }
        ],
        output_dir=output_dir,
        image_columns=image_columns,
        name_prefix=name_prefix,
        prefer_s3_videos=prefer_s3_videos,
    )
    return batch_result["results"]["single"]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="解析 parquet 文件")
    parser.add_argument(
        "--file",
        dest="file",
        default=None,
        help="parquet 文件路径；不传则自动取 parquet_file 目录最新文件",
    )
    parser.add_argument(
        "--rows",
        dest="rows",
        type=int,
        default=5,
        help="预览行数，默认5",
    )
    parser.add_argument(
        "--dir",
        dest="parquet_dir",
        default=None,
        help="自动查找 parquet 时使用的目录，默认项目根目录/parquet_file",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    parquet_dir = Path(args.parquet_dir) if args.parquet_dir else (project_root / "parquet_file")

    parquet_path = Path(args.file) if args.file else _find_latest_parquet(parquet_dir)
    print(f"[parquet] 使用文件: {parquet_path}")

    result = parse_parquet_file(parquet_path=parquet_path, preview_rows=args.rows)
    result_text = json.dumps(result, ensure_ascii=False, indent=2, default=str)

    analysis_file = project_root / "parquet_file" / "parquet_analysis.txt"
    analysis_file.parent.mkdir(parents=True, exist_ok=True)
    analysis_file.write_text(result_text, encoding="utf-8")

    # print(result_text)
    print(f"[parquet] 解析结果已写入: {analysis_file}")


if __name__ == "__main__":
    main()


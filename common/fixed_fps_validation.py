"""校验转换输出是否遵循 Episode 主相机首帧起始的固定 30 FPS 时间轴。"""

from typing import Any, Sequence

from common.mcap_source import adaptive_search_windows, read_mcap_window_messages
from common.parquet_source import (
    open_parquet_file,
    parquet_source_name,
    resolve_unique_parquet_episode_source,
)


NANOSECONDS_PER_SECOND = 1_000_000_000
FPS = 30
TIMESTAMP_GRID_TOLERANCE_SECONDS = 2e-5
T0_ALIGNMENT_TOLERANCE_NS = 1_000


def find_episode_first_main_frame_ns(
    mcap_sources: Sequence[Any],
    main_camera_topic: str,
    episode_start_ns: int,
    episode_end_ns: int,
    *,
    require_chunk_indexes: bool,
) -> int:
    """在 Episode 范围内查找主相机的第一条消息，避免读取完整 Episode。"""
    for window_ns in adaptive_search_windows(episode_end_ns - episode_start_ns):
        records, _ = read_mcap_window_messages(
            mcap_sources,
            [main_camera_topic],
            episode_start_ns,
            min(episode_start_ns + window_ns + 1, episode_end_ns + 1),
            require_chunk_indexes=require_chunk_indexes,
        )
        timestamps = [
            int(record["message"].log_time)
            for record in records
            if record["channel"].topic == main_camera_topic
        ]
        if timestamps:
            return min(timestamps)
    raise ValueError(
        f"Episode 范围内未找到主相机帧: topic={main_camera_topic}, "
        f"range=[{episode_start_ns}, {episode_end_ns}]"
    )


def find_episode_last_main_frame_ns(
    mcap_sources: Sequence[Any],
    main_camera_topic: str,
    episode_start_ns: int,
    episode_end_ns: int,
    *,
    require_chunk_indexes: bool,
) -> int:
    """在 Episode 范围内查找主相机最后一帧，避免读取完整 Episode。"""
    for window_ns in adaptive_search_windows(episode_end_ns - episode_start_ns):
        records, _ = read_mcap_window_messages(
            mcap_sources,
            [main_camera_topic],
            max(episode_start_ns, episode_end_ns - window_ns),
            episode_end_ns + 1,
            require_chunk_indexes=require_chunk_indexes,
        )
        timestamps = [
            int(record["message"].log_time)
            for record in records
            if record["channel"].topic == main_camera_topic
        ]
        if timestamps:
            return max(timestamps)
    raise ValueError(
        f"Episode 范围内未找到主相机帧: topic={main_camera_topic}, "
        f"range=[{episode_start_ns}, {episode_end_ns}]"
    )


def _read_episode_parquet_rows(
    parquet_source: Any,
    original_timestamp_column: str,
    timestamp_column: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open_parquet_file(parquet_source) as parquet_file:
        required_columns = [
            original_timestamp_column,
            timestamp_column,
            "frame_index",
        ]
        missing_columns = [
            column for column in required_columns
            if column not in parquet_file.schema_arrow.names
        ]
        if missing_columns:
            raise ValueError(
                f"Parquet 缺少列 {missing_columns}: {parquet_source_name(parquet_source)}"
            )
        for row_group_index in range(parquet_file.metadata.num_row_groups):
            values = parquet_file.read_row_group(
                row_group_index,
                columns=required_columns,
                use_threads=False,
            ).to_pylist()
            for row in values:
                if any(row.get(column) is None for column in required_columns):
                    raise ValueError(
                        f"Parquet 固定 FPS 必需字段存在空值: "
                        f"{parquet_source_name(parquet_source)}"
                    )
                rows.append(
                    {
                        "source": parquet_source_name(parquet_source),
                        "original_timestamp_ns": int(row[original_timestamp_column]),
                        "timestamp": float(row[timestamp_column]),
                        "frame_index": int(row["frame_index"]),
                    }
                )
    return rows


def validate_episode_fixed_fps_timeline(
    *,
    mcap_sources: Sequence[Any],
    parquet_sources: Any,
    main_camera_topic: str,
    episode_start_ns: int,
    episode_end_ns: int,
    require_chunk_indexes: bool = True,
    timestamp_column: str = "timestamp",
    original_timestamp_column: str = "original_timestamp_ns",
) -> dict[str, Any]:
    """独立复算 `T0 + n / 30`，校验 Episode 的 Parquet 输出时间轴。

    `timestamp` 是输出的秒级固定时间轴；`original_timestamp_ns` 是对应帧的采样目标时间。
    主相机消息时间与转换端一致，统一使用 MCAP `log_time`。
    """
    episode_start_ns = int(episode_start_ns)
    episode_end_ns = int(episode_end_ns)
    if episode_start_ns >= episode_end_ns:
        raise ValueError("episode_start_ns 必须早于 episode_end_ns")

    t0_ns = find_episode_first_main_frame_ns(
        mcap_sources,
        main_camera_topic,
        episode_start_ns,
        episode_end_ns,
        require_chunk_indexes=require_chunk_indexes,
    )
    last_main_frame_ns = find_episode_last_main_frame_ns(
        mcap_sources,
        main_camera_topic,
        episode_start_ns,
        episode_end_ns,
        require_chunk_indexes=require_chunk_indexes,
    )
    timeline_end_ns = min(episode_end_ns, last_main_frame_ns)
    episode_record = resolve_unique_parquet_episode_source(
        parquet_sources,
        episode_start_ns,
        episode_end_ns,
    )
    rows = _read_episode_parquet_rows(
        episode_record["source"],
        original_timestamp_column,
        timestamp_column,
    )
    if not rows:
        raise ValueError(
            f"Parquet 在 Episode 输出范围内没有 {original_timestamp_column}: "
            f"[{t0_ns}, {episode_end_ns}]"
        )

    failures: list[str] = []
    t0_alignment_difference_ns = abs(rows[0]["original_timestamp_ns"] - t0_ns)
    if t0_alignment_difference_ns > T0_ALIGNMENT_TOLERANCE_NS:
        failures.append(
            "首帧原始时间未与主相机 T0 对齐: "
            f"parquet={rows[0]['original_timestamp_ns']}, T0={t0_ns}, "
            f"diff_ns={t0_alignment_difference_ns}, "
            f"tolerance_ns={T0_ALIGNMENT_TOLERANCE_NS}"
        )
    if abs(rows[0]["timestamp"]) > 1e-9:
        failures.append(f"固定时间轴首帧不是 0 秒: timestamp={rows[0]['timestamp']}")

    previous_original_timestamp_ns: int | None = None
    for expected_frame_index, row in enumerate(rows):
        timestamp_seconds = row["timestamp"]
        frame_index = row["frame_index"]
        expected_seconds = expected_frame_index / FPS
        if frame_index != expected_frame_index:
            failures.append(
                f"frame_index 不连续: actual={frame_index}, expected={expected_frame_index}"
            )
        if abs(timestamp_seconds - expected_seconds) > TIMESTAMP_GRID_TOLERANCE_SECONDS:
            failures.append(
                f"时间戳不在 30 FPS 网格: timestamp={timestamp_seconds}, "
                f"expected={expected_seconds}, frame_index={expected_frame_index}"
            )
        expected_target_ns = t0_ns + round(expected_seconds * NANOSECONDS_PER_SECOND)
        target_difference_ns = abs(row["original_timestamp_ns"] - expected_target_ns)
        if target_difference_ns > T0_ALIGNMENT_TOLERANCE_NS:
            failures.append(
                "original_timestamp_ns 不是固定时间轴采样目标: "
                f"actual={row['original_timestamp_ns']}, expected={expected_target_ns}, "
                f"diff_ns={target_difference_ns}"
            )
        if (
            previous_original_timestamp_ns is not None
            and row["original_timestamp_ns"] <= previous_original_timestamp_ns
        ):
            failures.append(
                "original_timestamp_ns 未严格递增: "
                f"{row['original_timestamp_ns']} <= {previous_original_timestamp_ns}"
            )
        previous_original_timestamp_ns = row["original_timestamp_ns"]

    expected_frame_count = ((timeline_end_ns - t0_ns) * FPS) // NANOSECONDS_PER_SECOND + 1
    if len(rows) != expected_frame_count:
        failures.append(
            f"输出帧数不符合 Episode 30 FPS 时间轴: "
            f"actual={len(rows)}, expected={expected_frame_count}"
        )

    return {
        "is_consistent": not failures,
        "fps": FPS,
        "timestamp_grid_tolerance_seconds": TIMESTAMP_GRID_TOLERANCE_SECONDS,
        "t0_alignment_tolerance_ns": T0_ALIGNMENT_TOLERANCE_NS,
        "t0_alignment_difference_ns": t0_alignment_difference_ns,
        "main_camera_topic": main_camera_topic,
        "parquet_source": episode_record["source_name"],
        "episode_id": episode_record["episode_id"],
        "episode_index": episode_record["episode_index"],
        "l1_id": episode_record["l1_id"],
        "time_field": "log_time",
        "episode_start_ns": episode_start_ns,
        "episode_end_ns": episode_end_ns,
        "t0_ns": t0_ns,
        "last_main_frame_ns": last_main_frame_ns,
        "timeline_end_ns": timeline_end_ns,
        "timestamp_column": timestamp_column,
        "original_timestamp_column": original_timestamp_column,
        "actual_frame_count": len(rows),
        "expected_frame_count": expected_frame_count,
        "first_parquet_timestamp": rows[0]["timestamp"],
        "last_parquet_timestamp": rows[-1]["timestamp"],
        "first_original_timestamp_ns": rows[0]["original_timestamp_ns"],
        "last_original_timestamp_ns": rows[-1]["original_timestamp_ns"],
        "failures": failures,
    }

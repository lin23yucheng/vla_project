"""校验转换输出是否遵循 Episode 主相机首帧起始的固定 30 FPS 时间轴。"""

from typing import Any, Sequence

from common.mcap_source import adaptive_search_windows, read_mcap_window_messages
from common.parquet_source import normalize_parquet_sources, open_parquet_file, parquet_source_name


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
    parquet_sources: Any,
    start_ns: int,
    end_ns: int,
    original_timestamp_column: str,
    timestamp_column: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in normalize_parquet_sources(parquet_sources):
        with open_parquet_file(source) as parquet_file:
            required_columns = [original_timestamp_column, timestamp_column]
            missing_columns = [
                column for column in required_columns
                if column not in parquet_file.schema_arrow.names
            ]
            if missing_columns:
                raise ValueError(
                    f"Parquet 缺少列 {missing_columns}: {parquet_source_name(source)}"
                )
            for row_group_index in range(parquet_file.metadata.num_row_groups):
                values = parquet_file.read_row_group(
                    row_group_index,
                    columns=required_columns,
                    use_threads=False,
                ).to_pylist()
                rows.extend(
                    {
                        "source": parquet_source_name(source),
                        "original_timestamp_ns": int(row[original_timestamp_column]),
                        "timestamp": float(row[timestamp_column]),
                    }
                    for row in values
                    if row.get(original_timestamp_column) is not None
                    and row.get(timestamp_column) is not None
                    and start_ns <= int(row[original_timestamp_column]) <= end_ns
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
    rows = _read_episode_parquet_rows(
        parquet_sources,
        t0_ns,
        timeline_end_ns,
        original_timestamp_column,
        timestamp_column,
    )
    if not rows:
        raise ValueError(
            f"Parquet 在 Episode 输出范围内没有 {original_timestamp_column}: "
            f"[{t0_ns}, {episode_end_ns}]"
        )

    rows.sort(key=lambda row: row["timestamp"])
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

    frame_indices: list[int] = []
    for row in rows:
        timestamp_seconds = row["timestamp"]
        frame_index = round(timestamp_seconds * FPS)
        expected_seconds = frame_index / FPS
        if abs(timestamp_seconds - expected_seconds) > TIMESTAMP_GRID_TOLERANCE_SECONDS:
            failures.append(
                f"时间戳不在 30 FPS 网格: timestamp={timestamp_seconds}, "
                f"expected={expected_seconds}, frame_index={frame_index}"
            )
        frame_indices.append(frame_index)
        expected_target_ns = t0_ns + round(expected_seconds * NANOSECONDS_PER_SECOND)
        target_difference_ns = abs(row["original_timestamp_ns"] - expected_target_ns)
        if target_difference_ns > T0_ALIGNMENT_TOLERANCE_NS:
            failures.append(
                "original_timestamp_ns 不是固定时间轴采样目标: "
                f"actual={row['original_timestamp_ns']}, expected={expected_target_ns}, "
                f"diff_ns={target_difference_ns}"
            )

    duplicate_indices = [
        frame_indices[index]
        for index in range(1, len(frame_indices))
        if frame_indices[index] == frame_indices[index - 1]
    ]
    if duplicate_indices:
        failures.append(f"30 FPS 时间轴存在重复帧索引: {duplicate_indices[:10]}")
    if any(frame_indices[index] != frame_indices[index - 1] + 1 for index in range(1, len(frame_indices))):
        failures.append("30 FPS 时间轴存在跳帧或非递增帧索引")

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

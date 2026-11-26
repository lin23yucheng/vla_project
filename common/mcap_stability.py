"""Audit raw robot MCAP files for structural and camera-stream issues."""

from __future__ import annotations

import json
import math
from bisect import bisect_left
from collections import Counter
from pathlib import Path
from typing import Any

from mcap.reader import make_reader
import yaml

from common.extract_mcap_image import decode_image_for_preview, parse_ros2_image_message


CAMERA_VIEW_ROLES = {
    "cam_head": "主视角",
    "cam_chest": "全景",
    "cam_left": "左腕",
    "cam_right": "右腕",
}
CAMERA_TOPIC_PRIORITY = (
    "/cam_head/color_compressed",
    "/cam_chest/color_compressed",
    "/cam_left/color_compressed",
    "/cam_right/color_compressed",
)
IMAGE_SCHEMA_SUFFIXES = ("/image", "/compressedimage")
EXPECTED_CAMERA_SCHEMA = "sensor_msgs/msg/CompressedImage"
TREND_WINDOW_NS = 60_000_000_000


def _is_image_schema(schema_name: str | None) -> bool:
    normalized = (schema_name or "").strip().lower()
    return normalized.endswith(IMAGE_SCHEMA_SUFFIXES)


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _timestamp_summary(timestamps: list[int]) -> dict[str, Any]:
    intervals_ns: list[int] = []
    out_of_order_count = 0
    duplicate_count = 0
    longest_gap: dict[str, int] | None = None

    for previous, current in zip(timestamps, timestamps[1:]):
        interval = current - previous
        if interval < 0:
            out_of_order_count += 1
            continue
        if interval == 0:
            duplicate_count += 1
            continue
        intervals_ns.append(interval)
        if longest_gap is None or interval > longest_gap["interval_ns"]:
            longest_gap = {
                "interval_ns": interval,
                "start_ns": previous,
                "end_ns": current,
            }

    return {
        "message_count": len(timestamps),
        "first_ns": timestamps[0] if timestamps else None,
        "last_ns": timestamps[-1] if timestamps else None,
        "minimum_ns": min(timestamps) if timestamps else None,
        "maximum_ns": max(timestamps) if timestamps else None,
        "observed_rate_hz": (
            (len(timestamps) - 1) * 1_000_000_000 / (max(timestamps) - min(timestamps))
            if len(timestamps) > 1 and max(timestamps) > min(timestamps)
            else None
        ),
        "out_of_order_count": out_of_order_count,
        "duplicate_count": duplicate_count,
        "interval_count": len(intervals_ns),
        "interval_mean_ns": sum(intervals_ns) / len(intervals_ns) if intervals_ns else None,
        "interval_p50_ns": _percentile(intervals_ns, 0.50),
        "interval_p95_ns": _percentile(intervals_ns, 0.95),
        "interval_p99_ns": _percentile(intervals_ns, 0.99),
        "interval_max_ns": max(intervals_ns) if intervals_ns else None,
        "longest_gap": longest_gap,
    }


def _window_summaries(timestamps: list[int]) -> list[dict[str, Any]]:
    if not timestamps:
        return []

    origin_ns = timestamps[0]
    windows: dict[int, dict[str, Any]] = {}
    previous_ns = None
    for timestamp_ns in timestamps:
        window_index = max(0, (timestamp_ns - origin_ns) // TREND_WINDOW_NS)
        window = windows.setdefault(
            window_index,
            {"message_count": 0, "intervals_ns": []},
        )
        window["message_count"] += 1
        if previous_ns is not None:
            interval_ns = timestamp_ns - previous_ns
            if interval_ns > 0:
                window["intervals_ns"].append(interval_ns)
        previous_ns = timestamp_ns

    return [
        {
            "start_offset_seconds": window_index * TREND_WINDOW_NS / 1_000_000_000,
            "end_offset_seconds": (window_index + 1) * TREND_WINDOW_NS / 1_000_000_000,
            "message_count": window["message_count"],
            "interval_count": len(window["intervals_ns"]),
            "interval_mean_ns": (
                sum(window["intervals_ns"]) / len(window["intervals_ns"])
                if window["intervals_ns"]
                else None
            ),
            "interval_p95_ns": _percentile(window["intervals_ns"], 0.95),
            "interval_max_ns": max(window["intervals_ns"])
            if window["intervals_ns"]
            else None,
        }
        for window_index, window in sorted(windows.items())
    ]


def _four_camera_header_skew(header_times: dict[str, list[int]]) -> dict[str, Any] | None:
    if len(header_times) != 4 or any(not timestamps for timestamps in header_times.values()):
        return None

    reference_topic = next(
        (topic for topic in CAMERA_TOPIC_PRIORITY if topic in header_times),
        next(iter(header_times)),
    )
    reference_times = header_times[reference_topic]
    other_times = {
        topic: sorted(timestamps)
        for topic, timestamps in header_times.items()
        if topic != reference_topic
    }
    skews_ns = []
    for reference_ns in reference_times:
        matched = [reference_ns]
        for timestamps in other_times.values():
            position = bisect_left(timestamps, reference_ns)
            candidates = timestamps[max(0, position - 1):min(len(timestamps), position + 1)]
            matched.append(min(candidates, key=lambda value: abs(value - reference_ns)))
        skews_ns.append(max(matched) - min(matched))

    return {
        "reference_topic": reference_topic,
        "interpretation": "只有四路 header.stamp 使用同一时钟时，差值才可解释为相机同步差",
        "sample_count": len(skews_ns),
        "p50_ns": _percentile(skews_ns, 0.50),
        "p95_ns": _percentile(skews_ns, 0.95),
        "p99_ns": _percentile(skews_ns, 0.99),
        "maximum_ns": max(skews_ns),
    }


def _load_episode_metadata(mcap_path: Path, root: Path) -> dict[str, Any] | None:
    for parent in (mcap_path.parent, *mcap_path.parents):
        metadata_path = parent / "meta.json"
        if metadata_path.is_file():
            with metadata_path.open("r", encoding="utf-8") as file:
                metadata = json.load(file)
            metadata["_path"] = str(metadata_path)
            return metadata
        if parent == root:
            break
    return None


def _load_rosbag_metadata(mcap_path: Path) -> dict[str, Any] | None:
    metadata_path = mcap_path.parent / "metadata.yaml"
    if not metadata_path.is_file():
        return None
    with metadata_path.open("r", encoding="utf-8") as file:
        document = yaml.safe_load(file)
    information = document.get("rosbag2_bagfile_information") if isinstance(document, dict) else None
    if not isinstance(information, dict):
        raise ValueError(f"metadata.yaml 缺少 rosbag2_bagfile_information: {metadata_path}")

    file_entries = information.get("files", [])
    if not isinstance(file_entries, list):
        file_entries = []
    matching_entry = next(
        (
            entry
            for entry in file_entries
            if isinstance(entry, dict)
            and Path(str(entry.get("path", ""))).name == mcap_path.name
        ),
        None,
    )
    topic_counts = {}
    topic_entries = information.get("topics_with_message_count", [])
    if not isinstance(topic_entries, list):
        topic_entries = []
    for item in topic_entries:
        if not isinstance(item, dict):
            continue
        topic_metadata = item.get("topic_metadata", {})
        topic = topic_metadata.get("name") if isinstance(topic_metadata, dict) else None
        if topic:
            topic_counts[topic] = int(item.get("message_count", 0))

    return {
        "path": str(metadata_path),
        "bag_message_count": information.get("message_count"),
        "file_message_count": matching_entry.get("message_count") if matching_entry else None,
        "file_duration_ns": (
            matching_entry.get("duration", {}).get("nanoseconds")
            if matching_entry
            else None
        ),
        "topic_message_counts": topic_counts,
        "is_single_file_bag": len(file_entries) == 1,
    }


def _new_topic_record() -> dict[str, Any]:
    return {
        "message_count": 0,
        "schemas": set(),
        "log_times": [],
        "header_times": [],
        "image_decode_failures": 0,
        "image_dimensions": Counter(),
        "image_encodings": Counter(),
    }


def load_robot_camera_config(config_path: str | Path) -> tuple[dict[str, str], list[str], list[str]]:
    """Load camera topic labels and configured action/state topics."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"机器人配置文件不存在: {path}")
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"机器人配置必须是 YAML mapping: {path}")

    cameras = config.get("cameras")
    if not isinstance(cameras, dict) or len(cameras) != 4:
        raise ValueError("robot_config.yaml 的 cameras 必须定义四路相机")

    camera_labels = {}
    for camera_name, camera in cameras.items():
        if not isinstance(camera, dict) or not camera.get("topic") or not camera.get("label"):
            raise ValueError(f"机器人相机配置不完整: {camera_name}")
        view_role = CAMERA_VIEW_ROLES.get(camera_name, camera["label"])
        camera_labels[camera["topic"]] = f"{view_role}（{camera['label']}）"
    if len(camera_labels) != 4:
        raise ValueError("robot_config.yaml 的 cameras 必须包含四个不同的 topic")

    configured_topics = set()
    for section in ("actions", "states"):
        entries = config.get(section, {})
        if not isinstance(entries, dict):
            raise ValueError(f"机器人配置中的 {section} 必须是 mapping")
        configured_topics.update(
            entry["topic"]
            for entry in entries.values()
            if isinstance(entry, dict) and isinstance(entry.get("topic"), str)
        )

    return camera_labels, sorted(configured_topics), sorted(cameras)


def audit_mcap_file(
    mcap_path: str | Path,
    root: str | Path,
    camera_labels: dict[str, str],
    configured_topics: list[str],
) -> dict[str, Any]:
    """Read one MCAP to EOF, checking CRCs, topics, timestamps, and image decoding."""
    path = Path(mcap_path)
    root_path = Path(root)
    topics: dict[str, dict[str, Any]] = {}
    total_messages = 0

    with path.open("rb") as stream:
        reader = make_reader(stream, validate_crcs=True)
        summary = reader.get_summary()
        summary_message_count = (
            int(summary.statistics.message_count)
            if summary is not None and summary.statistics is not None
            else None
        )

        for schema, channel, message in reader.iter_messages():
            total_messages += 1
            topic = channel.topic
            record = topics.setdefault(topic, _new_topic_record())
            record["message_count"] += 1
            record["log_times"].append(int(message.log_time))
            schema_name = schema.name if schema is not None else None
            if schema_name:
                record["schemas"].add(schema_name)

            if not _is_image_schema(schema_name):
                continue

            image_info = parse_ros2_image_message(message.data, schema_name)
            if image_info is None:
                record["image_decode_failures"] += 1
                continue

            try:
                preview = decode_image_for_preview(image_info)
            except Exception:
                record["image_decode_failures"] += 1
                continue

            record["image_dimensions"][f"{preview.width}x{preview.height}"] += 1
            record["image_encodings"][image_info["encoding"]] += 1
            header_ns = (
                int(image_info["header_stamp_sec"]) * 1_000_000_000
                + int(image_info["header_stamp_nanosec"])
            )
            if header_ns > 0:
                record["header_times"].append(header_ns)

    metadata = _load_episode_metadata(path, root_path)
    rosbag_metadata = _load_rosbag_metadata(path)
    declared_topics = []
    if metadata:
        declared_topics = metadata.get("upper", {}).get("topics", [])
        if not isinstance(declared_topics, list):
            declared_topics = []

    actual_topic_names = set(topics)
    declared_topics_not_recorded = sorted(set(declared_topics) - actual_topic_names)
    image_topics = sorted(
        topic
        for topic, record in topics.items()
        if any(_is_image_schema(schema) for schema in record["schemas"])
    )
    camera_topics = sorted(topic for topic in camera_labels if topic in image_topics)
    missing_camera_topics = sorted(set(camera_labels) - set(camera_topics))
    unconfigured_image_topics = sorted(set(image_topics) - set(camera_labels))
    configured_topics_not_recorded = sorted(set(configured_topics) - actual_topic_names)

    topic_results = {}
    image_decode_failures = 0
    for topic, record in sorted(topics.items()):
        is_camera = topic in camera_topics
        result = {
            "role": camera_labels.get(topic),
            "message_count": record["message_count"],
            "schemas": sorted(record["schemas"]),
            "log_time": _timestamp_summary(record["log_times"]),
        }
        if is_camera:
            result["header_stamp"] = _timestamp_summary(record["header_times"])
            result["sixty_second_trend"] = _window_summaries(record["log_times"])
            result["image_decode_failures"] = record["image_decode_failures"]
            result["image_dimensions"] = dict(record["image_dimensions"])
            result["image_encodings"] = dict(record["image_encodings"])
            image_decode_failures += record["image_decode_failures"]
        topic_results[topic] = result

    errors = []
    if summary_message_count is not None and summary_message_count != total_messages:
        errors.append(
            f"MCAP summary 消息数 {summary_message_count} 与实际遍历数 {total_messages} 不一致"
        )
    if missing_camera_topics:
        errors.append(f"robot_config.yaml 配置但 MCAP 缺失或非图像格式的相机 topic: {missing_camera_topics}")
    for topic in camera_topics:
        schemas = topics[topic]["schemas"]
        if schemas != {EXPECTED_CAMERA_SCHEMA}:
            errors.append(
                f"相机 topic={topic} schema 应为 {EXPECTED_CAMERA_SCHEMA}，实际为 {sorted(schemas)}"
            )
    yaml_message_count = (
        rosbag_metadata["file_message_count"]
        if rosbag_metadata and rosbag_metadata["file_message_count"] is not None
        else None
    )
    if yaml_message_count is not None and yaml_message_count != total_messages:
        errors.append(
            f"metadata.yaml 声明此文件 {yaml_message_count} 条消息，实际遍历 {total_messages} 条"
        )
    if rosbag_metadata and rosbag_metadata["is_single_file_bag"]:
        for topic, expected_count in rosbag_metadata["topic_message_counts"].items():
            actual_count = topics.get(topic, {}).get("message_count", 0)
            if actual_count != expected_count:
                errors.append(
                    f"metadata.yaml topic={topic} 声明 {expected_count} 条，实际 {actual_count} 条"
                )
    if image_decode_failures:
        errors.append(f"有 {image_decode_failures} 帧图像解析或解码失败")
    if unconfigured_image_topics:
        warnings = [f"MCAP 中存在未在 robot_config.yaml 配置的图像 topic: {unconfigured_image_topics}"]
    else:
        warnings = []
    if configured_topics_not_recorded:
        warnings.append(
            "robot_config.yaml 引用但本 MCAP 未录到的动作/状态 topic（请结合机器人实际硬件确认）: "
            f"{configured_topics_not_recorded}"
        )
    metadata_only_topics_not_recorded = sorted(
        set(declared_topics_not_recorded) - set(configured_topics)
    )
    if metadata_only_topics_not_recorded:
        warnings.append(
            "meta.json 列出但 robot_config.yaml 未引用、且本 MCAP 未录到的 topic: "
            f"{metadata_only_topics_not_recorded}"
        )

    for topic in camera_topics:
        log_time = topic_results[topic]["log_time"]
        header_stamp = topic_results[topic]["header_stamp"]
        if log_time["out_of_order_count"] or log_time["duplicate_count"]:
            errors.append(
                f"相机 topic={topic} 的 log_time 存在倒退"
                f"{log_time['out_of_order_count']} 次或重复 {log_time['duplicate_count']} 次"
            )
        if header_stamp["out_of_order_count"] or header_stamp["duplicate_count"]:
            errors.append(
                f"相机 topic={topic} 的 header.stamp 存在倒退"
                f"{header_stamp['out_of_order_count']} 次或重复 {header_stamp['duplicate_count']} 次"
            )

    camera_header_times = {
        topic: topics[topic]["header_times"]
        for topic in camera_topics
    }
    camera_time_relationship = _four_camera_header_skew(camera_header_times)

    return {
        "file": str(path),
        "episode": metadata.get("episode") if metadata else None,
        "task": metadata.get("task") if metadata else None,
        "operator": metadata.get("operator") if metadata else None,
        "meta_json": metadata.get("_path") if metadata else None,
        "rosbag_metadata": rosbag_metadata,
        "metadata_topics": declared_topics,
        "total_message_count": total_messages,
        "summary_message_count": summary_message_count,
        "topic_count": len(topics),
        "camera_topics": camera_topics,
        "unconfigured_image_topics": unconfigured_image_topics,
        "declared_topics_not_recorded": declared_topics_not_recorded,
        "configured_topics_not_recorded": configured_topics_not_recorded,
        "missing_camera_topics": missing_camera_topics,
        "image_decode_failures": image_decode_failures,
        "four_camera_header_time_relationship": camera_time_relationship,
        "topics": topic_results,
        "warnings": warnings,
        "errors": errors,
        "is_valid": not errors,
    }


def audit_mcap_directory(root: str | Path) -> dict[str, Any]:
    """Audit every MCAP recursively found under an episode directory."""
    root_path = Path(root)
    if not root_path.is_dir():
        raise FileNotFoundError(f"MCAP 数据目录不存在: {root_path}")

    mcap_files = sorted(path for path in root_path.rglob("*.mcap") if path.is_file())
    if not mcap_files:
        raise FileNotFoundError(f"目录及子目录下未找到 .mcap 文件: {root_path}")

    config_path = Path(__file__).resolve().parent.parent / "Robot_Configuration" / "robot_config.yaml"
    camera_labels, configured_topics, camera_names = load_robot_camera_config(config_path)

    files = []
    for mcap_path in mcap_files:
        try:
            files.append(
                audit_mcap_file(
                    mcap_path,
                    root_path,
                    camera_labels,
                    configured_topics,
                )
            )
        except Exception as exc:
            files.append(
                {
                    "file": str(mcap_path),
                    "errors": [f"MCAP 完整遍历失败: {type(exc).__name__}: {exc}"],
                    "is_valid": False,
                }
            )

    camera_trends: dict[str, list[dict[str, Any]]] = {}
    for file_result in files:
        for topic in file_result.get("camera_topics", []):
            camera = file_result["topics"][topic]
            timing = camera["log_time"]
            camera_trends.setdefault(topic, []).append(
                {
                    "episode": file_result.get("episode"),
                    "file": file_result["file"],
                    "message_count": camera["message_count"],
                    "duration_seconds": (
                        (timing["maximum_ns"] - timing["minimum_ns"]) / 1_000_000_000
                        if timing["minimum_ns"] is not None
                        else None
                    ),
                    "interval_mean_ns": timing["interval_mean_ns"],
                    "interval_p95_ns": timing["interval_p95_ns"],
                    "interval_max_ns": timing["interval_max_ns"],
                    "image_decode_failures": camera["image_decode_failures"],
                }
            )

    return {
        "root": str(root_path),
        "robot_config": str(config_path),
        "configured_camera_names": camera_names,
        "file_count": len(files),
        "valid_file_count": sum(file["is_valid"] for file in files),
        "camera_episode_comparison": camera_trends,
        "files": files,
        "is_valid": all(file["is_valid"] for file in files),
    }

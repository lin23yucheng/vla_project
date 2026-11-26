"""Audit raw robot MCAP files for structural and camera-stream issues."""

from __future__ import annotations

import json
import math
import csv
import argparse
from bisect import bisect_left
from collections import Counter
from pathlib import Path
from typing import Any

import yaml
from mcap.reader import make_reader

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
DEFAULT_AUDIT_CONFIG = Path(__file__).resolve().parent.parent / "config" / "mcap_audit_config.json"
DEFAULT_REPORT_DIR = Path(__file__).resolve().parent.parent / "report" / "mcap_audit"
EXPECTED_CAMERA_FPS = 30.0
CAMERA_FPS_TOLERANCE = 5.0
MIN_CAMERA_FPS = EXPECTED_CAMERA_FPS - CAMERA_FPS_TOLERANCE
MAX_CAMERA_FPS = EXPECTED_CAMERA_FPS + CAMERA_FPS_TOLERANCE
MAX_CAMERA_INTERVAL_NS = 40_000_000
TREND_WINDOW_NS = 60_000_000_000


def _is_image_schema(schema_name: str | None) -> bool:
    normalized = (schema_name or "").strip().lower()
    return normalized.endswith(IMAGE_SCHEMA_SUFFIXES)


def load_audit_config(config_path: str | Path = DEFAULT_AUDIT_CONFIG) -> dict[str, Any]:
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"MCAP 审计配置不存在: {path}")
    with path.open("r", encoding="utf-8") as file:
        config = json.load(file)
    target_fps = float(config["camera_fps_target"])
    tolerance = float(config["camera_fps_tolerance"])
    window_seconds = int(config["window_seconds"])
    max_interval_ms = float(config["max_frame_interval_ms"])
    if target_fps <= 0 or tolerance < 0 or window_seconds <= 0 or max_interval_ms <= 0:
        raise ValueError("MCAP 审计配置中的 FPS、窗口长度和最大间隔必须为正数")
    config["camera_fps_min"] = max(0.0, target_fps - tolerance)
    config["camera_fps_max"] = target_fps + tolerance
    config["window_ns"] = window_seconds * 1_000_000_000
    config["max_frame_interval_ns"] = round(max_interval_ms * 1_000_000)
    config["config_path"] = str(path)
    return config


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


def _event_timestamp_anomalies(
    events: list[dict[str, Any]],
    file_path: Path,
    topic: str,
    timestamp_key: str,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    anomalies = []
    previous = None
    for event in events:
        if timestamp_key == "header_stamp_ns" and not event.get("header_valid"):
            continue
        timestamp_ns = event[timestamp_key]
        if timestamp_ns is None or timestamp_ns <= 0:
            continue
        if previous is not None:
            previous_ns = previous[timestamp_key]
            interval_ns = timestamp_ns - previous_ns
            if interval_ns < 0:
                anomalies.append(
                    _anomaly(
                        file_path,
                        topic,
                        event["message_index"],
                        event["event_index"],
                        timestamp_ns,
                        f"{timestamp_key} 倒退",
                        {"previous_ns": previous_ns, "current_ns": timestamp_ns},
                        ">= previous timestamp",
                    )
                )
            elif interval_ns == 0:
                anomalies.append(
                    _anomaly(
                        file_path,
                        topic,
                        event["message_index"],
                        event["event_index"],
                        timestamp_ns,
                        f"{timestamp_key} 重复",
                        interval_ns,
                        "> 0 ns",
                    )
                )
            elif timestamp_key == "log_time_ns" and interval_ns > config["max_frame_interval_ns"]:
                target_interval_ns = 1_000_000_000 / config["camera_fps_target"]
                estimated_missing = (
                    max(0, round(interval_ns / target_interval_ns) - 1)
                    if config["estimate_dropped_frames"]
                    else None
                )
                anomalies.append(
                    _anomaly(
                        file_path,
                        topic,
                        event["message_index"],
                        event["event_index"],
                        timestamp_ns,
                        "帧间隔超阈值（疑似丢帧）",
                        {
                            "interval_ms": interval_ns / 1_000_000,
                            "estimated_missing_frames": estimated_missing,
                            "previous_timestamp_ns": previous_ns,
                        },
                        config["max_frame_interval_ms"],
                    )
                )
        previous = event
    return anomalies


def _window_summaries(
    timestamps: list[int],
    window_ns: int = 60_000_000_000,
    max_frame_interval_ns: int = 40_000_000,
    target_fps: float = 30.0,
) -> list[dict[str, Any]]:
    if not timestamps:
        return []

    origin_ns = timestamps[0]
    windows: dict[int, dict[str, Any]] = {}
    previous_ns = None
    for timestamp_ns in timestamps:
        window_index = max(0, (timestamp_ns - origin_ns) // window_ns)
        window = windows.setdefault(
            window_index,
            {"message_count": 0, "intervals_ns": [], "gap_events": []},
        )
        window["message_count"] += 1
        if previous_ns is not None:
            interval_ns = timestamp_ns - previous_ns
            if interval_ns > 0:
                window["intervals_ns"].append(interval_ns)
                if interval_ns > max_frame_interval_ns:
                    estimated_missing = max(
                        0,
                        round(interval_ns / (1_000_000_000 / target_fps)) - 1,
                    )
                    window["gap_events"].append(estimated_missing)
        previous_ns = timestamp_ns

    return [
        {
            "start_offset_seconds": window_index * window_ns / 1_000_000_000,
            "end_offset_seconds": (window_index + 1) * window_ns / 1_000_000_000,
            "message_count": window["message_count"],
            "interval_count": len(window["intervals_ns"]),
            "observed_rate_hz": (
                1_000_000_000 / (sum(window["intervals_ns"]) / len(window["intervals_ns"]))
                if window["intervals_ns"]
                else None
            ),
            "interval_mean_ns": (
                sum(window["intervals_ns"]) / len(window["intervals_ns"])
                if window["intervals_ns"]
                else None
            ),
            "interval_p50_ns": _percentile(window["intervals_ns"], 0.50),
            "interval_p95_ns": _percentile(window["intervals_ns"], 0.95),
            "interval_p99_ns": _percentile(window["intervals_ns"], 0.99),
            "interval_max_ns": max(window["intervals_ns"])
            if window["intervals_ns"]
            else None,
            "over_threshold_gap_count": len(window["gap_events"]),
            "estimated_dropped_frames": sum(window["gap_events"]),
        }
        for window_index, window in sorted(windows.items())
    ]


def _event_window_summaries(
    events: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    valid_log_times = [event["log_time_ns"] for event in events if event["log_time_ns"] > 0]
    if not valid_log_times:
        return []
    origin_ns = min(valid_log_times)
    windows: dict[int, dict[str, Any]] = {}
    previous = None
    for event in events:
        log_time_ns = event["log_time_ns"]
        window_index = max(0, (log_time_ns - origin_ns) // config["window_ns"])
        window = windows.setdefault(
            window_index,
            {
                "message_count": 0,
                "intervals_ns": [],
                "over_threshold_gap_count": 0,
                "estimated_dropped_frames": 0,
                "timestamp_reversal_count": 0,
                "timestamp_duplicate_count": 0,
                "invalid_log_time_count": 0,
                "invalid_header_stamp_count": 0,
                "image_decode_failures": 0,
                "record_delay_ns": [],
                "timestamps": [],
            },
        )
        window["message_count"] += 1
        if log_time_ns <= 0:
            window["invalid_log_time_count"] += 1
        else:
            window["timestamps"].append(log_time_ns)
        if event.get("image_valid") is False:
            window["image_decode_failures"] += 1
        if event.get("header_stamp_ns") is not None and not event.get("header_valid"):
            window["invalid_header_stamp_count"] += 1
        if event.get("header_valid"):
            window["record_delay_ns"].append(log_time_ns - event["header_stamp_ns"])
        if previous is not None and log_time_ns > 0 and previous["log_time_ns"] > 0:
            interval_ns = log_time_ns - previous["log_time_ns"]
            if interval_ns < 0:
                window["timestamp_reversal_count"] += 1
            elif interval_ns == 0:
                window["timestamp_duplicate_count"] += 1
            else:
                window["intervals_ns"].append(interval_ns)
                if interval_ns > config["max_frame_interval_ns"]:
                    window["over_threshold_gap_count"] += 1
                    if config["estimate_dropped_frames"]:
                        window["estimated_dropped_frames"] += max(
                            0,
                            round(interval_ns / (1_000_000_000 / config["camera_fps_target"])) - 1,
                        )
        previous = event

    results = []
    for index, window in sorted(windows.items()):
        intervals = window["intervals_ns"]
        delays = window["record_delay_ns"]
        results.append(
            {
                "start_offset_seconds": index * config["window_seconds"],
                "end_offset_seconds": (index + 1) * config["window_seconds"],
                "message_count": window["message_count"],
                "interval_count": len(intervals),
                "observed_rate_hz": 1e9 / (sum(intervals) / len(intervals)) if intervals else None,
                "interval_mean_ns": sum(intervals) / len(intervals) if intervals else None,
                "interval_p50_ns": _percentile(intervals, 0.5),
                "interval_p95_ns": _percentile(intervals, 0.95),
                "interval_p99_ns": _percentile(intervals, 0.99),
                "interval_max_ns": max(intervals) if intervals else None,
                "over_threshold_gap_count": window["over_threshold_gap_count"],
                "estimated_dropped_frames": window["estimated_dropped_frames"],
                "timestamp_reversal_count": window["timestamp_reversal_count"],
                "timestamp_duplicate_count": window["timestamp_duplicate_count"],
                "invalid_log_time_count": window["invalid_log_time_count"],
                "invalid_header_stamp_count": window["invalid_header_stamp_count"],
                "image_decode_failures": window["image_decode_failures"],
                "first_log_time_ns": min(window["timestamps"]) if window["timestamps"] else None,
                "last_log_time_ns": max(window["timestamps"]) if window["timestamps"] else None,
                "record_delay_p50_ns": _percentile(delays, 0.5),
                "record_delay_p95_ns": _percentile(delays, 0.95),
                "record_delay_max_ns": max(delays) if delays else None,
            }
        )
    return results


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


def _four_camera_skew_windows(
    header_times: dict[str, list[int]],
    window_ns: int,
) -> list[dict[str, Any]]:
    available = [timestamp for values in header_times.values() for timestamp in values]
    if len(header_times) != 4 or not available:
        return []
    origin_ns = min(available)
    buckets: dict[int, dict[str, list[int]]] = {}
    for topic, timestamps in header_times.items():
        for timestamp_ns in timestamps:
            index = max(0, (timestamp_ns - origin_ns) // window_ns)
            buckets.setdefault(index, {}).setdefault(topic, []).append(timestamp_ns)
    windows = []
    for index, topics in sorted(buckets.items()):
        skew = _four_camera_header_skew(topics)
        if skew is not None:
            windows.append(
                {
                    "start_offset_seconds": index * window_ns / 1_000_000_000,
                    "end_offset_seconds": (index + 1) * window_ns / 1_000_000_000,
                    **skew,
                }
            )
    return windows


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
        "events": [],
    }


def _anomaly(
    file_path: Path,
    topic: str,
    message_index: int,
    event_index: int,
    timestamp_ns: int | None,
    kind: str,
    actual: Any,
    threshold: Any = None,
) -> dict[str, Any]:
    return {
        "file": str(file_path),
        "topic": topic,
        "message_index": message_index,
        "event_index": event_index,
        "timestamp_ns": timestamp_ns,
        "anomaly_type": kind,
        "actual_value": actual,
        "threshold": threshold,
    }


def _ensure_failures_have_anomalies(
    file_path: Path,
    errors: list[str],
    anomalies: list[dict[str, Any]],
    camera_topics: list[str],
) -> None:
    for error in errors:
        if any(
            anomaly["anomaly_type"] in error
            and str(anomaly["actual_value"]) in error
            for anomaly in anomalies
        ):
            continue
        topic = next((name for name in camera_topics if name in error), "")
        anomalies.append(
            _anomaly(
                file_path,
                topic,
                "",
                "",
                None,
                "结构性校验失败",
                error,
                "MCAP、metadata 与机器人相机配置一致",
            )
        )


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
    audit_config: dict[str, Any],
) -> dict[str, Any]:
    """Read one MCAP to EOF, checking CRCs, topics, timestamps, and image decoding."""
    path = Path(mcap_path)
    root_path = Path(root)
    topics: dict[str, dict[str, Any]] = {}
    total_messages = 0
    anomalies: list[dict[str, Any]] = []

    with path.open("rb") as stream:
        reader = make_reader(stream, validate_crcs=True)
        summary = reader.get_summary()
        summary_message_count = (
            int(summary.statistics.message_count)
            if summary is not None and summary.statistics is not None
            else None
        )

        for schema, channel, message in reader.iter_messages(log_time_order=False):
            total_messages += 1
            topic = channel.topic
            record = topics.setdefault(topic, _new_topic_record())
            record["message_count"] += 1
            message_index = record["message_count"]
            log_time_ns = int(message.log_time)
            publish_time_ns = int(message.publish_time)
            record["log_times"].append(log_time_ns)
            schema_name = schema.name if schema is not None else None
            if schema_name:
                record["schemas"].add(schema_name)

            event = {
                "message_index": message_index,
                "event_index": total_messages,
                "log_time_ns": log_time_ns,
                "publish_time_ns": publish_time_ns,
                "header_stamp_ns": None,
                "header_valid": None,
                "sequence": None,
                "image_valid": None,
            }
            record["events"].append(event)
            if topic in camera_labels and log_time_ns <= 0:
                anomalies.append(
                    _anomaly(path, topic, message_index, total_messages, log_time_ns,
                             "非法 log_time", log_time_ns, "> 0")
                )
            if topic in camera_labels and publish_time_ns <= 0:
                anomalies.append(
                    _anomaly(path, topic, message_index, total_messages, publish_time_ns,
                             "非法 publish_time", publish_time_ns, "> 0")
                )

            if not _is_image_schema(schema_name):
                continue

            image_info = parse_ros2_image_message(message.data, schema_name)
            if image_info is None:
                record["image_decode_failures"] += 1
                event["image_valid"] = False
                anomalies.append(
                    _anomaly(path, topic, message_index, total_messages, log_time_ns,
                             "图像消息无法解析", len(message.data), "有效 ROS 图像消息")
                )
                continue

            stamp_sec = int(image_info["header_stamp_sec"])
            stamp_nanosec = int(image_info["header_stamp_nanosec"])
            header_ns = stamp_sec * 1_000_000_000 + stamp_nanosec
            event["header_stamp_ns"] = header_ns
            event["image_valid"] = True
            event["header_valid"] = (
                stamp_sec >= 0
                and 0 <= stamp_nanosec < 1_000_000_000
                and header_ns > 0
            )
            if not event["header_valid"]:
                anomalies.append(
                    _anomaly(
                        path,
                        topic,
                        message_index,
                        total_messages,
                        header_ns,
                        "非法 header.stamp",
                        {"sec": stamp_sec, "nanosec": stamp_nanosec},
                        "sec >= 0, 0 <= nanosec < 1e9, timestamp > 0",
                    )
                )
            try:
                preview = decode_image_for_preview(image_info)
            except Exception:
                record["image_decode_failures"] += 1
                event["image_valid"] = False
                anomalies.append(
                    _anomaly(path, topic, message_index, total_messages, log_time_ns,
                             "图像解码失败", image_info.get("data_length"), "可解码图像")
                )
                continue

            record["image_dimensions"][f"{preview.width}x{preview.height}"] += 1
            record["image_encodings"][image_info["encoding"]] += 1
            if event["header_valid"]:
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
            result["sixty_second_trend"] = _event_window_summaries(
                record["events"], audit_config
            )
            result["image_decode_failures"] = record["image_decode_failures"]
            result["image_dimensions"] = dict(record["image_dimensions"])
            result["image_encodings"] = dict(record["image_encodings"])
            result["header_log_offsets_ns"] = [
                event["log_time_ns"] - event["header_stamp_ns"]
                for event in record["events"]
                if event.get("header_valid")
            ]
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
        anomalies.extend(
            _event_timestamp_anomalies(
                topics[topic]["events"], path, topic, "log_time_ns", audit_config
            )
        )
        anomalies.extend(
            _event_timestamp_anomalies(
                topics[topic]["events"], path, topic, "header_stamp_ns", audit_config
            )
        )
        schemas = topics[topic]["schemas"]
        if schemas != {EXPECTED_CAMERA_SCHEMA}:
            errors.append(
                f"相机 topic={topic} schema 应为 {EXPECTED_CAMERA_SCHEMA}，实际为 {sorted(schemas)}"
            )
        dimensions = topic_results[topic]["image_dimensions"]
        if len(dimensions) != 1:
            errors.append(
                f"相机 topic={topic} 图像分辨率在本轮采集中不一致或没有成功解码帧: "
                f"{dimensions}"
            )
        encodings = topic_results[topic]["image_encodings"]
        if len(encodings) != 1:
            errors.append(
                f"相机 topic={topic} 解码图像格式在本轮采集中不一致或没有成功解码帧: "
                f"{encodings}"
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

    camera_header_times = {
        topic: topics[topic]["header_times"]
        for topic in camera_topics
    }
    camera_time_relationship = _four_camera_header_skew(camera_header_times)
    camera_skew_windows = _four_camera_skew_windows(
        camera_header_times, audit_config["window_ns"]
    )
    if audit_config["max_cross_camera_skew_ms"] is not None:
        skew_limit_ns = audit_config["max_cross_camera_skew_ms"] * 1_000_000
        for window in camera_skew_windows:
            if window["maximum_ns"] > skew_limit_ns:
                anomalies.append(
                    _anomaly(
                        path,
                        "四路相机",
                        0,
                        0,
                        None,
                        "窗口相机时间差超阈值",
                        {
                            "window_start_seconds": window["start_offset_seconds"],
                            "maximum_skew_ms": window["maximum_ns"] / 1_000_000,
                        },
                        audit_config["max_cross_camera_skew_ms"],
                    )
                )
    for topic in camera_topics:
        timing = topic_results[topic]["log_time"]
        if timing["observed_rate_hz"] is None:
            errors.append(f"相机 topic={topic} 消息不足，无法计算有效帧率")
        elif not audit_config["camera_fps_min"] <= timing["observed_rate_hz"] <= audit_config["camera_fps_max"]:
            anomalies.append(
                _anomaly(
                    path, topic, 0, 0, timing["first_ns"], "整段帧率超范围",
                    timing["observed_rate_hz"],
                    [audit_config["camera_fps_min"], audit_config["camera_fps_max"]],
                )
            )
            errors.append(
                f"相机 topic={topic} 整段观测帧率 {timing['observed_rate_hz']:.3f} FPS，"
                f"超出允许范围 {audit_config['camera_fps_min']:.0f}-"
                f"{audit_config['camera_fps_max']:.0f} FPS"
            )
        for window in topic_results[topic]["sixty_second_trend"]:
            window_rate = window["observed_rate_hz"]
            if window_rate is not None and not audit_config["camera_fps_min"] <= window_rate <= audit_config["camera_fps_max"]:
                anomalies.append(
                    _anomaly(
                        path, topic, 0, 0, None, "窗口帧率超范围",
                        {
                            "window_start_seconds": window["start_offset_seconds"],
                            "fps": window_rate,
                        },
                        [audit_config["camera_fps_min"], audit_config["camera_fps_max"]],
                    )
                )
                errors.append(
                    f"相机 topic={topic} 在 {window['start_offset_seconds']:.0f}-"
                    f"{window['end_offset_seconds']:.0f}s 窗口帧率 {window_rate:.3f} FPS 超出范围"
                )
    delay_values = [
        event["log_time_ns"] - event["header_stamp_ns"]
        for topic in camera_topics
        for event in topics[topic]["events"]
        if event.get("header_valid")
    ]
    capture_record_delay = (
        {
            "sample_count": len(delay_values),
            "minimum_ns": min(delay_values),
            "p50_ns": _percentile(delay_values, 0.5),
            "p95_ns": _percentile(delay_values, 0.95),
            "p99_ns": _percentile(delay_values, 0.99),
            "maximum_ns": max(delay_values),
            "negative_count": sum(value < 0 for value in delay_values),
        }
        if delay_values
        else None
    )
    if (
        capture_record_delay
        and audit_config["header_and_log_share_clock"]
        and audit_config["max_capture_record_delay_ms"] is not None
    ):
        threshold_ns = audit_config["max_capture_record_delay_ms"] * 1_000_000
        for topic in camera_topics:
            for event in topics[topic]["events"]:
                header_ns = event["header_stamp_ns"]
                if not event.get("header_valid"):
                    continue
                delay_ns = event["log_time_ns"] - header_ns
                if delay_ns > threshold_ns:
                    anomalies.append(
                        _anomaly(path, topic, event["message_index"], event["event_index"],
                                 event["log_time_ns"], "采集到记录延迟超阈值",
                                 delay_ns / 1_000_000,
                                 audit_config["max_capture_record_delay_ms"])
                    )
    for anomaly in anomalies:
        errors.append(
            f"{anomaly['topic']} 消息#{anomaly['message_index']} "
            f"{anomaly['anomaly_type']}：实际值={anomaly['actual_value']}，"
            f"阈值={anomaly['threshold']}"
        )
    _ensure_failures_have_anomalies(path, errors, anomalies, list(camera_labels))
    if not audit_config["header_and_log_share_clock"] and capture_record_delay:
        delay_interpretation = "时钟关系未确认，该差值仅作诊断"
    else:
        delay_interpretation = "按配置的共同时钟假设解释为采集到记录时间差"

    coverage = {}
    for topic in camera_topics:
        header_summary = topic_results[topic]["header_stamp"]
        coverage[topic] = {
            "first_ns": header_summary["first_ns"],
            "last_ns": header_summary["last_ns"],
            "duration_ns": (
                header_summary["last_ns"] - header_summary["first_ns"]
                if header_summary["first_ns"] is not None and header_summary["last_ns"] is not None
                else None
            ),
        }

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
        "four_camera_skew_windows": camera_skew_windows,
        "camera_coverage": coverage,
        "capture_record_delay": capture_record_delay,
        "capture_record_delay_interpretation": delay_interpretation,
        "sequence_check_available": False,
        "sequence_check_note": "sensor_msgs/msg/CompressedImage 的 Header 不包含序列号，无法据此确认丢帧",
        "anomalies": anomalies,
        "topics": topic_results,
        "warnings": warnings,
        "errors": errors,
        "is_valid": not errors,
    }


def audit_mcap_directory(
    root: str | Path,
    audit_config_path: str | Path = DEFAULT_AUDIT_CONFIG,
    report_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Audit every MCAP recursively found under an episode directory."""
    root_path = Path(root)
    if not root_path.is_dir():
        raise FileNotFoundError(f"MCAP 数据目录不存在: {root_path}")

    mcap_files = sorted(path for path in root_path.rglob("*.mcap") if path.is_file())
    if not mcap_files:
        raise FileNotFoundError(f"目录及子目录下未找到 .mcap 文件: {root_path}")

    config_path = Path(__file__).resolve().parent.parent / "Robot_Configuration" / "robot_config.yaml"
    audit_config = load_audit_config(audit_config_path)
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
                    audit_config,
                )
            )
        except Exception as exc:
            error_text = f"MCAP 完整遍历失败: {type(exc).__name__}: {exc}"
            files.append(
                {
                    "file": str(mcap_path),
                    "errors": [error_text],
                    "anomalies": [
                        {
                            "file": str(mcap_path),
                            "topic": "",
                            "message_index": "",
                            "event_index": "",
                            "timestamp_ns": "",
                            "anomaly_type": "文件解析失败",
                            "actual_value": error_text,
                            "threshold": "文件应能完整读取",
                        }
                    ],
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

    result = {
        "root": str(root_path),
        "robot_config": str(config_path),
        "audit_config": audit_config["config_path"],
        "configured_camera_names": camera_names,
        "camera_fps_expectation": {
            "target_fps": audit_config["camera_fps_target"],
            "tolerance_fps": audit_config["camera_fps_tolerance"],
            "accepted_range_fps": [audit_config["camera_fps_min"], audit_config["camera_fps_max"]],
            "maximum_interval_ms": audit_config["max_frame_interval_ms"],
            "window_seconds": audit_config["window_seconds"],
        },
        "file_count": len(files),
        "valid_file_count": sum(file["is_valid"] for file in files),
        "camera_episode_comparison": camera_trends,
        "files": files,
        "is_valid": all(file["is_valid"] for file in files),
    }
    _add_cross_camera_coverage(result)
    _add_episode_trends(result)
    selected_report_dir = Path(report_dir) if report_dir else DEFAULT_REPORT_DIR
    result["report_files"] = write_mcap_csv_reports(result, selected_report_dir)
    return result


def _add_cross_camera_coverage(result: dict[str, Any]) -> None:
    for file_result in result["files"]:
        coverage = file_result.get("camera_coverage", {})
        spans = [
            value
            for value in coverage.values()
            if value.get("first_ns") is not None and value.get("last_ns") is not None
        ]
        if not spans:
            file_result["four_camera_coverage"] = None
            continue
        common_start = max(item["first_ns"] for item in spans)
        common_end = min(item["last_ns"] for item in spans)
        file_result["four_camera_coverage"] = {
            "common_start_ns": common_start,
            "common_end_ns": common_end,
            "common_duration_ns": max(0, common_end - common_start),
            "first_frame_spread_ns": max(item["first_ns"] for item in spans)
            - min(item["first_ns"] for item in spans),
            "last_frame_spread_ns": max(item["last_ns"] for item in spans)
            - min(item["last_ns"] for item in spans),
            "per_camera": coverage,
        }


def _add_episode_trends(result: dict[str, Any]) -> None:
    trends = {}
    ordered_files = sorted(
        result["files"],
        key=lambda item: (
            item.get("episode") is None,
            item.get("episode") if item.get("episode") is not None else item.get("file", ""),
        ),
    )
    all_topics = sorted(
        {
            topic
            for file_result in ordered_files
            for topic in file_result.get("camera_topics", [])
        }
    )
    for topic_name in all_topics:
        rows = []
        for file_result in ordered_files:
            camera = file_result.get("topics", {}).get(topic_name)
            if camera is None:
                continue
            timing = camera["log_time"]
            rows.append(
                {
                    "episode": file_result.get("episode"),
                    "file": file_result["file"],
                    "fps": timing["observed_rate_hz"],
                    "max_interval_ns": timing["interval_max_ns"],
                    "decode_failures": camera.get("image_decode_failures", 0),
                }
            )
            file_result.setdefault("within_episode_trends", {})[topic_name] = {
                "fps_direction": _direction(
                    [window["observed_rate_hz"] for window in camera.get("sixty_second_trend", [])]
                ),
                "max_interval_direction": _direction(
                    [window["interval_max_ns"] for window in camera.get("sixty_second_trend", [])]
                ),
            }
        trends[topic_name] = {
            "episodes": rows,
            "fps_direction": _direction([row["fps"] for row in rows]),
            "max_interval_direction": _direction(
                [row["max_interval_ns"] for row in rows]
            ),
            "interpretation": "描述性趋势，轮次动作/环境不同，不能单独归因于版本变化",
        }
    result["episode_trends"] = trends


def _direction(values: list[float | int | None]) -> str:
    usable = [value for value in values if value is not None]
    if len(usable) < 2:
        return "样本不足"
    if all(right >= left for left, right in zip(usable, usable[1:])) and any(
        right > left for left, right in zip(usable, usable[1:])
    ):
        return "逐轮上升"
    if all(right <= left for left, right in zip(usable, usable[1:])) and any(
        right < left for left, right in zip(usable, usable[1:])
    ):
        return "逐轮下降"
    return "无单向趋势"


def write_mcap_csv_reports(
    result: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    summary_path = output_path / "summary.csv"
    window_path = output_path / "window_summary.csv"
    anomalies_path = output_path / "anomalies.csv"

    summary_fields = [
        "episode", "file", "topic", "role", "message_count", "fps",
        "duration_seconds", "first_header_ns", "last_header_ns",
        "p50_interval_ms", "p95_interval_ms", "p99_interval_ms",
        "max_interval_ms", "over_threshold_gap_count", "estimated_dropped_frames",
        "resolution_counts", "decode_failures", "timestamp_reversals",
        "timestamp_duplicates", "header_log_offset_p50_ms", "header_log_offset_p95_ms",
        "common_coverage_seconds", "first_frame_spread_ms", "last_frame_spread_ms",
        "sequence_check_available", "status",
    ]
    window_fields = [
        "episode", "file", "topic", "role", "window_start_seconds",
        "window_end_seconds", "message_count", "fps", "p50_interval_ms",
        "p95_interval_ms", "p99_interval_ms", "max_interval_ms",
        "over_threshold_gap_count", "estimated_dropped_frames",
        "decode_failures", "timestamp_reversals", "timestamp_duplicates",
        "invalid_log_time_count", "invalid_header_stamp_count",
        "record_delay_p50_ms", "record_delay_p95_ms", "record_delay_max_ms",
        "four_camera_skew_p95_ms", "four_camera_skew_max_ms",
    ]
    anomaly_fields = [
        "file", "episode", "topic", "message_index", "event_index",
        "timestamp_ns", "anomaly_type", "actual_value", "threshold",
    ]

    with summary_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=summary_fields)
        writer.writeheader()
        for file_result in result["files"]:
            for topic in file_result.get("camera_topics", []):
                camera = file_result["topics"][topic]
                timing = camera["log_time"]
                header = camera["header_stamp"]
                coverage = file_result.get("four_camera_coverage") or {}
                header_log_offsets = camera.get("header_log_offsets_ns", [])
                writer.writerow(
                    {
                        "episode": file_result.get("episode"),
                        "file": file_result["file"],
                        "topic": topic,
                        "role": camera.get("role"),
                        "message_count": camera["message_count"],
                        "fps": timing["observed_rate_hz"],
                        "duration_seconds": (
                            (timing["maximum_ns"] - timing["minimum_ns"]) / 1e9
                            if timing["minimum_ns"] is not None
                            else None
                        ),
                        "first_header_ns": header["first_ns"],
                        "last_header_ns": header["last_ns"],
                        "p50_interval_ms": _ms(timing["interval_p50_ns"]),
                        "p95_interval_ms": _ms(timing["interval_p95_ns"]),
                        "p99_interval_ms": _ms(timing["interval_p99_ns"]),
                        "max_interval_ms": _ms(timing["interval_max_ns"]),
                        "over_threshold_gap_count": sum(
                            1 for item in file_result.get("anomalies", [])
                            if item["topic"] == topic and item["anomaly_type"] == "帧间隔超阈值（疑似丢帧）"
                        ),
                        "estimated_dropped_frames": sum(
                            item["actual_value"].get("estimated_missing_frames") or 0
                            for item in file_result.get("anomalies", [])
                            if item["topic"] == topic and isinstance(item["actual_value"], dict)
                        ),
                        "resolution_counts": json.dumps(camera["image_dimensions"], ensure_ascii=False),
                        "decode_failures": camera["image_decode_failures"],
                        "timestamp_reversals": timing["out_of_order_count"],
                        "timestamp_duplicates": timing["duplicate_count"],
                        "header_log_offset_p50_ms": _ms(_percentile(header_log_offsets, 0.5)),
                        "header_log_offset_p95_ms": _ms(_percentile(header_log_offsets, 0.95)),
                        "common_coverage_seconds": (
                            coverage.get("common_duration_ns", 0) / 1e9
                            if coverage
                            else None
                        ),
                        "first_frame_spread_ms": _ms(coverage.get("first_frame_spread_ns")),
                        "last_frame_spread_ms": _ms(coverage.get("last_frame_spread_ns")),
                        "sequence_check_available": file_result.get("sequence_check_available", False),
                        "status": "PASS" if file_result["is_valid"] else "FAIL",
                    }
                )

    with window_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=window_fields)
        writer.writeheader()
        for file_result in result["files"]:
            for topic in file_result.get("camera_topics", []):
                camera = file_result["topics"][topic]
                for window in camera["sixty_second_trend"]:
                    matching_skew = next(
                        (
                            item
                            for item in file_result.get("four_camera_skew_windows", [])
                            if item["start_offset_seconds"] == window["start_offset_seconds"]
                        ),
                        None,
                    )
                    writer.writerow(
                        {
                            "episode": file_result.get("episode"),
                            "file": file_result["file"],
                            "topic": topic,
                            "role": camera.get("role"),
                            "window_start_seconds": window["start_offset_seconds"],
                            "window_end_seconds": window["end_offset_seconds"],
                            "message_count": window["message_count"],
                            "fps": window["observed_rate_hz"],
                            "p50_interval_ms": _ms(window["interval_p50_ns"]),
                            "p95_interval_ms": _ms(window["interval_p95_ns"]),
                            "p99_interval_ms": _ms(window["interval_p99_ns"]),
                            "max_interval_ms": _ms(window["interval_max_ns"]),
                            "over_threshold_gap_count": window["over_threshold_gap_count"],
                            "estimated_dropped_frames": window["estimated_dropped_frames"],
                            "decode_failures": window["image_decode_failures"],
                            "timestamp_reversals": window["timestamp_reversal_count"],
                            "timestamp_duplicates": window["timestamp_duplicate_count"],
                            "invalid_log_time_count": window["invalid_log_time_count"],
                            "invalid_header_stamp_count": window["invalid_header_stamp_count"],
                            "record_delay_p50_ms": _ms(window["record_delay_p50_ns"]),
                            "record_delay_p95_ms": _ms(window["record_delay_p95_ns"]),
                            "record_delay_max_ms": _ms(window["record_delay_max_ns"]),
                            "four_camera_skew_p95_ms": (
                                _ms(matching_skew["p95_ns"]) if matching_skew else None
                            ),
                            "four_camera_skew_max_ms": (
                                _ms(matching_skew["maximum_ns"]) if matching_skew else None
                            ),
                        }
                    )

    with anomalies_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=anomaly_fields)
        writer.writeheader()
        for file_result in result["files"]:
            for anomaly in file_result.get("anomalies", []):
                writer.writerow({"episode": file_result.get("episode"), **anomaly})

    return {
        "summary_csv": str(summary_path),
        "window_summary_csv": str(window_path),
        "anomalies_csv": str(anomalies_path),
    }


def _ms(value: int | float | None) -> float | None:
    return value / 1_000_000 if value is not None else None


def render_mcap_report(result: dict[str, Any]) -> str:
    """Render a concise human-readable report while retaining raw JSON separately."""
    expectation = result["camera_fps_expectation"]
    status = "PASS" if result["is_valid"] else "FAIL"
    lines = [
        f"原始 MCAP 稳定性审计：{status}",
        f"数据目录：{result['root']}",
        f"机器人配置：{result['robot_config']}",
        f"审计阈值配置：{result.get('audit_config', '未记录')}",
        (
            f"判定标准：目标 {expectation['target_fps']:.0f} FPS，允许 "
            f"{expectation['accepted_range_fps'][0]:.0f}-"
            f"{expectation['accepted_range_fps'][1]:.0f} FPS；"
            f"单次最大帧间隔 {expectation['maximum_interval_ms']:.1f} ms"
        ),
        f"Episode 文件：{result['valid_file_count']}/{result['file_count']} 通过",
    ]

    for file_result in result["files"]:
        episode = file_result.get("episode")
        episode_label = f"Episode {episode}" if episode is not None else "Episode 未知"
        episode_status = "PASS" if file_result.get("is_valid") else "FAIL"
        lines.extend(
            [
                "",
                f"{episode_label} [{episode_status}]",
                f"文件：{file_result['file']}",
                "视角 | 帧数 | FPS | 间隔 P50/P95/P99 | 最大间隔 | 分辨率 | 解码失败",
            ]
        )

        for topic in file_result.get("camera_topics", []):
            camera = file_result.get("topics", {}).get(topic)
            if not camera:
                continue
            timing = camera["log_time"]
            interval_values = [
                timing.get("interval_p50_ns"),
                timing.get("interval_p95_ns"),
                timing.get("interval_p99_ns"),
            ]
            interval_text = "/".join(
                f"{value / 1_000_000:.2f}"
                if value is not None
                else "N/A"
                for value in interval_values
            )
            maximum_interval = timing.get("interval_max_ns")
            maximum_text = (
                f"{maximum_interval / 1_000_000:.2f} ms"
                if maximum_interval is not None
                else "N/A"
            )
            rate = timing.get("observed_rate_hz")
            rate_text = f"{rate:.2f}" if rate is not None else "N/A"
            dimensions = ", ".join(
                f"{size} ({count})"
                for size, count in sorted(camera.get("image_dimensions", {}).items())
            ) or "无有效图像"
            lines.append(
                f"{camera.get('role') or topic} | {camera['message_count']} | "
                f"{rate_text} | {interval_text} ms | {maximum_text} | "
                f"{dimensions} | {camera.get('image_decode_failures', 'N/A')}"
            )

        relationship = file_result.get("four_camera_header_time_relationship")
        if relationship:
            lines.append(
                "四路 header.stamp 最近帧差（同一时钟假设）: "
                f"P50 {relationship['p50_ns'] / 1_000_000:.3f} ms, "
                f"P95 {relationship['p95_ns'] / 1_000_000:.3f} ms, "
                f"最大 {relationship['maximum_ns'] / 1_000_000:.3f} ms"
            )
        skew_windows = file_result.get("four_camera_skew_windows", [])
        if skew_windows:
            lines.append(
                "四路时间差窗口 P95："
                + "; ".join(
                    f"{window['start_offset_seconds']:.0f}-"
                    f"{window['end_offset_seconds']:.0f}s "
                    f"{window['p95_ns'] / 1_000_000:.3f} ms"
                    for window in skew_windows
                )
            )
        coverage = file_result.get("four_camera_coverage")
        if coverage:
            lines.append(
                "四路共同有效覆盖："
                f"{coverage['common_duration_ns'] / 1e9:.3f}s；"
                f"首帧跨度 {coverage['first_frame_spread_ns'] / 1e6:.3f} ms，"
                f"末帧跨度 {coverage['last_frame_spread_ns'] / 1e6:.3f} ms"
            )
        delay = file_result.get("capture_record_delay")
        if delay:
            lines.append(
                "header.stamp 到 log_time 差值（"
                f"{file_result.get('capture_record_delay_interpretation', '诊断')}）："
                f"P50 {_ms(delay['p50_ns']):.3f} ms，P95 {_ms(delay['p95_ns']):.3f} ms，"
                f"最大 {_ms(delay['maximum_ns']):.3f} ms"
            )
        lines.append(file_result.get("sequence_check_note", "序列号检查未配置"))

        window_rows = []
        for topic in file_result.get("camera_topics", []):
            camera = file_result.get("topics", {}).get(topic, {})
            for window in camera.get("sixty_second_trend", []):
                rate = window.get("observed_rate_hz")
                rate_text = f"{rate:.2f} FPS" if rate is not None else "N/A"
                window_rows.append(
                    f"{camera.get('role') or topic} "
                    f"{window['start_offset_seconds']:.0f}-"
                    f"{window['end_offset_seconds']:.0f}s {rate_text}"
                )
        if window_rows:
            window_seconds = expectation.get("window_seconds", 60)
            lines.append(f"{window_seconds} 秒窗口帧率：" + "; ".join(window_rows))
        within_trends = file_result.get("within_episode_trends", {})
        if within_trends:
            lines.append(
                "本轮窗口趋势："
                + "; ".join(
                    f"{file_result['topics'][topic].get('role') or topic} "
                    f"FPS {trend['fps_direction']}，最大间隔 {trend['max_interval_direction']}"
                    for topic, trend in within_trends.items()
                )
            )
        errors = file_result.get("errors", [])
        warnings = file_result.get("warnings", [])
        if errors:
            lines.append("异常：")
            lines.extend(f"  - {error}" for error in errors)
        if warnings:
            lines.append("数据提示（不计入相机审计失败）：")
            lines.extend(f"  - {warning}" for warning in warnings)

    if result.get("episode_trends"):
        lines.extend(["", "连续多轮趋势（描述性，不单独判失败）："])
        for topic, trend in result["episode_trends"].items():
            lines.append(
                f"{topic}: FPS {trend['fps_direction']}，"
                f"最大间隔 {trend['max_interval_direction']}"
            )

    if result.get("report_files"):
        lines.extend(
            [
                "",
                "CSV 报告：",
                f"  汇总：{result['report_files']['summary_csv']}",
                f"  分窗口：{result['report_files']['window_summary_csv']}",
                f"  异常明细：{result['report_files']['anomalies_csv']}",
            ]
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="审计机器人原始 MCAP 相机数据")
    parser.add_argument("--input-dir", required=True, help="包含 episode 子目录的 MCAP 根目录")
    parser.add_argument(
        "--audit-config",
        default=str(DEFAULT_AUDIT_CONFIG),
        help="独立 MCAP 审计 JSON 配置路径",
    )
    parser.add_argument("--output-dir", help="CSV 输出目录，默认写入项目 report/mcap_audit")
    args = parser.parse_args(argv)
    result = audit_mcap_directory(args.input_dir, args.audit_config, args.output_dir)
    print(render_mcap_report(result))
    return 0 if result["is_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

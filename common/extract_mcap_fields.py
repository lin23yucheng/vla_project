"""按机器人配置从多个 MCAP 文件提取指定时间附近的 ROS2 字段值。"""

import json
import math
from pathlib import Path
from typing import Any

from mcap_ros2.reader import read_ros2_messages


POSE_TOPICS = ("/pika_pose_l", "/pika_pose_r")
CONFIG_SECTIONS = ("action", "observation_state")


def load_configured_topic_fields(
    config_path: str | Path,
    sections: tuple[str, ...] = CONFIG_SECTIONS,
    topics: tuple[str, ...] = POSE_TOPICS,
) -> list[dict[str, Any]]:
    """读取配置中指定 section/topic 的 fields 列表。"""
    config_file = Path(config_path)
    if not config_file.is_file():
        raise FileNotFoundError(f"机器人配置文件不存在: {config_file}")

    with config_file.open("r", encoding="utf-8") as file:
        config_data = json.load(file)

    entries: list[dict[str, Any]] = []
    for section in sections:
        section_items = config_data.get(section)
        if not isinstance(section_items, list):
            raise ValueError(f"机器人配置缺少 list 类型的 {section}")

        for topic in topics:
            matching_items = [item for item in section_items if isinstance(item, dict) and item.get("topic") == topic]
            if len(matching_items) != 1:
                raise ValueError(f"机器人配置 {section} 中 topic={topic} 应有且仅有一个条目，实际为 {len(matching_items)}")

            item = matching_items[0]
            fields = item.get("fields")
            if not isinstance(fields, list) or not fields or not all(isinstance(field, str) for field in fields):
                raise ValueError(f"机器人配置 {section} 中 topic={topic} 的 fields 必须是非空字符串列表")
            entries.append(
                {
                    "section": section,
                    "name": item.get("name"),
                    "group": item.get("group"),
                    "topic": topic,
                    "fields": fields,
                }
            )
    return entries


def get_nested_field_value(message: Any, field_path: str) -> Any:
    """从动态 ROS2 消息对象中按点分路径读取字段。"""
    current = message
    for part in field_path.split("."):
        if isinstance(current, dict):
            if part not in current:
                raise KeyError(f"字段路径 {field_path} 缺少节点 {part}")
            current = current[part]
        else:
            if not hasattr(current, part):
                raise AttributeError(f"字段路径 {field_path} 缺少属性 {part}")
            current = getattr(current, part)
    return current


def _scan_nearest_messages(
    mcap_files: list[Path],
    topics: list[str],
    target_ns: int,
    start_time: int | None,
    end_time: int | None,
) -> tuple[dict[str, dict[str, Any]], int]:
    nearest: dict[str, dict[str, Any]] = {}
    scanned_messages = 0
    for mcap_file in mcap_files:
        for decoded in read_ros2_messages(
            mcap_file,
            topics=topics,
            start_time=start_time,
            end_time=end_time,
        ):
            scanned_messages += 1
            topic = decoded.channel.topic
            diff_ns = abs(decoded.publish_time_ns - target_ns)
            current = nearest.get(topic)
            if current is None or diff_ns < current["diff_ns"]:
                nearest[topic] = {
                    "mcap_file": str(mcap_file),
                    "topic": topic,
                    "schema_name": decoded.schema.name,
                    "matched_publish_ns": decoded.publish_time_ns,
                    "matched_log_ns": decoded.log_time_ns,
                    "diff_ns": diff_ns,
                    "ros_message": decoded.ros_msg,
                }
    return nearest, scanned_messages


def extract_configured_topic_fields_at_time(
    config_path: str | Path,
    mcap_dir: str | Path,
    target_ns: int,
    search_window_ns: int = 1_000_000_000,
) -> dict[str, Any]:
    """从多个 MCAP 中查找最近消息，并提取配置 fields 对应的值。"""
    config_entries = load_configured_topic_fields(config_path)
    mcap_root = Path(mcap_dir)
    if not mcap_root.is_dir():
        raise FileNotFoundError(f"mcap 目录不存在: {mcap_root}")
    mcap_files = sorted(path for path in mcap_root.glob("*.mcap") if path.is_file())
    if not mcap_files:
        raise FileNotFoundError(f"mcap 目录下未找到 .mcap 文件: {mcap_root}")

    topic_fields: dict[str, list[str]] = {}
    for entry in config_entries:
        topic_fields.setdefault(entry["topic"], [])
        for field in entry["fields"]:
            if field not in topic_fields[entry["topic"]]:
                topic_fields[entry["topic"]].append(field)

    target_ns = int(target_ns)
    topics = list(topic_fields)
    nearest, scanned_messages = _scan_nearest_messages(
        mcap_files=mcap_files,
        topics=topics,
        target_ns=target_ns,
        start_time=target_ns - search_window_ns,
        end_time=target_ns + search_window_ns + 1,
    )

    missing_topics = [topic for topic in topics if topic not in nearest]
    used_full_scan_fallback = bool(missing_topics)
    if missing_topics:
        fallback_nearest, fallback_count = _scan_nearest_messages(
            mcap_files=mcap_files,
            topics=missing_topics,
            target_ns=target_ns,
            start_time=None,
            end_time=None,
        )
        nearest.update(fallback_nearest)
        scanned_messages += fallback_count

    still_missing = [topic for topic in topics if topic not in nearest]
    if still_missing:
        raise ValueError(f"多个 mcap 文件中未找到 topic: {still_missing}")

    topic_results: dict[str, dict[str, Any]] = {}
    for topic, fields in topic_fields.items():
        nearest_message = nearest[topic]
        values = {
            field: get_nested_field_value(nearest_message["ros_message"], field)
            for field in fields
        }
        topic_results[topic] = {
            key: value
            for key, value in nearest_message.items()
            if key != "ros_message"
        }
        topic_results[topic]["fields"] = fields
        topic_results[topic]["values"] = values

    configured_results = []
    for entry in config_entries:
        topic_result = topic_results[entry["topic"]]
        configured_results.append(
            {
                **entry,
                "mcap_file": topic_result["mcap_file"],
                "schema_name": topic_result["schema_name"],
                "target_ns": target_ns,
                "matched_publish_ns": topic_result["matched_publish_ns"],
                "matched_log_ns": topic_result["matched_log_ns"],
                "diff_ns": topic_result["diff_ns"],
                "values": {
                    field: topic_result["values"][field]
                    for field in entry["fields"]
                },
            }
        )

    return {
        "config_file": str(Path(config_path)),
        "mcap_dir": str(mcap_root),
        "target_ns": target_ns,
        "search_window_ns": search_window_ns,
        "used_full_scan_fallback": used_full_scan_fallback,
        "scanned_messages": scanned_messages,
        "topic_results": topic_results,
        "configured_results": configured_results,
    }


def quaternion_to_euler_zyx(qx: float, qy: float, qz: float, qw: float) -> dict[str, float]:
    """按 ZYX 旋转顺序将四元数转换为 yaw、pitch、roll。"""
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm == 0:
        raise ValueError("四元数模长为 0，无法转换欧拉角")
    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm

    sin_roll_cos_pitch = 2.0 * (qw * qx + qy * qz)
    cos_roll_cos_pitch = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sin_roll_cos_pitch, cos_roll_cos_pitch)

    sin_pitch = 2.0 * (qw * qy - qz * qx)
    pitch = math.copysign(math.pi / 2.0, sin_pitch) if abs(sin_pitch) >= 1.0 else math.asin(sin_pitch)

    sin_yaw_cos_pitch = 2.0 * (qw * qz + qx * qy)
    cos_yaw_cos_pitch = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(sin_yaw_cos_pitch, cos_yaw_cos_pitch)
    return {"yaw": yaw, "pitch": pitch, "roll": roll}


def _load_observation_gripper_entries(config_path: str | Path) -> dict[str, dict[str, Any]]:
    with Path(config_path).open("r", encoding="utf-8") as file:
        config_data = json.load(file)
    entries: dict[str, dict[str, Any]] = {}
    for item in config_data.get("observation_state", []):
        if not isinstance(item, dict) or item.get("fields") != ["angle"]:
            continue
        group = item.get("group")
        topic = item.get("topic")
        if group in {"left", "right"} and isinstance(topic, str):
            entries[group] = {
                "name": item.get("name"),
                "group": group,
                "topic": topic,
                "fields": ["angle"],
            }
    missing_groups = [group for group in ("left", "right") if group not in entries]
    if missing_groups:
        raise ValueError(f"observation_state 缺少夹爪 angle 配置: {missing_groups}")
    return entries


def extract_robot_vectors_at_time(
    config_path: str | Path,
    mcap_dir: str | Path,
    target_ns: int,
    search_window_ns: int = 1_000_000_000,
) -> dict[str, Any]:
    """从 MCAP 提取位姿和夹爪值，生成四组 [x,y,z,roll,pitch,yaw,angle]。"""
    pose_entries = load_configured_topic_fields(config_path)
    gripper_entries = _load_observation_gripper_entries(config_path)
    mcap_root = Path(mcap_dir)
    if not mcap_root.is_dir():
        raise FileNotFoundError(f"mcap 目录不存在: {mcap_root}")
    mcap_files = sorted(path for path in mcap_root.glob("*.mcap") if path.is_file())
    if not mcap_files:
        raise FileNotFoundError(f"mcap 目录下未找到 .mcap 文件: {mcap_root}")

    topics = list(dict.fromkeys(
        [entry["topic"] for entry in pose_entries]
        + [entry["topic"] for entry in gripper_entries.values()]
    ))
    target_ns = int(target_ns)
    nearest, scanned_messages = _scan_nearest_messages(
        mcap_files,
        topics,
        target_ns,
        target_ns - search_window_ns,
        target_ns + search_window_ns + 1,
    )
    missing_topics = [topic for topic in topics if topic not in nearest]
    used_full_scan_fallback = bool(missing_topics)
    if missing_topics:
        fallback, fallback_count = _scan_nearest_messages(
            mcap_files, missing_topics, target_ns, None, None
        )
        nearest.update(fallback)
        scanned_messages += fallback_count
    still_missing = [topic for topic in topics if topic not in nearest]
    if still_missing:
        raise ValueError(f"多个 mcap 文件中未找到 topic: {still_missing}")

    vector_labels = ["x", "y", "z", "roll", "pitch", "yaw", "angle"]
    vectors = []
    for pose_entry in pose_entries:
        group = pose_entry["group"]
        if group not in gripper_entries:
            raise ValueError(f"位姿配置缺少有效 group: {pose_entry}")
        pose_message = nearest[pose_entry["topic"]]
        pose_values = {
            field: float(get_nested_field_value(pose_message["ros_message"], field))
            for field in pose_entry["fields"]
        }
        euler = quaternion_to_euler_zyx(
            pose_values["pose.orientation.x"],
            pose_values["pose.orientation.y"],
            pose_values["pose.orientation.z"],
            pose_values["pose.orientation.w"],
        )

        gripper_entry = gripper_entries[group]
        gripper_message = nearest[gripper_entry["topic"]]
        angle = float(get_nested_field_value(gripper_message["ros_message"], "angle"))
        values_by_label = {
            "x": pose_values["pose.position.x"],
            "y": pose_values["pose.position.y"],
            "z": pose_values["pose.position.z"],
            "yaw": euler["yaw"],
            "pitch": euler["pitch"],
            "roll": euler["roll"],
            "angle": angle,
        }
        vectors.append(
            {
                "section": pose_entry["section"],
                "group": group,
                "pose_topic": pose_entry["topic"],
                "gripper_topic": gripper_entry["topic"],
                "pose_fields": pose_entry["fields"],
                "gripper_fields": gripper_entry["fields"],
                "raw_pose_values": pose_values,
                "euler_order": "ZYX",
                "values_by_label": values_by_label,
                "vector_labels": vector_labels,
                "vector": [values_by_label[label] for label in vector_labels],
                "pose_match": {
                    key: value for key, value in pose_message.items() if key != "ros_message"
                },
                "gripper_match": {
                    key: value for key, value in gripper_message.items() if key != "ros_message"
                },
            }
        )

    return {
        "config_file": str(Path(config_path)),
        "mcap_dir": str(mcap_root),
        "target_ns": target_ns,
        "search_window_ns": search_window_ns,
        "used_full_scan_fallback": used_full_scan_fallback,
        "scanned_messages": scanned_messages,
        "vector_labels": vector_labels,
        "vectors": vectors,
    }

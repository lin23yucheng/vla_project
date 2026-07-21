"""按机器人配置从多个 MCAP 文件提取指定时间附近的 ROS2 字段值。"""

import json
import math
from pathlib import Path
from typing import Any

from mcap_ros2.reader import read_ros2_messages


POSE_TOPICS = ("/pika_pose_l", "/pika_pose_r")
CONFIG_SECTIONS = ("action", "observation_state")
# 机器人配置中的 sync_policy="3" 表示 3 ms；本模块统一换算为纳秒。
DEFAULT_MAX_TOLERANCE_NS = 3_000_000


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


def _message_record(decoded: Any, mcap_file: Path, target_ns: int) -> dict[str, Any]:
    publish_ns = int(decoded.publish_time_ns)
    return {
        "mcap_file": str(mcap_file),
        "topic": decoded.channel.topic,
        "schema_name": decoded.schema.name,
        "matched_publish_ns": publish_ns,
        "matched_log_ns": int(decoded.log_time_ns),
        "diff_ns": abs(publish_ns - target_ns),
        "ros_message": decoded.ros_msg,
    }


def _scan_interpolation_neighbors(
    mcap_files: list[Path],
    topics: list[str],
    target_ns: int,
    start_time: int | None,
    end_time: int | None,
) -> tuple[dict[str, dict[str, dict[str, Any] | None]], int]:
    """为每个 topic 收集目标时间最近点及左右两个插值端点。"""
    neighbors = {
        topic: {"nearest": None, "before": None, "after": None}
        for topic in topics
    }
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
            publish_ns = int(decoded.publish_time_ns)
            record = _message_record(decoded, mcap_file, target_ns)
            topic_neighbors = neighbors[topic]

            nearest = topic_neighbors["nearest"]
            if nearest is None or record["diff_ns"] < nearest["diff_ns"]:
                topic_neighbors["nearest"] = record

            before = topic_neighbors["before"]
            if publish_ns <= target_ns and (
                before is None or publish_ns > before["matched_publish_ns"]
            ):
                topic_neighbors["before"] = record

            after = topic_neighbors["after"]
            if publish_ns >= target_ns and (
                after is None or publish_ns < after["matched_publish_ns"]
            ):
                topic_neighbors["after"] = record
    return neighbors, scanned_messages


def _linear_interpolate(start: float, end: float, ratio: float) -> float:
    return start + (end - start) * ratio


def _normalize_quaternion(quaternion: list[float] | tuple[float, ...]) -> tuple[float, float, float, float]:
    if len(quaternion) != 4:
        raise ValueError(f"四元数必须包含 4 个值，实际为 {len(quaternion)}")
    norm = math.sqrt(sum(float(value) ** 2 for value in quaternion))
    if norm == 0:
        raise ValueError("四元数模长为 0，无法进行旋转插值")
    return tuple(float(value) / norm for value in quaternion)  # type: ignore[return-value]


def slerp_quaternion(
    start: list[float] | tuple[float, ...],
    end: list[float] | tuple[float, ...],
    ratio: float,
) -> tuple[float, float, float, float]:
    """按最短旋转路径对 [qx, qy, qz, qw] 做球面线性插值。"""
    start_q = _normalize_quaternion(start)
    end_q = _normalize_quaternion(end)
    dot = sum(left * right for left, right in zip(start_q, end_q))
    if dot < 0.0:
        end_q = tuple(-value for value in end_q)
        dot = -dot
    dot = min(1.0, max(-1.0, dot))

    if dot > 0.9995:
        interpolated = tuple(
            _linear_interpolate(left, right, ratio)
            for left, right in zip(start_q, end_q)
        )
        return _normalize_quaternion(interpolated)

    angle = math.acos(dot)
    sin_angle = math.sin(angle)
    start_weight = math.sin((1.0 - ratio) * angle) / sin_angle
    end_weight = math.sin(ratio * angle) / sin_angle
    return tuple(
        start_weight * left + end_weight * right
        for left, right in zip(start_q, end_q)
    )  # type: ignore[return-value]


def interpolate_pose7d(
    start: list[float] | tuple[float, ...],
    end: list[float] | tuple[float, ...],
    ratio: float,
) -> list[float]:
    """对 [x,y,z,qx,qy,qz,qw] 做位置线性插值和旋转 SLERP。"""
    if len(start) != 7 or len(end) != 7:
        raise ValueError("pose7d 的起止向量都必须包含 7 个值")
    position = [
        _linear_interpolate(float(start[index]), float(end[index]), ratio)
        for index in range(3)
    ]
    quaternion = slerp_quaternion(start[3:7], end[3:7], ratio)
    return [*position, *quaternion]


def _interpolation_ratio(before_ns: int, after_ns: int, target_ns: int) -> float:
    if after_ns <= before_ns:
        raise ValueError(f"插值端点时间无效: before={before_ns}, after={after_ns}")
    return (target_ns - before_ns) / (after_ns - before_ns)


def _resolve_interpolation(
    topic_neighbors: dict[str, dict[str, Any] | None],
    target_ns: int,
    max_tolerance_ns: int,
) -> tuple[str, float | None, dict[str, Any], dict[str, Any] | None]:
    nearest = topic_neighbors["nearest"]
    if nearest is None:
        raise ValueError("未找到可用消息")
    if nearest["diff_ns"] <= max_tolerance_ns:
        return "nearest", None, nearest, None

    before = topic_neighbors["before"]
    after = topic_neighbors["after"]
    if before is None or after is None or before["matched_publish_ns"] == after["matched_publish_ns"]:
        return "nearest_missing_bracket", None, nearest, None
    ratio = _interpolation_ratio(
        before["matched_publish_ns"],
        after["matched_publish_ns"],
        target_ns,
    )
    return "interpolated", ratio, before, after


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


def _load_main_time_topic(config_path: str | Path) -> str:
    with Path(config_path).open("r", encoding="utf-8") as file:
        config_data = json.load(file)
    main_time_topic = config_data.get("main_time_topic")
    if not isinstance(main_time_topic, str) or not main_time_topic:
        raise ValueError("机器人配置缺少有效 main_time_topic")
    return main_time_topic


def _public_message_metadata(message: dict[str, Any] | None) -> dict[str, Any] | None:
    if message is None:
        return None
    return {key: value for key, value in message.items() if key != "ros_message"}


def _build_interpolation_metadata(
    topic_neighbors: dict[str, dict[str, Any] | None],
    strategy: str,
    ratio: float | None,
    first: dict[str, Any],
    second: dict[str, Any] | None,
    target_ns: int,
) -> dict[str, Any]:
    return {
        "strategy": strategy,
        "target_ns": target_ns,
        "ratio": ratio,
        "nearest": _public_message_metadata(topic_neighbors["nearest"]),
        "before": _public_message_metadata(first if strategy == "interpolated" else topic_neighbors["before"]),
        "after": _public_message_metadata(second if strategy == "interpolated" else topic_neighbors["after"]),
    }


def _merge_neighbors(
    current: dict[str, dict[str, dict[str, Any] | None]],
    fallback: dict[str, dict[str, dict[str, Any] | None]],
    topics: list[str],
) -> None:
    for topic in topics:
        for position in ("nearest", "before", "after"):
            candidate = fallback[topic][position]
            if candidate is not None:
                current[topic][position] = candidate


def extract_robot_vectors_at_time(
    config_path: str | Path,
    mcap_dir: str | Path,
    target_ns: int,
    search_window_ns: int = 1_000_000_000,
    max_tolerance_ns: int = DEFAULT_MAX_TOLERANCE_NS,
) -> dict[str, Any]:
    """按主相机帧同步并插值 MCAP 数据，生成四组机器人七维向量。"""
    if search_window_ns < 0:
        raise ValueError("search_window_ns 不能小于 0")
    if max_tolerance_ns < 0:
        raise ValueError("max_tolerance_ns 不能小于 0")
    pose_entries = load_configured_topic_fields(config_path)
    gripper_entries = _load_observation_gripper_entries(config_path)
    main_time_topic = _load_main_time_topic(config_path)
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

    main_nearest, scanned_messages = _scan_nearest_messages(
        mcap_files,
        [main_time_topic],
        target_ns,
        target_ns - search_window_ns,
        target_ns + search_window_ns + 1,
    )
    used_full_scan_fallback = main_time_topic not in main_nearest
    if main_time_topic not in main_nearest:
        fallback_main, fallback_count = _scan_nearest_messages(
            mcap_files, [main_time_topic], target_ns, None, None
        )
        main_nearest.update(fallback_main)
        scanned_messages += fallback_count
    if main_time_topic not in main_nearest:
        raise ValueError(f"多个 mcap 文件中未找到主相机 topic: {main_time_topic}")

    main_message = main_nearest[main_time_topic]
    main_frame_ns = int(main_message["matched_publish_ns"])
    neighbors, sensor_scan_count = _scan_interpolation_neighbors(
        mcap_files,
        topics,
        main_frame_ns,
        main_frame_ns - search_window_ns,
        main_frame_ns + search_window_ns + 1,
    )
    scanned_messages += sensor_scan_count
    incomplete_topics = []
    for topic in topics:
        topic_neighbors = neighbors[topic]
        nearest = topic_neighbors["nearest"]
        needs_bracket = nearest is not None and nearest["diff_ns"] > max_tolerance_ns
        if nearest is None or (needs_bracket and (
            topic_neighbors["before"] is None or topic_neighbors["after"] is None
        )):
            incomplete_topics.append(topic)
    if incomplete_topics:
        fallback_neighbors, fallback_count = _scan_interpolation_neighbors(
            mcap_files, incomplete_topics, main_frame_ns, None, None
        )
        _merge_neighbors(neighbors, fallback_neighbors, incomplete_topics)
        scanned_messages += fallback_count
        used_full_scan_fallback = True

    missing_topics = [topic for topic in topics if neighbors[topic]["nearest"] is None]
    if missing_topics:
        raise ValueError(f"多个 mcap 文件中未找到 topic: {missing_topics}")

    vector_labels = ["x", "y", "z", "roll", "pitch", "yaw", "angle"]
    vectors = []
    for pose_entry in pose_entries:
        group = pose_entry["group"]
        if group not in gripper_entries:
            raise ValueError(f"位姿配置缺少有效 group: {pose_entry}")
        pose_neighbors = neighbors[pose_entry["topic"]]
        pose_strategy, pose_ratio, pose_first, pose_second = _resolve_interpolation(
            pose_neighbors, main_frame_ns, max_tolerance_ns
        )
        first_pose = [
            float(get_nested_field_value(pose_first["ros_message"], field))
            for field in pose_entry["fields"]
        ]
        pose7d = first_pose
        if pose_strategy == "interpolated" and pose_second is not None and pose_ratio is not None:
            second_pose = [
                float(get_nested_field_value(pose_second["ros_message"], field))
                for field in pose_entry["fields"]
            ]
            pose7d = interpolate_pose7d(first_pose, second_pose, pose_ratio)
        pose_values = dict(zip(pose_entry["fields"], pose7d))
        euler = quaternion_to_euler_zyx(
            pose_values["pose.orientation.x"],
            pose_values["pose.orientation.y"],
            pose_values["pose.orientation.z"],
            pose_values["pose.orientation.w"],
        )

        gripper_entry = gripper_entries[group]
        gripper_neighbors = neighbors[gripper_entry["topic"]]
        gripper_strategy, gripper_ratio, gripper_first, gripper_second = _resolve_interpolation(
            gripper_neighbors, main_frame_ns, max_tolerance_ns
        )
        angle = float(get_nested_field_value(gripper_first["ros_message"], "angle"))
        if (
            gripper_strategy == "interpolated"
            and gripper_second is not None
            and gripper_ratio is not None
        ):
            second_angle = float(get_nested_field_value(gripper_second["ros_message"], "angle"))
            angle = _linear_interpolate(angle, second_angle, gripper_ratio)
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
                "pose_match": _build_interpolation_metadata(
                    pose_neighbors,
                    pose_strategy,
                    pose_ratio,
                    pose_first,
                    pose_second,
                    main_frame_ns,
                ),
                "gripper_match": _build_interpolation_metadata(
                    gripper_neighbors,
                    gripper_strategy,
                    gripper_ratio,
                    gripper_first,
                    gripper_second,
                    main_frame_ns,
                ),
            }
        )

    return {
        "config_file": str(Path(config_path)),
        "mcap_dir": str(mcap_root),
        "target_ns": target_ns,
        "main_time_topic": main_time_topic,
        "main_frame_ns": main_frame_ns,
        "main_frame_match": _public_message_metadata(main_message),
        "search_window_ns": search_window_ns,
        "max_tolerance_ns": max_tolerance_ns,
        "used_full_scan_fallback": used_full_scan_fallback,
        "scanned_messages": scanned_messages,
        "vector_labels": vector_labels,
        "vectors": vectors,
    }

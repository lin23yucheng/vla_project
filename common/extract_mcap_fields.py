"""按机器人配置从多个 MCAP 文件提取指定时间附近的 ROS2 字段值。"""

import json
import math
import re
from pathlib import Path
from typing import Any, Sequence

from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

from common.mcap_source import (
    DEFAULT_INITIAL_SEARCH_WINDOW_NS,
    adaptive_search_windows,
    local_mcap_sources,
    mcap_source_name,
    open_mcap_source,
    read_mcap_window_messages,
    select_mcap_sources_for_window,
)


POSE_TOPICS = ("/pika_pose_l", "/pika_pose_r")
CONFIG_SECTIONS = ("action", "observation_state")
# 机器人配置中的 sync_policy="3" 表示 3 ms；本模块统一换算为纳秒。
DEFAULT_MAX_TOLERANCE_NS = 3_000_000


def _sync_policy_to_tolerance_ns(value: Any) -> int:
    """将 machine_config 的毫秒同步策略转换为纳秒。"""
    normalized = "3" if str(value).strip().lower() == "exact_frame" else str(value).strip()
    if not normalized:
        return DEFAULT_MAX_TOLERANCE_NS
    try:
        tolerance_ns = round(float(normalized) * 1_000_000)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"无法识别 sync_policy={value!r}") from exc
    if tolerance_ns < 0:
        raise ValueError(f"sync_policy 不能小于 0: {value!r}")
    return tolerance_ns


def load_configured_topic_fields(
    config_path: str | Path,
    sections: tuple[str, ...] = CONFIG_SECTIONS,
    topics: tuple[str, ...] | None = POSE_TOPICS,
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

        section_topics = topics
        if section_topics is None:
            section_topics = tuple(
                item.get("topic")
                for item in section_items
                if isinstance(item, dict)
                and item.get("topic")
                and (
                    item.get("parser") == "pose7d"
                    or len(item.get("fields", [])) == 7
                )
            )
        for topic in section_topics:
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
                    "sync_policy": item.get("sync_policy"),
                    "sync_tolerance_ns": _sync_policy_to_tolerance_ns(
                        item.get("sync_policy", "3")
                    ),
                }
            )
    return entries


def get_nested_field_value(message: Any, field_path: str) -> Any:
    """从动态 ROS2 消息对象中按点分路径读取字段，支持数组下标如 ``data[0]``。"""
    current = message
    for part in field_path.split("."):
        match = re.fullmatch(r"([^\[\]]+)(?:\[(\d+)\])?", part)
        if match is None:
            raise ValueError(f"字段路径格式无效: {field_path}")
        field_name, index_text = match.groups()
        if isinstance(current, dict):
            if field_name not in current:
                raise KeyError(f"字段路径 {field_path} 缺少节点 {field_name}")
            current = current[field_name]
        else:
            if not hasattr(current, field_name):
                raise AttributeError(f"字段路径 {field_path} 缺少属性 {field_name}")
            current = getattr(current, field_name)
        if index_text is not None:
            try:
                current = current[int(index_text)]
            except (IndexError, KeyError, TypeError) as exc:
                raise IndexError(f"字段路径 {field_path} 的下标 {index_text} 无效") from exc
    return current


def _scan_nearest_messages(
    mcap_files: Sequence[Any],
    topics: list[str],
    target_ns: int,
    start_time: int | None,
    end_time: int | None,
    require_chunk_indexes: bool = False,
) -> tuple[dict[str, dict[str, Any]], int]:
    nearest: dict[str, dict[str, Any]] = {}
    scanned_messages = 0
    selected_sources = select_mcap_sources_for_window(mcap_files, start_time, end_time)
    for mcap_file in selected_sources:
        with open_mcap_source(mcap_file) as stream:
            reader = make_reader(stream, decoder_factories=[DecoderFactory()])
            if require_chunk_indexes:
                summary = reader.get_summary()
                if summary is None or not summary.chunk_indexes:
                    raise ValueError(
                        f"远端 MCAP 缺少 chunk index，拒绝顺序读取完整对象: "
                        f"{mcap_source_name(mcap_file)}"
                    )
            for schema, channel, message, ros_message in reader.iter_decoded_messages(
                topics=topics,
                start_time=start_time,
                end_time=end_time,
            ):
                if schema is None:
                    raise ValueError(f"MCAP topic={channel.topic} 缺少 ROS2 schema: {mcap_source_name(mcap_file)}")
                scanned_messages += 1
                topic = channel.topic
                log_ns = int(message.log_time)
                diff_ns = abs(log_ns - target_ns)
                current = nearest.get(topic)
                if current is None or diff_ns < current["diff_ns"]:
                    nearest[topic] = {
                        "mcap_file": mcap_source_name(mcap_file),
                        "topic": topic,
                        "schema_name": schema.name,
                        "matched_publish_ns": int(message.publish_time),
                        "matched_log_ns": log_ns,
                        "diff_ns": diff_ns,
                        "ros_message": ros_message,
                    }
    return nearest, scanned_messages


def _message_record(
    schema: Any,
    channel: Any,
    message: Any,
    ros_message: Any,
    mcap_file: Any,
    target_ns: int,
) -> dict[str, Any]:
    log_ns = int(message.log_time)
    return {
        "mcap_file": mcap_source_name(mcap_file),
        "topic": channel.topic,
        "schema_name": schema.name,
        "matched_publish_ns": int(message.publish_time),
        "matched_log_ns": log_ns,
        "diff_ns": abs(log_ns - target_ns),
        "ros_message": ros_message,
    }


def _scan_main_frame_and_sensor_messages(
    mcap_files: Sequence[Any],
    main_time_topic: str,
    sensor_topics: list[str],
    target_ns: int,
    start_time: int,
    end_time: int,
    require_chunk_indexes: bool = False,
) -> tuple[dict[str, Any] | None, dict[str, list[dict[str, Any]]], int, bool]:
    """单次读取中定位主帧并收集传感器消息，避免解码主相机图像。"""
    main_nearest: dict[str, Any] | None = None
    sensor_records = {topic: [] for topic in sensor_topics}
    scanned_messages = 0
    scan_topics = list(dict.fromkeys([main_time_topic, *sensor_topics]))
    window_records, cache_hit = read_mcap_window_messages(
        mcap_files,
        scan_topics,
        start_time,
        end_time,
        require_chunk_indexes=require_chunk_indexes,
    )
    decoder_factories: dict[str, DecoderFactory] = {}
    for window_record in window_records:
        scanned_messages += 1
        schema = window_record["schema"]
        channel = window_record["channel"]
        message = window_record["message"]
        mcap_file = window_record["mcap_file"]
        if schema is None:
            raise ValueError(f"MCAP topic={channel.topic} 缺少 ROS2 schema: {mcap_file}")

        log_ns = int(message.log_time)
        topic = channel.topic
        if topic == main_time_topic:
            candidate = {
                "mcap_file": mcap_file,
                "topic": topic,
                "schema_name": schema.name,
                "matched_publish_ns": int(message.publish_time),
                "matched_log_ns": log_ns,
                "diff_ns": abs(log_ns - target_ns),
            }
            if main_nearest is None or candidate["diff_ns"] < main_nearest["diff_ns"]:
                main_nearest = candidate

        if topic not in sensor_records:
            continue
        decoder_factory = decoder_factories.setdefault(mcap_file, DecoderFactory())
        decoder = decoder_factory.decoder_for(channel.message_encoding, schema)
        if decoder is None:
            raise ValueError(f"MCAP topic={topic} 无法按 ROS2 CDR 解码: {mcap_file}")
        sensor_records[topic].append(
            {
                "mcap_file": mcap_file,
                "topic": topic,
                "schema_name": schema.name,
                "matched_publish_ns": int(message.publish_time),
                "matched_log_ns": log_ns,
                "ros_message": decoder(message.data),
            }
        )

    return main_nearest, sensor_records, scanned_messages, cache_hit


def _build_interpolation_neighbors(
    records_by_topic: dict[str, list[dict[str, Any]]],
    topics: list[str],
    target_ns: int,
) -> dict[str, dict[str, dict[str, Any] | None]]:
    """从已读取的传感器记录构建最近值和插值两侧端点。"""
    neighbors = {
        topic: {"nearest": None, "before": None, "after": None}
        for topic in topics
    }
    for topic in topics:
        for raw_record in records_by_topic.get(topic, []):
            log_ns = int(raw_record["matched_log_ns"])
            record = {**raw_record, "diff_ns": abs(log_ns - target_ns)}
            topic_neighbors = neighbors[topic]

            nearest = topic_neighbors["nearest"]
            if nearest is None or record["diff_ns"] < nearest["diff_ns"]:
                topic_neighbors["nearest"] = record

            before = topic_neighbors["before"]
            if log_ns <= target_ns and (
                before is None or log_ns > before["matched_log_ns"]
            ):
                topic_neighbors["before"] = record

            after = topic_neighbors["after"]
            if log_ns >= target_ns and (
                after is None or log_ns < after["matched_log_ns"]
            ):
                topic_neighbors["after"] = record
    return neighbors


def _incomplete_interpolation_topics(
    neighbors: dict[str, dict[str, dict[str, Any] | None]],
    topics: list[str],
    max_tolerance_ns: int | dict[str, int],
) -> list[str]:
    incomplete_topics = []
    for topic in topics:
        topic_neighbors = neighbors[topic]
        nearest = topic_neighbors["nearest"]
        tolerance_ns = (
            max_tolerance_ns.get(topic, DEFAULT_MAX_TOLERANCE_NS)
            if isinstance(max_tolerance_ns, dict)
            else max_tolerance_ns
        )
        needs_bracket = nearest is not None and nearest["diff_ns"] > tolerance_ns
        if nearest is None or (needs_bracket and (
            topic_neighbors["before"] is None or topic_neighbors["after"] is None
        )):
            incomplete_topics.append(topic)
    return incomplete_topics


def _scan_interpolation_neighbors(
    mcap_files: Sequence[Any],
    topics: list[str],
    target_ns: int,
    start_time: int | None,
    end_time: int | None,
    require_chunk_indexes: bool = False,
) -> tuple[dict[str, dict[str, dict[str, Any] | None]], int]:
    """为每个 topic 收集目标时间最近点及左右两个插值端点。"""
    neighbors = {
        topic: {"nearest": None, "before": None, "after": None}
        for topic in topics
    }
    scanned_messages = 0
    selected_sources = select_mcap_sources_for_window(mcap_files, start_time, end_time)
    for mcap_file in selected_sources:
        with open_mcap_source(mcap_file) as stream:
            reader = make_reader(stream, decoder_factories=[DecoderFactory()])
            if require_chunk_indexes:
                summary = reader.get_summary()
                if summary is None or not summary.chunk_indexes:
                    raise ValueError(
                        f"远端 MCAP 缺少 chunk index，拒绝顺序读取完整对象: "
                        f"{mcap_source_name(mcap_file)}"
                    )
            for schema, channel, message, ros_message in reader.iter_decoded_messages(
                topics=topics,
                start_time=start_time,
                end_time=end_time,
            ):
                if schema is None:
                    raise ValueError(f"MCAP topic={channel.topic} 缺少 ROS2 schema: {mcap_source_name(mcap_file)}")
                scanned_messages += 1
                topic = channel.topic
                record = _message_record(
                    schema,
                    channel,
                    message,
                    ros_message,
                    mcap_file,
                    target_ns,
                )
                topic_neighbors = neighbors[topic]

                nearest = topic_neighbors["nearest"]
                if nearest is None or record["diff_ns"] < nearest["diff_ns"]:
                    topic_neighbors["nearest"] = record

                before = topic_neighbors["before"]
                log_ns = int(message.log_time)
                if log_ns <= target_ns and (
                    before is None or log_ns > before["matched_log_ns"]
                ):
                    topic_neighbors["before"] = record

                after = topic_neighbors["after"]
                if log_ns >= target_ns and (
                    after is None or log_ns < after["matched_log_ns"]
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
    """位置线性插值，姿态按 [qx, qy, qz, qw] 做最短路径 SLERP。"""
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
    if before is None and after is not None:
        return "clamp_first", None, after, None
    if after is None and before is not None:
        return "clamp_last", None, before, None
    if before is None or after is None:
        return "single_value", None, nearest, None
    if before["matched_log_ns"] == after["matched_log_ns"]:
        return "single_value", None, nearest, None
    ratio = _interpolation_ratio(
        before["matched_log_ns"],
        after["matched_log_ns"],
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


def _load_gripper_entries(config_path: str | Path) -> dict[tuple[str, str], dict[str, Any]]:
    with Path(config_path).open("r", encoding="utf-8") as file:
        config_data = json.load(file)
    entries: dict[tuple[str, str], dict[str, Any]] = {}
    for section in CONFIG_SECTIONS:
        for item in config_data.get(section, []):
            if not isinstance(item, dict):
                continue
            group = item.get("group")
            topic = item.get("topic")
            fields = item.get("fields")
            is_gripper = item.get("parser") == "gripper" or fields == ["angle"]
            if (
                is_gripper
                and group in {"left", "right"}
                and isinstance(topic, str)
                and isinstance(fields, list)
                and len(fields) == 1
                and isinstance(fields[0], str)
            ):
                entries[(section, group)] = {
                    "section": section,
                    "name": item.get("name"),
                    "group": group,
                    "topic": topic,
                    "fields": fields,
                    "sync_policy": item.get("sync_policy"),
                    "sync_tolerance_ns": _sync_policy_to_tolerance_ns(
                        item.get("sync_policy", "3")
                    ),
                }
    expected_keys = [
        (section, group)
        for section in CONFIG_SECTIONS
        for group in ("left", "right")
    ]
    missing_keys = [key for key in expected_keys if key not in entries]
    if missing_keys:
        raise ValueError(f"机器人配置缺少有效夹爪配置: {missing_keys}")
    return entries


def _load_main_time_topic(config_path: str | Path) -> str:
    with Path(config_path).open("r", encoding="utf-8") as file:
        config_data = json.load(file)
    main_time_topic = config_data.get("main_time_topic")
    if not isinstance(main_time_topic, str) or not main_time_topic:
        raise ValueError("机器人配置缺少有效 main_time_topic")
    return main_time_topic


def load_robot_vector_topics(config_path: str | Path) -> list[str]:
    """返回主时间、位姿和夹爪 topic，供相同窗口的其他提取流程预取。"""
    pose_entries = load_configured_topic_fields(config_path, topics=None)
    gripper_entries = _load_gripper_entries(config_path)
    main_time_topic = _load_main_time_topic(config_path)
    return list(dict.fromkeys(
        [main_time_topic]
        + [entry["topic"] for entry in pose_entries]
        + [entry["topic"] for entry in gripper_entries.values()]
    ))


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
        candidate_nearest = fallback[topic]["nearest"]
        current_nearest = current[topic]["nearest"]
        if candidate_nearest is not None and (
            current_nearest is None
            or candidate_nearest["diff_ns"] < current_nearest["diff_ns"]
        ):
            current[topic]["nearest"] = candidate_nearest

        candidate_before = fallback[topic]["before"]
        current_before = current[topic]["before"]
        if candidate_before is not None and (
            current_before is None
            or candidate_before["matched_log_ns"] > current_before["matched_log_ns"]
        ):
            current[topic]["before"] = candidate_before

        candidate_after = fallback[topic]["after"]
        current_after = current[topic]["after"]
        if candidate_after is not None and (
            current_after is None
            or candidate_after["matched_log_ns"] < current_after["matched_log_ns"]
        ):
            current[topic]["after"] = candidate_after


def extract_robot_vectors_at_time(
    config_path: str | Path,
    mcap_dir: str | Path | None,
    target_ns: int,
    search_window_ns: int = 1_000_000_000,
    max_tolerance_ns: int | None = None,
    *,
    initial_search_window_ns: int = DEFAULT_INITIAL_SEARCH_WINDOW_NS,
    align_to_main_camera: bool = True,
    mcap_sources: Sequence[Any] | None = None,
    mcap_source_label: str | None = None,
    require_chunk_indexes: bool = False,
    allow_full_scan_fallback: bool = True,
    episode_start_ns: int | None = None,
    episode_end_ns: int | None = None,
) -> dict[str, Any]:
    """按主相机帧或指定时间轴同步并插值 MCAP 数据，生成四组机器人七维向量。"""
    if search_window_ns < 0:
        raise ValueError("search_window_ns 不能小于 0")
    if max_tolerance_ns is not None and max_tolerance_ns < 0:
        raise ValueError("max_tolerance_ns 不能小于 0")
    if (episode_start_ns is None) != (episode_end_ns is None):
        raise ValueError("episode_start_ns 和 episode_end_ns 必须同时提供")
    if episode_start_ns is not None and episode_end_ns is not None:
        episode_start_ns = int(episode_start_ns)
        episode_end_ns = int(episode_end_ns)
        if episode_start_ns >= episode_end_ns:
            raise ValueError("episode_start_ns 必须早于 episode_end_ns")
    if require_chunk_indexes and allow_full_scan_fallback:
        raise ValueError(
            "远端 indexed MCAP 模式禁止 allow_full_scan_fallback，避免读取完整对象"
        )
    # Pose topics are configuration-specific.  The legacy default remains
    # available for callers that explicitly request it, while robot-vector
    # extraction follows the pose7d entries in the supplied JSON.
    pose_entries = load_configured_topic_fields(config_path, topics=None)
    gripper_entries = _load_gripper_entries(config_path)
    main_time_topic = _load_main_time_topic(config_path)
    if mcap_sources is None:
        if mcap_dir is None:
            raise ValueError("mcap_dir 和 mcap_sources 不能同时为空")
        mcap_files, resolved_source_label = local_mcap_sources(mcap_dir)
    else:
        mcap_files = list(mcap_sources)
        if not mcap_files:
            raise ValueError("mcap_sources 不能为空")
        resolved_source_label = mcap_source_label or "remote-mcap-sources"

    topics = list(dict.fromkeys(
        [entry["topic"] for entry in pose_entries]
        + [entry["topic"] for entry in gripper_entries.values()]
    ))
    configured_tolerances: dict[str, int] = {}
    for entry in [*pose_entries, *gripper_entries.values()]:
        topic = entry["topic"]
        tolerance_ns = int(entry.get("sync_tolerance_ns", DEFAULT_MAX_TOLERANCE_NS))
        existing = configured_tolerances.get(topic)
        if existing is not None and existing != tolerance_ns:
            raise ValueError(f"同一 topic 的 sync_policy 不一致: topic={topic}")
        configured_tolerances[topic] = tolerance_ns
    topic_tolerances = (
        {topic: int(max_tolerance_ns) for topic in topics}
        if max_tolerance_ns is not None
        else configured_tolerances
    )
    target_ns = int(target_ns)

    def bounded_window(window_ns: int) -> tuple[int, int]:
        start_ns = target_ns - window_ns
        end_ns = target_ns + window_ns + 1
        if episode_start_ns is not None and episode_end_ns is not None:
            start_ns = max(start_ns, episode_start_ns)
            end_ns = min(end_ns, episode_end_ns + 1)
        return start_ns, end_ns

    scanned_messages = 0
    window_cache_hits = 0
    used_full_scan_fallback = False
    used_sensor_window_extension = False
    search_attempt_windows_ns: list[int] = []
    effective_search_window_ns: int | None = None
    main_message: dict[str, Any] | None = None
    neighbors: dict[str, dict[str, dict[str, Any] | None]] | None = None
    incomplete_topics: list[str] = []

    for window_ns in adaptive_search_windows(search_window_ns, initial_search_window_ns):
        search_attempt_windows_ns.append(window_ns)
        effective_search_window_ns = window_ns
        scan_start_ns, scan_end_ns = bounded_window(window_ns)
        if scan_start_ns >= scan_end_ns:
            continue
        if align_to_main_camera:
            candidate_main, sensor_records, scan_count, cache_hit = _scan_main_frame_and_sensor_messages(
                mcap_files,
                main_time_topic,
                topics,
                target_ns,
                scan_start_ns,
                scan_end_ns,
                require_chunk_indexes=require_chunk_indexes,
            )
            scanned_messages += scan_count
            window_cache_hits += int(cache_hit)
            if candidate_main is None:
                continue
            candidate_neighbors = _build_interpolation_neighbors(
                sensor_records,
                topics,
                int(candidate_main["matched_log_ns"]),
            )
        else:
            candidate_main = None
            candidate_neighbors, scan_count = _scan_interpolation_neighbors(
                mcap_files,
                topics,
                target_ns,
                scan_start_ns,
                scan_end_ns,
                require_chunk_indexes=require_chunk_indexes,
            )
            scanned_messages += scan_count
        candidate_incomplete_topics = _incomplete_interpolation_topics(
            candidate_neighbors,
            topics,
            topic_tolerances,
        )
        if episode_start_ns is not None and episode_end_ns is not None:
            candidate_incomplete_topics = [
                topic
                for topic in candidate_incomplete_topics
                if not (
                    candidate_neighbors[topic]["nearest"] is not None
                    and (
                        (
                            candidate_neighbors[topic]["before"] is None
                            and candidate_neighbors[topic]["after"] is not None
                            and scan_start_ns == episode_start_ns
                        )
                        or (
                            candidate_neighbors[topic]["after"] is None
                            and candidate_neighbors[topic]["before"] is not None
                            and scan_end_ns == episode_end_ns + 1
                        )
                    )
                )
            ]
        main_message = candidate_main
        neighbors = candidate_neighbors
        incomplete_topics = candidate_incomplete_topics
        if not incomplete_topics:
            break

    if align_to_main_camera and main_message is None and allow_full_scan_fallback:
        fallback_main, fallback_count = _scan_nearest_messages(
            mcap_files,
            [main_time_topic],
            target_ns,
            episode_start_ns,
            episode_end_ns + 1 if episode_end_ns is not None else None,
            require_chunk_indexes=require_chunk_indexes,
        )
        scanned_messages += fallback_count
        used_full_scan_fallback = True
        main_message = fallback_main.get(main_time_topic)
    if align_to_main_camera and main_message is None:
        raise ValueError(f"多个 mcap 文件中未找到主相机 topic: {main_time_topic}")

    sampling_reference_ns = (
        int(main_message["matched_log_ns"])
        if main_message is not None
        else target_ns
    )
    if neighbors is None:
        fallback_start_ns = (
            episode_start_ns
            if episode_start_ns is not None
            else sampling_reference_ns - search_window_ns
        )
        fallback_end_ns = (
            episode_end_ns + 1
            if episode_end_ns is not None
            else sampling_reference_ns + search_window_ns + 1
        )
        neighbors, sensor_scan_count = _scan_interpolation_neighbors(
            mcap_files,
            topics,
            sampling_reference_ns,
            fallback_start_ns,
            fallback_end_ns,
            require_chunk_indexes=require_chunk_indexes,
        )
        scanned_messages += sensor_scan_count
        incomplete_topics = _incomplete_interpolation_topics(
            neighbors,
            topics,
            topic_tolerances,
        )
    elif incomplete_topics:
        extension_start_ns = (
            episode_start_ns
            if episode_start_ns is not None
            else sampling_reference_ns - search_window_ns
        )
        extension_end_ns = (
            episode_end_ns + 1
            if episode_end_ns is not None
            else sampling_reference_ns + search_window_ns + 1
        )
        extension_neighbors, extension_count = _scan_interpolation_neighbors(
            mcap_files,
            incomplete_topics,
            sampling_reference_ns,
            extension_start_ns,
            extension_end_ns,
            require_chunk_indexes=require_chunk_indexes,
        )
        _merge_neighbors(neighbors, extension_neighbors, incomplete_topics)
        scanned_messages += extension_count
        used_sensor_window_extension = True
        incomplete_topics = _incomplete_interpolation_topics(
            neighbors,
            topics,
            topic_tolerances,
        )

    if incomplete_topics and allow_full_scan_fallback:
        fallback_start_ns = episode_start_ns
        fallback_end_ns = episode_end_ns + 1 if episode_end_ns is not None else None
        fallback_neighbors, fallback_count = _scan_interpolation_neighbors(
            mcap_files,
            incomplete_topics,
            sampling_reference_ns,
            fallback_start_ns,
            fallback_end_ns,
            require_chunk_indexes=require_chunk_indexes,
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
        section = pose_entry["section"]
        group = pose_entry["group"]
        gripper_key = (section, group)
        if gripper_key not in gripper_entries:
            raise ValueError(f"位姿配置缺少对应夹爪配置: {pose_entry}")
        pose_neighbors = neighbors[pose_entry["topic"]]
        pose_strategy, pose_ratio, pose_first, pose_second = _resolve_interpolation(
            pose_neighbors,
            sampling_reference_ns,
            topic_tolerances[pose_entry["topic"]],
        )
        first_pose = [
            float(get_nested_field_value(pose_first["ros_message"], field))
            for field in pose_entry["fields"]
        ]
        pose7d = first_pose
        first_pose_values = dict(zip(pose_entry["fields"], first_pose))
        euler = quaternion_to_euler_zyx(
            first_pose_values["pose.orientation.x"],
            first_pose_values["pose.orientation.y"],
            first_pose_values["pose.orientation.z"],
            first_pose_values["pose.orientation.w"],
        )
        if pose_strategy == "interpolated" and pose_second is not None and pose_ratio is not None:
            second_pose = [
                float(get_nested_field_value(pose_second["ros_message"], field))
                for field in pose_entry["fields"]
            ]
            pose7d = interpolate_pose7d(first_pose, second_pose, pose_ratio)
            euler = quaternion_to_euler_zyx(*pose7d[3:7])
        pose_values = dict(zip(pose_entry["fields"], pose7d))

        gripper_entry = gripper_entries[gripper_key]
        gripper_neighbors = neighbors[gripper_entry["topic"]]
        gripper_strategy, gripper_ratio, gripper_first, gripper_second = _resolve_interpolation(
            gripper_neighbors,
            sampling_reference_ns,
            topic_tolerances[gripper_entry["topic"]],
        )
        gripper_field = gripper_entry["fields"][0]
        angle = float(get_nested_field_value(gripper_first["ros_message"], gripper_field))
        if (
            gripper_strategy == "interpolated"
            and gripper_second is not None
            and gripper_ratio is not None
        ):
            second_angle = float(get_nested_field_value(gripper_second["ros_message"], gripper_field))
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
                    sampling_reference_ns,
                ),
                "gripper_match": _build_interpolation_metadata(
                    gripper_neighbors,
                    gripper_strategy,
                    gripper_ratio,
                    gripper_first,
                    gripper_second,
                    sampling_reference_ns,
                ),
            }
        )

    return {
        "config_file": str(Path(config_path)),
        "mcap_dir": resolved_source_label,
        "mcap_source_count": len(mcap_files),
        "target_ns": target_ns,
        "main_time_topic": main_time_topic,
        "align_to_main_camera": align_to_main_camera,
        "sampling_reference_ns": sampling_reference_ns,
        "main_frame_ns": (
            int(main_message["matched_log_ns"])
            if main_message is not None
            else None
        ),
        "main_frame_match": _public_message_metadata(main_message),
        "search_window_ns": search_window_ns,
        "initial_search_window_ns": initial_search_window_ns,
        "effective_search_window_ns": effective_search_window_ns,
        "search_attempt_windows_ns": search_attempt_windows_ns,
        "max_tolerance_ns": max_tolerance_ns,
        "topic_tolerances_ns": topic_tolerances,
        "time_field": "log_time",
        "episode_start_ns": episode_start_ns,
        "episode_end_ns": episode_end_ns,
        "used_full_scan_fallback": used_full_scan_fallback,
        "used_sensor_window_extension": used_sensor_window_extension,
        "window_cache_hits": window_cache_hits,
        "scanned_messages": scanned_messages,
        "vector_labels": vector_labels,
        "vectors": vectors,
    }

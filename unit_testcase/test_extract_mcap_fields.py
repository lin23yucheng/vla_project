"""
MCAP 字段提取功能单元测试。

测试范围：
- 从 JSON 配置文件加载 topic 与字段映射
- 通过点号路径从嵌套对象中获取属性值
- 四元数转欧拉角（ZYX 顺序）
"""

import json
import math
from pathlib import Path

import pytest

from common.extract_mcap_fields import (
    DEFAULT_MAX_TOLERANCE_NS,
    _build_interpolation_neighbors,
    _incomplete_interpolation_topics,
    _load_gripper_entries,
    _resolve_interpolation,
    _sync_policy_to_tolerance_ns,
    extract_robot_vectors_at_time,
    get_nested_field_value,
    interpolate_pose7d,
    load_configured_topic_fields,
    quaternion_to_euler_zyx,
    slerp_quaternion,
)
from common.mcap_source import adaptive_search_windows


class Position:
    """模拟 ROS 消息中的位置对象。"""
    x = 1.25


class Pose:
    """模拟 ROS 消息中的位姿对象。"""
    position = Position()


class Message:
    """模拟 ROS 消息对象。"""
    pose = Pose()


def test_load_configured_topic_fields(tmp_path: Path):
    """验证从 JSON 配置中正确加载所有 topic 与字段定义。"""
    config_path = tmp_path / "robot.json"
    config_path.write_text(
        json.dumps(
            {
                "action": [
                    {"name": "left_action", "topic": "/pika_pose_l", "fields": ["pose.position.x"]},
                    {"name": "right_action", "topic": "/pika_pose_r", "fields": ["pose.position.y"]},
                ],
                "observation_state": [
                    {"name": "left_state", "topic": "/pika_pose_l", "fields": ["pose.position.z"]},
                    {"name": "right_state", "topic": "/pika_pose_r", "fields": ["pose.orientation.w"]},
                ],
            }
        ),
        encoding="utf-8",
    )

    result = load_configured_topic_fields(config_path)

    assert [(item["section"], item["topic"]) for item in result] == [
        ("action", "/pika_pose_l"),
        ("action", "/pika_pose_r"),
        ("observation_state", "/pika_pose_l"),
        ("observation_state", "/pika_pose_r"),
    ]


def test_load_configured_topic_fields_discovers_pose7d_topics(tmp_path: Path):
    """未指定 topic 时，应从每个 section 的 pose7d 条目发现左右臂配置。"""
    pose_fields = [
        "pose.position.x",
        "pose.position.y",
        "pose.position.z",
        "pose.orientation.x",
        "pose.orientation.y",
        "pose.orientation.z",
        "pose.orientation.w",
    ]
    section_items = [
        {
            "name": "left_arm",
            "group": "left",
            "topic": "/jbt_arm_L/current_arm_tcp_pose",
            "fields": pose_fields,
            "parser": "pose7d",
        },
        {
            "name": "left_gripper",
            "group": "left",
            "topic": "/gripper/gripper_l/data",
            "fields": ["angle"],
            "parser": "gripper",
        },
        {
            "name": "right_arm",
            "group": "right",
            "topic": "/jbt_arm_R/current_arm_tcp_pose",
            "fields": pose_fields,
            "parser": "pose7d",
        },
    ]
    config_path = tmp_path / "robot.json"
    config_path.write_text(
        json.dumps({"action": section_items, "observation_state": section_items}),
        encoding="utf-8",
    )

    result = load_configured_topic_fields(config_path, topics=None)

    assert [(item["section"], item["topic"]) for item in result] == [
        ("action", "/jbt_arm_L/current_arm_tcp_pose"),
        ("action", "/jbt_arm_R/current_arm_tcp_pose"),
        ("observation_state", "/jbt_arm_L/current_arm_tcp_pose"),
        ("observation_state", "/jbt_arm_R/current_arm_tcp_pose"),
    ]


def test_get_nested_field_value():
    """验证可通过点号路径从嵌套对象中正确读取属性值。"""
    assert get_nested_field_value(Message(), "pose.position.x") == 1.25


def test_get_nested_field_value_supports_array_index():
    """天机夹爪反馈的数值位于 ROS 数组字段 data[0]。"""
    assert get_nested_field_value({"data": [1.25]}, "data[0]") == 1.25


def test_load_tianji_gripper_entries_uses_data_index():
    config_path = Path(__file__).parent.parent / "Robot_Configuration" / "天机构型.json"

    entries = _load_gripper_entries(config_path)

    assert len(entries) == 4
    assert {entry["topic"] for entry in entries.values()} == {
        "/info/gripper_feedback_L", "/info/gripper_feedback_R",
    }
    assert all(entry["fields"] == ["data[0]"] for entry in entries.values())


def test_get_nested_field_value_rejects_missing_path():
    """当点号路径中存在不存在的属性时，应抛出 AttributeError。"""
    with pytest.raises(AttributeError, match="missing"):
        get_nested_field_value(Message(), "pose.position.missing")


def test_quaternion_to_euler_zyx_identity():
    """验证单位四元数 (0, 0, 0, 1) 转换为欧拉角后均为 0。"""
    assert quaternion_to_euler_zyx(0.0, 0.0, 0.0, 1.0) == {
        "yaw": 0.0,
        "pitch": 0.0,
        "roll": 0.0,
    }


def test_default_tolerance_is_three_milliseconds_in_nanoseconds():
    """配置中的 3 ms 在纳秒时间轴上应换算为 3,000,000 ns。"""
    assert DEFAULT_MAX_TOLERANCE_NS == 3_000_000


def test_interpolate_pose7d_uses_linear_position_and_slerp():
    """位置应线性插值，旋转应沿最短路径做球面线性插值。"""
    result = interpolate_pose7d(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [10.0, 20.0, 30.0, 0.0, 0.0, 1.0, 0.0],
        0.5,
    )

    assert result[:3] == pytest.approx([5.0, 10.0, 15.0])
    euler = quaternion_to_euler_zyx(*result[3:])
    assert euler == pytest.approx({"yaw": math.pi / 2.0, "pitch": 0.0, "roll": 0.0})


def test_slerp_quaternion_treats_negated_quaternions_as_same_rotation():
    """q 和 -q 表示同一旋转，插值不应绕远路。"""
    result = slerp_quaternion(
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, -1.0],
        0.5,
    )

    assert result == pytest.approx((0.0, 0.0, 0.0, 1.0))


def test_adaptive_search_windows_expand_to_configured_maximum():
    assert adaptive_search_windows(1_000, 100) == [100, 200, 400, 800, 1_000]


def test_build_interpolation_neighbors_collects_nearest_and_bracket():
    records = {
        "/pose": [
            {"matched_publish_ns": 900, "matched_log_ns": 90, "ros_message": object()},
            {"matched_publish_ns": 800, "matched_log_ns": 110, "ros_message": object()},
        ]
    }

    neighbors = _build_interpolation_neighbors(records, ["/pose"], target_ns=100)

    assert neighbors["/pose"]["nearest"]["matched_log_ns"] == 90
    assert neighbors["/pose"]["before"]["matched_log_ns"] == 90
    assert neighbors["/pose"]["after"]["matched_log_ns"] == 110
    assert _incomplete_interpolation_topics(neighbors, ["/pose"], max_tolerance_ns=5) == []


def test_sync_policy_is_parsed_as_milliseconds():
    assert _sync_policy_to_tolerance_ns("3") == 3_000_000
    assert _sync_policy_to_tolerance_ns("exact_frame") == 3_000_000
    assert _sync_policy_to_tolerance_ns("0.5") == 500_000


def test_resolve_interpolation_clamps_to_episode_first_and_last_values():
    first = {"matched_log_ns": 110, "diff_ns": 10, "ros_message": object()}
    last = {"matched_log_ns": 90, "diff_ns": 10, "ros_message": object()}

    strategy, ratio, selected, second = _resolve_interpolation(
        {"nearest": first, "before": None, "after": first},
        target_ns=100,
        max_tolerance_ns=5,
    )
    assert (strategy, ratio, selected, second) == ("clamp_first", None, first, None)

    strategy, ratio, selected, second = _resolve_interpolation(
        {"nearest": last, "before": last, "after": None},
        target_ns=100,
        max_tolerance_ns=5,
    )
    assert (strategy, ratio, selected, second) == ("clamp_last", None, last, None)


def test_robot_vector_extraction_uses_one_combined_window_scan(monkeypatch):
    target_ns = 1_000_000_000
    pose_fields = [
        "pose.position.x",
        "pose.position.y",
        "pose.position.z",
        "pose.orientation.x",
        "pose.orientation.y",
        "pose.orientation.z",
        "pose.orientation.w",
    ]
    pose_entries = [
        {"section": section, "group": group, "topic": pose_topic, "fields": pose_fields}
        for section in ("action", "observation_state")
        for group, pose_topic in (("left", "/pose_l"), ("right", "/pose_r"))
    ]
    gripper_entries = {
        (section, group): {"topic": gripper_topic, "fields": ["angle"]}
        for section in ("action", "observation_state")
        for group, gripper_topic in (("left", "/gripper_l"), ("right", "/gripper_r"))
    }
    pose_message = {
        "pose": {
            "position": {"x": 1.0, "y": 2.0, "z": 3.0},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        }
    }
    calls = []

    monkeypatch.setattr("common.extract_mcap_fields.load_configured_topic_fields", lambda *args, **kwargs: pose_entries)
    monkeypatch.setattr("common.extract_mcap_fields._load_gripper_entries", lambda *args, **kwargs: gripper_entries)
    monkeypatch.setattr("common.extract_mcap_fields._load_main_time_topic", lambda *args, **kwargs: "/main")

    def fake_scan(mcap_files, main_time_topic, sensor_topics, scan_target_ns, start_time, end_time, **kwargs):
        calls.append((main_time_topic, sensor_topics, scan_target_ns, start_time, end_time))
        records = {}
        for topic in sensor_topics:
            records[topic] = [{
                "mcap_file": "fake.mcap",
                "topic": topic,
                "schema_name": "fake",
                "matched_publish_ns": target_ns,
                "matched_log_ns": target_ns,
                "ros_message": pose_message if topic.startswith("/pose") else {"angle": 1.5},
            }]
        return (
            {
                "mcap_file": "fake.mcap",
                "topic": "/main",
                "schema_name": "sensor_msgs/msg/Image",
                "matched_publish_ns": target_ns,
                "matched_log_ns": target_ns,
                "diff_ns": 0,
            },
            records,
            5,
            False,
        )

    monkeypatch.setattr("common.extract_mcap_fields._scan_main_frame_and_sensor_messages", fake_scan)

    result = extract_robot_vectors_at_time(
        config_path="unused.json",
        mcap_dir=None,
        target_ns=target_ns,
        mcap_sources=[object()],
        mcap_source_label="test",
        require_chunk_indexes=True,
        allow_full_scan_fallback=False,
    )

    assert len(calls) == 1
    assert calls[0][3:] == (target_ns - 100_000_000, target_ns + 100_000_000 + 1)
    assert result["scanned_messages"] == 5
    assert result["search_attempt_windows_ns"] == [100_000_000]
    assert result["used_sensor_window_extension"] is False
    assert result["window_cache_hits"] == 0
    assert len(result["vectors"]) == 4


def test_main_camera_fallback_stays_inside_episode(monkeypatch):
    fallback_ranges = []
    monkeypatch.setattr(
        "common.extract_mcap_fields.load_configured_topic_fields",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        "common.extract_mcap_fields._load_gripper_entries",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        "common.extract_mcap_fields._load_main_time_topic",
        lambda *args, **kwargs: "/main",
    )
    monkeypatch.setattr(
        "common.extract_mcap_fields._scan_main_frame_and_sensor_messages",
        lambda *args, **kwargs: (None, {}, 0, False),
    )

    def fake_fallback(mcap_files, topics, target_ns, start_time, end_time, **kwargs):
        fallback_ranges.append((start_time, end_time))
        return {}, 0

    monkeypatch.setattr("common.extract_mcap_fields._scan_nearest_messages", fake_fallback)

    with pytest.raises(ValueError, match="未找到主相机 topic"):
        extract_robot_vectors_at_time(
            config_path="unused.json",
            mcap_dir=None,
            target_ns=150,
            mcap_sources=[object()],
            allow_full_scan_fallback=True,
            episode_start_ns=100,
            episode_end_ns=200,
        )

    assert fallback_ranges == [(100, 201)]


def test_robot_vector_extraction_reads_tianji_gripper_data_index(monkeypatch):
    """夹爪输出标签固定为 angle，但数值须遵循构型中的 data[0] 字段。"""
    target_ns = 1_000_000_000
    pose_fields = [
        "pose.position.x", "pose.position.y", "pose.position.z",
        "pose.orientation.x", "pose.orientation.y", "pose.orientation.z", "pose.orientation.w",
    ]
    pose_entries = [
        {"section": section, "group": group, "topic": f"/pose_{group}", "fields": pose_fields}
        for section in ("action", "observation_state")
        for group in ("left", "right")
    ]
    gripper_entries = {
        (section, group): {"topic": f"/gripper_{group}", "fields": ["data[0]"]}
        for section in ("action", "observation_state")
        for group in ("left", "right")
    }
    pose_message = {
        "pose": {
            "position": {"x": 1.0, "y": 2.0, "z": 3.0},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        }
    }
    monkeypatch.setattr("common.extract_mcap_fields.load_configured_topic_fields", lambda *args, **kwargs: pose_entries)
    monkeypatch.setattr("common.extract_mcap_fields._load_gripper_entries", lambda *args, **kwargs: gripper_entries)
    monkeypatch.setattr("common.extract_mcap_fields._load_main_time_topic", lambda *args, **kwargs: "/main")

    def fake_scan(mcap_files, main_time_topic, sensor_topics, scan_target_ns, start_time, end_time, **kwargs):
        records = {
            topic: [{
                "mcap_file": "fake.mcap", "topic": topic, "schema_name": "fake",
                "matched_publish_ns": target_ns, "matched_log_ns": target_ns,
                "ros_message": pose_message if topic.startswith("/pose") else {"data": [2.5]},
            }]
            for topic in sensor_topics
        }
        return (
            {
                "mcap_file": "fake.mcap", "topic": "/main", "schema_name": "image",
                "matched_publish_ns": target_ns, "matched_log_ns": target_ns, "diff_ns": 0,
            },
            records,
            len(records),
            False,
        )

    monkeypatch.setattr("common.extract_mcap_fields._scan_main_frame_and_sensor_messages", fake_scan)

    result = extract_robot_vectors_at_time(
        config_path="unused.json", mcap_dir=None, target_ns=target_ns,
        mcap_sources=[object()], mcap_source_label="test", require_chunk_indexes=True,
        allow_full_scan_fallback=False,
    )

    assert [item["values_by_label"]["angle"] for item in result["vectors"]] == [2.5] * 4
    assert all(item["gripper_fields"] == ["data[0]"] for item in result["vectors"])


def test_robot_vector_extraction_uses_slerp_in_interpolated_path(monkeypatch):
    target_ns = 100
    pose_fields = [
        "pose.position.x", "pose.position.y", "pose.position.z",
        "pose.orientation.x", "pose.orientation.y", "pose.orientation.z", "pose.orientation.w",
    ]
    pose_entries = [
        {
            "section": section,
            "group": group,
            "topic": f"/pose_{group}",
            "fields": pose_fields,
            "sync_tolerance_ns": 0,
        }
        for section in ("action", "observation_state")
        for group in ("left", "right")
    ]
    gripper_entries = {
        (section, group): {
            "topic": f"/gripper_{group}",
            "fields": ["angle"],
            "sync_tolerance_ns": 0,
        }
        for section in ("action", "observation_state")
        for group in ("left", "right")
    }
    start_q = [math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)]
    end_q = [0.0, math.sqrt(0.5), 0.0, math.sqrt(0.5)]

    def pose_message(quaternion):
        return {
            "pose": {
                "position": {"x": 0.0, "y": 0.0, "z": 0.0},
                "orientation": dict(zip(("x", "y", "z", "w"), quaternion)),
            }
        }

    monkeypatch.setattr("common.extract_mcap_fields.load_configured_topic_fields", lambda *args, **kwargs: pose_entries)
    monkeypatch.setattr("common.extract_mcap_fields._load_gripper_entries", lambda *args, **kwargs: gripper_entries)
    monkeypatch.setattr("common.extract_mcap_fields._load_main_time_topic", lambda *args, **kwargs: "/main")

    def fake_scan(mcap_files, topics, scan_target_ns, start_time, end_time, **kwargs):
        result = {}
        for topic in topics:
            is_pose = topic.startswith("/pose")
            before = {
                "matched_log_ns": 0, "matched_publish_ns": 10,
                "diff_ns": 100,
                "ros_message": pose_message(start_q) if is_pose else {"angle": 0.0},
            }
            after = {
                "matched_log_ns": 200, "matched_publish_ns": 210,
                "diff_ns": 100,
                "ros_message": pose_message(end_q) if is_pose else {"angle": 2.0},
            }
            result[topic] = {"nearest": before, "before": before, "after": after}
        return result, len(topics) * 2

    monkeypatch.setattr("common.extract_mcap_fields._scan_interpolation_neighbors", fake_scan)

    result = extract_robot_vectors_at_time(
        config_path="unused.json",
        mcap_dir=None,
        target_ns=target_ns,
        align_to_main_camera=False,
        mcap_sources=[object()],
        mcap_source_label="test",
        episode_start_ns=0,
        episode_end_ns=200,
    )

    expected_euler = quaternion_to_euler_zyx(*slerp_quaternion(start_q, end_q, 0.5))
    assert result["vectors"][0]["values_by_label"] == pytest.approx(
        {"x": 0.0, "y": 0.0, "z": 0.0, **expected_euler, "angle": 1.0}
    )
    assert result["vectors"][0]["pose_match"]["strategy"] == "interpolated"

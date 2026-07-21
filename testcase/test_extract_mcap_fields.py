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
    get_nested_field_value,
    interpolate_pose7d,
    load_configured_topic_fields,
    quaternion_to_euler_zyx,
    slerp_quaternion,
)


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

"""
MCAP 字段提取功能单元测试。

测试范围：
- 从 JSON 配置文件加载 topic 与字段映射
- 通过点号路径从嵌套对象中获取属性值
- 四元数转欧拉角（ZYX 顺序）
"""

import json
from pathlib import Path

import pytest

from common.extract_mcap_fields import (
    get_nested_field_value,
    load_configured_topic_fields,
    quaternion_to_euler_zyx,
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

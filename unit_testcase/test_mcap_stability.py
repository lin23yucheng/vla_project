from pathlib import Path

from common.mcap_stability import (
    CAMERA_VIEW_ROLES,
    _four_camera_header_skew,
    _is_image_schema,
    load_robot_camera_config,
    _timestamp_summary,
    _window_summaries,
)


def test_timestamp_summary_reports_duplicates_reversals_and_gaps():
    result = _timestamp_summary([10, 20, 20, 15, 45])

    assert result["out_of_order_count"] == 1
    assert result["duplicate_count"] == 1
    assert result["interval_count"] == 2
    assert result["interval_mean_ns"] == 20
    assert result["interval_p50_ns"] == 20
    assert result["longest_gap"] == {
        "interval_ns": 30,
        "start_ns": 15,
        "end_ns": 45,
    }


def test_window_summaries_split_long_capture_by_sixty_seconds():
    result = _window_summaries([0, 1_000_000_000, 60_000_000_000, 61_000_000_000])

    assert [window["message_count"] for window in result] == [2, 2]
    assert [window["interval_max_ns"] for window in result] == [
        1_000_000_000,
        59_000_000_000,
    ]


def test_camera_roles_and_ros_image_schema_detection():
    assert CAMERA_VIEW_ROLES["cam_head"] == "主视角"
    assert _is_image_schema("sensor_msgs/msg/CompressedImage")
    assert _is_image_schema("sensor_msgs/msg/Image")
    assert not _is_image_schema("sensor_msgs/msg/JointState")


def test_four_camera_header_skew_matches_nearest_frames():
    result = _four_camera_header_skew(
        {
            "/cam_head/color_compressed": [100, 200],
            "/cam_chest/color_compressed": [101, 201],
            "/cam_left/color_compressed": [99, 199],
            "/cam_right/color_compressed": [102, 202],
        }
    )

    assert result["reference_topic"] == "/cam_head/color_compressed"
    assert result["sample_count"] == 2
    assert result["maximum_ns"] == 3


def test_robot_config_supplies_camera_topics_and_joint_state_topics():
    config_path = Path(__file__).parent.parent / "Robot_Configuration" / "robot_config.yaml"

    camera_labels, configured_topics, camera_names = load_robot_camera_config(config_path)

    assert len(camera_labels) == 4
    assert camera_labels["/cam_head/color_compressed"] == "主视角（头部相机）"
    assert camera_labels["/cam_chest/color_compressed"] == "全景（胸部相机）"
    assert "/left_gripper/joint_states" in configured_topics
    assert camera_names == ["cam_chest", "cam_head", "cam_left", "cam_right"]

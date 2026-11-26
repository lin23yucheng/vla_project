from pathlib import Path

import csv
import pytest
from mcap.writer import Writer

from common.mcap_stability import (
    MAX_CAMERA_INTERVAL_NS,
    CAMERA_VIEW_ROLES,
    _four_camera_header_skew,
    _event_window_summaries,
    _is_image_schema,
    audit_mcap_file,
    load_robot_camera_config,
    load_audit_config,
    render_mcap_report,
    write_mcap_csv_reports,
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


def test_camera_rate_tolerance_is_applied_as_thirty_plus_or_minus_five_fps():
    assert MAX_CAMERA_INTERVAL_NS == 40_000_000
    windows = _window_summaries([0, 33_333_333, 66_666_666])
    assert windows[0]["observed_rate_hz"] == pytest.approx(30)


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


def test_report_puts_verdict_and_camera_metrics_before_warnings():
    result = {
        "is_valid": True,
        "root": "capture",
        "robot_config": "robot_config.yaml",
        "audit_config": "mcap_audit_config.json",
        "camera_fps_expectation": {
            "target_fps": 30,
            "accepted_range_fps": [25, 35],
            "maximum_interval_ms": 40,
        },
        "file_count": 1,
        "valid_file_count": 1,
        "files": [
            {
                "episode": 7,
                "file": "episode_7/rosbag_0.mcap",
                "is_valid": True,
                "camera_topics": ["/cam_head/color_compressed"],
                "topics": {
                    "/cam_head/color_compressed": {
                        "role": "主视角（头部相机）",
                        "message_count": 100,
                        "image_dimensions": {"1280x720": 100},
                        "image_decode_failures": 0,
                        "log_time": {
                            "observed_rate_hz": 30.0,
                            "interval_p50_ns": 33_000_000,
                            "interval_p95_ns": 34_000_000,
                            "interval_p99_ns": 35_000_000,
                            "interval_max_ns": 36_000_000,
                        },
                        "sixty_second_trend": [],
                    }
                },
                "warnings": ["未安装夹爪"],
                "errors": [],
            }
        ],
    }

    report = render_mcap_report(result)

    assert report.index("原始 MCAP 稳定性审计：PASS") < report.index("主视角")
    assert report.index("主视角") < report.index("数据提示")
    assert "25-35 FPS" in report
    assert "未安装夹爪" in report


def test_event_windows_report_gaps_timestamps_and_delay():
    config = load_audit_config()
    events = [
        {
            "log_time_ns": 1_000_000_000,
            "header_stamp_ns": 999_000_000,
            "header_valid": True,
            "image_valid": True,
        },
        {
            "log_time_ns": 1_100_000_000,
            "header_stamp_ns": 1_099_000_000,
            "header_valid": True,
            "image_valid": False,
        },
    ]

    windows = _event_window_summaries(events, config)

    assert windows[0]["over_threshold_gap_count"] == 1
    assert windows[0]["estimated_dropped_frames"] == 2
    assert windows[0]["image_decode_failures"] == 1
    assert windows[0]["record_delay_p50_ns"] == 1_000_000


def test_raw_mcap_scan_detects_log_time_reversal_in_file_order(tmp_path):
    topic = "/cam_head/color_compressed"
    mcap_path = tmp_path / "reversed.mcap"
    with mcap_path.open("wb") as stream:
        writer = Writer(stream)
        writer.start()
        schema_id = writer.register_schema(
            name="sensor_msgs/msg/CompressedImage",
            encoding="ros2msg",
            data=b"",
        )
        channel_id = writer.register_channel(
            topic=topic,
            message_encoding="cdr",
            schema_id=schema_id,
        )
        for timestamp_ns in (2_000_000_000, 1_000_000_000, 1_100_000_000):
            writer.add_message(
                channel_id=channel_id,
                log_time=timestamp_ns,
                publish_time=timestamp_ns,
                data=b"invalid image payload",
            )
        writer.finish()

    result = audit_mcap_file(
        mcap_path,
        tmp_path,
        {topic: "主视角"},
        [],
        load_audit_config(),
    )

    reversals = [
        item for item in result["anomalies"] if item["anomaly_type"] == "log_time_ns 倒退"
    ]
    assert len(reversals) == 1
    assert reversals[0]["message_index"] == 2
    assert reversals[0]["event_index"] == 2
    assert any(item["anomaly_type"] == "帧间隔超阈值（疑似丢帧）" for item in result["anomalies"])


def test_csv_report_writer_creates_expected_headers(tmp_path):
    paths = write_mcap_csv_reports(
        {"files": [{"file": "bad.mcap", "episode": 1, "anomalies": []}]},
        tmp_path,
    )

    assert all(Path(path).is_file() for path in paths.values())
    with Path(paths["anomalies_csv"]).open(encoding="utf-8-sig", newline="") as file:
        assert next(csv.reader(file)) == [
            "file", "episode", "topic", "message_index", "event_index",
            "timestamp_ns", "anomaly_type", "actual_value", "threshold",
        ]

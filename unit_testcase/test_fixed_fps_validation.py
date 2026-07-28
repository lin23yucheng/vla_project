from types import SimpleNamespace

import json
import pyarrow as pa
import pyarrow.parquet as pq

from common.fixed_fps_validation import (
    find_episode_first_main_frame_ns,
    find_episode_last_main_frame_ns,
    validate_episode_fixed_fps_timeline,
)


def test_episode_main_frame_bounds_use_log_time(monkeypatch):
    messages = [
        SimpleNamespace(log_time=110, publish_time=900),
        SimpleNamespace(log_time=190, publish_time=100),
    ]

    def fake_read(*args, **kwargs):
        return ([
            {
                "channel": SimpleNamespace(topic="/main"),
                "message": message,
            }
            for message in messages
        ], False)

    monkeypatch.setattr(
        "common.fixed_fps_validation.read_mcap_window_messages",
        fake_read,
    )

    assert find_episode_first_main_frame_ns(
        [object()], "/main", 100, 200, require_chunk_indexes=True,
    ) == 110
    assert find_episode_last_main_frame_ns(
        [object()], "/main", 100, 200, require_chunk_indexes=True,
    ) == 190


def _write_fixed_fps_episode(path, frame_indices):
    episode_start_ns = 90
    episode_end_ns = 70_000_000
    hierarchy = json.dumps(
        [
            {
                "level": 1,
                "layer_id": "l1",
                "id": "l1-episode",
                "name": "episode",
                "start_timestamp_ns": episode_start_ns,
                "end_timestamp_ns": episode_end_ns,
            }
        ]
    )
    pq.write_table(
        pa.table(
            {
                "timestamp": pa.array([0.0, 1 / 30, 2 / 30]),
                "original_timestamp_ns": pa.array(
                    [100, 33_333_433, 66_666_767],
                    type=pa.int64(),
                ),
                "frame_index": pa.array(frame_indices, type=pa.int64()),
                "episode_id": pa.array(["episode-1"] * 3),
                "episode_index": pa.array([0] * 3, type=pa.int64()),
                "annotation.hierarchy_json": pa.array([hierarchy] * 3),
            }
        ),
        path,
    )
    return episode_start_ns, episode_end_ns


def test_fixed_fps_validation_checks_frame_index_in_file_order(tmp_path, monkeypatch):
    parquet_path = tmp_path / "episode.parquet"
    episode_start_ns, episode_end_ns = _write_fixed_fps_episode(
        parquet_path,
        [0, 2, 2],
    )
    monkeypatch.setattr(
        "common.fixed_fps_validation.find_episode_first_main_frame_ns",
        lambda *args, **kwargs: 100,
    )
    monkeypatch.setattr(
        "common.fixed_fps_validation.find_episode_last_main_frame_ns",
        lambda *args, **kwargs: 66_666_767,
    )

    result = validate_episode_fixed_fps_timeline(
        mcap_sources=[object()],
        parquet_sources=[parquet_path],
        main_camera_topic="/main",
        episode_start_ns=episode_start_ns,
        episode_end_ns=episode_end_ns,
    )

    assert result["is_consistent"] is False
    assert any("frame_index 不连续" in failure for failure in result["failures"])


def test_fixed_fps_validation_accepts_complete_episode_timeline(tmp_path, monkeypatch):
    parquet_path = tmp_path / "valid_episode.parquet"
    episode_start_ns, episode_end_ns = _write_fixed_fps_episode(
        parquet_path,
        [0, 1, 2],
    )
    monkeypatch.setattr(
        "common.fixed_fps_validation.find_episode_first_main_frame_ns",
        lambda *args, **kwargs: 100,
    )
    monkeypatch.setattr(
        "common.fixed_fps_validation.find_episode_last_main_frame_ns",
        lambda *args, **kwargs: 66_666_767,
    )

    result = validate_episode_fixed_fps_timeline(
        mcap_sources=[object()],
        parquet_sources=[parquet_path],
        main_camera_topic="/main",
        episode_start_ns=episode_start_ns,
        episode_end_ns=episode_end_ns,
    )

    assert result["is_consistent"] is True
    assert result["actual_frame_count"] == result["expected_frame_count"] == 3
    assert result["episode_id"] == "episode-1"

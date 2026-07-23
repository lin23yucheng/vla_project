import json

import pyarrow as pa
import pyarrow.parquet as pq

from common.parse_parquet_file import (
    compare_parquet_annotations,
    extract_parquet_annotation_summary,
)


def _hierarchy(*items):
    return json.dumps(items)


def _annotation(level, segment_id, name, start_ns, end_ns):
    return {
        "level": level,
        "id": segment_id,
        "name": name,
        "layer_id": f"l{level}",
        "start_timestamp_ns": str(start_ns),
        "end_timestamp_ns": str(end_ns),
    }


def _write_annotation_parquet(path):
    l1 = _annotation(1, "l1-1", "task-all", 100, 400)
    l2_first = _annotation(2, "l2-1", "pick", 100, 199)
    l2_second = _annotation(2, "l2-2", "place", 200, 400)
    rows = []
    for frame_index, (timestamp_ns, leaf, progress, done) in enumerate(
        (
            (150, l2_first, 0.5, False),
            (250, l2_second, 0.2, False),
            (350, l2_second, 0.9, True),
        )
    ):
        rows.append(
            {
                "original_timestamp_ns": timestamp_ns,
                "timestamp": frame_index / 50,
                "episode_index": 0,
                "frame_index": frame_index,
                "index": frame_index,
                "episode_id": "episode-1",
                "task": l1["name"],
                "subtask": leaf["name"],
                "subtask_id": leaf["id"],
                "subtask_progress": progress,
                "annotation.level_1_id": l1["id"],
                "annotation.level_1_name": l1["name"],
                "annotation.leaf_id": leaf["id"],
                "annotation.leaf_name": leaf["name"],
                "annotation.path": f"{l1['name']}/{leaf['name']}",
                "annotation.hierarchy_json": _hierarchy(l1, leaf),
                "done": done,
            }
        )
    pq.write_table(pa.Table.from_pylist(rows), path)


def _expected_layers():
    return [
        {
            "layerId": "l1",
            "segments": [
                {
                    "segmentId": "l1-1",
                    "description": "task-all",
                    "startTimeNs": "100",
                    "endTimeNs": "400",
                }
            ],
        },
        {
            "layerId": "l2",
            "segments": [
                {
                    "segmentId": "l2-1",
                    "description": "pick",
                    "startTimeNs": "100",
                    "endTimeNs": "199",
                },
                {
                    "segmentId": "l2-2",
                    "description": "place",
                    "startTimeNs": "200",
                    "endTimeNs": "400",
                },
            ],
        },
    ]


def test_extract_parquet_annotation_summary(tmp_path):
    parquet_path = tmp_path / "annotations.parquet"
    _write_annotation_parquet(parquet_path)

    summary = extract_parquet_annotation_summary(parquet_path)

    assert summary["level_counts"] == {"L1": 1, "L2": 2, "L3": 0}
    assert summary["validation_error_count"] == 0
    assert summary["annotations"][0]["row_count"] == 3


def test_compare_parquet_annotations_reports_field_differences(tmp_path):
    parquet_path = tmp_path / "annotations.parquet"
    _write_annotation_parquet(parquet_path)

    consistent = compare_parquet_annotations(parquet_path, _expected_layers())
    assert consistent["is_consistent"] is True

    mismatched_layers = _expected_layers()
    mismatched_layers[1]["segments"][1]["description"] = "wrong-name"
    mismatched_layers[1]["segments"][1]["endTimeNs"] = "390"
    mismatched = compare_parquet_annotations(parquet_path, mismatched_layers)

    assert mismatched["is_consistent"] is False
    difference_fields = {
        difference["field"]
        for comparison in mismatched["segment_comparisons"]
        for difference in comparison["differences"]
    }
    assert {"name", "end_timestamp_ns"}.issubset(difference_fields)


def test_compare_parquet_annotations_reports_l1_playback_duration_difference(tmp_path):
    parquet_path = tmp_path / "annotations.parquet"
    _write_annotation_parquet(parquet_path)
    expected_layers = _expected_layers()
    expected_layers[0]["segments"][0]["endTimeNs"] = "100000100"
    result = compare_parquet_annotations(
        parquet_path,
        expected_layers,
        validate_l1_playback_duration=True,
    )

    assert result["is_consistent"] is False
    assert result["l1_playback_duration_comparisons"] == [
        {
            "expected_duration_ns": 100000000,
            "expected_duration_seconds": 0.1,
            "parquet_playback_start_seconds": 0.0,
            "parquet_playback_end_seconds": 0.04,
            "parquet_playback_duration_seconds": 0.04,
            "is_consistent": False,
        }
    ]
    assert any(
        "L1第1条播放时长不一致: parquet timestamp 0.0 -> 0.04，时长 0.04 秒；"
        "代码标注时长 0.1 秒（100000000 ns），允许误差 0.04 秒" == failure
        for failure in result["failures"]
    )


def test_compare_parquet_annotations_validates_each_l1_playback_independently(tmp_path):
    parquet_path = tmp_path / "two_l1_episodes.parquet"
    l1_first = _annotation(1, "l1-first", "first-task", 1_000_000_000, 3_000_000_000)
    l1_second = _annotation(1, "l1-second", "second-task", 10_000_000_000, 13_000_000_000)
    rows = []
    for frame_index, (l1, timestamp, original_timestamp_ns) in enumerate(
        (
            (l1_first, 0.0, 1_100_000_000),
            (l1_first, 2.0, 2_900_000_000),
            (l1_second, 0.0, 10_100_000_000),
            (l1_second, 3.0, 12_900_000_000),
        )
    ):
        rows.append(
            {
                "timestamp": timestamp,
                "original_timestamp_ns": original_timestamp_ns,
                "episode_index": frame_index // 2,
                "frame_index": frame_index,
                "index": frame_index,
                "episode_id": f"episode-{frame_index // 2}",
                "task": l1["name"],
                "subtask": l1["name"],
                "subtask_id": l1["id"],
                "subtask_progress": 1.0,
                "annotation.level_1_id": l1["id"],
                "annotation.level_1_name": l1["name"],
                "annotation.leaf_id": l1["id"],
                "annotation.leaf_name": l1["name"],
                "annotation.path": l1["name"],
                "annotation.hierarchy_json": _hierarchy(l1),
                "done": frame_index in (1, 3),
            }
        )
    pq.write_table(pa.Table.from_pylist(rows), parquet_path)
    expected_layers = [
        {
            "layerId": "l1",
            "segments": [
                {
                    "segmentId": l1_first["id"],
                    "description": l1_first["name"],
                    "startTimeNs": l1_first["start_timestamp_ns"],
                    "endTimeNs": l1_first["end_timestamp_ns"],
                },
                {
                    "segmentId": l1_second["id"],
                    "description": l1_second["name"],
                    "startTimeNs": l1_second["start_timestamp_ns"],
                    "endTimeNs": l1_second["end_timestamp_ns"],
                },
            ],
        }
    ]

    result = compare_parquet_annotations(
        parquet_path,
        expected_layers,
        validate_l1_playback_duration=True,
    )

    assert result["is_consistent"] is True
    assert [
        comparison["parquet_playback_duration_seconds"]
        for comparison in result["l1_playback_duration_comparisons"]
    ] == [2.0, 3.0]

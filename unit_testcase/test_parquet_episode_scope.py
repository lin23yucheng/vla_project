import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from common.parquet_source import (
    find_nearest_parquet_row,
    resolve_unique_parquet_episode_source,
)


def _write_episode(path, *, episode_id, episode_index, l1_id, start_ns, end_ns, values):
    hierarchy = json.dumps(
        [
            {
                "level": 1,
                "layer_id": "l1",
                "id": l1_id,
                "name": l1_id,
                "start_timestamp_ns": start_ns,
                "end_timestamp_ns": end_ns,
            }
        ]
    )
    pq.write_table(
        pa.table(
            {
                "original_timestamp_ns": pa.array(values, type=pa.int64()),
                "episode_id": pa.array([episode_id] * len(values)),
                "episode_index": pa.array([episode_index] * len(values), type=pa.int64()),
                "annotation.hierarchy_json": pa.array([hierarchy] * len(values)),
            }
        ),
        path,
    )


def test_nearest_parquet_row_is_limited_to_unique_episode(tmp_path):
    first = tmp_path / "episode_000000.parquet"
    second = tmp_path / "episode_000001.parquet"
    _write_episode(
        first,
        episode_id="episode-a",
        episode_index=0,
        l1_id="l1-a",
        start_ns=100,
        end_ns=300,
        values=[150, 200],
    )
    _write_episode(
        second,
        episode_id="episode-b",
        episode_index=1,
        l1_id="l1-b",
        start_ns=180,
        end_ns=400,
        values=[200, 250],
    )

    match = find_nearest_parquet_row(
        [first, second],
        200,
        episode_start_ns=180,
        episode_end_ns=400,
    )

    assert match["source"] == second
    assert match["episode_id"] == "episode-b"
    assert match["episode_index"] == 1
    assert match["l1_id"] == "l1-b"


def test_episode_resolution_rejects_non_unique_l1_mapping(tmp_path):
    sources = []
    for index in range(2):
        path = tmp_path / f"duplicate_{index}.parquet"
        _write_episode(
            path,
            episode_id=f"episode-{index}",
            episode_index=index,
            l1_id=f"l1-{index}",
            start_ns=100,
            end_ns=300,
            values=[150, 200],
        )
        sources.append(path)

    with pytest.raises(ValueError, match="无法唯一映射"):
        resolve_unique_parquet_episode_source(sources, 100, 300)


def test_nearest_row_rejects_global_matching_across_sources(tmp_path):
    sources = []
    for index in range(2):
        path = tmp_path / f"episode_{index}.parquet"
        _write_episode(
            path,
            episode_id=f"episode-{index}",
            episode_index=index,
            l1_id=f"l1-{index}",
            start_ns=index * 100,
            end_ns=index * 100 + 90,
            values=[index * 100 + 10],
        )
        sources.append(path)

    with pytest.raises(ValueError, match="禁止全局最近行匹配"):
        find_nearest_parquet_row(sources, 50)

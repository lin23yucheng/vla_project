import io
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from common.parse_parquet_file import extract_nearest_parquet_images_batch


def _png_bytes(color: tuple[int, int, int]) -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", (2, 2), color).save(stream, format="PNG")
    return stream.getvalue()


def test_batch_image_extraction_reads_shared_row_group_once(tmp_path: Path):
    image_type = pa.struct([
        pa.field("bytes", pa.binary()),
        pa.field("path", pa.string()),
    ])
    timestamps = [100, 200, 300]
    left_images = [
        {"bytes": _png_bytes((index, 0, 0)), "path": f"left_{index}.png"}
        for index in range(3)
    ]
    right_images = [
        {"bytes": _png_bytes((0, index, 0)), "path": f"right_{index}.png"}
        for index in range(3)
    ]
    table = pa.table(
        {
            "original_timestamp_ns": pa.array(timestamps, type=pa.int64()),
            "wrist_image_left": pa.array(left_images, type=image_type),
            "wrist_image_right": pa.array(right_images, type=image_type),
        }
    )
    parquet_path = tmp_path / "images.parquet"
    pq.write_table(table, parquet_path, row_group_size=3)

    result = extract_nearest_parquet_images_batch(
        parquet_path=parquet_path,
        target_requests=[
            {"key": "start", "target_ns": 101, "extra_name_parts": ["start"]},
            {"key": "end", "target_ns": 299, "extra_name_parts": ["end"]},
        ],
        output_dir=tmp_path / "output",
        name_prefix="batch",
    )

    assert result["target_count"] == 2
    assert result["row_groups_read"] == 1
    assert result["results"]["start"]["matched_row_index"] == 0
    assert result["results"]["end"]["matched_row_index"] == 2
    assert result["results"]["start"]["batch_row_group_target_count"] == 2
    for extract_result in result["results"].values():
        assert len(extract_result["saved_files"]) == 2
        assert all(Path(item["saved_path"]).is_file() for item in extract_result["saved_files"])

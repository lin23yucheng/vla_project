import io
import shutil
from pathlib import Path

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from common.parse_parquet_file import extract_nearest_parquet_images_batch


def _png_bytes(color: tuple[int, int, int]) -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", (2, 2), color).save(stream, format="PNG")
    return stream.getvalue()


def _write_test_video(path: Path, colors: list[tuple[int, int, int]]) -> None:
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=10)
        stream.width = 16
        stream.height = 16
        stream.pix_fmt = "yuv420p"
        for color in colors:
            pixels = np.full((16, 16, 3), color, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


class _FakeS3Client:
    def __init__(self, objects: dict[str, Path]):
        self.objects = objects
        self.downloads: list[str] = []

    def fget_object(self, bucket: str, object_name: str, destination: str):
        self.downloads.append(object_name)
        shutil.copyfile(self.objects[object_name], destination)


class _FakeS3ParquetSource:
    def __init__(self, parquet_path: Path, client: _FakeS3Client):
        self.parquet_path = parquet_path
        self.client = client
        self.bucket = "test-bucket"
        self.object_name = (
            "task/final-data/lerobot_v2/data/chunk-000/episode_000000.parquet"
        )

    def open(self):
        return self.parquet_path.open("rb")

    def __str__(self) -> str:
        return f"s3://{self.bucket}/{self.object_name}"


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


def test_batch_image_extraction_supports_tianji_camera_names(tmp_path: Path):
    image_type = pa.struct([
        pa.field("bytes", pa.binary()),
        pa.field("path", pa.string()),
    ])
    camera_names = ["left_eye", "right_eye", "left_image", "right_image"]
    table = pa.table(
        {
            "original_timestamp_ns": pa.array([100], type=pa.int64()),
            **{
                name: pa.array(
                    [{"bytes": _png_bytes((index, index, index)), "path": f"{name}.png"}],
                    type=image_type,
                )
                for index, name in enumerate(camera_names)
            },
        }
    )
    parquet_path = tmp_path / "tianji_images.parquet"
    pq.write_table(table, parquet_path)

    result = extract_nearest_parquet_images_batch(
        parquet_path=parquet_path,
        target_requests=[{"key": "start", "target_ns": 100}],
        output_dir=tmp_path / "output",
        image_columns=camera_names,
    )

    saved_files = result["results"]["start"]["saved_files"]
    assert [item["column"] for item in saved_files] == camera_names
    assert all(Path(item["saved_path"]).is_file() for item in saved_files)


def test_batch_image_extraction_prefers_s3_video_frames(tmp_path: Path):
    image_type = pa.struct([
        pa.field("bytes", pa.binary()),
        pa.field("path", pa.string()),
    ])
    images = [
        {"bytes": _png_bytes(color), "path": f"frame_{index:06d}.png"}
        for index, color in enumerate([(255, 0, 0), (0, 255, 0), (0, 0, 255)])
    ]
    parquet_path = tmp_path / "episode_000000.parquet"
    pq.write_table(
        pa.table(
            {
                "original_timestamp_ns": pa.array([100, 200, 300], type=pa.int64()),
                "left_eye": pa.array(images, type=image_type),
            }
        ),
        parquet_path,
    )
    video_path = tmp_path / "episode_000000.mp4"
    _write_test_video(video_path, [(255, 0, 0), (0, 255, 0), (0, 0, 255)])
    video_object = (
        "task/final-data/lerobot_v2/videos/left_eye/chunk-000/episode_000000.mp4"
    )
    client = _FakeS3Client({video_object: video_path})
    source = _FakeS3ParquetSource(parquet_path, client)

    result = extract_nearest_parquet_images_batch(
        parquet_path=[source],
        target_requests=[
            {"key": "start", "target_ns": 101},
            {"key": "end", "target_ns": 299},
        ],
        output_dir=tmp_path / "output",
        image_columns=["left_eye"],
        prefer_s3_videos=True,
    )

    assert result["image_source_mode"] == "s3_video"
    assert result["row_groups_read"] == 0
    assert result["video_files_read"] == 1
    assert client.downloads == [video_object]
    assert result["results"]["start"]["saved_files"][0]["video_frame_index"] == 0
    assert result["results"]["end"]["saved_files"][0]["video_frame_index"] == 2
    for extract_result in result["results"].values():
        saved_file = extract_result["saved_files"][0]
        assert saved_file["decode_mode"] == "s3_video_frame"
        assert Path(saved_file["saved_path"]).is_file()

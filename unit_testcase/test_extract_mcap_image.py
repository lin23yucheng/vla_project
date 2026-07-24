import io
import struct

from mcap.writer import Writer
from PIL import Image

from common.extract_mcap_image import (
    extract_global_nearest_image_from_mcap_sources,
    parse_ros2_compressed_image,
)


def _cdr_string(value: str) -> bytes:
    encoded = value.encode("utf-8") + b"\x00"
    return struct.pack("<I", len(encoded)) + encoded


def _align_4(data: bytes) -> bytes:
    return data + b"\x00" * (-len(data) % 4)


def _compressed_image_cdr(image_data: bytes) -> bytes:
    data = b"\x00\x01\x00\x00"
    data += struct.pack("<iI", 14_939, 56_887_039)
    data += _cdr_string("camera")
    data = _align_4(data)
    data += _cdr_string("png")
    data = _align_4(data)
    data += struct.pack("<I", len(image_data)) + image_data
    return data


def _png_bytes() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (3, 2), (10, 20, 30)).save(output, format="PNG")
    return output.getvalue()


def test_parse_ros2_compressed_image():
    parsed = parse_ros2_compressed_image(_compressed_image_cdr(_png_bytes()))

    assert parsed is not None
    assert (parsed["width"], parsed["height"]) == (3, 2)
    assert parsed["encoding"] == "rgb8"
    assert parsed["step"] == 9
    assert len(parsed["data"]) == 18


def test_extract_compressed_image_from_indexed_mcap(tmp_path):
    topic = "/camera/top_left/image_raw"
    target_ns = 14_939_056_887_039
    mcap_path = tmp_path / "compressed.mcap"
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
        writer.add_message(
            channel_id=channel_id,
            log_time=target_ns,
            publish_time=target_ns,
            data=_compressed_image_cdr(_png_bytes()),
        )
        writer.finish()

    result = extract_global_nearest_image_from_mcap_sources(
        mcap_sources=[mcap_path],
        mcap_source_label=str(tmp_path),
        topics=[topic],
        target_ns=target_ns,
        output_dir=tmp_path / "output",
        require_chunk_indexes=True,
    )

    assert result["errors"] == []
    assert result["results"][0]["schema_name"] == "sensor_msgs/msg/CompressedImage"
    preview_path = result["results"][0]["preview_path"]
    assert preview_path is not None
    with Image.open(preview_path) as preview:
        assert preview.size == (3, 2)

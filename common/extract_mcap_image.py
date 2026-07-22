"""
从 MCAP 文件中提取指定 topic 在指定纳秒时间戳附近的图像消息
支持保存图像元数据及解码后的图片文件
"""

import json
import os
import re
import sys
import struct
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Any, Optional, Sequence, TypedDict

from mcap.reader import make_reader

from common.mcap_source import (
    DEFAULT_INITIAL_SEARCH_WINDOW_NS,
    adaptive_search_windows,
    local_mcap_sources,
    mcap_source_name,
    open_mcap_source,
    read_mcap_window_messages,
    select_mcap_sources_for_window,
)


class ParsedImageInfo(TypedDict):
    header_stamp_sec: int
    header_stamp_nanosec: int
    frame_id: str
    height: int
    width: int
    encoding: str
    is_bigendian: int
    step: int
    data: bytes
    data_length: int


class SaveImageResult(TypedDict):
    saved_paths: list[str]
    preview_path: Optional[str]
    preview_error: Optional[str]


def _sanitize_filename_component(value: str) -> str:
    sanitized = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff_-]+", "_", str(value).strip())
    return sanitized.strip("_") or "unknown"


def _infer_side_label(topic_name: str) -> str:
    lower_topic = topic_name.lower()
    if "_l/" in lower_topic or "left" in lower_topic:
        return "left"
    if "_r/" in lower_topic or "right" in lower_topic:
        return "right"
    return _sanitize_filename_component(lower_topic)


def clean_path(raw: str) -> str:
    """去除路径中可能混入的不可见 Unicode 控制字符"""
    invisible = "\u202a\u202b\u202c\u202d\u202e\u200e\u200f\ufeff"
    for ch in invisible:
        raw = raw.replace(ch, "")
    return raw.strip().strip('"')


def ns_to_str(ns: int) -> str:
    dt = datetime.fromtimestamp(ns / 1e9)
    return dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def parse_ros2_image(data: bytes) -> Optional[ParsedImageInfo]:
    """
    手动解析 ROS2 sensor_msgs/Image 的 CDR 序列化数据。
    不依赖 ROS2 环境，仅解析前几个固定字段获取图像元信息。
    返回字典或 None。
    """
    try:
        offset = 0

        def align(boundary: int):
            nonlocal offset
            remainder = offset % boundary
            if remainder:
                offset += boundary - remainder

        def ensure_size(size: int):
            if offset + size > len(data):
                raise ValueError("消息长度不足，无法继续解析")

        # CDR 头部 4 字节（通常跳过）
        if len(data) < 4:
            return None
        offset += 4

        # --- header.stamp.sec (uint32) ---
        align(4)
        ensure_size(4)
        sec = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        # --- header.stamp.nanosec (uint32) ---
        ensure_size(4)
        nanosec = struct.unpack_from("<I", data, offset)[0]
        offset += 4

        # --- header.frame_id (string) ---
        # ROS2 CDR string: 4-byte length (含末尾\0), 内容, 4-byte 对齐
        align(4)
        ensure_size(4)
        fid_len = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        ensure_size(fid_len)
        frame_id = data[offset:offset + fid_len - 1].decode("utf-8", errors="ignore") if fid_len > 0 else ""
        offset += fid_len
        # 4 字节对齐
        align(4)

        # --- height (uint32) ---
        ensure_size(4)
        height = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        # --- width (uint32) ---
        ensure_size(4)
        width = struct.unpack_from("<I", data, offset)[0]
        offset += 4

        # --- encoding (string) ---
        align(4)
        ensure_size(4)
        enc_len = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        ensure_size(enc_len)
        encoding = data[offset:offset + enc_len - 1].decode("utf-8", errors="ignore") if enc_len > 0 else ""
        offset += enc_len

        # --- is_bigendian (uint8) ---
        ensure_size(1)
        is_bigendian = struct.unpack_from("<B", data, offset)[0]
        offset += 1
        align(4)

        # --- step (uint32) ---
        ensure_size(4)
        step = struct.unpack_from("<I", data, offset)[0]
        offset += 4

        # --- data (uint8[]) ---
        align(4)
        ensure_size(4)
        data_len = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        ensure_size(data_len)
        image_data = data[offset:offset + data_len]

        result: ParsedImageInfo = {
            "header_stamp_sec": sec,
            "header_stamp_nanosec": nanosec,
            "frame_id": frame_id,
            "height": height,
            "width": width,
            "encoding": encoding,
            "is_bigendian": is_bigendian,
            "step": step,
            "data": image_data,
            "data_length": data_len,
        }
        return result
    except Exception:
        return None


def decode_image_for_preview(image_info: ParsedImageInfo):
    """将常见 ROS2 Image 编码转换为 PIL Image，供保存和预览使用。"""
    from PIL import Image
    import numpy as np

    encoding = image_info["encoding"]
    img_data = image_info["data"]
    width = image_info["width"]
    height = image_info["height"]
    step = image_info.get("step", 0)

    if width <= 0 or height <= 0:
        raise ValueError(f"无效图像尺寸: {width}x{height}")

    def reshape_rows(dtype, channels=1):
        bytes_per_channel = np.dtype(dtype).itemsize
        row_pixels = width * channels
        expected_row_bytes = row_pixels * bytes_per_channel
        row_bytes = step or expected_row_bytes
        expected_total_bytes = row_bytes * height

        if len(img_data) < expected_total_bytes:
            raise ValueError(
                f"图像数据长度不足: 期望至少 {expected_total_bytes} bytes，实际 {len(img_data)} bytes"
            )

        arr = np.frombuffer(img_data[:expected_total_bytes], dtype=np.uint8).reshape((height, row_bytes))
        trimmed = arr[:, :expected_row_bytes]
        return trimmed.view(dtype).reshape((height, width) if channels == 1 else (height, width, channels))

    if encoding in ("rgb8", "RGB8"):
        arr = reshape_rows(np.uint8, 3)
        return Image.fromarray(arr, mode="RGB")
    if encoding in ("bgr8", "BGR8"):
        arr = reshape_rows(np.uint8, 3)[:, :, ::-1]
        return Image.fromarray(arr, mode="RGB")
    if encoding in ("rgba8", "RGBA8"):
        arr = reshape_rows(np.uint8, 4)
        return Image.fromarray(arr, mode="RGBA")
    if encoding in ("bgra8", "BGRA8"):
        arr = reshape_rows(np.uint8, 4)[:, :, [2, 1, 0, 3]]
        return Image.fromarray(arr, mode="RGBA")
    if encoding in ("mono8", "MONO8", "8UC1"):
        arr = reshape_rows(np.uint8)
        return Image.fromarray(arr, mode="L")
    if encoding in ("mono16", "MONO16", "16UC1"):
        arr = reshape_rows(np.uint16)
        arr = (arr / 256).astype(np.uint8)
        return Image.fromarray(arr, mode="L")

    raise ValueError(f"暂不支持直接预览的图像编码: {encoding}")


def open_image_file(image_path: str) -> tuple[bool, Optional[str]]:
    """使用系统默认图片查看器打开图片。"""
    try:
        if sys.platform.startswith("win"):
            os.startfile(image_path)
        elif sys.platform == "darwin":
            subprocess.run(["open", image_path], check=True)
        else:
            subprocess.run(["xdg-open", image_path], check=True)
        return True, None
    except Exception as exc:
        return False, str(exc)


def save_image(
    image_info: ParsedImageInfo,
    out_dir: str,
    base_name: str,
    include_raw_bin: bool = True,
    include_meta_json: bool = True,
) -> SaveImageResult:
    """尝试将解析出的图像数据保存为常见格式，同时保存元数据 JSON。"""
    encoding = image_info["encoding"]
    img_data = image_info["data"]
    width = image_info["width"]
    height = image_info["height"]

    saved = []
    preview_path = None
    preview_error = None

    # 按需保存元数据 JSON（mcap 图片人工比对场景可关闭）
    if include_meta_json:
        meta = {
            "width": width,
            "height": height,
            "encoding": encoding,
            "step": image_info.get("step", 0),
            "is_bigendian": image_info.get("is_bigendian", 0),
            "data_length": len(img_data),
        }
        meta_path = os.path.join(out_dir, f"{base_name}.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        saved.append(meta_path)

    # 生成可直接打开的预览图
    try:
        img = decode_image_for_preview(image_info)
        preview_path = os.path.join(out_dir, f"{base_name}.png")
        img.save(preview_path)
        saved.append(preview_path)
    except Exception as exc:
        preview_error = str(exc)

    # 按需保存原始数据（批量图片比对场景可关闭，仅保留 png/json）
    if include_raw_bin:
        raw_path = os.path.join(out_dir, f"{base_name}_{encoding.replace(' ', '_')}.bin")
        with open(raw_path, "wb") as f:
            f.write(img_data)
        saved.append(raw_path)

    return {
        "saved_paths": saved,
        "preview_path": preview_path,
        "preview_error": preview_error,
    }


def extract_nearest_image_from_mcap(
    mcap_path: str | Path,
    topic_name: str,
    target_ns: int,
    output_dir: str | Path,
    name_prefix: str | None = None,
    extra_name_parts: list[str] | None = None,
) -> dict:
    """从单个 MCAP 中提取指定 topic 在目标时间附近最近的一帧图像并保存。"""
    mcap_file_path = Path(mcap_path)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    best_msg = None
    best_diff: Optional[int] = None
    total_topic_messages = 0

    with open(mcap_file_path, "rb") as f:
        reader = make_reader(f)
        for schema, channel, message in reader.iter_messages():
            if channel.topic != topic_name:
                continue
            total_topic_messages += 1
            diff = abs(message.publish_time - target_ns)
            if best_diff is None or diff < best_diff:
                best_diff = diff
                best_msg = message

    if best_msg is None or best_diff is None:
        raise ValueError(f"MCAP 文件 {mcap_file_path} 未找到 topic {topic_name} 的消息")

    img_info = parse_ros2_image(best_msg.data)
    if not img_info:
        raise ValueError(f"MCAP 文件 {mcap_file_path} 中 topic {topic_name} 的最近消息无法解析为 ROS2 Image")

    matched_ns = int(best_msg.publish_time)
    side_label = _infer_side_label(topic_name)
    name_parts = []
    if name_prefix:
        name_parts.append(_sanitize_filename_component(name_prefix))
    name_parts.extend(
        [
            "mcap",
            _sanitize_filename_component(mcap_file_path.stem),
            side_label,
            f"target_{int(target_ns)}",
            f"matched_{matched_ns}",
        ]
    )
    for extra in extra_name_parts or []:
        if extra:
            name_parts.append(_sanitize_filename_component(extra))
    base_name = "_".join(name_parts)
    save_result = save_image(img_info, str(output_root), base_name)

    return {
        "mcap_file": str(mcap_file_path),
        "topic_name": topic_name,
        "side": side_label,
        "target_ns": int(target_ns),
        "matched_ns": matched_ns,
        "diff_ns": best_diff,
        "total_topic_messages": total_topic_messages,
        "saved_paths": save_result["saved_paths"],
        "preview_path": save_result["preview_path"],
        "preview_error": save_result["preview_error"],
        "encoding": img_info.get("encoding"),
        "width": img_info.get("width"),
        "height": img_info.get("height"),
    }


def extract_nearest_images_from_mcap_directory(
    mcap_dir: str | Path,
    topics: list[str],
    target_ns: int,
    output_dir: str | Path,
    name_prefix: str | None = None,
    extra_name_parts: list[str] | None = None,
) -> dict:
    """遍历目录中的所有 mcap 文件，为每个文件每个 topic 提取最近图片。"""
    mcap_root = Path(mcap_dir)
    if not mcap_root.exists():
        raise FileNotFoundError(f"mcap 目录不存在: {mcap_root}")

    mcap_files = sorted([p for p in mcap_root.glob("*.mcap") if p.is_file()])
    if not mcap_files:
        raise FileNotFoundError(f"mcap 目录下未找到 .mcap 文件: {mcap_root}")

    results = []
    errors = []
    for mcap_file in mcap_files:
        for topic in topics:
            try:
                result = extract_nearest_image_from_mcap(
                    mcap_path=mcap_file,
                    topic_name=topic,
                    target_ns=target_ns,
                    output_dir=output_dir,
                    name_prefix=name_prefix,
                    extra_name_parts=extra_name_parts,
                )
                results.append(result)
            except Exception as exc:
                errors.append(
                    {
                        "mcap_file": str(mcap_file),
                        "topic_name": topic,
                        "error": str(exc),
                    }
                )

    return {
        "mcap_dir": str(mcap_root),
        "target_ns": int(target_ns),
        "topics": topics,
        "output_dir": str(Path(output_dir)),
        "results": results,
        "errors": errors,
    }


def extract_global_nearest_image_from_mcap_directory(
    mcap_dir: str | Path,
    topics: list[str],
    target_ns: int,
    output_dir: str | Path,
    name_prefix: str | None = None,
    extra_name_parts: list[str] | None = None,
) -> dict[str, Any]:
    """在目录全部 mcap 文件中，按每个 topic 全局找最近的一帧（每个 topic 只输出一张图）。"""
    mcap_files, source_label = local_mcap_sources(mcap_dir)
    return extract_global_nearest_image_from_mcap_sources(
        mcap_sources=mcap_files,
        mcap_source_label=source_label,
        topics=topics,
        target_ns=target_ns,
        output_dir=output_dir,
        name_prefix=name_prefix,
        extra_name_parts=extra_name_parts,
        search_window_ns=None,
        allow_full_scan_fallback=True,
        require_chunk_indexes=False,
    )


def extract_global_nearest_image_from_mcap_sources(
    mcap_sources: Sequence[Any],
    mcap_source_label: str,
    topics: list[str],
    target_ns: int,
    output_dir: str | Path,
    name_prefix: str | None = None,
    extra_name_parts: list[str] | None = None,
    search_window_ns: int | None = 1_000_000_000,
    allow_full_scan_fallback: bool = False,
    require_chunk_indexes: bool = True,
    initial_search_window_ns: int = DEFAULT_INITIAL_SEARCH_WINDOW_NS,
    additional_cache_topics: Sequence[str] | None = None,
) -> dict[str, Any]:
    """从本地或远端 MCAP 源按索引提取目标时间附近的多路图片。"""
    mcap_files = list(mcap_sources)
    if not mcap_files:
        raise ValueError("mcap_sources 不能为空")

    if not topics:
        raise ValueError("topics 不能为空")
    if search_window_ns is not None and search_window_ns < 0:
        raise ValueError("search_window_ns 不能小于 0")
    if require_chunk_indexes and search_window_ns is None:
        raise ValueError("远端 indexed MCAP 模式必须设置有限的 search_window_ns")
    if require_chunk_indexes and allow_full_scan_fallback:
        raise ValueError(
            "远端 indexed MCAP 模式禁止 allow_full_scan_fallback，避免读取完整对象"
        )

    target_ns = int(target_ns)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    best_by_topic: dict[str, dict[str, Any]] = {}
    topic_message_counts = {topic: 0 for topic in topics}
    scanned_messages = 0
    window_cache_hits = 0

    def scan(
        scan_topics: list[str],
        start_time: int | None,
        end_time: int | None,
    ) -> None:
        nonlocal scanned_messages, window_cache_hits
        if start_time is not None and end_time is not None:
            read_topics = list(dict.fromkeys([
                *scan_topics,
                *(additional_cache_topics or []),
            ]))
            window_records, cache_hit = read_mcap_window_messages(
                mcap_files,
                read_topics,
                start_time,
                end_time,
                require_chunk_indexes=require_chunk_indexes,
            )
            window_cache_hits += int(cache_hit)
            for window_record in window_records:
                channel = window_record["channel"]
                message = window_record["message"]
                topic_name = channel.topic
                if topic_name not in scan_topics:
                    continue
                scanned_messages += 1
                topic_message_counts[topic_name] += 1
                diff_ns = abs(int(message.publish_time) - target_ns)
                current = best_by_topic.get(topic_name)
                if current is None or diff_ns < current["diff_ns"]:
                    best_by_topic[topic_name] = {
                        "mcap_file": window_record["mcap_file"],
                        "matched_ns": int(message.publish_time),
                        "diff_ns": diff_ns,
                        "data": bytes(message.data),
                    }
            return

        selected_sources = select_mcap_sources_for_window(
            mcap_files,
            start_time,
            end_time,
        )
        for mcap_file in selected_sources:
            with open_mcap_source(mcap_file) as stream:
                reader = make_reader(stream)
                if require_chunk_indexes:
                    summary = reader.get_summary()
                    if summary is None or not summary.chunk_indexes:
                        raise ValueError(
                            f"远端 MCAP 缺少 chunk index，拒绝顺序读取完整对象: "
                            f"{mcap_source_name(mcap_file)}"
                        )
                for _, channel, message in reader.iter_messages(
                    topics=scan_topics,
                    start_time=start_time,
                    end_time=end_time,
                ):
                    scanned_messages += 1
                    topic_name = channel.topic
                    topic_message_counts[topic_name] += 1
                    diff_ns = abs(int(message.publish_time) - target_ns)
                    current = best_by_topic.get(topic_name)
                    if current is None or diff_ns < current["diff_ns"]:
                        best_by_topic[topic_name] = {
                            "mcap_file": mcap_source_name(mcap_file),
                            "matched_ns": int(message.publish_time),
                            "diff_ns": diff_ns,
                            "data": bytes(message.data),
                        }

    search_attempt_windows_ns: list[int | None] = []
    effective_search_window_ns: int | None = None
    if search_window_ns is None:
        search_attempt_windows_ns.append(None)
        scan(topics, None, None)
    else:
        missing_topics = list(topics)
        for window_ns in adaptive_search_windows(
            search_window_ns,
            initial_search_window_ns,
        ):
            search_attempt_windows_ns.append(window_ns)
            effective_search_window_ns = window_ns
            scan(
                missing_topics,
                target_ns - window_ns,
                target_ns + window_ns + 1,
            )
            missing_topics = [topic for topic in topics if topic not in best_by_topic]
            if not missing_topics:
                break
    missing_topics = [topic for topic in topics if topic not in best_by_topic]
    used_full_scan_fallback = False
    if missing_topics and allow_full_scan_fallback and search_window_ns is not None:
        scan(missing_topics, None, None)
        used_full_scan_fallback = True

    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for topic_name in topics:
        best = best_by_topic.get(topic_name)
        if best is None:
            errors.append(
                {
                    "topic_name": topic_name,
                    "error": (
                        f"在 {mcap_source_label} 的"
                        f"{'全部数据' if search_window_ns is None else f'目标时间 ±{search_window_ns}ns 内'}"
                        f"未找到 topic={topic_name} 的可用图像消息"
                    ),
                }
            )
            continue

        img_info = parse_ros2_image(best["data"])
        if not img_info:
            errors.append({"topic_name": topic_name, "error": f"topic={topic_name} 的最近消息无法解析为 ROS2 Image"})
            continue

        side_label = _infer_side_label(topic_name)
        name_parts = []
        if name_prefix:
            name_parts.append(_sanitize_filename_component(name_prefix))
        name_parts.extend(
            [
                "mcap",
                side_label,
                _sanitize_filename_component(topic_name),
                f"target_{target_ns}",
                f"matched_{best['matched_ns']}",
            ]
        )
        for extra in extra_name_parts or []:
            if extra:
                name_parts.append(_sanitize_filename_component(extra))
        base_name = "_".join(name_parts)

        save_result = save_image(
            img_info,
            str(output_root),
            base_name,
            include_raw_bin=False,
            include_meta_json=False,
        )
        results.append(
            {
                "mcap_file": best["mcap_file"],
                "topic_name": topic_name,
                "side": side_label,
                "target_ns": target_ns,
                "matched_ns": best["matched_ns"],
                "diff_ns": best["diff_ns"],
                "topic_message_count": topic_message_counts[topic_name],
                "saved_paths": save_result["saved_paths"],
                "preview_path": save_result["preview_path"],
                "preview_error": save_result["preview_error"],
                "encoding": img_info.get("encoding"),
                "width": img_info.get("width"),
                "height": img_info.get("height"),
            }
        )

    if not results:
        raise ValueError(f"在 {mcap_source_label} 中未提取到任何 topic={topics} 的图片")

    return {
        "mcap_dir": mcap_source_label,
        "mcap_source_count": len(mcap_files),
        "target_ns": target_ns,
        "topics": topics,
        "output_dir": str(output_root),
        "search_window_ns": search_window_ns,
        "initial_search_window_ns": initial_search_window_ns if search_window_ns is not None else None,
        "effective_search_window_ns": effective_search_window_ns,
        "search_attempt_windows_ns": search_attempt_windows_ns,
        "used_full_scan_fallback": used_full_scan_fallback,
        "window_cache_hits": window_cache_hits,
        "scanned_messages": scanned_messages,
        "results": results,
        "errors": errors,
    }


def extract_image(mcap_path: str, topic_name: str, target_ns: int):
    """查找 topic 中 publish_time 最接近 target_ns 的图像消息并输出信息"""
    with open(mcap_path, "rb") as f:
        reader = make_reader(f)

        best_msg = None
        best_diff: Optional[int] = None
        msg_count = 0

        for schema, channel, message in reader.iter_messages():
            if channel.topic == topic_name:
                msg_count += 1
                diff = abs(message.publish_time - target_ns)
                if best_diff is None or diff < best_diff:
                    best_diff = diff
                    best_msg = message
        if best_msg is None:
            print(f"未在文件中找到 topic: {topic_name}")
            print(f"该 topic 下消息数: {msg_count}")
            return

        assert best_diff is not None
        publish_time = best_msg.publish_time
        assert publish_time is not None

        print("=" * 60)
        print(f"目标时间: {ns_to_str(target_ns)} ({target_ns} ns)")
        print(f"最近消息 publish_time: {ns_to_str(publish_time)} ({publish_time} ns)")
        diff_ms = float(best_diff) / 1e6
        print(f"时间偏差: {diff_ms:.3f} ms")
        print(f"数据大小: {len(best_msg.data)} bytes")
        print("=" * 60)

        # 尝试解析为 ROS2 Image
        img_info = parse_ros2_image(best_msg.data)

        if img_info:
            print("\n【图像信息】")
            print(f"  宽度 (width):    {img_info['width']}")
            print(f"  高度 (height):   {img_info['height']}")
            print(f"  编码 (encoding): {img_info['encoding']}")
            print(f"  步长 (step):     {img_info['step']}")
            print(f"  大端序:          {img_info['is_bigendian']}")
            print(f"  数据长度:        {img_info['data_length']} bytes")

            # 保存图像
            out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../extracted_images")
            os.makedirs(out_dir, exist_ok=True)
            base_name = f"{topic_name.replace('/', '_').strip('_')}_{target_ns}"
            save_result = save_image(img_info, out_dir, base_name)

            print(f"\n【输出文件】")
            for p in save_result["saved_paths"]:
                print(f"  {p}")

            preview_path = save_result["preview_path"]
            if preview_path:
                print(f"\n正在打开图片预览: {preview_path}")
                opened, open_error = open_image_file(preview_path)
                if opened:
                    print("图片已用系统默认查看器打开，可直接查看。")
                else:
                    print(f"打开图片失败，请手动打开该 PNG 文件: {open_error}")
            else:
                print("\n未生成可直接预览的 PNG 图片。")
                if save_result["preview_error"]:
                    print(f"原因: {save_result['preview_error']}")
                print("你仍可使用输出的 .bin + .json 文件进一步转换。")
        else:
            print("\n无法按 ROS2 sensor_msgs/Image 解析，可能为其他消息格式。")
            print(f"原始数据前 32 字节 (hex): {best_msg.data[:32].hex()}")

            # 保存原始数据供手动分析
            out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../extracted_images")
            os.makedirs(out_dir, exist_ok=True)
            raw_path = os.path.join(out_dir, f"{topic_name.replace('/', '_').strip('_')}_{target_ns}.bin")
            with open(raw_path, "wb") as fp:
                fp.write(best_msg.data)
            print(f"原始数据已保存: {raw_path}")



def main():
    print("MCAP 图像提取工具")
    print("-" * 60)

    mcap_file = clean_path(input("请输入 MCAP 文件路径: "))
    if not mcap_file or not os.path.exists(mcap_file):
        print(f"文件不存在或路径为空: {mcap_file}")
        sys.exit(1)

    topic_name = input("请输入图像 topic 名称 (如 /camera/color/image_raw): ").strip()
    if not topic_name:
        print("topic 名称不能为空")
        sys.exit(1)

    target_ns_str = input("请输入目标纳秒时间戳: ").strip()
    try:
        target_ns = int(target_ns_str)
    except ValueError:
        print("时间戳必须是整数（纳秒）")
        sys.exit(1)

    try:
        extract_image(mcap_file, topic_name, target_ns)
    except Exception as e:
        print(f"提取失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

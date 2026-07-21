#!/usr/bin/env python3
"""解析 parquet 文件并输出基础信息。"""

from __future__ import annotations

import argparse
import io
import json
import re
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


def _find_latest_parquet(parquet_dir: Path) -> Path:
    parquet_files = sorted(
        [p for p in parquet_dir.glob("*.parquet") if p.is_file()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not parquet_files:
        raise FileNotFoundError(f"目录中未找到 parquet 文件: {parquet_dir}")
    return parquet_files[0]


def _preview_rows(parquet_file: pq.ParquetFile, rows: int) -> list[dict[str, Any]]:
    table = parquet_file.read().slice(0, max(rows, 0))
    if table.num_rows == 0:
        return []

    columns = table.column_names
    pydict = table.to_pydict()
    preview: list[dict[str, Any]] = []
    for idx in range(table.num_rows):
        row = {col: pydict[col][idx] for col in columns}
        preview.append(row)
    return preview


def parse_parquet_file(parquet_path: str | Path, preview_rows: int = 5) -> dict[str, Any]:
    """解析 parquet，返回结构化结果，供外部调用。"""
    parquet_file_path = Path(parquet_path)
    if not parquet_file_path.exists():
        raise FileNotFoundError(f"parquet 文件不存在: {parquet_file_path}")

    parquet_file = pq.ParquetFile(parquet_file_path)
    schema = parquet_file.schema_arrow

    result = {
        "file": str(parquet_file_path),
        "num_rows": parquet_file.metadata.num_rows,
        "num_columns": len(schema.names),
        "num_row_groups": parquet_file.metadata.num_row_groups,
        "columns": [
            {
                "name": field.name,
                "type": str(field.type),
                "nullable": field.nullable,
            }
            for field in schema
        ],
        "preview_rows": _preview_rows(parquet_file, preview_rows),
    }
    return result


def _discover_image_columns(schema: pa.Schema) -> list[str]:
    """查找形如 struct<bytes: binary, path: string> 的图片列。"""
    image_columns: list[str] = []
    for field in schema:
        if not pa.types.is_struct(field.type):
            continue
        child_names = {child.name for child in field.type}
        if {"bytes", "path"}.issubset(child_names):
            image_columns.append(field.name)
    return image_columns


def _sanitize_filename_component(value: Any) -> str:
    text = str(value).strip()
    sanitized = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff_-]+", "_", text)
    return sanitized.strip("_") or "unknown"


def _infer_side_label(name: str) -> str:
    lower_name = name.lower()
    if "left" in lower_name or "_l" in lower_name:
        return "left"
    if "right" in lower_name or "_r" in lower_name:
        return "right"
    return _sanitize_filename_component(lower_name)


def _to_int_ns(raw_value: Any) -> int | None:
    if raw_value is None:
        return None
    try:
        return int(raw_value)
    except (TypeError, ValueError):
        return None


def _save_preview_image(image_bytes: bytes, output_png_path: Path) -> tuple[Path, str]:
    """优先将 bytes 解析为可展示 PNG；失败则退化为 bin。"""
    try:
        from PIL import Image

        with Image.open(io.BytesIO(image_bytes)) as img:
            output_png_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(output_png_path, format="PNG")
        return output_png_path, "direct_image_bytes"
    except Exception:
        pass

    try:
        from common.extract_mcap_image import decode_image_for_preview, parse_ros2_image

        image_info = parse_ros2_image(image_bytes)
        if image_info:
            output_png_path.parent.mkdir(parents=True, exist_ok=True)
            decode_image_for_preview(image_info).save(output_png_path, format="PNG")
            return output_png_path, "ros2_image_message"
    except Exception:
        pass

    bin_path = output_png_path.with_suffix(".bin")
    bin_path.parent.mkdir(parents=True, exist_ok=True)
    bin_path.write_bytes(image_bytes)
    return bin_path, "raw_bin_fallback"


def extract_nearest_parquet_images(
    parquet_path: str | Path,
    target_ns: int,
    output_dir: str | Path,
    image_columns: list[str] | None = None,
    name_prefix: str | None = None,
    extra_name_parts: list[str] | None = None,
) -> dict[str, Any]:
    """按目标纳秒时间在 parquet 中寻找最近帧并导出图片。"""
    parquet_file_path = Path(parquet_path)
    if not parquet_file_path.exists():
        raise FileNotFoundError(f"parquet 文件不存在: {parquet_file_path}")

    parquet_file = pq.ParquetFile(parquet_file_path)
    schema = parquet_file.schema_arrow
    all_column_names = set(schema.names)

    discovered_image_columns = _discover_image_columns(schema)
    if image_columns is None:
        selected_image_columns = discovered_image_columns
    else:
        selected_image_columns = [name for name in image_columns if name in discovered_image_columns]

    if not selected_image_columns:
        raise ValueError("parquet 中未找到可用图片列（期望 struct<bytes, path>）")

    timestamp_column = "original_timestamp_ns" if "original_timestamp_ns" in all_column_names else None
    if timestamp_column is None:
        raise ValueError("parquet 缺少 original_timestamp_ns 列，无法做纳秒时间匹配")

    read_columns = [timestamp_column, *selected_image_columns]
    table = parquet_file.read(columns=read_columns)
    rows = table.to_pylist()

    target_ns = int(target_ns)
    nearest_index = None
    nearest_timestamp_ns = None
    nearest_diff_ns = None

    for index, row in enumerate(rows):
        row_ns = _to_int_ns(row.get(timestamp_column))
        if row_ns is None:
            continue
        diff_ns = abs(row_ns - target_ns)
        if nearest_diff_ns is None or diff_ns < nearest_diff_ns:
            nearest_index = index
            nearest_timestamp_ns = row_ns
            nearest_diff_ns = diff_ns

    if nearest_index is None or nearest_timestamp_ns is None or nearest_diff_ns is None:
        raise ValueError("parquet 中没有可用于时间匹配的 original_timestamp_ns 数据")

    nearest_row = rows[nearest_index]
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    saved_files: list[dict[str, Any]] = []
    for column_name in selected_image_columns:
        image_struct = nearest_row.get(column_name)
        if not isinstance(image_struct, dict):
            continue

        image_bytes = image_struct.get("bytes")
        image_path_in_row = image_struct.get("path")
        if isinstance(image_bytes, memoryview):
            image_bytes = image_bytes.tobytes()
        elif isinstance(image_bytes, bytearray):
            image_bytes = bytes(image_bytes)

        if not isinstance(image_bytes, (bytes, bytearray)):
            continue

        side_label = _infer_side_label(column_name)
        name_parts = []
        if name_prefix:
            name_parts.append(_sanitize_filename_component(name_prefix))
        name_parts.extend(["parquet", side_label, f"target_{target_ns}", f"matched_{nearest_timestamp_ns}"])
        for extra in extra_name_parts or []:
            if extra:
                name_parts.append(_sanitize_filename_component(extra))
        output_png = output_root / f"{'_'.join(name_parts)}.png"
        saved_path, decode_mode = _save_preview_image(bytes(image_bytes), output_png)
        saved_files.append(
            {
                "column": column_name,
                "side": side_label,
                "row_path": image_path_in_row,
                "saved_path": str(saved_path),
                "decode_mode": decode_mode,
                "bytes_length": len(image_bytes),
            }
        )

    if not saved_files:
        raise ValueError("最近时间行未提取到有效图片 bytes")

    return {
        "parquet_file": str(parquet_file_path),
        "target_ns": target_ns,
        "matched_row_index": nearest_index,
        "matched_timestamp_ns": nearest_timestamp_ns,
        "diff_ns": nearest_diff_ns,
        "used_nearest": nearest_diff_ns != 0,
        "image_columns": selected_image_columns,
        "output_dir": str(output_root),
        "saved_files": saved_files,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="解析 parquet 文件")
    parser.add_argument(
        "--file",
        dest="file",
        default=None,
        help="parquet 文件路径；不传则自动取 parquet_file 目录最新文件",
    )
    parser.add_argument(
        "--rows",
        dest="rows",
        type=int,
        default=5,
        help="预览行数，默认5",
    )
    parser.add_argument(
        "--dir",
        dest="parquet_dir",
        default=None,
        help="自动查找 parquet 时使用的目录，默认项目根目录/parquet_file",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    parquet_dir = Path(args.parquet_dir) if args.parquet_dir else (project_root / "parquet_file")

    parquet_path = Path(args.file) if args.file else _find_latest_parquet(parquet_dir)
    print(f"[parquet] 使用文件: {parquet_path}")

    result = parse_parquet_file(parquet_path=parquet_path, preview_rows=args.rows)
    result_text = json.dumps(result, ensure_ascii=False, indent=2, default=str)

    analysis_file = project_root / "parquet_file" / "parquet_analysis.txt"
    analysis_file.parent.mkdir(parents=True, exist_ok=True)
    analysis_file.write_text(result_text, encoding="utf-8")

    # print(result_text)
    print(f"[parquet] 解析结果已写入: {analysis_file}")


if __name__ == "__main__":
    main()


"""统一打开本地或远端的单个/多个 Parquet 数据源。"""

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence

import pyarrow.parquet as pq


_EPISODE_INDEX_CACHE: dict[tuple[str, ...], list[dict[str, Any]]] = {}
_IMAGE_COLUMN_METADATA_CACHE: dict[str, dict[str, Any]] = {}


def parquet_source_name(source: Any) -> str:
    return str(source)


def normalize_parquet_sources(sources: Any) -> list[Any]:
    if isinstance(sources, (str, Path)) or callable(getattr(sources, "open", None)):
        normalized = [sources]
    elif isinstance(sources, Sequence):
        normalized = list(sources)
    else:
        raise TypeError(f"不支持的 parquet 数据源: {sources!r}")
    if not normalized:
        raise ValueError("parquet 数据源不能为空")
    return normalized


@contextmanager
def open_parquet_source(source: Any) -> Iterator[Any]:
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(f"parquet 文件不存在: {path}")
        with path.open("rb") as stream:
            yield stream
        return

    opener = getattr(source, "open", None)
    if not callable(opener):
        raise TypeError(f"parquet 数据源不支持 open(): {source!r}")
    stream = opener()
    try:
        if not stream.seekable():
            raise ValueError(f"parquet 数据源必须支持 seek: {source}")
        yield stream
    finally:
        stream.close()


@contextmanager
def open_parquet_file(
    source: Any,
    *,
    pre_buffer: bool = False,
) -> Iterator[pq.ParquetFile]:
    """Open a Parquet source, optionally coalescing remote column reads."""
    with open_parquet_source(source) as stream:
        parquet_file = pq.ParquetFile(stream, pre_buffer=pre_buffer)
        try:
            yield parquet_file
        finally:
            parquet_file.close(force=True)


def _collect_image_column_metadata(parquet_file: pq.ParquetFile) -> dict[str, Any]:
    leaf_indices: dict[str, dict[str, int]] = {}
    for index in range(parquet_file.metadata.num_columns):
        path = parquet_file.metadata.schema.column(index).path
        if "." not in path:
            continue
        column, suffix = path.rsplit(".", 1)
        if suffix in {"bytes", "path"}:
            leaf_indices.setdefault(column, {})[suffix] = index

    columns: dict[str, Any] = {}
    for column, suffix_indices in leaf_indices.items():
        if set(suffix_indices) != {"bytes", "path"}:
            continue
        result = {
            "row_count": parquet_file.metadata.num_rows,
            "bytes_value_count": 0,
            "path_value_count": 0,
            "bytes_null_count": 0,
            "path_null_count": 0,
            "metadata_complete": True,
        }
        for suffix, leaf_index in suffix_indices.items():
            for row_group_index in range(parquet_file.metadata.num_row_groups):
                metadata = parquet_file.metadata.row_group(row_group_index).column(leaf_index)
                result[f"{suffix}_value_count"] += int(metadata.num_values)
                statistics = metadata.statistics
                if statistics is None or statistics.null_count is None:
                    result["metadata_complete"] = False
                else:
                    result[f"{suffix}_null_count"] += int(statistics.null_count)
        columns[column] = result
    return {
        "row_count": parquet_file.metadata.num_rows,
        "columns": columns,
    }


def get_parquet_image_column_metadata(source: Any) -> dict[str, Any]:
    """返回图片叶子列统计，并复用 Episode 索引阶段已读取的 footer。"""
    source_name = parquet_source_name(source)
    cached = _IMAGE_COLUMN_METADATA_CACHE.get(source_name)
    if cached is not None:
        return cached
    with open_parquet_file(source) as parquet_file:
        result = _collect_image_column_metadata(parquet_file)
    _IMAGE_COLUMN_METADATA_CACHE[source_name] = result
    return result


def _parse_l1_definition(raw_hierarchy: Any) -> tuple[str, str, int, int] | None:
    try:
        hierarchy = json.loads(raw_hierarchy) if isinstance(raw_hierarchy, str) else raw_hierarchy
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(hierarchy, list):
        return None
    for item in hierarchy:
        if not isinstance(item, dict):
            continue
        try:
            level = int(item.get("level"))
        except (TypeError, ValueError):
            continue
        if level != 1:
            continue
        segment_id = str(item.get("id") or item.get("segmentId") or "").strip()
        name = str(item.get("name") or item.get("description") or item.get("prompt") or "").strip()
        start_value = item.get("start_timestamp_ns", item.get("startTimeNs"))
        end_value = item.get("end_timestamp_ns", item.get("endTimeNs"))
        try:
            return segment_id, name, int(start_value), int(end_value)
        except (TypeError, ValueError):
            return None
    return None


def build_parquet_episode_index(sources: Any) -> list[dict[str, Any]]:
    """读取轻量标注列，为每个单 Episode Parquet 建立 L1 映射。"""
    normalized_sources = normalize_parquet_sources(sources)
    source_names = tuple(parquet_source_name(source) for source in normalized_sources)
    cached = _EPISODE_INDEX_CACHE.get(source_names)
    if cached is not None:
        sources_by_name = {
            parquet_source_name(source): source for source in normalized_sources
        }
        return [
            {**item, "source": sources_by_name[item["source_name"]]}
            for item in cached
        ]

    records: list[dict[str, Any]] = []
    cache_records: list[dict[str, Any]] = []
    required_columns = [
        "episode_id",
        "episode_index",
        "annotation.hierarchy_json",
    ]
    for source in normalized_sources:
        source_name = parquet_source_name(source)
        with open_parquet_file(source, pre_buffer=True) as parquet_file:
            _IMAGE_COLUMN_METADATA_CACHE[source_name] = _collect_image_column_metadata(
                parquet_file
            )
            missing_columns = [
                column for column in required_columns
                if column not in parquet_file.schema_arrow.names
            ]
            if missing_columns:
                raise ValueError(
                    f"Parquet Episode 索引缺少列 {missing_columns}: {source_name}"
                )
            table = parquet_file.read(columns=required_columns, use_threads=False)

        episode_identities: set[tuple[str, int]] = set()
        l1_definitions: set[tuple[str, str, int, int]] = set()
        for row in table.to_pylist():
            episode_id = str(row.get("episode_id") or "").strip()
            try:
                episode_index = int(row.get("episode_index"))
            except (TypeError, ValueError):
                raise ValueError(
                    f"Parquet 包含无效 episode_index: {source_name}"
                ) from None
            if not episode_id:
                raise ValueError(f"Parquet 包含空 episode_id: {source_name}")
            episode_identities.add((episode_id, episode_index))
            l1_definition = _parse_l1_definition(row.get("annotation.hierarchy_json"))
            if l1_definition is not None:
                l1_definitions.add(l1_definition)

        if len(episode_identities) != 1:
            raise ValueError(
                f"Parquet 文件必须只包含一个 Episode: {source_name}, "
                f"实际={sorted(episode_identities)}"
            )
        if len(l1_definitions) != 1:
            raise ValueError(
                f"Parquet 文件必须唯一对应一个 L1: {source_name}, "
                f"实际={sorted(l1_definitions)}"
            )
        episode_id, episode_index = next(iter(episode_identities))
        l1_id, l1_name, l1_start_ns, l1_end_ns = next(iter(l1_definitions))
        cached_record = {
            "source_name": source_name,
            "episode_id": episode_id,
            "episode_index": episode_index,
            "l1_id": l1_id,
            "l1_name": l1_name,
            "l1_start_ns": l1_start_ns,
            "l1_end_ns": l1_end_ns,
        }
        cache_records.append(cached_record)
        records.append({**cached_record, "source": source})

    _EPISODE_INDEX_CACHE[source_names] = cache_records
    return records


def resolve_unique_parquet_episode_source(
    sources: Any,
    episode_start_ns: int,
    episode_end_ns: int,
    *,
    expected_l1_id: str | None = None,
) -> dict[str, Any]:
    """按 L1 定义解析唯一 Parquet Episode，禁止跨 Episode 猜测最近行。"""
    episode_start_ns = int(episode_start_ns)
    episode_end_ns = int(episode_end_ns)
    expected_l1_id = str(expected_l1_id or "").strip() or None
    matches = [
        item
        for item in build_parquet_episode_index(sources)
        if item["l1_start_ns"] == episode_start_ns
        and item["l1_end_ns"] == episode_end_ns
        and (expected_l1_id is None or item["l1_id"] == expected_l1_id)
    ]
    if len(matches) != 1:
        candidates = [
            {
                key: item[key]
                for key in (
                    "source_name",
                    "episode_id",
                    "episode_index",
                    "l1_id",
                    "l1_start_ns",
                    "l1_end_ns",
                )
            }
            for item in build_parquet_episode_index(sources)
        ]
        raise ValueError(
            "L1 无法唯一映射到 Parquet Episode: "
            f"l1_id={expected_l1_id!r}, range=[{episode_start_ns}, {episode_end_ns}], "
            f"matches={len(matches)}, candidates={candidates}"
        )
    return matches[0]


def find_nearest_parquet_row(
    sources: Any,
    target_ns: int,
    timestamp_column: str = "original_timestamp_ns",
    *,
    episode_start_ns: int | None = None,
    episode_end_ns: int | None = None,
    expected_l1_id: str | None = None,
) -> dict[str, Any]:
    """仅读取时间戳列，在指定 Episode 内定位最近行。"""
    return find_nearest_parquet_rows(
        sources,
        [target_ns],
        timestamp_column=timestamp_column,
        episode_start_ns=episode_start_ns,
        episode_end_ns=episode_end_ns,
        expected_l1_id=expected_l1_id,
    )[0]


def find_nearest_parquet_rows(
    sources: Any,
    target_ns_values: Sequence[int],
    timestamp_column: str = "original_timestamp_ns",
    *,
    episode_start_ns: int | None = None,
    episode_end_ns: int | None = None,
    expected_l1_id: str | None = None,
) -> list[dict[str, Any]]:
    """一次扫描指定 Episode 的时间戳列，为多个目标时间定位最近行。"""
    targets = [int(target_ns) for target_ns in target_ns_values]
    if not targets:
        raise ValueError("target_ns_values 不能为空")
    best_matches: list[dict[str, Any] | None] = [None] * len(targets)

    if (episode_start_ns is None) != (episode_end_ns is None):
        raise ValueError("episode_start_ns 和 episode_end_ns 必须同时提供")
    episode_record = None
    normalized_sources = normalize_parquet_sources(sources)
    if episode_start_ns is not None and episode_end_ns is not None:
        episode_record = resolve_unique_parquet_episode_source(
            normalized_sources,
            episode_start_ns,
            episode_end_ns,
            expected_l1_id=expected_l1_id,
        )
        normalized_sources = [episode_record["source"]]
    elif len(normalized_sources) > 1:
        raise ValueError(
            "多个 Parquet source 禁止全局最近行匹配，必须提供所属 L1 Episode 边界"
        )

    for source in normalized_sources:
        source_name = parquet_source_name(source)
        with open_parquet_file(source) as parquet_file:
            if timestamp_column not in parquet_file.schema_arrow.names:
                raise ValueError(
                    f"parquet 缺少 {timestamp_column} 列: {source_name}"
                )
            global_row_offset = 0
            for row_group_index in range(parquet_file.metadata.num_row_groups):
                table = parquet_file.read_row_group(
                    row_group_index,
                    columns=[timestamp_column],
                    use_threads=False,
                )
                values = table.column(timestamp_column).to_pylist()
                for row_in_group, raw_value in enumerate(values):
                    try:
                        row_ns = int(raw_value)
                    except (TypeError, ValueError):
                        continue
                    for target_index, target_ns in enumerate(targets):
                        diff_ns = abs(row_ns - target_ns)
                        best = best_matches[target_index]
                        if best is None or diff_ns < best["diff_ns"]:
                            best_matches[target_index] = {
                                "source": source,
                                "source_name": source_name,
                                "row_group_index": row_group_index,
                                "row_in_group": row_in_group,
                                "row_index": global_row_offset + row_in_group,
                                "timestamp_ns": row_ns,
                                "diff_ns": diff_ns,
                                "episode_id": episode_record["episode_id"] if episode_record else None,
                                "episode_index": episode_record["episode_index"] if episode_record else None,
                                "l1_id": episode_record["l1_id"] if episode_record else None,
                            }
                global_row_offset += len(values)

    if any(match is None for match in best_matches):
        raise ValueError(
            f"parquet 中没有可用于时间匹配的 {timestamp_column} 数据"
        )
    return [match for match in best_matches if match is not None]


def read_matched_parquet_row(
    match: dict[str, Any],
    columns: list[str],
) -> tuple[dict[str, Any], list[str]]:
    source = match["source"]
    with open_parquet_file(source) as parquet_file:
        available_columns = parquet_file.schema_arrow.names
        missing_columns = [name for name in columns if name not in available_columns]
        if missing_columns:
            raise ValueError(
                f"parquet 缺少列 {missing_columns}: {parquet_source_name(source)}"
            )
        table = parquet_file.read_row_group(
            int(match["row_group_index"]),
            columns=columns,
            use_threads=False,
        )
        row_in_group = int(match["row_in_group"])
        if row_in_group >= table.num_rows:
            raise IndexError(
                f"parquet row group 行号越界: {row_in_group} >= {table.num_rows}"
            )
        return table.slice(row_in_group, 1).to_pylist()[0], available_columns

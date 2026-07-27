"""统一打开本地或远端的单个/多个 Parquet 数据源。"""

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence

import pyarrow.parquet as pq


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


def find_nearest_parquet_row(
    sources: Any,
    target_ns: int,
    timestamp_column: str = "original_timestamp_ns",
) -> dict[str, Any]:
    """仅读取时间戳列，在所有 parquet/row group 中定位全局最近行。"""
    return find_nearest_parquet_rows(
        sources,
        [target_ns],
        timestamp_column=timestamp_column,
    )[0]


def find_nearest_parquet_rows(
    sources: Any,
    target_ns_values: Sequence[int],
    timestamp_column: str = "original_timestamp_ns",
) -> list[dict[str, Any]]:
    """一次扫描时间戳列，为多个目标时间定位全局最近行。"""
    targets = [int(target_ns) for target_ns in target_ns_values]
    if not targets:
        raise ValueError("target_ns_values 不能为空")
    best_matches: list[dict[str, Any] | None] = [None] * len(targets)

    for source in normalize_parquet_sources(sources):
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

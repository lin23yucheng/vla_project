"""统一打开本地或远端 MCAP 数据源。"""

from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path
from threading import RLock
from typing import Any, Iterator, Sequence

from mcap.reader import make_reader


DEFAULT_INITIAL_SEARCH_WINDOW_NS = 100_000_000
DEFAULT_MCAP_WINDOW_CACHE_BYTES = 256 * 1024 * 1024
_mcap_window_cache: OrderedDict[tuple[Any, ...], dict[str, Any]] = OrderedDict()
_mcap_window_cache_bytes = 0
_mcap_window_cache_lock = RLock()


def mcap_source_name(source: Any) -> str:
    return str(source)


@contextmanager
def open_mcap_source(source: Any) -> Iterator[Any]:
    """以二进制 seekable stream 打开 MCAP 数据源。"""
    if isinstance(source, (str, Path)):
        with Path(source).open("rb") as stream:
            yield stream
        return

    opener = getattr(source, "open", None)
    if not callable(opener):
        raise TypeError(f"MCAP 数据源不支持 open(): {source!r}")
    stream = opener()
    try:
        if not stream.seekable():
            raise ValueError(f"MCAP 数据源必须支持 seek: {mcap_source_name(source)}")
        yield stream
    finally:
        stream.close()


def local_mcap_sources(mcap_dir: str | Path) -> tuple[list[Path], str]:
    root = Path(mcap_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"mcap 目录不存在: {root}")
    sources = sorted(path for path in root.glob("*.mcap") if path.is_file())
    if not sources:
        raise FileNotFoundError(f"mcap 目录下未找到 .mcap 文件: {root}")
    return sources, str(root)


def select_mcap_sources_for_window(
    sources: Sequence[Any],
    start_time: int | None,
    end_time: int | None,
) -> list[Any]:
    """利用远端源预读的 MCAP 时间范围跳过无关对象。"""
    selected = []
    for source in sources:
        source_start = getattr(source, "message_start_time_ns", None)
        source_end = getattr(source, "message_end_time_ns", None)
        if source_start is None or source_end is None:
            selected.append(source)
            continue
        if start_time is not None and source_end < start_time:
            continue
        if end_time is not None and source_start >= end_time:
            continue
        selected.append(source)
    return selected


def adaptive_search_windows(
    max_window_ns: int,
    initial_window_ns: int = DEFAULT_INITIAL_SEARCH_WINDOW_NS,
) -> list[int]:
    """返回从小到大、以最大窗口结束的对称时间搜索窗口。"""
    if max_window_ns < 0:
        raise ValueError("max_window_ns 不能小于 0")
    if initial_window_ns <= 0:
        raise ValueError("initial_window_ns 必须大于 0")
    if max_window_ns == 0:
        return [0]

    windows = [min(initial_window_ns, max_window_ns)]
    while windows[-1] < max_window_ns:
        windows.append(min(windows[-1] * 2, max_window_ns))
    return windows


def _mcap_source_fingerprint(source: Any) -> tuple[Any, ...]:
    return (
        id(source),
        mcap_source_name(source),
        getattr(source, "size_bytes", None),
        getattr(source, "message_start_time_ns", None),
        getattr(source, "message_end_time_ns", None),
    )


def _get_cached_mcap_window(
    base_key: tuple[Any, ...],
    topics: frozenset[str],
) -> list[dict[str, Any]] | None:
    with _mcap_window_cache_lock:
        for cache_key in reversed(list(_mcap_window_cache)):
            entry = _mcap_window_cache[cache_key]
            if cache_key[0] != base_key or not topics.issubset(entry["topics"]):
                continue
            _mcap_window_cache.move_to_end(cache_key)
            return [
                record
                for record in entry["records"]
                if record["channel"].topic in topics
            ]
    return None


def _put_cached_mcap_window(
    base_key: tuple[Any, ...],
    topics: frozenset[str],
    records: list[dict[str, Any]],
    max_cache_bytes: int,
) -> None:
    global _mcap_window_cache_bytes
    approx_bytes = sum(len(record["message"].data) + 256 for record in records)
    if approx_bytes > max_cache_bytes:
        return

    cache_key = (base_key, topics)
    with _mcap_window_cache_lock:
        previous = _mcap_window_cache.pop(cache_key, None)
        if previous is not None:
            _mcap_window_cache_bytes -= previous["approx_bytes"]
        _mcap_window_cache[cache_key] = {
            "topics": topics,
            "records": records,
            "approx_bytes": approx_bytes,
        }
        _mcap_window_cache_bytes += approx_bytes
        while _mcap_window_cache and _mcap_window_cache_bytes > max_cache_bytes:
            _, evicted = _mcap_window_cache.popitem(last=False)
            _mcap_window_cache_bytes -= evicted["approx_bytes"]


def clear_mcap_window_cache() -> None:
    """清空进程内 MCAP 有限窗口缓存，主要供任务边界和测试使用。"""
    global _mcap_window_cache_bytes
    with _mcap_window_cache_lock:
        _mcap_window_cache.clear()
        _mcap_window_cache_bytes = 0


def read_mcap_window_messages(
    mcap_files: Sequence[Any],
    topics: Sequence[str],
    start_time: int,
    end_time: int,
    *,
    require_chunk_indexes: bool = False,
    use_cache: bool = True,
    max_cache_bytes: int = DEFAULT_MCAP_WINDOW_CACHE_BYTES,
) -> tuple[list[dict[str, Any]], bool]:
    """读取有限时间窗口消息，并在同一进程内复用相同或更完整的 topic 集合。"""
    if end_time <= start_time:
        raise ValueError("MCAP 窗口 end_time 必须大于 start_time")
    normalized_topics = list(dict.fromkeys(topics))
    if not normalized_topics:
        raise ValueError("MCAP 窗口 topics 不能为空")
    if max_cache_bytes <= 0:
        raise ValueError("max_cache_bytes 必须大于 0")

    selected_sources = select_mcap_sources_for_window(mcap_files, start_time, end_time)
    base_key = (
        tuple(_mcap_source_fingerprint(source) for source in selected_sources),
        int(start_time),
        int(end_time),
        bool(require_chunk_indexes),
    )
    topic_set = frozenset(normalized_topics)
    if use_cache:
        cached = _get_cached_mcap_window(base_key, topic_set)
        if cached is not None:
            return cached, True

    records: list[dict[str, Any]] = []
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
            for schema, channel, message in reader.iter_messages(
                topics=normalized_topics,
                start_time=start_time,
                end_time=end_time,
            ):
                records.append(
                    {
                        "mcap_file": mcap_source_name(mcap_file),
                        "schema": schema,
                        "channel": channel,
                        "message": message,
                    }
                )

    if use_cache:
        _put_cached_mcap_window(base_key, topic_set, records, max_cache_bytes)
    return records, False

"""统一打开本地或远端 MCAP 数据源。"""

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence


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

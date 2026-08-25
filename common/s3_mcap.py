"""通过 S3 Range 请求提供可随机访问的远端 MCAP 数据源。"""

import io
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

from mcap.reader import make_reader


DEFAULT_READ_AHEAD_BYTES = 1024 * 1024
DEFAULT_MAX_RANGE_REQUEST_BYTES = 1024 * 1024 * 1024


@dataclass(frozen=True)
class S3McapConfig:
    endpoint_url: str
    access_key: str
    secret_key: str = field(repr=False)
    bucket: str = "vla-label-studio"
    prefix: str = ""
    region: str | None = None
    read_ahead_bytes: int = DEFAULT_READ_AHEAD_BYTES
    max_range_request_bytes: int = DEFAULT_MAX_RANGE_REQUEST_BYTES

    def __post_init__(self) -> None:
        if not self.endpoint_url.strip():
            raise ValueError("S3 endpoint_url 不能为空")
        if not self.access_key.strip():
            raise ValueError("S3 access_key 不能为空")
        if not self.secret_key:
            raise ValueError("S3 secret_key 不能为空")
        if not self.bucket.strip():
            raise ValueError("S3 bucket 不能为空")
        if self.read_ahead_bytes <= 0:
            raise ValueError("S3 read_ahead_bytes 必须大于 0")
        if self.max_range_request_bytes <= 0:
            raise ValueError("S3 max_range_request_bytes 必须大于 0")
        if self.read_ahead_bytes > self.max_range_request_bytes:
            raise ValueError(
                "S3 read_ahead_bytes 不能大于 max_range_request_bytes"
            )

    @property
    def normalized_prefix(self) -> str:
        prefix = self.prefix.strip().strip("/")
        return f"{prefix}/" if prefix else ""


class S3RangeReader(io.RawIOBase):
    """将 S3 对象映射成只读、可 seek 的文件接口，不落地完整对象。"""

    def __init__(
        self,
        client: Any,
        bucket: str,
        object_name: str,
        object_size: int,
        read_ahead_bytes: int = DEFAULT_READ_AHEAD_BYTES,
        max_range_request_bytes: int = DEFAULT_MAX_RANGE_REQUEST_BYTES,
    ) -> None:
        super().__init__()
        if object_size < 0:
            raise ValueError("S3 对象大小不能小于 0")
        self._client = client
        self._bucket = bucket
        self._object_name = object_name
        self._object_size = int(object_size)
        self._read_ahead_bytes = int(read_ahead_bytes)
        self._max_range_request_bytes = int(max_range_request_bytes)
        self._position = 0
        self._cache_start = 0
        self._cache = b""
        self.range_request_count = 0
        self.range_bytes_read = 0

    @property
    def name(self) -> str:
        return f"s3://{self._bucket}/{self._object_name}"

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def writable(self) -> bool:
        return False

    def _ensure_open(self) -> None:
        if self.closed:
            raise ValueError("I/O operation on closed S3 Range reader")

    def tell(self) -> int:
        self._ensure_open()
        return self._position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        self._ensure_open()
        if whence == io.SEEK_SET:
            position = int(offset)
        elif whence == io.SEEK_CUR:
            position = self._position + int(offset)
        elif whence == io.SEEK_END:
            position = self._object_size + int(offset)
        else:
            raise ValueError(f"不支持的 whence: {whence}")
        if position < 0:
            raise ValueError(f"不能 seek 到负数位置: {position}")
        self._position = position
        return position

    def _request_range(self, offset: int, length: int) -> bytes:
        if length > self._max_range_request_bytes:
            raise OSError(
                f"拒绝单次读取 {length} 字节的远端 MCAP 范围，"
                f"上限为 {self._max_range_request_bytes} 字节；"
                "请确认 MCAP 使用了合理大小的 indexed chunks"
            )
        response = self._client.get_object(
            self._bucket,
            self._object_name,
            offset=offset,
            length=length,
        )
        try:
            data = response.read()
        finally:
            response.close()
            response.release_conn()
        if len(data) != length:
            raise OSError(
                f"S3 Range 返回长度不完整: object={self.name}, "
                f"offset={offset}, expected={length}, actual={len(data)}"
            )
        self.range_request_count += 1
        self.range_bytes_read += len(data)
        return data

    def read(self, size: int = -1) -> bytes:
        self._ensure_open()
        if self._position >= self._object_size:
            return b""
        remaining = self._object_size - self._position
        if size is None or size < 0:
            size = remaining
        size = min(int(size), remaining)
        if size <= 0:
            return b""

        cache_end = self._cache_start + len(self._cache)
        request_end = self._position + size
        if self._cache_start <= self._position and request_end <= cache_end:
            cache_offset = self._position - self._cache_start
            data = self._cache[cache_offset:cache_offset + size]
            self._position += len(data)
            return data

        fetch_length = min(
            remaining,
            max(size, self._read_ahead_bytes),
        )
        data = self._request_range(self._position, fetch_length)
        if fetch_length <= self._read_ahead_bytes:
            self._cache_start = self._position
            self._cache = data
        else:
            self._cache = b""
        result = data[:size]
        self._position += len(result)
        return result

    def readinto(self, buffer: Any) -> int:
        data = self.read(len(buffer))
        buffer[:len(data)] = data
        return len(data)

    def close(self) -> None:
        self._cache = b""
        super().close()


@dataclass
class S3McapSource:
    client: Any = field(repr=False)
    bucket: str
    object_name: str
    size_bytes: int
    read_ahead_bytes: int = DEFAULT_READ_AHEAD_BYTES
    max_range_request_bytes: int = DEFAULT_MAX_RANGE_REQUEST_BYTES
    message_start_time_ns: int | None = None
    message_end_time_ns: int | None = None
    chunk_count: int | None = None
    image_start_time_ns: int | None = None
    image_end_time_ns: int | None = None

    def __str__(self) -> str:
        return f"s3://{self.bucket}/{self.object_name}"

    def open(self) -> io.BufferedReader:
        raw_stream = S3RangeReader(
            client=self.client,
            bucket=self.bucket,
            object_name=self.object_name,
            object_size=self.size_bytes,
            read_ahead_bytes=self.read_ahead_bytes,
            max_range_request_bytes=self.max_range_request_bytes,
        )
        # MCAP 会为 RawIOBase 临时创建 BufferedReader；临时对象释放时会提前
        # 关闭底层 Range 流。这里保持一个覆盖完整读取周期的稳定包装。
        return io.BufferedReader(raw_stream)

    def inspect_index(self) -> None:
        """仅通过 Range 请求读取 MCAP 摘要并记录对象覆盖的时间范围。"""
        with self.open() as stream:
            reader = make_reader(stream)
            summary = reader.get_summary()
        if summary is None or not summary.chunk_indexes:
            raise ValueError(
                f"远端 MCAP 缺少 chunk index，拒绝顺序读取完整对象: {self}"
            )
        self.message_start_time_ns = min(
            int(chunk.message_start_time) for chunk in summary.chunk_indexes
        )
        self.message_end_time_ns = max(
            int(chunk.message_end_time) for chunk in summary.chunk_indexes
        )
        self.chunk_count = len(summary.chunk_indexes)

    @staticmethod
    def _is_image_message(schema: Any, channel: Any) -> bool:
        schema_name = str(getattr(schema, "name", "") or "").lower()
        topic = str(getattr(channel, "topic", "") or "").lower()
        return (
            "sensor_msgs/msg/image" in schema_name
            or "sensor_msgs/msg/compressedimage" in schema_name
            or schema_name.endswith("/image")
            or schema_name.endswith("/compressedimage")
            or "/image" in topic
            or "camera" in topic and ("image" in topic or "rgb" in topic or "color" in topic)
        )

    def inspect_image_time_range(self) -> None:
        """利用 MCAP 索引仅读取图片 topic 的首尾消息时间戳。"""
        image_topics = set()
        with self.open() as stream:
            reader = make_reader(stream)
            summary = reader.get_summary()
            if summary is None:
                raise ValueError(f"远端 MCAP 缺少摘要: {self}")
            for channel in summary.channels.values():
                schema = summary.schemas.get(channel.schema_id)
                if self._is_image_message(schema, channel):
                    image_topics.add(channel.topic)
        if not image_topics:
            raise ValueError(f"远端 MCAP 未找到图片消息: {self}")

        with self.open() as stream:
            reader = make_reader(stream)
            first = next(reader.iter_messages(topics=image_topics), None)
        with self.open() as stream:
            reader = make_reader(stream)
            last = next(reader.iter_messages(topics=image_topics, reverse=True), None)
        if first is None or last is None:
            raise ValueError(f"远端 MCAP 图片 topic 没有可读取消息: {self}")
        first_message = first[2]
        last_message = last[2]
        self.image_start_time_ns = int(
            getattr(first_message, "publish_time", None)
            or getattr(first_message, "log_time")
        )
        self.image_end_time_ns = int(
            getattr(last_message, "publish_time", None)
            or getattr(last_message, "log_time")
        )


class S3McapStore:
    def __init__(self, config: S3McapConfig) -> None:
        self.config = config
        self._active_prefix = config.normalized_prefix
        endpoint, secure = self._parse_endpoint(config.endpoint_url)
        try:
            from minio import Minio
        except ImportError as exc:
            raise RuntimeError(
                "缺少 minio 依赖，请先安装 requirements.txt"
            ) from exc
        self.client = Minio(
            endpoint,
            access_key=config.access_key,
            secret_key=config.secret_key,
            secure=secure,
            region=config.region,
        )

    @staticmethod
    def _parse_endpoint(endpoint_url: str) -> tuple[str, bool]:
        parsed = urlparse(endpoint_url.strip())
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError(
                f"S3 endpoint 必须是完整的 http(s) API 地址: {endpoint_url!r}"
            )
        if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
            raise ValueError(
                f"S3 endpoint 不能包含登录路径或查询参数: {endpoint_url!r}"
            )
        return parsed.netloc, parsed.scheme == "https"

    @property
    def source_label(self) -> str:
        return f"s3://{self.config.bucket}/{self._active_prefix}"

    def resolve_unique_child_subdirectory(self, subdirectory: str) -> str:
        """在当前前缀的唯一一级目录下继续进入指定子目录。"""
        child_subdirectory = subdirectory.strip().strip("/")
        if not child_subdirectory:
            raise ValueError("S3 子目录名称不能为空")

        child_directories: set[str] = set()
        for item in self.client.list_objects(
            self.config.bucket,
            prefix=self._active_prefix,
            recursive=False,
        ):
            object_name = str(item.object_name or "")
            if not object_name.startswith(self._active_prefix):
                continue
            relative_name = object_name[len(self._active_prefix):]
            if not relative_name:
                continue
            is_directory = bool(getattr(item, "is_dir", False))
            if not is_directory and "/" not in relative_name:
                continue
            child_name = relative_name.split("/", 1)[0].strip()
            if child_name:
                child_directories.add(child_name)

        if len(child_directories) != 1:
            discovered = sorted(child_directories)
            raise ValueError(
                f"S3 路径 {self.source_label} 下应有且仅有一个一级文件夹，"
                f"实际发现 {len(discovered)} 个: {discovered}"
            )

        child_name = next(iter(child_directories))
        self._active_prefix = (
            f"{self._active_prefix}{child_name}/{child_subdirectory}/"
        )
        return self._active_prefix

    def list_indexed_mcap_sources(self, progress_callback=None) -> list[S3McapSource]:
        """列出并解析 MCAP；可通过回调报告逐文件进度。"""
        sources = []
        for item in self.client.list_objects(
            self.config.bucket,
            prefix=self._active_prefix,
            recursive=True,
        ):
            object_name = str(item.object_name or "")
            if not object_name.lower().endswith(".mcap"):
                continue
            if item.size is None or int(item.size) <= 0:
                raise ValueError(
                    f"S3 MCAP 对象大小无效: s3://{self.config.bucket}/{item.object_name}"
                )
            source = S3McapSource(
                client=self.client,
                bucket=self.config.bucket,
                object_name=object_name,
                size_bytes=int(item.size),
                read_ahead_bytes=self.config.read_ahead_bytes,
                max_range_request_bytes=self.config.max_range_request_bytes,
            )
            if progress_callback is not None:
                progress_callback("start", source, len(sources) + 1)
            source.inspect_image_time_range()
            if progress_callback is not None:
                progress_callback("done", source, len(sources) + 1)
            sources.append(source)
        sources.sort(key=lambda source: source.object_name)
        if not sources:
            raise FileNotFoundError(
                f"S3 路径下未找到 .mcap 对象: {self.source_label}"
            )
        return sources

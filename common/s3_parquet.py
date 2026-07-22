"""通过 S3 Range 请求提供可随机访问的远端 Parquet 数据源。"""

import io
from dataclasses import dataclass, field
from typing import Any

from common.s3_mcap import S3McapConfig, S3McapStore, S3RangeReader


@dataclass(frozen=True)
class S3ParquetSource:
    client: Any = field(repr=False)
    bucket: str
    object_name: str
    size_bytes: int
    read_ahead_bytes: int
    max_range_request_bytes: int

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
        return io.BufferedReader(raw_stream)


class S3ParquetStore:
    def __init__(self, config: S3McapConfig, prefix: str) -> None:
        self.config = config
        self.prefix = prefix.strip().strip("/") + "/"
        endpoint, secure = S3McapStore._parse_endpoint(config.endpoint_url)
        try:
            from minio import Minio
        except ImportError as exc:
            raise RuntimeError("缺少 minio 依赖，请先安装 requirements.txt") from exc
        self.client = Minio(
            endpoint,
            access_key=config.access_key,
            secret_key=config.secret_key,
            secure=secure,
            region=config.region,
        )

    @property
    def source_label(self) -> str:
        return f"s3://{self.config.bucket}/{self.prefix}"

    def list_parquet_sources(self) -> list[S3ParquetSource]:
        sources: list[S3ParquetSource] = []
        for item in self.client.list_objects(
            self.config.bucket,
            prefix=self.prefix,
            recursive=True,
        ):
            object_name = str(item.object_name or "")
            if not object_name.lower().endswith(".parquet"):
                continue
            if item.size is None or int(item.size) <= 0:
                raise ValueError(
                    f"S3 parquet 对象大小无效: "
                    f"s3://{self.config.bucket}/{object_name}"
                )
            sources.append(
                S3ParquetSource(
                    client=self.client,
                    bucket=self.config.bucket,
                    object_name=object_name,
                    size_bytes=int(item.size),
                    read_ahead_bytes=self.config.read_ahead_bytes,
                    max_range_request_bytes=self.config.max_range_request_bytes,
                )
            )

        sources.sort(key=lambda source: source.object_name)
        if not sources:
            raise FileNotFoundError(
                f"S3 路径下未找到 parquet 文件: {self.source_label}"
            )
        return sources

"""天机数据 V2 任务校验流程。"""

import configparser
import json
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any

import allure
import pytest

from api import all_api
from common import Assert
from common.client_factory import create_lazy_yixiu_client
from common.extract_mcap_fields import extract_robot_vectors_at_time
from common.extract_mcap_image import extract_global_nearest_image_from_mcap_sources
from common.extract_parquet_fields import (
    compare_robot_vectors,
    extract_parquet_robot_vectors_at_time,
)
from common.image_compare import compare_images
from common.parse_parquet_file import (
    compare_parquet_annotations,
    extract_nearest_parquet_images,
)
from common.s3_mcap import S3McapConfig, S3McapStore
from common.s3_parquet import S3ParquetStore

NANOSECONDS_PER_SECOND = 1_000_000_000
ROBOT_CONFIG_NAME = "天机构型.json"
BASELINE_CAMERA_NAME = "right_image"
BASELINE_CAMERA_KEY = "camera_bottom_right"
EXPECTED_SEGMENT_COUNTS = {"l1": 2, "l2": 4, "l3": 3}
EXPECTED_CAMERA_COUNT = 4
CONVERSION_POLL_INTERVAL_SECONDS = 10
CONVERSION_POLL_TIMEOUT_SECONDS = 15 * 60

# start/end 为 MCAP 绝对纳秒；存在跳帧时需同时填写接口返回的 episode 起止纳秒。
TASK_ANNOTATION_CONFIG = {
    "baseline_start_time_ns": 14792726535008,
    "baseline_end_time_ns": 15097130516310,
    "timeline_duration_sec": "304.337277241",
    "layers": [
        {
            "name": "L1 Episode",
            "type": "episode",
            "layerId": "l1",
            "category": "episode",
            "segments": [
                {
                    "key": "l1_01",
                    "description": "整理桌面",
                    "startTimeNs": 14939056887039,
                    "endTimeNs": 14986368887039,
                    "episodeStartTimeNs": 146297000000,
                    "episodeEndTimeNs": 193609000000,
                    "current_sequence": 4,
                },
                {
                    "key": "l1_02",
                    "description": "整理桌面",
                    "startTimeNs": 15041297239069,
                    "endTimeNs": 15069225239069,
                    "episodeStartTimeNs": 248504000000,
                    "episodeEndTimeNs": 276432000000,
                    "current_sequence": 6,
                },
            ],
        },
        {
            "name": "L2 Detail",
            "type": "detail",
            "layerId": "l2",
            "category": "detail",
            "segments": [
                {
                    "key": "l2_01",
                    "description": "右臂移动夹笔",
                    "startTimeNs": 14944854887039,
                    "endTimeNs": 14961012887039,
                    "episodeStartTimeNs": 152095000000,
                    "episodeEndTimeNs": 168253000000,
                    "current_sequence": 4,
                },
                {
                    "key": "l2_02",
                    "description": "右臂移动放笔",
                    "startTimeNs": 14961478887039,
                    "endTimeNs": 14969005887039,
                    "episodeStartTimeNs": 168719000000,
                    "episodeEndTimeNs": 176246000000,
                    "current_sequence": 4,
                },
                {
                    "key": "l2_03",
                    "description": "左臂夹笔",
                    "startTimeNs": 15042913239069,
                    "endTimeNs": 15046977239069,
                    "episodeStartTimeNs": 250120000000,
                    "episodeEndTimeNs": 254184000000,
                    "current_sequence": 6,
                },
                {
                    "key": "l2_04",
                    "description": "左臂放笔回位",
                    "startTimeNs": 15047720239069,
                    "endTimeNs": 15071751239069,
                    "episodeStartTimeNs": 254927000000,
                    "episodeEndTimeNs": 278958000000,
                    "current_sequence": 6,
                },
            ],
        },
        {
            "name": "L3 Detail",
            "type": "detail",
            "layerId": "l3",
            "category": "detail",
            "segments": [
                {
                    "key": "l3_01",
                    "description": "右臂夹住",
                    "startTimeNs": 14959023887039,
                    "endTimeNs": 14961174887039,
                    "episodeStartTimeNs": 166264000000,
                    "episodeEndTimeNs": 168415000000,
                    "current_sequence": 4,
                },
                {
                    "key": "l3_02",
                    "description": "右臂松开",
                    "startTimeNs": 14967688887039,
                    "endTimeNs": 14970636887039,
                    "episodeStartTimeNs": 174929000000,
                    "episodeEndTimeNs": 177877000000,
                    "current_sequence": 4,
                },
                {
                    "key": "l3_03",
                    "description": "左臂夹到放",
                    "startTimeNs": 15045620239069,
                    "endTimeNs": 15060575239069,
                    "episodeStartTimeNs": 252827000000,
                    "episodeEndTimeNs": 267782000000,
                    "current_sequence": 6,
                },
            ],
        },
    ],
}

assertions = Assert.Assertions()
_last_snowflake_13 = 0


def load_task_context() -> tuple[str, str, str, S3McapConfig]:
    """读取当前环境的任务号和天机 S3 MCAP 配置。"""
    project_root = Path(__file__).resolve().parent.parent
    config_path = project_root / "config" / "env_config.ini"
    config = configparser.ConfigParser()
    if not config.read(config_path, encoding="utf-8"):
        raise FileNotFoundError(f"未找到配置文件: {config_path}")

    execution_env = config.get("environment", "execution_env", fallback="").strip().lower()
    if execution_env not in {"dev", "fat", "prod"}:
        raise ValueError(f"execution_env 配置错误: {execution_env}，仅支持 dev、fat 或 prod")

    section = f"{execution_env}-vla"
    if not config.has_section(section):
        raise ValueError(f"配置文件缺少节: [{section}]")

    task_no = config.get(section, "tianji_task_no", fallback="").strip()
    if not task_no:
        raise ValueError(f"配置文件节 [{section}] 中缺少 tianji_task_no")

    endpoint_url = config.get(section, "s3_endpoint", fallback="").strip()
    access_key = (
        os.environ.get("VLA_S3_ACCESS_KEY")
        or config.get(section, "s3_access_key", fallback="").strip()
    )
    secret_key = (
        os.environ.get("VLA_S3_SECRET_KEY")
        or config.get(section, "s3_secret_key", fallback="")
    )
    bucket = config.get(section, "s3_bucket", fallback="").strip()

    s3_config = S3McapConfig(
        endpoint_url=endpoint_url,
        access_key=access_key,
        secret_key=secret_key,
        bucket=bucket,
        prefix=f"{task_no}/original-data",
        region=config.get(section, "s3_region", fallback="").strip() or None,
        read_ahead_bytes=config.getint(
            section,
            "s3_read_ahead_bytes",
            fallback=1024 * 1024,
        ),
        max_range_request_bytes=config.getint(
            section,
            "s3_max_range_request_bytes",
            fallback=1024 * 1024 * 1024,
        ),
    )
    return execution_env, section, task_no, s3_config


def load_robot_config() -> tuple[Path, dict[str, Any]]:
    """加载天机构型 JSON。"""
    config_path = Path(__file__).resolve().parent.parent / "Robot_Configuration" / ROBOT_CONFIG_NAME
    with config_path.open("r", encoding="utf-8") as file:
        robot_config = json.load(file)
    if not isinstance(robot_config, dict):
        raise ValueError(f"机器人构型文件根节点必须是对象: {config_path}")
    return config_path, robot_config


def resolve_robot_cameras(robot_config: dict[str, Any]) -> tuple[list[dict[str, str]], str]:
    """解析构型中的全部相机；天机的四路相机都会参与图片校验。"""
    raw_cameras = robot_config.get("cameras")
    if not isinstance(raw_cameras, list) or not raw_cameras:
        raise ValueError("天机构型中的 cameras 必须是非空列表")

    cameras: list[dict[str, str]] = []
    seen_names: set[str] = set()
    seen_keys: set[str] = set()
    seen_topics: set[str] = set()
    for index, raw_camera in enumerate(raw_cameras, start=1):
        if not isinstance(raw_camera, dict):
            raise ValueError(f"天机构型 cameras 第{index}项必须是对象")
        camera = {
            field: str(raw_camera.get(field, "")).strip()
            for field in ("key", "name", "group", "topic")
        }
        missing = [field for field, value in camera.items() if not value]
        if missing:
            raise ValueError(f"天机构型 cameras 第{index}项缺少字段: {missing}")
        if camera["name"] in seen_names:
            raise ValueError(f"天机构型相机 name 重复: {camera['name']}")
        if camera["key"] in seen_keys:
            raise ValueError(f"天机构型相机 key 重复: {camera['key']}")
        if camera["topic"] in seen_topics:
            raise ValueError(f"天机构型相机 topic 重复: {camera['topic']}")
        seen_names.add(camera["name"])
        seen_keys.add(camera["key"])
        seen_topics.add(camera["topic"])
        cameras.append(camera)

    if len(cameras) != EXPECTED_CAMERA_COUNT:
        raise ValueError(
            f"天机构型必须配置 {EXPECTED_CAMERA_COUNT} 路相机，实际为 {len(cameras)} 路"
        )

    main_time_topic = str(robot_config.get("main_time_topic", "")).strip()
    if not main_time_topic:
        raise ValueError("天机构型缺少有效的 main_time_topic")
    return cameras, main_time_topic


def iter_layer_segments():
    for layer in TASK_ANNOTATION_CONFIG.get("layers", []):
        for position, segment in enumerate(layer.get("segments", []), start=1):
            yield layer, position, segment


def get_segment_definition(segment_key: str) -> tuple[dict[str, Any], dict[str, Any]]:
    for layer, _, segment in iter_layer_segments():
        if segment.get("key") == segment_key:
            return layer, segment
    raise KeyError(f"未找到标注配置: {segment_key}")


def _parse_ns(value: Any, field_label: str) -> int:
    if isinstance(value, bool) or value in (None, ""):
        raise ValueError(f"{field_label} 尚未填写")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_label} 必须是整数纳秒值，当前为 {value!r}") from exc


def get_task_time_range_ns() -> tuple[int, int]:
    start_ns = _parse_ns(
        TASK_ANNOTATION_CONFIG.get("baseline_start_time_ns"),
        "baseline_start_time_ns",
    )
    end_ns = _parse_ns(
        TASK_ANNOTATION_CONFIG.get("baseline_end_time_ns"),
        "baseline_end_time_ns",
    )
    if start_ns >= end_ns:
        raise ValueError("baseline_start_time_ns 必须早于 baseline_end_time_ns")
    return start_ns, end_ns


def validate_annotation_config() -> None:
    """在调用接口前一次性报告待填项、数量和时间范围问题。"""
    failures: list[str] = []
    try:
        baseline_start_ns, baseline_end_ns = get_task_time_range_ns()
    except ValueError as exc:
        failures.append(str(exc))
        baseline_start_ns = baseline_end_ns = None

    raw_layers = TASK_ANNOTATION_CONFIG.get("layers")
    if not isinstance(raw_layers, list):
        failures.append("TASK_ANNOTATION_CONFIG.layers 必须是列表")
        raw_layers = []

    layers_by_id: dict[str, dict[str, Any]] = {}
    seen_keys: set[str] = set()
    for layer in raw_layers:
        if not isinstance(layer, dict):
            failures.append(f"标注层必须是对象: {layer!r}")
            continue
        layer_id = str(layer.get("layerId", "")).strip().lower()
        if layer_id in layers_by_id:
            failures.append(f"标注层重复: {layer_id}")
        layers_by_id[layer_id] = layer
        segments = layer.get("segments")
        if not isinstance(segments, list):
            failures.append(f"{layer_id}.segments 必须是列表")
            continue
        expected_count = EXPECTED_SEGMENT_COUNTS.get(layer_id)
        if expected_count is None:
            failures.append(f"存在未支持的标注层: {layer_id!r}")
        elif len(segments) != expected_count:
            failures.append(f"{layer_id} 标注数量应为 {expected_count}，实际为 {len(segments)}")

        for position, segment in enumerate(segments, start=1):
            if not isinstance(segment, dict):
                failures.append(f"{layer_id}第{position}段必须是对象")
                continue
            segment_key = str(segment.get("key", "")).strip()
            if not segment_key:
                failures.append(f"{layer_id}第{position}段缺少 key")
            elif segment_key in seen_keys:
                failures.append(f"标注 key 重复: {segment_key}")
            else:
                seen_keys.add(segment_key)

            description = str(segment.get("description", "")).strip()
            if not description or description.upper().startswith("TODO"):
                failures.append(f"{segment_key or f'{layer_id}第{position}段'} description 尚未填写")
            try:
                start_ns = _parse_ns(segment.get("startTimeNs"), f"{segment_key}.startTimeNs")
                end_ns = _parse_ns(segment.get("endTimeNs"), f"{segment_key}.endTimeNs")
            except ValueError as exc:
                failures.append(str(exc))
                continue
            if start_ns >= end_ns:
                failures.append(f"{segment_key} 的 startTimeNs 必须早于 endTimeNs")
                continue
            if (
                baseline_start_ns is not None
                and baseline_end_ns is not None
                and (start_ns < baseline_start_ns or end_ns > baseline_end_ns)
            ):
                failures.append(f"{segment_key} 的时间超出任务基准时间范围")

    for layer_id, expected_count in EXPECTED_SEGMENT_COUNTS.items():
        if layer_id not in layers_by_id:
            failures.append(f"缺少 {layer_id} 标注层（应包含 {expected_count} 段）")

    if failures:
        raise ValueError("天机标注配置未完成或不合法:\n- " + "\n- ".join(failures))


def build_annotation_time_fields(segment_key: str) -> dict[str, Any]:
    _, segment = get_segment_definition(segment_key)
    start_time_ns = _parse_ns(segment.get("startTimeNs"), f"{segment_key}.startTimeNs")
    end_time_ns = _parse_ns(segment.get("endTimeNs"), f"{segment_key}.endTimeNs")
    baseline_start_ns, baseline_end_ns = get_task_time_range_ns()
    if start_time_ns >= end_time_ns:
        raise ValueError(f"{segment_key} 的 startTimeNs 必须早于 endTimeNs")
    if start_time_ns < baseline_start_ns or end_time_ns > baseline_end_ns:
        raise ValueError(f"{segment_key} 的时间超出任务基准时间范围")

    configured_episode_start_ns = segment.get("episodeStartTimeNs")
    configured_episode_end_ns = segment.get("episodeEndTimeNs")
    has_explicit_episode_time = (
        configured_episode_start_ns not in (None, "")
        or configured_episode_end_ns not in (None, "")
    )
    if has_explicit_episode_time:
        episode_start_ns = _parse_ns(
            configured_episode_start_ns,
            f"{segment_key}.episodeStartTimeNs",
        )
        episode_end_ns = _parse_ns(
            configured_episode_end_ns,
            f"{segment_key}.episodeEndTimeNs",
        )
    else:
        episode_start_ns = start_time_ns - baseline_start_ns
        episode_end_ns = end_time_ns - baseline_start_ns
    if episode_start_ns < 0 or episode_start_ns >= episode_end_ns:
        raise ValueError(f"{segment_key} 的 episode 起止时间无效")
    episode_start_sec = episode_start_ns / NANOSECONDS_PER_SECOND
    episode_end_sec = episode_end_ns / NANOSECONDS_PER_SECOND

    def format_ns_as_seconds(value_ns: int) -> str:
        seconds, nanoseconds = divmod(value_ns, NANOSECONDS_PER_SECOND)
        return f"{seconds}.{nanoseconds:09d}"

    return {
        "startTimeNs": str(start_time_ns),
        "endTimeNs": str(end_time_ns),
        "startTimestampNs": str(start_time_ns),
        "endTimestampNs": str(end_time_ns),
        "episodeStartTimeNs": str(episode_start_ns),
        "episodeEndTimeNs": str(episode_end_ns),
        "episode_start_time": f"{episode_start_sec:.9f}",
        "episode_end_time": f"{episode_end_sec:.9f}",
        "timeline_start_sec": f"{episode_start_sec:.9f}",
        "timeline_end_sec": f"{episode_end_sec:.9f}",
        "startSec": episode_start_sec,
        "endSec": episode_end_sec,
        "start_time": format_ns_as_seconds(start_time_ns),
        "end_time": format_ns_as_seconds(end_time_ns),
    }


def build_annotation_text_fields(segment_key: str) -> dict[str, str]:
    _, segment = get_segment_definition(segment_key)
    description = str(segment.get("description", "")).strip()
    if not description or description.upper().startswith("TODO"):
        raise ValueError(f"{segment_key}.description 尚未填写")
    return {"description": description, "prompt": description}


def generate_snowflake_id(prefix: str) -> str:
    global _last_snowflake_13
    now_ms = int(time.time() * 1000)
    if now_ms <= _last_snowflake_13:
        now_ms = _last_snowflake_13 + 1
    _last_snowflake_13 = now_ms
    return f"{prefix}_{now_ms}"


def find_conversion_entry(conversion_list, task_id):
    for item in conversion_list or []:
        if str(item.get("task_id", "")).strip() == str(task_id):
            return item
    return None


global_client = create_lazy_yixiu_client()


@pytest.mark.annotation_v2
@allure.feature("场景：天机数据-V2格式信息校验")
class TestTianjiV2Verify:
    @classmethod
    def setup_class(cls):
        validate_annotation_config()
        cls.api_all = all_api.ApiAll(global_client)
        cls.execution_env, cls.section, cls.task_no, cls.s3_config = load_task_context()
        print(
            f"[前置] 正在发现S3上传目录: "
            f"s3://{cls.s3_config.bucket}/{cls.s3_config.normalized_prefix}",
            flush=True,
        )
        cls.s3_store = S3McapStore(cls.s3_config)
        cls.s3_store.resolve_unique_child_subdirectory("split")
        print(
            f"[前置] 已定位split目录，正在检查MCAP文件及chunk index: "
            f"{cls.s3_store.source_label}",
            flush=True,
        )
        cls.mcap_sources = cls.s3_store.list_indexed_mcap_sources()
        cls.mcap_source_label = cls.s3_store.source_label
        cls.robot_config_path, cls.robot_config = load_robot_config()
        cls.cameras, cls.main_time_topic = resolve_robot_cameras(cls.robot_config)
        cls.baseline_camera = next(
            (camera for camera in cls.cameras if camera["name"] == BASELINE_CAMERA_NAME),
            None,
        )
        if cls.baseline_camera is None:
            raise ValueError(
                f"天机构型中未找到 BASELINE_CAMERA_NAME={BASELINE_CAMERA_NAME!r} 对应相机"
            )

        total_mcap_size = sum(source.size_bytes for source in cls.mcap_sources)
        print(f"[前置] execution_env: {cls.execution_env}", flush=True)
        print(f"[前置] Task_no: {cls.task_no}", flush=True)
        print(f"[前置] S3 source: {cls.mcap_source_label}", flush=True)
        print(
            f"[前置] S3 MCAP文件数: {len(cls.mcap_sources)}，"
            f"总大小: {total_mcap_size / (1024 ** 3):.3f} GiB",
            flush=True,
        )
        for index, source in enumerate(cls.mcap_sources, start=1):
            print(
                f"[前置][MCAP {index}/{len(cls.mcap_sources)}] "
                f"{source.object_name}，大小={source.size_bytes / (1024 ** 3):.3f} GiB，"
                f"chunks={source.chunk_count}",
                flush=True,
            )
        for index, camera in enumerate(cls.cameras, start=1):
            print(
                f"[前置][相机 {index}/{len(cls.cameras)}] "
                f"{camera['name']}: {camera['topic']} -> {camera['key']}",
                flush=True,
            )

        cls.task_name = None
        cls.task_status_label = None
        cls.workflow_mode = "normal"
        cls.task_id = None
        cls.scene_tags = []
        cls.annotation_id = None
        cls.episode_id = None
        cls.created_segments = {layer_id: [] for layer_id in EXPECTED_SEGMENT_COUNTS}
        cls.created_segments_by_key = {}
        cls.submit_playback = None
        cls.target_format = None
        cls.finished_conversion_entry = None
        cls.parquet_store = None
        cls.parquet_sources = []
        cls.submitted_annotation_json = None
        cls.image_validation_results = {}
        cls.vector_validation_results = {}
        cls.parquet_annotation_validation_result = None
        cls.output_dirs_prepared = False

    def _build_playback(self, current_sequence: int) -> dict[str, Any]:
        baseline_start_ns, baseline_end_ns = get_task_time_range_ns()
        duration_sec = str(
            TASK_ANNOTATION_CONFIG.get("timeline_duration_sec", "")
        ).strip()
        try:
            if float(duration_sec) <= 0:
                raise ValueError
        except (TypeError, ValueError) as exc:
            raise ValueError("timeline_duration_sec 必须是大于0的秒数") from exc
        return {
            "topic": self.baseline_camera["topic"],
            "gap_policy": "skip_on_playback",
            "current_sequence": current_sequence,
            "baseline_camera_key": BASELINE_CAMERA_KEY,
            "baseline_end_time_ns": str(baseline_end_ns),
            "baseline_camera_label": self.baseline_camera["name"],
            "timeline_duration_sec": duration_sec,
            "baseline_start_time_ns": str(baseline_start_ns),
        }

    @classmethod
    def _discover_remote_parquet_sources(cls):
        parquet_prefix = (
            f"{cls.task_no}/final-data/lerobot_v2/data/chunk-000"
        )
        cls.parquet_store = S3ParquetStore(cls.s3_config, parquet_prefix)
        cls.parquet_sources = cls.parquet_store.list_parquet_sources()
        print(
            f"[步骤9] 远端parquet目录: {cls.parquet_store.source_label}",
            flush=True,
        )
        print(
            f"[步骤9] 发现parquet文件数: {len(cls.parquet_sources)}",
            flush=True,
        )
        for index, source in enumerate(cls.parquet_sources, start=1):
            print(
                f"[步骤9][Parquet {index}/{len(cls.parquet_sources)}] "
                f"{source.object_name}，大小={source.size_bytes / (1024 ** 3):.3f} GiB",
                flush=True,
            )
        return cls.parquet_sources

    def _resolve_segment_time_ns(self, segment_key: str, field_name: str) -> int:
        created_segment = self.created_segments_by_key.get(segment_key)
        if isinstance(created_segment, dict) and created_segment.get(field_name) not in (None, ""):
            return int(created_segment[field_name])
        return int(build_annotation_time_fields(segment_key)[field_name])

    def _configured_expected_layers(self) -> list[dict[str, Any]]:
        expected_layers = []
        for layer in TASK_ANNOTATION_CONFIG["layers"]:
            layer_id = layer["layerId"]
            expected_layers.append(
                {
                    "name": layer["name"],
                    "type": layer["type"],
                    "layerId": layer_id,
                    "segments": [
                        {
                            "layerId": layer_id,
                            **build_annotation_text_fields(segment["key"]),
                            **build_annotation_time_fields(segment["key"]),
                        }
                        for segment in layer["segments"]
                    ],
                }
            )
        return expected_layers

    def _expected_annotation_layers(self) -> list[dict[str, Any]]:
        if isinstance(self.submitted_annotation_json, dict):
            layers = self.submitted_annotation_json.get("layers")
            if isinstance(layers, list) and layers:
                return layers
        return self._configured_expected_layers()

    @classmethod
    def _prepare_output_dirs_once(cls):
        if cls.output_dirs_prepared:
            return
        project_root = Path(__file__).resolve().parent.parent
        for folder_name in ("parquet_image", "mcap_image", "image_compare"):
            folder_path = project_root / folder_name
            folder_path.mkdir(parents=True, exist_ok=True)
            for child in folder_path.iterdir():
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
        cls.output_dirs_prepared = True

    @staticmethod
    def _parquet_image_path(extract_result: dict[str, Any], camera_key: str) -> Path:
        for item in extract_result.get("saved_files", []):
            if item.get("column") == camera_key and item.get("saved_path"):
                return Path(item["saved_path"])
        raise AssertionError(f"parquet 提取结果中未找到相机列 {camera_key}")

    @staticmethod
    def _mcap_image_path(extract_result: dict[str, Any], topic: str) -> Path:
        for item in extract_result.get("results", []):
            if item.get("topic_name") != topic:
                continue
            preview_path = item.get("preview_path")
            if preview_path:
                return Path(preview_path)
        raise AssertionError(f"MCAP 提取结果中未找到相机 topic {topic} 的 PNG 图片")

    def _extract_images_at_time(
        self,
        segment_key: str,
        time_key: str,
        target_ns: int,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """一次 Range 扫描同时提取目标时间的四路远端 MCAP 图片。"""
        assert self.parquet_sources, "步骤9未发现远端 parquet 文件"
        project_root = Path(__file__).resolve().parent.parent
        common_name_parts = [segment_key, time_key]
        parquet_result = extract_nearest_parquet_images(
            parquet_path=self.parquet_sources,
            target_ns=target_ns,
            output_dir=project_root / "parquet_image",
            image_columns=[camera["key"] for camera in self.cameras],
            name_prefix=self.task_no,
            extra_name_parts=common_name_parts,
        )
        mcap_result = extract_global_nearest_image_from_mcap_sources(
            mcap_sources=self.mcap_sources,
            mcap_source_label=self.mcap_source_label,
            topics=[camera["topic"] for camera in self.cameras],
            target_ns=target_ns,
            output_dir=project_root / "mcap_image",
            name_prefix=self.task_no,
            extra_name_parts=common_name_parts,
            allow_full_scan_fallback=False,
            require_chunk_indexes=True,
        )
        allure.attach(
            json.dumps(
                {
                    "parquet": parquet_result,
                    "mcap_s3": mcap_result,
                },
                ensure_ascii=False,
                indent=2,
            ),
            name=f"{segment_key}-{time_key}-四路图片提取结果",
            attachment_type=allure.attachment_type.JSON,
        )
        return parquet_result, mcap_result

    def _verify_one_camera_image(
        self,
        segment_key: str,
        time_key: str,
        target_ns: int,
        camera: dict[str, str],
        parquet_result: dict[str, Any],
        mcap_result: dict[str, Any],
    ) -> dict[str, Any]:
        project_root = Path(__file__).resolve().parent.parent
        parquet_image = self._parquet_image_path(parquet_result, camera["key"])
        mcap_image = self._mcap_image_path(mcap_result, camera["topic"])
        assert parquet_image.suffix.lower() == ".png", f"parquet 图片未解码为 PNG: {parquet_image}"
        assert mcap_image.suffix.lower() == ".png", f"MCAP 图片未解码为 PNG: {mcap_image}"
        assert parquet_image.is_file(), f"parquet 图片不存在: {parquet_image}"
        assert mcap_image.is_file(), f"MCAP 图片不存在: {mcap_image}"

        diff_path = (
            project_root
            / "image_compare"
            / f"{self.task_no}_{segment_key}_{time_key}_{camera['name']}_diff_{target_ns}.png"
        )
        comparison = compare_images(
            reference_path=mcap_image,
            candidate_path=parquet_image,
            diff_output_path=diff_path,
        )
        comparison.update(
            {
                "segment_key": segment_key,
                "time_point": time_key,
                "target_ns": target_ns,
                "camera": camera,
            }
        )

        attachment_prefix = f"{segment_key}-{time_key}-{camera['name']}"
        allure.attach.file(
            str(mcap_image),
            name=f"{attachment_prefix}-MCAP",
            attachment_type=allure.attachment_type.PNG,
        )
        allure.attach.file(
            str(parquet_image),
            name=f"{attachment_prefix}-Parquet",
            attachment_type=allure.attachment_type.PNG,
        )
        if comparison.get("diff_path"):
            allure.attach.file(
                comparison["diff_path"],
                name=f"{attachment_prefix}-像素差异图",
                attachment_type=allure.attachment_type.PNG,
            )
        allure.attach(
            json.dumps(comparison, ensure_ascii=False, indent=2),
            name=f"{attachment_prefix}-图片一致性指标",
            attachment_type=allure.attachment_type.JSON,
        )
        assert comparison["is_consistent"], (
            f"{segment_key}/{time_key}/{camera['name']} 图片不一致: "
            f"similarity={comparison.get('similarity_percent')}%, "
            f"mean_absolute_error={comparison.get('mean_absolute_error')}"
        )
        return comparison

    def _verify_one_robot_vector_time(
        self,
        segment_key: str,
        time_key: str,
        target_ns: int,
    ) -> dict[str, Any]:
        assert self.parquet_sources, "步骤9未发现远端 parquet 文件"
        mcap_result = extract_robot_vectors_at_time(
            config_path=self.robot_config_path,
            mcap_dir=None,
            target_ns=target_ns,
            mcap_sources=self.mcap_sources,
            mcap_source_label=self.mcap_source_label,
            require_chunk_indexes=True,
            allow_full_scan_fallback=False,
        )
        parquet_result = extract_parquet_robot_vectors_at_time(
            parquet_path=self.parquet_sources,
            target_ns=target_ns,
        )
        assert len(mcap_result.get("vectors", [])) == 4, (
            f"{segment_key}/{time_key} MCAP 预期4组七维向量，"
            f"实际为 {len(mcap_result.get('vectors', []))}"
        )
        assert len(parquet_result.get("vectors", [])) == 4, (
            f"{segment_key}/{time_key} parquet 预期4组七维向量，"
            f"实际为 {len(parquet_result.get('vectors', []))}"
        )
        comparison = compare_robot_vectors(
            mcap_result=mcap_result,
            parquet_result=parquet_result,
        )
        result = {
            "segment_key": segment_key,
            "time_point": time_key,
            "target_ns": target_ns,
            "mcap_result": mcap_result,
            "parquet_result": parquet_result,
            "comparison": comparison,
        }
        allure.attach(
            json.dumps(result, ensure_ascii=False, indent=2),
            name=f"{segment_key}-{time_key}-七维向量对比",
            attachment_type=allure.attachment_type.JSON,
        )
        if not comparison["is_consistent"]:
            differences = []
            for item in comparison["comparisons"]:
                for dimension in item["dimensions"]:
                    if not dimension["is_consistent"]:
                        differences.append(
                            f"{item['section']}/{item['group']}/{dimension['label']}: "
                            f"MCAP={dimension['mcap_value']}, "
                            f"Parquet={dimension['parquet_value']}, "
                            f"差值={dimension['absolute_difference']}"
                        )
            raise AssertionError(
                f"{segment_key}/{time_key} 七维向量不一致:\n" + "\n".join(differences)
            )
        return result

    @pytest.mark.order(1)
    @allure.story("步骤1：查询任务列表")
    def test_query_task_info(self):
        with allure.step("步骤1：查询任务列表"):
            allure.attach(
                f"execution_env={self.execution_env}\n"
                f"section=[{self.section}]\n"
                f"Task_no={self.task_no}\n"
                f"s3_endpoint={self.s3_config.endpoint_url}\n"
                f"s3_source={self.mcap_source_label}\n"
                f"s3_mcap_count={len(self.mcap_sources)}\n"
                f"s3_mcap_total_size={sum(source.size_bytes for source in self.mcap_sources)} bytes\n"
                f"robot_config={self.robot_config_path}\n"
                f"camera_topics={[camera['topic'] for camera in self.cameras]}\n"
                f"main_time_topic={self.main_time_topic}",
                name="前置配置上下文",
                attachment_type=allure.attachment_type.TEXT,
            )
            response = self.api_all.query_task_list()
            assertions.assert_code(response.status_code, 200)
            response_data = response.json()
            assertions.assert_text(response_data.get("msg", ""), "success")
            target_task = next(
                (
                    task
                    for task in response_data.get("data", {}).get("list", [])
                    if str(task.get("task_no", "")).strip() == self.task_no
                ),
                None,
            )
            if not target_task:
                pytest.fail(f"未找到 task_no={self.task_no!r} 的任务")

            TestTianjiV2Verify.task_name = target_task.get("name")
            TestTianjiV2Verify.task_status_label = target_task.get("status_label")
            TestTianjiV2Verify.task_id = target_task.get("id")
            TestTianjiV2Verify.scene_tags = target_task.get("scene_tags", [])
            print(f"[步骤1] task_name: {self.task_name}", flush=True)
            print(f"[步骤1] status_label: {self.task_status_label}", flush=True)
            print(f"[步骤1] task_id: {self.task_id}", flush=True)
            print(f"[步骤1] scene_tags: {self.scene_tags}", flush=True)
            allure.attach(
                json.dumps(target_task, ensure_ascii=False, indent=2),
                name="任务信息",
                attachment_type=allure.attachment_type.JSON,
            )
            if self.task_status_label == "已转换":
                TestTianjiV2Verify.workflow_mode = "converted"
            elif self.task_status_label != "已采集":
                pytest.fail(
                    f"任务状态应为'已采集'或'已转换'，实际为 {self.task_status_label!r}"
                )

    @pytest.mark.order(2)
    @allure.story("步骤2：查询标注 workspace")
    def test_query_annotation_workspace(self):
        with allure.step("步骤2：查询标注 workspace"):
            if self.workflow_mode == "converted":
                pytest.skip("任务已转换，跳过 workspace 查询")
            assertions.assert_is_not_none(self.task_id, "步骤1未提取到 task_id")
            response = self.api_all.query_annotation_workspace(task_id=self.task_id)
            assertions.assert_code(response.status_code, 200)
            response_data = response.json()
            assertions.assert_text(response_data.get("msg", ""), "success")
            data = response_data.get("data", {})
            workspace_task_no = str(data.get("task_no", "")).strip()
            TestTianjiV2Verify.annotation_id = data.get("annotation", {}).get("annotation_id")
            TestTianjiV2Verify.episode_id = data.get("episode", {}).get("episode_id")
            assertions.assert_text(workspace_task_no, self.task_no)
            assertions.assert_is_not_none(self.annotation_id, "annotation_id 不能为空")
            assertions.assert_is_not_none(self.episode_id, "episode_id 不能为空")
            print(f"[步骤2] config task_no: {self.task_no}", flush=True)
            print(f"[步骤2] workspace task_no: {workspace_task_no}", flush=True)
            print(f"[步骤2] annotation_id: {self.annotation_id}", flush=True)
            print(f"[步骤2] episode_id: {self.episode_id}", flush=True)
            allure.attach(
                f"task_no={workspace_task_no}\nannotation_id={self.annotation_id}\nepisode_id={self.episode_id}",
                name="workspace字段",
                attachment_type=allure.attachment_type.TEXT,
            )

    @pytest.mark.order(3)
    @allure.story("步骤3：新增2个L1、4个L2和3个L3标注")
    def test_create_annotation_segments(self):
        with allure.step("步骤3：逐段新增 L1/L2/L3 标注"):
            if self.workflow_mode == "converted":
                pytest.skip("任务已转换，跳过新增标注")
            assertions.assert_is_not_none(self.annotation_id, "步骤2未提取到 annotation_id")
            tag_vocabulary = self.scene_tags or []
            total_segments = sum(EXPECTED_SEGMENT_COUNTS.values())
            segment_index = 0

            for layer, position, definition in iter_layer_segments():
                segment_index += 1
                layer_id = layer["layerId"]
                segment_key = definition["key"]
                selected_scene_tag = random.choice(tag_vocabulary) if tag_vocabulary else None
                scene_tags = [selected_scene_tag] if selected_scene_tag else []
                id_prefix = "episode" if layer_id == "l1" else "seg"
                segment_id = generate_snowflake_id(id_prefix)
                segment = {
                    "segmentId": segment_id,
                    "id": segment_id,
                    "layerId": layer_id,
                    "category": layer["category"],
                    **build_annotation_text_fields(segment_key),
                    "attributes": {
                        "scene": "tabletop",
                        "sceneTags": scene_tags,
                    },
                    **build_annotation_time_fields(segment_key),
                    "baseline_camera_key": BASELINE_CAMERA_KEY,
                }
                playback = self._build_playback(
                    current_sequence=int(
                        definition.get("current_sequence", int(layer_id[1:]))
                    )
                )
                with allure.step(f"新增{layer_id.upper()}第{position}段：{segment_key}"):
                    response = self.api_all.create_task_annotation_segments(
                        annotation_id=self.annotation_id,
                        tag_vocabulary=tag_vocabulary,
                        segments=[segment],
                        playback=playback,
                    )
                    assertions.assert_code(response.status_code, 200)
                    response_data = response.json()
                    assertions.assert_text(response_data.get("msg", ""), "success")
                    print(
                        f"[步骤3][{segment_index}/{total_segments}] "
                        f"{layer_id.upper()}第{position}段 {segment_key} 创建成功，"
                        f"description={segment['description']!r}，"
                        f"sceneTags={scene_tags}，segmentId={segment_id}，"
                        f"time=[{segment['startTimeNs']}, {segment['endTimeNs']}]",
                        flush=True,
                    )
                    allure.attach(
                        json.dumps(
                            {"segment": segment, "playback": playback},
                            ensure_ascii=False,
                            indent=2,
                        ),
                        name=f"{segment_key}-请求参数",
                        attachment_type=allure.attachment_type.JSON,
                    )
                    allure.attach(
                        json.dumps(response_data, ensure_ascii=False, indent=2),
                        name=f"{segment_key}-响应",
                        attachment_type=allure.attachment_type.JSON,
                    )
                TestTianjiV2Verify.created_segments[layer_id].append(segment)
                TestTianjiV2Verify.created_segments_by_key[segment_key] = segment
                if TestTianjiV2Verify.submit_playback is None:
                    TestTianjiV2Verify.submit_playback = playback

    @pytest.mark.order(4)
    @allure.story("步骤4：提交 L1/L2/L3 标注")
    def test_submit_annotation(self):
        with allure.step("步骤4：提交 L1/L2/L3 标注"):
            if self.workflow_mode == "converted":
                pytest.skip("任务已转换，跳过提交标注")
            assertions.assert_is_not_none(self.annotation_id, "缺少 annotation_id")
            assertions.assert_is_not_none(self.episode_id, "缺少 episode_id")
            assertions.assert_is_not_none(self.submit_playback, "缺少 playback")
            for layer_id, expected_count in EXPECTED_SEGMENT_COUNTS.items():
                actual_count = len(self.created_segments[layer_id])
                assert actual_count == expected_count, (
                    f"{layer_id.upper()} 创建数量不正确: 期望 {expected_count}，实际 {actual_count}"
                )

            layers = [
                {
                    "name": layer["name"],
                    "type": layer["type"],
                    "layerId": layer["layerId"],
                    "segments": self.created_segments[layer["layerId"]],
                }
                for layer in TASK_ANNOTATION_CONFIG["layers"]
            ]
            leaf_segments = self.created_segments["l3"]
            annotation_json = {
                "layers": layers,
                "playback": self.submit_playback,
                "segments": leaf_segments,
                "timeUnit": "source_timestamp_sec",
                "episodeId": str(self.episode_id),
                "tagVocabulary": self.scene_tags or [],
                "episodeTimeUnit": "episode_sec",
            }
            TestTianjiV2Verify.submitted_annotation_json = annotation_json
            allure.attach(
                json.dumps(annotation_json, ensure_ascii=False, indent=2),
                name="步骤4提交标注JSON",
                attachment_type=allure.attachment_type.JSON,
            )
            response = self.api_all.submit_task_annotation(
                annotation_id=self.annotation_id,
                tag_vocabulary=self.scene_tags or [],
                annotation_json=annotation_json,
            )
            assertions.assert_code(response.status_code, 200)
            response_data = response.json()
            assertions.assert_text(response_data.get("msg", ""), "success")
            print(
                "[步骤4] 标注提交成功: L1=2，L2=4，L3=3，合计=9",
                flush=True,
            )
            allure.attach(
                json.dumps(response_data, ensure_ascii=False, indent=2),
                name="步骤4响应",
                attachment_type=allure.attachment_type.JSON,
            )

    @pytest.mark.order(5)
    @allure.story("步骤5：完成质检")
    def test_complete_task_qc(self):
        with allure.step("步骤5：完成质检"):
            if self.workflow_mode == "converted":
                pytest.skip("任务已转换，跳过完成质检")
            assertions.assert_is_not_none(self.task_id, "步骤1未提取到 task_id")
            response = self.api_all.complete_task_qc(task_id=self.task_id)
            assertions.assert_code(response.status_code, 200)
            response_data = response.json()
            assertions.assert_in_text(response_data, "success")
            print(f"[步骤5] 质检完成: task_id={self.task_id}", flush=True)
            allure.attach(
                json.dumps(response_data, ensure_ascii=False, indent=2),
                name="步骤5响应",
                attachment_type=allure.attachment_type.JSON,
            )

    @pytest.mark.order(6)
    @allure.story("步骤6：发起 V2 数据转换")
    def test_create_conversion(self):
        with allure.step("步骤6：发起 V2 数据转换"):
            assertions.assert_is_not_none(self.task_id, "步骤1未提取到 task_id")
            TestTianjiV2Verify.target_format = "lerobot_v2"
            if self.workflow_mode == "converted":
                print(
                    f"[步骤6] 任务已转换，复用格式: {self.target_format}",
                    flush=True,
                )
                return
            print(
                f"[步骤6] 发起转换: task_id={self.task_id}，"
                f"target_format={self.target_format}",
                flush=True,
            )
            response = self.api_all.create_conversion(
                task_id=self.task_id,
                target_format=self.target_format,
                quality_labels=[],
            )
            assertions.assert_code(response.status_code, 200)
            response_data = response.json()
            assertions.assert_in_text(response_data, "success")
            print(f"[步骤6] 转换任务创建成功: task_id={self.task_id}", flush=True)
            allure.attach(
                json.dumps(response_data, ensure_ascii=False, indent=2),
                name="步骤6响应",
                attachment_type=allure.attachment_type.JSON,
            )

    @pytest.mark.order(7)
    @allure.story("步骤7：轮询正在转换列表")
    def test_poll_active_conversion(self):
        with allure.step("步骤7：轮询正在转换列表"):
            if self.workflow_mode == "converted":
                pytest.skip("任务已转换，跳过 active 列表轮询")
            assertions.assert_is_not_none(self.task_id, "步骤1未提取到 task_id")
            poll_started_at = time.monotonic()
            poll_deadline = poll_started_at + CONVERSION_POLL_TIMEOUT_SECONDS
            attempt = 0
            last_status = None
            print(
                f"[步骤7] 开始轮询转换状态: task_id={self.task_id}，"
                f"间隔={CONVERSION_POLL_INTERVAL_SECONDS}秒，"
                f"超时={CONVERSION_POLL_TIMEOUT_SECONDS // 60}分钟",
                flush=True,
            )
            while True:
                attempt += 1
                response = self.api_all.query_active_conversion_list()
                assertions.assert_code(response.status_code, 200)
                response_data = response.json()
                assertions.assert_text(response_data.get("msg", ""), "success")
                conversion_list = response_data.get("data", {}).get("list", [])
                target_entry = find_conversion_entry(conversion_list, self.task_id)
                if not target_entry:
                    elapsed_seconds = time.monotonic() - poll_started_at
                    print(
                        f"[步骤7] 第{attempt}次轮询: active列表未找到任务，"
                        f"已结束转换阶段，耗时={elapsed_seconds:.1f}秒",
                        flush=True,
                    )
                    allure.attach(
                        f"attempt={attempt}\ntask_id={self.task_id} 已离开 active 列表",
                        name="步骤7轮询结果",
                        attachment_type=allure.attachment_type.TEXT,
                    )
                    return
                status = str(target_entry.get("status", "")).strip().lower()
                last_status = status
                elapsed_seconds = time.monotonic() - poll_started_at
                print(
                    f"[步骤7] 第{attempt}次轮询: status={status}，"
                    f"active列表条数={len(conversion_list)}，"
                    f"已等待={elapsed_seconds:.1f}秒",
                    flush=True,
                )
                allure.attach(
                    json.dumps(target_entry, ensure_ascii=False, indent=2),
                    name=f"步骤7第{attempt}次轮询",
                    attachment_type=allure.attachment_type.JSON,
                )
                if status not in {"running", "queued"}:
                    pytest.fail(f"active 转换状态异常: {status!r}")

                remaining_seconds = poll_deadline - time.monotonic()
                if remaining_seconds <= 0:
                    elapsed_seconds = time.monotonic() - poll_started_at
                    allure.attach(
                        f"task_id={self.task_id}\n"
                        f"attempts={attempt}\n"
                        f"last_status={last_status}\n"
                        f"elapsed_seconds={elapsed_seconds:.3f}\n"
                        f"timeout_seconds={CONVERSION_POLL_TIMEOUT_SECONDS}",
                        name="步骤7轮询超时",
                        attachment_type=allure.attachment_type.TEXT,
                    )
                    pytest.fail(
                        f"转换轮询超过 {CONVERSION_POLL_TIMEOUT_SECONDS // 60} 分钟，"
                        f"task_id={self.task_id}，最后状态={last_status!r}"
                    )
                time.sleep(
                    min(CONVERSION_POLL_INTERVAL_SECONDS, remaining_seconds)
                )

    @pytest.mark.order(8)
    @allure.story("步骤8：查询已转换完成列表")
    def test_check_finished_conversion(self):
        with allure.step("步骤8：查询已转换完成列表"):
            if self.workflow_mode == "converted":
                pytest.skip("任务已转换，跳过 finished 列表确认")
            response = self.api_all.query_finished_conversion_list()
            assertions.assert_code(response.status_code, 200)
            response_data = response.json()
            assertions.assert_text(response_data.get("msg", ""), "success")
            target_entry = find_conversion_entry(
                response_data.get("data", {}).get("list", []),
                self.task_id,
            )
            if not target_entry:
                pytest.fail(f"finished 列表中未找到 task_id={self.task_id}")
            TestTianjiV2Verify.finished_conversion_entry = target_entry
            status = str(target_entry.get("status", "")).strip().lower()
            print(
                f"[步骤8] finished列表条数: "
                f"{len(response_data.get('data', {}).get('list', []))}",
                flush=True,
            )
            print(
                f"[步骤8] 找到task_id={self.task_id}，status={status}",
                flush=True,
            )
            assert status == "completed", f"finished 转换状态应为 completed，实际为 {status!r}"
            allure.attach(
                json.dumps(target_entry, ensure_ascii=False, indent=2),
                name="步骤8转换条目",
                attachment_type=allure.attachment_type.JSON,
            )

    @pytest.mark.order(9)
    @allure.story("步骤9：发现 S3 中全部 V2 parquet 文件")
    def test_discover_remote_parquet_files(self):
        with allure.step("步骤9：枚举 chunk-000 中全部远端 parquet 文件"):
            assertions.assert_is_not_none(self.target_format, "步骤6未设置 target_format")
            if self.workflow_mode != "converted":
                assertions.assert_is_not_none(
                    self.finished_conversion_entry,
                    "步骤8未确认转换完成",
                )
            parquet_sources = self._discover_remote_parquet_sources()
            assert parquet_sources, "S3 chunk-000 中未发现 parquet 文件"
            allure.attach(
                "\n".join(str(source) for source in parquet_sources),
                name="步骤9远端parquet文件列表",
                attachment_type=allure.attachment_type.TEXT,
            )

    @pytest.mark.order(10)
    @allure.story("步骤10：校验全部标注起止时间的四路相机图片")
    def test_compare_all_annotation_camera_images(self):
        with allure.step("步骤10：校验9段标注起止时间的四路相机图片"):
            self._prepare_output_dirs_once()
            failures: list[str] = []
            results: dict[str, Any] = {}
            total_time_points = sum(EXPECTED_SEGMENT_COUNTS.values()) * 2
            total_image_cases = total_time_points * len(self.cameras)
            time_point_index = 0
            image_case_index = 0
            print(
                f"[步骤10] 开始校验: {total_time_points}个时间点，"
                f"{len(self.cameras)}路相机，共{total_image_cases}组图片",
                flush=True,
            )
            for _, _, definition in iter_layer_segments():
                segment_key = definition["key"]
                segment_results = {}
                for time_key, field_name, time_label in (
                    ("start", "startTimeNs", "开始时间"),
                    ("end", "endTimeNs", "结束时间"),
                ):
                    time_point_index += 1
                    target_ns = self._resolve_segment_time_ns(segment_key, field_name)
                    time_results = {}
                    print(
                        f"[步骤10][时间点 {time_point_index}/{total_time_points}] "
                        f"{segment_key}/{time_key} target_ns={target_ns}，"
                        "开始提取parquet与S3 MCAP四路图片",
                        flush=True,
                    )
                    try:
                        parquet_result, mcap_result = self._extract_images_at_time(
                            segment_key=segment_key,
                            time_key=time_key,
                            target_ns=target_ns,
                        )
                    except Exception as exc:
                        error_message = (
                            f"{segment_key}/{time_key} 四路图片提取失败: {exc}"
                        )
                        failures.append(error_message)
                        image_case_index += len(self.cameras)
                        print(
                            f"[步骤10][图片 {image_case_index - len(self.cameras) + 1}-"
                            f"{image_case_index}/{total_image_cases}] {error_message}",
                            flush=True,
                        )
                        for camera in self.cameras:
                            time_results[camera["name"]] = {"error": str(exc)}
                        allure.attach(
                            error_message,
                            name=f"{segment_key}-{time_key}-四路图片提取异常",
                            attachment_type=allure.attachment_type.TEXT,
                        )
                        segment_results[time_key] = time_results
                        continue
                    for camera in self.cameras:
                        image_case_index += 1
                        case_key = f"{segment_key}/{time_key}/{camera['name']}"
                        with allure.step(
                            f"{segment_key}{time_label}-{camera['name']}图片对比"
                        ):
                            try:
                                comparison = self._verify_one_camera_image(
                                    segment_key=segment_key,
                                    time_key=time_key,
                                    target_ns=target_ns,
                                    camera=camera,
                                    parquet_result=parquet_result,
                                    mcap_result=mcap_result,
                                )
                                time_results[camera["name"]] = comparison
                                print(
                                    f"[步骤10][图片 {image_case_index}/{total_image_cases}] "
                                    f"{case_key} consistent={comparison['is_consistent']}，"
                                    f"similarity={comparison.get('similarity_percent')}%，"
                                    f"mean_absolute_error="
                                    f"{comparison.get('mean_absolute_error')}",
                                    flush=True,
                                )
                            except (
                                AssertionError,
                                FileNotFoundError,
                                KeyError,
                                OSError,
                                TypeError,
                                ValueError,
                            ) as exc:
                                error_message = f"{case_key} 图片校验失败: {exc}"
                                time_results[camera["name"]] = {"error": str(exc)}
                                failures.append(error_message)
                                print(
                                    f"[步骤10][图片 {image_case_index}/{total_image_cases}] "
                                    f"{error_message}",
                                    flush=True,
                                )
                                allure.attach(
                                    error_message,
                                    name=f"{case_key}-异常",
                                    attachment_type=allure.attachment_type.TEXT,
                                )
                    segment_results[time_key] = time_results
                results[segment_key] = segment_results
            TestTianjiV2Verify.image_validation_results = results
            allure.attach(
                json.dumps(results, ensure_ascii=False, indent=2),
                name="步骤10全部图片校验汇总",
                attachment_type=allure.attachment_type.JSON,
            )
            print(
                f"[步骤10] 图片校验完成: 总数={total_image_cases}，"
                f"失败={len(failures)}",
                flush=True,
            )
            assert not failures, "全部标注图片校验存在失败:\n" + "\n".join(failures)

    @pytest.mark.order(11)
    @allure.story("步骤11：校验全部标注起止时间的七维向量")
    def test_compare_all_annotation_robot_vectors(self):
        with allure.step("步骤11：校验9段标注起止时间的MCAP与parquet七维向量"):
            failures: list[str] = []
            results: dict[str, Any] = {}
            for _, _, definition in iter_layer_segments():
                segment_key = definition["key"]
                segment_results = {}
                for time_key, field_name, time_label in (
                    ("start", "startTimeNs", "开始时间"),
                    ("end", "endTimeNs", "结束时间"),
                ):
                    target_ns = self._resolve_segment_time_ns(segment_key, field_name)
                    with allure.step(f"{segment_key}{time_label}七维向量对比"):
                        try:
                            segment_results[time_key] = self._verify_one_robot_vector_time(
                                segment_key=segment_key,
                                time_key=time_key,
                                target_ns=target_ns,
                            )
                        except (
                            AssertionError,
                            FileNotFoundError,
                            KeyError,
                            OSError,
                            TypeError,
                            ValueError,
                        ) as exc:
                            error_message = f"{segment_key}/{time_key} 七维向量校验失败: {exc}"
                            segment_results[time_key] = {"error": str(exc)}
                            failures.append(error_message)
                            allure.attach(
                                error_message,
                                name=f"{segment_key}-{time_key}-异常",
                                attachment_type=allure.attachment_type.TEXT,
                            )
                results[segment_key] = segment_results
            TestTianjiV2Verify.vector_validation_results = results
            allure.attach(
                json.dumps(results, ensure_ascii=False, indent=2),
                name="步骤11全部七维向量校验汇总",
                attachment_type=allure.attachment_type.JSON,
            )
            assert not failures, "全部标注七维向量校验存在失败:\n" + "\n".join(failures)

    @pytest.mark.order(12)
    @allure.story("步骤12：校验 parquet 中的 L1/L2/L3 标注")
    def test_compare_parquet_annotations(self):
        with allure.step("步骤12：逐项校验 parquet 中的2个L1、4个L2和3个L3标注"):
            assert self.parquet_sources, "步骤9未发现远端 parquet 文件"
            expected_layers = self._expected_annotation_layers()
            result = compare_parquet_annotations(
                parquet_path=self.parquet_sources,
                expected_layers=expected_layers,
            )
            TestTianjiV2Verify.parquet_annotation_validation_result = result
            print(
                f"[步骤12] parquet标注校验: consistent={result['is_consistent']}，"
                f"失败项={len(result.get('failures', []))}",
                flush=True,
            )
            allure.attach(
                json.dumps(expected_layers, ensure_ascii=False, indent=2),
                name="步骤12期望标注",
                attachment_type=allure.attachment_type.JSON,
            )
            allure.attach(
                json.dumps(result, ensure_ascii=False, indent=2),
                name="步骤12 parquet 标注校验结果",
                attachment_type=allure.attachment_type.JSON,
            )
            assert result["is_consistent"], (
                "parquet 标注数据与代码配置不一致:\n" + "\n".join(result["failures"])
            )

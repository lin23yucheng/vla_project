"""摇操数据 V3 任务校验流程。"""

import configparser
import json
import os
import random
import shutil
import time
from pathlib import Path

import allure
import pytest

from api import all_api
from common import Assert
from common.client_factory import create_lazy_yixiu_client
from common.extract_mcap_fields import extract_robot_vectors_at_time, load_robot_vector_topics
from common.extract_mcap_image import extract_global_nearest_image_from_mcap_sources
from common.extract_parquet_fields import compare_robot_vectors, extract_parquet_robot_vectors_at_time
from common.image_compare import compare_images
from common.parse_parquet_file import (
    compare_parquet_annotations,
    extract_nearest_parquet_images_batch,
    format_l1_playback_duration_comparison,
)
from common.s3_mcap import S3McapConfig, S3McapStore
from common.s3_parquet import S3ParquetStore


START_TIME_NS = 1768893104830136738
END_TIME_NS = 1768893114656136738
BASELINE_START_TIME_NS = 1768893097352136738
BASELINE_END_TIME_NS = 1768893130775500327
ANNOTATION_DESCRIPTION = "移动夹笔放回全流程"
ROBOT_CONFIG_NAME = "遥操机械臂.json"
CONVERSION_POLL_INTERVAL_SECONDS = 10
CONVERSION_POLL_TIMEOUT_SECONDS = 15 * 60

assertions = Assert.Assertions()
_last_snowflake_13 = 0


def load_task_context() -> tuple[str, str, str, S3McapConfig]:
    """读取当前环境的任务编号和 MinIO 连接配置。"""
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

    task_no = config.get(section, "shake_task_no", fallback="").strip()
    if not task_no:
        raise ValueError(f"配置文件节 [{section}] 中缺少 shake_task_no")

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


def generate_episode_snowflake_id() -> str:
    """生成 episode_13 位毫秒时间戳 ID。"""
    global _last_snowflake_13
    now_ms = int(time.time() * 1000)
    if now_ms <= _last_snowflake_13:
        now_ms = _last_snowflake_13 + 1
    _last_snowflake_13 = now_ms
    return f"episode_{now_ms}"


def find_conversion_entry(conversion_list: list[dict], task_id) -> dict | None:
    for item in conversion_list or []:
        if str(item.get("task_id", "")).strip() == str(task_id):
            return item
    return None


def load_robot_config() -> tuple[Path, dict]:
    config_path = Path(__file__).resolve().parent.parent / "Robot_Configuration" / ROBOT_CONFIG_NAME
    with config_path.open("r", encoding="utf-8") as file:
        return config_path, json.load(file)


global_client = create_lazy_yixiu_client()


@pytest.mark.annotation_v3
@allure.feature("场景：摇操数据-V3格式信息校验")
class TestShakeV3Verify:
    _TEST_STEP_NAMES = {
        "test_query_task_info": "1",
        "test_query_annotation_workspace": "2",
        "test_create_l1_annotation_segment": "3",
        "test_submit_l1_annotation": "4",
        "test_complete_task_qc": "5",
        "test_create_conversion": "6",
        "test_poll_active_conversion": "7",
        "test_check_finished_conversion": "8",
        "test_discover_remote_parquet_files": "9",
        "test_extract_l1_parquet_images": "10",
        "test_extract_l1_mcap_images": "11",
        "test_compare_l1_camera_images": "12",
        "test_extract_l1_robot_vectors": "13",
        "test_compare_l1_robot_vectors": "14",
        "test_compare_parquet_annotations": "15",
    }

    @classmethod
    def setup_class(cls):
        cls.api_all = all_api.ApiAll(global_client)
        cls.execution_env, cls.section, cls.task_no, cls.s3_config = load_task_context()
        cls.mcap_store = S3McapStore(cls.s3_config)
        cls.mcap_sources = cls.mcap_store.list_indexed_mcap_sources()
        cls.mcap_source_label = cls.mcap_store.source_label
        cls.robot_config_path, cls.robot_config = load_robot_config()
        cls.camera_topics = cls._resolve_camera_topics(cls.robot_config)
        cls.mcap_prefetch_topics = load_robot_vector_topics(cls.robot_config_path)
        cls.task_name = None
        cls.task_status_label = None
        cls.workflow_mode = "normal"
        cls.task_id = None
        cls.scene_tags = None
        cls.annotation_id = None
        cls.episode_id = None
        cls.l1_segment = None
        cls.submit_playback = None
        cls.target_format = None
        cls.active_conversion_entry = None
        cls.finished_conversion_entry = None
        cls.parquet_store = None
        cls.parquet_sources = []
        cls.parquet_image_extract_result = None
        cls.parquet_image_batch_extract_results = None
        cls.mcap_image_extract_result = None
        cls.mcap_vector_extract_results = None
        cls.parquet_vector_extract_results = None
        cls.submitted_annotation_json = None
        cls.parquet_annotation_validation_result = None
        cls.image_output_dirs_prepared = False

    def setup_method(self, method):
        self._step_started_at = time.perf_counter()
        self._step_number = self._TEST_STEP_NAMES.get(method.__name__, method.__name__)
        print(f"[步骤{self._step_number}] 开始执行: {method.__name__}", flush=True)

    def teardown_method(self, method):
        elapsed_seconds = time.perf_counter() - self._step_started_at
        print(
            f"[步骤{self._step_number}] 执行结束: {method.__name__}，耗时 {elapsed_seconds:.3f}s",
            flush=True,
        )

    def _resolve_l1_time_ns(self, field_name: str, fallback: int) -> int:
        if isinstance(self.l1_segment, dict):
            value = self.l1_segment.get(field_name)
            if value not in (None, ""):
                try:
                    return int(str(value))
                except (TypeError, ValueError):
                    pass
        return fallback

    def _resolve_l1_start_time_ns(self) -> int:
        return self._resolve_l1_time_ns("startTimeNs", START_TIME_NS)

    def _resolve_l1_end_time_ns(self) -> int:
        return self._resolve_l1_time_ns("endTimeNs", END_TIME_NS)

    def _expected_annotation_layers(self) -> list[dict]:
        if isinstance(self.submitted_annotation_json, dict):
            layers = self.submitted_annotation_json.get("layers")
            if isinstance(layers, list) and layers:
                return layers

        # 已转换任务会跳过标注步骤，使用本用例中定义的同一组标注期望值。
        return [
            {
                "name": "L1 Episode",
                "type": "episode",
                "layerId": "l1",
                "segments": [
                    {
                        "layerId": "l1",
                        "description": ANNOTATION_DESCRIPTION,
                        "startTimeNs": str(START_TIME_NS),
                        "endTimeNs": str(END_TIME_NS),
                    }
                ],
            }
        ]

    def _prepare_image_output_dirs_once(self) -> None:
        if TestShakeV3Verify.image_output_dirs_prepared:
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
        TestShakeV3Verify.image_output_dirs_prepared = True

    @staticmethod
    def _resolve_camera_topics(robot_config: dict) -> list[str]:
        topics_by_group = {
            item.get("group"): item.get("topic")
            for item in robot_config.get("cameras", [])
            if isinstance(item, dict)
        }
        missing = [side for side in ("left", "right") if not topics_by_group.get(side)]
        if missing:
            raise ValueError(f"{ROBOT_CONFIG_NAME} cameras 缺少相机配置: {missing}")
        return [topics_by_group["left"], topics_by_group["right"]]

    @staticmethod
    def _find_parquet_image(extract_result: dict, side: str) -> Path:
        for item in extract_result.get("saved_files", []):
            if item.get("side") == side and item.get("saved_path"):
                return Path(item["saved_path"])
        raise AssertionError(f"parquet 提取结果中未找到 {side} 相机图片")

    @staticmethod
    def _find_mcap_image(extract_result: dict, side: str) -> Path:
        for item in extract_result.get("results", []):
            if item.get("side") != side:
                continue
            if item.get("preview_path"):
                return Path(item["preview_path"])
            for saved_path in item.get("saved_paths", []):
                path = Path(saved_path)
                if path.suffix.lower() == ".png":
                    return path
        raise AssertionError(f"mcap 提取结果中未找到 {side} 相机图片")

    def _extract_parquet_images(self, workflow_step: str) -> dict:
        assert self.parquet_sources, f"步骤9未发现远端 parquet 文件，无法执行步骤{workflow_step}"
        allure.attach(
            json.dumps(
                {
                    "parquet_source": self.parquet_store.source_label,
                    "parquet_files": [str(source) for source in self.parquet_sources],
                    "startTimeNs": self._resolve_l1_start_time_ns(),
                    "endTimeNs": self._resolve_l1_end_time_ns(),
                },
                ensure_ascii=False,
                indent=2,
            ),
            name=f"步骤{workflow_step}-parquet基础信息",
            attachment_type=allure.attachment_type.JSON,
        )

        output_dir = Path(__file__).resolve().parent.parent / "parquet_image"
        batch_results = self._ensure_parquet_image_batch_extract_results()
        results = {}
        for substep, result_key, time_key, time_label, target_ns in (
            (f"{workflow_step}.1", "start_time_extract", "start", "开始时间", self._resolve_l1_start_time_ns()),
            (f"{workflow_step}.2", "end_time_extract", "end", "结束时间", self._resolve_l1_end_time_ns()),
        ):
            with allure.step(f"步骤{substep}：提取L1标注{time_label} parquet 图片"):
                result = batch_results[result_key]
                assert result["target_ns"] == target_ns, (
                    f"步骤{substep}批量图片目标时间不一致: "
                    f"{result['target_ns']} != {target_ns}"
                )
                results[result_key] = result
                print(
                    f"[步骤{substep}] parquet图片 target_ns={target_ns} "
                    f"matched_timestamp_ns={result.get('matched_timestamp_ns')} "
                    f"diff_ns={result.get('diff_ns')} "
                    f"saved_files={len(result.get('saved_files', []))}",
                    flush=True,
                )
                allure.attach(
                    json.dumps(result, ensure_ascii=False, indent=2),
                    name=f"步骤{substep}-图片提取结果",
                    attachment_type=allure.attachment_type.JSON,
                )
                assert result.get("saved_files"), f"步骤{substep}未生成 parquet 图片"
                for item in result["saved_files"]:
                    assert Path(item["saved_path"]).is_file(), f"提取文件不存在: {item['saved_path']}"
        return results

    def _ensure_parquet_image_batch_extract_results(self) -> dict:
        if TestShakeV3Verify.parquet_image_batch_extract_results is not None:
            return TestShakeV3Verify.parquet_image_batch_extract_results

        target_requests = [
            {
                "key": "start_time_extract",
                "target_ns": self._resolve_l1_start_time_ns(),
                "extra_name_parts": ["l1", "start"],
            },
            {
                "key": "end_time_extract",
                "target_ns": self._resolve_l1_end_time_ns(),
                "extra_name_parts": ["l1", "end"],
            },
        ]
        output_dir = Path(__file__).resolve().parent.parent / "parquet_image"
        batch_result = extract_nearest_parquet_images_batch(
            parquet_path=self.parquet_sources,
            target_requests=target_requests,
            output_dir=output_dir,
            name_prefix=self.task_no,
        )
        TestShakeV3Verify.parquet_image_batch_extract_results = batch_result["results"]
        allure.attach(
            json.dumps(
                {
                    "target_count": batch_result["target_count"],
                    "source_count": batch_result["source_count"],
                    "row_groups_read": batch_result["row_groups_read"],
                    "output_dir": batch_result["output_dir"],
                },
                ensure_ascii=False,
                indent=2,
            ),
            name="parquet图片批量读取汇总",
            attachment_type=allure.attachment_type.JSON,
        )
        return TestShakeV3Verify.parquet_image_batch_extract_results

    def _extract_mcap_images(self, workflow_step: str) -> dict:
        assert self.mcap_sources, f"S3 original-data 下未发现 MCAP，无法执行步骤{workflow_step}"
        assertions.assert_is_not_none(
            self.parquet_image_extract_result,
            f"未提取 parquet 图片，无法执行步骤{workflow_step}的同步取帧",
        )
        topics = self.camera_topics
        output_dir = Path(__file__).resolve().parent.parent / "mcap_image"
        results = {}
        for substep, result_key, time_key, time_label, annotation_target_ns in (
            (f"{workflow_step}.1", "start_time_extract", "start", "开始时间", self._resolve_l1_start_time_ns()),
            (f"{workflow_step}.2", "end_time_extract", "end", "结束时间", self._resolve_l1_end_time_ns()),
        ):
            with allure.step(f"步骤{substep}：提取L1标注{time_label} MCAP 图片"):
                parquet_result = (self.parquet_image_extract_result or {}).get(result_key)
                assertions.assert_is_not_none(
                    parquet_result,
                    f"缺少L1标注{time_label} parquet 图片提取结果",
                )
                parquet_matched_ns = parquet_result.get("matched_timestamp_ns")
                assertions.assert_is_not_none(
                    parquet_matched_ns,
                    f"L1标注{time_label} parquet 提取结果缺少 matched_timestamp_ns",
                )
                mcap_target_ns = int(parquet_matched_ns)
                result = extract_global_nearest_image_from_mcap_sources(
                    mcap_sources=self.mcap_sources,
                    mcap_source_label=self.mcap_source_label,
                    topics=topics,
                    target_ns=mcap_target_ns,
                    output_dir=output_dir,
                    name_prefix=self.task_no,
                    extra_name_parts=["l1", time_key],
                    additional_cache_topics=self.mcap_prefetch_topics,
                )
                result["annotation_target_ns"] = annotation_target_ns
                result["parquet_matched_timestamp_ns"] = mcap_target_ns
                results[result_key] = result
                print(
                    f"[步骤{substep}] L1标注{time_label} MCAP图片提取完成："
                    f"标注时间={annotation_target_ns}ns，"
                    f"Parquet匹配时间={mcap_target_ns}ns，"
                    f"已扫描消息数={result.get('scanned_messages', '未提供')}，"
                    f"窗口缓存命中数={result.get('window_cache_hits', '未提供')}",
                    flush=True,
                )
                allure.attach(
                    json.dumps(result, ensure_ascii=False, indent=2),
                    name=f"步骤{substep}-图片提取结果",
                    attachment_type=allure.attachment_type.JSON,
                )
                saved_files = [
                    Path(path)
                    for item in result.get("results", [])
                    for path in item.get("saved_paths", [])
                ]
                assert saved_files, f"步骤{substep}未生成 MCAP 图片"
                for path in saved_files:
                    assert path.is_file(), f"提取文件不存在: {path}"
        return results

    def _compare_images(self, workflow_step: str) -> None:
        failures = []
        comparisons = {}
        cases = (
            (f"{workflow_step}.1", "start_time_extract", "start", "开始", "left", "左", self._resolve_l1_start_time_ns()),
            (f"{workflow_step}.2", "start_time_extract", "start", "开始", "right", "右", self._resolve_l1_start_time_ns()),
            (f"{workflow_step}.3", "end_time_extract", "end", "结束", "left", "左", self._resolve_l1_end_time_ns()),
            (f"{workflow_step}.4", "end_time_extract", "end", "结束", "right", "右", self._resolve_l1_end_time_ns()),
        )
        for substep, extract_key, time_key, time_label, side, side_label, target_ns in cases:
            with allure.step(f"步骤{substep}：对比L1标注{time_label}时间{side_label}相机图片"):
                try:
                    parquet_result = (self.parquet_image_extract_result or {}).get(extract_key, {})
                    mcap_result = (self.mcap_image_extract_result or {}).get(extract_key, {})
                    parquet_image = self._find_parquet_image(parquet_result, side)
                    mcap_image = self._find_mcap_image(mcap_result, side)
                    diff_path = (
                        Path(__file__).resolve().parent.parent
                        / "image_compare"
                        / f"{self.task_no}_l1_{time_key}_{side}_diff_{target_ns}.png"
                    )
                    comparison = compare_images(
                        reference_path=mcap_image,
                        candidate_path=parquet_image,
                        diff_output_path=diff_path,
                    )
                    comparisons[f"{time_key}_{side}"] = comparison
                    if not comparison["dimension_match"]:
                        result_message = (
                            f"{time_label}时间{side_label}相机图片尺寸不一致："
                            f"MCAP={comparison['reference_size']}，"
                            f"Parquet={comparison['candidate_size']}"
                        )
                    else:
                        result_message = (
                            f"{time_label}时间{side_label}相机图片一致性="
                            f"{comparison['is_consistent']}，"
                            f"容差内像素占比={comparison['similarity_percent']:.4f}%，"
                            f"平均绝对误差={comparison['mean_absolute_error']:.4f}，"
                            f"尺寸={comparison['reference_size']}"
                        )
                    print(f"[步骤{substep}] {result_message}", flush=True)
                    allure.attach.file(str(mcap_image), name=f"步骤{substep}-MCAP", attachment_type=allure.attachment_type.PNG)
                    allure.attach.file(str(parquet_image), name=f"步骤{substep}-Parquet", attachment_type=allure.attachment_type.PNG)
                    if comparison.get("diff_path"):
                        allure.attach.file(comparison["diff_path"], name=f"步骤{substep}-差异图", attachment_type=allure.attachment_type.PNG)
                    if not comparison["is_consistent"]:
                        failures.append(f"{time_label}时间{side_label}相机图片不一致: {comparison}")
                except (AssertionError, FileNotFoundError, OSError, ValueError) as exc:
                    comparisons[f"{time_key}_{side}"] = {"error": str(exc)}
                    failures.append(f"{time_label}时间{side_label}相机比较失败: {exc}")
        allure.attach(
            json.dumps(comparisons, ensure_ascii=False, indent=2),
            name=f"步骤{workflow_step}-L1图片对比汇总",
            attachment_type=allure.attachment_type.JSON,
        )
        assert not failures, "L1 标注图片一致性校验失败:\n" + "\n".join(failures)

    def _extract_mcap_vectors(self, workflow_step: str) -> dict:
        config_path, _ = load_robot_config()
        expected_pose_fields = [
            "pose.position.x",
            "pose.position.y",
            "pose.position.z",
            "pose.orientation.x",
            "pose.orientation.y",
            "pose.orientation.z",
            "pose.orientation.w",
        ]
        expected_topics = {
            "left": ("/jbt_arm_L/current_arm_tcp_pose", "/gripper/gripper_l/data"),
            "right": ("/jbt_arm_R/current_arm_tcp_pose", "/gripper/gripper_r/data"),
        }
        results = {}
        for substep, time_key, time_label, target_ns in (
            (f"{workflow_step}.1", "start", "开始时间", self._resolve_l1_start_time_ns()),
            (f"{workflow_step}.2", "end", "结束时间", self._resolve_l1_end_time_ns()),
        ):
            with allure.step(f"步骤{substep}：提取L1标注{time_label} MCAP 七维向量"):
                result = extract_robot_vectors_at_time(
                    config_path=config_path,
                    mcap_dir=None,
                    target_ns=target_ns,
                    mcap_sources=self.mcap_sources,
                    mcap_source_label=self.mcap_source_label,
                    require_chunk_indexes=True,
                    allow_full_scan_fallback=False,
                )
                results[time_key] = result
                vectors = result.get("vectors", [])
                assert len(vectors) == 4, f"预期提取 action/observation_state 左右共4组向量，实际为 {len(vectors)}"
                for item in vectors:
                    assert item["vector_labels"] == ["x", "y", "z", "roll", "pitch", "yaw", "angle"]
                    assert item["section"] in {"action", "observation_state"}
                    pose_topic, gripper_topic = expected_topics[item["group"]]
                    assert item["pose_topic"] == pose_topic
                    assert item["pose_fields"] == expected_pose_fields
                    assert item["gripper_topic"] == gripper_topic
                    assert item["gripper_fields"] == ["angle"]
                    print(
                        f"[步骤{substep}][{time_key}][{item['section']}][{item['group']}] "
                        f"pose_topic={item['pose_topic']} gripper_topic={item['gripper_topic']}",
                        flush=True,
                    )
                    print(f"  labels = {item['vector_labels']}", flush=True)
                    print(f"  vector = {item['vector']}", flush=True)
                allure.attach(
                    json.dumps(result, ensure_ascii=False, indent=2),
                    name=f"步骤{substep}-MCAP七维向量",
                    attachment_type=allure.attachment_type.JSON,
                )
        return results

    def _extract_parquet_vectors(self, workflow_step: str) -> dict:
        assert self.parquet_sources, "步骤9未发现远端 parquet 文件"
        results = {}
        for substep, time_key, time_label, target_ns in (
            (f"{workflow_step}.1", "start", "开始时间", self._resolve_l1_start_time_ns()),
            (f"{workflow_step}.2", "end", "结束时间", self._resolve_l1_end_time_ns()),
        ):
            with allure.step(f"步骤{substep}：提取L1标注{time_label} parquet 机器人向量"):
                result = extract_parquet_robot_vectors_at_time(
                    parquet_path=self.parquet_sources,
                    target_ns=target_ns,
                )
                results[time_key] = result
                vectors = result.get("vectors", [])
                assert len(vectors) == 4, "预期提取 action/observation_state 左右共4组向量"
                for item in vectors:
                    assert item["vector_labels"] == ["x", "y", "z", "roll", "pitch", "yaw", "angle"]
                    assert len(item["vector"]) == 7
                    print(
                        f"[步骤{substep}][{time_key}][{item['section']}][{item['group']}] "
                        f"target_ns={item['target_ns']} matched_timestamp_ns={item['matched_timestamp_ns']} "
                        f"diff_ns={item['diff_ns']} parquet_column={item['parquet_column']}",
                        flush=True,
                    )
                    print(f"  labels = {item['vector_labels']}", flush=True)
                    print(f"  vector = {item['vector']}", flush=True)
                allure.attach(
                    json.dumps(result, ensure_ascii=False, indent=2),
                    name=f"步骤{substep}-Parquet机器人向量",
                    attachment_type=allure.attachment_type.JSON,
                )
        return results

    def _compare_vectors(self, workflow_step: str) -> None:
        failures = []
        results = {}
        for substep, time_key, time_label in (
            (f"{workflow_step}.1", "start", "开始时间"),
            (f"{workflow_step}.2", "end", "结束时间"),
        ):
            with allure.step(f"步骤{substep}：对比L1标注{time_label}可用向量维度"):
                try:
                    mcap_result = (self.mcap_vector_extract_results or {}).get(time_key)
                    parquet_result = (self.parquet_vector_extract_results or {}).get(time_key)
                    assertions.assert_is_not_none(mcap_result, f"未提取{time_label} MCAP 向量")
                    assertions.assert_is_not_none(parquet_result, f"未提取{time_label} parquet 向量")
                    comparison = compare_robot_vectors(mcap_result=mcap_result, parquet_result=parquet_result)
                    results[time_key] = comparison
                    for item in comparison.get("comparisons", []):
                        print(
                            f"[步骤{substep}][{time_key}][{item['section']}][{item['group']}] "
                            f"consistent={item['is_consistent']}",
                            flush=True,
                        )
                        for dimension in item.get("dimensions", []):
                            print(
                                f"  {dimension['label']}: mcap={dimension['mcap_value']} "
                                f"parquet={dimension['parquet_value']} "
                                f"diff={dimension['absolute_difference']} "
                                f"consistent={dimension['is_consistent']}",
                                flush=True,
                            )
                            if not dimension["is_consistent"]:
                                failures.append(
                                    f"{time_label}/{item['section']}/{item['group']}/{dimension['label']}: "
                                    f"MCAP={dimension['mcap_value']}, Parquet={dimension['parquet_value']}, "
                                    f"差值={dimension['absolute_difference']}"
                                )
                except (AssertionError, FileNotFoundError, OSError, ValueError) as exc:
                    results[time_key] = {"error": str(exc)}
                    failures.append(f"{time_label}机器人向量比较失败: {exc}")
        allure.attach(
            json.dumps(results, ensure_ascii=False, indent=2),
            name=f"步骤{workflow_step}-L1机器人向量对比汇总",
            attachment_type=allure.attachment_type.JSON,
        )
        assert not failures, "L1 标注机器人向量不一致:\n" + "\n".join(failures)

    @classmethod
    def _discover_remote_parquet_sources(cls):
        parquet_prefix = (
            f"{cls.task_no}/final-data/lerobot_v3/data/chunk-000"
        )
        cls.parquet_store = S3ParquetStore(cls.s3_config, parquet_prefix)
        cls.parquet_sources = cls.parquet_store.list_parquet_sources()
        return cls.parquet_sources

    @pytest.mark.order(1)
    @allure.story("步骤1：查询任务列表")
    def test_query_task_info(self):
        with allure.step("步骤1：查询任务列表"):
            allure.attach(
                f"execution_env={self.execution_env}\nsection=[{self.section}]\n"
                f"Task_no={self.task_no}\n"
                f"mcap_source={self.mcap_source_label}\n"
                f"mcap_count={len(self.mcap_sources)}\n"
                f"mcap_total_size={sum(source.size_bytes for source in self.mcap_sources)} bytes",
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
                pytest.fail(f"未找到 task_no 为 '{self.task_no}' 的任务")

            TestShakeV3Verify.task_name = target_task.get("name")
            TestShakeV3Verify.task_status_label = target_task.get("status_label")
            TestShakeV3Verify.task_id = target_task.get("id")
            TestShakeV3Verify.scene_tags = target_task.get("scene_tags", [])
            print(f"[步骤1] task_name: {self.task_name}", flush=True)
            print(f"[步骤1] status_label: {self.task_status_label}", flush=True)
            print(f"[步骤1] task_id: {self.task_id}", flush=True)
            print(f"[步骤1] scene_tags: {self.scene_tags}", flush=True)
            if self.task_status_label == "已转换":
                TestShakeV3Verify.workflow_mode = "converted"
            elif self.task_status_label != "已采集":
                pytest.fail(
                    f"任务状态不符合预期: {self.task_status_label!r}，预期为 '已采集' 或 '已转换'"
                )

    @pytest.mark.order(2)
    @allure.story("步骤2：查询标注workspace")
    def test_query_annotation_workspace(self):
        with allure.step("步骤2：查询标注workspace"):
            if self.workflow_mode == "converted":
                pytest.skip("任务已转换，跳过标注流程")
            response = self.api_all.query_annotation_workspace(task_id=self.task_id)
            assertions.assert_code(response.status_code, 200)
            data = response.json().get("data", {})
            assertions.assert_text(str(data.get("task_no", "")).strip(), self.task_no)
            TestShakeV3Verify.annotation_id = data.get("annotation", {}).get("annotation_id")
            TestShakeV3Verify.episode_id = data.get("episode", {}).get("episode_id")
            print(f"[步骤2] config task_no: {self.task_no}", flush=True)
            print(f"[步骤2] workspace task_no: {data.get('task_no')}", flush=True)
            print(f"[步骤2] annotation_id: {self.annotation_id}", flush=True)
            print(f"[步骤2] episode_id: {self.episode_id}", flush=True)
            assertions.assert_is_not_none(self.annotation_id, "annotation_id 不能为空")
            assertions.assert_is_not_none(self.episode_id, "episode_id 不能为空")

    @pytest.mark.order(3)
    @allure.story("步骤3：新增L1层标注")
    def test_create_l1_annotation_segment(self):
        with allure.step("步骤3：新增L1层标注"):
            if self.workflow_mode == "converted":
                pytest.skip("任务已转换，跳过标注流程")
            assertions.assert_is_not_none(self.annotation_id, "未查询到 annotation_id")

            tag_vocabulary = self.scene_tags or []
            selected_scene_tag = random.choice(tag_vocabulary) if tag_vocabulary else None
            relative_start_ns = START_TIME_NS - BASELINE_START_TIME_NS
            relative_end_ns = END_TIME_NS - BASELINE_START_TIME_NS
            segment_id = generate_episode_snowflake_id()
            segment = {
                "segmentId": segment_id,
                "id": segment_id,
                "layerId": "l1",
                "category": "episode",
                "description": ANNOTATION_DESCRIPTION,
                "prompt": ANNOTATION_DESCRIPTION,
                "attributes": {
                    "scene": "tabletop",
                    "sceneTags": [selected_scene_tag] if selected_scene_tag else [],
                },
                "startTimeNs": str(START_TIME_NS),
                "endTimeNs": str(END_TIME_NS),
                "startTimestampNs": str(START_TIME_NS),
                "endTimestampNs": str(END_TIME_NS),
                "episodeStartTimeNs": str(relative_start_ns),
                "episodeEndTimeNs": str(relative_end_ns),
                "episode_start_time": f"{relative_start_ns / 1_000_000_000:.9f}",
                "episode_end_time": f"{relative_end_ns / 1_000_000_000:.9f}",
                "timeline_start_sec": f"{relative_start_ns / 1_000_000_000:.9f}",
                "timeline_end_sec": f"{relative_end_ns / 1_000_000_000:.9f}",
                "baseline_camera_key": "gripper_fisheye_r_color",
                "startSec": relative_start_ns / 1_000_000_000,
                "endSec": relative_end_ns / 1_000_000_000,
                "start_time": f"{START_TIME_NS // 1_000_000_000}.{START_TIME_NS % 1_000_000_000:09d}",
                "end_time": f"{END_TIME_NS // 1_000_000_000}.{END_TIME_NS % 1_000_000_000:09d}",
            }
            playback = {
                "topic": self.camera_topics[1],
                "gap_policy": "skip_on_playback",
                "current_sequence": 1,
                "baseline_camera_key": "gripper_fisheye_r_color",
                "baseline_end_time_ns": str(BASELINE_END_TIME_NS),
                "baseline_camera_label": "wrist_image_right",
                "timeline_duration_sec": f"{(BASELINE_END_TIME_NS - BASELINE_START_TIME_NS) / 1_000_000_000:.9f}",
                "baseline_start_time_ns": str(BASELINE_START_TIME_NS),
            }
            TestShakeV3Verify.l1_segment = segment
            TestShakeV3Verify.submit_playback = playback
            print(f"[步骤3] annotation_id: {self.annotation_id}", flush=True)
            print(f"[步骤3] tag_vocabulary(scene_tags): {tag_vocabulary}", flush=True)
            print(f"[步骤3] selected scene tag: {selected_scene_tag}", flush=True)
            print(f"[步骤3] generated segmentId/id: {segment_id}", flush=True)
            allure.attach(
                json.dumps({"segments": [segment], "playback": playback}, ensure_ascii=False, indent=2),
                name="步骤3请求参数",
                attachment_type=allure.attachment_type.JSON,
            )
            response = self.api_all.create_task_annotation_segments(
                annotation_id=self.annotation_id,
                tag_vocabulary=tag_vocabulary,
                segments=[segment],
                playback=playback,
            )
            assertions.assert_code(response.status_code, 200)
            assertions.assert_text(response.json().get("msg", ""), "success")

    @pytest.mark.order(4)
    @allure.story("步骤4：提交仅包含L1层的标注")
    def test_submit_l1_annotation(self):
        with allure.step("步骤4：提交仅包含L1层的标注"):
            if self.workflow_mode == "converted":
                pytest.skip("任务已转换，跳过标注流程")
            assertions.assert_is_not_none(self.annotation_id, "缺少 annotation_id")
            assertions.assert_is_not_none(self.episode_id, "缺少 episode_id")
            assertions.assert_is_not_none(self.l1_segment, "缺少 L1 分段")
            assertions.assert_is_not_none(self.submit_playback, "缺少 playback")

            tag_vocabulary = self.scene_tags or []
            l1_segments = [self.l1_segment]
            annotation_json = {
                "layers": [
                    {
                        "name": "L1 Episode",
                        "type": "episode",
                        "layerId": "l1",
                        "segments": l1_segments,
                    }
                ],
                "playback": self.submit_playback,
                "segments": l1_segments,
                "timeUnit": "source_timestamp_sec",
                "episodeId": str(self.episode_id),
                "tagVocabulary": tag_vocabulary,
                "episodeTimeUnit": "episode_sec",
            }
            TestShakeV3Verify.submitted_annotation_json = annotation_json
            print(f"[步骤4] annotation_id: {self.annotation_id}", flush=True)
            print(f"[步骤4] episode_id: {self.episode_id}", flush=True)
            print(f"[步骤4] tag_vocabulary(scene_tags): {tag_vocabulary}", flush=True)
            allure.attach(
                json.dumps(annotation_json, ensure_ascii=False, indent=2),
                name="步骤4提交标注JSON",
                attachment_type=allure.attachment_type.JSON,
            )
            response = self.api_all.submit_task_annotation(
                annotation_id=self.annotation_id,
                tag_vocabulary=tag_vocabulary,
                annotation_json=annotation_json,
            )
            assertions.assert_code(response.status_code, 200)
            assertions.assert_text(response.json().get("msg", ""), "success")

    @pytest.mark.order(5)
    @allure.story("步骤5：完成质检")
    def test_complete_task_qc(self):
        with allure.step("步骤5：完成质检"):
            if self.workflow_mode == "converted":
                pytest.skip("任务已转换，跳过质检流程")
            assertions.assert_is_not_none(self.task_id, "缺少 task_id")
            print(f"[步骤5] task_id: {self.task_id}", flush=True)
            response = self.api_all.complete_task_qc(task_id=self.task_id)
            assertions.assert_code(response.status_code, 200)
            assertions.assert_in_text(response.json(), "success")

    @pytest.mark.order(6)
    @allure.story("步骤6：发起数据转换")
    def test_create_conversion(self):
        with allure.step("步骤6：发起数据转换"):
            assertions.assert_is_not_none(self.task_id, "缺少 task_id")
            TestShakeV3Verify.target_format = "lerobot_v3"
            print(f"[步骤6] task_id: {self.task_id}", flush=True)
            print(f"[步骤6] target_format: {self.target_format}", flush=True)
            if self.workflow_mode == "converted":
                return
            response = self.api_all.create_conversion(
                task_id=self.task_id,
                target_format=self.target_format,
                quality_labels=[],
            )
            assertions.assert_code(response.status_code, 200)
            assertions.assert_in_text(response.json(), "success")

    @pytest.mark.order(7)
    @allure.story("步骤7：轮询正在转换列表")
    def test_poll_active_conversion(self):
        with allure.step("步骤7：轮询正在转换列表"):
            if self.workflow_mode == "converted":
                pytest.skip("任务已转换，跳过转换轮询")
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
                entry = find_conversion_entry(conversion_list, self.task_id)
                print(f"[步骤7] 第{attempt}次轮询，active列表条数: {len(conversion_list)}", flush=True)
                if entry is None:
                    print(f"[步骤7] active列表未找到 task_id={self.task_id}，进入步骤8", flush=True)
                    TestShakeV3Verify.active_conversion_entry = None
                    break
                TestShakeV3Verify.active_conversion_entry = entry
                status = str(entry.get("status", "")).strip().lower()
                last_status = status
                print(f"[步骤7] 找到task_id={self.task_id}，status={status}", flush=True)
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
                pytest.skip("任务已转换，跳过完成列表确认")
            response = self.api_all.query_finished_conversion_list()
            assertions.assert_code(response.status_code, 200)
            response_data = response.json()
            assertions.assert_text(response_data.get("msg", ""), "success")
            conversion_list = response_data.get("data", {}).get("list", [])
            entry = find_conversion_entry(conversion_list, self.task_id)
            print(f"[步骤8] finished列表条数: {len(conversion_list)}", flush=True)
            assertions.assert_is_not_none(entry, f"finished 列表中未找到 task_id={self.task_id}")
            assert str(entry.get("status", "")).strip().lower() == "completed", f"转换未完成: {entry}"
            print(f"[步骤8] 找到task_id={self.task_id}，status=completed", flush=True)
            TestShakeV3Verify.finished_conversion_entry = entry

    @pytest.mark.order(9)
    @allure.story("步骤9：发现S3中的全部V3 parquet文件")
    def test_discover_remote_parquet_files(self):
        with allure.step("步骤9：枚举chunk-000中的全部远端parquet文件"):
            if self.workflow_mode != "converted":
                assertions.assert_is_not_none(self.finished_conversion_entry, "尚未确认转换完成")
            assertions.assert_is_not_none(self.target_format, "缺少转换格式")
            assert self.target_format == "lerobot_v3", (
                f"摇操转换格式应为 lerobot_v3，实际为 {self.target_format!r}"
            )
            parquet_sources = self._discover_remote_parquet_sources()
            assert parquet_sources, "S3 chunk-000 中未发现 parquet 文件"
            print(f"[步骤9] task_no: {self.task_no}", flush=True)
            print(f"[步骤9] parquet_source: {self.parquet_store.source_label}", flush=True)
            print(f"[步骤9] parquet_count: {len(parquet_sources)}", flush=True)
            for index, source in enumerate(parquet_sources, start=1):
                print(
                    f"[步骤9][Parquet {index}/{len(parquet_sources)}] "
                    f"{source.object_name}, size={source.size_bytes} bytes",
                    flush=True,
                )
            allure.attach(
                "\n".join(str(source) for source in parquet_sources),
                name="步骤9远端parquet文件列表",
                attachment_type=allure.attachment_type.TEXT,
            )

    @pytest.mark.order(10)
    @allure.story("步骤10：按L1标注开始和结束时间提取parquet图片")
    def test_extract_l1_parquet_images(self):
        with allure.step("步骤10：按L1标注开始和结束时间提取parquet图片"):
            self._prepare_image_output_dirs_once()
            TestShakeV3Verify.parquet_image_extract_result = self._extract_parquet_images("10")

    @pytest.mark.order(11)
    @allure.story("步骤11：按L1标注开始和结束时间从S3 MCAP提取图片")
    def test_extract_l1_mcap_images(self):
        with allure.step("步骤11：按L1标注开始和结束时间从S3 MCAP提取图片"):
            self._prepare_image_output_dirs_once()
            TestShakeV3Verify.mcap_image_extract_result = self._extract_mcap_images("11")

    @pytest.mark.order(12)
    @allure.story("步骤12：校验L1标注开始和结束时间的左右相机图片一致性")
    def test_compare_l1_camera_images(self):
        with allure.step("步骤12：校验L1标注开始和结束时间的左右相机图片一致性"):
            self._compare_images("12")

    @pytest.mark.order(13)
    @allure.story("步骤13：提取L1标注开始和结束时间的MCAP与parquet机器人向量")
    def test_extract_l1_robot_vectors(self):
        with allure.step("步骤13：提取L1标注开始和结束时间的MCAP与parquet机器人向量"):
            try:
                TestShakeV3Verify.mcap_vector_extract_results = self._extract_mcap_vectors("13.1")
            except (
                AssertionError,
                AttributeError,
                FileNotFoundError,
                KeyError,
                OSError,
                TypeError,
                ValueError,
            ) as exc:
                error_message = f"MCAP 原始数据校验失败: {exc}"
                allure.attach(
                    error_message,
                    name="步骤13.1-MCAP原始数据失败原因",
                    attachment_type=allure.attachment_type.TEXT,
                )
                raise AssertionError(error_message) from exc

            try:
                TestShakeV3Verify.parquet_vector_extract_results = self._extract_parquet_vectors("13.2")
            except (
                AssertionError,
                FileNotFoundError,
                OSError,
                TypeError,
                ValueError,
            ) as exc:
                error_message = f"Parquet 转换数据校验失败: {exc}"
                allure.attach(
                    error_message,
                    name="步骤13.2-Parquet转换数据失败原因",
                    attachment_type=allure.attachment_type.TEXT,
                )
                raise AssertionError(error_message) from exc

    @pytest.mark.order(14)
    @allure.story("步骤14：对比L1标注开始和结束时间的MCAP与parquet七维向量")
    def test_compare_l1_robot_vectors(self):
        with allure.step("步骤14：对比L1标注开始和结束时间的MCAP与parquet七维向量"):
            self._compare_vectors("14")

    @pytest.mark.order(15)
    @allure.story("步骤15：校验parquet中的L1/L2/L3标注数据")
    def test_compare_parquet_annotations(self):
        with allure.step("步骤15：解析parquet并与代码中提交的L1/L2/L3标注逐项对比"):
            assert self.parquet_sources, "步骤9未发现远端 parquet 文件"
            expected_layers = self._expected_annotation_layers()
            result = compare_parquet_annotations(
                parquet_path=self.parquet_sources,
                expected_layers=expected_layers,
                validate_l1_playback_duration=True,
            )
            TestShakeV3Verify.parquet_annotation_validation_result = result
            print(
                f"[步骤15] annotation consistent={result['is_consistent']} "
                f"failure_count={len(result.get('failures', []))}",
                flush=True,
            )
            for comparison_index, comparison in enumerate(
                result["l1_playback_duration_comparisons"], start=1
            ):
                print(
                    f"[步骤15] "
                    f"{format_l1_playback_duration_comparison(comparison, comparison_index)}",
                    flush=True,
                )
            allure.attach(
                json.dumps(expected_layers, ensure_ascii=False, indent=2),
                name="步骤15-代码中的期望标注",
                attachment_type=allure.attachment_type.JSON,
            )
            allure.attach(
                json.dumps(result, ensure_ascii=False, indent=2),
                name="步骤15-parquet标注校验结果",
                attachment_type=allure.attachment_type.JSON,
            )
            assert result["is_consistent"], "parquet 标注数据与代码中提交的数据不一致:\n" + "\n".join(
                result["failures"]
            )

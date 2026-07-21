"""
夹爪数据V2任务校验流程
"""

import configparser
import json
import random
import shutil
import time
from pathlib import Path

import allure
import pytest

from api import all_api
from common import Assert
from common.client_factory import create_lazy_yixiu_client
from common.extract_mcap_image import extract_global_nearest_image_from_mcap_directory
from common.extract_mcap_fields import extract_robot_vectors_at_time
from common.image_compare import compare_images
from common.extract_parquet_fields import compare_robot_vectors, extract_parquet_robot_vectors_at_time
from common.parse_parquet_file import extract_nearest_parquet_images, parse_parquet_file
from common.parquet_download_service import download_converted_parquet

NANOSECONDS_PER_SECOND = 1_000_000_000
TASK_ANNOTATION_CONFIG = {
    "baseline_start_time_ns": 1776150106321320964,
    "baseline_end_time_ns": 1776150323218997544,
    "segments": {
        "l1": {
            "description": "拿笔放笔",
            "startTimeNs": 1776150112347320964,
            "endTimeNs": 1776150118194320964,
        },
        "l2_first": {
            "description": "移动夹笔",
            "startTimeNs": 1776150112601320964,
            "endTimeNs": 1776150114616320964,
        },
        "l2_second": {
            "description": "夹住放回",
            "startTimeNs": 1776150114616320964,
            "endTimeNs": 1776150117885320964,
        },
    },
}

assertions = Assert.Assertions()
_last_snowflake_13 = 0


def load_task_context():
    """读取执行环境并解析当前环境下的任务上下文。"""
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

    task_no = config.get(section, "Task_no", fallback="").strip()
    if not task_no:
        raise ValueError(f"配置文件节 [{section}] 中缺少 Task_no")

    mcap_path = config.get(section, "mcap_path", fallback="").strip()

    return execution_env, section, task_no, mcap_path


def generate_episode_snowflake_id():
    """生成 episode_13位雪花ID（毫秒时间戳）。"""
    global _last_snowflake_13
    now_ms = int(time.time() * 1000)
    if now_ms <= _last_snowflake_13:
        now_ms = _last_snowflake_13 + 1
    _last_snowflake_13 = now_ms
    return f"episode_{now_ms}"


def generate_seg_snowflake_id():
    """生成 seg_13位雪花ID（毫秒时间戳）。"""
    global _last_snowflake_13
    now_ms = int(time.time() * 1000)
    if now_ms <= _last_snowflake_13:
        now_ms = _last_snowflake_13 + 1
    _last_snowflake_13 = now_ms
    return f"seg_{now_ms}"


def find_conversion_entry(conversion_list, task_id):
    """从转换列表中按 task_id 查找对应条目。"""
    for item in conversion_list or []:
        if str(item.get("task_id", "")).strip() == str(task_id):
            return item
    return None


def _get_task_time_range_ns() -> tuple[int, int]:
    try:
        baseline_start_time_ns = int(TASK_ANNOTATION_CONFIG["baseline_start_time_ns"])
        baseline_end_time_ns = int(TASK_ANNOTATION_CONFIG["baseline_end_time_ns"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("任务基准时间必须包含有效的起止纳秒值") from exc
    if baseline_start_time_ns > baseline_end_time_ns:
        raise ValueError("baseline_start_time_ns 不能晚于 baseline_end_time_ns")
    return baseline_start_time_ns, baseline_end_time_ns


def build_annotation_time_fields(segment_key: str) -> dict:
    """由任务基准时间与 segment 的绝对纳秒生成全部标注时间字段。"""
    try:
        segment_time = TASK_ANNOTATION_CONFIG["segments"][segment_key]
        start_time_ns = int(segment_time["startTimeNs"])
        end_time_ns = int(segment_time["endTimeNs"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"标注 {segment_key!r} 缺少有效的绝对起止纳秒值") from exc

    baseline_start_time_ns, baseline_end_time_ns = _get_task_time_range_ns()
    if start_time_ns > end_time_ns:
        raise ValueError("startTimeNs 不能晚于 endTimeNs")
    if start_time_ns < baseline_start_time_ns or end_time_ns > baseline_end_time_ns:
        raise ValueError(f"标注 {segment_key!r} 的绝对时间超出任务时间轴范围")

    episode_start_time_ns = start_time_ns - baseline_start_time_ns
    episode_end_time_ns = end_time_ns - baseline_start_time_ns

    episode_start_sec = episode_start_time_ns / NANOSECONDS_PER_SECOND
    episode_end_sec = episode_end_time_ns / NANOSECONDS_PER_SECOND

    def format_ns_as_seconds(value_ns: int) -> str:
        seconds, nanoseconds = divmod(value_ns, NANOSECONDS_PER_SECOND)
        return f"{seconds}.{nanoseconds:09d}"

    return {
        "startTimeNs": str(start_time_ns),
        "endTimeNs": str(end_time_ns),
        "startTimestampNs": str(start_time_ns),
        "endTimestampNs": str(end_time_ns),
        "episodeStartTimeNs": str(episode_start_time_ns),
        "episodeEndTimeNs": str(episode_end_time_ns),
        "episode_start_time": f"{episode_start_sec:.9f}",
        "episode_end_time": f"{episode_end_sec:.9f}",
        "timeline_start_sec": f"{episode_start_sec:.9f}",
        "timeline_end_sec": f"{episode_end_sec:.9f}",
        "startSec": episode_start_sec,
        "endSec": episode_end_sec,
        "start_time": format_ns_as_seconds(start_time_ns),
        "end_time": format_ns_as_seconds(end_time_ns),
    }


def build_annotation_text_fields(segment_key: str) -> dict:
    """使用同一份描述生成 description 和 prompt。"""
    try:
        description = str(
            TASK_ANNOTATION_CONFIG["segments"][segment_key]["description"]
        ).strip()
    except (KeyError, TypeError) as exc:
        raise ValueError(f"标注 {segment_key!r} 缺少 description") from exc
    if not description:
        raise ValueError(f"标注 {segment_key!r} 的 description 不能为空")
    return {"description": description, "prompt": description}


def build_playback(current_sequence: int) -> dict:
    """由统一任务时间配置生成 playback 数据。"""
    baseline_start_time_ns, baseline_end_time_ns = _get_task_time_range_ns()
    timeline_duration_sec = (
        baseline_end_time_ns - baseline_start_time_ns
    ) / NANOSECONDS_PER_SECOND
    return {
        "topic": "/sensor/camera_fisheye_r/color/image_raw",
        "gap_policy": "skip_on_playback",
        "current_sequence": current_sequence,
        "baseline_camera_key": "sensor_fisheye_r_color",
        "baseline_end_time_ns": str(baseline_end_time_ns),
        "baseline_camera_label": "wrist_image_right",
        "timeline_duration_sec": f"{timeline_duration_sec:.9f}",
        "baseline_start_time_ns": str(baseline_start_time_ns),
    }


global_client = create_lazy_yixiu_client()


@pytest.mark.annotation_v2
@allure.feature("场景：夹爪数据-V2格式信息校验")
class TestV2Verify:
    @classmethod
    def setup_class(cls):
        """初始化接口封装实例与配置上下文。"""
        cls.api_all = all_api.ApiAll(global_client)
        cls.execution_env, cls.section, cls.task_no, cls.mcap_path = load_task_context()
        cls.task_name = None
        cls.task_status_label = None
        cls.workflow_mode = "normal"
        cls.task_id = None
        cls.scene_tags = None
        cls.annotation_id = None
        cls.episode_id = None
        cls.l1_segment = None
        cls.l2_segment_first = None
        cls.l2_segment_second = None
        cls.submit_playback = None
        cls.target_format = None
        cls.active_conversion_entry = None
        cls.finished_conversion_entry = None
        cls.downloaded_parquet_file = None
        cls.parquet_image_extract_result = None
        cls.mcap_image_extract_result = None
        cls.step4_parquet_image_extract_result = None
        cls.step4_mcap_image_extract_result = None
        cls.step5_parquet_image_extract_result = None
        cls.step5_mcap_image_extract_result = None
        cls.step3_pose_field_extract_results = None
        cls.step3_parquet_pose_field_extract_results = None
        cls.step4_pose_field_extract_results = None
        cls.step4_parquet_pose_field_extract_results = None
        cls.step5_pose_field_extract_results = None
        cls.step5_parquet_pose_field_extract_results = None
        cls.image_output_dirs_prepared = False

    def _resolve_step3_start_time_ns(self) -> int:
        """优先取步骤3生成的 startTimeNs；步骤3被跳过时使用默认值。"""
        if isinstance(self.l1_segment, dict):
            start_time_ns = self.l1_segment.get("startTimeNs", "")
            if start_time_ns != "":
                try:
                    return int(str(start_time_ns))
                except (TypeError, ValueError):
                    pass
        return int(build_annotation_time_fields("l1")["startTimeNs"])

    def _resolve_step3_end_time_ns(self) -> int:
        """优先取步骤3生成的 endTimeNs；步骤3被跳过时使用默认值。"""
        if isinstance(self.l1_segment, dict):
            end_time_ns = self.l1_segment.get("endTimeNs", "")
            if end_time_ns != "":
                try:
                    return int(str(end_time_ns))
                except (TypeError, ValueError):
                    pass
        return int(build_annotation_time_fields("l1")["endTimeNs"])

    @staticmethod
    def _resolve_segment_time_ns(segment: dict | None, field_name: str, fallback: int) -> int:
        if isinstance(segment, dict):
            value = segment.get(field_name, "")
            if value != "":
                try:
                    return int(str(value))
                except (TypeError, ValueError):
                    pass
        return fallback

    def _resolve_step4_start_time_ns(self) -> int:
        return self._resolve_segment_time_ns(
            self.l2_segment_first,
            "startTimeNs",
            int(build_annotation_time_fields("l2_first")["startTimeNs"]),
        )

    def _resolve_step4_end_time_ns(self) -> int:
        return self._resolve_segment_time_ns(
            self.l2_segment_first,
            "endTimeNs",
            int(build_annotation_time_fields("l2_first")["endTimeNs"]),
        )

    def _resolve_step5_start_time_ns(self) -> int:
        return self._resolve_segment_time_ns(
            self.l2_segment_second,
            "startTimeNs",
            int(build_annotation_time_fields("l2_second")["startTimeNs"]),
        )

    def _resolve_step5_end_time_ns(self) -> int:
        return self._resolve_segment_time_ns(
            self.l2_segment_second,
            "endTimeNs",
            int(build_annotation_time_fields("l2_second")["endTimeNs"]),
        )

    def _prepare_image_output_dirs_once(self):
        """首次进入图片提取步骤时，清空图片提取与比较输出目录。"""
        if TestV2Verify.image_output_dirs_prepared:
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

        TestV2Verify.image_output_dirs_prepared = True

    @staticmethod
    def _find_parquet_image(extract_result: dict, side: str) -> Path:
        """从 parquet 提取结果中定位指定相机图片。"""
        for item in extract_result.get("saved_files", []):
            if item.get("side") == side and item.get("saved_path"):
                return Path(item["saved_path"])
        raise AssertionError(f"parquet 提取结果中未找到 {side} 相机图片")

    @staticmethod
    def _find_mcap_image(extract_result: dict, side: str) -> Path:
        """从 mcap 提取结果中定位指定相机的 PNG 预览图。"""
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

    def _extract_parquet_images_for_annotation(
        self,
        annotation_key: str,
        annotation_label: str,
        workflow_step: str,
        start_time_ns: int,
        end_time_ns: int,
    ) -> dict:
        assertions.assert_is_not_none(self.downloaded_parquet_file, f"步骤11未下载 parquet 文件，无法执行步骤{workflow_step}")
        parquet_path = Path(self.downloaded_parquet_file)
        assert parquet_path.exists(), f"步骤11下载的 parquet 文件不存在: {parquet_path}"
        parquet_info = parse_parquet_file(parquet_path=parquet_path, preview_rows=1)
        allure.attach(
            json.dumps(
                {
                    "file": parquet_info.get("file"),
                    "num_rows": parquet_info.get("num_rows"),
                    "num_columns": parquet_info.get("num_columns"),
                    "startTimeNs": start_time_ns,
                    "endTimeNs": end_time_ns,
                },
                ensure_ascii=False,
            ),
            name=f"步骤{workflow_step}-{annotation_label}parquet基础信息与匹配参数",
            attachment_type=allure.attachment_type.TEXT,
        )

        output_dir = Path(__file__).resolve().parent.parent / "parquet_image"
        extract_results = {}
        for substep, result_key, time_key, time_label, target_ns in (
            (f"{workflow_step}.1", "start_time_extract", "start", "开始时间", start_time_ns),
            (f"{workflow_step}.2", "end_time_extract", "end", "结束时间", end_time_ns),
        ):
            with allure.step(f"步骤{substep}：提取{annotation_label}{time_label}parquet图片（左/右各一张）"):
                extract_result = extract_nearest_parquet_images(
                    parquet_path=parquet_path,
                    target_ns=target_ns,
                    output_dir=output_dir,
                    name_prefix=self.task_no,
                    extra_name_parts=[annotation_key, time_key],
                )
                extract_results[result_key] = extract_result
                allure.attach(
                    json.dumps(extract_result, ensure_ascii=False, indent=2),
                    name=f"步骤{substep}-{annotation_label}{time_label}图片提取结果",
                    attachment_type=allure.attachment_type.TEXT,
                )
                saved_files = extract_result.get("saved_files", [])
                assert saved_files, f"步骤{substep}未生成任何 parquet 图片文件"
                for file_info in saved_files:
                    saved_path = Path(file_info.get("saved_path", ""))
                    assert saved_path.exists(), f"步骤{substep}提取的文件不存在: {saved_path}"
        return extract_results

    def _extract_mcap_images_for_annotation(
        self,
        annotation_key: str,
        annotation_label: str,
        workflow_step: str,
        start_time_ns: int,
        end_time_ns: int,
    ) -> dict:
        assertions.assert_is_not_none(self.mcap_path, f"配置文件未提供 mcap_path，无法执行步骤{workflow_step}")
        mcap_dir = Path(self.mcap_path)
        assert mcap_dir.exists(), f"配置中的 mcap_path 不存在: {mcap_dir}"
        assert mcap_dir.is_dir(), f"配置中的 mcap_path 不是文件夹: {mcap_dir}"
        topics = [
            "/sensor/camera_fisheye_l/color/image_raw",
            "/sensor/camera_fisheye_r/color/image_raw",
        ]
        output_dir = Path(__file__).resolve().parent.parent / "mcap_image"
        allure.attach(
            f"mcap_dir={mcap_dir}\nstartTimeNs={start_time_ns}\nendTimeNs={end_time_ns}\ntopics={topics}",
            name=f"步骤{workflow_step}-{annotation_label}提取参数",
            attachment_type=allure.attachment_type.TEXT,
        )

        extract_results = {}
        for substep, result_key, time_key, time_label, target_ns in (
            (f"{workflow_step}.1", "start_time_extract", "start", "开始时间", start_time_ns),
            (f"{workflow_step}.2", "end_time_extract", "end", "结束时间", end_time_ns),
        ):
            with allure.step(f"步骤{substep}：提取{annotation_label}{time_label}mcap图片（左/右各一张）"):
                extract_result = extract_global_nearest_image_from_mcap_directory(
                    mcap_dir=mcap_dir,
                    topics=topics,
                    target_ns=target_ns,
                    output_dir=output_dir,
                    name_prefix=self.task_no,
                    extra_name_parts=[annotation_key, time_key],
                )
                extract_results[result_key] = extract_result
                allure.attach(
                    json.dumps(extract_result, ensure_ascii=False, indent=2),
                    name=f"步骤{substep}-{annotation_label}{time_label}图片提取结果",
                    attachment_type=allure.attachment_type.TEXT,
                )
                saved_files = []
                for item in extract_result.get("results", []):
                    saved_files.extend([Path(path) for path in item.get("saved_paths", [])])
                assert saved_files, f"步骤{substep}未生成任何 mcap 图片文件"
                for saved_path in saved_files:
                    assert saved_path.exists(), f"步骤{substep}提取的 mcap 文件不存在: {saved_path}"
        return extract_results

    def _compare_camera_pair(
        self,
        mcap_extract_result: dict,
        parquet_extract_result: dict,
        time_key: str,
        time_label: str,
        side: str,
        side_label: str,
        target_ns: int,
        substep: str,
        annotation_key: str,
    ) -> tuple[dict, str]:
        """比较一组 MCAP/Parquet 相机图片并附加 Allure 证据。"""
        parquet_image = self._find_parquet_image(parquet_extract_result, side)
        mcap_image = self._find_mcap_image(mcap_extract_result, side)
        assert parquet_image.is_file(), f"parquet {side_label}图片不存在: {parquet_image}"
        assert mcap_image.is_file(), f"mcap {side_label}图片不存在: {mcap_image}"

        project_root = Path(__file__).resolve().parent.parent
        diff_path = project_root / "image_compare" / f"{self.task_no}_{annotation_key}_{time_key}_{side}_diff_{target_ns}.png"
        comparison = compare_images(
            reference_path=mcap_image,
            candidate_path=parquet_image,
            diff_output_path=diff_path,
        )
        comparison["time_point"] = time_key
        comparison["camera_side"] = side
        comparison["target_ns"] = target_ns

        attachment_prefix = f"步骤{substep}-{annotation_key}-{time_label}{side_label}"
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
                name=f"{attachment_prefix}-像素差异图(差异放大4倍)",
                attachment_type=allure.attachment_type.PNG,
            )
        allure.attach(
            json.dumps(comparison, ensure_ascii=False, indent=2),
            name=f"{attachment_prefix}-一致性指标",
            attachment_type=allure.attachment_type.JSON,
        )

        if not comparison["dimension_match"]:
            result_message = (
                f"{time_label}{side_label}图片尺寸不一致："
                f"MCAP={comparison['reference_size']}，"
                f"Parquet={comparison['candidate_size']}"
            )
        else:
            result_message = (
                f"{time_label}{side_label}图片一致性={comparison['is_consistent']}，"
                f"容差内像素占比={comparison['similarity_percent']:.4f}%，"
                f"平均绝对误差={comparison['mean_absolute_error']:.4f}，"
                f"尺寸={comparison['reference_size']}"
            )

        print(f"[步骤{substep}] {result_message}")
        allure.attach(
            result_message,
            name=f"{attachment_prefix}-结果",
            attachment_type=allure.attachment_type.TEXT,
        )
        return comparison, result_message

    def _compare_annotation_image_results(
        self,
        parquet_results: dict,
        mcap_results: dict,
        annotation_key: str,
        annotation_label: str,
        workflow_step: str,
        start_time_ns: int,
        end_time_ns: int,
    ) -> None:
        comparison_cases = [
            (f"{workflow_step}.1", "start_time_extract", "start", "开始时间", "left", "左相机", start_time_ns),
            (f"{workflow_step}.2", "start_time_extract", "start", "开始时间", "right", "右相机", start_time_ns),
            (f"{workflow_step}.3", "end_time_extract", "end", "结束时间", "left", "左相机", end_time_ns),
            (f"{workflow_step}.4", "end_time_extract", "end", "结束时间", "right", "右相机", end_time_ns),
        ]
        comparison_results = {}
        failures = []
        for substep, extract_key, time_key, time_label, side, side_label, target_ns in comparison_cases:
            case_name = f"{annotation_label}{time_label}{side_label}"
            with allure.step(f"步骤{substep}：对比{case_name}图片"):
                try:
                    parquet_extract_result = parquet_results.get(extract_key)
                    mcap_extract_result = mcap_results.get(extract_key)
                    assertions.assert_is_not_none(
                        parquet_extract_result,
                        f"未生成{case_name} parquet 图片",
                    )
                    assertions.assert_is_not_none(
                        mcap_extract_result,
                        f"未生成{case_name} mcap 图片",
                    )
                    comparison, result_message = self._compare_camera_pair(
                        mcap_extract_result=mcap_extract_result,
                        parquet_extract_result=parquet_extract_result,
                        time_key=time_key,
                        time_label=time_label,
                        side=side,
                        side_label=side_label,
                        target_ns=target_ns,
                        substep=substep,
                        annotation_key=annotation_key,
                    )
                    comparison_results[f"{time_key}_{side}"] = comparison
                    if not comparison["is_consistent"]:
                        failures.append(f"{annotation_label}{result_message}")
                except (AssertionError, FileNotFoundError, OSError, ValueError) as exc:
                    error_message = f"{case_name}图片比较失败：{exc}"
                    comparison_results[f"{time_key}_{side}"] = {"error": str(exc)}
                    failures.append(error_message)
                    allure.attach(
                        error_message,
                        name=f"步骤{substep}-{case_name}-异常",
                        attachment_type=allure.attachment_type.TEXT,
                    )

        allure.attach(
            json.dumps(comparison_results, ensure_ascii=False, indent=2),
            name=f"步骤{workflow_step}-{annotation_label}四组图片汇总",
            attachment_type=allure.attachment_type.JSON,
        )
        assert not failures, f"步骤{workflow_step}{annotation_label}图片一致性校验失败：\n" + "\n".join(failures)
        allure.attach(
            f"步骤{workflow_step}执行成功：{annotation_label}开始和结束时间的左右相机图片均一致",
            name=f"步骤{workflow_step}结果",
            attachment_type=allure.attachment_type.TEXT,
        )

    def _extract_mcap_vectors_for_annotation(
        self,
        annotation_key: str,
        annotation_label: str,
        workflow_step: str,
        start_time_ns: int,
        end_time_ns: int,
    ) -> dict:
        assertions.assert_is_not_none(self.mcap_path, f"配置文件未提供 mcap_path，无法执行步骤{workflow_step}")
        config_path = Path(__file__).resolve().parent.parent / "Robot_Configuration" / "夹爪手持构型.json"
        results = {}
        for substep, time_key, time_label, target_ns in (
            (f"{workflow_step}.1", "start", "开始时间", start_time_ns),
            (f"{workflow_step}.2", "end", "结束时间", end_time_ns),
        ):
            with allure.step(f"步骤{substep}：提取{annotation_label}{time_label}MCAP七维向量"):
                extract_result = extract_robot_vectors_at_time(
                    config_path=config_path,
                    mcap_dir=self.mcap_path,
                    target_ns=target_ns,
                )
                results[time_key] = extract_result
                vectors = extract_result.get("vectors", [])
                assert len(vectors) == 4, f"步骤{substep}预期提取4组七维向量，实际为 {len(vectors)}"
                for item in vectors:
                    print(
                        f"[步骤{substep}][{annotation_key}][{time_key}]"
                        f"[{item['section']}][{item['group']}] "
                        f"pose_topic={item['pose_topic']} "
                        f"gripper_topic={item['gripper_topic']}"
                    )
                    print(f"  labels = {item['vector_labels']}")
                    print(f"  vector = {item['vector']}")
                allure.attach(
                    json.dumps(extract_result, ensure_ascii=False, indent=2),
                    name=f"步骤{substep}-{annotation_label}{time_label}MCAP七维向量",
                    attachment_type=allure.attachment_type.JSON,
                )
        return results

    def _extract_parquet_vectors_for_annotation(
        self,
        annotation_key: str,
        annotation_label: str,
        workflow_step: str,
        start_time_ns: int,
        end_time_ns: int,
    ) -> dict:
        assertions.assert_is_not_none(self.downloaded_parquet_file, f"步骤11未下载 parquet 文件，无法执行步骤{workflow_step}")
        results = {}
        for substep, time_key, time_label, target_ns in (
            (f"{workflow_step}.1", "start", "开始时间", start_time_ns),
            (f"{workflow_step}.2", "end", "结束时间", end_time_ns),
        ):
            with allure.step(f"步骤{substep}：提取{annotation_label}{time_label}parquet七维向量"):
                extract_result = extract_parquet_robot_vectors_at_time(
                    parquet_path=self.downloaded_parquet_file,
                    target_ns=target_ns,
                )
                results[time_key] = extract_result
                vectors = extract_result.get("vectors", [])
                assert len(vectors) == 4, f"步骤{substep}预期提取4组七维向量，实际为 {len(vectors)}"
                for item in vectors:
                    print(
                        f"[步骤{substep}][{annotation_key}][{time_key}]"
                        f"[{item['section']}][{item['group']}] "
                        f"target_ns={item['target_ns']} "
                        f"matched_timestamp_ns={item['matched_timestamp_ns']} "
                        f"diff_ns={item['diff_ns']} "
                        f"parquet_column={item['parquet_column']}"
                    )
                    print(f"  labels = {item['vector_labels']}")
                    print(f"  vector = {item['vector']}")
                allure.attach(
                    json.dumps(extract_result, ensure_ascii=False, indent=2),
                    name=f"步骤{substep}-{annotation_label}{time_label}parquet七维向量",
                    attachment_type=allure.attachment_type.JSON,
                )
        return results

    def _compare_vectors_for_annotation(
        self,
        mcap_results: dict,
        parquet_results: dict,
        annotation_key: str,
        annotation_label: str,
        workflow_step: str,
    ) -> None:
        comparison_results = {}
        failures = []
        for substep, time_key, time_label in (
            (f"{workflow_step}.1", "start", "开始时间"),
            (f"{workflow_step}.2", "end", "结束时间"),
        ):
            with allure.step(f"步骤{substep}：对比{annotation_label}{time_label}七维向量"):
                try:
                    mcap_result = mcap_results.get(time_key)
                    parquet_result = parquet_results.get(time_key)
                    assertions.assert_is_not_none(mcap_result, f"未提取{annotation_label}{time_label}MCAP七维向量")
                    assertions.assert_is_not_none(parquet_result, f"未提取{annotation_label}{time_label}parquet七维向量")
                    comparison = compare_robot_vectors(mcap_result=mcap_result, parquet_result=parquet_result)
                    comparison_results[time_key] = comparison
                    for item in comparison["comparisons"]:
                        print(
                            f"[步骤{substep}][{annotation_key}][{time_key}]"
                            f"[{item['section']}][{item['group']}] consistent={item['is_consistent']}"
                        )
                        for dimension in item["dimensions"]:
                            print(
                                f"  {dimension['label']}: mcap={dimension['mcap_value']} "
                                f"parquet={dimension['parquet_value']} "
                                f"diff={dimension['absolute_difference']} "
                                f"consistent={dimension['is_consistent']}"
                            )
                            if not dimension["is_consistent"]:
                                failures.append(
                                    f"{annotation_label}{time_label}/"
                                    f"{item['section']}/{item['group']}/{dimension['label']}: "
                                    f"MCAP={dimension['mcap_value']}, "
                                    f"Parquet={dimension['parquet_value']}, "
                                    f"差值={dimension['absolute_difference']}"
                                )
                    allure.attach(
                        json.dumps(comparison, ensure_ascii=False, indent=2),
                        name=f"步骤{substep}-{annotation_label}{time_label}七维向量对比",
                        attachment_type=allure.attachment_type.JSON,
                    )
                except (AssertionError, FileNotFoundError, OSError, ValueError) as exc:
                    error_message = f"{annotation_label}{time_label}七维向量比较失败：{exc}"
                    comparison_results[time_key] = {"error": str(exc)}
                    failures.append(error_message)
                    allure.attach(
                        error_message,
                        name=f"步骤{substep}-{annotation_label}{time_label}-异常",
                        attachment_type=allure.attachment_type.TEXT,
                    )

        allure.attach(
            json.dumps(comparison_results, ensure_ascii=False, indent=2),
            name=f"步骤{workflow_step}-{annotation_label}开始结束时间七维向量汇总",
            attachment_type=allure.attachment_type.JSON,
        )
        assert not failures, f"步骤{workflow_step}{annotation_label}七维向量不一致：\n" + "\n".join(failures)

    @pytest.mark.order(1)
    @allure.story("步骤1：查询任务列表")
    def test_query_task_info(self):
        with allure.step("步骤1：查询任务列表"):
            allure.attach(
                f"execution_env={self.execution_env}\nsection=[{self.section}]\nTask_no={self.task_no}\nmcap_path={self.mcap_path}",
                name="前置配置上下文",
                attachment_type=allure.attachment_type.TEXT,
            )

            response = self.api_all.query_task_list()
            assertions.assert_code(response.status_code, 200)
            response_data = response.json()
            assertions.assert_text(response_data.get("msg", ""), "success")

            task_list = response_data.get("data", {}).get("list", [])
            target_task = None
            for task in task_list:
                if str(task.get("task_no", "")).strip() == self.task_no:
                    target_task = task
                    break

            if not target_task:
                error_msg = f"错误: 未找到 task_no 为 '{self.task_no}' 的任务"
                allure.attach(error_msg, name="任务查找失败", attachment_type=allure.attachment_type.TEXT)
                pytest.fail(error_msg)

            task_name = target_task.get("name")
            status_label = target_task.get("status_label")
            task_id = target_task.get("id")

            TestV2Verify.task_name = task_name
            TestV2Verify.task_status_label = status_label
            TestV2Verify.task_id = task_id
            TestV2Verify.scene_tags = target_task.get("scene_tags", [])

            print(f"[步骤1] task_name: {task_name}")
            print(f"[步骤1] status_label: {status_label}")
            print(f"[步骤1] task_id: {task_id}")
            print(f"[步骤1] scene_tags: {TestV2Verify.scene_tags}")

            task_info = (
                f"Task_no: {self.task_no}\n"
                f"name: {task_name}\n"
                f"status_label: {status_label}\n"
                f"id: {task_id}\n"
                f"scene_tags: {TestV2Verify.scene_tags}"
            )
            allure.attach(task_info, name="任务信息", attachment_type=allure.attachment_type.TEXT)

            if status_label == "已采集":
                allure.attach("状态校验成功：任务为已采集，步骤1通过", name="状态校验结果",
                              attachment_type=allure.attachment_type.TEXT)
            elif status_label == "已转换":
                TestV2Verify.workflow_mode = "converted"
                allure.attach(
                    "状态校验成功：任务为已转换，将跳过步骤2-7和步骤9-10，步骤11直接执行",
                    name="状态校验结果",
                    attachment_type=allure.attachment_type.TEXT,
                )
            else:
                error_msg = (
                    f"错误: 任务状态不符合预期，当前 status_label 为 '{status_label}'，"
                    f"预期为 '已采集' 或 '已转换'，脚本终止，不执行后续步骤"
                )
                allure.attach(error_msg, name="状态校验失败", attachment_type=allure.attachment_type.TEXT)
                pytest.fail(error_msg)

    @pytest.mark.order(2)
    @allure.story("步骤2：查询标注workspace")
    def test_query_annotation_workspace(self):
        with allure.step("步骤2：查询标注workspace"):
            if self.workflow_mode == "converted":
                allure.attach(
                    "任务已转换：跳过步骤2，直接进入后续下载流程",
                    name="步骤2跳过",
                    attachment_type=allure.attachment_type.TEXT,
                )
                pytest.skip("任务已转换，跳过步骤2")

            allure.attach(
                f"task_id={self.task_id}",
                name="查询参数",
                attachment_type=allure.attachment_type.TEXT,
            )

            response = self.api_all.query_annotation_workspace(task_id=self.task_id)
            assertions.assert_code(response.status_code, 200)
            response_data = response.json()
            assertions.assert_text(response_data.get("msg", ""), "success")

            data = response_data.get("data", {})
            annotation = data.get("annotation", {})
            episode = data.get("episode", {})
            workspace_task_no = str(data.get("task_no", "")).strip()
            annotation_id = annotation.get("annotation_id")
            episode_id = episode.get("episode_id")

            TestV2Verify.annotation_id = annotation_id
            TestV2Verify.episode_id = episode_id

            print(f"[步骤2] config task_no: {self.task_no}")
            print(f"[步骤2] workspace task_no: {workspace_task_no}")
            print(f"[步骤2] annotation_id: {annotation_id}")
            print(f"[步骤2] episode_id: {episode_id}")

            workspace_info = (
                f"config_task_no: {self.task_no}\n"
                f"workspace_task_no: {workspace_task_no}\n"
                f"annotation_id: {annotation_id}\n"
                f"episode_id: {episode_id}"
            )
            allure.attach(workspace_info, name="workspace字段", attachment_type=allure.attachment_type.TEXT)

            assertions.assert_text(workspace_task_no, self.task_no)
            assertions.assert_is_not_none(annotation_id, "data.annotation.annotation_id 不能为空")
            assertions.assert_is_not_none(episode_id, "data.episode.episode_id 不能为空")

            allure.attach("步骤2执行成功：已查询标注workspace", name="步骤2结果",
                          attachment_type=allure.attachment_type.TEXT)

    @pytest.mark.order(3)
    @allure.story("步骤3：新增L1层标注")
    def test_create_l1_annotation_segments(self):
        with allure.step("步骤3：新增L1层标注"):
            if self.workflow_mode == "converted":
                allure.attach(
                    "任务已转换：跳过步骤3，直接进入后续下载流程",
                    name="步骤3跳过",
                    attachment_type=allure.attachment_type.TEXT,
                )
                pytest.skip("任务已转换，跳过步骤3")

            assertions.assert_is_not_none(self.annotation_id, "步骤2未提取到 annotation_id，无法执行步骤3")

            annotation_id = self.annotation_id
            tag_vocabulary = self.scene_tags or []

            selected_scene_tag = random.choice(tag_vocabulary) if tag_vocabulary else None
            scene_tags_for_segment = [selected_scene_tag] if selected_scene_tag else []
            episode_snowflake_id = generate_episode_snowflake_id()

            segments = [
                {
                    "segmentId": episode_snowflake_id,
                    "id": episode_snowflake_id,
                    "layerId": "l1",
                    "category": "episode",
                    **build_annotation_text_fields("l1"),
                    "attributes": {
                        "scene": "tabletop",
                        "sceneTags": scene_tags_for_segment,
                    },
                    **build_annotation_time_fields("l1"),
                    "baseline_camera_key": "sensor_fisheye_r_color",
                }
            ]
            TestV2Verify.l1_segment = segments[0]

            playback = build_playback(current_sequence=1)
            TestV2Verify.submit_playback = playback

            print(f"[步骤3] annotation_id: {annotation_id}")
            print(f"[步骤3] tag_vocabulary(scene_tags): {tag_vocabulary}")
            print(f"[步骤3] selected scene tag: {selected_scene_tag}")
            print(f"[步骤3] generated segmentId/id: {episode_snowflake_id}")

            request_info = (
                f"annotation_id: {annotation_id}\n"
                f"tag_vocabulary: {tag_vocabulary}\n"
                f"selected_scene_tag: {selected_scene_tag}\n"
                f"segmentId/id: {episode_snowflake_id}\n"
                f"segments: {segments}\n"
                f"playback: {playback}"
            )
            allure.attach(request_info, name="步骤3请求参数", attachment_type=allure.attachment_type.TEXT)

            response = self.api_all.create_task_annotation_segments(
                annotation_id=annotation_id,
                tag_vocabulary=tag_vocabulary,
                segments=segments,
                playback=playback,
            )
            assertions.assert_code(response.status_code, 200)
            response_data = response.json()
            allure.attach(str(response_data), name="步骤3响应", attachment_type=allure.attachment_type.TEXT)
            assertions.assert_text(response_data.get("msg", ""), "success")

            allure.attach("步骤3执行成功：已新增L1层标注", name="步骤3结果", attachment_type=allure.attachment_type.TEXT)

    @pytest.mark.order(4)
    @allure.story("步骤4：新增L2层标注第一段")
    def test_create_l2_annotation_segment_first(self):
        with allure.step("步骤4：新增L2层标注第一段"):
            if self.workflow_mode == "converted":
                allure.attach(
                    "任务已转换：跳过步骤4，直接进入后续下载流程",
                    name="步骤4跳过",
                    attachment_type=allure.attachment_type.TEXT,
                )
                pytest.skip("任务已转换，跳过步骤4")

            assertions.assert_is_not_none(self.annotation_id, "步骤2未提取到 annotation_id，无法执行步骤4")

            annotation_id = self.annotation_id
            tag_vocabulary = self.scene_tags or []
            selected_scene_tag = random.choice(tag_vocabulary) if tag_vocabulary else None
            scene_tags_for_segment = [selected_scene_tag] if selected_scene_tag else []
            seg_snowflake_id = generate_seg_snowflake_id()

            segments = [
                {
                    "segmentId": seg_snowflake_id,
                    "id": seg_snowflake_id,
                    "layerId": "l2",
                    "category": "detail",
                    **build_annotation_text_fields("l2_first"),
                    "attributes": {
                        "scene": "tabletop",
                        "sceneTags": scene_tags_for_segment,
                    },
                    **build_annotation_time_fields("l2_first"),
                    "baseline_camera_key": "sensor_fisheye_r_color",
                }
            ]
            TestV2Verify.l2_segment_first = segments[0]

            playback = build_playback(current_sequence=2)

            print(f"[步骤4] annotation_id: {annotation_id}")
            print(f"[步骤4] tag_vocabulary(scene_tags): {tag_vocabulary}")
            print(f"[步骤4] selected scene tag: {selected_scene_tag}")
            print(f"[步骤4] generated segmentId/id: {seg_snowflake_id}")

            request_info = (
                f"annotation_id: {annotation_id}\n"
                f"tag_vocabulary: {tag_vocabulary}\n"
                f"selected_scene_tag: {selected_scene_tag}\n"
                f"segmentId/id: {seg_snowflake_id}\n"
                f"segments: {segments}\n"
                f"playback: {playback}"
            )
            allure.attach(request_info, name="步骤4请求参数", attachment_type=allure.attachment_type.TEXT)

            response = self.api_all.create_task_annotation_segments(
                annotation_id=annotation_id,
                tag_vocabulary=tag_vocabulary,
                segments=segments,
                playback=playback,
            )
            assertions.assert_code(response.status_code, 200)
            response_data = response.json()
            allure.attach(str(response_data), name="步骤4响应", attachment_type=allure.attachment_type.TEXT)
            assertions.assert_text(response_data.get("msg", ""), "success")

            allure.attach("步骤4执行成功：已新增L2层标注第一段", name="步骤4结果",
                          attachment_type=allure.attachment_type.TEXT)

    @pytest.mark.order(5)
    @allure.story("步骤5：新增L2层标注第二段")
    def test_create_l2_annotation_segment_second(self):
        with allure.step("步骤5：新增L2层标注第二段"):
            if self.workflow_mode == "converted":
                allure.attach(
                    "任务已转换：跳过步骤5，直接进入后续下载流程",
                    name="步骤5跳过",
                    attachment_type=allure.attachment_type.TEXT,
                )
                pytest.skip("任务已转换，跳过步骤5")

            assertions.assert_is_not_none(self.annotation_id, "步骤2未提取到 annotation_id，无法执行步骤5")

            annotation_id = self.annotation_id
            tag_vocabulary = self.scene_tags or []
            selected_scene_tag = random.choice(tag_vocabulary) if tag_vocabulary else None
            scene_tags_for_segment = [selected_scene_tag] if selected_scene_tag else []
            seg_snowflake_id = generate_seg_snowflake_id()

            segments = [
                {
                    "segmentId": seg_snowflake_id,
                    "id": seg_snowflake_id,
                    "layerId": "l2",
                    "category": "detail",
                    **build_annotation_text_fields("l2_second"),
                    "attributes": {
                        "scene": "tabletop",
                        "sceneTags": scene_tags_for_segment,
                    },
                    **build_annotation_time_fields("l2_second"),
                    "baseline_camera_key": "sensor_fisheye_r_color",
                }
            ]
            TestV2Verify.l2_segment_second = segments[0]

            playback = build_playback(current_sequence=2)

            print(f"[步骤5] annotation_id: {annotation_id}")
            print(f"[步骤5] tag_vocabulary(scene_tags): {tag_vocabulary}")
            print(f"[步骤5] selected scene tag: {selected_scene_tag}")
            print(f"[步骤5] generated segmentId/id: {seg_snowflake_id}")

            request_info = (
                f"annotation_id: {annotation_id}\n"
                f"tag_vocabulary: {tag_vocabulary}\n"
                f"selected_scene_tag: {selected_scene_tag}\n"
                f"segmentId/id: {seg_snowflake_id}\n"
                f"segments: {segments}\n"
                f"playback: {playback}"
            )
            allure.attach(request_info, name="步骤5请求参数", attachment_type=allure.attachment_type.TEXT)

            response = self.api_all.create_task_annotation_segments(
                annotation_id=annotation_id,
                tag_vocabulary=tag_vocabulary,
                segments=segments,
                playback=playback,
            )
            assertions.assert_code(response.status_code, 200)
            response_data = response.json()
            allure.attach(str(response_data), name="步骤5响应", attachment_type=allure.attachment_type.TEXT)
            assertions.assert_text(response_data.get("msg", ""), "success")

            allure.attach("步骤5执行成功：已新增L2层标注第二段", name="步骤5结果",
                          attachment_type=allure.attachment_type.TEXT)

    @pytest.mark.order(6)
    @allure.story("步骤6：提交标注")
    def test_submit_annotation(self):
        with allure.step("步骤6：提交标注"):
            if self.workflow_mode == "converted":
                allure.attach(
                    "任务已转换：跳过步骤6，直接进入后续下载流程",
                    name="步骤6跳过",
                    attachment_type=allure.attachment_type.TEXT,
                )
                pytest.skip("任务已转换，跳过步骤6")

            assertions.assert_is_not_none(self.annotation_id, "缺少 annotation_id，无法提交标注")
            assertions.assert_is_not_none(self.episode_id, "缺少 episode_id，无法提交标注")
            assertions.assert_is_not_none(self.l1_segment, "缺少步骤3的L1分段数据")
            assertions.assert_is_not_none(self.l2_segment_first, "缺少步骤4的L2第一段数据")
            assertions.assert_is_not_none(self.l2_segment_second, "缺少步骤5的L2第二段数据")
            assertions.assert_is_not_none(self.submit_playback, "缺少播放上下文数据")

            tag_vocabulary = self.scene_tags or []
            l2_segments = [self.l2_segment_first, self.l2_segment_second]
            annotation_json = {
                "layers": [
                    {
                        "name": "L1 Episode",
                        "type": "episode",
                        "layerId": "l1",
                        "segments": [self.l1_segment],
                    },
                    {
                        "name": "L2 Detail",
                        "type": "detail",
                        "layerId": "l2",
                        "segments": l2_segments,
                    },
                ],
                "playback": self.submit_playback,
                "segments": l2_segments,
                "timeUnit": "source_timestamp_sec",
                "episodeId": str(self.episode_id),
                "tagVocabulary": tag_vocabulary,
                "episodeTimeUnit": "episode_sec",
            }

            submit_payload = {
                "tag_vocabulary": tag_vocabulary,
                "annotation_json": annotation_json,
            }
            submit_payload_json = json.dumps(submit_payload, ensure_ascii=False, indent=2)

            allure.attach(submit_payload_json, name="步骤6提交大json", attachment_type=allure.attachment_type.TEXT)

            response = self.api_all.submit_task_annotation(
                annotation_id=self.annotation_id,
                tag_vocabulary=tag_vocabulary,
                annotation_json=annotation_json,
            )
            assertions.assert_code(response.status_code, 200)
            response_data = response.json()
            allure.attach(str(response_data), name="步骤6响应", attachment_type=allure.attachment_type.TEXT)
            assertions.assert_text(response_data.get("msg", ""), "success")

            allure.attach("步骤6执行成功：已提交标注", name="步骤6结果", attachment_type=allure.attachment_type.TEXT)

    @pytest.mark.order(7)
    @allure.story("步骤7：完成质检")
    def test_complete_task_qc(self):
        with allure.step("步骤7：完成质检"):
            if self.workflow_mode == "converted":
                allure.attach(
                    "任务已转换：跳过步骤7，直接进入后续下载流程",
                    name="步骤7跳过",
                    attachment_type=allure.attachment_type.TEXT,
                )
                pytest.skip("任务已转换，跳过步骤7")

            assertions.assert_is_not_none(self.task_id, "步骤1未提取到 task_id，无法执行步骤7")

            print(f"[步骤7] task_id: {self.task_id}")
            allure.attach(
                f"task_id={self.task_id}",
                name="步骤7请求参数",
                attachment_type=allure.attachment_type.TEXT,
            )

            response = self.api_all.complete_task_qc(task_id=self.task_id)
            assertions.assert_code(response.status_code, 200)
            response_data = response.json()
            allure.attach(str(response_data), name="步骤7响应", attachment_type=allure.attachment_type.TEXT)
            assertions.assert_in_text(response_data, "success")

            allure.attach("步骤7执行成功：已完成质检", name="步骤7结果", attachment_type=allure.attachment_type.TEXT)

    @pytest.mark.order(8)
    @allure.story("步骤8：发起数据转换")
    def test_create_conversion(self):
        with allure.step("步骤8：发起数据转换"):
            assertions.assert_is_not_none(self.task_id, "步骤1未提取到 task_id，无法执行步骤8")

            target_format = "lerobot_v2"
            TestV2Verify.target_format = target_format
            if self.workflow_mode == "converted":
                allure.attach(
                    f"task_id={self.task_id}\ntarget_format={target_format}\n说明=任务已转换，步骤8仅保留 folder 值，不再发起转换",
                    name="步骤8请求参数",
                    attachment_type=allure.attachment_type.TEXT,
                )
                allure.attach("步骤8执行完成：已转换模式下仅保留folder信息", name="步骤8结果",
                              attachment_type=allure.attachment_type.TEXT)
                return

            print(f"[步骤8] task_id: {self.task_id}")
            allure.attach(
                f"task_id={self.task_id}\ntarget_format={target_format}\nquality_labels=[]",
                name="步骤8请求参数",
                attachment_type=allure.attachment_type.TEXT,
            )

            response = self.api_all.create_conversion(task_id=self.task_id, target_format=target_format,
                                                      quality_labels=[])
            assertions.assert_code(response.status_code, 200)
            response_data = response.json()
            allure.attach(str(response_data), name="步骤8响应", attachment_type=allure.attachment_type.TEXT)
            assertions.assert_in_text(response_data, "success")

            allure.attach("步骤8执行成功：已发起数据转换", name="步骤8结果", attachment_type=allure.attachment_type.TEXT)

    @pytest.mark.order(9)
    @allure.story("步骤9：轮询正在转换列表")
    def test_poll_active_conversion(self):
        with allure.step("步骤9：轮询正在转换列表"):
            if self.workflow_mode == "converted":
                allure.attach(
                    "任务已转换：跳过步骤9，直接进入步骤11",
                    name="步骤9跳过",
                    attachment_type=allure.attachment_type.TEXT,
                )
                pytest.skip("任务已转换，跳过步骤9")

            assertions.assert_is_not_none(self.task_id, "步骤1未提取到 task_id，无法执行步骤9")

            max_attempts = 60
            for attempt in range(1, max_attempts + 1):
                response = self.api_all.query_active_conversion_list()
                assertions.assert_code(response.status_code, 200)
                response_data = response.json()
                assertions.assert_text(response_data.get("msg", ""), "success")

                conversion_list = response_data.get("data", {}).get("list", [])
                target_entry = find_conversion_entry(conversion_list, self.task_id)

                print(f"[步骤9] 第{attempt}次轮询，active列表条数: {len(conversion_list)}")
                if target_entry:
                    status = str(target_entry.get("status", "")).strip().lower()
                    print(f"[步骤9] 找到task_id={self.task_id}，status={status}")
                    allure.attach(
                        f"attempt={attempt}\nstatus={status}\nentry={target_entry}",
                        name="步骤9转换条目",
                        attachment_type=allure.attachment_type.TEXT,
                    )
                    TestV2Verify.active_conversion_entry = target_entry

                    if status in {"running", "queued"}:
                        if attempt < max_attempts:
                            time.sleep(5)
                            continue
                        error_msg = f"错误: task_id={self.task_id} 在 active 列表中持续处于 {status}，已达到最大轮询次数 {max_attempts}"
                        allure.attach(error_msg, name="步骤9轮询超时", attachment_type=allure.attachment_type.TEXT)
                        pytest.fail(error_msg)

                    error_msg = f"错误: task_id={self.task_id} 在 active 列表中状态不是 running，而是 '{status}'"
                    allure.attach(error_msg, name="步骤9状态异常", attachment_type=allure.attachment_type.TEXT)
                    pytest.fail(error_msg)

                print(f"[步骤9] active列表未找到 task_id={self.task_id}，进入步骤10")
                allure.attach(
                    f"task_id={self.task_id}\nactive列表未找到，进入步骤10",
                    name="步骤9未找到条目",
                    attachment_type=allure.attachment_type.TEXT,
                )
                TestV2Verify.active_conversion_entry = None
                break

            allure.attach("步骤9执行完成：active轮询结束", name="步骤9结果", attachment_type=allure.attachment_type.TEXT)

    @pytest.mark.order(10)
    @allure.story("步骤10：查询已转换完成列表")
    def test_check_finished_conversion(self):
        with allure.step("步骤10：查询已转换完成列表"):
            if self.workflow_mode == "converted":
                allure.attach(
                    "任务已转换：跳过步骤10，直接进入步骤11",
                    name="步骤10跳过",
                    attachment_type=allure.attachment_type.TEXT,
                )
                pytest.skip("任务已转换，跳过步骤10")

            assertions.assert_is_not_none(self.task_id, "步骤1未提取到 task_id，无法执行步骤10")

            response = self.api_all.query_finished_conversion_list()
            assertions.assert_code(response.status_code, 200)
            response_data = response.json()
            assertions.assert_text(response_data.get("msg", ""), "success")

            conversion_list = response_data.get("data", {}).get("list", [])
            target_entry = find_conversion_entry(conversion_list, self.task_id)

            print(f"[步骤10] finished列表条数: {len(conversion_list)}")
            if not target_entry:
                error_msg = f"错误: 已完成转换列表中未找到 task_id={self.task_id} 的数据"
                allure.attach(error_msg, name="步骤10未找到条目", attachment_type=allure.attachment_type.TEXT)
                pytest.fail(error_msg)

            status = str(target_entry.get("status", "")).strip().lower()
            print(f"[步骤10] 找到task_id={self.task_id}，status={status}")
            allure.attach(
                f"task_id={self.task_id}\nstatus={status}\nentry={target_entry}",
                name="步骤10转换条目",
                attachment_type=allure.attachment_type.TEXT,
            )
            TestV2Verify.finished_conversion_entry = target_entry

            if status != "completed":
                error_msg = f"错误: task_id={self.task_id} 在已完成列表中的状态不是 completed，而是 '{status}'"
                allure.attach(error_msg, name="步骤10状态异常", attachment_type=allure.attachment_type.TEXT)
                pytest.fail(error_msg)

            allure.attach("步骤10执行成功：数据转换已完成", name="步骤10结果",
                          attachment_type=allure.attachment_type.TEXT)

    @pytest.mark.order(11)
    @allure.story("步骤11：下载parquet文件")
    def test_download_parquet_file(self):
        with allure.step("步骤11：下载parquet文件"):
            assertions.assert_is_not_none(self.task_no, "步骤1未提取到 task_no，无法执行步骤11")
            if self.workflow_mode != "converted":
                assertions.assert_is_not_none(self.finished_conversion_entry, "步骤10未确认转换完成，无法执行步骤11")
            else:
                allure.attach(
                    "任务已转换：步骤11直接执行，步骤10结果不再作为前置依赖",
                    name="步骤11前置说明",
                    attachment_type=allure.attachment_type.TEXT,
                )
            assertions.assert_is_not_none(self.target_format, "步骤8未提取到 target_format，无法执行步骤11")

            folder = self.target_format
            print(f"[步骤11] task_no: {self.task_no}")
            print(f"[步骤11] folder: {folder}")
            allure.attach(
                f"task_no={self.task_no}\nfolder={folder}",
                name="步骤11下载参数",
                attachment_type=allure.attachment_type.TEXT,
            )

            downloaded_file = download_converted_parquet(task_no=self.task_no, folder=folder)
            TestV2Verify.downloaded_parquet_file = downloaded_file
            print(f"[步骤11] downloaded_parquet_file: {downloaded_file}")

            allure.attach(
                f"downloaded_file={downloaded_file}",
                name="步骤11下载结果",
                attachment_type=allure.attachment_type.TEXT,
            )

            assertions.assert_is_not_none(downloaded_file, "下载 parquet 失败，未返回文件路径")
            assert downloaded_file.exists(), f"下载后的 parquet 文件不存在: {downloaded_file}"

            allure.attach("步骤11执行成功：parquet文件已下载", name="步骤11结果",
                          attachment_type=allure.attachment_type.TEXT)

    @pytest.mark.order(12)
    @allure.story("步骤12：按步骤3标注开始和结束时间提取parquet图片")
    def test_extract_parquet_images_by_step3_times(self):
        with allure.step("步骤12：解析parquet并提取开始和结束时间图片"):
            self._prepare_image_output_dirs_once()
            extract_results = self._extract_parquet_images_for_annotation(
                annotation_key="step3",
                annotation_label="步骤3标注",
                workflow_step="12",
                start_time_ns=self._resolve_step3_start_time_ns(),
                end_time_ns=self._resolve_step3_end_time_ns(),
            )
            TestV2Verify.parquet_image_extract_result = extract_results
            allure.attach(
                "步骤12执行成功：已输出开始和结束时间的parquet左右相机图片",
                name="步骤12结果",
                attachment_type=allure.attachment_type.TEXT,
            )

    @pytest.mark.order(13)
    @allure.story("步骤13：按步骤3标注开始和结束时间提取原始mcap图片")
    def test_extract_mcap_images_by_step3_times(self):
        with allure.step("步骤13：在全部mcap中提取开始和结束时间图片"):
            self._prepare_image_output_dirs_once()
            extract_results = self._extract_mcap_images_for_annotation(
                annotation_key="step3",
                annotation_label="步骤3标注",
                workflow_step="13",
                start_time_ns=self._resolve_step3_start_time_ns(),
                end_time_ns=self._resolve_step3_end_time_ns(),
            )
            TestV2Verify.mcap_image_extract_result = extract_results
            allure.attach(
                "步骤13执行成功：已输出开始和结束时间的mcap左右相机图片",
                name="步骤13结果",
                attachment_type=allure.attachment_type.TEXT,
            )

    @pytest.mark.order(14)
    @allure.story("步骤14：校验步骤3标注开始和结束时间的左右相机图片一致性")
    def test_compare_mcap_and_parquet_camera_images(self):
        with allure.step("步骤14：对比mcap与parquet四组相机图片"):
            self._compare_annotation_image_results(
                parquet_results=self.parquet_image_extract_result or {},
                mcap_results=self.mcap_image_extract_result or {},
                annotation_key="step3",
                annotation_label="步骤3标注",
                workflow_step="14",
                start_time_ns=self._resolve_step3_start_time_ns(),
                end_time_ns=self._resolve_step3_end_time_ns(),
            )

    @pytest.mark.order(15)
    @allure.story("步骤15：按步骤4标注开始和结束时间提取parquet图片")
    def test_extract_parquet_images_by_step4_times(self):
        with allure.step("步骤15：提取步骤4标注开始和结束时间parquet图片"):
            self._prepare_image_output_dirs_once()
            extract_results = self._extract_parquet_images_for_annotation(
                annotation_key="step4",
                annotation_label="步骤4标注",
                workflow_step="15",
                start_time_ns=self._resolve_step4_start_time_ns(),
                end_time_ns=self._resolve_step4_end_time_ns(),
            )
            TestV2Verify.step4_parquet_image_extract_result = extract_results

    @pytest.mark.order(16)
    @allure.story("步骤16：按步骤4标注开始和结束时间提取原始mcap图片")
    def test_extract_mcap_images_by_step4_times(self):
        with allure.step("步骤16：提取步骤4标注开始和结束时间mcap图片"):
            self._prepare_image_output_dirs_once()
            extract_results = self._extract_mcap_images_for_annotation(
                annotation_key="step4",
                annotation_label="步骤4标注",
                workflow_step="16",
                start_time_ns=self._resolve_step4_start_time_ns(),
                end_time_ns=self._resolve_step4_end_time_ns(),
            )
            TestV2Verify.step4_mcap_image_extract_result = extract_results

    @pytest.mark.order(17)
    @allure.story("步骤17：校验步骤4标注开始和结束时间的左右相机图片一致性")
    def test_compare_step4_mcap_and_parquet_camera_images(self):
        with allure.step("步骤17：对比步骤4标注mcap与parquet四组相机图片"):
            self._compare_annotation_image_results(
                parquet_results=self.step4_parquet_image_extract_result or {},
                mcap_results=self.step4_mcap_image_extract_result or {},
                annotation_key="step4",
                annotation_label="步骤4标注",
                workflow_step="17",
                start_time_ns=self._resolve_step4_start_time_ns(),
                end_time_ns=self._resolve_step4_end_time_ns(),
            )

    @pytest.mark.order(18)
    @allure.story("步骤18：按步骤5标注开始和结束时间提取parquet图片")
    def test_extract_parquet_images_by_step5_times(self):
        with allure.step("步骤18：提取步骤5标注开始和结束时间parquet图片"):
            self._prepare_image_output_dirs_once()
            extract_results = self._extract_parquet_images_for_annotation(
                annotation_key="step5",
                annotation_label="步骤5标注",
                workflow_step="18",
                start_time_ns=self._resolve_step5_start_time_ns(),
                end_time_ns=self._resolve_step5_end_time_ns(),
            )
            TestV2Verify.step5_parquet_image_extract_result = extract_results

    @pytest.mark.order(19)
    @allure.story("步骤19：按步骤5标注开始和结束时间提取原始mcap图片")
    def test_extract_mcap_images_by_step5_times(self):
        with allure.step("步骤19：提取步骤5标注开始和结束时间mcap图片"):
            self._prepare_image_output_dirs_once()
            extract_results = self._extract_mcap_images_for_annotation(
                annotation_key="step5",
                annotation_label="步骤5标注",
                workflow_step="19",
                start_time_ns=self._resolve_step5_start_time_ns(),
                end_time_ns=self._resolve_step5_end_time_ns(),
            )
            TestV2Verify.step5_mcap_image_extract_result = extract_results

    @pytest.mark.order(20)
    @allure.story("步骤20：校验步骤5标注开始和结束时间的左右相机图片一致性")
    def test_compare_step5_mcap_and_parquet_camera_images(self):
        with allure.step("步骤20：对比步骤5标注mcap与parquet四组相机图片"):
            self._compare_annotation_image_results(
                parquet_results=self.step5_parquet_image_extract_result or {},
                mcap_results=self.step5_mcap_image_extract_result or {},
                annotation_key="step5",
                annotation_label="步骤5标注",
                workflow_step="20",
                start_time_ns=self._resolve_step5_start_time_ns(),
                end_time_ns=self._resolve_step5_end_time_ns(),
            )

    @pytest.mark.order(21)
    @allure.story("步骤21：提取步骤3开始和结束时间的MCAP七维向量")
    def test_extract_step3_robot_vectors_from_mcap(self):
        with allure.step("步骤21：提取步骤3开始和结束时间MCAP七维向量"):
            results = self._extract_mcap_vectors_for_annotation(
                annotation_key="step3",
                annotation_label="步骤3标注",
                workflow_step="21",
                start_time_ns=self._resolve_step3_start_time_ns(),
                end_time_ns=self._resolve_step3_end_time_ns(),
            )
            TestV2Verify.step3_pose_field_extract_results = results

    @pytest.mark.order(22)
    @allure.story("步骤22：提取步骤3开始和结束时间的parquet七维向量")
    def test_extract_step3_robot_vectors_from_parquet(self):
        with allure.step("步骤22：提取步骤3开始和结束时间parquet七维向量"):
            results = self._extract_parquet_vectors_for_annotation(
                annotation_key="step3",
                annotation_label="步骤3标注",
                workflow_step="22",
                start_time_ns=self._resolve_step3_start_time_ns(),
                end_time_ns=self._resolve_step3_end_time_ns(),
            )
            TestV2Verify.step3_parquet_pose_field_extract_results = results

    @pytest.mark.order(23)
    @allure.story("步骤23：对比步骤3开始和结束时间的MCAP与parquet七维向量")
    def test_compare_step3_robot_vectors(self):
        with allure.step("步骤23：对比步骤3开始和结束时间七维向量"):
            self._compare_vectors_for_annotation(
                mcap_results=self.step3_pose_field_extract_results or {},
                parquet_results=self.step3_parquet_pose_field_extract_results or {},
                annotation_key="step3",
                annotation_label="步骤3标注",
                workflow_step="23",
            )

    @pytest.mark.order(24)
    @allure.story("步骤24：提取步骤4开始和结束时间的MCAP七维向量")
    def test_extract_step4_robot_vectors_from_mcap(self):
        with allure.step("步骤24：提取步骤4开始和结束时间MCAP七维向量"):
            results = self._extract_mcap_vectors_for_annotation(
                annotation_key="step4",
                annotation_label="步骤4标注",
                workflow_step="24",
                start_time_ns=self._resolve_step4_start_time_ns(),
                end_time_ns=self._resolve_step4_end_time_ns(),
            )
            TestV2Verify.step4_pose_field_extract_results = results

    @pytest.mark.order(25)
    @allure.story("步骤25：提取步骤4开始和结束时间的parquet七维向量")
    def test_extract_step4_robot_vectors_from_parquet(self):
        with allure.step("步骤25：提取步骤4开始和结束时间parquet七维向量"):
            results = self._extract_parquet_vectors_for_annotation(
                annotation_key="step4",
                annotation_label="步骤4标注",
                workflow_step="25",
                start_time_ns=self._resolve_step4_start_time_ns(),
                end_time_ns=self._resolve_step4_end_time_ns(),
            )
            TestV2Verify.step4_parquet_pose_field_extract_results = results

    @pytest.mark.order(26)
    @allure.story("步骤26：对比步骤4开始和结束时间的MCAP与parquet七维向量")
    def test_compare_step4_robot_vectors(self):
        with allure.step("步骤26：对比步骤4开始和结束时间七维向量"):
            self._compare_vectors_for_annotation(
                mcap_results=self.step4_pose_field_extract_results or {},
                parquet_results=self.step4_parquet_pose_field_extract_results or {},
                annotation_key="step4",
                annotation_label="步骤4标注",
                workflow_step="26",
            )

    @pytest.mark.order(27)
    @allure.story("步骤27：提取步骤5开始和结束时间的MCAP七维向量")
    def test_extract_step5_robot_vectors_from_mcap(self):
        with allure.step("步骤27：提取步骤5开始和结束时间MCAP七维向量"):
            results = self._extract_mcap_vectors_for_annotation(
                annotation_key="step5",
                annotation_label="步骤5标注",
                workflow_step="27",
                start_time_ns=self._resolve_step5_start_time_ns(),
                end_time_ns=self._resolve_step5_end_time_ns(),
            )
            TestV2Verify.step5_pose_field_extract_results = results

    @pytest.mark.order(28)
    @allure.story("步骤28：提取步骤5开始和结束时间的parquet七维向量")
    def test_extract_step5_robot_vectors_from_parquet(self):
        with allure.step("步骤28：提取步骤5开始和结束时间parquet七维向量"):
            results = self._extract_parquet_vectors_for_annotation(
                annotation_key="step5",
                annotation_label="步骤5标注",
                workflow_step="28",
                start_time_ns=self._resolve_step5_start_time_ns(),
                end_time_ns=self._resolve_step5_end_time_ns(),
            )
            TestV2Verify.step5_parquet_pose_field_extract_results = results

    @pytest.mark.order(29)
    @allure.story("步骤29：对比步骤5开始和结束时间的MCAP与parquet七维向量")
    def test_compare_step5_robot_vectors(self):
        with allure.step("步骤29：对比步骤5开始和结束时间七维向量"):
            self._compare_vectors_for_annotation(
                mcap_results=self.step5_pose_field_extract_results or {},
                parquet_results=self.step5_parquet_pose_field_extract_results or {},
                annotation_key="step5",
                annotation_label="步骤5标注",
                workflow_step="29",
            )

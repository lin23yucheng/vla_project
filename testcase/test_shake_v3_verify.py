"""摇操数据 V3 任务校验流程。"""

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
from common.extract_mcap_fields import extract_robot_vectors_at_time
from common.extract_mcap_image import extract_global_nearest_image_from_mcap_directory
from common.extract_parquet_fields import compare_robot_vectors, extract_parquet_robot_vectors_at_time
from common.image_compare import compare_images
from common.parse_parquet_file import (
    compare_parquet_annotations,
    extract_nearest_parquet_images,
    parse_parquet_file,
)
from common.parquet_download_service import download_converted_parquet


START_TIME_NS = 1768893104830136738
END_TIME_NS = 1768893114656136738
BASELINE_START_TIME_NS = 1768893097352136738
BASELINE_END_TIME_NS = 1768893130775500327
ANNOTATION_DESCRIPTION = "移动夹笔放回全流程"
ROBOT_CONFIG_NAME = "遥操机械臂.json"

assertions = Assert.Assertions()
_last_snowflake_13 = 0


def load_task_context() -> tuple[str, str, str, str]:
    """读取当前环境的任务编号和摇操 MCAP 目录。"""
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
    shake_path = config.get(section, "shake_path", fallback="").strip()
    if not shake_path:
        raise ValueError(f"配置文件节 [{section}] 中缺少 shake_path")
    return execution_env, section, task_no, shake_path


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
    @classmethod
    def setup_class(cls):
        cls.api_all = all_api.ApiAll(global_client)
        cls.execution_env, cls.section, cls.task_no, cls.shake_path = load_task_context()
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
        cls.downloaded_parquet_file = None
        cls.parquet_image_extract_result = None
        cls.mcap_image_extract_result = None
        cls.mcap_vector_extract_results = None
        cls.parquet_vector_extract_results = None
        cls.submitted_annotation_json = None
        cls.parquet_annotation_validation_result = None
        cls.image_output_dirs_prepared = False

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
    def _camera_topics() -> list[str]:
        _, robot_config = load_robot_config()
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
        assertions.assert_is_not_none(
            self.downloaded_parquet_file,
            f"未下载 parquet 文件，无法执行步骤{workflow_step}",
        )
        parquet_path = Path(self.downloaded_parquet_file)
        assert parquet_path.is_file(), f"下载的 parquet 文件不存在: {parquet_path}"
        parquet_info = parse_parquet_file(parquet_path=parquet_path, preview_rows=1)
        allure.attach(
            json.dumps(
                {
                    "file": parquet_info.get("file"),
                    "num_rows": parquet_info.get("num_rows"),
                    "num_columns": parquet_info.get("num_columns"),
                    "num_row_groups": parquet_info.get("num_row_groups"),
                    "startTimeNs": self._resolve_l1_start_time_ns(),
                    "endTimeNs": self._resolve_l1_end_time_ns(),
                },
                ensure_ascii=False,
                indent=2,
            ),
            name=f"步骤{workflow_step}-parquet基础信息",
            attachment_type=allure.attachment_type.JSON,
        )

        results = {}
        output_dir = Path(__file__).resolve().parent.parent / "parquet_image"
        for substep, result_key, time_key, time_label, target_ns in (
            (f"{workflow_step}.1", "start_time_extract", "start", "开始时间", self._resolve_l1_start_time_ns()),
            (f"{workflow_step}.2", "end_time_extract", "end", "结束时间", self._resolve_l1_end_time_ns()),
        ):
            with allure.step(f"步骤{substep}：提取L1标注{time_label} parquet 图片"):
                result = extract_nearest_parquet_images(
                    parquet_path=parquet_path,
                    target_ns=target_ns,
                    output_dir=output_dir,
                    name_prefix=self.task_no,
                    extra_name_parts=["l1", time_key],
                )
                results[result_key] = result
                allure.attach(
                    json.dumps(result, ensure_ascii=False, indent=2),
                    name=f"步骤{substep}-图片提取结果",
                    attachment_type=allure.attachment_type.JSON,
                )
                assert result.get("saved_files"), f"步骤{substep}未生成 parquet 图片"
                for item in result["saved_files"]:
                    assert Path(item["saved_path"]).is_file(), f"提取文件不存在: {item['saved_path']}"
        return results

    def _extract_mcap_images(self, workflow_step: str) -> dict:
        mcap_dir = Path(self.shake_path)
        assert mcap_dir.is_dir(), f"配置中的 shake_path 不是有效目录: {mcap_dir}"
        assertions.assert_is_not_none(
            self.parquet_image_extract_result,
            f"未提取 parquet 图片，无法执行步骤{workflow_step}的同步取帧",
        )
        topics = self._camera_topics()
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
                result = extract_global_nearest_image_from_mcap_directory(
                    mcap_dir=mcap_dir,
                    topics=topics,
                    target_ns=mcap_target_ns,
                    output_dir=output_dir,
                    name_prefix=self.task_no,
                    extra_name_parts=["l1", time_key],
                )
                result["annotation_target_ns"] = annotation_target_ns
                result["parquet_matched_timestamp_ns"] = mcap_target_ns
                results[result_key] = result
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
                    mcap_dir=self.shake_path,
                    target_ns=target_ns,
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
                allure.attach(
                    json.dumps(result, ensure_ascii=False, indent=2),
                    name=f"步骤{substep}-MCAP七维向量",
                    attachment_type=allure.attachment_type.JSON,
                )
        return results

    def _extract_parquet_vectors(self, workflow_step: str) -> dict:
        assertions.assert_is_not_none(self.downloaded_parquet_file, "未下载 parquet 文件")
        results = {}
        for substep, time_key, time_label, target_ns in (
            (f"{workflow_step}.1", "start", "开始时间", self._resolve_l1_start_time_ns()),
            (f"{workflow_step}.2", "end", "结束时间", self._resolve_l1_end_time_ns()),
        ):
            with allure.step(f"步骤{substep}：提取L1标注{time_label} parquet 机器人向量"):
                result = extract_parquet_robot_vectors_at_time(
                    parquet_path=self.downloaded_parquet_file,
                    target_ns=target_ns,
                )
                results[time_key] = result
                vectors = result.get("vectors", [])
                assert len(vectors) == 4, "预期提取 action/observation_state 左右共4组向量"
                for item in vectors:
                    assert item["vector_labels"] == ["x", "y", "z", "roll", "pitch", "yaw", "angle"]
                    assert len(item["vector"]) == 7
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
                        for dimension in item.get("dimensions", []):
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

    @pytest.mark.order(1)
    @allure.story("步骤1：查询任务列表")
    def test_query_task_info(self):
        with allure.step("步骤1：查询任务列表"):
            allure.attach(
                f"execution_env={self.execution_env}\nsection=[{self.section}]\n"
                f"Task_no={self.task_no}\nshake_path={self.shake_path}",
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
                "topic": self._camera_topics()[1],
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
            response = self.api_all.complete_task_qc(task_id=self.task_id)
            assertions.assert_code(response.status_code, 200)
            assertions.assert_in_text(response.json(), "success")

    @pytest.mark.order(6)
    @allure.story("步骤6：发起数据转换")
    def test_create_conversion(self):
        with allure.step("步骤6：发起数据转换"):
            assertions.assert_is_not_none(self.task_id, "缺少 task_id")
            TestShakeV3Verify.target_format = "lerobot_v3"
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
            for attempt in range(1, 61):
                response = self.api_all.query_active_conversion_list()
                assertions.assert_code(response.status_code, 200)
                response_data = response.json()
                assertions.assert_text(response_data.get("msg", ""), "success")
                entry = find_conversion_entry(response_data.get("data", {}).get("list", []), self.task_id)
                if entry is None:
                    TestShakeV3Verify.active_conversion_entry = None
                    break
                TestShakeV3Verify.active_conversion_entry = entry
                status = str(entry.get("status", "")).strip().lower()
                if status not in {"running", "queued"}:
                    pytest.fail(f"active 转换状态异常: {status!r}")
                if attempt == 60:
                    pytest.fail(f"转换在 {60 * 5} 秒内未离开 active 列表")
                time.sleep(5)

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
            entry = find_conversion_entry(response_data.get("data", {}).get("list", []), self.task_id)
            assertions.assert_is_not_none(entry, f"finished 列表中未找到 task_id={self.task_id}")
            assert str(entry.get("status", "")).strip().lower() == "completed", f"转换未完成: {entry}"
            TestShakeV3Verify.finished_conversion_entry = entry

    @pytest.mark.order(9)
    @allure.story("步骤9：下载parquet文件")
    def test_download_parquet_file(self):
        with allure.step("步骤9：下载parquet文件"):
            if self.workflow_mode != "converted":
                assertions.assert_is_not_none(self.finished_conversion_entry, "尚未确认转换完成")
            assertions.assert_is_not_none(self.target_format, "缺少转换格式")
            downloaded_file = download_converted_parquet(
                task_no=self.task_no,
                folder=self.target_format,
            )
            assertions.assert_is_not_none(downloaded_file, "下载 parquet 未返回文件路径")
            assert Path(downloaded_file).is_file(), f"下载后的 parquet 文件不存在: {downloaded_file}"
            TestShakeV3Verify.downloaded_parquet_file = Path(downloaded_file)

    @pytest.mark.order(10)
    @allure.story("步骤10：按L1标注开始和结束时间提取parquet图片")
    def test_extract_l1_parquet_images(self):
        with allure.step("步骤10：按L1标注开始和结束时间提取parquet图片"):
            self._prepare_image_output_dirs_once()
            TestShakeV3Verify.parquet_image_extract_result = self._extract_parquet_images("10")

    @pytest.mark.order(11)
    @allure.story("步骤11：按L1标注开始和结束时间从shake_path提取MCAP图片")
    def test_extract_l1_mcap_images(self):
        with allure.step("步骤11：按L1标注开始和结束时间从shake_path提取MCAP图片"):
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
            assertions.assert_is_not_none(self.downloaded_parquet_file, "未下载 parquet 文件")
            expected_layers = self._expected_annotation_layers()
            result = compare_parquet_annotations(
                parquet_path=self.downloaded_parquet_file,
                expected_layers=expected_layers,
            )
            TestShakeV3Verify.parquet_annotation_validation_result = result
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

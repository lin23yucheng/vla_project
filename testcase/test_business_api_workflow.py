"""业务接口自动化全流程/工作台数据校验"""

import os
import json
import configparser
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path

import allure
import pytest

from api import all_api
from common import Assert
from common.client_factory import create_lazy_yixiu_client

assertions = Assert.Assertions()
global_client = create_lazy_yixiu_client()
# 填写已有任务编号时，跳过创建、上传和采集完成步骤，直接从步骤 8 开始。
# 为空字符串时执行完整流程；也可通过 VLA_TASK_NO 环境变量覆盖。
TASK_NO = "TASK-2026-038"
TASK_NO = os.environ.get("VLA_TASK_NO", TASK_NO).strip()
UPLOAD_POLL_INTERVAL_SECONDS = 10
UPLOAD_POLL_TIMEOUT_SECONDS = 30 * 60
LOCAL_UPLOAD_PROGRESS_POLL_SECONDS = 30
AUTO_LABELING_POLL_INTERVAL_SECONDS = 10
AUTO_LABELING_POLL_TIMEOUT_SECONDS = 60 * 60


def format_elapsed_minutes_seconds(seconds):
    """将耗时格式化为 X 分 X 秒。"""
    total_seconds = max(0, int(round(seconds)))
    minutes, remaining_seconds = divmod(total_seconds, 60)
    return f"{minutes}分{remaining_seconds}秒"


def response_list(response, interface_name):
    """校验列表接口响应并返回 data.list。"""
    assertions.assert_code(response.status_code, 200)
    response_data = response.json()
    assertions.assert_text(response_data.get("msg", ""), "success")

    data_list = response_data.get("data", {}).get("list", [])
    if not isinstance(data_list, list):
        pytest.fail(f"{interface_name} 返回的 data.list 不是列表")
    return data_list


def select_robot_config(robot_configs):
    """展示构型名称并返回用户选择的构型。"""
    configs_by_name = {
        str(config.get("name", "")).strip(): config
        for config in robot_configs
        if str(config.get("name", "")).strip()
    }
    if not configs_by_name:
        pytest.fail("机器人构型列表为空或未包含有效 name")

    available_names = list(configs_by_name)
    print("可选机器人构型：")
    for name in available_names:
        print(f"- {name}")

    selected_name = os.environ.get("VLA_ROBOT_CONFIG_NAME", "").strip()
    if not selected_name:
        selected_name = input("您需要创建哪一种构型的任务：").strip()

    selected_config = configs_by_name.get(selected_name)
    if selected_config is None:
        pytest.fail(
            f"机器人构型 {selected_name!r} 不存在，可选构型：{', '.join(available_names)}"
        )
    return selected_config


def collect_local_files():
    """读取用户指定路径下的全部普通文件，并保留相对路径。"""
    collection_path = os.environ.get("VLA_COLLECTION_PATH", "").strip()
    if not collection_path:
        collection_path = input("请提供采集文件的本地路径：").strip()
    if not collection_path:
        pytest.fail("未提供采集文件的本地路径")

    root_path = Path(collection_path).expanduser().resolve()
    if not root_path.exists():
        pytest.fail(f"采集文件路径不存在：{root_path}")

    if root_path.is_file():
        return root_path.parent, [root_path]
    if not root_path.is_dir():
        pytest.fail(f"采集文件路径不是文件或目录：{root_path}")

    files = sorted(path for path in root_path.rglob("*") if path.is_file())
    if not files:
        pytest.fail(f"采集文件目录为空：{root_path}")
    return root_path, files


def query_all_collect_files(api_all, task_id, status_group, page_size=20):
    """分页获取指定上传状态的全部采集文件。"""
    page_index = 1
    files = []
    while True:
        response = api_all.query_collect_files(
            task_id=task_id,
            status_group=status_group,
            page_index=page_index,
            page_size=page_size,
        )
        assertions.assert_code(response.status_code, 200)
        response_data = response.json()
        assertions.assert_text(response_data.get("msg", ""), "success")
        data = response_data.get("data", {})
        page_files = data.get("list", [])
        if not isinstance(page_files, list):
            pytest.fail(f"{status_group} 采集文件列表的 data.list 不是列表")
        files.extend(page_files)

        if page_index >= int(data.get("pages", 0) or 0):
            return files
        page_index += 1


def print_active_upload_progress(api_all, task_id):
    """查询并输出正在进行的本地上传进度。"""
    active_files = query_all_collect_files(api_all, task_id, "active")
    uploaded_files = query_all_collect_files(api_all, task_id, "uploaded")
    if not active_files:
        summary_response = api_all.query_collect_files_summary(task_id)
        if summary_response.status_code == 200:
            summary_data = summary_response.json().get("data", {})
            print(
                "[步骤4] 上传进度：active 列表暂为空，"
                f"已上传={len(uploaded_files)}，汇总={summary_data}",
                flush=True,
            )
        else:
            print(
                "[步骤4] 上传进度：active 列表暂为空，"
                f"已上传={len(uploaded_files)}，汇总接口状态={summary_response.status_code}",
                flush=True,
            )
        return

    for active_file in active_files:
        print(
            f"[步骤4] 上传进度："
            f"{active_file.get('relative_path') or active_file.get('filename')}，"
            f"状态={active_file.get('upload_status_label')}，"
            f"local_progress={active_file.get('local_progress', 0)}%，"
            f"progress={active_file.get('progress', 0)}%",
            flush=True,
        )


def _first_value(item, *keys):
    if not isinstance(item, dict):
        return None
    for key in keys:
        value = item.get(key)
        if value is not None and value != "":
            return value
    return None


def _auto_segment(item, layer_id, category, job_id, index, playback):
    """将自动标注结果中的 episode/sub-task 转为工作台 segment 格式。"""
    if not isinstance(item, dict):
        return None
    start_ns = _first_value(item, "startTimeNs", "startTimestampNs", "start_time_ns", "start_timestamp_ns")
    end_ns = _first_value(item, "endTimeNs", "endTimestampNs", "end_time_ns", "end_timestamp_ns")
    # 自动标注结果接口可能只返回相对视频的 startSec/endSec。
    if start_ns is None or end_ns is None:
        start_sec = _first_value(item, "startSec", "start_sec")
        end_sec = _first_value(item, "endSec", "end_sec")
        try:
            baseline_start = int(playback["baseline_start_time_ns"])
            if start_sec is not None and end_sec is not None:
                start_ns = baseline_start + round(float(start_sec) * 1_000_000_000)
                end_ns = baseline_start + round(float(end_sec) * 1_000_000_000)
        except (KeyError, TypeError, ValueError) as exc:
            pytest.fail(f"自动标注结果 startSec/endSec 或 playback 无效：{exc}")
    if start_ns is None or end_ns is None:
        return None
    try:
        start_ns_int, end_ns_int = int(start_ns), int(end_ns)
        baseline_start_ns = int(playback["baseline_start_time_ns"])
    except (KeyError, TypeError, ValueError) as exc:
        pytest.fail(f"自动标注结果或 playback 缺少有效纳秒时间：{exc}")
    if end_ns_int < start_ns_int:
        pytest.fail(f"自动标注片段结束时间早于开始时间：{item}")
    start_ns, end_ns = str(start_ns_int), str(end_ns_int)
    relative_start_ns, relative_end_ns = start_ns_int - baseline_start_ns, end_ns_int - baseline_start_ns
    segment_id_value = _first_value(item, "segmentId", "segment_id", "id")
    if segment_id_value is None:
        segment_id_value = f"auto:{job_id}:{category}:{index}"
    segment_id = str(segment_id_value)
    description_value = _first_value(item, "description", "prompt", "name", "label", "text")
    if description_value is None:
        pytest.fail(
            f"自动标注结果缺少 description/prompt/name，不能准确回写 {category}[{index}]：{item}"
        )
    description = str(description_value)
    prompt_value = _first_value(item, "prompt") or description
    layer_value = _first_value(item, "layerId", "layer_id")
    if layer_value is None:
        # 简化自动标注结果不携带层级时，由结果所在集合的协议层级确定。
        layer_value = layer_id
    source_value = _first_value(item, "source") or "auto"
    auto_review = item.get("autoReview")
    if not isinstance(auto_review, dict):
        auto_review = {"score": 0, "decision": "", "comment": ""}
    attributes = item.get("attributes")
    if not isinstance(attributes, dict):
        attributes = {}
    segment = dict(item)
    segment.update({
        "segmentId": segment_id,
        "id": segment_id,
        "layerId": str(layer_value),
        "category": category,
        "description": description,
        "prompt": str(prompt_value),
        "source": str(source_value),
        "autoReview": auto_review,
        "attributes": {
            **attributes,
            # 两项由本次接口链路确定，不依赖前端或算法结果重复返回。
            "autoJobId": str(job_id),
            "autoResultType": category,
        },
        "startTimeNs": start_ns,
        "endTimeNs": end_ns,
        "startTimestampNs": start_ns,
        "endTimestampNs": end_ns,
    })
    relative_start_sec = relative_start_ns / 1_000_000_000
    relative_end_sec = relative_end_ns / 1_000_000_000
    segment.update({
        "episodeStartTimeNs": str(relative_start_ns),
        "episodeEndTimeNs": str(relative_end_ns),
        "episode_start_time": f"{relative_start_sec:.9f}",
        "episode_end_time": f"{relative_end_sec:.9f}",
        "timeline_start_sec": f"{relative_start_sec:.9f}",
        "timeline_end_sec": f"{relative_end_sec:.9f}",
        "baseline_camera_key": playback["baseline_camera_key"],
        "startSec": relative_start_sec,
        "endSec": relative_end_sec,
        "start_time": f"{start_ns_int // 1_000_000_000}.{start_ns_int % 1_000_000_000:09d}",
        "end_time": f"{end_ns_int // 1_000_000_000}.{end_ns_int % 1_000_000_000:09d}",
    })
    return segment


def _build_auto_segments(result, job_id, playback):
    episodes = result.get("episodes_annotation") if isinstance(result, dict) else None
    if not isinstance(episodes, list) or not episodes:
        return []
    segments = []
    for episode_index, episode in enumerate(episodes):
        episode_item = episode.get("segment", episode) if isinstance(episode, dict) else episode
        episode_segment = _auto_segment(episode_item, "l1", "episode", job_id, episode_index, playback)
        if episode_segment:
            segments.append(episode_segment)
        subtasks = _first_value(
            episode,
            "sub_tasks_annotation",
            "subtasks_annotation",
            "sub_tasks",
            "subtasks",
            "sub_task_annotation",
        )
        if not isinstance(subtasks, list) and isinstance(episode_item, dict):
            subtasks = _first_value(
                episode_item,
                "sub_tasks_annotation",
                "subtasks_annotation",
                "sub_tasks",
                "subtasks",
            )
        if isinstance(subtasks, list):
            for subtask_index, subtask in enumerate(subtasks):
                subtask_item = subtask.get("segment", subtask) if isinstance(subtask, dict) else subtask
                segment = _auto_segment(
                    subtask_item, "l2", "sub_task", job_id, subtask_index, playback
                )
                if segment:
                    segments.append(segment)
    # 简化结果中 sub_tasks_annotation 与 episodes_annotation 平级返回。
    if not any(item.get("category") == "sub_task" for item in segments):
        top_level_subtasks = result.get("sub_tasks_annotation") or result.get("subtasks_annotation") or []
        if isinstance(top_level_subtasks, list):
            for subtask_index, subtask in enumerate(top_level_subtasks):
                subtask_item = subtask.get("segment", subtask) if isinstance(subtask, dict) else subtask
                segment = _auto_segment(
                    subtask_item, "l2", "sub_task", job_id, subtask_index, playback
                )
                if segment:
                    segments.append(segment)
    return segments


def _workspace_playback(workspace_data, playlist_data):
    candidates = [
        (workspace_data or {}).get("playback"),
        ((workspace_data or {}).get("annotation") or {}).get("playback"),
        (playlist_data or {}).get("playback"),
        playlist_data,
    ]
    for candidate in candidates:
        if isinstance(candidate, dict) and candidate.get("baseline_start_time_ns"):
            return candidate
    # video-playlist 的实际响应将 playback 拆分为 timeline 和 cameras。
    if isinstance(playlist_data, dict):
        cameras = playlist_data.get("cameras")
        timeline = playlist_data.get("timeline") or {}
        if isinstance(cameras, list) and cameras:
            requested_key = os.environ.get("VLA_BASELINE_CAMERA_KEY", "").strip()
            camera = next(
                (item for item in cameras if item.get("camera_key") == requested_key),
                None,
            ) if requested_key else None
            camera = camera or cameras[0]
            camera_key = camera.get("camera_key")
            camera_segments = camera.get("segments") or []
            camera_segment = camera_segments[0] if camera_segments else {}
            start_ns = camera_segment.get("global_start_time_ns") or timeline.get("start_time_ns")
            end_ns = camera_segment.get("global_end_time_ns") or timeline.get("end_time_ns")
            topic = camera_segment.get("image_topic") or camera_segment.get("topic")
            sequence = camera_segment.get("sequence", 1)
            if camera_key and start_ns and end_ns:
                return {
                    "baseline_camera_key": camera_key,
                    "baseline_camera_label": camera_key,
                    "topic": topic,
                    "baseline_start_time_ns": str(start_ns),
                    "baseline_end_time_ns": str(end_ns),
                    "timeline_duration_sec": str(
                        timeline.get("duration_sec")
                        or (int(end_ns) - int(start_ns)) / 1_000_000_000
                    ),
                    "gap_policy": "skip_on_playback",
                    "current_sequence": sequence,
                }
    return None


@allure.feature("业务接口自动化全流程")
class TestBusinessApiWorkflow:
    @classmethod
    def setup_class(cls):
        cls.api_all = all_api.ApiAll(global_client)
        cls.task_name = None
        cls.category_name = None
        cls.category_label = None
        cls.category_id = None
        cls.scene_tags = None
        cls.robot_config_name = None
        cls.robot_config_id = None
        cls.task_no = None
        cls.task_id = None
        cls.local_collection_path = None
        cls.upload_files = []
        cls.auto_labeling_job_id = None
        cls.auto_labeling_result = None
        cls.auto_labeling_job_result = None
        cls.annotation_video_duration_sec = None

    @pytest.mark.order(1)
    @allure.story("步骤1：创建任务")
    def test_create_task(self):
        if TASK_NO:
            pytest.skip(f"已指定任务 {TASK_NO}，跳过创建任务")
        with allure.step("获取创建任务所需的分类、标签和机器人构型"):
            categories = response_list(
                self.api_all.query_data_categories(), "获取数据分类接口"
            )
            if not categories:
                pytest.fail("数据分类列表为空，无法创建任务")
            category = categories[-1]
            self.__class__.category_name = category.get("name")
            self.__class__.category_label = category.get("label")
            self.__class__.category_id = category.get("id")
            if not self.category_name:
                pytest.fail("最后一条数据分类未包含有效 name")

            scene_tag_list = response_list(
                self.api_all.query_scene_tags(), "获取标签接口"
            )
            if len(scene_tag_list) < 2:
                pytest.fail("标签列表少于两条，无法创建任务")
            last_two_scene_tags = scene_tag_list[-2:]
            last_two_names = [str(tag.get("name", "")).strip() for tag in last_two_scene_tags]
            if any(not name for name in last_two_names):
                pytest.fail("最后两条标签中存在空 name")
            self.__class__.scene_tags = list(reversed(last_two_names))

            robot_configs = response_list(
                self.api_all.query_robot_configs(), "获取机器人构型接口"
            )
            selected_robot_config = select_robot_config(robot_configs)
            self.__class__.robot_config_name = selected_robot_config.get("name")
            self.__class__.robot_config_id = selected_robot_config.get("id")
            if not self.robot_config_id:
                pytest.fail(f"机器人构型 {self.robot_config_name!r} 未包含有效 id")

        self.__class__.task_name = (
            f"{self.robot_config_name}-业务接口自动化-"
            f"{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        )
        with allure.step("创建任务"):
            response = self.api_all.create_task(
                name=self.task_name,
                category=self.category_name,
                scene_tags=self.scene_tags,
                robot_config_id=self.robot_config_id,
            )
            assertions.assert_code(response.status_code, 201)
            assertions.assert_text(response.json().get("msg", ""), "success")

            allure.attach(
                "\n".join(
                    [
                        f"task_name: {self.task_name}",
                        f"category_name: {self.category_name}",
                        f"category_label: {self.category_label}",
                        f"category_id: {self.category_id}",
                        f"scene_tags: {self.scene_tags}",
                        f"robot_config_name: {self.robot_config_name}",
                        f"robot_config_id: {self.robot_config_id}",
                    ]
                ),
                name="创建任务参数",
                attachment_type=allure.attachment_type.TEXT,
            )

    @pytest.mark.order(2)
    @allure.story("步骤2：查询任务管理列表")
    def test_query_created_task(self):
        if TASK_NO:
            task_list = response_list(
                self.api_all.query_task_list(), "查询任务管理列表接口"
            )
            target_task = next(
                (task for task in task_list if str(task.get("task_no", "")).strip() == TASK_NO),
                None,
            )
            if target_task is None:
                pytest.fail(f"任务管理列表未找到任务编号：{TASK_NO}")
            self.__class__.task_no = target_task.get("task_no")
            self.__class__.task_id = target_task.get("id")
            self.__class__.task_name = target_task.get("name")
            self.__class__.robot_config_id = (
                target_task.get("robot_config_id")
                or target_task.get("robotConfigId")
                or (target_task.get("robot_config") or {}).get("id")
            )
            self.__class__.robot_config_name = (
                target_task.get("robot_config_name")
                or target_task.get("robotConfigName")
                or (target_task.get("robot_config") or {}).get("name")
            )
            scene_tags = target_task.get("scene_tags") or target_task.get("sceneTags") or []
            self.__class__.scene_tags = scene_tags if isinstance(scene_tags, list) else []
            if not self.task_id or not self.task_name:
                pytest.fail(f"任务 {TASK_NO} 未包含有效 id 或 name")
            if not self.robot_config_id and self.robot_config_name:
                configs = response_list(self.api_all.query_robot_configs(), "获取机器人构型接口")
                matched = next((item for item in configs if item.get("name") == self.robot_config_name), None)
                self.__class__.robot_config_id = matched.get("id") if matched else None
            if not self.robot_config_id:
                pytest.fail(f"任务 {TASK_NO} 未包含有效 robot_config_id")
            return
        if not self.task_name:
            pytest.fail("步骤1未生成任务名称，无法查询创建结果")

        with allure.step("按任务名称查询任务管理列表"):
            task_list = response_list(
                self.api_all.query_task_list(), "查询任务管理列表接口"
            )
            target_task = next(
                (
                    task
                    for task in task_list
                    if str(task.get("name", "")).strip() == self.task_name
                ),
                None,
            )
            if target_task is None:
                pytest.fail(f"任务管理列表未找到刚创建的任务：{self.task_name}")

            self.__class__.task_no = target_task.get("task_no")
            self.__class__.task_id = target_task.get("id")
            if not self.task_no or not self.task_id:
                pytest.fail(f"任务 {self.task_name} 未包含有效 task_no 或 id")

            allure.attach(
                f"task_name: {self.task_name}\n"
                f"task_no: {self.task_no}\n"
                f"id: {self.task_id}",
                name="创建任务查询结果",
                attachment_type=allure.attachment_type.TEXT,
            )

    @pytest.mark.order(3)
    @allure.story("步骤3：查询数据采集列表")
    def test_query_created_task_in_data_collection_list(self):
        if TASK_NO:
            pytest.skip(f"已指定任务 {TASK_NO}，跳过数据采集列表校验")
        if not self.task_id or not self.task_no or not self.task_name:
            pytest.fail("步骤2未获取完整任务信息，无法查询数据采集列表")

        with allure.step("按任务 ID 查询数据采集列表并核对任务信息"):
            task_list = response_list(
                self.api_all.query_data_collection_list(), "查询数据采集列表接口"
            )
            target_task = next(
                (
                    task
                    for task in task_list
                    if str(task.get("id", "")).strip() == str(self.task_id).strip()
                ),
                None,
            )
            if target_task is None:
                pytest.fail(f"数据采集列表未找到刚创建的任务：id={self.task_id}")

            actual_task_id = target_task.get("id")
            actual_task_name = target_task.get("name")
            actual_task_no = target_task.get("task_no")
            assertions.assert_text(str(actual_task_id), str(self.task_id))
            assertions.assert_text(str(actual_task_name), str(self.task_name))
            assertions.assert_text(str(actual_task_no), str(self.task_no))

            allure.attach(
                f"id: {actual_task_id}\n"
                f"name: {actual_task_name}\n"
                f"task_no: {actual_task_no}",
                name="数据采集列表任务核对结果",
                attachment_type=allure.attachment_type.TEXT,
            )

    @pytest.mark.order(4)
    @allure.story("步骤4：采集文件上传")
    def test_upload_collection_files(self):
        if TASK_NO:
            pytest.skip(f"已指定任务 {TASK_NO}，跳过文件上传")
        if not self.task_id:
            pytest.fail("步骤2未获取 task_id，无法上传采集文件")

        root_path, local_files = collect_local_files()
        upload_files = [
            {
                "local_path": local_path,
                "filename": local_path.name,
                "relative_path": local_path.relative_to(root_path).as_posix(),
                "size_bytes": local_path.stat().st_size,
            }
            for local_path in local_files
        ]
        registration_files = [
            {
                "filename": file_info["filename"],
                "relative_path": file_info["relative_path"],
                "size_bytes": file_info["size_bytes"],
            }
            for file_info in upload_files
        ]

        with allure.step("登记待上传采集文件"):
            response = self.api_all.register_collect_files(
                task_id=self.task_id, files=registration_files
            )
            assertions.assert_code(response.status_code, 200)
            response_data = response.json()
            assertions.assert_text(response_data.get("msg", ""), "success")
            registered_files = response_data.get("data", {}).get("files", [])
            if len(registered_files) != len(upload_files):
                pytest.fail(
                    f"登记文件数量不一致：本地 {len(upload_files)} 个，"
                    f"接口返回 {len(registered_files)} 个"
                )

            # 某些环境批量登记接口会返回成功但未实际落库，单文件登记可正常落库。
            summary_response = self.api_all.query_collect_files_summary(self.task_id)
            assertions.assert_code(summary_response.status_code, 200)
            summary_data = summary_response.json().get("data", {})
            if int(summary_data.get("total", 0) or 0) == 0:
                registered_files = []
                for registration_file in registration_files:
                    single_response = self.api_all.register_collect_files(
                        task_id=self.task_id, files=[registration_file]
                    )
                    assertions.assert_code(single_response.status_code, 200)
                    single_data = single_response.json()
                    assertions.assert_text(single_data.get("msg", ""), "success")
                    single_files = single_data.get("data", {}).get("files", [])
                    if len(single_files) != 1:
                        pytest.fail(
                            f"单文件登记响应数量异常：{registration_file['relative_path']}，"
                            f"返回 {len(single_files)} 个"
                        )
                    registered_files.extend(single_files)

                summary_response = self.api_all.query_collect_files_summary(self.task_id)
                assertions.assert_code(summary_response.status_code, 200)
                summary_data = summary_response.json().get("data", {})
                if int(summary_data.get("total", 0) or 0) != len(upload_files):
                    pytest.fail(
                        "文件登记后服务端汇总数量仍不正确："
                        f"期望 {len(upload_files)}，实际 {summary_data.get('total', 0)}"
                    )

            registered_by_path = {
                str(file_info.get("relative_path", "")): file_info
                for file_info in registered_files
            }
            for file_info in upload_files:
                registered_file = registered_by_path.get(file_info["relative_path"])
                if not registered_file or not registered_file.get("id"):
                    pytest.fail(
                        f"登记响应未返回文件 {file_info['relative_path']} 的 task_file_id"
                    )
                file_info["task_file_id"] = registered_file["id"]

        self.__class__.local_collection_path = root_path
        self.__class__.upload_files = upload_files
        allure.attach(
            "\n".join(
                f"{file_info['relative_path']} ({file_info['size_bytes']} bytes)"
                for file_info in upload_files
            ),
            name="待上传采集文件",
            attachment_type=allure.attachment_type.TEXT,
        )

        upload_started_at = time.monotonic()
        for index, file_info in enumerate(upload_files, start=1):
            with allure.step(
                f"上传文件 {index}/{len(upload_files)}：{file_info['relative_path']}"
            ):
                print(
                    f"[步骤4] 开始上传 {index}/{len(upload_files)}："
                    f"{file_info['relative_path']}（{file_info['size_bytes']} bytes）",
                    flush=True,
                )
                resume_response = self.api_all.query_collect_file_resume(
                    task_id=self.task_id,
                    task_file_id=file_info["task_file_id"],
                )
                assertions.assert_code(resume_response.status_code, 200)
                resume_data = resume_response.json().get("data", {})
                offset = int(resume_data.get("uploaded_bytes", 0) or 0)
                if offset < 0 or offset > file_info["size_bytes"]:
                    pytest.fail(
                        f"文件 {file_info['relative_path']} 的断点位置无效：{offset}"
                    )

                upload_result = {}

                def upload_file():
                    try:
                        with file_info["local_path"].open("rb") as file_stream:
                            file_stream.seek(offset)
                            upload_result["response"] = self.api_all.upload_collect_file_stream(
                                task_id=self.task_id,
                                filename=file_info["filename"],
                                relative_path=file_info["relative_path"],
                                task_file_id=file_info["task_file_id"],
                                offset=offset,
                                file_stream=file_stream,
                                content_length=file_info["size_bytes"] - offset,
                            )
                    except Exception as exc:
                        upload_result["error"] = exc

                upload_thread = threading.Thread(
                    target=upload_file,
                    name=f"upload-{file_info['filename']}",
                )
                upload_thread.start()
                while upload_thread.is_alive():
                    upload_thread.join(LOCAL_UPLOAD_PROGRESS_POLL_SECONDS)
                    if upload_thread.is_alive():
                        print_active_upload_progress(self.api_all, self.task_id)

                if "error" in upload_result:
                    raise upload_result["error"]
                response = upload_result.get("response")
                if response is None:
                    pytest.fail(f"文件 {file_info['relative_path']} 上传未返回响应")
                assertions.assert_code(response.status_code, 201)
                response_data = response.json()
                assertions.assert_text(response_data.get("msg", ""), "success")
                uploaded_file = response_data.get("data", {}).get("file", {})
                assertions.assert_text(
                    uploaded_file.get("upload_status_label", ""), "本地上传成功"
                )
                print(
                    f"[步骤4] 本地上传完成 {index}/{len(upload_files)}："
                    f"{file_info['relative_path']}，状态="
                    f"{uploaded_file.get('upload_status_label')}，"
                    f"local_progress={uploaded_file.get('local_progress', 0)}%",
                    flush=True,
                )

        upload_elapsed_seconds = int(time.monotonic() - upload_started_at)
        upload_minutes, upload_seconds = divmod(upload_elapsed_seconds, 60)
        print(
            f"[步骤4] 全部文件上传完成，总耗时："
            f"{upload_minutes}分{upload_seconds}秒",
            flush=True,
        )

    @pytest.mark.order(5)
    @allure.story("步骤5：轮询采集文件上传结果")
    def test_wait_for_collection_upload_completion(self):
        if TASK_NO:
            pytest.skip(f"已指定任务 {TASK_NO}，跳过上传轮询")
        if not self.task_id or not self.upload_files:
            pytest.fail("步骤4未准备上传文件，无法检查上传结果")

        expected_relative_paths = {
            file_info["relative_path"] for file_info in self.upload_files
        }
        started_at = time.monotonic()
        while True:
            active_files = query_all_collect_files(self.api_all, self.task_id, "active")
            uploaded_files = query_all_collect_files(self.api_all, self.task_id, "uploaded")
            summary_response = self.api_all.query_collect_files_summary(self.task_id)
            assertions.assert_code(summary_response.status_code, 200)
            summary_response_data = summary_response.json()
            assertions.assert_text(summary_response_data.get("msg", ""), "success")
            summary_data = summary_response_data.get("data", {})

            remaining_count = len(active_files)
            print(
                f"[步骤5] 上传轮询：总文件 {len(self.upload_files)} 个，"
                f"已完成 {len(uploaded_files)} 个，剩余 {remaining_count} 个，"
                f"整体进度={summary_data.get('overall_progress', 0)}%",
                flush=True,
            )
            for active_file in active_files:
                print(
                    f"[步骤5] 进行中：{active_file.get('relative_path') or active_file.get('filename')}，"
                    f"状态={active_file.get('upload_status_label')}，"
                    f"local_progress={active_file.get('local_progress', 0)}%，"
                    f"progress={active_file.get('progress', 0)}%",
                    flush=True,
                )

            allure.attach(
                f"active_count: {len(active_files)}\n"
                f"uploaded_count: {len(uploaded_files)}\n"
                f"active_files: {[(file_info.get('relative_path'), file_info.get('upload_status_label')) for file_info in active_files]}\n"
                f"uploaded_files: {[(file_info.get('relative_path'), file_info.get('upload_status_label')) for file_info in uploaded_files]}\n"
                f"summary: {summary_data}",
                name="上传轮询状态",
                attachment_type=allure.attachment_type.TEXT,
            )

            if not active_files:
                actual_relative_paths = {
                    str(file_info.get("relative_path", "")) for file_info in uploaded_files
                }
                actual_filenames = {
                    str(file_info.get("filename", "")) for file_info in uploaded_files
                }
                expected_filenames = {
                    file_info["filename"] for file_info in self.upload_files
                }
                assertions.assert_text(
                    str(len(uploaded_files)), str(len(self.upload_files))
                )
                assertions.assert_text(
                    str(sorted(actual_relative_paths)),
                    str(sorted(expected_relative_paths)),
                )
                assertions.assert_text(
                    str(sorted(actual_filenames)), str(sorted(expected_filenames))
                )
                for file_info in uploaded_files:
                    assertions.assert_text(
                        file_info.get("upload_status_label", ""), "上传成功"
                    )
                assertions.assert_text(
                    str(summary_data.get("success", 0)), str(len(self.upload_files))
                )
                return

            elapsed_seconds = time.monotonic() - started_at
            if elapsed_seconds >= UPLOAD_POLL_TIMEOUT_SECONDS:
                pytest.fail(
                    f"等待服务器转存上传完成超时：{UPLOAD_POLL_TIMEOUT_SECONDS} 秒"
                )
            time.sleep(UPLOAD_POLL_INTERVAL_SECONDS)

    @pytest.mark.order(6)
    @allure.story("步骤6：提交采集完成")
    def test_complete_collection_files(self):
        if TASK_NO:
            pytest.skip(f"已指定任务 {TASK_NO}，跳过提交采集完成")
        if not self.task_id or not self.upload_files:
            pytest.fail("步骤4未准备上传文件，无法提交采集完成")

        with allure.step("核对本地文件与服务端已上传文件数量"):
            uploaded_files = query_all_collect_files(
                self.api_all, self.task_id, "uploaded"
            )
            active_files = query_all_collect_files(
                self.api_all, self.task_id, "active"
            )
            expected_count = len(self.upload_files)
            if active_files:
                pytest.fail(
                    f"仍有文件处于上传中，无法提交采集完成：{len(active_files)} 个"
                )
            assertions.assert_text(str(len(uploaded_files)), str(expected_count))
            for uploaded_file in uploaded_files:
                assertions.assert_text(
                    uploaded_file.get("upload_status_label", ""), "上传成功"
                )

        with allure.step("调用采集完成接口"):
            response = self.api_all.complete_collect_files(self.task_id)
            assertions.assert_code(response.status_code, 200)
            assertions.assert_text(response.json().get("msg", ""), "success")

    @pytest.mark.order(7)
    @allure.story("步骤7：验证任务进入标注列表")
    def test_verify_collection_completed_task_in_annotation_list(self):
        if TASK_NO:
            pytest.skip(f"已指定任务 {TASK_NO}，跳过采集完成后的列表校验")
        if not self.task_id or not self.task_name:
            pytest.fail("步骤2未获取完整任务信息，无法验证采集完成结果")

        with allure.step("采集列表中不再显示该任务"):
            collection_tasks = response_list(
                self.api_all.query_data_collection_list(), "查询数据采集列表接口"
            )
            collection_match = next(
                (
                    task
                    for task in collection_tasks
                    if str(task.get("id", "")).strip()
                    == str(self.task_id).strip()
                ),
                None,
            )
            if collection_match is not None:
                pytest.fail(
                    f"采集完成任务仍存在于数据采集列表：id={self.task_id}"
                )

        with allure.step("标注列表中显示该任务"):
            annotation_tasks = response_list(
                self.api_all.query_annotation_list(), "查询标注列表接口"
            )
            annotation_match = next(
                (
                    task
                    for task in annotation_tasks
                    if str(task.get("id", "")).strip()
                    == str(self.task_id).strip()
                    or str(task.get("name", "")).strip() == self.task_name
                ),
                None,
            )
            if annotation_match is None:
                pytest.fail(
                    f"采集完成任务未出现在标注列表：id={self.task_id}，name={self.task_name}"
                )

    @pytest.mark.order(8)
    @allure.story("步骤8：参数预检")
    def test_auto_labeling_pre_check(self):
        if not self.task_id or not self.robot_config_id:
            pytest.fail("缺少 task_id 或 robot_config_id，无法进行参数预检")

        config_response = self.api_all.query_robot_config_detail(self.robot_config_id)
        assertions.assert_code(config_response.status_code, 200)
        config_data = config_response.json().get("data", {})
        robot_config = config_data.get("config_json")
        if not isinstance(robot_config, dict):
            pytest.fail("机器人构型详情缺少有效 config_json")

        uploaded_files = query_all_collect_files(self.api_all, self.task_id, "uploaded")
        file_paths = [item.get("s3_uri") for item in uploaded_files if item.get("s3_uri")]
        if not file_paths:
            pytest.fail("没有可用于自动化标注的 S3 文件路径")

        playlist_response = self.api_all.query_task_video_playlist(self.task_id)
        assertions.assert_code(playlist_response.status_code, 200)
        playlist_data = playlist_response.json().get("data")
        if not isinstance(playlist_data, dict):
            pytest.fail("视频播放清单响应缺少有效 data")

        pre_check_job_id = os.environ.get("VLA_AUTO_LABELING_JOB_ID", "").strip()
        if not pre_check_job_id:
            # 预检接口要求 job_id 字符串，但尚未进入正式任务创建阶段；使用本次请求的数字追踪 ID。
            pre_check_job_id = str(time.time_ns())[-16:]
        payload = {
            "job_id": pre_check_job_id,
            "basic": {"task_id": str(self.task_id), "user_config_path": None},
            "input": {
                "file_path": file_paths,
                "file_type": "mcap",
                "video_playlist": json.dumps(playlist_data, ensure_ascii=False, separators=(",", ":")),
                "robot_config": json.dumps(robot_config, ensure_ascii=False, separators=(",", ":")),
                "user_prompt": os.environ.get(
                    "VLA_AUTO_LABELING_PROMPT",
                    "请描述机器人完成的子任务,可选子任务为：\n"
                    "夹取物体：机械臂移动靠近物体，末端夹爪角度减小以夹持物体；\n"
                    "移动物体：夹取物体后的移动状态；\n"
                    "放下物体：机械臂末端夹爪张开，放开物体。",
                ),
                "video_fps": int(os.environ.get("VLA_AUTO_LABELING_FPS", "30")),
            },
        }
        response = self.api_all.auto_labeling_pre_check(payload)
        if response.status_code != 200:
            pytest.fail(
                f"参数预检失败：HTTP {response.status_code}，响应：{response.text}"
            )
        assertions.assert_code(response.status_code, 200)
        assertions.assert_text(response.json().get("msg", ""), "ok")

    @pytest.mark.order(9)
    @allure.story("步骤9：提交自动化标注任务")
    def test_submit_auto_labeling(self):
        if not self.task_id or not self.robot_config_id:
            pytest.fail("缺少任务或机器人构型信息，无法提交自动化标注")

        job_response = self.api_all.create_auto_labeling_job_id()
        assertions.assert_code(job_response.status_code, 200)
        job_id = job_response.json().get("data", {}).get("job_id")
        if not job_id:
            pytest.fail("生成 job_id 失败")

        config_response = self.api_all.query_robot_config_detail(self.robot_config_id)
        assertions.assert_code(config_response.status_code, 200)
        robot_config = config_response.json().get("data", {}).get("config_json")
        files = query_all_collect_files(self.api_all, self.task_id, "uploaded")
        playlist_response = self.api_all.query_task_video_playlist(self.task_id)
        assertions.assert_code(playlist_response.status_code, 200)
        playlist = playlist_response.json().get("data")
        if not isinstance(robot_config, dict) or not isinstance(playlist, dict):
            pytest.fail("自动化标注提交所需配置不完整")

        payload = {
            "job_id": str(job_id),
            "basic": {"task_id": str(self.task_id), "user_config_path": None},
            "input": {
                "file_path": [item["s3_uri"] for item in files if item.get("s3_uri")],
                "file_type": "mcap",
                "video_playlist": json.dumps(playlist, ensure_ascii=False, separators=(",", ":")),
                "robot_config": json.dumps(robot_config, ensure_ascii=False, separators=(",", ":")),
                "user_prompt": os.environ.get(
                    "VLA_AUTO_LABELING_PROMPT",
                    "请描述机器人完成的子任务,可选子任务为：\n"
                    "夹取物体：机械臂移动靠近物体，末端夹爪角度减小以夹持物体；\n"
                    "移动物体：夹取物体后的移动状态；\n"
                    "放下物体：机械臂末端夹爪张开，放开物体。",
                ),
                "video_fps": int(os.environ.get("VLA_AUTO_LABELING_FPS", "30")),
            },
        }
        response = self.api_all.submit_auto_labeling_task(payload)
        if response.status_code not in {200, 202}:
            pytest.fail(f"提交自动化标注失败：HTTP {response.status_code}，响应：{response.text}")
        submitted = response.json().get("data", {})
        self.__class__.auto_labeling_job_id = submitted.get("job_id") or job_id
        if submitted.get("status") not in {"QUEUED", "RUNNING"}:
            pytest.fail(f"自动化标注提交后状态异常：{submitted}")

    @pytest.mark.order(10)
    @allure.story("步骤10：轮询自动化标注状态")
    def test_wait_for_auto_labeling(self):
        if not self.task_id or not self.auto_labeling_job_id:
            pytest.fail("缺少自动化标注 job_id")
        started_at = time.monotonic()
        while True:
            response = self.api_all.query_auto_labeling_task(self.task_id, self.auto_labeling_job_id)
            assertions.assert_code(response.status_code, 200)
            data = response.json().get("data", {})
            status = data.get("status")
            status_code = data.get("status_code")
            print(
                f"[步骤10] 自动化标注轮询：status={status}，"
                f"status_code={status_code}，"
                f"progress={data.get('progress_percent', 0)}%，"
                f"phase={data.get('progress_phase')}，"
                f"message={data.get('progress_message')}",
                flush=True,
            )
            if status == "SUCCEEDED" or status_code == 4:
                # 页面点击“查看并编辑自动标注结果”时会调用该任务级接口，
                # 由接口触发/刷新结果落库；自动化流程中也需在任务完成后主动调用一次。
                result_response = self.api_all.query_auto_labeling_job_result(
                    self.task_id, self.auto_labeling_job_id
                )
                assertions.assert_code(result_response.status_code, 200)
                result_data = result_response.json()
                if result_data.get("msg") not in {"success", "ok"}:
                    pytest.fail(
                        "刷新自动化标注结果失败："
                        f"HTTP {result_response.status_code}，响应：{result_data}"
                    )
                # 保存“查看并编辑自动标注结果”接口的完整响应；该响应是
                # 组装工作台 segments 的权威自动标注结果来源。
                self.__class__.auto_labeling_job_result = result_data.get("data", {})
                elapsed = time.monotonic() - started_at
                video_duration_text = "未知"
                playlist_response = self.api_all.query_task_video_playlist(self.task_id)
                if playlist_response.status_code == 200:
                    playlist_data = playlist_response.json().get("data", {})
                    timeline = playlist_data.get("timeline", {}) if isinstance(playlist_data, dict) else {}
                    try:
                        video_duration_sec = float(timeline.get("duration_sec"))
                    except (TypeError, ValueError):
                        video_duration_sec = None
                    if video_duration_sec is not None and video_duration_sec >= 0:
                        self.__class__.annotation_video_duration_sec = video_duration_sec
                        video_duration_text = format_elapsed_minutes_seconds(video_duration_sec)
                print(
                    f"[步骤10] 自动化标注完成，耗时：{format_elapsed_minutes_seconds(elapsed)} "
                    f"（{elapsed:.2f}秒）；标注视频总时长：{video_duration_text}",
                    flush=True,
                )
                print("[步骤10] 已刷新自动化标注结果，页面可读取结果", flush=True)
                return
            if status in {"FAILED", "CANCELED", "CANCELLED", "TIMEOUT"} or status_code in {5, 6, 7, 8}:
                elapsed = time.monotonic() - started_at
                pytest.fail(
                    f"自动化标注执行失败（已耗时：{format_elapsed_minutes_seconds(elapsed)}）：{data}"
                )
            if time.monotonic() - started_at >= AUTO_LABELING_POLL_TIMEOUT_SECONDS:
                pytest.fail(f"自动化标注轮询超时：{AUTO_LABELING_POLL_TIMEOUT_SECONDS} 秒")
            time.sleep(AUTO_LABELING_POLL_INTERVAL_SECONDS)

    @pytest.mark.order(11)
    @allure.story("步骤11：校验自动化标注结果")
    def test_verify_auto_labeling_result(self):
        if not self.task_id:
            pytest.fail("缺少 task_id，无法查询自动化标注结果")
        response = self.api_all.query_latest_auto_labeling_job(self.task_id)
        assertions.assert_code(response.status_code, 200)
        data = response.json().get("data", {})
        if data.get("status") != "SUCCEEDED" or data.get("status_code") != 4:
            pytest.fail(f"自动化标注最终状态异常：{data}")
        result_data = self.auto_labeling_job_result or data
        result = result_data.get("result", {}) if isinstance(result_data, dict) else {}
        episodes = result.get("episodes_annotation", [])
        if not isinstance(episodes, list) or not episodes:
            pytest.fail("自动化标注成功，但 episodes_annotation 为空")
        self.__class__.auto_labeling_result = result_data

        workspace_response = self.api_all.query_annotation_workspace(self.task_id)
        assertions.assert_code(workspace_response.status_code, 200)
        workspace_response_data = workspace_response.json()
        assertions.assert_text(workspace_response_data.get("msg", ""), "success")
        workspace_data = workspace_response_data.get("data", {})
        annotation = workspace_data.get("annotation", {}) if isinstance(workspace_data, dict) else {}
        annotation_id = annotation.get("annotation_id") or annotation.get("id")
        if not annotation_id:
            pytest.fail(f"标注工作区未返回 annotation_id：{workspace_data}")

        playlist_response = self.api_all.query_task_video_playlist(self.task_id)
        assertions.assert_code(playlist_response.status_code, 200)
        playlist_data = playlist_response.json().get("data", {})
        playback = _workspace_playback(workspace_data, playlist_data)
        if not playback:
            pytest.fail(
                "无法从 annotation-workspace 或 video-playlist 获取 playback；"
                f"workspace={workspace_data}，playlist={playlist_data}"
            )
        required_playback_fields = ("baseline_camera_key", "baseline_start_time_ns")
        missing_playback_fields = [key for key in required_playback_fields if not playback.get(key)]
        if missing_playback_fields:
            pytest.fail(f"playback 缺少字段 {missing_playback_fields}：{playback}")

        job_id = data.get("job_id") or self.auto_labeling_job_id
        if not job_id:
            pytest.fail("自动标注结果未返回 job_id")
        segments = _build_auto_segments(result, job_id, playback)
        if not segments:
            pytest.fail(
                "无法从 episodes_annotation 构造可回显的 segments；"
                f"结果字段：{json.dumps(result, ensure_ascii=False)}"
            )
        if not any(segment["category"] == "episode" for segment in segments):
            pytest.fail("自动标注结果未构造出 episode 分段，拒绝写入")

        tag_vocabulary = (
            annotation.get("tag_vocabulary")
            or annotation.get("tagVocabulary")
            or workspace_data.get("tag_vocabulary")
            or workspace_data.get("tagVocabulary")
            or self.scene_tags
            or []
        )
        if not isinstance(tag_vocabulary, list):
            pytest.fail(f"tag_vocabulary 不是列表：{tag_vocabulary!r}")
        payload = {
            "tag_vocabulary": tag_vocabulary,
            "segments": segments,
            "deleted_segments": [],
            "playback": playback,
            "status": "DRAFT",
        }
        allure.attach(
            json.dumps(payload, ensure_ascii=False, indent=2),
            name="自动标注结果回写 segments 请求参数",
            attachment_type=allure.attachment_type.JSON,
        )
        patch_response = self.api_all.create_task_annotation_segments(
            annotation_id=annotation_id,
            tag_vocabulary=tag_vocabulary,
            segments=segments,
            playback=playback,
        )
        assertions.assert_code(patch_response.status_code, 200)
        assertions.assert_text(patch_response.json().get("msg", ""), "success")
        print(
            f"[步骤11] 已回写自动标注 segments：annotation_id={annotation_id}，"
            f"episode={sum(item['category'] == 'episode' for item in segments)}，"
            f"sub_task={sum(item['category'] == 'sub_task' for item in segments)}",
            flush=True,
        )


@allure.feature("工作台数据校验")
class TestWorkbenchData:
    """校验工作台任务概览与 PostgreSQL tasks 表统计一致。"""

    @classmethod
    def setup_class(cls):
        cls.api_all = all_api.ApiAll(global_client)
        cls.collection_duration_recalculated_sec = None
        cls.annotation_duration_recalculated_sec = None
        cls.episode_count_recalculated = None
        cls.episode_duration_recalculated_sec = None
        cls.workforce_people = None
        config_path = Path(__file__).resolve().parent.parent / "config" / "env_config.ini"
        cls.config = configparser.ConfigParser()
        if not cls.config.read(config_path, encoding="utf-8"):
            pytest.fail(f"无法读取配置文件：{config_path}")
        if not cls.config.has_section("fat-vla"):
            pytest.fail("配置文件缺少 [fat-vla] 节")

    @contextmanager
    def _postgres_connection(self):
        """按环境配置创建 PostgreSQL 连接，并确保离开作用域时关闭。"""
        try:
            import psycopg2
        except ImportError:
            pytest.fail("缺少 PostgreSQL 驱动，请先安装 requirements.txt 中的 psycopg2-binary")

        section = self.config["fat-vla"]
        connection = psycopg2.connect(
            host=section.get("db_host"),
            port=section.getint("db_port", fallback=5432),
            user=section.get("db_user"),
            password=section.get("db_password"),
            dbname=section.get("db_name"),
            connect_timeout=30,
        )
        try:
            yield connection
        finally:
            connection.close()

    def _database_counts(self):
        """查询 tasks 总数及各状态数量，连接在查询完成后关闭。"""
        try:
            with self._postgres_connection() as connection, connection.cursor() as cursor:
                cursor.execute(
                    "SELECT COUNT(*) FROM tasks"
                )
                counts = {"total": int(cursor.fetchone()[0])}
                for status in range(1, 6):
                    cursor.execute(
                        "SELECT COUNT(*) FROM tasks WHERE status = %s", (status,)
                    )
                    counts[status] = int(cursor.fetchone()[0])
                return counts
        except Exception as exc:
            pytest.fail(f"查询 PostgreSQL tasks 表失败：{exc}")

    def _collected_task_nos(self):
        """返回已采集、已标注、已质检或已转换任务的 task_no。"""
        try:
            with self._postgres_connection() as connection, connection.cursor() as cursor:
                cursor.execute(
                    "SELECT task_no FROM tasks WHERE status IN (2, 3, 4, 5) "
                    "AND task_no IS NOT NULL AND task_no <> '' ORDER BY task_no"
                )
                return [str(row[0]).strip() for row in cursor.fetchall() if str(row[0]).strip()]
        except Exception as exc:
            pytest.fail(f"查询已采集任务 task_no 失败：{exc}")

    def _annotation_duration_rows(self, task_nos):
        """查询指定任务的全部标注片段起止纳秒数。"""
        if not task_nos:
            return []
        try:
            with self._postgres_connection() as connection, connection.cursor() as cursor:
                cursor.execute(
                    "SELECT task_no, start_time_ns, end_time_ns "
                    "FROM annotation_segment_items "
                    "WHERE task_no = ANY(%s) AND deleted_flag = FALSE "
                    "ORDER BY task_no, segment_item_id",
                    (task_nos,),
                )
                return cursor.fetchall()
        except Exception as exc:
            pytest.fail(f"查询 annotation_segment_items 表失败：{exc}")

    def _episode_count(self, task_nos):
        """统计指定任务下 segment_kind=episode 的标注数量。"""
        if not task_nos:
            return 0
        try:
            with self._postgres_connection() as connection, connection.cursor() as cursor:
                cursor.execute(
                    "SELECT COUNT(*) FROM annotation_segment_items "
                    "WHERE task_no = ANY(%s) AND segment_kind = %s "
                    "AND deleted_flag = FALSE",
                    (task_nos, "episode"),
                )
                return int(cursor.fetchone()[0])
        except Exception as exc:
            pytest.fail(f"查询 episode 标注数量失败：{exc}")

    def _episode_duration_statistics(self):
        """按开发口径统计全部视频转换 Episode 的总时长及数量。"""
        try:
            with self._postgres_connection() as connection, connection.cursor() as cursor:
                cursor.execute(
                    "SELECT COALESCE(SUM(dur), 0), COUNT(*) "
                    "FROM ("
                    "    SELECT conversion_outputs.task_id, conversion_outputs.episode_id, "
                    "           MAX(conversion_outputs.duration_sec) AS dur "
                    "    FROM conversion_outputs "
                    "    WHERE conversion_outputs.kind = %s "
                    "    GROUP BY conversion_outputs.task_id, conversion_outputs.episode_id"
                    ") AS episode_durations",
                    ("video",),
                )
                total_duration_sec, episode_count = cursor.fetchone()
                # 所有 Episode 时长汇总后，再按页面规则统一保留两位小数。
                return (
                    Decimal(str(total_duration_sec)).quantize(
                        Decimal("0.01"), rounding=ROUND_HALF_UP
                    ),
                    int(episode_count),
                )
        except Exception as exc:
            pytest.fail(f"查询 Episode 标注时长失败：{exc}")

    def _workforce_completion_rows(self):
        """按采集、标注、质检和转换子任务统计人员完成数及完成率。"""
        query = """
            WITH assignments AS (
                SELECT task_id, collector_id AS user_id,
                       CASE WHEN status IN (2, 3, 4, 5) THEN 1 ELSE 0 END AS is_completed
                FROM tasks WHERE collector_id IS NOT NULL

                UNION ALL

                SELECT task_id, annotator_id AS user_id,
                       CASE WHEN status IN (3, 4, 5) THEN 1 ELSE 0 END AS is_completed
                FROM tasks WHERE annotator_id IS NOT NULL

                UNION ALL

                SELECT task_id, qc_id AS user_id,
                       CASE WHEN status IN (4, 5) THEN 1 ELSE 0 END AS is_completed
                FROM tasks WHERE qc_id IS NOT NULL

                UNION ALL

                SELECT
                    task_id,
                    creator_id AS user_id,
                    CASE WHEN BOOL_OR(status = 'completed') THEN 1 ELSE 0 END AS is_completed
                FROM conversion_jobs
                WHERE creator_id IS NOT NULL
                GROUP BY task_id, creator_id
            ),
            totals AS (
                SELECT user_id, COUNT(*) AS task_count,
                       SUM(is_completed) AS completed_task_count
                FROM assignments
                GROUP BY user_id
            )
            SELECT
                u.user_id,
                u.username,
                u.full_name,
                COALESCE(t.task_count, 0) AS task_count,
                COALESCE(t.completed_task_count, 0) AS completed_task_count,
                ROUND(
                    COALESCE(t.completed_task_count, 0)::numeric
                    / NULLIF(COALESCE(t.task_count, 0), 0),
                    2
                ) AS completion_rate
            FROM users u
            LEFT JOIN totals t ON t.user_id = u.user_id
            ORDER BY u.user_id
        """
        try:
            with self._postgres_connection() as connection, connection.cursor() as cursor:
                cursor.execute(query)
                columns = [description[0] for description in cursor.description]
                return [dict(zip(columns, row)) for row in cursor.fetchall()]
        except Exception as exc:
            pytest.fail(f"查询人员完成任务数失败：{exc}")

    @pytest.mark.order(12)
    @allure.story("步骤1：任务统计-验证任务状态数据")
    def test_task_status_statistics(self):
        with allure.step("调用工作台任务概览接口"):
            response = self.api_all.query_task_dashboard()
            assertions.assert_code(response.status_code, 200)
            response_data = response.json()
            assertions.assert_text(response_data.get("msg", ""), "success")
            statistics = response_data.get("data", {}).get("statistics")
            if not isinstance(statistics, dict):
                pytest.fail("工作台任务概览响应缺少 data.statistics")

        db_counts = self._database_counts()
        expected = {
            "total": db_counts["total"],
            "created": db_counts[1],
            "collected": db_counts[2],
            "annotated": db_counts[3],
            "qc_done": db_counts[4],
            "converted": db_counts[5],
        }
        actual = {}
        for key, db_value in expected.items():
            value = statistics.get(key)
            if value is None:
                pytest.fail(f"工作台统计缺少字段：data.statistics.{key}")
            try:
                actual[key] = int(value)
            except (TypeError, ValueError):
                pytest.fail(f"工作台统计字段 {key} 不是数字：{value!r}")
        display_names = {
            "total": "总任务数量",
            "created": "已创建",
            "collected": "已采集",
            "annotated": "已标注",
            "qc_done": "已质检",
            "converted": "已转换",
        }
        actual_display = {display_names[key]: value for key, value in actual.items()}
        expected_display = {display_names[key]: value for key, value in expected.items()}
        print("[工作台任务状态] 接口统计：", actual_display, flush=True)
        print("[工作台任务状态] 数据库统计：", expected_display, flush=True)
        print("[工作台任务状态] 对比结果：", flush=True)
        for key, db_value in expected.items():
            print(
                f"  {display_names[key]}: 接口={actual[key]}，数据库={db_value}，"
                f"结果={'一致' if actual[key] == db_value else '不一致'}",
                flush=True,
            )
            assertions.assert_text(str(actual[key]), str(db_value))

        allure.attach(
            json.dumps({"api": actual, "database": expected}, ensure_ascii=False, indent=2),
            name="工作台任务状态统计对比",
            attachment_type=allure.attachment_type.JSON,
        )

    @pytest.mark.order(13)
    @allure.story("步骤2：任务统计-验证总采集时长")
    def test_collection_duration_statistics(self):
        """按任务转换表中的 MCAP 帧相对时间重算总采集时长。"""
        print("[步骤2] 开始调用采集时长统计接口...", flush=True)
        response = self.api_all.query_duration_stats(period="all")
        assertions.assert_code(response.status_code, 200)
        response_data = response.json()
        assertions.assert_text(response_data.get("msg", ""), "success")
        metrics = response_data.get("data", {}).get("metrics")
        if not isinstance(metrics, dict):
            pytest.fail("工作台采集时长响应缺少 data.metrics")
        try:
            api_collect_sec = Decimal(str(metrics["collect_duration_sec"])).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
        except (KeyError, TypeError, ValueError, InvalidOperation) as exc:
            pytest.fail(f"data.metrics.collect_duration_sec 不是有效数字：{exc}")
        print(f"[步骤2] 接口采集总时长：{api_collect_sec} 秒", flush=True)

        print("[步骤2] 正在查询符合条件的任务（状态 2/3/4/5）...", flush=True)
        task_nos = self._collected_task_nos()
        print(f"[步骤2] 共找到 {len(task_nos)} 个任务，开始查询数据库 MCAP 转换表...", flush=True)
        total_duration_sec = Decimal("0")
        task_details = []
        try:
            from psycopg2 import sql

            with self._postgres_connection() as connection, connection.cursor() as cursor:
                for task_index, task_no in enumerate(task_nos, start=1):
                    # TASK-2026-006 -> mcap_video_frames_task_2026_006.
                    normalized_task_no = str(task_no).strip().lower().replace("-", "_")
                    if not normalized_task_no or not normalized_task_no.startswith("task_"):
                        raise ValueError(f"任务编号格式无效，无法定位转换表：{task_no!r}")
                    table_name = f"mcap_video_frames_{normalized_task_no}"
                    cursor.execute(
                        sql.SQL(
                            "SELECT task_id, task_file_id, MAX(relative_time_sec) AS duration_sec "
                            "FROM {} GROUP BY task_id, task_file_id ORDER BY task_id, task_file_id"
                        ).format(sql.Identifier(table_name))
                    )
                    rows = cursor.fetchall()
                    task_duration_sec = Decimal("0")
                    for _task_id, _task_file_id, duration in rows:
                        if duration is None:
                            continue
                        duration_decimal = Decimal(str(duration))
                        if duration_decimal < 0:
                            raise ValueError(
                                f"任务 {task_no} 转换表 {table_name} 存在负数 MCAP 时长：{duration}"
                            )
                        task_duration_sec += duration_decimal
                    total_duration_sec += task_duration_sec
                    task_details.append(
                        {
                            "task_no": task_no,
                            "table_name": table_name,
                            "mcap_count": len(rows),
                            "duration_sec": float(task_duration_sec),
                        }
                    )
                    print(
                        f"[步骤2] ({task_index}/{len(task_nos)}) 任务 {task_no} 完成，"
                        f"{len(rows)} 个 MCAP，时长 {task_duration_sec:.6f} 秒",
                        flush=True,
                    )
        except Exception as exc:
            pytest.fail(f"查询 MCAP 转换表统计采集时长失败：{exc}")

        # 所有任务、所有 MCAP 汇总后按页面展示规则统一保留两位小数。
        expected_collect_sec = total_duration_sec.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        self.__class__.collection_duration_recalculated_sec = expected_collect_sec
        total_mcap_count = sum(detail["mcap_count"] for detail in task_details)
        actual_error_sec = abs(api_collect_sec - expected_collect_sec)
        print(
            f"[工作台采集时长] 接口={api_collect_sec} 秒，数据库重算={expected_collect_sec} 秒，"
            f"任务数={len(task_nos)}，MCAP数量={total_mcap_count}，实际误差={actual_error_sec:.2f} 秒",
            flush=True,
        )
        if actual_error_sec != Decimal("0.00"):
            pytest.fail(
                f"采集总时长不一致：接口={api_collect_sec}，数据库重算={expected_collect_sec}，"
                f"实际误差={actual_error_sec:.2f} 秒"
            )
        allure.attach(
            json.dumps(
                {
                    "api_collect_sec": api_collect_sec,
                    "db_collect_sec": expected_collect_sec,
                    "mcap_count": total_mcap_count,
                    "actual_error_sec": actual_error_sec,
                    "tasks": task_details,
                },
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
            name="采集总时长统计对比",
            attachment_type=allure.attachment_type.JSON,
        )

    @pytest.mark.order(14)
    @allure.story("步骤3：任务统计-验证标注总时长")
    def test_annotation_duration_statistics(self):
        """按已标注及后续状态任务的标注片段重算标注总时长。"""
        print("[步骤3] 开始读取工作台标注时长统计接口...", flush=True)
        response = self.api_all.query_duration_stats(period="all")
        assertions.assert_code(response.status_code, 200)
        response_data = response.json()
        assertions.assert_text(response_data.get("msg", ""), "success")
        metrics = response_data.get("data", {}).get("metrics")
        if not isinstance(metrics, dict):
            pytest.fail("工作台标注时长响应缺少 data.metrics")
        try:
            api_annotation_sec = Decimal(str(metrics["annotation_duration_sec"])).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
        except (KeyError, TypeError, ValueError, InvalidOperation) as exc:
            pytest.fail(f"data.metrics.annotation_duration_sec 不是有效数字：{exc}")
        print(f"[步骤3] 接口标注总时长：{api_annotation_sec} 秒", flush=True)

        print("[步骤3] 正在查询已标注/已质检/已转换任务...", flush=True)
        # 标注时长只统计已标注、已质检、已转换任务。
        try:
            with self._postgres_connection() as connection:
                with connection.cursor() as cursor:
                    cursor.execute(
                        "SELECT task_no FROM tasks WHERE status IN (3, 4, 5) "
                        "AND task_no IS NOT NULL AND task_no <> '' ORDER BY task_no"
                    )
                    task_nos = [str(row[0]).strip() for row in cursor.fetchall()]
        except Exception as exc:
            pytest.fail(f"查询标注状态任务失败：{exc}")
        print(f"[步骤3] 共找到 {len(task_nos)} 个任务，查询标注片段...", flush=True)

        rows = self._annotation_duration_rows(task_nos)
        total_annotation_ns = 0
        task_details = {task_no: {"segment_count": 0, "duration_sec": 0.0} for task_no in task_nos}
        for task_no, start_time_ns, end_time_ns in rows:
            try:
                start = int(start_time_ns)
                end = int(end_time_ns)
            except (TypeError, ValueError) as exc:
                pytest.fail(f"任务 {task_no} 存在无效标注时间：{exc}")
            duration = end - start
            if duration < 0:
                pytest.fail(f"任务 {task_no} 标注时间范围无效：start={start}，end={end}")
            total_annotation_ns += duration
            detail = task_details.setdefault(str(task_no), {"segment_count": 0, "duration_sec": 0.0})
            detail["segment_count"] += 1
            detail["duration_sec"] += duration / 1_000_000_000
        # 标注片段全部汇总后，按页面展示规则统一保留两位小数。
        total_annotation_sec = (
                Decimal(total_annotation_ns) / Decimal("1000000000")
        ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        self.__class__.annotation_duration_recalculated_sec = total_annotation_sec
        actual_error_sec = abs(api_annotation_sec - total_annotation_sec)
        print(
            f"[步骤3] 标注片段数={len(rows)}，数据库={total_annotation_sec:.6f} 秒，"
            f"接口={api_annotation_sec:.6f} 秒，误差={actual_error_sec:.6f} 秒",
            flush=True,
        )
        if actual_error_sec > Decimal("0.00"):
            pytest.fail(
                f"标注总时长不一致：接口={api_annotation_sec}，数据库={total_annotation_sec}，"
                f"误差={actual_error_sec:.6f} 秒"
            )
        allure.attach(
            json.dumps(
                {"api_annotation_sec": float(api_annotation_sec), "db_annotation_sec": float(total_annotation_sec),
                 "segment_count": len(rows), "tasks": task_details},
                ensure_ascii=False, indent=2,
            ),
            name="标注总时长统计对比", attachment_type=allure.attachment_type.JSON,
        )

    @pytest.mark.order(15)
    @allure.story("步骤4：任务统计-验证 Episode 总个数")
    def test_episode_count_statistics(self):
        """按任务状态和 annotation_segment_items 重算 Episode 数量。"""
        print("[步骤4] 开始读取工作台 Episode 统计接口...", flush=True)
        response = self.api_all.query_duration_stats(period="all")
        assertions.assert_code(response.status_code, 200)
        response_data = response.json()
        assertions.assert_text(response_data.get("msg", ""), "success")
        metrics = response_data.get("data", {}).get("metrics")
        if not isinstance(metrics, dict):
            pytest.fail("工作台 Episode 统计响应缺少 data.metrics")
        try:
            api_episode_count = int(metrics["episode_count"])
        except (KeyError, TypeError, ValueError) as exc:
            pytest.fail(f"data.metrics.episode_count 不是有效整数：{exc}")
        print(f"[步骤4] 接口 Episode 总数：{api_episode_count}", flush=True)

        print("[步骤4] 正在查询已转换任务...", flush=True)
        task_nos = []
        try:
            with self._postgres_connection() as connection:
                with connection.cursor() as cursor:
                    cursor.execute(
                        "SELECT task_no FROM tasks WHERE status = 5 "
                        "AND task_no IS NOT NULL AND task_no <> '' ORDER BY task_no"
                    )
                    task_nos = [str(row[0]).strip() for row in cursor.fetchall()]
        except Exception as exc:
            pytest.fail(f"查询 Episode 状态任务失败：{exc}")

        db_episode_count = self._episode_count(task_nos)
        self.__class__.episode_count_recalculated = db_episode_count
        print(
            f"[步骤4] 任务数={len(task_nos)}，数据库 Episode 数={db_episode_count}，"
            f"接口 Episode 数={api_episode_count}",
            flush=True,
        )
        if api_episode_count != db_episode_count:
            pytest.fail(
                f"Episode 总数不一致：接口={api_episode_count}，数据库={db_episode_count}"
            )
        allure.attach(
            json.dumps(
                {"api_episode_count": api_episode_count, "db_episode_count": db_episode_count,
                 "task_count": len(task_nos), "task_nos": task_nos},
                ensure_ascii=False, indent=2,
            ),
            name="Episode 总数统计对比", attachment_type=allure.attachment_type.JSON,
        )

    @pytest.mark.order(16)
    @allure.story("步骤5：任务统计-验证 Episode 总时长")
    def test_episode_duration_statistics(self):
        """按全部视频转换产物重算 Episode 总时长。"""
        print("[步骤5] 开始读取工作台转换时长统计接口...", flush=True)
        response = self.api_all.query_duration_stats(period="all")
        assertions.assert_code(response.status_code, 200)
        response_data = response.json()
        assertions.assert_text(response_data.get("msg", ""), "success")
        metrics = response_data.get("data", {}).get("metrics")
        if not isinstance(metrics, dict):
            pytest.fail("工作台 Episode 时长响应缺少 data.metrics")
        try:
            api_conversion_sec = Decimal(str(metrics["episode_duration_sec"])).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
        except (KeyError, TypeError, ValueError, InvalidOperation) as exc:
            pytest.fail(f"data.metrics.episode_duration_sec 不是有效数字：{exc}")
        print(f"[步骤5] 接口 Episode 总时长：{api_conversion_sec} 秒", flush=True)

        print("[步骤5] 正在汇总全部视频转换产物...", flush=True)
        db_conversion_sec, episode_count = self._episode_duration_statistics()
        self.__class__.episode_duration_recalculated_sec = db_conversion_sec
        actual_error_sec = abs(api_conversion_sec - db_conversion_sec)
        print(
            f"[步骤5] 去重后 Episode 数={episode_count}，数据库 Episode 时长={db_conversion_sec:.6f} 秒，"
            f"接口={api_conversion_sec:.6f} 秒，误差={actual_error_sec:.6f} 秒",
            flush=True,
        )
        if actual_error_sec > Decimal("0.00"):
            pytest.fail(
                f"Episode 总时长不一致：接口={api_conversion_sec}，"
                f"数据库={db_conversion_sec}，误差={actual_error_sec:.6f} 秒"
            )
        allure.attach(
            json.dumps(
                {"api_conversion_sec": float(api_conversion_sec), "db_episode_sec": float(db_conversion_sec),
                 "episode_count": episode_count,
                 "aggregation": "每个 task_id + episode_id 取最大 video 时长后汇总",
                 "rounding": "所有 Episode 时长先累加，最后统一四舍五入到两位小数"},
                ensure_ascii=False, indent=2,
            ),
            name="Episode 总时长统计对比", attachment_type=allure.attachment_type.JSON,
        )

    @pytest.mark.order(17)
    @allure.story("步骤6：人效统计-验证参与人员")
    def test_workforce_people_statistics(self):
        """按人效统计明细中的角色重算参与人员数量。"""
        print("[步骤6] 开始读取工作台人效统计接口...", flush=True)
        response = self.api_all.query_workforce_stats(period="all")
        assertions.assert_code(response.status_code, 200)
        response_data = response.json()
        assertions.assert_text(response_data.get("msg", ""), "success")

        data = response_data.get("data", {})
        rows = data.get("rows")
        summary_people = data.get("summary", {}).get("people")
        if not isinstance(rows, list):
            pytest.fail("工作台人效统计响应缺少有效 data.rows 列表")
        if not isinstance(summary_people, dict):
            pytest.fail("工作台人效统计响应缺少有效 data.summary.people")

        role_to_summary_key = {
            "采集工程师": "collection",
            "标注工程师": "annotation",
            "质检工程师": "qc",
            "算法工程师": "algorithm",
        }
        expected_people = {
            "total": len(rows),
            "collection": 0,
            "annotation": 0,
            "qc": 0,
            "algorithm": 0,
        }
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                pytest.fail(f"data.rows[{index}] 不是对象")
            role_names = row.get("role_names", [])
            if role_names is None:
                role_names = []
            if not isinstance(role_names, list):
                pytest.fail(f"data.rows[{index}].role_names 不是列表")
            # 一个用户拥有多个目标角色时，按角色分别计数；管理员同时计入所有目标角色。
            role_names = {str(role).strip() for role in role_names}
            if "管理员" in role_names:
                role_names.update(role_to_summary_key)
            for role, summary_key in role_to_summary_key.items():
                if role in role_names:
                    expected_people[summary_key] += 1

        actual_people = {}
        for key, expected_value in expected_people.items():
            value = summary_people.get(key)
            try:
                actual_value = int(value)
            except (TypeError, ValueError):
                pytest.fail(f"data.summary.people.{key} 不是有效整数：{value!r}")
            if actual_value != expected_value:
                pytest.fail(
                    f"参与人员统计不一致：{key} 接口={actual_value}，"
                    f"按 data.rows 角色重算={expected_value}"
                )
            actual_people[key] = actual_value

        print(
            f"[步骤6] 参与人员统计校验通过：总人数={actual_people['total']}，"
            f"采集={actual_people['collection']}，标注={actual_people['annotation']}，"
            f"质检={actual_people['qc']}，算法={actual_people['algorithm']}",
            flush=True,
        )
        allure.attach(
            json.dumps(
                {"summary_people": actual_people, "rows_count": len(rows)},
                ensure_ascii=False,
                indent=2,
            ),
            name="参与人员统计对比",
            attachment_type=allure.attachment_type.JSON,
        )
        self.__class__.workforce_people = actual_people

    @pytest.mark.order(18)
    @allure.story("步骤7：人效统计-验证采集统计")
    def test_workforce_collection_statistics(self):
        """核对采集任务数、采集总时长及按参与人员计算的平均值。"""
        print("[步骤7] 开始读取工作台人效采集统计接口...", flush=True)
        # 在实际校验前刷新一次数据库基准，避免使用步骤2的过期结果。
        self.test_collection_duration_statistics()
        if not isinstance(self.workforce_people, dict):
            pytest.fail("步骤6未保存参与人员统计结果，无法验证步骤7")

        response = self.api_all.query_workforce_stats(period="all")
        assertions.assert_code(response.status_code, 200)
        response_data = response.json()
        assertions.assert_text(response_data.get("msg", ""), "success")
        collect_summary = response_data.get("data", {}).get("summary", {}).get("collect")
        if not isinstance(collect_summary, dict):
            pytest.fail("工作台人效统计响应缺少有效 data.summary.collect")

        expected_task_count = len(self._collected_task_nos())
        try:
            api_task_count = int(collect_summary["task_count"])
            api_total_duration_sec = Decimal(str(collect_summary["total_duration_sec"]))
            api_avg_tasks = Decimal(str(collect_summary["avg_tasks"]))
            api_avg_duration_sec = Decimal(str(collect_summary["avg_duration_sec"]))
        except (KeyError, TypeError, ValueError, InvalidOperation) as exc:
            pytest.fail(f"data.summary.collect 存在无效统计值：{exc}")

        if api_task_count != expected_task_count:
            pytest.fail(
                f"采集任务数不一致：接口={api_task_count}，"
                f"tasks 表状态 2/3/4/5={expected_task_count}"
            )

        expected_duration_sec = Decimal(str(self.collection_duration_recalculated_sec))
        duration_error = abs(api_total_duration_sec - expected_duration_sec)
        if duration_error > Decimal("1.00"):
            pytest.fail(
                f"采集总时长不一致：接口={api_total_duration_sec}，"
                f"步骤2重算={expected_duration_sec}，误差={duration_error} 秒"
            )

        collection_people = int(self.workforce_people.get("collection", 0))
        if collection_people <= 0:
            pytest.fail(f"步骤6采集人员数量无效：{collection_people}")
        expected_avg_tasks = (
                Decimal(expected_task_count) / Decimal(collection_people)
        ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        expected_avg_duration = (
                expected_duration_sec / Decimal(collection_people)
        ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        actual_avg_tasks = api_avg_tasks.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        actual_avg_duration = api_avg_duration_sec.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        if actual_avg_tasks != expected_avg_tasks:
            pytest.fail(
                f"采集平均任务数不一致：接口={actual_avg_tasks}，"
                f"任务数/采集人员数({collection_people})={expected_avg_tasks}"
            )
        if actual_avg_duration != expected_avg_duration:
            pytest.fail(
                f"采集平均时长不一致：接口={actual_avg_duration}，"
                f"步骤2重算采集总时长/采集人员数({collection_people})={expected_avg_duration}"
            )

        print(
            f"[步骤7] 采集统计校验通过：任务数={api_task_count}，"
            f"总时长={api_total_duration_sec} 秒，采集人员数={collection_people}，"
            f"平均任务数={actual_avg_tasks}，平均时长={actual_avg_duration} 秒",
            flush=True,
        )
        allure.attach(
            json.dumps(
                {
                    "api": {
                        "task_count": api_task_count,
                        "total_duration_sec": str(api_total_duration_sec),
                        "avg_tasks": str(api_avg_tasks),
                        "avg_duration_sec": str(api_avg_duration_sec),
                    },
                    "expected": {
                        "task_count": expected_task_count,
                        "duration_source": "步骤2重算",
                        "people_count": collection_people,
                        "avg_tasks": str(expected_avg_tasks),
                        "avg_duration_sec": str(expected_avg_duration),
                    },
                },
                ensure_ascii=False,
                indent=2,
            ),
            name="采集统计对比",
            attachment_type=allure.attachment_type.JSON,
        )

    @pytest.mark.order(19)
    @allure.story("步骤8：人效统计-验证标注统计")
    def test_workforce_annotation_statistics(self):
        """核对标注任务数、标注总时长及按参与人员计算的平均值。"""
        print("[步骤8] 开始读取工作台人效标注统计接口...", flush=True)
        # 在实际校验前刷新一次数据库基准，避免使用步骤3的过期结果。
        self.test_annotation_duration_statistics()
        if not isinstance(self.workforce_people, dict):
            pytest.fail("步骤6未保存参与人员统计结果，无法验证步骤8")

        response = self.api_all.query_workforce_stats(period="all")
        assertions.assert_code(response.status_code, 200)
        response_data = response.json()
        assertions.assert_text(response_data.get("msg", ""), "success")
        annotation_summary = response_data.get("data", {}).get("summary", {}).get("annotation")
        if not isinstance(annotation_summary, dict):
            pytest.fail("工作台人效统计响应缺少有效 data.summary.annotation")

        try:
            with self._postgres_connection() as connection:
                with connection.cursor() as cursor:
                    cursor.execute(
                        "SELECT COUNT(*) FROM tasks WHERE status IN (3, 4, 5)"
                    )
                    expected_task_count = int(cursor.fetchone()[0])
        except Exception as exc:
            pytest.fail(f"查询标注任务数失败：{exc}")

        try:
            api_task_count = int(annotation_summary["task_count"])
            api_total_duration_sec = Decimal(str(annotation_summary["total_duration_sec"]))
            api_avg_tasks = Decimal(str(annotation_summary["avg_tasks"]))
            api_avg_duration_sec = Decimal(str(annotation_summary["avg_duration_sec"]))
        except (KeyError, TypeError, ValueError, InvalidOperation) as exc:
            pytest.fail(f"data.summary.annotation 存在无效统计值：{exc}")

        if api_task_count != expected_task_count:
            pytest.fail(
                f"标注任务数不一致：接口={api_task_count}，"
                f"tasks 表状态 3/4/5={expected_task_count}"
            )

        expected_duration_sec = self.annotation_duration_recalculated_sec
        if api_total_duration_sec != expected_duration_sec:
            pytest.fail(
                f"标注总时长不一致：接口={api_total_duration_sec}，"
                f"步骤3重算={expected_duration_sec}"
            )

        annotation_people = int(self.workforce_people.get("annotation", 0))
        if annotation_people <= 0:
            pytest.fail(f"步骤6标注人员数量无效：{annotation_people}")
        expected_avg_tasks = (
                Decimal(expected_task_count) / Decimal(annotation_people)
        ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        expected_avg_duration = (
                expected_duration_sec / Decimal(annotation_people)
        ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        actual_avg_tasks = api_avg_tasks.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        actual_avg_duration = api_avg_duration_sec.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        if actual_avg_tasks != expected_avg_tasks:
            pytest.fail(
                f"标注平均任务数不一致：接口={actual_avg_tasks}，"
                f"任务数/标注人员数({annotation_people})={expected_avg_tasks}"
            )
        if actual_avg_duration != expected_avg_duration:
            pytest.fail(
                f"标注平均时长不一致：接口={actual_avg_duration}，"
                f"步骤3标注总时长/标注人员数({annotation_people})={expected_avg_duration}"
            )

        print(
            f"[步骤8] 标注统计校验通过：任务数={api_task_count}，"
            f"总时长={api_total_duration_sec} 秒，标注人员数={annotation_people}，"
            f"平均任务数={actual_avg_tasks}，平均时长={actual_avg_duration} 秒",
            flush=True,
        )
        allure.attach(
            json.dumps(
                {
                    "api": {
                        "task_count": api_task_count,
                        "total_duration_sec": str(api_total_duration_sec),
                        "avg_tasks": str(api_avg_tasks),
                        "avg_duration_sec": str(api_avg_duration_sec),
                    },
                    "expected": {
                        "task_count": expected_task_count,
                        "duration_source": "步骤3重算",
                        "people_count": annotation_people,
                        "avg_tasks": str(expected_avg_tasks),
                        "avg_duration_sec": str(expected_avg_duration),
                    },
                },
                ensure_ascii=False,
                indent=2,
            ),
            name="标注统计对比",
            attachment_type=allure.attachment_type.JSON,
        )

    @pytest.mark.order(20)
    @allure.story("步骤9：人效统计-验证Episode统计")
    def test_workforce_episode_statistics(self):
        """核对 Episode 总数、总时长及按参与人员计算的平均值。"""
        print("[步骤9] 开始读取工作台人效 Episode 统计接口...", flush=True)
        # Episode 数量和时长在校验前各刷新一次，使用当前数据库数据。
        self.test_episode_count_statistics()
        self.test_episode_duration_statistics()
        if not isinstance(self.workforce_people, dict):
            pytest.fail("步骤6未保存参与人员统计结果，无法验证步骤9")

        response = self.api_all.query_workforce_stats(period="all")
        assertions.assert_code(response.status_code, 200)
        response_data = response.json()
        assertions.assert_text(response_data.get("msg", ""), "success")
        episode_summary = response_data.get("data", {}).get("summary", {}).get("episode")
        if not isinstance(episode_summary, dict):
            pytest.fail("工作台人效统计响应缺少有效 data.summary.episode")

        try:
            api_episode_count = int(episode_summary["task_count"])
            api_total_duration_sec = Decimal(str(episode_summary["total_duration_sec"]))
            api_avg_tasks = Decimal(str(episode_summary["avg_tasks"]))
            api_avg_duration_sec = Decimal(str(episode_summary["avg_duration_sec"]))
        except (KeyError, TypeError, ValueError, InvalidOperation) as exc:
            pytest.fail(f"data.summary.episode 存在无效统计值：{exc}")

        expected_episode_count = int(self.episode_count_recalculated)
        expected_duration_sec = self.episode_duration_recalculated_sec
        if api_episode_count != expected_episode_count:
            pytest.fail(
                f"Episode 总个数不一致：接口={api_episode_count}，"
                f"步骤4重算={expected_episode_count}"
            )
        if api_total_duration_sec != expected_duration_sec:
            pytest.fail(
                f"Episode 总时长不一致：接口={api_total_duration_sec}，"
                f"步骤5重算={expected_duration_sec}"
            )

        people_counts = [
            int(self.workforce_people.get(key, 0))
            for key in ("collection", "annotation", "qc", "algorithm")
        ]
        max_people = max(people_counts, default=0)
        if max_people <= 0:
            pytest.fail(f"步骤6参与人员数量无有效最大值：{people_counts}")
        expected_avg_tasks = (
                Decimal(expected_episode_count) / Decimal(max_people)
        ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        expected_avg_duration = (
                expected_duration_sec / Decimal(max_people)
        ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        actual_avg_tasks = api_avg_tasks.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        actual_avg_duration = api_avg_duration_sec.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        if actual_avg_tasks != expected_avg_tasks:
            pytest.fail(
                f"Episode 平均个数不一致：接口={actual_avg_tasks}，"
                f"总个数/{max_people}={expected_avg_tasks}"
            )
        if actual_avg_duration != expected_avg_duration:
            pytest.fail(
                f"Episode 平均时长不一致：接口={actual_avg_duration}，"
                f"总时长/{max_people}={expected_avg_duration}"
            )

        print(
            f"[步骤9] Episode 统计校验通过：总个数={api_episode_count}，"
            f"总时长={api_total_duration_sec} 秒，最大参与人员数={max_people}，"
            f"平均个数={actual_avg_tasks}，平均时长={actual_avg_duration} 秒",
            flush=True,
        )
        allure.attach(
            json.dumps(
                {
                    "api": {
                        "task_count": api_episode_count,
                        "total_duration_sec": str(api_total_duration_sec),
                        "avg_tasks": str(api_avg_tasks),
                        "avg_duration_sec": str(api_avg_duration_sec),
                    },
                    "expected": {
                        "task_count": expected_episode_count,
                        "duration_source": "步骤5重算",
                        "max_people": max_people,
                        "avg_tasks": str(expected_avg_tasks),
                        "avg_duration_sec": str(expected_avg_duration),
                    },
                },
                ensure_ascii=False,
                indent=2,
            ),
            name="Episode 人效统计对比",
            attachment_type=allure.attachment_type.JSON,
        )

    @pytest.mark.order(21)
    @allure.story("步骤10：人效统计-验证列表页人员各项统计数据")
    def test_workforce_rows_totals(self):
        """汇总人效列表 rows，核对页面总时长、Episode 数量及转换时长。"""
        print("[步骤10] 开始汇总人效列表页人员数据...", flush=True)
        required_totals = {
            "collection_duration_recalculated_sec": self.collection_duration_recalculated_sec,
            "annotation_duration_recalculated_sec": self.annotation_duration_recalculated_sec,
            "episode_count_recalculated": self.episode_count_recalculated,
            "episode_duration_recalculated_sec": self.episode_duration_recalculated_sec,
        }
        missing_totals = [name for name, value in required_totals.items() if value is None]
        if missing_totals:
            pytest.fail(f"步骤2-5未获取完整统计基准：{', '.join(missing_totals)}")

        response = self.api_all.query_workforce_stats(period="all")
        assertions.assert_code(response.status_code, 200)
        response_data = response.json()
        assertions.assert_text(response_data.get("msg", ""), "success")
        rows = response_data.get("data", {}).get("rows")
        if not isinstance(rows, list):
            pytest.fail("工作台人效统计响应缺少有效 data.rows 列表")

        field_names = (
            "collect_duration_sec",
            "annotation_duration_sec",
            "episode_count",
            "episode_duration_sec",
            "conversion_duration_sec",
        )
        row_totals = {field: Decimal("0") for field in field_names}
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                pytest.fail(f"data.rows[{index}] 不是对象")
            for field in field_names:
                value = row.get(field, 0)
                try:
                    row_totals[field] += Decimal(str(value or 0))
                except (TypeError, ValueError, InvalidOperation) as exc:
                    pytest.fail(f"data.rows[{index}].{field} 不是有效数字：{value!r}，{exc}")

        expected_totals = {
            "collect_duration_sec": Decimal(str(self.collection_duration_recalculated_sec)),
            "annotation_duration_sec": Decimal(str(self.annotation_duration_recalculated_sec)),
            "episode_count": Decimal(str(self.episode_count_recalculated)),
            "episode_duration_sec": Decimal(str(self.episode_duration_recalculated_sec)),
            "conversion_duration_sec": Decimal(str(self.episode_duration_recalculated_sec)),
        }
        for field, expected in expected_totals.items():
            actual = row_totals[field]
            tolerance = Decimal("0.01") if field != "episode_count" else Decimal("0")
            if abs(actual - expected) > tolerance:
                pytest.fail(
                    f"人效列表汇总不一致：{field}，接口 rows 汇总={actual}，"
                    f"步骤基准={expected}，误差={abs(actual - expected)}"
                )

        print(
            f"[步骤10] rows 汇总校验通过：采集={row_totals['collect_duration_sec']} 秒，"
            f"标注={row_totals['annotation_duration_sec']} 秒，"
            f"Episode={row_totals['episode_count']} 个，"
            f"Episode时长={row_totals['episode_duration_sec']} 秒，"
            f"转换时长={row_totals['conversion_duration_sec']} 秒",
            flush=True,
        )
        allure.attach(
            json.dumps(
                {
                    "row_count": len(rows),
                    "rows_totals": {key: str(value) for key, value in row_totals.items()},
                    "expected_totals": {key: str(value) for key, value in expected_totals.items()},
                },
                ensure_ascii=False,
                indent=2,
            ),
            name="人效列表人员数据汇总对比",
            attachment_type=allure.attachment_type.JSON,
        )

    @pytest.mark.order(22)
    @allure.story("步骤11：人效统计-验证人员完成任务数/完成率")
    def test_workforce_completion_statistics(self):
        """逐人员核对人效列表的完成任务数和完成率。"""
        print("[步骤11] 开始核对人员完成任务数和完成率...", flush=True)
        response = self.api_all.query_workforce_stats(period="all")
        assertions.assert_code(response.status_code, 200)
        response_data = response.json()
        assertions.assert_text(response_data.get("msg", ""), "success")
        api_rows = response_data.get("data", {}).get("rows")
        if not isinstance(api_rows, list):
            pytest.fail("工作台人效统计响应缺少有效 data.rows 列表")

        db_rows = self._workforce_completion_rows()
        db_by_user_id = {
            str(row.get("user_id")): row
            for row in db_rows
            if row.get("user_id") is not None
        }
        comparisons = []
        errors = []
        for index, api_row in enumerate(api_rows):
            if not isinstance(api_row, dict):
                errors.append(f"data.rows[{index}] 不是对象")
                continue
            user_id = api_row.get("user_id")
            if user_id is None or str(user_id).strip() == "":
                errors.append(f"data.rows[{index}] 缺少有效 user_id")
                continue
            user_id_key = str(user_id).strip()
            full_name = str(api_row.get("name") or "").strip()
            role_code = str(api_row.get("role_code") or "").strip()
            db_row = db_by_user_id.get(user_id_key)
            if db_row is None:
                errors.append(
                    f"数据库未找到接口人员：user_id={user_id_key}，name={full_name!r}"
                )
                continue
            try:
                api_completed = int(api_row.get("completed_tasks", 0) or 0)
                api_rate = Decimal(str(api_row.get("completion_rate", 0) or 0)).quantize(
                    Decimal("0.01"), rounding=ROUND_HALF_UP
                )
                db_completed = int(db_row["completed_task_count"] or 0)
                db_rate = Decimal(str(db_row["completion_rate"] or 0)).quantize(
                    Decimal("0.01"), rounding=ROUND_HALF_UP
                )
            except (TypeError, ValueError, InvalidOperation, KeyError) as exc:
                errors.append(f"人员 {full_name!r} 完成统计存在无效值：{exc}")
                continue
            if api_completed != db_completed or api_rate != db_rate:
                errors.append(
                    f"人员完成统计不一致：user_id={user_id_key}，name={full_name}，role_code={role_code}，"
                    f"完成数接口={api_completed}、数据库={db_completed}，"
                    f"完成率接口={api_rate}、数据库={db_rate}"
                )
            comparisons.append(
                {
                    "user_id": user_id_key,
                    "name": full_name,
                    "role_code": role_code,
                    "completed_tasks": api_completed,
                    "completion_rate": str(api_rate),
                    "matched": not (api_completed != db_completed or api_rate != db_rate),
                }
            )

        allure.attach(
            json.dumps(
                {"comparisons": comparisons, "errors": errors},
                ensure_ascii=False,
                indent=2,
            ),
            name="人员完成任务数和完成率对比",
            attachment_type=allure.attachment_type.JSON,
        )
        if errors:
            print(
                f"[步骤11] 人员完成任务数/完成率校验发现 {len(errors)} 条异常：",
                flush=True,
            )
            for error in errors:
                print(f"  - {error}", flush=True)
            pytest.fail(
                f"人员完成任务数/完成率校验失败，共 {len(errors)} 条异常：\n"
                + "\n".join(f"- {error}" for error in errors)
            )
        print(f"[步骤11] 人员完成任务数/完成率校验通过：共 {len(comparisons)} 人", flush=True)

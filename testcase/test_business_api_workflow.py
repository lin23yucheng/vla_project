"""业务接口自动化全流程。"""

import os
import json
import configparser
import threading
import time
from datetime import datetime
from pathlib import Path

import allure
import pytest

from api import all_api
from common import Assert
from common.client_factory import create_lazy_yixiu_client
from common.s3_mcap import S3McapConfig, S3McapStore


assertions = Assert.Assertions()
global_client = create_lazy_yixiu_client()
# 填写已有任务编号时，跳过创建、上传和采集完成步骤，直接从步骤 8 开始。
# 为空字符串时执行完整流程；也可通过 VLA_TASK_NO 环境变量覆盖。
TASK_NO = "TASK-2026-012"
TASK_NO = os.environ.get("VLA_TASK_NO", TASK_NO).strip()
UPLOAD_POLL_INTERVAL_SECONDS = 10
UPLOAD_POLL_TIMEOUT_SECONDS = 30 * 60
LOCAL_UPLOAD_PROGRESS_POLL_SECONDS = 30
AUTO_LABELING_POLL_INTERVAL_SECONDS = 10
AUTO_LABELING_POLL_TIMEOUT_SECONDS = 60 * 60


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


# @allure.feature("业务接口自动化全流程")
# class TestBusinessApiWorkflow:
#     @classmethod
#     def setup_class(cls):
#         cls.api_all = all_api.ApiAll(global_client)
#         cls.task_name = None
#         cls.category_name = None
#         cls.category_label = None
#         cls.category_id = None
#         cls.scene_tags = None
#         cls.robot_config_name = None
#         cls.robot_config_id = None
#         cls.task_no = None
#         cls.task_id = None
#         cls.local_collection_path = None
#         cls.upload_files = []
#         cls.auto_labeling_job_id = None
#         cls.auto_labeling_result = None
#
#     @pytest.mark.order(1)
#     @allure.story("步骤1：创建任务")
#     def test_create_task(self):
#         if TASK_NO:
#             pytest.skip(f"已指定任务 {TASK_NO}，跳过创建任务")
#         with allure.step("获取创建任务所需的分类、标签和机器人构型"):
#             categories = response_list(
#                 self.api_all.query_data_categories(), "获取数据分类接口"
#             )
#             if not categories:
#                 pytest.fail("数据分类列表为空，无法创建任务")
#             category = categories[-1]
#             self.__class__.category_name = category.get("name")
#             self.__class__.category_label = category.get("label")
#             self.__class__.category_id = category.get("id")
#             if not self.category_name:
#                 pytest.fail("最后一条数据分类未包含有效 name")
#
#             scene_tag_list = response_list(
#                 self.api_all.query_scene_tags(), "获取标签接口"
#             )
#             if len(scene_tag_list) < 2:
#                 pytest.fail("标签列表少于两条，无法创建任务")
#             last_two_scene_tags = scene_tag_list[-2:]
#             last_two_names = [str(tag.get("name", "")).strip() for tag in last_two_scene_tags]
#             if any(not name for name in last_two_names):
#                 pytest.fail("最后两条标签中存在空 name")
#             self.__class__.scene_tags = list(reversed(last_two_names))
#
#             robot_configs = response_list(
#                 self.api_all.query_robot_configs(), "获取机器人构型接口"
#             )
#             selected_robot_config = select_robot_config(robot_configs)
#             self.__class__.robot_config_name = selected_robot_config.get("name")
#             self.__class__.robot_config_id = selected_robot_config.get("id")
#             if not self.robot_config_id:
#                 pytest.fail(f"机器人构型 {self.robot_config_name!r} 未包含有效 id")
#
#         self.__class__.task_name = (
#             f"{self.robot_config_name}-业务接口自动化-"
#             f"{datetime.now().strftime('%Y%m%d-%H%M%S')}"
#         )
#         with allure.step("创建任务"):
#             response = self.api_all.create_task(
#                 name=self.task_name,
#                 category=self.category_name,
#                 scene_tags=self.scene_tags,
#                 robot_config_id=self.robot_config_id,
#             )
#             assertions.assert_code(response.status_code, 201)
#             assertions.assert_text(response.json().get("msg", ""), "success")
#
#             allure.attach(
#                 "\n".join(
#                     [
#                         f"task_name: {self.task_name}",
#                         f"category_name: {self.category_name}",
#                         f"category_label: {self.category_label}",
#                         f"category_id: {self.category_id}",
#                         f"scene_tags: {self.scene_tags}",
#                         f"robot_config_name: {self.robot_config_name}",
#                         f"robot_config_id: {self.robot_config_id}",
#                     ]
#                 ),
#                 name="创建任务参数",
#                 attachment_type=allure.attachment_type.TEXT,
#             )
#
#     @pytest.mark.order(2)
#     @allure.story("步骤2：查询任务管理列表")
#     def test_query_created_task(self):
#         if TASK_NO:
#             task_list = response_list(
#                 self.api_all.query_task_list(), "查询任务管理列表接口"
#             )
#             target_task = next(
#                 (task for task in task_list if str(task.get("task_no", "")).strip() == TASK_NO),
#                 None,
#             )
#             if target_task is None:
#                 pytest.fail(f"任务管理列表未找到任务编号：{TASK_NO}")
#             self.__class__.task_no = target_task.get("task_no")
#             self.__class__.task_id = target_task.get("id")
#             self.__class__.task_name = target_task.get("name")
#             self.__class__.robot_config_id = (
#                 target_task.get("robot_config_id")
#                 or target_task.get("robotConfigId")
#                 or (target_task.get("robot_config") or {}).get("id")
#             )
#             self.__class__.robot_config_name = (
#                 target_task.get("robot_config_name")
#                 or target_task.get("robotConfigName")
#                 or (target_task.get("robot_config") or {}).get("name")
#             )
#             if not self.task_id or not self.task_name:
#                 pytest.fail(f"任务 {TASK_NO} 未包含有效 id 或 name")
#             if not self.robot_config_id and self.robot_config_name:
#                 configs = response_list(self.api_all.query_robot_configs(), "获取机器人构型接口")
#                 matched = next((item for item in configs if item.get("name") == self.robot_config_name), None)
#                 self.__class__.robot_config_id = matched.get("id") if matched else None
#             if not self.robot_config_id:
#                 pytest.fail(f"任务 {TASK_NO} 未包含有效 robot_config_id")
#             return
#         if not self.task_name:
#             pytest.fail("步骤1未生成任务名称，无法查询创建结果")
#
#         with allure.step("按任务名称查询任务管理列表"):
#             task_list = response_list(
#                 self.api_all.query_task_list(), "查询任务管理列表接口"
#             )
#             target_task = next(
#                 (
#                     task
#                     for task in task_list
#                     if str(task.get("name", "")).strip() == self.task_name
#                 ),
#                 None,
#             )
#             if target_task is None:
#                 pytest.fail(f"任务管理列表未找到刚创建的任务：{self.task_name}")
#
#             self.__class__.task_no = target_task.get("task_no")
#             self.__class__.task_id = target_task.get("id")
#             if not self.task_no or not self.task_id:
#                 pytest.fail(f"任务 {self.task_name} 未包含有效 task_no 或 id")
#
#             allure.attach(
#                 f"task_name: {self.task_name}\n"
#                 f"task_no: {self.task_no}\n"
#                 f"id: {self.task_id}",
#                 name="创建任务查询结果",
#                 attachment_type=allure.attachment_type.TEXT,
#             )
#
#     @pytest.mark.order(3)
#     @allure.story("步骤3：查询数据采集列表")
#     def test_query_created_task_in_data_collection_list(self):
#         if TASK_NO:
#             pytest.skip(f"已指定任务 {TASK_NO}，跳过数据采集列表校验")
#         if not self.task_id or not self.task_no or not self.task_name:
#             pytest.fail("步骤2未获取完整任务信息，无法查询数据采集列表")
#
#         with allure.step("按任务 ID 查询数据采集列表并核对任务信息"):
#             task_list = response_list(
#                 self.api_all.query_data_collection_list(), "查询数据采集列表接口"
#             )
#             target_task = next(
#                 (
#                     task
#                     for task in task_list
#                     if str(task.get("id", "")).strip() == str(self.task_id).strip()
#                 ),
#                 None,
#             )
#             if target_task is None:
#                 pytest.fail(f"数据采集列表未找到刚创建的任务：id={self.task_id}")
#
#             actual_task_id = target_task.get("id")
#             actual_task_name = target_task.get("name")
#             actual_task_no = target_task.get("task_no")
#             assertions.assert_text(str(actual_task_id), str(self.task_id))
#             assertions.assert_text(str(actual_task_name), str(self.task_name))
#             assertions.assert_text(str(actual_task_no), str(self.task_no))
#
#             allure.attach(
#                 f"id: {actual_task_id}\n"
#                 f"name: {actual_task_name}\n"
#                 f"task_no: {actual_task_no}",
#                 name="数据采集列表任务核对结果",
#                 attachment_type=allure.attachment_type.TEXT,
#             )
#
#     @pytest.mark.order(4)
#     @allure.story("步骤4：采集文件上传")
#     def test_upload_collection_files(self):
#         if TASK_NO:
#             pytest.skip(f"已指定任务 {TASK_NO}，跳过文件上传")
#         if not self.task_id:
#             pytest.fail("步骤2未获取 task_id，无法上传采集文件")
#
#         root_path, local_files = collect_local_files()
#         upload_files = [
#             {
#                 "local_path": local_path,
#                 "filename": local_path.name,
#                 "relative_path": local_path.relative_to(root_path).as_posix(),
#                 "size_bytes": local_path.stat().st_size,
#             }
#             for local_path in local_files
#         ]
#         registration_files = [
#             {
#                 "filename": file_info["filename"],
#                 "relative_path": file_info["relative_path"],
#                 "size_bytes": file_info["size_bytes"],
#             }
#             for file_info in upload_files
#         ]
#
#         with allure.step("登记待上传采集文件"):
#             response = self.api_all.register_collect_files(
#                 task_id=self.task_id, files=registration_files
#             )
#             assertions.assert_code(response.status_code, 200)
#             response_data = response.json()
#             assertions.assert_text(response_data.get("msg", ""), "success")
#             registered_files = response_data.get("data", {}).get("files", [])
#             if len(registered_files) != len(upload_files):
#                 pytest.fail(
#                     f"登记文件数量不一致：本地 {len(upload_files)} 个，"
#                     f"接口返回 {len(registered_files)} 个"
#                 )
#
#             # 某些环境批量登记接口会返回成功但未实际落库，单文件登记可正常落库。
#             summary_response = self.api_all.query_collect_files_summary(self.task_id)
#             assertions.assert_code(summary_response.status_code, 200)
#             summary_data = summary_response.json().get("data", {})
#             if int(summary_data.get("total", 0) or 0) == 0:
#                 registered_files = []
#                 for registration_file in registration_files:
#                     single_response = self.api_all.register_collect_files(
#                         task_id=self.task_id, files=[registration_file]
#                     )
#                     assertions.assert_code(single_response.status_code, 200)
#                     single_data = single_response.json()
#                     assertions.assert_text(single_data.get("msg", ""), "success")
#                     single_files = single_data.get("data", {}).get("files", [])
#                     if len(single_files) != 1:
#                         pytest.fail(
#                             f"单文件登记响应数量异常：{registration_file['relative_path']}，"
#                             f"返回 {len(single_files)} 个"
#                         )
#                     registered_files.extend(single_files)
#
#                 summary_response = self.api_all.query_collect_files_summary(self.task_id)
#                 assertions.assert_code(summary_response.status_code, 200)
#                 summary_data = summary_response.json().get("data", {})
#                 if int(summary_data.get("total", 0) or 0) != len(upload_files):
#                     pytest.fail(
#                         "文件登记后服务端汇总数量仍不正确："
#                         f"期望 {len(upload_files)}，实际 {summary_data.get('total', 0)}"
#                     )
#
#             registered_by_path = {
#                 str(file_info.get("relative_path", "")): file_info
#                 for file_info in registered_files
#             }
#             for file_info in upload_files:
#                 registered_file = registered_by_path.get(file_info["relative_path"])
#                 if not registered_file or not registered_file.get("id"):
#                     pytest.fail(
#                         f"登记响应未返回文件 {file_info['relative_path']} 的 task_file_id"
#                     )
#                 file_info["task_file_id"] = registered_file["id"]
#
#         self.__class__.local_collection_path = root_path
#         self.__class__.upload_files = upload_files
#         allure.attach(
#             "\n".join(
#                 f"{file_info['relative_path']} ({file_info['size_bytes']} bytes)"
#                 for file_info in upload_files
#             ),
#             name="待上传采集文件",
#             attachment_type=allure.attachment_type.TEXT,
#         )
#
#         upload_started_at = time.monotonic()
#         for index, file_info in enumerate(upload_files, start=1):
#             with allure.step(
#                 f"上传文件 {index}/{len(upload_files)}：{file_info['relative_path']}"
#             ):
#                 print(
#                     f"[步骤4] 开始上传 {index}/{len(upload_files)}："
#                     f"{file_info['relative_path']}（{file_info['size_bytes']} bytes）",
#                     flush=True,
#                 )
#                 resume_response = self.api_all.query_collect_file_resume(
#                     task_id=self.task_id,
#                     task_file_id=file_info["task_file_id"],
#                 )
#                 assertions.assert_code(resume_response.status_code, 200)
#                 resume_data = resume_response.json().get("data", {})
#                 offset = int(resume_data.get("uploaded_bytes", 0) or 0)
#                 if offset < 0 or offset > file_info["size_bytes"]:
#                     pytest.fail(
#                         f"文件 {file_info['relative_path']} 的断点位置无效：{offset}"
#                     )
#
#                 upload_result = {}
#
#                 def upload_file():
#                     try:
#                         with file_info["local_path"].open("rb") as file_stream:
#                             file_stream.seek(offset)
#                             upload_result["response"] = self.api_all.upload_collect_file_stream(
#                                 task_id=self.task_id,
#                                 filename=file_info["filename"],
#                                 relative_path=file_info["relative_path"],
#                                 task_file_id=file_info["task_file_id"],
#                                 offset=offset,
#                                 file_stream=file_stream,
#                                 content_length=file_info["size_bytes"] - offset,
#                             )
#                     except Exception as exc:
#                         upload_result["error"] = exc
#
#                 upload_thread = threading.Thread(
#                     target=upload_file,
#                     name=f"upload-{file_info['filename']}",
#                 )
#                 upload_thread.start()
#                 while upload_thread.is_alive():
#                     upload_thread.join(LOCAL_UPLOAD_PROGRESS_POLL_SECONDS)
#                     if upload_thread.is_alive():
#                         print_active_upload_progress(self.api_all, self.task_id)
#
#                 if "error" in upload_result:
#                     raise upload_result["error"]
#                 response = upload_result.get("response")
#                 if response is None:
#                     pytest.fail(f"文件 {file_info['relative_path']} 上传未返回响应")
#                 assertions.assert_code(response.status_code, 201)
#                 response_data = response.json()
#                 assertions.assert_text(response_data.get("msg", ""), "success")
#                 uploaded_file = response_data.get("data", {}).get("file", {})
#                 assertions.assert_text(
#                     uploaded_file.get("upload_status_label", ""), "本地上传成功"
#                 )
#                 print(
#                     f"[步骤4] 本地上传完成 {index}/{len(upload_files)}："
#                     f"{file_info['relative_path']}，状态="
#                     f"{uploaded_file.get('upload_status_label')}，"
#                     f"local_progress={uploaded_file.get('local_progress', 0)}%",
#                     flush=True,
#                 )
#
#         upload_elapsed_seconds = int(time.monotonic() - upload_started_at)
#         upload_minutes, upload_seconds = divmod(upload_elapsed_seconds, 60)
#         print(
#             f"[步骤4] 全部文件上传完成，总耗时："
#             f"{upload_minutes}分{upload_seconds}秒",
#             flush=True,
#         )
#
#     @pytest.mark.order(5)
#     @allure.story("步骤5：轮询采集文件上传结果")
#     def test_wait_for_collection_upload_completion(self):
#         if TASK_NO:
#             pytest.skip(f"已指定任务 {TASK_NO}，跳过上传轮询")
#         if not self.task_id or not self.upload_files:
#             pytest.fail("步骤4未准备上传文件，无法检查上传结果")
#
#         expected_relative_paths = {
#             file_info["relative_path"] for file_info in self.upload_files
#         }
#         started_at = time.monotonic()
#         while True:
#             active_files = query_all_collect_files(self.api_all, self.task_id, "active")
#             uploaded_files = query_all_collect_files(self.api_all, self.task_id, "uploaded")
#             summary_response = self.api_all.query_collect_files_summary(self.task_id)
#             assertions.assert_code(summary_response.status_code, 200)
#             summary_response_data = summary_response.json()
#             assertions.assert_text(summary_response_data.get("msg", ""), "success")
#             summary_data = summary_response_data.get("data", {})
#
#             remaining_count = len(active_files)
#             print(
#                 f"[步骤5] 上传轮询：总文件 {len(self.upload_files)} 个，"
#                 f"已完成 {len(uploaded_files)} 个，剩余 {remaining_count} 个，"
#                 f"整体进度={summary_data.get('overall_progress', 0)}%",
#                 flush=True,
#             )
#             for active_file in active_files:
#                 print(
#                     f"[步骤5] 进行中：{active_file.get('relative_path') or active_file.get('filename')}，"
#                     f"状态={active_file.get('upload_status_label')}，"
#                     f"local_progress={active_file.get('local_progress', 0)}%，"
#                     f"progress={active_file.get('progress', 0)}%",
#                     flush=True,
#                 )
#
#             allure.attach(
#                 f"active_count: {len(active_files)}\n"
#                 f"uploaded_count: {len(uploaded_files)}\n"
#                 f"active_files: {[(file_info.get('relative_path'), file_info.get('upload_status_label')) for file_info in active_files]}\n"
#                 f"uploaded_files: {[(file_info.get('relative_path'), file_info.get('upload_status_label')) for file_info in uploaded_files]}\n"
#                 f"summary: {summary_data}",
#                 name="上传轮询状态",
#                 attachment_type=allure.attachment_type.TEXT,
#             )
#
#             if not active_files:
#                 actual_relative_paths = {
#                     str(file_info.get("relative_path", "")) for file_info in uploaded_files
#                 }
#                 actual_filenames = {
#                     str(file_info.get("filename", "")) for file_info in uploaded_files
#                 }
#                 expected_filenames = {
#                     file_info["filename"] for file_info in self.upload_files
#                 }
#                 assertions.assert_text(
#                     str(len(uploaded_files)), str(len(self.upload_files))
#                 )
#                 assertions.assert_text(
#                     str(sorted(actual_relative_paths)),
#                     str(sorted(expected_relative_paths)),
#                 )
#                 assertions.assert_text(
#                     str(sorted(actual_filenames)), str(sorted(expected_filenames))
#                 )
#                 for file_info in uploaded_files:
#                     assertions.assert_text(
#                         file_info.get("upload_status_label", ""), "上传成功"
#                     )
#                 assertions.assert_text(
#                     str(summary_data.get("success", 0)), str(len(self.upload_files))
#                 )
#                 return
#
#             elapsed_seconds = time.monotonic() - started_at
#             if elapsed_seconds >= UPLOAD_POLL_TIMEOUT_SECONDS:
#                 pytest.fail(
#                     f"等待服务器转存上传完成超时：{UPLOAD_POLL_TIMEOUT_SECONDS} 秒"
#                 )
#             time.sleep(UPLOAD_POLL_INTERVAL_SECONDS)
#
#     @pytest.mark.order(6)
#     @allure.story("步骤6：提交采集完成")
#     def test_complete_collection_files(self):
#         if TASK_NO:
#             pytest.skip(f"已指定任务 {TASK_NO}，跳过提交采集完成")
#         if not self.task_id or not self.upload_files:
#             pytest.fail("步骤4未准备上传文件，无法提交采集完成")
#
#         with allure.step("核对本地文件与服务端已上传文件数量"):
#             uploaded_files = query_all_collect_files(
#                 self.api_all, self.task_id, "uploaded"
#             )
#             active_files = query_all_collect_files(
#                 self.api_all, self.task_id, "active"
#             )
#             expected_count = len(self.upload_files)
#             if active_files:
#                 pytest.fail(
#                     f"仍有文件处于上传中，无法提交采集完成：{len(active_files)} 个"
#                 )
#             assertions.assert_text(str(len(uploaded_files)), str(expected_count))
#             for uploaded_file in uploaded_files:
#                 assertions.assert_text(
#                     uploaded_file.get("upload_status_label", ""), "上传成功"
#                 )
#
#         with allure.step("调用采集完成接口"):
#             response = self.api_all.complete_collect_files(self.task_id)
#             assertions.assert_code(response.status_code, 200)
#             assertions.assert_text(response.json().get("msg", ""), "success")
#
#     @pytest.mark.order(7)
#     @allure.story("步骤7：验证任务进入标注列表")
#     def test_verify_collection_completed_task_in_annotation_list(self):
#         if TASK_NO:
#             pytest.skip(f"已指定任务 {TASK_NO}，跳过采集完成后的列表校验")
#         if not self.task_id or not self.task_name:
#             pytest.fail("步骤2未获取完整任务信息，无法验证采集完成结果")
#
#         with allure.step("采集列表中不再显示该任务"):
#             collection_tasks = response_list(
#                 self.api_all.query_data_collection_list(), "查询数据采集列表接口"
#             )
#             collection_match = next(
#                 (
#                     task
#                     for task in collection_tasks
#                     if str(task.get("id", "")).strip()
#                     == str(self.task_id).strip()
#                 ),
#                 None,
#             )
#             if collection_match is not None:
#                 pytest.fail(
#                     f"采集完成任务仍存在于数据采集列表：id={self.task_id}"
#                 )
#
#         with allure.step("标注列表中显示该任务"):
#             annotation_tasks = response_list(
#                 self.api_all.query_annotation_list(), "查询标注列表接口"
#             )
#             annotation_match = next(
#                 (
#                     task
#                     for task in annotation_tasks
#                     if str(task.get("id", "")).strip()
#                     == str(self.task_id).strip()
#                     or str(task.get("name", "")).strip() == self.task_name
#                 ),
#                 None,
#             )
#             if annotation_match is None:
#                 pytest.fail(
#                     f"采集完成任务未出现在标注列表：id={self.task_id}，name={self.task_name}"
#                 )
#
#     @pytest.mark.order(8)
#     @allure.story("步骤8：参数预检")
#     def test_auto_labeling_pre_check(self):
#         if not self.task_id or not self.robot_config_id:
#             pytest.fail("缺少 task_id 或 robot_config_id，无法进行参数预检")
#
#         config_response = self.api_all.query_robot_config_detail(self.robot_config_id)
#         assertions.assert_code(config_response.status_code, 200)
#         config_data = config_response.json().get("data", {})
#         robot_config = config_data.get("config_json")
#         if not isinstance(robot_config, dict):
#             pytest.fail("机器人构型详情缺少有效 config_json")
#
#         uploaded_files = query_all_collect_files(self.api_all, self.task_id, "uploaded")
#         file_paths = [item.get("s3_uri") for item in uploaded_files if item.get("s3_uri")]
#         if not file_paths:
#             pytest.fail("没有可用于自动化标注的 S3 文件路径")
#
#         playlist_response = self.api_all.query_task_video_playlist(self.task_id)
#         assertions.assert_code(playlist_response.status_code, 200)
#         playlist_data = playlist_response.json().get("data")
#         if not isinstance(playlist_data, dict):
#             pytest.fail("视频播放清单响应缺少有效 data")
#
#         pre_check_job_id = os.environ.get("VLA_AUTO_LABELING_JOB_ID", "").strip()
#         if not pre_check_job_id:
#             # 预检接口要求 job_id 字符串，但尚未进入正式任务创建阶段；使用本次请求的数字追踪 ID。
#             pre_check_job_id = str(time.time_ns())[-16:]
#         payload = {
#             "job_id": pre_check_job_id,
#             "basic": {"task_id": str(self.task_id), "user_config_path": None},
#             "input": {
#                 "file_path": file_paths,
#                 "file_type": "mcap",
#                 "video_playlist": json.dumps(playlist_data, ensure_ascii=False, separators=(",", ":")),
#                 "robot_config": json.dumps(robot_config, ensure_ascii=False, separators=(",", ":")),
#                 "user_prompt": os.environ.get(
#                     "VLA_AUTO_LABELING_PROMPT",
#                     "请描述机器人完成的子任务,可选子任务为：夹取物体，移动物体，放下物体",
#                 ),
#                 "video_fps": int(os.environ.get("VLA_AUTO_LABELING_FPS", "30")),
#             },
#         }
#         response = self.api_all.auto_labeling_pre_check(payload)
#         if response.status_code != 200:
#             pytest.fail(
#                 f"参数预检失败：HTTP {response.status_code}，响应：{response.text}"
#             )
#         assertions.assert_code(response.status_code, 200)
#         assertions.assert_text(response.json().get("msg", ""), "ok")
#
#     @pytest.mark.order(9)
#     @allure.story("步骤9：提交自动化标注任务")
#     def test_submit_auto_labeling(self):
#         if not self.task_id or not self.robot_config_id:
#             pytest.fail("缺少任务或机器人构型信息，无法提交自动化标注")
#
#         job_response = self.api_all.create_auto_labeling_job_id()
#         assertions.assert_code(job_response.status_code, 200)
#         job_id = job_response.json().get("data", {}).get("job_id")
#         if not job_id:
#             pytest.fail("生成 job_id 失败")
#
#         config_response = self.api_all.query_robot_config_detail(self.robot_config_id)
#         assertions.assert_code(config_response.status_code, 200)
#         robot_config = config_response.json().get("data", {}).get("config_json")
#         files = query_all_collect_files(self.api_all, self.task_id, "uploaded")
#         playlist_response = self.api_all.query_task_video_playlist(self.task_id)
#         assertions.assert_code(playlist_response.status_code, 200)
#         playlist = playlist_response.json().get("data")
#         if not isinstance(robot_config, dict) or not isinstance(playlist, dict):
#             pytest.fail("自动化标注提交所需配置不完整")
#
#         payload = {
#             "job_id": str(job_id),
#             "basic": {"task_id": str(self.task_id), "user_config_path": None},
#             "input": {
#                 "file_path": [item["s3_uri"] for item in files if item.get("s3_uri")],
#                 "file_type": "mcap",
#                 "video_playlist": json.dumps(playlist, ensure_ascii=False, separators=(",", ":")),
#                 "robot_config": json.dumps(robot_config, ensure_ascii=False, separators=(",", ":")),
#                 "user_prompt": os.environ.get(
#                     "VLA_AUTO_LABELING_PROMPT",
#                     "请描述机器人完成的子任务,可选子任务为：夹取物体，移动物体，放下物体",
#                 ),
#                 "video_fps": int(os.environ.get("VLA_AUTO_LABELING_FPS", "30")),
#             },
#         }
#         response = self.api_all.submit_auto_labeling_task(payload)
#         if response.status_code not in {200, 202}:
#             pytest.fail(f"提交自动化标注失败：HTTP {response.status_code}，响应：{response.text}")
#         submitted = response.json().get("data", {})
#         self.__class__.auto_labeling_job_id = submitted.get("job_id") or job_id
#         if submitted.get("status") not in {"QUEUED", "RUNNING"}:
#             pytest.fail(f"自动化标注提交后状态异常：{submitted}")
#
#     @pytest.mark.order(10)
#     @allure.story("步骤10：轮询自动化标注状态")
#     def test_wait_for_auto_labeling(self):
#         if not self.task_id or not self.auto_labeling_job_id:
#             pytest.fail("缺少自动化标注 job_id")
#         started_at = time.monotonic()
#         while True:
#             response = self.api_all.query_auto_labeling_task(self.task_id, self.auto_labeling_job_id)
#             assertions.assert_code(response.status_code, 200)
#             data = response.json().get("data", {})
#             status = data.get("status")
#             status_code = data.get("status_code")
#             print(
#                 f"[步骤10] 自动化标注轮询：status={status}，"
#                 f"status_code={status_code}，"
#                 f"progress={data.get('progress_percent', 0)}%，"
#                 f"phase={data.get('progress_phase')}，"
#                 f"message={data.get('progress_message')}",
#                 flush=True,
#             )
#             if status == "SUCCEEDED" or status_code == 4:
#                 return
#             if status in {"FAILED", "CANCELED", "CANCELLED", "TIMEOUT"} or status_code in {5, 6, 7, 8}:
#                 pytest.fail(f"自动化标注执行失败：{data}")
#             if time.monotonic() - started_at >= AUTO_LABELING_POLL_TIMEOUT_SECONDS:
#                 pytest.fail(f"自动化标注轮询超时：{AUTO_LABELING_POLL_TIMEOUT_SECONDS} 秒")
#             time.sleep(AUTO_LABELING_POLL_INTERVAL_SECONDS)
#
#     @pytest.mark.order(11)
#     @allure.story("步骤11：校验自动化标注结果")
#     def test_verify_auto_labeling_result(self):
#         if not self.task_id:
#             pytest.fail("缺少 task_id，无法查询自动化标注结果")
#         response = self.api_all.query_latest_auto_labeling_job(self.task_id)
#         assertions.assert_code(response.status_code, 200)
#         data = response.json().get("data", {})
#         if data.get("status") != "SUCCEEDED" or data.get("status_code") != 4:
#             pytest.fail(f"自动化标注最终状态异常：{data}")
#         episodes = data.get("result", {}).get("episodes_annotation", [])
#         if not isinstance(episodes, list) or not episodes:
#             pytest.fail("自动化标注成功，但 episodes_annotation 为空")
#         self.__class__.auto_labeling_result = data
#
#         workspace_response = self.api_all.query_annotation_workspace(self.task_id)
#         assertions.assert_code(workspace_response.status_code, 200)
#         assertions.assert_text(workspace_response.json().get("msg", ""), "success")


@allure.feature("工作台数据校验")
class TestWorkbenchData:
    """校验工作台任务概览与 PostgreSQL tasks 表统计一致。"""

    @classmethod
    def setup_class(cls):
        cls.api_all = all_api.ApiAll(global_client)
        config_path = Path(__file__).resolve().parent.parent / "config" / "env_config.ini"
        cls.config = configparser.ConfigParser()
        if not cls.config.read(config_path, encoding="utf-8"):
            pytest.fail(f"无法读取配置文件：{config_path}")
        if not cls.config.has_section("fat-vla"):
            pytest.fail("配置文件缺少 [fat-vla] 节")

    def _database_counts(self):
        """查询 tasks 总数及各状态数量，连接在查询完成后关闭。"""
        try:
            import psycopg2
        except ImportError as exc:
            pytest.fail("缺少 PostgreSQL 驱动，请先安装 requirements.txt 中的 psycopg2-binary")

        section = self.config["fat-vla"]
        connection = None
        try:
            connection = psycopg2.connect(
                host=section.get("db_host"),
                port=section.getint("db_port", fallback=5432),
                user=section.get("db_user"),
                password=section.get("db_password"),
                dbname=section.get("db_name"),
                connect_timeout=30,
            )
            with connection.cursor() as cursor:
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
        finally:
            if connection is not None:
                connection.close()

    def _collected_task_nos(self):
        """返回已采集、已标注、已质检或已转换任务的 task_no。"""
        try:
            import psycopg2
        except ImportError as exc:
            pytest.fail("缺少 PostgreSQL 驱动，请先安装 requirements.txt 中的 psycopg2-binary")

        section = self.config["fat-vla"]
        connection = None
        try:
            connection = psycopg2.connect(
                host=section.get("db_host"),
                port=section.getint("db_port", fallback=5432),
                user=section.get("db_user"),
                password=section.get("db_password"),
                dbname=section.get("db_name"),
                connect_timeout=30,
            )
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT task_no FROM tasks WHERE status IN (2, 3, 4, 5) "
                    "AND task_no IS NOT NULL AND task_no <> '' ORDER BY task_no"
                )
                return [str(row[0]).strip() for row in cursor.fetchall() if str(row[0]).strip()]
        except Exception as exc:
            pytest.fail(f"查询已采集任务 task_no 失败：{exc}")
        finally:
            if connection is not None:
                connection.close()

    def _task_s3_config(self, task_no):
        return S3McapConfig(
            endpoint_url=self.config.get("fat-vla", "s3_endpoint"),
            access_key=self.config.get("fat-vla", "s3_access_key"),
            secret_key=self.config.get("fat-vla", "s3_secret_key"),
            bucket=self.config.get("fat-vla", "s3_bucket", fallback="vla-label-studio"),
            prefix=f"{task_no}/original-data",
            region=self.config.get("fat-vla", "s3_region", fallback="").strip() or None,
            read_ahead_bytes=self.config.getint("fat-vla", "s3_read_ahead_bytes", fallback=1024 * 1024),
            max_range_request_bytes=self.config.getint(
                "fat-vla", "s3_max_range_request_bytes", fallback=1024 * 1024 * 1024
            ),
        )

    def _annotation_duration_rows(self, task_nos):
        """查询指定任务的全部标注片段起止纳秒数。"""
        if not task_nos:
            return []
        try:
            import psycopg2
        except ImportError as exc:
            pytest.fail("缺少 PostgreSQL 驱动，请先安装 requirements.txt 中的 psycopg2-binary")

        section = self.config["fat-vla"]
        connection = None
        try:
            connection = psycopg2.connect(
                host=section.get("db_host"),
                port=section.getint("db_port", fallback=5432),
                user=section.get("db_user"),
                password=section.get("db_password"),
                dbname=section.get("db_name"),
                connect_timeout=30,
            )
            with connection.cursor() as cursor:
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
        finally:
            if connection is not None:
                connection.close()

    def _episode_count(self, task_nos):
        """统计指定任务下 segment_kind=episode 的标注数量。"""
        if not task_nos:
            return 0
        try:
            import psycopg2
        except ImportError as exc:
            pytest.fail("缺少 PostgreSQL 驱动，请先安装 requirements.txt 中的 psycopg2-binary")
        section = self.config["fat-vla"]
        connection = None
        try:
            connection = psycopg2.connect(
                host=section.get("db_host"), port=section.getint("db_port", fallback=5432),
                user=section.get("db_user"), password=section.get("db_password"),
                dbname=section.get("db_name"), connect_timeout=30,
            )
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT COUNT(*) FROM annotation_segment_items "
                    "WHERE task_no = ANY(%s) AND segment_kind = %s "
                    "AND deleted_flag = FALSE",
                    (task_nos, "episode"),
                )
                return int(cursor.fetchone()[0])
        except Exception as exc:
            pytest.fail(f"查询 episode 标注数量失败：{exc}")
        finally:
            if connection is not None:
                connection.close()

    def _episode_duration_sec(self, task_nos):
        """累加所有 Episode 纳秒时长，最后统一换算秒并四舍五入。"""
        if not task_nos:
            return 0
        try:
            import psycopg2
        except ImportError as exc:
            pytest.fail("缺少 PostgreSQL 驱动，请先安装 requirements.txt 中的 psycopg2-binary")
        section = self.config["fat-vla"]
        connection = None
        try:
            connection = psycopg2.connect(
                host=section.get("db_host"), port=section.getint("db_port", fallback=5432),
                user=section.get("db_user"), password=section.get("db_password"),
                dbname=section.get("db_name"), connect_timeout=30,
            )
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT task_no, start_time_ns, end_time_ns "
                    "FROM annotation_segment_items "
                    "WHERE task_no = ANY(%s) AND segment_kind = %s "
                    "AND deleted_flag = FALSE",
                    (task_nos, "episode"),
                )
                total_ns = 0
                for task_no, start_time_ns, end_time_ns in cursor.fetchall():
                    try:
                        start = int(start_time_ns)
                        end = int(end_time_ns)
                    except (TypeError, ValueError) as exc:
                        pytest.fail(f"任务 {task_no} Episode 时间无效：{exc}")
                    duration_ns = end - start
                    if duration_ns < 0:
                        pytest.fail(
                            f"任务 {task_no} Episode 时间范围无效："
                            f"start={start}，end={end}"
                        )
                    total_ns += duration_ns
                # 所有 Episode 计算完成后，再统一换算为秒并四舍五入。
                return int(total_ns / 1_000_000_000 + 0.5)
        except Exception as exc:
            pytest.fail(f"查询 Episode 标注时长失败：{exc}")
        finally:
            if connection is not None:
                connection.close()

    @pytest.mark.order(12)
    @allure.story("步骤1：验证任务状态数据")
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
    @allure.story("步骤2：验证总采集时长")
    def test_collection_duration_statistics(self):
        """按 tasks 状态和 S3 MCAP 时间范围重算总采集时长。"""
        print("[步骤2] 开始调用采集时长统计接口...", flush=True)
        response = self.api_all.query_duration_stats(period="all")
        assertions.assert_code(response.status_code, 200)
        response_data = response.json()
        assertions.assert_text(response_data.get("msg", ""), "success")
        trend = response_data.get("data", {}).get("duration_trend")
        if not isinstance(trend, list) or not trend:
            pytest.fail("工作台采集时长响应缺少 data.duration_trend[0]")
        try:
            api_collect_sec = float(trend[0]["collect_sec"])
        except (KeyError, TypeError, ValueError) as exc:
            pytest.fail(f"data.duration_trend[0].collect_sec 不是有效数字：{exc}")
        print(f"[步骤2] 接口采集总时长：{api_collect_sec} 秒", flush=True)

        print("[步骤2] 正在查询符合条件的任务（状态 2/3/4/5）...", flush=True)
        task_nos = self._collected_task_nos()
        print(f"[步骤2] 共找到 {len(task_nos)} 个任务，开始读取 S3 MCAP...", flush=True)
        total_duration_ns = 0
        task_details = []
        for task_index, task_no in enumerate(task_nos, start=1):
            print(f"[步骤2] ({task_index}/{len(task_nos)}) 开始处理任务 {task_no}...", flush=True)
            store = S3McapStore(self._task_s3_config(task_no))

            def report_mcap_progress(stage, source, mcap_index):
                if stage == "start":
                    print(f"[步骤2]   MCAP {mcap_index}: 开始解析 {source.object_name}", flush=True)
                else:
                    duration_sec = (int(source.message_end_time_ns) - int(source.message_start_time_ns)) / 1_000_000_000
                    print(f"[步骤2]   MCAP {mcap_index}: 完成，时长 {duration_sec:.6f} 秒", flush=True)

            sources = store.list_indexed_mcap_sources(progress_callback=report_mcap_progress)
            task_duration_ns = 0
            for source in sources:
                if source.message_start_time_ns is None or source.message_end_time_ns is None:
                    pytest.fail(f"MCAP 未解析出时间范围：{source}")
                duration_ns = int(source.message_end_time_ns) - int(source.message_start_time_ns)
                if duration_ns < 0:
                    pytest.fail(f"MCAP 时间范围无效：{source}")
                task_duration_ns += duration_ns
            total_duration_ns += task_duration_ns
            task_details.append({"task_no": task_no, "mcap_count": len(sources), "duration_sec": task_duration_ns / 1_000_000_000})
            print(f"[步骤2] ({task_index}/{len(task_nos)}) 任务 {task_no} 完成，{len(sources)} 个 MCAP，时长 {task_duration_ns / 1_000_000_000:.6f} 秒", flush=True)

        expected_collect_sec = total_duration_ns / 1_000_000_000
        total_mcap_count = sum(detail["mcap_count"] for detail in task_details)
        allowed_error_sec = total_mcap_count * 0.066
        actual_error_sec = abs(api_collect_sec - expected_collect_sec)
        print(
            f"[工作台采集时长] 接口={api_collect_sec} 秒，S3重算={expected_collect_sec} 秒，"
            f"任务数={len(task_nos)}，MCAP数量={total_mcap_count}，"
            f"允许误差={allowed_error_sec:.6f} 秒，实际误差={actual_error_sec:.6f} 秒",
            flush=True,
        )
        if actual_error_sec > allowed_error_sec:
            pytest.fail(
                f"采集总时长不一致：接口={api_collect_sec}，S3重算={expected_collect_sec}，"
                f"实际误差={actual_error_sec:.6f} 秒，允许误差={allowed_error_sec:.6f} 秒"
            )
        allure.attach(
            json.dumps(
                {
                    "api_collect_sec": api_collect_sec,
                    "s3_collect_sec": expected_collect_sec,
                    "mcap_count": total_mcap_count,
                    "allowed_error_sec": allowed_error_sec,
                    "actual_error_sec": actual_error_sec,
                    "tasks": task_details,
                },
                ensure_ascii=False,
                indent=2,
            ),
            name="采集总时长统计对比",
            attachment_type=allure.attachment_type.JSON,
        )

    @pytest.mark.order(14)
    @allure.story("步骤3：验证标注总时长")
    def test_annotation_duration_statistics(self):
        """按已标注及后续状态任务的标注片段重算标注总时长。"""
        print("[步骤3] 开始读取工作台标注时长统计接口...", flush=True)
        response = self.api_all.query_duration_stats(period="all")
        assertions.assert_code(response.status_code, 200)
        response_data = response.json()
        assertions.assert_text(response_data.get("msg", ""), "success")
        trend = response_data.get("data", {}).get("duration_trend")
        if not isinstance(trend, list) or not trend:
            pytest.fail("工作台采集时长响应缺少 data.duration_trend[0]")
        try:
            api_annotation_sec = float(trend[0]["annotation_sec"])
        except (KeyError, TypeError, ValueError) as exc:
            pytest.fail(f"data.duration_trend[0].annotation_sec 不是有效数字：{exc}")
        print(f"[步骤3] 接口标注总时长：{api_annotation_sec} 秒", flush=True)

        print("[步骤3] 正在查询已标注/已质检/已转换任务...", flush=True)
        # 标注时长只统计已标注、已质检、已转换任务。
        try:
            import psycopg2
            section = self.config["fat-vla"]
            with psycopg2.connect(
                host=section.get("db_host"), port=section.getint("db_port", fallback=5432),
                user=section.get("db_user"), password=section.get("db_password"),
                dbname=section.get("db_name"), connect_timeout=30,
            ) as connection:
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
        # 标注片段全部计算完成后，再统一换算为秒并四舍五入。
        total_annotation_sec = int(total_annotation_ns / 1_000_000_000 + 0.5)
        actual_error_sec = abs(api_annotation_sec - total_annotation_sec)
        print(
            f"[步骤3] 标注片段数={len(rows)}，数据库={total_annotation_sec:.6f} 秒，"
            f"接口={api_annotation_sec:.6f} 秒，误差={actual_error_sec:.6f} 秒",
            flush=True,
        )
        if actual_error_sec > 1e-6:
            pytest.fail(
                f"标注总时长不一致：接口={api_annotation_sec}，数据库={total_annotation_sec}，"
                f"误差={actual_error_sec:.6f} 秒"
            )
        allure.attach(
            json.dumps(
                {"api_annotation_sec": api_annotation_sec, "db_annotation_sec": total_annotation_sec,
                 "segment_count": len(rows), "tasks": task_details},
                ensure_ascii=False, indent=2,
            ),
            name="标注总时长统计对比", attachment_type=allure.attachment_type.JSON,
        )

    @pytest.mark.order(15)
    @allure.story("步骤4：验证 Episode 总个数")
    def test_episode_count_statistics(self):
        """按任务状态和 annotation_segment_items 重算 Episode 数量。"""
        print("[步骤4] 开始读取工作台 Episode 统计接口...", flush=True)
        response = self.api_all.query_duration_stats(period="all")
        assertions.assert_code(response.status_code, 200)
        response_data = response.json()
        assertions.assert_text(response_data.get("msg", ""), "success")
        trend = response_data.get("data", {}).get("episode_trend")
        if not isinstance(trend, list) or not trend:
            pytest.fail("工作台统计响应缺少 data.episode_trend[0]")
        try:
            api_episode_count = int(trend[0]["episode_count"])
        except (KeyError, TypeError, ValueError) as exc:
            pytest.fail(f"data.episode_trend[0].episode_count 不是有效整数：{exc}")
        print(f"[步骤4] 接口 Episode 总数：{api_episode_count}", flush=True)

        print("[步骤4] 正在查询已转换任务...", flush=True)
        task_nos = []
        try:
            import psycopg2
            section = self.config["fat-vla"]
            with psycopg2.connect(
                host=section.get("db_host"), port=section.getint("db_port", fallback=5432),
                user=section.get("db_user"), password=section.get("db_password"),
                dbname=section.get("db_name"), connect_timeout=30,
            ) as connection:
                with connection.cursor() as cursor:
                    cursor.execute(
                        "SELECT task_no FROM tasks WHERE status = 5 "
                        "AND task_no IS NOT NULL AND task_no <> '' ORDER BY task_no"
                    )
                    task_nos = [str(row[0]).strip() for row in cursor.fetchall()]
        except Exception as exc:
            pytest.fail(f"查询 Episode 状态任务失败：{exc}")

        db_episode_count = self._episode_count(task_nos)
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
    @allure.story("步骤5：验证 Episode 总时长")
    def test_episode_duration_statistics(self):
        """按已转换任务的 Episode 标注时间范围重算总时长。"""
        print("[步骤5] 开始读取工作台转换时长统计接口...", flush=True)
        response = self.api_all.query_duration_stats(period="all")
        assertions.assert_code(response.status_code, 200)
        response_data = response.json()
        assertions.assert_text(response_data.get("msg", ""), "success")
        trend = response_data.get("data", {}).get("duration_trend")
        if not isinstance(trend, list) or not trend:
            pytest.fail("工作台统计响应缺少 data.duration_trend[0]")
        try:
            api_conversion_sec = float(trend[0]["conversion_sec"])
        except (KeyError, TypeError, ValueError) as exc:
            pytest.fail(f"data.duration_trend[0].conversion_sec 不是有效数字：{exc}")
        print(f"[步骤5] 接口 Episode 总时长：{api_conversion_sec} 秒", flush=True)

        print("[步骤5] 正在查询已转换任务...", flush=True)
        task_nos = []
        try:
            import psycopg2
            section = self.config["fat-vla"]
            with psycopg2.connect(
                host=section.get("db_host"), port=section.getint("db_port", fallback=5432),
                user=section.get("db_user"), password=section.get("db_password"),
                dbname=section.get("db_name"), connect_timeout=30,
            ) as connection:
                with connection.cursor() as cursor:
                    cursor.execute(
                        "SELECT task_no FROM tasks WHERE status = 5 "
                        "AND task_no IS NOT NULL AND task_no <> '' ORDER BY task_no"
                    )
                    task_nos = [str(row[0]).strip() for row in cursor.fetchall()]
        except Exception as exc:
            pytest.fail(f"查询已转换任务失败：{exc}")

        db_conversion_sec = self._episode_duration_sec(task_nos)
        actual_error_sec = abs(api_conversion_sec - db_conversion_sec)
        print(
            f"[步骤5] 已转换任务数={len(task_nos)}，数据库 Episode 时长={db_conversion_sec:.6f} 秒，"
            f"接口={api_conversion_sec:.6f} 秒，误差={actual_error_sec:.6f} 秒",
            flush=True,
        )
        if actual_error_sec > 1e-6:
            pytest.fail(
                f"Episode 总时长不一致：接口={api_conversion_sec}，"
                f"数据库={db_conversion_sec}，误差={actual_error_sec:.6f} 秒"
            )
        allure.attach(
            json.dumps(
                {"api_conversion_sec": api_conversion_sec, "db_episode_sec": db_conversion_sec,
                 "rounding": "所有 Episode 时长先累加，最后统一四舍五入到秒",
                 "task_count": len(task_nos),
                 "task_nos": task_nos},
                ensure_ascii=False, indent=2,
            ),
            name="Episode 总时长统计对比", attachment_type=allure.attachment_type.JSON,
        )

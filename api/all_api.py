"""
VLA所有接口
"""
from api import api_login
from common.Request_Response import ApiClient

env = api_login.url


class ApiAll():
    def __init__(self, client: ApiClient):
        self.client = client

    # ==================== 工作台 ====================

    # ==================== 任务管理 ====================
    # 创建任务
    def create_task(self, name, category, scene_tags, robot_config_id):
        url = f"{env}/api/v1/tasks/add"
        payload = {
            "name": name,
            "description": "自动化描述",
            "annotation_levels": [],
            "category": category,
            "scene_tags": scene_tags,
            "robot_config_id": robot_config_id,
            "collect_requirement": "自动化要求",
            "target_count": 10,
            "priority": "high",
        }

        response = self.client.post_with_retry(url, json=payload)
        return response

    # 删除任务
    def delete_task(self, task_id):
        url = f"{env}/api/v1/tasks/{task_id}"

        response = self.client.request_with_retry("DELETE", url)
        return response

    # 查询任务管理列表页
    def query_task_list(self, page_index=1, page_size=100):
        url = f"{env}/api/v1/tasks"
        params = {"page_index": page_index, "page_size": page_size}

        response = self.client.get_with_retry(url, params=params)
        return response

    # 获取数据分类
    def query_data_categories(self, page_index=1, page_size=100):
        url = f"{env}/api/v1/data-categories"
        params = {"page_index": page_index, "page_size": page_size}

        response = self.client.get_with_retry(url, params=params)
        return response

    # 获取标签
    def query_scene_tags(self, enabled=True, page_index=1, page_size=100):
        url = f"{env}/api/v1/scene-tags"
        params = {
            "enabled": enabled,
            "page_index": page_index,
            "page_size": page_size,
        }

        response = self.client.get_with_retry(url, params=params)
        return response

    # 获取机器人构型
    def query_robot_configs(self, page_index=1, page_size=100):
        url = f"{env}/api/v1/robot-configs"
        params = {"page_index": page_index, "page_size": page_size}

        response = self.client.get_with_retry(url, params=params)
        return response

    # ==================== 数据采集 ====================
    # 查询数据采集列表页
    def query_data_collection_list(self, page_index=1, page_size=100):
        url = f"{env}/api/v1/tasks"
        params = {
            "page_index": page_index,
            "page_size": page_size,
            "status": 1,
        }

        response = self.client.get_with_retry(url, params=params)
        return response

    # 提交采集完成
    def complete_collect_files(self, task_id):
        url = f"{env}/api/v1/tasks/{task_id}/collect-files/complete"

        response = self.client.post_with_retry(url)
        return response

    # 登记待上传采集文件
    def register_collect_files(self, task_id, files):
        url = f"{env}/api/v1/tasks/{task_id}/collect-files/register"
        payload = {"files": files}

        response = self.client.post_with_retry(url, json=payload)
        return response

    # 查询采集文件上传汇总
    def query_collect_files_summary(self, task_id):
        url = f"{env}/api/v1/tasks/{task_id}/collect-files/summary"

        response = self.client.get_with_retry(url)
        return response

    # 按上传状态查询采集文件
    def query_collect_files(self, task_id, status_group, page_index=1, page_size=100):
        url = f"{env}/api/v1/tasks/{task_id}/collect-files"
        params = {
            "status_group": status_group,
            "page_index": page_index,
            "page_size": page_size,
        }

        response = self.client.get_with_retry(url, params=params)
        return response

    # 查询采集文件断点续传位置
    def query_collect_file_resume(self, task_id, task_file_id):
        url = f"{env}/api/v1/tasks/{task_id}/collect-files/resume/{task_file_id}"

        response = self.client.get_with_retry(url)
        return response

    # 上传采集文件二进制流
    def upload_collect_file_stream(
            self,
            task_id,
            filename,
            relative_path,
            task_file_id,
            offset,
            file_stream,
            content_length=None,
            timeout=3600,
    ):
        url = f"{env}/api/v1/tasks/{task_id}/collect-files/stream"
        params = {
            "filename": filename,
            "relative_path": relative_path,
            "task_file_id": task_file_id,
            "offset": offset,
        }
        headers = {"Content-Type": "application/octet-stream"}
        if content_length is not None:
            headers["Content-Length"] = str(int(content_length))

        response = self.client.post_with_retry(
            url,
            params=params,
            data=file_stream,
            headers=headers,
            timeout=timeout,
        )
        return response

    # ==================== 数据标注 ====================
    # 查询数据标注列表页
    def query_annotation_list(
            self, page_index=1, page_size=99, annotation_status="all"
    ):
        url = f"{env}/api/v1/tasks"
        params = {
            "page_index": page_index,
            "page_size": page_size,
            "annotation_status": annotation_status,
        }

        response = self.client.get_with_retry(url, params=params)
        return response

    # 查询标注工作区
    def query_annotation_workspace(self, task_id):
        url = f"{env}/api/v1/task-annotations/tasks/{task_id}/annotation-workspace"

        response = self.client.get_with_retry(url)
        return response

    # 新增标注接口
    def create_task_annotation_segments(self, annotation_id, tag_vocabulary, segments, playback):
        url = f"{env}/api/v1/task-annotations/annotations/{annotation_id}/segments"
        payload = {
            "tag_vocabulary": tag_vocabulary,
            "segments": segments,
            "deleted_segments": [],
            "playback": playback,
            "status": "DRAFT",
        }

        response = self.client.patch_with_retry(url, json=payload)
        return response

    # 提交标注接口
    def submit_task_annotation(self, annotation_id, tag_vocabulary, annotation_json):
        url = f"{env}/api/v1/task-annotations/annotations/{annotation_id}/submit"
        payload = {
            "tag_vocabulary": tag_vocabulary,
            "annotation_json": annotation_json,
        }

        response = self.client.post_with_retry(url, json=payload)
        return response

    # 查询任务视频播放清单（提交自动化标注时序列化为 input.video_playlist）
    def query_task_video_playlist(self, task_id):
        url = f"{env}/api/v1/task-annotations/tasks/{task_id}/video-playlist"
        return self.client.get_with_retry(url)

    # 获取单个机器人构型详情（包含 config_json）
    def query_robot_config_detail(self, robot_config_id):
        url = f"{env}/api/v1/robot-configs/{robot_config_id}"
        return self.client.get_with_retry(url)

    # 参数预检
    def auto_labeling_pre_check(self, payload):
        url = f"{env}/api/v1/auto-labeling/pre-check"
        # 422 是参数校验失败，重试不会改变请求内容；保留响应体便于定位具体字段。
        return self.client.post_no_raise(url, json=payload)

    # 生成自动化标注 job_id
    def create_auto_labeling_job_id(self):
        url = f"{env}/api/v1/task-annotations/auto-labeling/job-id"
        return self.client.post_with_retry(url)

    # 提交自动化标注任务
    def submit_auto_labeling_task(self, payload):
        url = f"{env}/api/v1/auto-labeling/tasks"
        return self.client.post_with_retry(url, json=payload)

    # 查询自动化标注任务状态
    def query_auto_labeling_task(self, task_id, job_id):
        url = f"{env}/api/v1/auto-labeling/tasks/{task_id}/{job_id}"
        return self.client.get_with_retry(url)

    # 查询任务最近一次自动化标注结果
    def query_latest_auto_labeling_job(self, task_id):
        url = f"{env}/api/v1/task-annotations/tasks/{task_id}/auto-labeling/jobs/latest"
        return self.client.get_with_retry(url)

    # ==================== 人工质检 ====================
    # 完成质检接口
    def complete_task_qc(self, task_id):
        url = f"{env}/api/v1/tasks/{task_id}/complete-qc"

        response = self.client.post_with_retry(url)
        return response

    # ==================== 数据转换 ====================
    # 创建数据转换任务
    def create_conversion(self, task_id, target_format, quality_labels):
        url = f"{env}/api/v1/conversions"
        payload = {
            "task_ids": [str(task_id)],
            "target_format": target_format,
            "quality_labels": quality_labels,
        }

        response = self.client.post_with_retry(url, json=payload)
        return response

    # 查询全部数据转换任务列表页
    def query_conversion_list(self):
        url = f"{env}/api/v1/conversions"
        params = {
            "view": "all",
            "page_index": 1,
            "page_size": 100,
        }

        response = self.client.get_no_raise(url, params=params)
        return response

    # ==================== 数据可视化 ====================

    # ==================== 用户与权限 ====================

    # ==================== 基础配置 ====================

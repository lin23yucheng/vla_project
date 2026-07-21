"""
VLA所有接口
"""
from api import api_login
from common.Request_Response import ApiClient

env = api_login.url


class ApiAll():
    def __init__(self, client: ApiClient):
        self.client = client

    # 查询正在转换的任务列表页
    def query_active_conversion_list(self):
        url = f"{env}/api/v1/conversions"
        params = {
            "view": "active",
            "page_index": 1,
            "page_size": 99,
        }

        response = self.client.get_with_retry(url, params=params)
        return response

    # 查询已转换完成的任务列表页
    def query_finished_conversion_list(self):
        url = f"{env}/api/v1/conversions"
        params = {
            "view": "finished",
            "page_index": 1,
            "page_size": 99,
        }

        response = self.client.get_with_retry(url, params=params)
        return response

    # 数据转换接口
    def create_conversion(self, task_id, target_format, quality_labels):
        url = f"{env}/api/v1/conversions"
        payload = {
            "task_ids": [str(task_id)],
            "target_format": target_format,
            "quality_labels": quality_labels,
        }

        response = self.client.post_with_retry(url, json=payload)
        return response

    # 完成质检接口
    def complete_task_qc(self, task_id):
        url = f"{env}/api/v1/tasks/{task_id}/complete-qc"

        response = self.client.post_with_retry(url)
        return response

    # 查询标注workspace
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

    # 查询任务管理列表页
    def query_task_list(self, page_index=1, page_size=99):
        url = f"{env}/api/v1/tasks"
        params = {"page_index": page_index, "page_size": page_size}

        response = self.client.get_with_retry(url, params=params)
        return response

    # 查询数据转换列表页
    def query_conversion_list(self, view="finished", page_index=1, page_size=100):
        url = f"{env}/api/v1/conversions"
        params = {
            "view": view,
            "page_index": page_index,
            "page_size": page_size,
        }

        response = self.client.get_no_raise(url, params=params)
        return response

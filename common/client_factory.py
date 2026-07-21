"""HTTP 客户端工厂辅助工具。

懒加载客户端使模块导入无副作用：仅在测试或 API 封装层实际发送请求时才进行认证。
"""

from api import api_login
from common.Request_Response import ApiClient


def create_yixiu_headers():
    return {
        "Authorization": api_login.ApiLogin().login()
    }


def create_yixiu_client():
    return ApiClient(base_headers=create_yixiu_headers())


class LazyApiClient:
    def __init__(self, factory):
        self._factory = factory
        self._client = None

    def get_client(self):
        if self._client is None:
            self._client = self._factory()
        return self._client

    def __getattr__(self, name):
        return getattr(self.get_client(), name)


def create_lazy_yixiu_client():
    def factory():
        return create_yixiu_client()

    return LazyApiClient(factory)

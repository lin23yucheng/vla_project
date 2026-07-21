"""
封装Assert方法
"""
import json
import traceback
import sys
from common.Log import MyLog


class Assertions:
    def __init__(self):
        self.log = MyLog  # 使用新的日志类

    @staticmethod
    def assert_is_not_none(value, message):
        """检查值是否为 None，否则抛出断言错误"""
        if value is None:
            raise AssertionError(message)
        return True

    def assert_code(self, actual_code, expected_code):
        """
        验证response状态码
        :param actual_code: 实际状态码
        :param expected_code: 预期状态码
        :return:
        """
        try:
            assert actual_code == expected_code
            return True
        except AssertionError as e:
            error_msg = f"状态码错误\n预期: {expected_code}\n实际: {actual_code}"
            # 简化错误信息，不再记录位置和堆栈
            raise AssertionError(error_msg) from e

    def assert_body(self, body, body_msg, expected_msg):
        """
        验证response body中任意属性的值
        :param body: 响应体
        :param body_msg: 要检查的属性名
        :param expected_msg: 预期值
        :return:
        """
        try:
            msg = body[body_msg]
            assert msg == expected_msg
            return True
        except KeyError:
            error_msg = f"响应体中缺少属性 '{body_msg}'"
            # 简化错误信息
            raise AssertionError(error_msg)
        except AssertionError as e:
            error_msg = (f"响应体属性值不匹配\n"
                         f"属性: '{body_msg}'\n"
                         f"预期值: '{expected_msg}'\n"
                         f"实际值: '{body.get(body_msg, '')}'")
            # 简化错误信息
            raise AssertionError(error_msg) from e

    # noinspection PyUnboundLocalVariable
    def assert_in_text(self, body, expected_msg):
        """
        验证response body中是否包含预期字符串
        :param body: 响应体
        :param expected_msg: 预期包含的字符串
        :return:
        """
        try:
            # 处理不同类型的数据
            if isinstance(body, dict) or isinstance(body, list):
                text = json.dumps(body, ensure_ascii=False)
            else:
                text = str(body)

            assert expected_msg in text
            return True
        except AssertionError as e:
            # 截取部分响应文本用于显示
            excerpt = text[:200] + "..." if len(text) > 200 else text
            error_msg = (f"响应不包含预期内容\n"
                         f"预期: '{expected_msg}'\n"
                         f"实际: '{excerpt}'")
            # 简化错误信息
            raise AssertionError(error_msg) from e

    def assert_text(self, body, expected_msg):
        """
        验证response body中是否等于预期字符串
        :param body: 响应体
        :param expected_msg: 预期字符串
        :return:
        """
        try:
            assert body == expected_msg
            return True
        except AssertionError as e:
            error_msg = (f"响应不等于预期值\n"
                         f"预期: '{expected_msg}'\n"
                         f"实际: '{body}'")
            # 简化错误信息
            raise AssertionError(error_msg) from e

    def assert_time(self, actual_time, expected_time):
        """
        验证response body响应时间小于预期最大响应时间,单位：毫秒
        :param actual_time: 实际响应时间
        :param expected_time: 预期最大响应时间
        :return:
        """
        try:
            assert actual_time < expected_time
            return True
        except AssertionError as e:
            error_msg = (f"响应时间超过预期\n"
                         f"预期最大时间: {expected_time}ms\n"
                         f"实际响应时间: {actual_time}ms")
            # 简化错误信息
            raise AssertionError(error_msg) from e

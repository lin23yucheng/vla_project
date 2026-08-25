import json
import os
import shutil
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

import pytest

from common.Log import MyLog, set_log_level

# 设置全局日志级别
set_log_level('info')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ALLURE_RESULTS = os.path.join(BASE_DIR, "report", "allure-results")
ALLURE_REPORT = os.path.join(BASE_DIR, "report", "allure-report")

# 在这里按顺序填写要执行的测试文件
TEST_FILES: List[str] = [
    # "testcase/test_gripper_v2_verify.py", # 夹爪
    # "testcase/test_shake_v3_verify.py",   # 摇操
    # "testcase/test_tianji_v2_verify.py",  # 天机
    "testcase/test_business_api_workflow.py"  # 业务API工作流
]


@dataclass(frozen=True)
class WorkflowFailurePolicy:
    """定义一个工作流中前置步骤和验证步骤的失败处理方式。"""

    critical_steps: frozenset[int]
    validation_groups: tuple[tuple[int, ...], ...]

    def validation_group_for(self, step: int) -> tuple[int, ...] | None:
        return next(
            (group for group in self.validation_groups if step in group),
            None,
        )


WORKFLOW_FAILURE_POLICIES = {
    "test_gripper_v2_verify.py": WorkflowFailurePolicy(
        critical_steps=frozenset(range(1, 12)),
        validation_groups=(
            (12, 13, 14),
            (15, 16, 17),
            (18, 19, 20),
            (21, 22, 23),
            (24, 25, 26),
            (27, 28, 29),
            (30,),
        ),
    ),
    "test_tianji_v2_verify.py": WorkflowFailurePolicy(
        critical_steps=frozenset(range(1, 10)),
        validation_groups=((10,), (11,), (12,)),
    ),
    "test_shake_v3_verify.py": WorkflowFailurePolicy(
        critical_steps=frozenset(range(1, 10)),
        validation_groups=((10, 11, 12), (13, 14), (15,)),
    ),
}


class WorkflowFailureController:
    """在验证失败后跳过同组剩余步骤，在前置失败后停止当前工作流。"""

    def __init__(self, policy: WorkflowFailurePolicy) -> None:
        self.policy = policy
        self.failed_groups: dict[tuple[int, ...], int] = {}

    @staticmethod
    def _step_number(item) -> int | None:
        marker = item.get_closest_marker("order")
        if marker is None or not marker.args:
            return None
        try:
            return int(marker.args[0])
        except (TypeError, ValueError):
            return None

    def pytest_runtest_setup(self, item) -> None:
        step = self._step_number(item)
        if step is None:
            return
        group = self.policy.validation_group_for(step)
        failed_step = self.failed_groups.get(group) if group else None
        if failed_step is not None:
            pytest.skip(
                f"步骤{failed_step}失败，跳过同组验证步骤{step}"
            )

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_makereport(self, item, call):
        outcome = yield
        report = outcome.get_result()
        if report.when not in {"setup", "call"} or not report.failed:
            return

        step = self._step_number(item)
        if step is None:
            return
        if step in self.policy.critical_steps:
            item.session.shouldstop = (
                f"前置步骤{step}失败，停止后续工作流步骤"
            )
            return

        group = self.policy.validation_group_for(step)
        if group is not None:
            self.failed_groups.setdefault(group, step)


def format_time(seconds: float) -> str:
    """将秒数转换为 X分X秒 格式。"""
    minutes = int(seconds // 60)
    remaining_seconds = int(seconds % 60)
    return f"{minutes}分{remaining_seconds}秒"


def reset_logs():
    """清除历史日志并重新初始化日志处理器。"""
    from common.Log import logger, MyLog

    log_dir = MyLog.get_log_dir()
    print(f"日志目录: {log_dir}")

    for handler in logger.handlers[:]:
        try:
            handler.flush()
            handler.close()
        except Exception as exc:
            print(f"关闭日志处理器失败: {exc}")
        finally:
            logger.removeHandler(handler)

    time.sleep(1)

    if os.path.exists(log_dir):
        try:
            shutil.rmtree(log_dir)
            print(f"已删除日志目录: {log_dir}")
        except Exception as exc:
            print(f"删除日志目录失败: {exc}")

    os.makedirs(log_dir, exist_ok=True)
    MyLog.reinit_handlers()
    MyLog.info("已清除历史日志文件")


def prepare_report_dirs():
    """清理并初始化 Allure 结果目录。"""
    if os.path.exists(ALLURE_RESULTS):
        shutil.rmtree(ALLURE_RESULTS)
    os.makedirs(ALLURE_RESULTS, exist_ok=True)
    MyLog.info("已清理历史报告数据")


def resolve_test_path(test_file: str) -> str:
    """将相对路径解析为项目下的绝对路径。"""
    if os.path.isabs(test_file):
        return test_file
    return os.path.join(BASE_DIR, test_file.replace('/', os.sep))


def execute_test(test_file: str) -> int:
    """执行单个测试文件。"""
    target_path: str = resolve_test_path(test_file)
    MyLog.info(f"开始执行测试文件: {test_file}")

    if not os.path.exists(target_path):
        MyLog.error(f"错误：测试文件不存在: {target_path}")
        return 1

    pytest_args: List[str] = []
    pytest_args.append("-v")
    pytest_args.append("-s")
    pytest_args.append(target_path)
    pytest_args.append(f"--alluredir={ALLURE_RESULTS}")

    policy = WORKFLOW_FAILURE_POLICIES.get(Path(target_path).name)
    plugins = [WorkflowFailureController(policy)] if policy else None
    exit_code = pytest.main(pytest_args, plugins=plugins)

    if exit_code == 0:
        MyLog.info(f"测试文件 {test_file} 全部通过")
    elif exit_code == 1:
        MyLog.error(f"测试文件 {test_file} 存在失败用例")
    else:
        MyLog.critical(f"测试文件 {test_file} 执行错误，退出码: {exit_code}")

    return exit_code


def generate_allure_report(total_time_text: str | None = None) -> None:
    """生成 Allure 报告。"""
    if not os.path.exists(ALLURE_RESULTS) or not os.listdir(ALLURE_RESULTS):
        MyLog.info("没有生成测试结果，跳过报告生成")
        return

    command = f'allure generate "{ALLURE_RESULTS}" -o "{ALLURE_REPORT}" --clean'
    result = os.system(command)
    if result != 0:
        MyLog.error("Allure 报告生成失败，请确认本机已正确安装 allure 命令")
        return

    report_env_file = os.path.join(ALLURE_REPORT, 'widgets', 'environment.json')
    if total_time_text and os.path.exists(report_env_file):
        try:
            with open(report_env_file, 'r', encoding='utf-8') as file:
                env_data = json.load(file)

            env_data.append({
                "name": "执行时间",
                "values": [f"整体耗时: {total_time_text}"],
            })

            with open(report_env_file, 'w', encoding='utf-8') as file:
                json.dump(env_data, file, ensure_ascii=False, indent=2)
        except Exception as exc:
            MyLog.warning(f"更新 Allure 环境信息失败: {exc}")

    report_url = f"file://{os.path.abspath(ALLURE_REPORT)}/index.html"
    MyLog.info(f"测试报告生成成功: {report_url}")
    print(f"测试报告生成成功: {report_url}")


def run_order_tests() -> int:
    """按 TEST_FILES 中的顺序依次执行测试，并在结束后生成 Allure 报告。"""
    if not TEST_FILES:
        raise ValueError("TEST_FILES 为空，请先在 run.py 中填写要执行的测试文件路径")

    sys.path.append(BASE_DIR)
    reset_logs()
    prepare_report_dirs()

    start_time = time.time()
    MyLog.info("===== 开始顺序执行测试任务 =====")
    MyLog.info(f"执行文件列表: {TEST_FILES}")

    file_times = {}
    final_exit_code = 0

    try:
        for test_file in TEST_FILES:
            file_start_time = time.time()
            exit_code = execute_test(test_file)
            elapsed = time.time() - file_start_time
            file_times[test_file] = elapsed

            formatted = format_time(elapsed)
            MyLog.info(f"测试文件 {test_file} 执行完成，耗时: {formatted}")

            if exit_code != 0:
                final_exit_code = exit_code
                MyLog.critical(f"执行测试文件 {test_file} 失败，停止后续执行")
                break

    except KeyboardInterrupt:
        final_exit_code = 130
        MyLog.warning("用户中断测试执行")
        print("\n用户中断测试执行，正在生成报告...")

    finally:
        total_time = format_time(time.time() - start_time)

        if file_times:
            MyLog.info("===== 测试文件耗时明细 =====")
            for file_name, seconds in file_times.items():
                MyLog.info(f"{file_name}: {format_time(seconds)}")

        generate_allure_report(total_time)
        MyLog.info(f"VLA_Project-整体耗时: {total_time}")
        MyLog.info("===== 测试任务完成 =====")
        print(f"\033[32mVLA_Project-整体耗时: {total_time}\033[0m")

    return final_exit_code


def signal_handler(sig, frame):
    """处理中断信号。"""
    MyLog.warning(f"接收到中断信号: {sig}")
    raise KeyboardInterrupt


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print("===== 测试执行程序启动 =====")
    MyLog.info("===== 测试执行程序启动 =====")

    exit_code = 0
    try:
        exit_code = run_order_tests()
    except Exception as exc:
        exit_code = 1
        MyLog.error(f"执行测试任务时发生异常: {exc}", exc_info=True)
        print(f"执行测试任务时发生异常: {exc}")
    finally:
        MyLog.info("===== 测试执行程序结束 =====")
        print("===== 测试执行程序结束 =====")

    sys.exit(exit_code)

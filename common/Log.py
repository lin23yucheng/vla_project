import logging
import os
import sys
import traceback
import time

LEVELS = {
    'debug': logging.DEBUG,
    'info': logging.INFO,
    'warning': logging.WARNING,
    'error': logging.ERROR,
    'critical': logging.CRITICAL
}

logger = logging.getLogger()
level = 'info'  # 默认日志级别设为info


def create_file(filename):
    """创建日志文件及目录"""
    try:
        path = os.path.dirname(filename)
        if not os.path.isdir(path):
            os.makedirs(path, exist_ok=True)
        if not os.path.isfile(filename):
            with open(filename, mode='w', encoding='utf-8') as f:
                pass
    except Exception as e:
        print(f"Error creating log file: {e}")


def set_log_level(log_level):
    """设置日志级别"""
    global level
    level = log_level
    logger.setLevel(LEVELS.get(level, logging.INFO))


class MyLog:
    """日志封装类"""

    @classmethod
    def get_base_path(cls):
        """动态获取项目根目录"""
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    @classmethod
    def get_log_dir(cls):
        """获取日志目录路径"""
        return os.path.join(cls.get_base_path(), 'Log')

    @classmethod
    def get_log_file(cls):
        """获取普通日志文件路径"""
        return os.path.join(cls.get_log_dir(), 'log.log')

    @classmethod
    def get_err_file(cls):
        """获取错误日志文件路径"""
        return os.path.join(cls.get_log_dir(), 'err.log')

    date_format = '%Y-%m-%d %H:%M:%S'

    # 增强的日志格式，包含文件名、行号、函数名
    detailed_format = '%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] %(funcName)s - %(message)s'

    # 初始化处理器
    @classmethod
    def init_handlers(cls):
        """初始化日志处理器"""
        # 确保目录存在
        log_dir = cls.get_log_dir()
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        # 创建文件
        create_file(cls.get_log_file())
        create_file(cls.get_err_file())

        # 配置处理器
        cls.handler = logging.FileHandler(cls.get_log_file(), encoding='utf-8')
        cls.err_handler = logging.FileHandler(cls.get_err_file(), encoding='utf-8')
        # cls.err_handler.setLevel(logging.WARNING)  # 确保错误处理器只记录警告及以上
        cls.err_handler.setLevel(logging.ERROR)  # 确保错误处理器只记录ERROR及以上

        # 设置详细格式
        formatter = logging.Formatter(
            cls.detailed_format,  # 使用包含位置信息的格式
            datefmt=cls.date_format
        )
        cls.handler.setFormatter(formatter)
        cls.err_handler.setFormatter(formatter)

        # 添加处理器
        logger.addHandler(cls.handler)
        logger.addHandler(cls.err_handler)
        logger.setLevel(LEVELS.get(level, logging.INFO))

    @classmethod
    def reinit_handlers(cls):
        """重新初始化日志处理器"""
        # 关闭现有处理器
        for handler in logger.handlers[:]:
            try:
                handler.close()
            except:
                pass
            logger.removeHandler(handler)

        # 重新初始化处理器
        cls.init_handlers()

    @staticmethod
    def debug(log_msg):
        logger.debug(log_msg)

    @staticmethod
    def info(log_msg):
        logger.info(log_msg)

    @staticmethod
    def warning(log_msg):
        logger.warning(log_msg)

    @staticmethod
    def error(log_msg, exc_info=False):
        """记录错误信息，可选包含异常堆栈"""
        if exc_info:
            # 获取完整的异常堆栈
            exc_type, exc_value, exc_traceback = sys.exc_info()
            stack_trace = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
            logger.error(f"{log_msg}\n堆栈跟踪:\n{stack_trace}")
        else:
            # 记录错误信息（自动包含位置）
            logger.error(log_msg)

    @staticmethod
    def critical(log_msg):
        logger.critical(log_msg)

    @staticmethod
    def log_exception():
        """记录当前异常的完整信息"""
        exc_type, exc_value, exc_traceback = sys.exc_info()
        stack_trace = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        logger.critical(f"未处理异常:\n{stack_trace}")


class AllureReporter:
    @classmethod
    def close(cls):
        """关闭Allure上下文"""
        pass

    @staticmethod
    def update_environment_info():
        """更新环境信息（空实现）"""
        pass


# 在模块加载时初始化处理器
MyLog.init_handlers()

if __name__ == "__main__":
    # 测试日志功能
    try:
        MyLog.debug("调试信息")
        MyLog.info("普通信息")
        MyLog.warning("警告信息")

        # 模拟错误
        1 / 0
    except Exception as e:
        MyLog.error(f"发生错误: {str(e)}", exc_info=True)

    MyLog.critical("严重错误")

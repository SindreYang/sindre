
    
import sys
import os
from loguru import logger


class CustomLogger:
    """
    一个使用 loguru 库的自定义日志记录器类，支持多进程、print重定向、日志分级存储等。

    Attributes:
        logger (logger): 配置好的 loguru 日志记录器实例。
    """

    def __init__(
        self,
        logger_name=None,
        level="DEBUG",
        log_dir="logs",
        console_output=True,
        file_output=False,
        capture_print=True,
        filter_log=None,
    ):
        """
        初始化 CustomLogger。

        Args:
            logger_name (str, optional): 日志记录器名称。
            level (str): 日志级别，默认为 'DEBUG'。
            log_dir (str): 日志文件存储目录。
            console_output (bool): 是否启用控制台输出。
            file_output (bool): 是否启用文件输出。
            capture_print (bool): 是否捕获 print 输出。
            filter_log (callable, optional): 自定义日志过滤函数。
        """
        # 参数校验
        #self._validate_level(level)

        self.level = level.upper()
        self.log_dir = log_dir
        self.console_output = console_output
        self.file_output = file_output
        self.capture_print = capture_print
        self.logger_name = logger_name
        self.filter_log = filter_log

        self.logger = logger
        self.logger.remove()

        
        self._configure_logger()

    def _validate_level(self, level):
        """验证日志级别有效性"""
        valid_levels = ['TRACE', 'DEBUG', 'INFO', 'SUCCESS', 'WARNING', 'ERROR', 'CRITICAL']
        level_upper = level.upper()
        
        if level_upper not in valid_levels:
            raise ValueError(
                f"无效的日志级别: {level}，有效级别为: {', '.join(valid_levels)}"
            )

    def _configure_logger(self):
        """配置日志处理器。"""
        self.logger.configure(extra={"name": self.logger_name or "SindreLogger"})
        log_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<blue>{extra[name]: <8}</blue> | "
            "<cyan>{file}</cyan>:<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        )

        if self.capture_print:
            self._redirect_print()

        # 控制台输出
        if self.console_output:
            self.logger.add(
                sys.stderr,
                level=self.level,
                format=log_format,
                filter=self.filter_log,
            )

        # 文件输出
        if self.file_output:
            os.makedirs(self.log_dir, exist_ok=True)
             # 运行日志，每天新创建
            run_log_path = os.path.join(self.log_dir, "run_{time:YYYY-MM-DD}.log")
            self.logger.add(
                run_log_path,
                level=self.level,
                format=log_format,
                filter=self.filter_log,
                enqueue=True,
                backtrace=True,
                diagnose=True,
                rotation="00:00"
            )

            # 错误日志，10MB 自动扩充
            error_log_path = os.path.join(self.log_dir, "error.log")
            self.logger.add(
                error_log_path,
                format=log_format,
                filter=self.filter_log,
                level="ERROR",
                enqueue=True,
                backtrace=True,
                diagnose=True,
                rotation="10 MB"
            )


    def _redirect_print(self):
        """重定向 print 输出到日志。"""
        class PrintInterceptor:
            def __init__(self, logger):
                self.logger = logger

            def write(self, message):
                if message.strip():
                    # 获取当前调用栈信息，从而获取准确的行号
                    import inspect
                    frame = inspect.currentframe().f_back
                    while frame and frame.f_code.co_name == 'write':
                        frame = frame.f_back
                    if frame:
                        line = frame.f_lineno
                    else:
                        line = None
                    if line:
                        self.logger.opt(depth=1).info(f"Print(line {line}): {message.strip()} ")
                    else:
                        self.logger.info("Print: {}", message.strip())

            def flush(self):
                pass

        sys.stdout = PrintInterceptor(self.logger)

    def get_logger(self):
        """获取配置好的日志记录器实例。"""
        return self.logger


# 示例用法

if __name__ == "__main__":
    # 创建一个日志记录器实例
    logger_name = "my_custom_logger"
    custom_logger = CustomLogger(logger_name, level="INFO", console_output=True,capture_print=False).get_logger()
    custom_logger.disable(__name__)

    # 记录不同级别的日志
    custom_logger.debug("This is a debug message.")
    custom_logger.enable(__name__)
    custom_logger.info("This is an info message.")
    custom_logger.warning("This is a warning message.")
    custom_logger.error("This is an error message.")

    # 捕获 print 输出
    print("This is a print statement that will be captured.")

    # 捕获异常
    @custom_logger.catch
    def fun():
        s = 10000 / 0
        print(s)
    fun()
    try:
        raise ValueError("22")
    except Exception:
        custom_logger.exception("555")

    # 创建一个日志过滤函数
    def filter_only_info(record):
        return record["level"].name == "INFO"

    # 创建一个带有过滤功能的日志记录器实例
    filtered_logger = CustomLogger(
        logger_name="filtered_logger",
        level="DEBUG",
        console_output=True,
        filter_log=filter_only_info
    ).get_logger()

    # 记录日志，只有 INFO 级别的日志会被输出
    filtered_logger.debug("This debug message should not be shown.")
    filtered_logger.info("This info message should be shown.")
    filtered_logger.warning("This warning message should not be shown.")
    
    
    # 基本用法
    base_logger = CustomLogger(
        logger_name="BaseLog",
        level="INFO",
        log_dir="example_logs",
        console_output=True,
        file_output=True,
        capture_print=True,
    ).get_logger()

    base_logger.debug("调试信息不可见")
    base_logger.info("普通运行信息")
    base_logger.warning("警告信息")
    
    # 原始 print 输出
    print("此消息会被捕获并添加日志来源")

    # 异常捕获示例
    @base_logger.catch
    def error_func():
        return 1 / 0

    error_func()

    # 过滤日志示例
    def info_only_filter(record):
        return record["level"].name == "INFO"

    filtered_logger = CustomLogger(
        logger_name="FilterLog",
        level="DEBUG",
        log_dir="example_logs",
        console_output=True,
        filter_log=info_only_filter,
    ).get_logger()

    filtered_logger.debug("调试信息不可见（被过滤）")
    filtered_logger.info("重要信息可见")
    filtered_logger.error("错误信息不可见（被过滤）")
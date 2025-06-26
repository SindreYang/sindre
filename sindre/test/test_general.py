"""
sindre.general模块测试用例
测试CustomLogger类的各种功能
"""

import pytest
import os
import sys
from sindre.general.logs import CustomLogger

def worker_function(logger):
    logger.info("来自子进程的日志")
class TestCustomLogger:
    """测试CustomLogger类的各种功能"""
    
    def test_init_basic(self, tmp_path):
        """测试基本初始化"""
        logger = CustomLogger(
            logger_name="test_logger",
            level="INFO",
            log_dir=str(tmp_path),
            console_output=True,
            file_output=False
        )
        
        assert logger.level == "INFO"
        assert logger.log_dir == str(tmp_path)
        assert logger.console_output is True
        assert logger.file_output is False
        assert logger.logger_name == "test_logger"
    
    def test_init_with_file_output(self, tmp_path):
        """测试启用文件输出的初始化"""
        logger = CustomLogger(
            logger_name="file_logger",
            level="DEBUG",
            log_dir=str(tmp_path),
            console_output=False,
            file_output=True
        )
        
        assert logger.file_output is True
        assert os.path.exists(str(tmp_path))
    
    def test_get_logger(self, tmp_path):
        """测试获取logger实例"""
        custom_logger = CustomLogger(
            logger_name="test_get_logger",
            level="INFO",
            log_dir=str(tmp_path)
        )
        
        logger_instance = custom_logger.get_logger()
        assert logger_instance is not None
        assert hasattr(logger_instance, 'info')
        assert hasattr(logger_instance, 'debug')
        assert hasattr(logger_instance, 'warning')
        assert hasattr(logger_instance, 'error')
    
    def test_log_levels(self, tmp_path):
        """测试不同日志级别的输出"""
        custom_logger = CustomLogger(
            logger_name="level_test",
            level="DEBUG",
            log_dir=str(tmp_path),
            console_output=False,
            file_output=True
        )
        
        logger = custom_logger.get_logger()
        
        # 测试各种日志级别
        logger.debug("调试信息")
        logger.info("信息")
        logger.warning("警告")
        logger.error("错误")
        
        # 验证日志文件是否创建
        log_files = os.listdir(str(tmp_path))
        assert len(log_files) >= 1  # 至少应该有运行日志文件
    
    def test_capture_print(self, tmp_path):
        """测试print输出捕获功能"""
        custom_logger = CustomLogger(
            logger_name="print_capture",
            level="INFO",
            log_dir=str(tmp_path),
            console_output=False,
            file_output=True,
            capture_print=True
        )
        
        logger = custom_logger.get_logger()
        
        # 测试print捕获
        print("这是一句print语句")
        
        # 恢复stdout
        sys.stdout = sys.__stdout__
    
    def test_filter_log(self, tmp_path):
        """测试日志过滤功能"""
        def only_info_filter(record):
            return record["level"].name == "INFO"
        
        custom_logger = CustomLogger(
            logger_name="filter_test",
            level="DEBUG",
            log_dir=str(tmp_path),
            console_output=False,
            file_output=True,
            filter_log=only_info_filter
        )
        
        logger = custom_logger.get_logger()
        
        # 这些日志应该被过滤掉
        logger.debug("调试信息 - 应该被过滤")
        logger.warning("警告信息 - 应该被过滤")
        
        # 这个日志应该保留
        logger.info("信息 - 应该保留")
    
    def test_exception_handling(self, tmp_path):
        """测试异常处理装饰器"""
        custom_logger = CustomLogger(
            logger_name="exception_test",
            level="ERROR",
            log_dir=str(tmp_path),
            console_output=False,
            file_output=True
        )
        logger = custom_logger.get_logger()
        @logger.catch(reraise=True)
        def error_function():
            return 1 / 0
        with pytest.raises(ZeroDivisionError):
            error_function()
    
    def test_invalid_level(self, tmp_path):
        """测试无效日志级别的处理"""
        with pytest.raises(ValueError):
            CustomLogger(
                logger_name="invalid_level_test",
                level="INVALID_LEVEL",  # 无效级别
                log_dir=str(tmp_path)
            )
    
    def test_logger_disable_enable(self, tmp_path):
        """测试logger的禁用和启用功能"""
        custom_logger = CustomLogger(
            logger_name="disable_test",
            level="INFO",
            log_dir=str(tmp_path)
        )
        
        logger = custom_logger.get_logger()
        
        # 禁用当前模块的日志
        logger.disable(__name__)
        
        # 启用当前模块的日志
        logger.enable(__name__)
    
    def test_multiprocessing_support(self, tmp_path):
        """测试多进程支持"""
        custom_logger = CustomLogger(
            logger_name="multiprocessing_test",
            level="INFO",
            log_dir=str(tmp_path),
            console_output=False,
            file_output=True
        )
        
        logger = custom_logger.get_logger()
        
        # 测试在多进程环境中使用
        import multiprocessing as mp
        
       
        
        # 创建子进程
        p = mp.Process(target=worker_function, args=(logger,))
        p.start()
        p.join()
        
        # 验证进程正常结束
        assert p.exitcode == 0


if __name__ == "__main__":
    pytest.main([__file__]) 
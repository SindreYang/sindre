# General 通用工具模块

> 提供通用工具和功能，包括高级日志记录、多进程支持、print重定向等

## 📋 目录

- [功能特性](#功能特性)
- [快速开始](#快速开始)
- [核心功能](#核心功能)
- [使用指南](#使用指南)
- [高级功能](#高级功能)
- [配置选项](#配置选项)
- [常见问题](#常见问题)

## ✨ 功能特性

- 📝 **高级日志**: 基于loguru的强大日志记录功能
- 🔄 **多进程支持**: 线程安全的日志记录
- 🖨️ **Print重定向**: 自动捕获print输出到日志
- 📊 **日志分级**: 支持多种日志级别和过滤
- 📁 **文件管理**: 自动日志文件轮转和管理
- 🎨 **彩色输出**: 美观的控制台彩色日志
- 🛡️ **异常捕获**: 自动异常记录和堆栈跟踪

## 🚀 快速开始

### 基本使用

```python
from sindre.general.logs import CustomLogger

# 创建基本日志记录器
logger = CustomLogger(
    logger_name="MyApp",
    level="INFO",
    console_output=True,
    file_output=True
).get_logger()

# 记录日志
logger.info("应用程序启动")
logger.warning("发现潜在问题")
logger.error("发生错误")
```

### 高级配置

```python
from sindre.general.logs import CustomLogger

# 创建高级日志记录器
logger = CustomLogger(
    logger_name="AdvancedApp",
    level="DEBUG",
    log_dir="logs",
    console_output=True,
    file_output=True,
    capture_print=True
).get_logger()

# 记录不同级别的日志
logger.debug("调试信息")
logger.info("普通信息")
logger.success("成功信息")
logger.warning("警告信息")
logger.error("错误信息")
logger.critical("严重错误")
```

## 🔧 核心功能

### CustomLogger 类

```python
class CustomLogger:
    """自定义日志记录器类"""
    
    def __init__(
        self,
        logger_name=None,
        level="DEBUG",
        log_dir="logs",
        console_output=True,
        file_output=False,
        capture_print=False,
        filter_log=None
    ):
        """
        初始化日志记录器
        
        Args:
            logger_name: 日志记录器名称
            level: 日志级别 (TRACE, DEBUG, INFO, SUCCESS, WARNING, ERROR, CRITICAL)
            log_dir: 日志文件存储目录
            console_output: 是否启用控制台输出
            file_output: 是否启用文件输出
            capture_print: 是否捕获print输出
            filter_log: 自定义日志过滤函数
        """
    
    def get_logger(self):
        """获取配置好的日志记录器实例"""
        return self.logger
```

## 📖 使用指南

### 1. 基本日志记录

#### 简单日志记录

```python
from sindre.general.logs import CustomLogger

# 创建基本日志记录器
logger = CustomLogger(
    logger_name="SimpleApp",
    level="INFO",
    console_output=True
).get_logger()

# 记录不同级别的日志
logger.debug("这是调试信息")
logger.info("这是普通信息")
logger.warning("这是警告信息")
logger.error("这是错误信息")
logger.critical("这是严重错误")

# 格式化日志
name = "张三"
age = 25
logger.info(f"用户 {name} 年龄 {age} 岁")
logger.info("用户 {} 年龄 {} 岁", name, age)  # loguru风格
```

#### 文件日志记录

```python
from sindre.general.logs import CustomLogger

# 创建文件日志记录器
logger = CustomLogger(
    logger_name="FileApp",
    level="DEBUG",
    log_dir="logs",
    console_output=True,
    file_output=True
).get_logger()

# 日志会自动保存到文件
logger.info("这条信息会同时显示在控制台和保存到文件")
logger.error("错误信息也会被记录")

# 查看生成的日志文件
# - logs/run_YYYY-MM-DD.log (运行日志，每天轮转)
# - logs/error.log (错误日志，10MB轮转)
```

### 2. Print重定向

#### 捕获Print输出

```python
from sindre.general.logs import CustomLogger

# 创建支持print重定向的日志记录器
logger = CustomLogger(
    logger_name="PrintCapture",
    level="INFO",
    console_output=True,
    capture_print=True
).get_logger()

# 普通的print语句会被自动捕获
print("这条print语句会被捕获并记录到日志中")
print("包含行号信息的print输出")

# 日志输出示例：
# 2024-01-01 12:00:00 | INFO     | PrintCapture | main.py:main:15 - Print(line 15): 这条print语句会被捕获并记录到日志中
```

#### 混合使用

```python
from sindre.general.logs import CustomLogger

logger = CustomLogger(
    logger_name="MixedApp",
    level="INFO",
    console_output=True,
    capture_print=True
).get_logger()

# 使用logger记录
logger.info("使用logger记录的信息")

# 使用print（会被自动捕获）
print("使用print输出的信息")

# 两者都会显示在日志中，但格式略有不同
```

### 3. 异常捕获

#### 自动异常捕获

```python
from sindre.general.logs import CustomLogger

logger = CustomLogger(
    logger_name="ExceptionApp",
    level="INFO",
    console_output=True
).get_logger()

# 使用装饰器自动捕获异常
@logger.catch
def risky_function():
    """可能出错的函数"""
    result = 10 / 0
    return result

# 调用函数，异常会被自动记录
try:
    risky_function()
except Exception:
    pass

# 手动捕获异常
try:
    raise ValueError("这是一个测试异常")
except Exception:
    logger.exception("捕获到异常")
```

### 4. 日志过滤

#### 自定义过滤器

```python
from sindre.general.logs import CustomLogger

# 定义过滤函数
def info_only_filter(record):
    """只显示INFO级别的日志"""
    return record["level"].name == "INFO"

def exclude_sensitive_filter(record):
    """排除包含敏感信息的日志"""
    sensitive_words = ["password", "token", "secret"]
    message = record["message"].lower()
    return not any(word in message for word in sensitive_words)

# 创建带过滤器的日志记录器
logger = CustomLogger(
    logger_name="FilteredApp",
    level="DEBUG",
    console_output=True,
    filter_log=info_only_filter
).get_logger()

# 只有INFO级别的日志会被显示
logger.debug("这条调试信息不会显示")
logger.info("这条信息会显示")
logger.warning("这条警告不会显示")
```

## 🚀 高级功能

### 1. 多进程日志

```python
from sindre.general.logs import CustomLogger
import multiprocessing as mp

def worker_process(logger_name, process_id):
    """工作进程函数"""
    logger = CustomLogger(
        logger_name=logger_name,
        level="INFO",
        log_dir="logs",
        console_output=False,
        file_output=True
    ).get_logger()
    
    logger.info(f"进程 {process_id} 开始工作")
    # 执行一些工作...
    logger.info(f"进程 {process_id} 完成工作")

# 主进程
if __name__ == "__main__":
    main_logger = CustomLogger(
        logger_name="MainApp",
        level="INFO",
        console_output=True,
        file_output=True
    ).get_logger()
    
    main_logger.info("启动多进程任务")
    
    # 创建多个进程
    processes = []
    for i in range(4):
        p = mp.Process(target=worker_process, args=("WorkerApp", i))
        processes.append(p)
        p.start()
    
    # 等待所有进程完成
    for p in processes:
        p.join()
    
    main_logger.info("所有进程完成")
```

### 2. 日志轮转

```python
from sindre.general.logs import CustomLogger

# 日志会自动轮转
logger = CustomLogger(
    logger_name="RotationApp",
    level="INFO",
    log_dir="logs",
    file_output=True
).get_logger()

# 运行日志：每天00:00轮转
# 错误日志：达到10MB时轮转
for i in range(1000):
    logger.info(f"这是第 {i} 条日志信息")
    if i % 100 == 0:
        logger.error(f"这是第 {i} 条错误信息")
```

### 3. 性能监控

```python
from sindre.general.logs import CustomLogger
import time

logger = CustomLogger(
    logger_name="PerformanceApp",
    level="INFO",
    console_output=True
).get_logger()

# 使用上下文管理器记录执行时间
@logger.catch
def performance_test():
    start_time = time.time()
    
    # 模拟耗时操作
    time.sleep(2)
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    logger.info(f"操作完成，耗时: {execution_time:.2f}秒")
    
    # 模拟异常
    if execution_time > 1.5:
        raise TimeoutError("操作超时")

# 执行性能测试
performance_test()
```

### 4. 结构化日志

```python
from sindre.general.logs import CustomLogger
import json

logger = CustomLogger(
    logger_name="StructuredApp",
    level="INFO",
    console_output=True
).get_logger()

# 记录结构化数据
user_data = {
    "user_id": 12345,
    "username": "张三",
    "action": "login",
    "timestamp": "2024-01-01T12:00:00Z"
}

logger.info("用户操作: {}", json.dumps(user_data, ensure_ascii=False))

# 记录业务事件
def log_business_event(event_type, data):
    """记录业务事件"""
    event = {
        "event_type": event_type,
        "data": data,
        "timestamp": time.time()
    }
    logger.info("业务事件: {}", json.dumps(event, ensure_ascii=False))

# 使用示例
log_business_event("user_login", {"user_id": 12345, "ip": "192.168.1.1"})
log_business_event("order_created", {"order_id": "ORD001", "amount": 99.99})
```

## ⚙️ 配置选项

### 日志级别

```python
# 支持的日志级别（从低到高）
levels = [
    "TRACE",      # 最详细的调试信息
    "DEBUG",      # 调试信息
    "INFO",       # 一般信息
    "SUCCESS",    # 成功信息
    "WARNING",    # 警告信息
    "ERROR",      # 错误信息
    "CRITICAL"    # 严重错误
]

# 设置日志级别
logger = CustomLogger(
    logger_name="LevelApp",
    level="WARNING",  # 只显示WARNING及以上级别的日志
    console_output=True
).get_logger()
```

### 日志格式

```python
# 默认日志格式
log_format = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<blue>{extra[name]: <8}</blue> | "
    "<cyan>{file}</cyan>:<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)

# 输出示例：
# 2024-01-01 12:00:00 | INFO     | MyApp     | main.py:main:15 - 应用程序启动
```

### 文件配置

```python
# 日志文件配置
logger = CustomLogger(
    logger_name="FileConfigApp",
    level="INFO",
    log_dir="custom_logs",  # 自定义日志目录
    console_output=True,
    file_output=True
).get_logger()

# 生成的文件：
# - custom_logs/run_2024-01-01.log (运行日志)
# - custom_logs/error.log (错误日志)
```

## ❓ 常见问题

### Q1: 如何禁用特定模块的日志？

**A**: 使用enable/disable方法：
```python
from sindre.general.logs import CustomLogger

logger = CustomLogger("MyApp").get_logger()

# 禁用当前模块的日志
logger.disable(__name__)
logger.info("这条日志不会显示")

# 重新启用
logger.enable(__name__)
logger.info("这条日志会显示")
```

### Q2: 如何自定义日志格式？

**A**: 修改CustomLogger类的_configure_logger方法：
```python
# 在CustomLogger类中修改log_format
log_format = (
    "{time:YYYY-MM-DD HH:mm:ss} | "
    "{level} | "
    "{name} | "
    "{message}"
)
```

### Q3: 如何处理大量日志？

**A**: 使用日志轮转和过滤：
```python
logger = CustomLogger(
    logger_name="HighVolumeApp",
    level="WARNING",  # 只记录重要日志
    log_dir="logs",
    file_output=True
).get_logger()

# 日志会自动轮转，避免文件过大
```

### Q4: 如何在不同进程间共享日志？

**A**: 使用文件输出和进程安全的配置：
```python
logger = CustomLogger(
    logger_name="MultiProcessApp",
    level="INFO",
    log_dir="logs",
    console_output=False,  # 避免控制台冲突
    file_output=True       # 使用文件输出
).get_logger()
```

### Q5: 如何捕获第三方库的日志？

**A**: 使用print重定向和异常捕获：
```python
logger = CustomLogger(
    logger_name="ThirdPartyApp",
    level="INFO",
    console_output=True,
    capture_print=True  # 捕获print输出
).get_logger()

# 第三方库的print输出会被捕获
import some_third_party_library
some_third_party_library.some_function()
```

### Q6: 如何调试日志配置问题？

**A**: 使用简单的配置进行测试：
```python
# 最简单的配置
logger = CustomLogger(
    logger_name="DebugApp",
    level="DEBUG",
    console_output=True,
    file_output=False
).get_logger()

# 逐步添加功能
logger.info("测试基本功能")
logger.debug("测试调试级别")
```

### Q7: 如何优化日志性能？

**A**: 使用适当的配置：
```python
logger = CustomLogger(
    logger_name="PerformanceApp",
    level="INFO",  # 避免过多DEBUG日志
    console_output=True,
    file_output=False,  # 如果不需要文件输出
    capture_print=False  # 如果不需要捕获print
).get_logger()
```

### Q8: 如何处理日志文件权限问题？

**A**: 确保目录权限正确：
```python
import os

# 确保日志目录存在且有写权限
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

logger = CustomLogger(
    logger_name="PermissionApp",
    log_dir=log_dir,
    file_output=True
).get_logger()
```

## 📊 性能基准

| 操作 | 日志数量 | 时间 | 内存使用 |
|------|----------|------|----------|
| 控制台输出 | 10,000条 | ~2s | ~10MB |
| 文件输出 | 10,000条 | ~5s | ~20MB |
| Print重定向 | 10,000条 | ~3s | ~15MB |
| 异常捕获 | 1,000次 | ~1s | ~5MB |

## 🔗 相关链接

- [Loguru文档](https://loguru.readthedocs.io/)
- [Python日志最佳实践](https://docs.python.org/3/howto/logging.html)
- [多进程日志处理](https://docs.python.org/3/library/multiprocessing.html)

---


如有问题，请查看 [常见问题](#常见问题) 或提交 [Issue](https://github.com/SindreYang/sindre/issues)

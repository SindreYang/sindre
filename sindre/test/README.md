# Sindre库测试套件

本目录包含sindre库的完整测试套件，基于pytest框架构建。

## 测试文件结构

```
sindre/test/
├── test_general.py      # general模块测试（日志系统等）
├── test_lmdb.py         # lmdb模块测试（数据库操作）
├── test_report.py       # report模块测试（报告生成）
├── test_utils3d.py      # utils3d模块测试（3D工具）
├── test_win_tools.py    # win_tools模块测试（Windows工具）
├── test_deploy.py       # deploy模块测试（部署相关）
├── test_all.py          # 完整测试套件
├── pytest.ini          # pytest配置文件
├── README.md           # 本文件
└── data/               # 测试数据目录
```

## 运行测试

### 1. 安装依赖

```bash
# 安装pytest
pip install pytest

# 安装可选依赖（用于覆盖率测试）
pip install pytest-cov

# 安装可选依赖（用于并行测试）
pip install pytest-xdist
```

### 2. 运行所有测试

```bash
# 在项目根目录运行
pytest sindre/test/

# 或者在test目录中运行
cd sindre/test
pytest
```

### 3. 运行特定模块测试

```bash
# 只运行general模块测试
pytest sindre/test/test_general.py

# 只运行lmdb模块测试
pytest sindre/test/test_lmdb.py

# 只运行report模块测试
pytest sindre/test/test_report.py
```

### 4. 运行完整测试套件

```bash
# 运行完整测试套件（包含详细输出）
python sindre/test/test_all.py
```

## 测试选项

### 基本选项

```bash
# 详细输出
pytest -v

# 显示最少的输出
pytest -q

# 显示本地变量
pytest -l

# 在第一个失败后停止
pytest -x

# 显示最长的测试
pytest --durations=10
```

### 标记过滤

```bash
# 只运行Windows特定测试
pytest -m windows

# 跳过慢速测试
pytest -m "not slow"

# 只运行单元测试
pytest -m unit

# 只运行集成测试
pytest -m integration
```

### 覆盖率测试

```bash
# 生成覆盖率报告
pytest --cov=sindre --cov-report=html

# 在终端显示覆盖率
pytest --cov=sindre --cov-report=term-missing

# 生成XML格式的覆盖率报告
pytest --cov=sindre --cov-report=xml
```

### 并行测试

```bash
# 使用所有CPU核心并行运行测试
pytest -n auto

# 使用指定数量的进程
pytest -n 4
```

## 测试标记说明

- `@pytest.mark.slow`: 标记为慢速测试
- `@pytest.mark.windows`: 仅在Windows平台上运行
- `@pytest.mark.linux`: 仅在Linux平台上运行
- `@pytest.mark.macos`: 仅在macOS平台上运行
- `@pytest.mark.integration`: 集成测试
- `@pytest.mark.unit`: 单元测试
- `@pytest.mark.smoke`: 冒烟测试

## 平台兼容性

### Windows平台
- 所有测试都可以运行
- Windows特定功能（win_tools）会进行完整测试

### Linux/macOS平台
- 大部分测试可以运行
- Windows特定功能会被跳过
- 某些依赖密集型模块（如utils3d）可能被跳过

## 测试数据

测试数据存储在 `data/` 目录中，包括：
- 示例LMDB数据库文件
- 测试图像文件
- 3D模型文件
- 配置文件

## 故障排除

### 常见问题

1. **导入错误**
   ```bash
   # 确保在正确的Python环境中
   python -c "import sindre"
   ```

2. **依赖缺失**
   ```bash
   # 安装所有依赖
   pip install -r requirements.txt
   ```

3. **权限问题**
   ```bash
   # 确保有写入权限
   chmod -R 755 sindre/test/
   ```

4. **路径问题**
   ```bash
   # 确保在正确的目录中运行
   cd /path/to/sindre_library_private
   ```

### 调试测试

```bash
# 使用pdb调试器
pytest --pdb

# 在失败时进入pdb
pytest --pdbcls=IPython.terminal.debugger:Pdb

# 显示完整的错误信息
pytest --tb=long
```

## 持续集成

测试套件支持持续集成环境：

```yaml
# GitHub Actions示例
- name: Run tests
  run: |
    pip install pytest pytest-cov
    pytest sindre/test/ --cov=sindre --cov-report=xml
```

## 贡献指南

添加新测试时请遵循以下规范：

1. 测试文件命名：`test_模块名.py`
2. 测试类命名：`Test类名`
3. 测试方法命名：`test_功能描述`
4. 使用中文注释说明测试目的
5. 包含适当的setup和teardown方法
6. 使用临时文件和目录进行测试
7. 添加适当的错误处理测试

## 测试覆盖率目标

- 总体覆盖率：> 80%
- 核心模块覆盖率：> 90%
- 关键功能覆盖率：> 95%

## 联系方式

如有测试相关问题，请提交Issue或联系维护者。 
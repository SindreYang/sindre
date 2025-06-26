"""
sindre.report模块测试用例
测试Report类的各种功能
"""

import pytest
from PIL import Image
from sindre.report.report import Report
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

class TestReport:
    """测试Report类的各种功能"""
    
    def test_report_init(self):
        """测试Report初始化"""
        report = Report()
        assert report.data["testPass"] == 0
        assert report.data["testFail"] == 0
        assert report.data["testSkip"] == 0
        assert report.data["testAll"] == 0
        assert "testResult" in report.data
        assert "beginTime" in report.data
        assert "totalTime" in report.data
    
    def test_append_row_success(self):
        """测试添加成功测试行"""
        report = Report()
        row = {
            "className": "TestClass",
            "methodName": "test_success",
            "description": "成功方法",
            "spendTime": "0.1 s",
            "status": "成功",
            "log": ["测试通过"]
        }
        
        report.append_row(row)
        
        assert len(report.data["testResult"]) == 1
        assert report.data["testResult"][0]["status"] == "成功"
    
    def test_append_row_failure(self):
        """测试添加失败测试行"""
        report = Report()
        row = {
            "className": "TestClass",
            "methodName": "test_failure",
            "description": "失败方法",
            "spendTime": "0.2 s",
            "status": "失败",
            "log": ["测试失败", "断言错误"]
        }
        
        report.append_row(row)
        
        assert len(report.data["testResult"]) == 1
        assert report.data["testResult"][0]["status"] == "失败"
    
    def test_append_row_skip(self):
        """测试添加跳过测试行"""
        report = Report()
        row = {
            "className": "TestClass",
            "methodName": "test_skip",
            "description": "跳过方法",
            "spendTime": "0.0 s",
            "status": "跳过",
            "log": ["测试被跳过"]
        }
        
        report.append_row(row)
        
        assert len(report.data["testResult"]) == 1
        assert report.data["testResult"][0]["status"] == "跳过"
    
    def test_write_report(self):
        """测试生成HTML报告，写入data目录"""
        report = Report()
        test_rows = [
            {"className": "TestClass1", "methodName": "test_success_1", "description": "成功的测试1", "spendTime": "0.1 s", "status": "成功", "log": ["测试通过"]},
            {"className": "TestClass1", "methodName": "test_failure_1", "description": "失败的测试1", "spendTime": "0.2 s", "status": "失败", "log": ["测试失败"]},
            {"className": "TestClass2", "methodName": "test_skip_1", "description": "跳过的测试1", "spendTime": "0.0 s", "status": "跳过", "log": ["测试跳过"]}
        ]
        for row in test_rows:
            report.append_row(row)
        report_path = os.path.join(DATA_DIR, "test_report")
        os.makedirs(report_path, exist_ok=True)
        report.write(str(report_path))
        report_file = os.path.join(report_path, "测试报告.html")
        assert os.path.exists(report_file)
        with open(report_file, 'r', encoding='utf-8') as f:
            content = f.read()
            assert "测试报告" in content
            assert "test_success_1" in content
            assert "test_failure_1" in content
            assert "test_skip_1" in content
        # 测试后清理
        os.remove(report_file)
        os.rmdir(report_path)
    
    def test_pil_to_b64(self):
        """测试PIL图像转Base64功能"""
        img = Image.new('RGB', (100, 100), color='red')
        
        b64_str = Report.PIL_To_B64(img)
        
        assert b64_str.startswith('<img src="data:image/png;base64,')
        assert b64_str.endswith('">')
        assert len(b64_str) > 100


if __name__ == "__main__":
    pytest.main([__file__]) 
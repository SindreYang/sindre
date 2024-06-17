
__author__ = 'sindre'

import os
import base64
import io
import json
import time


class Report:
    def __init__(self):
        self.data = {
            "testPass": 0,
            "testResult": [],
            "testName": "测试报告",
            "testAll": 0,
            "testFail": 0,
            "beginTime": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
            "totalTime": "",
            "testSkip": 0,
        }
        self.file_path = os.path.dirname(__file__)

    def append_row(self, row: dict):
        # 添加行内容
        # {
        #     "className": "UnittestCaseSecond",
        #     "methodName": "test_is_none",
        #     "description": "\n            test None object\n        :return:\n        ",
        #     "spendTime": "0.0 s",
        #     "status": "成功",
        #     "log": [
        #         "<br></br>",
        #         ""
        #     ]
        # }

        self.data["testResult"].append(row)

    def write(self, path="./"):
        self.data["testAll"] = len(self.data["testResult"])
        totalTime = 0
        for i in self.data["testResult"]:
            totalTime += i["spendTime"]
            if i["status"] == "成功":
                self.data["testPass"] += 1
            if i["status"] == "失败":
                self.data["testFail"] += 1
            if i["status"] == "跳过":
                self.data["testSkip"] += 1

        self.data["totalTime"] = f"{totalTime} 秒"

        with open(os.path.join(self.file_path, "template"), 'rb') as file:
            body = file.readlines()
        with open(os.path.join(path, "测试报告.html"), 'wb') as write_file:
            for item in body:
                if item.strip().startswith(b'var resultData'):
                    head = '    var resultData = '
                    item = item.decode().split(head)
                    item[1] = head + json.dumps(self.data, ensure_ascii=False, indent=4)
                    item = ''.join(item).encode()
                    item = bytes(item) + b';\n'
                write_file.write(item)

    @staticmethod
    def PIL_To_B64(data):
        buffer = io.BytesIO()
        data.save(buffer, format="png")
        img_b = buffer.getvalue()
        base64_str = base64.b64encode(img_b).decode("utf-8")
        return f'<img src="data:image/png;base64,{base64_str}">'.strip()

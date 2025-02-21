# -*- coding: UTF-8 -*-
import sys
if sys.platform.lower() == "win32":
    try:
        # 尝试导入 Windows 相关工具模块
        import sindre.win_tools.taskbar as taskbar
        import sindre.win_tools.tools as tools
    except ImportError:
        # 若导入模块时出现错误，捕获异常并输出错误信息
        print("注意：导入 Windows 工具时出错,请 pip install pywin32")
else:
    # 若当前系统不是 Windows，输出相应提示信息
    print("注意：当前系统不是 Windows,无法加载 Windows 工具.")
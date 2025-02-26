# -*- coding: UTF-8 -*-
import sys

if sys.platform.lower() == "win32":
    try:
        # 尝试导入 Windows 相关工具模块
        import sindre.win_tools.tools as tools
        import sindre.win_tools.taskbar as taskbar
    except ImportError:
        pass

# else:
#     # 若当前系统不是 Windows，输出相应提示信息
#     print("注意：当前系统不是 Windows,无法加载 Windows 工具.")
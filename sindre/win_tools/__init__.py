# -*- coding: UTF-8 -*-
import sys

if sys.platform.lower() == "win32":
    try:
        # 尝试导入 Windows 相关工具模块
        import sindre.win_tools.tools as tools
        import sindre.win_tools.taskbar as taskbar
    except ImportError:
        pass


# -*- coding: UTF-8 -*-



try:
    import win32gui
    import win32con
except ImportError:
    # 若导入模块时出现错误，捕获异常并输出错误信息
    print("注意：导入 Windows 工具时出错,请 pip install pywin32")


def HEXtoRGBAint(HEX: str):
    alpha = HEX[7:]
    blue = HEX[5:7]
    green = HEX[3:5]
    red = HEX[1:3]
    gradientColor = alpha + blue + green + red
    return int(gradientColor, base=16)


def get_windows_child(hWnd):

    # 获取所有子窗口
    hwndChildList = []

    win32gui.EnumChildWindows(hWnd, lambda hwnd, param: param.append(hwnd), hwndChildList)

    for hwnd in hwndChildList:
        print("句柄：", hwnd, "标题：", win32gui.GetWindowText(hwnd))
    return hwndChildList


def set_windows_alpha(alpha: int=255,class_name:str="Shell_TrayWnd"):
    """
      通过查找class_name,强制用于设置任务栏透明程度


      Args:
          alpha: 透明度，(0--完全透明，255--完全不透明）
          class_name: 窗口名

    """
    # 假设 hwnd 是你想要设置透明度的窗口句柄
    hWnd = win32gui.FindWindow(class_name, None)
    ex_style = win32gui.GetWindowLong(hWnd, win32con.GWL_EXSTYLE)
    # 设置为分层窗口并设置半透明
    ex_style |= win32con.WS_EX_LAYERED
    win32gui.SetWindowLong(hWnd, win32con.GWL_EXSTYLE, ex_style)
    win32gui.SetLayeredWindowAttributes(hWnd, 0, alpha, win32con.LWA_ALPHA)



"""Windows 環境感知：其他視窗的頂邊（讓寵物跳上去）、使用者閒置時間。

非 Windows 系統時全部回傳空值，寵物仍可在螢幕底部活動。
"""
from __future__ import annotations

import os
import sys

IS_WIN = sys.platform == "win32"

if IS_WIN:
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    dwmapi = ctypes.windll.dwmapi
    kernel32 = ctypes.windll.kernel32

    class RECT(ctypes.Structure):
        _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long),
                    ("right", ctypes.c_long), ("bottom", ctypes.c_long)]

    class MONITORINFOEXW(ctypes.Structure):
        _fields_ = [("cbSize", wintypes.DWORD), ("rcMonitor", RECT), ("rcWork", RECT),
                    ("dwFlags", wintypes.DWORD), ("szDevice", wintypes.WCHAR * 32)]

    class LASTINPUTINFO(ctypes.Structure):
        _fields_ = [("cbSize", wintypes.UINT), ("dwTime", wintypes.DWORD)]

    WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    MONITORENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HMONITOR, wintypes.HDC,
                                         ctypes.POINTER(RECT), wintypes.LPARAM)
    user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
    for _f in (user32.IsWindowVisible, user32.IsIconic, user32.IsZoomed):
        _f.argtypes = [wintypes.HWND]
    user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(RECT)]
    kernel32.GetTickCount.restype = wintypes.DWORD
    user32.GetClassNameW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    user32.GetMonitorInfoW.argtypes = [wintypes.HMONITOR, ctypes.POINTER(MONITORINFOEXW)]
    dwmapi.DwmGetWindowAttribute.argtypes = [wintypes.HWND, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD]

    DWMWA_EXTENDED_FRAME_BOUNDS = 9
    DWMWA_CLOAKED = 14
    SKIP_CLASSES = {"Progman", "WorkerW", "Shell_TrayWnd", "Shell_SecondaryTrayWnd",
                    "Windows.UI.Core.CoreWindow", "NotifyIconOverflowWindow", "TopLevelWindowForOverflowXamlIsland",
                    "XamlExplorerHostIslandWindow", "ForegroundStaging", "MultitaskingViewFrame"}
    _MY_PID = os.getpid()


def idle_seconds() -> float:
    """使用者多久沒動滑鼠鍵盤了。"""
    if not IS_WIN:
        return 0.0
    li = LASTINPUTINFO()
    li.cbSize = ctypes.sizeof(li)
    if not user32.GetLastInputInfo(ctypes.byref(li)):
        return 0.0
    return max(0, (kernel32.GetTickCount() - li.dwTime) & 0xFFFFFFFF) / 1000.0


def monitors() -> list[tuple[str, tuple[int, int, int, int]]]:
    """[(裝置名稱, 實體像素矩形)]"""
    if not IS_WIN:
        return []
    out = []

    def cb(hmon, hdc, lprect, lparam):
        mi = MONITORINFOEXW()
        mi.cbSize = ctypes.sizeof(mi)
        if user32.GetMonitorInfoW(hmon, ctypes.byref(mi)):
            r = mi.rcMonitor
            out.append((mi.szDevice, (r.left, r.top, r.right, r.bottom)))
        return True

    user32.EnumDisplayMonitors(None, None, MONITORENUMPROC(cb), 0)
    return out


def top_windows() -> list[tuple[int, tuple[int, int, int, int], bool]]:
    """可見的一般視窗 [(hwnd, 實體像素矩形, 是否最大化)]，依 Z 順序（最上層在前）。"""
    if not IS_WIN:
        return []
    out = []
    buf = ctypes.create_unicode_buffer(256)

    def cb(hwnd, lparam):
        try:
            if not user32.IsWindowVisible(hwnd) or user32.IsIconic(hwnd):
                return True
            pid = wintypes.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            if pid.value == _MY_PID:
                return True
            cloaked = ctypes.c_int(0)
            dwmapi.DwmGetWindowAttribute(hwnd, DWMWA_CLOAKED, ctypes.byref(cloaked), ctypes.sizeof(cloaked))
            if cloaked.value:
                return True
            user32.GetClassNameW(hwnd, buf, 256)
            if buf.value in SKIP_CLASSES:
                return True
            r = RECT()
            if dwmapi.DwmGetWindowAttribute(hwnd, DWMWA_EXTENDED_FRAME_BOUNDS, ctypes.byref(r), ctypes.sizeof(r)):
                user32.GetWindowRect(hwnd, ctypes.byref(r))
            w, h = r.right - r.left, r.bottom - r.top
            if w < 120 or h < 60:
                return True
            # 有標題的視窗才算（排除各種隱形的工具視窗）；最大化的也算，但頂邊會被當成螢幕頂端過濾掉
            if user32.GetWindowTextLengthW(hwnd) == 0:
                return True
            out.append((int(hwnd or 0), (r.left, r.top, r.right, r.bottom), bool(user32.IsZoomed(hwnd))))
        except Exception:
            pass
        return True

    user32.EnumWindows(WNDENUMPROC(cb), 0)
    return out

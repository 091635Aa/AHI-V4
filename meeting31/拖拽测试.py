# -*- coding: utf-8 -*-
"""meeting31/拖拽测试.py —— #290 连贯性操作测试 v2：单窗口内多选拖拽。

所有者测试设计：『左键拖拽选择文件，拖到目标——流畅不中断，两个拖拽步骤，
多选拖拽』。v1 双 explorer 窗口失败教训：Win11 explorer 单进程多标签合并+
路径漂移——v2 改单窗口内拖拽（源目录 3 文件+1 目标子文件夹，同屏）。
判读：①目标子文件夹文件数递增（文件系统真值）②鼠标轨迹监听（200ms 采样）：
拖拽期间轨迹连续（瞬移次数）。
"""
from __future__ import annotations

import ctypes
import json
import os
import shutil
import subprocess
import threading
import time

from ctypes import wintypes

HERE = os.path.dirname(os.path.abspath(__file__))
沙盒 = os.path.join(HERE, "拖拽沙盒")
源 = os.path.join(沙盒, "源文件夹")
目标 = os.path.join(源, "目标文件夹")
轨迹路径 = os.path.join(HERE, "拖拽轨迹.jsonl")
日志 = os.path.join(HERE, "拖拽测试.log")

import pyautogui
import pyperclip

pyautogui.FAILSAFE = False
监听中 = threading.Event()
轨迹: list = []


def 记(s: str):
    行 = "[%s] %s" % (time.strftime("%H:%M:%S"), s)
    print(行, flush=True)
    with open(日志, "a", encoding="utf-8") as f:
        f.write(行 + "\n")


def 监听线程():
    user32 = ctypes.windll.user32
    while True:
        pt = wintypes.POINT()
        user32.GetCursorPos(ctypes.byref(pt))
        按下 = bool(user32.GetKeyState(0x01) & 0x8000)
        轨迹.append(dict(t=round(time.time(), 3), x=pt.x, y=pt.y, 按下=按下))
        with open(轨迹路径, "a", encoding="utf-8") as f:
            f.write(json.dumps(轨迹[-1]) + "\n")
        if not 监听中.is_set() and 轨迹 and time.time() - 轨迹[-1]["t"] > 4:
            break
        time.sleep(0.2)


def 找窗口(标题关键词):
    user32 = ctypes.windll.user32
    命中 = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def cb(hwnd, lp):
        if user32.IsWindowVisible(hwnd):
            n = user32.GetWindowTextLengthW(hwnd)
            if n > 0:
                buf = ctypes.create_unicode_buffer(n + 1)
                user32.GetWindowTextW(hwnd, buf, n + 1)
                if 标题关键词 in buf.value:
                    命中.append((hwnd, buf.value))
        return True
    user32.EnumWindows(cb, 0)
    return 命中


def 拖拽(x1, y1, x2, y2, 悬停=0.9):
    """连贯拖拽：dragTo（官方拖放时序，实测兼容 explorer OLE 拖放）。"""
    pyautogui.moveTo(x1, y1)
    time.sleep(0.2)
    pyautogui.dragTo(x2, y2, duration=1.2, button="left")
    time.sleep(悬停)


def 导航目录(路径: str):
    """地址栏导航：剪贴板粘贴原样反斜杠路径（正斜杠/双斜杠会让 explorer 报
    『Windows 找不到』——实测弹窗实锤的 bug）。"""
    pyautogui.hotkey("ctrl", "l")
    time.sleep(0.3)
    pyperclip.copy(路径)
    pyautogui.hotkey("ctrl", "v")
    time.sleep(0.2)
    pyautogui.press("enter")
    time.sleep(1.5)


def 自动校准行():
    """终结坐标猜测：逐 y 试探点击+前后差分检测选中高亮（不预设颜色——暗色/亮色
    主题通用）。返回按 y 排序的 4 行 [甲, 乙, 丙, 目标] 坐标。"""
    import numpy as np
    行s = []
    y = 110
    前 = None
    while y < 700 and len(行s) < 4:
        pyautogui.moveTo(300, y)
        time.sleep(0.1)
        前 = np.asarray(pyautogui.screenshot().crop((150, y - 24, 800, y + 24))
                        .convert("L"), dtype=np.float32)
        pyautogui.click(300, y)
        time.sleep(0.3)
        后 = np.asarray(pyautogui.screenshot().crop((150, y - 24, 800, y + 24))
                        .convert("L"), dtype=np.float32)
        差 = np.abs(后 - 前).mean()
        if 差 > 3.0:                       # 点击引发行高亮=命中行
            行s.append(y)
            记("  校准命中 y=%d (亮度差 %.1f)" % (y, 差))
            y += 20
        else:
            y += 10
    if len(行s) != 4:
        return None
    return [(300, yy) for yy in 行s]


def 主():
    # 0) 沙盒：源目录=3 文件+1 目标子文件夹（同屏）
    shutil.rmtree(沙盒, ignore_errors=True)
    os.makedirs(目标, exist_ok=True)
    for i, 名 in enumerate(("测试文件甲.txt", "测试文件乙.txt", "测试文件丙.txt")):
        with open(os.path.join(源, 名), "w", encoding="utf-8") as f:
            f.write("拖拽测试文件 %d\n" % i * 20)
    记("沙盒就绪：源 3 文件+目标子文件夹（单窗口内拖拽）")

    # 1) explorer 打开源目录（单窗口）
    subprocess.Popen(["explorer", 源])
    time.sleep(3)
    w源 = 找窗口("源文件夹")
    user32 = ctypes.windll.user32
    if w源:
        user32.ShowWindow(w源[0][0], 9)
        user32.MoveWindow(w源[0][0], 0, 0, 1200, 800, True)
        user32.SwitchToThisWindow(w源[0][0], True)
        time.sleep(0.6)
        导航目录(源)
        pyautogui.click(600, 400)          # 聚焦内容视图
        time.sleep(0.4)
    记("窗口就位（最大化内容视图，4 行：甲/乙/丙/目标文件夹）")

    # 2) 监听启动
    监听中.set()
    t = threading.Thread(target=监听线程, daemon=True)
    t.start()

    # 详细信息视图行位置（实证汇总：乙=178 成功拖动、目标=266 成功拖放 → 行高 28）
    行甲, 行乙, 行丙, 行目标 = (230, 150), (230, 178), (230, 206), (230, 266)

    # 3) 拖拽一：先点选中甲 → F5 确保视图与文件系统同步 → 拖到目标文件夹
    pyautogui.press("f5")                  # explorer 视图刷新（防视图/文件系统不同步）
    time.sleep(1.2)
    pyautogui.click(行甲[0], 行甲[1])       # 选中甲（模仿手动成功流程）
    time.sleep(0.4)
    记("拖拽一：甲 %s → 目标文件夹 %s" % (行甲, 行目标))
    拖拽(行甲[0], 行甲[1], 行目标[0], 行目标[1])
    time.sleep(2)
    n目标1 = len(os.listdir(目标))
    记("拖拽一后 目标文件夹 %d 个文件" % n目标1)
    步骤一 = n目标1 == 1

    # 4) 拖拽二：多选（乙+丙 Ctrl+点击）→ 拖到目标文件夹
    pyautogui.click(行乙[0], 行乙[1])
    time.sleep(0.3)
    pyautogui.keyDown("ctrl")
    pyautogui.click(行丙[0], 行丙[1])
    time.sleep(0.2)
    pyautogui.keyUp("ctrl")
    time.sleep(0.3)
    记("拖拽二：多选 乙+丙 → 目标文件夹")
    pyautogui.moveTo(行乙[0], 行乙[1])
    pyautogui.mouseDown()
    for i in range(1, 17):
        pyautogui.moveTo(行乙[0] + (行目标[0] - 行乙[0]) * i // 16,
                         行乙[1] + (行目标[1] - 行乙[1]) * i // 16,
                         duration=0.05)
        time.sleep(0.03)
    time.sleep(0.9)
    pyautogui.mouseUp()
    time.sleep(2)
    n目标2 = len(os.listdir(目标))
    记("拖拽二后 目标文件夹 %d 个文件（应为 3）" % n目标2)
    步骤二 = n目标2 == 3 and len(os.listdir(源)) == 1   # 源只剩目标文件夹

    # 5) 轨迹分析
    拖拽段s = []
    按住段 = None
    for i, p in enumerate(轨迹):
        if p["按下"] and 按住段 is None:
            按住段 = i
        elif not p["按下"] and 按住段 is not None:
            拖拽段s.append(轨迹[按住段:i + 1])
            按住段 = None
    分析 = []
    for k, 段 in enumerate(拖拽段s):
        瞬移 = sum(1 for a, b in zip(段, 段[1:])
                   if abs(b["x"] - a["x"]) + abs(b["y"] - a["y"]) > 120)
        分析.append(dict(段=k, 采样点=len(段), 瞬移=瞬移,
                         起点=(段[0]["x"], 段[0]["y"]),
                         终点=(段[-1]["x"], 段[-1]["y"])))
    判一 = "PASS" if 步骤一 else "FAIL"
    判二 = "PASS" if 步骤二 else "FAIL"
    记("==== 判读 ====")
    记("拖拽一（单文件→目标文件夹）：%s ｜ 拖拽二（多选→目标文件夹）：%s"
       % (判一, 判二))
    for a in 分析:
        记("按住段 %s：采样 %d 点、瞬移 %d 次" % (a["段"], a["采样点"], a["瞬移"]))
    监听中.clear()
    with open(os.path.join(HERE, "拖拽测试_结果.json"), "w",
              encoding="utf-8") as f:
        json.dump(dict(拖拽一=判一, 拖拽二=判二, 轨迹分析=分析),
                  f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    主()

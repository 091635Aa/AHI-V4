# -*- coding: utf-8 -*-
"""meeting32/probe_基础真实.py —— 基础能力真实迁移：滚轮 + 中键（物理机实测）。

判据（先定）：
  R1 滚轮下：视图内容变化（差分 ≥5000px）
  R2 滚轮上：视图恢复（差分回落）
  R3 中键：新窗口/标签出现（新差分块）
"""
from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys
import time

from ctypes import wintypes

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "meeting31"))

import numpy as np
import pyautogui
import cv2

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0


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
                    命中.append(hwnd)
        return True
    user32.EnumWindows(cb, 0)
    return 命中


def 截图():
    return np.asarray(pyautogui.screenshot().crop((0, 100, 900, 700)),
                      dtype=np.int16)


def 存图(img, 名):
    cv2.imencode(".png", cv2.cvtColor(
        img.astype(np.uint8), cv2.COLOR_RGB2BGR))[1].tofile(
        os.path.join(HERE, "实录", 名 + ".png"))


def 差块(a, b, 阈=5000):
    d = (np.abs(b - a).sum(axis=2) > 60).astype(np.uint8)
    n, _, stats, cent = cv2.connectedComponentsWithStats(d)
    return [(int(stats[i, 4]), int(cent[i][0]), int(cent[i][1]))
            for i in range(1, n) if stats[i, 4] >= 阈]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    源 = os.path.join(HERE, "explorer沙盒", "源文件夹")
    shutil.rmtree(源, ignore_errors=True)
    os.makedirs(源, exist_ok=True)
    for i in range(45):                              # 45 个文件强制出滚动条
        with open(os.path.join(源, "滚动文件%02d.txt" % i), "w",
                  encoding="utf-8") as f:
            f.write("x" * 50)
    subprocess.Popen(["explorer", 源])
    time.sleep(3)
    hwnds = 找窗口("源文件夹")
    user32 = ctypes.windll.user32
    if hwnds:
        user32.MoveWindow(hwnds[0], 0, 0, 900, 700, True)
        user32.SwitchToThisWindow(hwnds[0], True)
    time.sleep(1.5)
    pyautogui.moveTo(450, 400)                       # 进列表区
    等秒 = time.sleep

    前 = 截图()
    存图(前, "滚前")

    def 滚(格数):
        """ctypes 直发 WHEEL_DELTA（pyautogui.scroll 实测发原始单位太小无效）"""
        ctypes.windll.user32.mouse_event(0x0800, 0, 0, int(-120 * 格数), 0)

    滚(3)                                            # 滚轮下（内容上移）
    等秒(1)
    后下 = 截图()
    存图(后下, "滚后")
    R1 = bool(差块(前, 后下))
    滚(-3)                                           # 滚轮上（正 delta）
    滚(-3)
    等秒(1)
    后上 = 截图()
    R2 = (np.abs(后上 - 前).sum() / 1e6) < (np.abs(后下 - 前).sum() / 1e6)
    print("[R1 滚轮下视图变化] %s" % ("PASS" if R1 else "FAIL"))
    print("[R2 滚轮上恢复] %s" % ("PASS" if R2 else "FAIL"))

    # 中键：点击第一个文件行 → 新窗口/标签（新差分块）
    pyautogui.moveTo(200, 250)
    pyautogui.click(button="middle")
    等秒(2)
    后中 = 截图()
    块 = 差块(后下, 后中, 8000)
    R3 = bool(块)
    print("[R3 中键新窗/标签] %s %s" % ("PASS" if R3 else "FAIL", 块[:2]))
    pyautogui.hotkey("alt", "f4")
    判 = R1 and R2 and R3
    print("==== 基础真实迁移：%s ====" % ("PASS" if 判 else "FAIL"))
    return 0 if 判 else 1


if __name__ == "__main__":
    sys.exit(main())

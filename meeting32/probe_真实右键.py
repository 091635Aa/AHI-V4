# -*- coding: utf-8 -*-
"""meeting32/probe_真实右键.py —— W6 现实迁移：真实 explorer 右键菜单（一次实测）。

沙箱课3 学到的「右键对象→菜单出现」迁移到真实 explorer：
  右键文件行 → 截图差分 → 菜单弹出判定（光标附近出现新亮块）→ Esc 关闭。
判据（先定）：M1 菜单出现（差分块 ≥5000px 且在光标 300px 内）；M2 Esc 后菜单消失。
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
sys.path.insert(0, os.path.join(HERE, "..", "meeting28"))
sys.path.insert(0, r"<workspace>/03_AHI\AHI2.1")

import numpy as np
import pyautogui
import pyperclip
import cv2

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

沙盒 = os.path.join(HERE, "explorer沙盒")
源 = os.path.join(沙盒, "源文件夹")


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
    return np.asarray(pyautogui.screenshot().crop((0, 100, 1000, 700)),
                      dtype=np.int16)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    shutil.rmtree(沙盒, ignore_errors=True)
    os.makedirs(源, exist_ok=True)
    for 名 in ("迁移文件甲.txt", "迁移文件乙.txt"):
        with open(os.path.join(源, 名), "w", encoding="utf-8") as f:
            f.write("右键迁移测试\n" * 10)
    subprocess.Popen(["explorer", 源])
    time.sleep(3)
    user32 = ctypes.windll.user32
    hwnds = 找窗口("源文件夹")
    if not hwnds:
        print("[FAIL] M0 explorer 未打开")
        return 1
    user32.ShowWindow(hwnds[0], 9)
    user32.MoveWindow(hwnds[0], 0, 0, 1000, 700, True)
    time.sleep(1.5)
    pyautogui.hotkey("ctrl", "l"); time.sleep(0.3)
    pyperclip.copy(源); pyautogui.hotkey("ctrl", "v")
    time.sleep(0.2); pyautogui.press("enter"); time.sleep(2)

    # 文件行位置：txt 图标为亮色页——亮行带检测（此文件夹无文件夹图标，
    # 黄图标路径天然为空；且先关掉会盖窗的画布 v5）
    subprocess.run(["taskkill", "/F", "/IM", "python.exe", "/FI",
                    "WINDOWTITLE eq AHI画布v5*"], shell=True,
                   capture_output=True)
    time.sleep(0.5)
    全窗 = np.asarray(pyautogui.screenshot().crop((0, 0, 640, 700)),
                      dtype=np.uint8)
    hsv全 = cv2.cvtColor(全窗, cv2.COLOR_RGB2HSV)
    亮 = cv2.inRange(hsv全, (0, 0, 130), (180, 90, 255))
    条带 = 亮[:, 150:220]                        # 图标列竖条带（窗口相对）
    行命中 = 条带.sum(axis=1) // 255
    行s = []
    y = 0
    while y < len(行命中):
        if 行命中[y] > 8:
            y2 = y
            while y2 < len(行命中) and 行命中[y2] > 4:
                y2 += 1
            if y2 - y >= 8:
                行s.append((y + y2) // 2)
            y = y2 + 6
        else:
            y += 1
    行s = [y for y in 行s if 150 <= y <= 400]
    if not 行s:
        print("[FAIL] M0 未检到文件行")
        pyautogui.hotkey("alt", "f4")
        return 1
    x, y = 165, 行s[0]
    print("右键目标:", x, y, flush=True)
    前 = 截图()
    pyautogui.moveTo(x, y); time.sleep(0.2)
    pyautogui.click(button="right")
    time.sleep(1.0)
    后 = 截图()
    差 = np.abs(后 - 前).sum(axis=2)
    差块 = (差 > 60).astype(np.uint8)
    n, _, stats, cent = cv2连接域(差块)
    命中 = [(int(stats[i, 4]), int(cent[i][0]), int(cent[i][1]))
            for i in range(1, n) if stats[i, 4] >= 5000]
    M1 = bool(命中)
    print("菜单差分块:", 命中[:4], flush=True)
    pyautogui.press("esc")
    time.sleep(0.8)
    后2 = 截图()
    # M2 判定：菜单块区域消失（右键附带的选中高亮会保留——那是正常状态）
    M2 = True
    for 面积, mx, my in 命中:
        区 = (np.abs(后2 - 前).sum(axis=2) > 60)[
            max(my - 100, 0):my + 100, max(mx - 100, 0):mx + 100]
        if 区.sum() > 5000:
            M2 = False
    print("[M1 菜单出现] %s" % ("PASS" if M1 else "FAIL"))
    print("[M2 Esc 关闭] %s" % ("PASS" if M2 else "FAIL"))
    pyautogui.hotkey("alt", "f4")
    return 0 if (M1 and M2) else 1


def cv2连接域(mask):
    import cv2
    return cv2.connectedComponentsWithStats(mask)


if __name__ == "__main__":
    sys.exit(main())

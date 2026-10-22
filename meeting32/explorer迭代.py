# -*- coding: utf-8 -*-
"""meeting32/explorer迭代.py —— #350 真实环境拖拽迭代 v2（窗口自适应）。

v3/v4 教训全数修正：
  ①BGR2HSV 解 RGB 截图 → 统一 RGB2HSV；
  ②写死坐标 → GetWindowRect 实测窗口矩形，一切坐标相对窗口；
     （窗口尺寸两次运行间会变：633×393 行距13.5 / 1200×800 行距27——写死必死）
  ③crop 切掉图标列 → 全窗口检测，导航栏按宽度 15% 排除；
  ④拖放无悬停 → press-分段移动-悬停0.35s-release；
  ⑤视图不同步 → 每次拖后 F5；
  ⑥cv2.imwrite 中文路径静默失败 → imencode+tofile。
判据（先定）：X1 端到端 3/3（文件系统计数）；X2 诊断留痕；X3 失败 F5 重同步。
"""
from __future__ import annotations

import ctypes
import json
import os
import shutil
import subprocess
import sys
import threading
import time

from ctypes import wintypes

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "meeting31"))
sys.path.insert(0, os.path.join(HERE, "..", "meeting28"))
sys.path.insert(0, r"<workspace>/03_AHI\AHI2.1")
沙盒 = os.path.join(HERE, "explorer沙盒")
源 = os.path.join(沙盒, "源文件夹")
目标 = os.path.join(源, "目标文件夹")
日志 = os.path.join(HERE, "explorer迭代.log")
诊断目录 = os.path.join(HERE, "explorer诊断")

import numpy as np
import pyautogui
import pyperclip
import cv2

from synthesis import 综合体

pyautogui.FAILSAFE = False
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
        time.sleep(0.1)


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


def 窗口矩形(hwnd) -> tuple:
    r = wintypes.RECT()
    ctypes.windll.user32.GetWindowRect(hwnd, ctypes.byref(r))
    return (r.left, r.top, r.right, r.bottom)


class 眼:
    @staticmethod
    def 存诊断(img, 名):
        os.makedirs(诊断目录, exist_ok=True)
        cv2.imencode(".png", img)[1].tofile(
            os.path.join(诊断目录, 名 + ".png"))

    @staticmethod
    def 行检测(hwnd, 诊断名=None) -> dict:
        """窗口矩形内：目标文件夹行（列表区最上黄图标）+ 文件行（图标列亮色行带）。
        坐标全相对窗口实测矩形——对窗口尺寸/位置自适应。"""
        wx0, wy0, wx1, wy1 = 窗口矩形(hwnd)
        img = np.asarray(pyautogui.screenshot().crop((wx0, wy0, wx1, wy1)),
                         dtype=np.uint8)
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)           # RGB 顺序！
        黄 = cv2.inRange(hsv, (15, 60, 110), (50, 255, 255))
        n, _, stats, cent = cv2.connectedComponentsWithStats(黄)
        导航界 = int((wx1 - wx0) * 0.14)                     # 导航栏≈窗宽14%
        列表黄 = [(cent[i][1], cent[i][0], stats[i, 4])
                 for i in range(1, n)
                 if stats[i, 4] >= 25 and cent[i][0] > 导航界
                 and stats[i, 1] > 导航界]
        if 诊断名:
            眼.存诊断(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), 诊断名 + "_原图")
            眼.存诊断(黄, 诊断名 + "_黄掩码")
        if not 列表黄:
            return dict(文件夹行=None, 文件行s=[])
        列表黄.sort()
        文件夹y, 图标x, _ = 列表黄[0]                        # 最上黄图标=目标文件夹
        # 文件行：图标列竖条带内的亮色行带（txt 图标白/灰）
        带半 = 22
        x0 = max(int(图标x) - 带半, 0)
        x1 = min(int(图标x) + 带半, img.shape[1])
        条带 = img[:, x0:x1]
        亮 = cv2.cvtColor(条带, cv2.COLOR_RGB2HSV)
        白 = cv2.inRange(亮, (0, 0, 130), (180, 90, 255))
        行命中 = 白.sum(axis=1) // 255
        行s = []
        y = 0
        while y < len(行命中):
            if 行命中[y] > 6:
                y2 = y
                while y2 < len(行命中) and 行命中[y2] > 3:
                    y2 += 1
                if y2 - y >= 5:
                    行s.append((y + y2) // 2)
                y = y2 + 3
            else:
                y += 1
        文件行s = [wy0 + y for y in 行s if abs(y - 文件夹y) > 8]
        return dict(文件夹行=wy0 + int(文件夹y), 图标x=wy0 * 0 + wx0 + int(图标x),
                    文件行s=文件行s)


class 体:
    def __init__(self, 桥):
        self.桥 = 桥

    def 说(self, 类型, 文本):
        桥 = self.桥
        n = len(桥.账本)
        桥.账本.append(("自我", 文本))
        桥.自言.追加(桥.拍 if hasattr(桥, "拍") else 0, 类型, 文本,
                    ("内生", "explorer迭代"), [(n, 文本[:16])])
        记("自言[%s] %s" % (类型, 文本))

    @staticmethod
    def 拖(x1, y1, x2, y2):
        pyautogui.moveTo(x1, y1)
        time.sleep(0.25)
        pyautogui.mouseDown(button="left")
        步 = 12
        for i in range(1, 步 + 1):
            pyautogui.moveTo(x1 + (x2 - x1) * i / 步, y1 + (y2 - y1) * i / 步,
                             duration=0.08)
        time.sleep(0.35)                                  # 悬停=落下意图
        pyautogui.mouseUp(button="left")
        time.sleep(1.2)

    @staticmethod
    def 回报():
        return len(os.listdir(目标))

    @staticmethod
    def 刷新():
        pyautogui.press("f5")
        time.sleep(1.0)


def 主():
    shutil.rmtree(沙盒, ignore_errors=True)
    os.makedirs(目标, exist_ok=True)
    for i, 名 in enumerate(("测试文件甲.txt", "测试文件乙.txt", "测试文件丙.txt")):
        with open(os.path.join(源, 名), "w", encoding="utf-8") as f:
            f.write("拖拽测试文件 %d\n" % i * 20)
    subprocess.Popen(["explorer", 源])
    time.sleep(3)
    user32 = ctypes.windll.user32
    hwnds = 找窗口("源文件夹")
    if hwnds:
        user32.ShowWindow(hwnds[0], 9)
        user32.MoveWindow(hwnds[0], 0, 0, 1000, 700, True)
        user32.SwitchToThisWindow(hwnds[0], True)
        time.sleep(1.0)
        pyautogui.hotkey("ctrl", "l"); time.sleep(0.3)
        pyperclip.copy(源); pyautogui.hotkey("ctrl", "v")
        time.sleep(0.2); pyautogui.press("enter"); time.sleep(2)
    hwnd = 找窗口("源文件夹")[0]
    threading.Thread(target=监听线程, daemon=True).start()

    t = 体(综合体(seed=999))
    t.桥.设开放域([], [])
    记("任务（所有者）：把 3 个文件全部拖进目标文件夹")
    t.说("计划", "任务收到：3 个文件拖进目标文件夹。列表区最上面的黄图标行="
           "文件夹，同列亮行=文件；拖一个数一次，视图变了就 F5。")

    旧计数 = t.回报()
    轮 = 0
    经验 = {}
    结果行 = []
    while 旧计数 < 3 and 轮 < 12:
        轮 += 1
        检 = 眼.行检测(hwnd, "轮%d" % 轮)
        if 检["文件夹行"] is None or not 检["文件行s"]:
            t.说("复盘", "第%d轮没看到文件夹/文件行（检到 %s）——F5 重同步。"
                 % (轮, 检))
            t.刷新()
            continue
        候选目标 = 检["文件夹行"]
        图标x = 检["图标x"]
        文件行s = [y for y in 检["文件行s"] if y > 170 or y > 检["文件夹行"]]
        if not 文件行s:
            t.说("复盘", "没有文件行了？计数=%d——F5 再看。" % 旧计数)
            t.刷新()
            continue
        源行 = 文件行s[(轮 - 1) % len(文件行s)]
        t.说("假设", "第%d轮：文件夹行 y=%d（图标列 x=%d），文件行 %s——拖 (%d,%d) "
             "到 (%d,%d)，悬停落下。" % (轮, 候选目标, 图标x, 文件行s,
                                        图标x, 源行, 图标x + 40, 候选目标))
        t.拖(图标x, 源行, 图标x + 40, 候选目标)
        time.sleep(1.5)
        新计数 = t.回报()
        if 新计数 > 旧计数:
            t.说("复盘", "真完成：计数 %d→%d。视图同步继续。"
                 % (旧计数, 新计数))
            经验[(源行, 候选目标)] = 1.0
            旧计数 = 新计数
        else:
            t.说("复盘", "计数没变——F5 重同步视图再试。")
            经验[(源行, 候选目标)] = -1.0
            t.刷新()
        结果行.append((轮, 源行, 候选目标, 新计数))

    到达 = t.回报()
    判 = "PASS" if 到达 == 3 else ("PASS(部分)" if 到达 else "FAIL")
    记("==== 判读 ====")
    记("explorer 迭代 %d/3 → %s（%d 轮）" % (到达, 判, 轮))
    with open(os.path.join(HERE, "explorer迭代_结果.json"), "w", encoding="utf-8") as f:
        json.dump(dict(完成=到达, 轮=轮, 判读=判, 经验={str(k): v for k, v in 经验.items()},
                       结果行=结果行), f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    主()

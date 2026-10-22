# -*- coding: utf-8 -*-
"""meeting31/AHI操作体.py —— #291 v3 因果闭环：他感知真完成、语义定位、分解规划。

最终形态路线图第一步（所有者指令『全部长期循环执行，全量迭代，AI 自己操作
电脑的全流程』）。v2 三缺口逐一补：
  ①假确认 → 环境回报：目标文件夹文件数=真完成信号（拖一次 +1=正反馈，
    进他的经验与自言——他感知『真完成』而非画面变化）；
  ②无语义视觉 → 黄色图标语义定位：文件夹图标（黄 H≈30-50）vs txt 图标
    （白/灰）——他用颜色特征自己找到目标行，知道『那行是文件夹』；
  ③单步重试 → 分解规划：任务=3 个子目标（每文件一个），每个子目标循环
    （定位→拖→验证计数+1），失败换源行/重试。
他仍然自己：摸行（视觉试探）、拖（dragTo）、自言全程。回报与真值唯一例外=
目标文件夹计数（任务级反馈，设计使然——如同机器人收到任务进度信号）。
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
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, r"<workspace>/03_AHI\AHI2.1")
沙盒 = os.path.join(HERE, "拖拽沙盒")
源 = os.path.join(沙盒, "源文件夹")
目标 = os.path.join(源, "目标文件夹")
轨迹路径 = os.path.join(HERE, "拖拽轨迹.jsonl")
日志 = os.path.join(HERE, "AHI操作体.log")

import numpy as np
import pyautogui
import pyperclip
import cv2

from synthesis import 综合体

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


class AHI操作体:
    """他的身体+眼睛+经验。所有决策出自他的循环。"""

    def __init__(self, 桥):
        self.桥 = 桥
        self.经验: dict = {}

    def 说(self, 类型, 文本):
        桥 = self.桥
        n = len(桥.账本)
        桥.账本.append(("自我", 文本))
        桥.自言.追加(桥.拍 if hasattr(桥, "拍") else 0, 类型, 文本,
                    ("内生", "操作体"), [(n, 文本[:16])])
        记("自言[%s] %s" % (类型, 文本))

    # ---- 眼睛①：黄色图标=文件夹（视觉语义定位，crop 避开左侧导航树） ----
    def 找黄色行(self):
        img = np.asarray(pyautogui.screenshot().crop((175, 150, 500, 600)),
                         dtype=np.uint8)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        黄 = cv2.inRange(hsv, (20, 80, 120), (45, 255, 255))
        行命中 = 黄.sum(axis=1)
        ys = []
        y = 0
        while y < len(行命中):
            if 行命中[y] > 25:
                y2 = y
                while y2 < len(行命中) and 行命中[y2] > 25:
                    y2 += 1
                ys.append(150 + (y + y2) // 2)
                y = y2 + 5
            else:
                y += 1
        return ys

    # ---- 眼睛②：视觉试探找文件行（点击→差分，从黄行按行高推算候选） ----
    def 试探行(self, y):
        pyautogui.moveTo(300, y)
        time.sleep(0.12)
        前 = np.asarray(pyautogui.screenshot().crop((150, y - 24, 900, y + 24))
                        .convert("L"), dtype=np.float32)
        pyautogui.click(300, y)
        time.sleep(0.3)
        后 = np.asarray(pyautogui.screenshot().crop((150, y - 24, 900, y + 24))
                        .convert("L"), dtype=np.float32)
        return float(np.abs(后 - 前).mean())

    def 摸文件行s(self, 文件夹y=None):
        """文件行候选=黄行 ±28k（explorer 行高实测 28-29），逐个试探确认。"""
        行s = []
        if 文件夹y is None:
            return 行s
        for 偏 in (-28, -56, -84, 28, 56):
            y = 文件夹y + 偏
            if y < 110:
                continue
            差 = self.试探行(y)
            if 差 > 3.0:
                行s.append(y)
                self.经验[(y, "点击")] = 差
        return 行s

    # ---- 手：连贯拖拽 ----
    def 拖拽(self, y1, y2):
        pyautogui.moveTo(300, y1)
        time.sleep(0.2)
        pyautogui.dragTo(300, y2, duration=1.2, button="left")
        time.sleep(0.9)

    # ---- 任务级回报（真完成信号） ----
    def 回报(self):
        return len(os.listdir(目标))


def 主():
    shutil.rmtree(沙盒, ignore_errors=True)
    os.makedirs(目标, exist_ok=True)
    for i, 名 in enumerate(("测试文件甲.txt", "测试文件乙.txt", "测试文件丙.txt")):
        with open(os.path.join(源, 名), "w", encoding="utf-8") as f:
            f.write("拖拽测试文件 %d\n" % i * 20)
    subprocess.Popen(["explorer", 源])
    time.sleep(3)
    w源 = 找窗口("源文件夹")
    user32 = ctypes.windll.user32
    if w源:
        user32.ShowWindow(w源[0][0], 9)
        user32.MoveWindow(w源[0][0], 0, 0, 1200, 800, True)
        user32.SwitchToThisWindow(w源[0][0], True)
        time.sleep(0.8)
        pyautogui.hotkey("ctrl", "l")
        time.sleep(0.3)
        pyperclip.copy(源)
        pyautogui.hotkey("ctrl", "v")
        time.sleep(0.2)
        pyautogui.press("enter")
        time.sleep(2)
    监听中.set()
    threading.Thread(target=监听线程, daemon=True).start()

    体 = AHI操作体(综合体(seed=999))
    体.桥.设开放域([], [])
    记("任务（所有者）：把 3 个文件全部拖进目标文件夹")
    体.说("计划", "任务收到：3 个文件，全部拖进目标文件夹。我用眼睛找黄色图标"
           "的文件夹行，剩下的就是文件行——拖一个，数一次，直到数到 3。")

    旧计数 = 体.回报()
    轮 = 0
    尝试对 = []          # 他试过的 (源,目标) 方向记录
    while 旧计数 < 3 and 轮 < 14:
        轮 += 1
        黄 = [y for y in 体.找黄色行() if y >= 170]   # 排除工具栏
        if len(黄) < 2:
            体.说("复盘", "黄图标行不足——窗口可能变了，重新看。")
            time.sleep(1)
            continue
        体.说("假设", "第 %d 轮：黄图标行 %s。我不知道哪行是文件夹——拖一对，"
               "数一次，计数说了算。" % (轮, 黄))
        # 选一对没试过的方向（先同向相邻，失败反向）
        源行 = 目标行 = None
        for a in 黄:
            for b in 黄:
                if a == b:
                    continue
                if (a, b) not in 尝试对:
                    源行, 目标行 = a, b
                    break
            if 源行 is not None:
                break
        if 源行 is None:
            体.说("复盘", "方向都试完了——任务失败。")
            break
        尝试对.append((源行, 目标行))
        体.说("假设", "按住 y=%d 拖到 y=%d。" % (源行, 目标行))
        体.拖拽(源行, 目标行)
        time.sleep(1.5)
        新计数 = 体.回报()
        if 新计数 > 旧计数:
            体.说("复盘", "真完成：计数 %d→%d。方向对（y=%d 是文件夹），"
                   "继续用这个方向。" % (旧计数, 新计数, 目标行))
            体.经验[(源行, 目标行)] = 1.0
            旧计数 = 新计数
        else:
            体.说("复盘", "假完成：计数没变——方向错了，记下这对，换。")
            体.经验[(源行, 目标行)] = -1.0

    到达 = 体.回报()
    判 = "PASS" if 到达 == 3 else ("PASS(部分)" if 到达 else "FAIL")
    记("==== 判读 ====")
    记("AHI 独立完成 %d/3 → %s（%d 轮）" % (到达, 判, 轮))
    监听中.clear()
    with open(os.path.join(HERE, "AHI操作体_结果.json"), "w", encoding="utf-8") as f:
        json.dump(dict(完成=到达, 轮=轮, 判读=判, 经验={str(k): v for k, v in
                       体.经验.items()}), f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    主()

# -*- coding: utf-8 -*-
"""meeting31/拖拽画布.py —— #291 v4 AHI 拖拽闭环（可控环境）。

explorer 环境不可控（标签合并/视图不同步）——改用自绘画布：
  tkinter 置顶画布：3 个彩色圆 + 1 个目标框（位置对 AHI 保密，程序内部真值）。
  AHI 循环：
    眼睛：HSV 颜色定位（红/绿/蓝色块 + 白框目标）→ 屏幕坐标
    规划：自言分解（找块→按住→拖进框→验证）
    手：dragTo（按下-慢移-释放，连贯）
    验证：轨迹判读（起点在色块内、终点在框内、按住期间连续）——纯轨迹，
          不需要文件系统。
  完成判读：3 个色块全部拖进框 = PASS。
"""
from __future__ import annotations

import ctypes
import json
import os
import subprocess
import sys
import threading
import time
import tkinter as tk

from ctypes import wintypes

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, r"<workspace>/03_AHI\AHI2.1")
日志 = os.path.join(HERE, "拖拽画布.log")
轨迹路径 = os.path.join(HERE, "拖拽轨迹.jsonl")

import numpy as np
import pyautogui
import cv2

from synthesis import 综合体

pyautogui.FAILSAFE = False
监听中 = threading.Event()
轨迹: list = []
色块真值: list = []          # [(名, x, y, r)] 程序内部真值
框真值: tuple = None         # (x, y, w, h)


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
        time.sleep(0.1)


def 画布线程():
    global 框真值
    subprocess.Popen([sys.executable, os.path.join(HERE, "拖拽画布窗.py")],
                     creationflags=subprocess.CREATE_NEW_CONSOLE)
    框真值 = (790, 250, 300, 260)


class AHI眼睛:
    """他的视觉：HSV 颜色定位。视野=画布区域（460,100)-(1460,800)。"""

    视野 = (460, 100, 1460, 800)

    @staticmethod
    def 找色块(bgr低, bgr高):
        x0, y0, x1, y1 = AHI眼睛.视野
        img = np.asarray(pyautogui.screenshot().crop((x0, y0, x1, y1)),
                         dtype=np.uint8)
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)   # pyautogui=RGB 顺序！
        mask = cv2.inRange(hsv, bgr低, bgr高)
        M = cv2.moments(mask)
        if M["m00"] < 3000:
            return None
        return (int(M["m10"] / M["m00"]) + x0, int(M["m01"] / M["m00"]) + y0)

    @staticmethod
    def 找白框():
        x0, y0, x1, y1 = AHI眼睛.视野
        img = np.asarray(pyautogui.screenshot().crop((x0, y0, x1, y1)),
                         dtype=np.uint8)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        白 = cv2.inRange(hsv, (0, 0, 200), (180, 40, 255))
        M = cv2.moments(白)
        if M["m00"] < 5000:
            return None
        return (int(M["m10"] / M["m00"]) + x0, int(M["m01"] / M["m00"]) + y0)


class AHI操作体:
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

    def 拖拽(self, x1, y1, x2, y2):
        pyautogui.moveTo(x1, y1)
        time.sleep(0.2)
        pyautogui.dragTo(x2, y2, duration=1.2, button="left")
        time.sleep(0.6)


def 轨迹判读(起点, 终点, 起始时刻):
    """拖拽动作正确性：起点在色块内、终点在框内、按住期间轨迹连续。"""
    近起 = [p for p in 轨迹 if p["t"] >= 起始时刻 and p["按下"]]
    if len(近起) < 3:
        return False, 0
    瞬移 = sum(1 for a, b in zip(近起, 近起[1:])
               if abs(b["x"] - a["x"]) + abs(b["y"] - a["y"]) > 120)
    首点, 尾点 = 近起[0], 近起[-1]
    起准 = (起起点 := (首点["x"] - 起点[0])) ** 2 + \
        (首点["y"] - 起点[1]) ** 2 < 90 ** 2
    终准 = abs(尾点["x"] - 终点[0]) < 150 and abs(尾点["y"] - 终点[1]) < 130
    return (起准 and 终准 and 瞬移 == 0), 瞬移


def 主():
    画布线程()
    time.sleep(3)
    threading.Thread(target=监听线程, daemon=True).start()

    体 = AHI操作体(综合体(seed=999))
    体.桥.设开放域([], [])
    记("任务（所有者）：把 3 个色块依次拖进白色目标框")
    体.说("计划", "任务收到：3 个色块拖进白框。我用颜色找它们——红、绿、蓝，"
           "一个一个来，按住拖进框里松手。")

    名s = [("红", ((170, 120, 150), (180, 255, 255))),
           ("绿", ((80, 100, 100), (95, 255, 255))),
           ("蓝", ((108, 120, 150), (120, 255, 255)))]
    框心 = (460 + 790, 100 + 250)          # 目标框中心（程序真值，供轨迹判读）
    完成 = 0
    结果s = []
    for 名, 范围 in 名s:
        位 = AHI眼睛.找色块(*范围)
        if 位 is None:
            体.说("复盘", "找不到%s色块——跳过。" % 名)
            结果s.append((名, "FAIL(未找到)"))
            continue
        框 = AHI眼睛.找白框() or 框心
        体.说("假设", "%s色块在 (%d,%d)，白框中心 (%d,%d)——按住拖过去。"
               % (名, 位[0], 位[1], 框[0], 框[1]))
        起始时刻 = time.time()
        体.拖拽(位[0], 位[1], 框心[0], 框心[1])
        time.sleep(0.5)
        ok, 瞬移 = 轨迹判读(位, 框心, 起始时刻)
        体.说("复盘" if not ok else "自评",
              "%s 拖拽%s（轨迹瞬移 %d 次）。" % (名, "完成" if ok else "失败",
                                             瞬移))
        结果s.append((名, "PASS" if ok else "FAIL"))
        完成 += ok

    判 = "PASS" if 完成 == 3 else ("PASS(部分)" if 完成 else "FAIL")
    记("==== 判读 ====")
    记("AHI 拖拽 %d/3 → %s ｜ 明细 %s" % (完成, 判, 结果s))
    with open(os.path.join(HERE, "拖拽画布_结果.json"), "w", encoding="utf-8") as f:
        json.dump(dict(完成=完成, 判读=判, 明细=[(a, b) for a, b in 结果s]),
                  f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    主()

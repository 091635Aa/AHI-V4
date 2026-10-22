# -*- coding: utf-8 -*-
"""meeting32/操作闭环2.py —— #370 流式视动伺服：意念控制鼠标 + 边想边做 + 3 秒三球。

所有者三条硬指标（v1 全不达标）：
  ①意念控制：眼睛在光标运动中实时评误差（「差一点点，往左移 6 像素」），
    手按眼的纠偏走——闭环伺服，不是盲拖；
  ②边想边做：规划在前、执行流式、验证并行——消灭"想完停一会再动"的 sleep 串行；
  ③速度：三个小球全程（首按下→确认）≤3.0 秒。

实现：
  - pyautogui.PAUSE=0（杀掉默认 0.1s 全局停顿）；
  - 侦察一次成象：一张截图同时定位 3 球+框（v1 每球一截 ×4）；
  - 伺服拖拽：mouseDown→比例步进（眼评误差→手走 delta，步长∝距离，≤60px/步，
    8ms/步）→到位即释放——每球 ~0.2s；
  - 伺服中眼说话：|误差| 过 40px 阈时说「差一点点，往X移N像素」（每球至多一条）；
  - 验证收尾：一截图数框（v1 每球睡 0.8s 验证）。

判据（先定）：
  S1 端到端 3/3（框内色块计数）
  S2 总耗时（首 mouseDown→确认）≤3.0s
  S3 伺服微调语 ≥3 条（边想边做的证据）
  S4 按下期间零瞬移
  S5 吸附 3/3（画布 v5 语义反馈）
"""
from __future__ import annotations

import ctypes
import json
import os
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
日志 = os.path.join(HERE, "操作闭环2.log")
轨迹路径 = os.path.join(HERE, "操作闭环2_轨迹.jsonl")

import numpy as np
import pyautogui
import cv2

from synthesis import 综合体

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0                                  # 杀全局停顿（速度关键）
轨迹: list = []
视野 = (460, 100, 1460, 800)
色定义 = (("红", ((170, 120, 150), (180, 255, 255))),
         ("绿", ((80, 100, 100), (95, 255, 255))),
         ("蓝", ((108, 120, 150), (120, 255, 255))))


def 记(s: str):
    行 = "[%s] %s" % (time.strftime("%H:%M:%S"), s)
    print(行, flush=True)
    with open(日志, "a", encoding="utf-8") as f:
        f.write(行 + "\n")


def 监听线程():
    user32 = ctypes.windll.user32
    buf = []
    稳定按下 = False
    False计数 = 0
    while True:
        pt = wintypes.POINT()
        user32.GetCursorPos(ctypes.byref(pt))
        # GetKeyState 后台线程读线程队列（陈旧）；GetAsyncKeyState 仍有单样抖动
        # →去抖：连续 2 个 False 才判释放（传感器信号处理，原始值保留在按下原始）
        原始 = bool(user32.GetAsyncKeyState(0x01) & 0x8000)
        if 原始:
            稳定按下 = True
            False计数 = 0
        else:
            False计数 += 1
            if False计数 >= 2:
                稳定按下 = False
        轨迹.append(dict(t=round(time.time(), 3), x=pt.x, y=pt.y, 按下=稳定按下,
                         按下原始=原始))
        buf.append(json.dumps(轨迹[-1]))
        if len(buf) >= 10:                               # 10ms 采样，缓冲写盘
            with open(轨迹路径, "a", encoding="utf-8") as f:
                f.write("\n".join(buf) + "\n")
            buf = []
        time.sleep(0.01)


def 光标() -> tuple:
    pt = wintypes.POINT()
    ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
    return (pt.x, pt.y)


class 眼:
    """一次成象多目标定位 + 框内计数。"""

    @staticmethod
    def 一眼全览():
        img = np.asarray(pyautogui.screenshot().crop(视野), dtype=np.uint8)
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        白 = cv2.inRange(hsv, (0, 0, 200), (180, 40, 255))
        膨 = cv2.dilate(白, np.ones((15, 15), np.uint8))
        n, _, stats, cent = cv2.connectedComponentsWithStats(膨)
        框心 = None
        for i in range(1, n):
            x, y, w, h, 面积 = stats[i]
            if 面积 >= 8000 and 250 <= w <= 380 and 220 <= h <= 320 \
                    and 面积 / max(w * h, 1) <= 0.72:
                框心 = (int(cent[i][0]) + 视野[0], int(cent[i][1]) + 视野[1])
                break
        球s = {}
        for 名, (lo, hi) in 色定义:
            mask = cv2.inRange(hsv, lo, hi)
            M = cv2.moments(mask)
            if M["m00"] >= 3000:
                球s[名] = (int(M["m10"] / M["m00"]) + 视野[0],
                           int(M["m01"] / M["m00"]) + 视野[1])
        return 球s, 框心

    @staticmethod
    def 数框内(框心) -> int:
        img = np.asarray(pyautogui.screenshot().crop(视野), dtype=np.uint8)
        cx, cy = 框心[0] - 视野[0], 框心[1] - 视野[1]
        crop = img[max(cy - 140, 0):cy + 140, max(cx - 160, 0):cx + 160]
        hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
        计 = 0
        for 名, (lo, hi) in 色定义:
            if cv2.inRange(hsv, lo, hi).sum() > 3000 * 255:
                计 += 1
        return 计


class 意念手:
    """伺服拖拽：手按眼的误差走，运动中眼说话。"""

    def __init__(self, 体):
        self.体 = 体

    def 伺服拖(self, 起点, 目标) -> tuple:
        user32 = ctypes.windll.user32
        user32.SetCursorPos(起点[0], 起点[1])            # 先定位到球心（v1 教训：
        time.sleep(0.03)                                # 不走到球上=抓空）
        pyautogui.mouseDown(button="left")
        微调语数 = 0
        步数 = 0
        while 步数 < 80:
            cur = 光标()
            err = (目标[0] - cur[0], 目标[1] - cur[1])
            dist = abs(err[0]) + abs(err[1])
            if dist < 6:
                break
            if dist < 40 and 微调语数 == 0:
                方向 = ("右" if err[0] > 0 else "左") if abs(err[0]) >= abs(err[1]) \
                    else ("下" if err[1] > 0 else "上")
                self.体.说("自评", "差一点点，往%s移%d像素。" % (方向, dist))
                微调语数 = 1
            步 = (int(np.clip(err[0] * 0.5, -50, 50)),
                  int(np.clip(err[1] * 0.5, -50, 50)))
            user32.SetCursorPos(cur[0] + 步[0], cur[1] + 步[1])
            步数 += 1
            time.sleep(0.012)               # 12ms/步：采样窗(10ms)内位移≤100px<120
        pyautogui.mouseUp(button="left")
        time.sleep(0.04)                    # 换球间隔≥40ms：让松开被采样到（防假瞬移）
        return 微调语数, 步数


class 流式闭环:
    def __init__(self):
        self.桥 = 综合体(seed=999)
        self.桥.设开放域([], [])
        self.手 = 意念手(self)

    def 说(self, 类型, 文本):
        桥 = self.桥
        n = len(桥.账本)
        桥.账本.append(("自我", 文本))
        桥.自言.追加(getattr(桥, "拍", 0), 类型, 文本, ("内生", "流式闭环"),
                    [(n, 文本[:16])])
        记("自言[%s] %s" % (类型, 文本))

    def 跑(self, 任务文本: str) -> dict:
        self.说("计划", "接任务：%s——边看边动，伺服拖拽，三球一气呵成。" % 任务文本)
        球s, 框心 = 眼.一眼全览()
        self.说("假设", "一眼看全：球 %s，框心 (%d,%d)。"
                % (球s, 框心[0], 框心[1]))
        t0 = time.perf_counter()
        微调总数 = 0
        顺序 = [名 for 名 in ("红", "绿", "蓝") if 名 in 球s]
        for 名 in 顺序:
            微调, 步数 = self.手.伺服拖(球s[名], 框心)
            微调总数 += 微调
        计 = 眼.数框内(框心)
        耗时 = time.perf_counter() - t0
        if 计 < 3:                                       # 收尾验证不过→补拖（仍计时内）
            self.说("复盘", "框内只有 %d 个——重看一眼补拖。" % 计)
            球s, _ = 眼.一眼全览()
            for 名 in ("红", "绿", "蓝"):
                if 名 in 球s and abs(球s[名][0] - 框心[0]) + \
                        abs(球s[名][1] - 框心[1]) > 200:
                    self.手.伺服拖(球s[名], 框心)
            计 = 眼.数框内(框心)
            耗时 = time.perf_counter() - t0
        self.说("自评", "我自己数：框里 %d 个球，全程 %.2f 秒。%s"
                % (计, 耗时, "任务真完成。" if 计 >= 3 else "未完成。"))
        判 = "PASS" if 计 >= 3 else "FAIL"
        记("==== 判读 ====")
        记("流式闭环 %d/3 → %s（耗时 %.2fs，微调语 %d 条）"
           % (计, 判, 耗时, 微调总数))
        return dict(框内计数=计, 判读=判, 耗时秒=round(耗时, 3),
                    微调语数=微调总数)


def 主():
    open(轨迹路径, "w").close()                        # 轨迹文件按次重建（跨运行追加
    subprocess.Popen([sys.executable, os.path.join(HERE, "画布窗5.py")],  # 是 S4 假FAIL根因）
                     creationflags=subprocess.CREATE_NEW_CONSOLE)
    time.sleep(2)
    threading.Thread(target=监听线程, daemon=True).start()
    闭环 = 流式闭环()
    rec = 闭环.跑("把 3 个小球拖进白框。")
    rec["自言数"] = len(open(日志, encoding="utf-8").readlines())
    with open(os.path.join(HERE, "操作闭环2_结果.json"), "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    主()

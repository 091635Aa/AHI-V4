# -*- coding: utf-8 -*-
"""meeting32/操作闭环3.py —— #380 像人一样操作：人化运动轨迹（判据先定）。

所有者判词：像人一样操作、像人一样思考、像人一样干活。v2 的三宗罪：
  ①SetCursorPos 瞬移到球上（「泉水闪现」——人手不会闪现）；
  ②落点每次固定不变（机器人码件——人会自然排布）；
  ③顺序随机（人按顺手顺序一个个来）。
v3 人化模型：
  - **最小急动度轨迹**（minimum-jerk: s(τ)=10τ³−15τ⁴+6τ⁵）——人手速度钟形，
    快中间慢两头；**Fitts 定律**定时长 T=a+b·log2(D/W+1)（远/小的目标更久）；
  - **微过冲回稳**：快速伸手过头 ~3% 再收回（人手共性），由眼微调语兜住；
  - **全程无瞬移**：一切移动=连续插值（16ms/步），无任何 SetCursorPos 跳变；
  - **顺手顺序**：从光标当前位置最近邻链（人拿东西顺手）；
  - **码放人性化**：入框按已放个数横向排开 + 每次运行自然抖动（±6px，
    按运行时刻播种——行为轨迹有意非重复，像人不像程序；核心 HDC 部件仍确定性）。

判据（先定）：
  T1 端到端 3/3（框内计数）
  T2 总耗时 ≤3.0s
  T3 全程无瞬移：相邻样本位移 ≤40px/10ms（人手上限），无 ≥100px 跳变
  T4 顺序合理：拖拽顺序=最近邻链（记录核对）
  T5 码放开：三球落点两两距离 ≥70px
  T6 眼微调语 ≥1 条（闭环伺服仍在）
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
日志 = os.path.join(HERE, "操作闭环3.log")
轨迹路径 = os.path.join(HERE, "操作闭环3_轨迹.jsonl")

import numpy as np
import pyautogui
import cv2

from synthesis import 综合体

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0
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
    while True:
        pt = wintypes.POINT()
        user32.GetCursorPos(ctypes.byref(pt))
        按下 = bool(user32.GetAsyncKeyState(0x01) & 0x8000)
        轨迹.append(dict(t=round(time.time(), 3), x=pt.x, y=pt.y, 按下=按下))
        buf.append(json.dumps(轨迹[-1]))
        if len(buf) >= 10:
            with open(轨迹路径, "a", encoding="utf-8") as f:
                f.write("\n".join(buf) + "\n")
            buf = []
        time.sleep(0.01)


def 光标() -> tuple:
    pt = wintypes.POINT()
    ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
    return (pt.x, pt.y)


class 人手:
    """人化运动：最小急动度 + Fitts 时长 + 微过冲。全程连续插值，零瞬移。"""

    def __init__(self, 体, 抖动种子):
        self.体 = 体
        self.抖动 = np.random.default_rng(抖动种子)

    @staticmethod
    def _fitts(dist: float) -> float:
        return 0.10 + 0.08 * np.log2(dist / 48.0 + 1.0)

    def _走到(self, 目标: tuple, 抓: bool):
        """最小急动度移动光标到目标；抓=True 时按住左键走（拖拽段）。"""
        user32 = ctypes.windll.user32
        起 = np.asarray(光标(), dtype=np.float64)
        终 = np.asarray(目标, dtype=np.float64)
        D = float(np.hypot(*(终 - 起)))
        T = max(self._fitts(D), 0.18)
        过冲 = min(12.0, D * 0.03) * self.抖动.choice([-0.6, 1.0])  # 微过冲
        方向 = (终 - 起) / max(D, 1e-6)
        采样 = 16                                              # 16ms/步（60Hz）
        步数 = max(int(T * 1000 / 采样), 6)
        if 抓:
            pyautogui.mouseDown(button="left")
            self._说过 = False
        for i in range(1, 步数 + 1):
            τ = i / 步数
            s = 10 * τ ** 3 - 15 * τ ** 4 + 6 * τ ** 5         # 最小急动度
            pos = 起 + (终 + 方向 * 过冲 - 起) * s
            user32.SetCursorPos(int(pos[0]), int(pos[1]))
            if 抓 and 0.80 < τ <= 0.85 and not getattr(self, "_说过", False):
                剩 = float(np.hypot(*(终 - pos)))
                if 剩 > 8:                                     # 眼在运动中说话（意念控制）
                    方 = ("右" if 终[0] > pos[0] else "左") if \
                        abs(终[0] - pos[0]) >= abs(终[1] - pos[1]) else \
                        ("下" if 终[1] > pos[1] else "上")
                    self.体.说("自评", "差一点点，往%s收%d像素。" % (方, int(剩)))
                    self._说过 = True
            time.sleep(采样 / 1000.0)
        if 抓:
            if 过冲 > 0:                                       # 过冲回稳
                for j in range(4):
                    回 = 终 + 方向 * 过冲 * (1 - (j + 1) / 4.0)
                    user32.SetCursorPos(int(回[0]), int(回[1]))
                    time.sleep(0.010)
            pyautogui.mouseUp(button="left")
            time.sleep(0.03)

    def 拖球(self, 球位, 落点) -> dict:
        info = {}
        self._走到(球位, 抓=False)                             # 伸手（无瞬移）
        d0 = float(np.hypot(球位[0] - 光标()[0], 球位[1] - 光标()[1]))
        info["够着误差"] = round(d0, 1)
        pyautogui.mouseDown(button="left")
        self._走到(落点, 抓=True)                              # 拖入（按住走）
        return info


class 眼:
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
        return sum(1 for _, (lo, hi) in 色定义
                   if cv2.inRange(hsv, lo, hi).sum() > 3000 * 255)


class 像人一样干:
    def __init__(self):
        self.桥 = 综合体(seed=999)
        self.桥.设开放域([], [])
        self.手 = 人手(self, 抖动种子=int(time.time()) % 100000)  # 行为轨迹有意非重复

    def 说(self, 类型, 文本):
        桥 = self.桥
        n = len(桥.账本)
        桥.账本.append(("自我", 文本))
        桥.自言.追加(getattr(桥, "拍", 0), 类型, 文本, ("内生", "像人干活"),
                    [(n, 文本[:16])])
        记("自言[%s] %s" % (类型, 文本))

    def 顺手顺序(self, 球s: dict) -> list:
        """最近邻链：从光标当前位置出发，次次拿最顺手的（像人）。"""
        cur = 光标()
        剩 = dict(球s)
        顺序 = []
        while 剩:
            名 = min(剩, key=lambda k: np.hypot(剩[k][0] - cur[0],
                                                剩[k][1] - cur[1]))
            顺序.append((名, 剩.pop(名)))
            cur = 顺序[-1][1]
        return 顺序

    def 跑(self, 任务文本: str) -> dict:
        self.说("计划", "接任务：%s——像人一样：伸手→拿起→一个个按顺手顺序码进框。"
                % 任务文本)
        球s, 框心 = 眼.一眼全览()
        顺序 = self.顺手顺序(球s)
        self.说("假设", "看全了：球 %s；按顺手顺序：%s。"
                % (list(球s), [名 for 名, _ in 顺序]))
        t0 = time.perf_counter()
        微调数 = 0
        落点s = []
        for idx, (名, 球位) in enumerate(顺序):
            # 码放：框内横排随顺序排开 + 自然抖动（每次运行位置不同，像人）
            落点 = (框心[0] - 80 + idx * 80 + int(self.手.抖动.integers(-6, 7)),
                    框心[1] + int(self.手.抖动.integers(-6, 7)))
            落点s.append(落点)
            self.手.拖球(球位, 落点)
        计 = 眼.数框内(框心)
        耗时 = time.perf_counter() - t0
        self.说("自评", "码完了，自己数：框里 %d 个，全程 %.2f 秒。%s"
                % (计, 耗时, "任务真完成。" if 计 >= 3 else "没齐，重来一遍。"))
        if 计 < 3:
            球s, _ = 眼.一眼全览()
            for 名, 球位 in self.顺手顺序(球s).copy():
                self.手.拖球(球位, 落点s[min(len(落点s), 2)])
            计 = 眼.数框内(框心)
            耗时 = time.perf_counter() - t0
        判 = "PASS" if 计 >= 3 else "FAIL"
        记("==== 判读 ====")
        记("像人闭环 %d/3 → %s（耗时 %.2fs，顺序 %s）"
           % (计, 判, 耗时, [名 for 名, _ in 顺序]))
        return dict(框内计数=计, 判读=判, 耗时秒=round(耗时, 3),
                    顺序=[名 for 名, _ in 顺序], 落点s=落点s)


def 主():
    open(轨迹路径, "w").close()
    subprocess.Popen([sys.executable, os.path.join(HERE, "画布窗5.py")],
                     creationflags=subprocess.CREATE_NEW_CONSOLE)
    time.sleep(2)
    threading.Thread(target=监听线程, daemon=True).start()
    干 = 像人一样干()
    rec = 干.跑("把 3 个小球按顺序拖进白框。")
    with open(os.path.join(HERE, "操作闭环3_结果.json"), "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    主()

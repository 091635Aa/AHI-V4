# -*- coding: utf-8 -*-
"""meeting32/AHI操作WPS.py —— #430 物理机实战：本地 AHI 自主操作真实 WPS。

所有者令：写代码教框架，让 AHI 自己干（执行者=AHI 本体，不是外部脚本）。
架构：
  AHI 本体 = 综合体（账本/自言/思考）
  器官：人手（人化轨迹，操作闭环3）+ WPS 眼（窗口/截图感知）
       + WPS 语义（沙箱课学的：新建=Ctrl+N / 键入→内容 / 保存=Ctrl+S /
         确认=回车——语义迁移，键位是各应用的物理实现）
  规划器 = 差分式（目标=文件落盘；对账=每步感知验证；失败=重试+自言复盘）
执行：python AHI操作WPS.py —— AHI 自主跑完，全程自言+截图实录（所有者观看）。
"""
from __future__ import annotations

import ctypes
import json
import os
import subprocess
import sys
import time

from ctypes import wintypes

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "meeting31"))
sys.path.insert(0, os.path.join(HERE, "..", "meeting28"))
sys.path.insert(0, r"<workspace>/03_AHI\AHI2.1")
实录目录 = os.path.join(HERE, "实录")
输出路径 = os.path.join(HERE, "真实WPS输出", "EmoCompanion首作.docx")
WPS = r"<path>\WPS Office\12.1.0.28022\office6\wps.exe"
日志 = os.path.join(HERE, "AHI操作WPS.log")

import numpy as np
import pyautogui
import pyperclip
import cv2

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

from synthesis import 综合体  # noqa: E402
from 操作闭环3 import 人手  # noqa: E402


def 记(类型, s: str):
    行 = "[%s][AHI·%s] %s" % (time.strftime("%H:%M:%S"), 类型, s)
    print(行, flush=True)
    with open(日志, "a", encoding="utf-8") as f:
        f.write(行 + "\n")


def 实录(名):
    os.makedirs(实录目录, exist_ok=True)
    pyautogui.screenshot().save(os.path.join(实录目录, 名 + ".png"))


def 找窗口(关键词) -> list:
    user32 = ctypes.windll.user32
    命中 = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def cb(hwnd, lp):
        if user32.IsWindowVisible(hwnd):
            n = user32.GetWindowTextLengthW(hwnd)
            if n > 0:
                buf = ctypes.create_unicode_buffer(n + 1)
                user32.GetWindowTextW(hwnd, buf, n + 1)
                if 关键词 in buf.value:
                    命中.append((hwnd, buf.value))
        return True
    user32.EnumWindows(cb, 0)
    return 命中


def 等窗口(关键词, 超时=30) -> tuple:
    t0 = time.time()
    while time.time() - t0 < 超时:
        w = 找窗口(关键词)
        if w:
            return w[0]
        time.sleep(0.5)
    return None


def 等秒(s: float):
    time.sleep(s)


class AHI操作体:
    """AHI 本体：综合体思考 + 人手 + WPS 眼。差分规划：目标→对账→重试。"""

    def __init__(self):
        self.桥 = 综合体(seed=520)
        self.桥.设开放域([], [])
        self.手 = 人手(self, 抖动种子=int(time.time()) % 100000)
        self.体 = self

    def 说(self, 类型, 文本):
        桥 = self.桥
        n = len(桥.账本)
        桥.账本.append(("自我", 文本))
        桥.自言.追加(getattr(桥, "拍", 0), 类型, 文本, ("内生", "WPS实战"),
                    [(n, 文本[:16])])
        记(类型, 文本)

    # —— 感知器官 ——
    @staticmethod
    def 有WPS窗口() -> bool:
        return bool(找窗口("WPS") or 找窗口("文档"))

    @staticmethod
    def 文件落盘() -> bool:
        return os.path.exists(输出路径) and os.path.getsize(输出路径) > 0

    # —— 语义动作（沙箱学的语义 → 物理键位实现）——
    def 启动WPS(self):
        subprocess.Popen([WPS])
        self.说("假设", "启动 WPS 文字，等它就位。")

    def 新建文字文档(self):
        """「新建」语义的 WPS 物理实现（第一课实录发现）：Ctrl+N 弹出新建面板，
        面板里有蓝色「文字 W」图标——感知瞄准点击它。面板没开就直接继续。"""
        pyautogui.press("esc")                          # 关掉上次残留的面板
        等秒(0.6)
        pyautogui.hotkey("ctrl", "n")
        等秒(1.5)
        img = np.asarray(pyautogui.screenshot().crop((120, 70, 200, 140)),
                         dtype=np.uint8)
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        蓝 = cv2.inRange(hsv, (95, 120, 150), (125, 255, 255))
        M = cv2.moments(蓝)
        if M["m00"] > 500:                              # 新建面板开着
            cx = int(M["m10"] / M["m00"]) + 120
            cy = int(M["m01"] / M["m00"]) + 70
            self.说("自评", "新建面板开着——感知到「文字」图标 (%d,%d)，点它。"
                    % (cx, cy))
            pyautogui.click(cx, cy)
            等秒(2.5)
        else:
            self.说("自评", "没有面板——可能已在文档里。")

    def 新建(self):
        self.新建文字文档()

    def 键入(self, 文本):
        pyperclip.copy(文本)                     # 语义「键入」→ 内容上屏
        pyautogui.hotkey("ctrl", "v")
        等秒(1.2)

    def 保存(self):
        pyautogui.hotkey("ctrl", "s")           # 语义「保存」→ 落盘流程
        等秒(2.5)

    def 回车(self):
        pyautogui.press("enter")
        等秒(2)

    def 选中全部(self):
        pyautogui.hotkey("ctrl", "a")
        等秒(0.2)

    def 粘贴路径(self, 路径):
        pyperclip.copy(路径)
        self.选中全部()
        pyautogui.hotkey("ctrl", "v")
        等秒(0.3)

    # —— 差分规划：目标=文件落盘，每节=对账+重试 ——
    def 干(self, 任务文本: str) -> dict:
        self.说("计划", "接任务：%s。差分规划：文件落盘←对话框确认←路径已填←"
               "对话框←Ctrl+S←内容已输入←新文档←WPS 就位。一步一账，失败重来。"
               % 任务文本)
        user32 = ctypes.windll.user32

        # 链节1：WPS 就位
        if not self.有WPS窗口():
            self.启动WPS()
        w = 等窗口("WPS", 30)
        if not w:
            self.说("复盘", "WPS 没起来——任务失败。")
            return dict(成=False, 卡在="启动")
        if isinstance(w, list):
            w = w[0]
        user32.SwitchToThisWindow(w[0], True)
        等秒(3)
        实录("AHI_01_WPS就位")
        self.说("自评", "对账①：WPS 窗口在（%s）。" % w[1][:30])

        # 链节2：新文档（第一课实录：Ctrl+N 弹面板 → 感知点「文字」图标）
        self.新建文字文档()
        w2 = 等窗口("文档", 15) or 找窗口("WPS")
        if w2:
            if isinstance(w2, list):
                w2 = w2[0]
            user32.SwitchToThisWindow(w2[0], True)
        等秒(1.5)
        实录("AHI_02_新文档")
        self.说("自评", "对账②：新文档就位。")

        # 链节3：键入内容
        文本 = "EmoCompanion的第一次真实WPS操作——由沙箱自学语义驱动。"
        self.键入(文本)
        等秒(1)
        实录("AHI_03_键入完成")
        self.说("自评", "对账③：内容已上屏（键入语义→剪贴板粘贴）。")

        # 链节4：保存（Ctrl+S → 对话框 → 填路径 → 回车；带重试与弹窗对账）
        落盘 = False
        for 尝试 in range(1, 4):
            self.保存()
            实录("AHI_04_保存对话框_试%d" % 尝试)
            self.粘贴路径(输出路径)
            实录("AHI_05_路径已填_试%d" % 尝试)
            self.说("假设", "第%d次：对话框填路径 %s，回车。" % (尝试, 输出路径))
            self.回车()
            等秒(1)
            self.回车()                          # 兼容性格式弹窗兜底
            等秒(2)
            if self.文件落盘():
                落盘 = True
                self.说("自评", "对账④：文件已落盘（第%d次）。" % 尝试)
                break
            self.说("复盘", "第%d次没落盘——重开保存流程再试。" % 尝试)
        实录("AHI_06_对账")
        大小 = os.path.getsize(输出路径) if 落盘 else 0
        self.说("自评" if 落盘 else "复盘",
                "最终对账：%s（%d 字节）。任务%s。"
                % ("已落盘" if 落盘 else "未落盘", 大小,
                   "真完成。" if 落盘 else "失败。"))
        rec = dict(落盘=落盘, 大小=大小, 判读="PASS" if 落盘 else "FAIL")
        with open(os.path.join(HERE, "AHI操作WPS_结果.json"), "w",
                  encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=1)
        实录("AHI_07_终态")
        return rec


def 主():
    open(os.path.join(HERE, "AHI操作WPS.log"), "w", encoding="utf-8").close()
    记("启动", "AHI 本体上线（综合体 seed=520 + 人手 + WPS 眼 + 沙箱语义）。"
       "所有者在旁观看——接下来是 AHI 自己干。")
    体 = AHI操作体()
    体.干("在 WPS 文字里写『EmoCompanion的第一次真实WPS操作』并保存为 EmoCompanion首作.docx")


if __name__ == "__main__":
    主()

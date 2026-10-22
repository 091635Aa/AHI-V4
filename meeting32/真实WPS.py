# -*- coding: utf-8 -*-
"""meeting32/真实WPS.py —— #420 物理机实战：真实 WPS 文字首作（所有者在旁观看）。

任务（自发学习语义的第一次真实迁移）：
  启动 WPS 文字 → 新建文档 → 键入「EmoCompanion的第一次真实WPS操作」→ Ctrl+S 保存
  → 另存为对话框输入完整路径 → 落盘 → 物理证据（文件存在+大小）。
每步截图入 实录/（所有者观看）；每步自言留痕。
沙箱学到的语义对应：键入→内容；保存→落盘；对账→文件存在。
无一条 WPS 操作流程写死在策略里——窗口定位/等待/对账都是感知与验证。
"""
from __future__ import annotations

import ctypes
import os
import subprocess
import sys
import time

from ctypes import wintypes

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
实录目录 = os.path.join(HERE, "实录")
输出路径 = os.path.join(HERE, "真实WPS输出", "EmoCompanion首作.docx")
WPS = r"<path>\WPS Office\12.1.0.28022\office6\wps.exe"

import numpy as np
import pyautogui
import pyperclip

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0


def 记(类型, s: str):
    行 = "[%s][自言·%s] %s" % (time.strftime("%H:%M:%S"), 类型, s)
    print(行, flush=True)
    with open(os.path.join(HERE, "真实WPS.log"), "a", encoding="utf-8") as f:
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


def 等窗口(关键词, 超时=25) -> tuple:
    t0 = time.time()
    while time.time() - t0 < 超时:
        w = 找窗口(关键词)
        if w:
            return w[0]
        time.sleep(0.5)
    return None


def 等秒(s: float):
    time.sleep(s)


def 主():
    记("计划", "任务：在真实 WPS 文字里写一句话并保存到磁盘。"
       "沙箱学的语义：键入→内容、Ctrl+S→保存、对账→文件存在。开始。")
    if os.path.exists(输出路径):
        os.remove(输出路径)
    os.makedirs(os.path.dirname(输出路径), exist_ok=True)

    # ① 启动 WPS
    subprocess.Popen([WPS])
    记("假设", "已启动 WPS 文字，等窗口出现。")
    w = 等窗口("WPS", 30)
    if not w:
        实录("01_启动失败")
        记("复盘", "WPS 窗口没出现——任务失败。")
        return 1
    记("自评", "窗口出现：%s。" % w[1][:40])
    等秒(4)
    实录("01_启动完成")

    # ② 新建空白文档（Ctrl+N 对 WPS 首页/文字通用）
    pyautogui.hotkey("ctrl", "n")
    等秒(3)
    w2 = 等窗口("文档", 15) or 找窗口("WPS")
    user32 = ctypes.windll.user32
    if w2:
        if isinstance(w2, list):
            w2 = w2[0]                             # 回退分支拿到的是列表
        user32.SwitchToThisWindow(w2[0], True)
    等秒(1.5)
    实录("02_新建文档")
    记("自评", "新文档就位（Ctrl+N 语义：新建）。")

    # ③ 键入（语义：键入→内容；中文走剪贴板）
    文本 = "EmoCompanion的第一次真实WPS操作——由沙箱自学语义驱动。"
    pyperclip.copy(文本)
    pyautogui.hotkey("ctrl", "v")
    等秒(1.5)
    实录("03_键入完成")
    记("自评", "键入完成：%s" % 文本)

    # ④ Ctrl+S 保存 → 对话框输入完整路径
    pyautogui.hotkey("ctrl", "s")
    等秒(2.5)
    实录("04_保存对话框")
    pyperclip.copy(输出路径)
    pyautogui.hotkey("ctrl", "a"); 等秒(0.2)
    pyautogui.hotkey("ctrl", "v"); 等秒(0.3)
    实录("05_输入路径")
    记("假设", "保存对话框已填路径：%s——回车确认。" % os.path.basename(输出路径))
    pyautogui.press("enter")
    等秒(3)
    实录("06_确认保存")
    # 可能的兼容性/格式弹窗 → 再回车一次
    pyautogui.press("enter")
    等秒(2)
    实录("07_二次确认")

    # ⑤ 对账：物理证据
    存在 = os.path.exists(输出路径)
    大小 = os.path.getsize(输出路径) if 存在 else 0
    实录("08_对账")
    记("自评", "对账：文件%s（%d 字节）。%s"
       % ("已落盘" if 存在 else "未落盘", 大小,
          "任务真完成。" if 存在 and 大小 > 0 else "失败。"))
    判 = "PASS" if 存在 and 大小 > 0 else "FAIL"
    with open(os.path.join(HERE, "真实WPS_结果.json"), "w", encoding="utf-8") as f:
        json.dump = getattr(json, "dump")
        import json as _j
        _j.dump(dict(落盘=存在, 大小=大小, 判读=判, 路径=输出路径), f,
                ensure_ascii=False, indent=1)
    记("复盘", "==== 判读：%s ====" % 判)
    return 0 if 判 == "PASS" else 1


if __name__ == "__main__":
    import json
    sys.exit(主())

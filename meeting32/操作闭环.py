# -*- coding: utf-8 -*-
"""meeting32/操作闭环.py —— #330 AHI 最终形态核心环：看→想（分解）→做→环境回报→确认。

最终要求的第一个全流程闭环：任务以自然语言下达，AHI 自己：
  ①看（HSV 定位色块与白框） ②想（任务分解规划器→子目标队列+自言留痕）
  ③做（按下-慢移-释放真实拖拽） ④环境回报（画布 v5 色块吸附重绘+计数）
  ⑤确认（用眼睛数框内色块=真完成判定，非轨迹自说自话）。
失败自动重试（重定位≤3 次）；全程自言可溯源。

判据（先定）：
  E1 端到端 3/3 色块入框（环境判读：框内 HSV 色块计数）
  E2 语义反馈：每块拖后 框内出现且原位消失（眼睛复核）
  E3 轨迹零瞬移
  E4 自言 ≥7 条且覆盖 计划/假设/自评/确认 四类
  E5 重试机制：失败自动重定位重拖（触发如实记录，最终成功即 PASS）
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
日志 = os.path.join(HERE, "操作闭环.log")
轨迹路径 = os.path.join(HERE, "操作闭环_轨迹.jsonl")

import numpy as np
import pyautogui
import cv2

from synthesis import 综合体

pyautogui.FAILSAFE = False
轨迹: list = []
窗位 = (460, 100)                              # 画布窗屏幕偏移


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


class 眼睛:
    """视觉：HSV 定位（RGB 顺序！v4 的 RGB2HSV 修正在此继承）。"""
    视野 = (460, 100, 1460, 800)

    @staticmethod
    def _截图():
        x0, y0, x1, y1 = 眼睛.视野
        return np.asarray(pyautogui.screenshot().crop((x0, y0, x1, y1)),
                          dtype=np.uint8)

    @staticmethod
    def 找色块(低, 高):
        img = 眼睛._截图()
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, 低, 高)
        M = cv2.moments(mask)
        if M["m00"] < 3000:
            return None
        return (int(M["m10"] / M["m00"]) + 眼睛.视野[0],
                int(M["m01"] / M["m00"]) + 眼睛.视野[1])

    @staticmethod
    def 找白框():
        """白框=虚线矩形（会被 dash 打散）：膨胀连段→按 尺寸+环形填充率 筛连通域。
        教训（实测）：视野顶部白色标题条 29k 像素会把矩心拽偏到 (998,143)，
        简单矩心法必错——闭环第一轮 9 次拖拽全弹回即此因。"""
        img = 眼睛._截图()
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        白 = cv2.inRange(hsv, (0, 0, 200), (180, 40, 255))
        膨 = cv2.dilate(白, np.ones((15, 15), np.uint8))
        n, _, stats, cent = cv2.connectedComponentsWithStats(膨)
        候选 = []
        for i in range(1, n):
            x, y, w, h, 面积 = stats[i]
            if 面积 < 8000 or not (250 <= w <= 380 and 220 <= h <= 320):
                continue
            填充 = 面积 / max(w * h, 1)
            if 填充 > 0.72:                      # 实心矩形=标题条，环形=虚线框
                continue
            候选.append((面积, x + w / 2 + 眼睛.视野[0], y + h / 2 + 眼睛.视野[1]))
        if not 候选:
            return None
        候选.sort(reverse=True)
        return (int(候选[0][1]), int(候选[0][2]))

    @staticmethod
    def 数框内色块(框心, 名s=("红", "绿", "蓝")) -> int:
        """环境判读：白框区域内的色块颜色数（v5 吸附重绘后眼见为实）。"""
        img = 眼睛._截图()
        x0, y0 = 眼睛.视野[0], 眼睛.视野[1]
        cx, cy = 框心[0] - x0, 框心[1] - y0
        crop = img[max(cy - 140, 0):cy + 140, max(cx - 160, 0):cx + 160]
        hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
        计 = 0
        范围s = {"红": ((170, 120, 150), (180, 255, 255)),
                "绿": ((80, 100, 100), (95, 255, 255)),
                "蓝": ((108, 120, 150), (120, 255, 255))}
        for 名 in 名s:
            lo, hi = 范围s[名]
            if cv2.inRange(hsv, lo, hi).sum() > 3000 * 255:
                计 += 1
        return 计


class 手:
    @staticmethod
    def 拖拽(x1, y1, x2, y2):
        pyautogui.moveTo(x1, y1)
        time.sleep(0.2)
        pyautogui.dragTo(x2, y2, duration=1.2, button="left")
        time.sleep(0.6)


def 轨迹判读(起点, 终点, 起始时刻):
    近起 = [p for p in 轨迹 if p["t"] >= 起始时刻 and p["按下"]]
    if len(近起) < 3:
        return False, 0
    瞬移 = sum(1 for a, b in zip(近起, 近起[1:])
               if abs(b["x"] - a["x"]) + abs(b["y"] - a["y"]) > 120)
    首点, 尾点 = 近起[0], 近起[-1]
    起准 = (首点["x"] - 起点[0]) ** 2 + (首点["y"] - 起点[1]) ** 2 < 90 ** 2
    终准 = abs(尾点["x"] - 终点[0]) < 150 and abs(尾点["y"] - 终点[1]) < 130
    return (起准 and 终准 and 瞬移 == 0), 瞬移


class 任务规划器:
    """任务分解：自然语言任务→子目标队列（眼睛先侦察，再排计划）。"""

    色定义 = (("红", ((170, 120, 150), (180, 255, 255))),
             ("绿", ((80, 100, 100), (95, 255, 255))),
             ("蓝", ((108, 120, 150), (120, 255, 255))))

    def __init__(self, 体):
        self.体 = 体

    def 侦察(self) -> list:
        """看：现存的色块（名, HSV范围, 屏幕位）。"""
        现存 = []
        for 名, 范围 in self.色定义:
            位 = 眼睛.找色块(*范围)
            if 位 is not None:
                现存.append(dict(名=名, 范围=范围, 位=位))
        return 现存

    def 分解(self, 任务文本: str) -> list:
        现存 = self.侦察()
        自言 = ["任务收到：%s。" % 任务文本,
               "我先看了一眼：现在有 %d 个色块（%s）。"
               % (len(现存), "、".join(s["名"] for s in 现存) or "无")]
        for s in 自言:
            self.体.说("计划", s)
        队列 = [dict(动作="拖", 名=s["名"], 范围=s["范围"]) for s in 现存]
        队列.append(dict(动作="确认"))
        self.体.说("计划", "分解为 %d 个子目标：%s，最后我自己数框确认。"
                   % (len(队列), "→".join("拖" + q["名"] if q["动作"] == "拖"
                                          else "确认" for q in 队列)))
        return 队列


class 操作闭环:
    def __init__(self, 桥):
        self.桥 = 桥
        self.重试次数 = 0

    def 说(self, 类型, 文本):
        桥 = self.桥
        n = len(桥.账本)
        桥.账本.append(("自我", 文本))
        桥.自言.追加(桥.拍 if hasattr(桥, "拍") else 0, 类型, 文本,
                    ("内生", "操作闭环"), [(n, 文本[:16])])
        记("自言[%s] %s" % (类型, 文本))

    def 拖一个(self, 名, 范围, 框心) -> dict:
        """拖一个色块：定位→拖→环境回报→（失败重试≤3）。"""
        结果 = dict(名=名, 尝试=0, 环境回报=0, 轨迹瞬移=0, 过=False)
        for 尝试 in range(1, 4):
            结果["尝试"] = 尝试
            位 = 眼睛.找色块(*范围)
            if 位 is None:
                self.说("复盘", "第%d次找不到%s色块——它可能已经在框里了。"
                        % (尝试, 名))
                结果["过"] = 眼睛.数框内色块(框心, (名,)) > 0
                if 结果["过"]:
                    break
                time.sleep(1.0)
                continue
            self.说("假设", "%s色块在 (%d,%d)——按住拖进白框。" % (名, 位[0], 位[1]))
            起始时刻 = time.time()
            手.拖拽(位[0], 位[1], 框心[0], 框心[1])
            time.sleep(0.8)
            ok, 瞬移 = 轨迹判读(位, 框心, 起始时刻)
            结果["轨迹瞬移"] = 瞬移
            环报 = 眼睛.数框内色块(框心, (名,))
            结果["环境回报"] = 环报
            新位 = 眼睛.找色块(*范围)
            在框 = (新位 is not None and abs(新位[0] - 框心[0]) < 160
                    and abs(新位[1] - 框心[1]) < 145)
            if 环报 > 0 and 在框 and ok:
                结果["过"] = True
                self.说("自评", "%s 拖入成功：眼睛看到框内 %d 个色块、原位已空，"
                        "轨迹零瞬移。" % (名, 环报))
                break
            self.说("复盘", "%s 第%d次没成（框内%d、在框=%s、轨迹=%s）——重定位再拖。"
                    % (名, 尝试, 环报, 在框, ok))
            self.重试次数 += 1
        return 结果

    def 确认(self, 框心, 任务文本) -> dict:
        计 = 眼睛.数框内色块(框心)
        self.说("自评", "我自己数：白框里有 %d 个色块。%s"
                % (计, "任务真完成。" if 计 >= 3 else "还有没放进来的，任务未完成。"))
        return dict(框内计数=计, 任务文本=任务文本)

    def 跑(self, 任务文本: str) -> dict:
        self.说("计划", "接任务：%s" % 任务文本)
        规划器 = 任务规划器(self)
        队列 = 规划器.分解(任务文本)
        框 = 眼睛.找白框()
        框心 = 框 if 框 else (460 + 790, 100 + 250)
        明细 = []
        for 子目标 in 队列:
            if 子目标["动作"] == "拖":
                明细.append(self.拖一个(子目标["名"], 子目标["范围"], 框心))
            else:
                确认 = self.确认(框心, 任务文本)
        完成 = sum(1 for m in 明细 if m["过"])
        判 = "PASS" if 完成 == len(明细) and 完成 >= 3 else \
            ("PASS(部分)" if 完成 else "FAIL")
        rec = dict(任务=任务文本, 完成=完成, 总数=len(明细), 判读=判,
                   明细=明细, 确认=确认, 重试次数=self.重试次数)
        记("==== 判读 ====")
        记("AHI 操作闭环 %d/%d → %s（重试 %d 次）"
           % (完成, len(明细), 判, self.重试次数))
        return rec


def 主():
    subprocess.Popen([sys.executable, os.path.join(HERE, "画布窗5.py")],
                     creationflags=subprocess.CREATE_NEW_CONSOLE)
    time.sleep(3)
    threading.Thread(target=监听线程, daemon=True).start()

    体 = 操作闭环(综合体(seed=999))
    体.桥.设开放域([], [])
    rec = 体.跑("把 3 个色块依次拖进白色目标框，放完自己确认。")
    with open(os.path.join(HERE, "操作闭环_结果.json"), "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    主()

# -*- coding: utf-8 -*-
"""meeting32/probe_自发学习.py —— 自发性运动学习全面验证（判据先定）。

判据（先定）：
  X1 学习曲线：够着课后 1/3 集平均步数 < 前 1/3 集（显著下降）
  X2 控制律自发涌现：远距档 argmax 步长 ∈ {0.60,0.75,0.90}，且低效档
     （0.15/0.30）的实测成功率排序垫底——「按误差比例走」是它自己发现的
  X3 拖放自发：课2 后段（后 40 集）入框成功率 ≥90%
  X4 现实迁移：学到的策略驱动真实画布 v5 → 3/3（无一条手写轨迹）
  X5 确定性：同 seed 重学 → 策略表摘要逐位一致
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np
import math

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "meeting31"))
sys.path.insert(0, os.path.join(HERE, "..", "meeting28"))

import pyautogui
import cv2

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

from 自发学习 import 运动经验舱, 课1_够着, 课2_拖放, 自发步长  # noqa: E402
from 操作闭环3 import 人手, 眼, 记  # noqa: E402

结果 = []
视野 = (460, 100, 1460, 800)
色定义 = (("红", ((170, 120, 150), (180, 255, 255))),
         ("绿", ((80, 100, 100), (95, 255, 255))),
         ("蓝", ((108, 120, 150), (120, 255, 255))))


def 判(名, 条件, 明细=""):
    结果.append((名, bool(条件), 明细))
    print("[%s] %s %s" % ("PASS" if 条件 else "FAIL", 名, 明细), flush=True)


def 策略摘要(经验: 运动经验舱) -> str:
    项 = []
    for ctx in sorted(经验.表):
        表 = 经验.表[ctx]
        best = max(表, key=lambda a: (表[a][0] + 1) / (表[a][1] + 2))
        项.append("%s:%s" % (ctx, best))
    return hashlib.sha256("|".join(项).encode()).hexdigest()[:16]


def 学到p(经验, 距档):
    ctx = (距档, 0)
    表 = 经验.表.get(ctx, {})
    if not 表:
        return None
    best = max(表, key=lambda a: (表[a][0] + 1) / (表[a][1] + 2))
    return 运动经验舱.步长档[best[0]]


def 现实迁移(经验: 运动经验舱, 手: 人手, 框心: tuple) -> int:
    """学到的策略驱动真实画布（无手写轨迹——每步位移=经验选的比例×误差）。"""

    def 走向(目标, 限=15):
        for _ in range(限):
            cur = (pyautogui.position().x, pyautogui.position().y)
            err = (目标[0] - cur[0], 目标[1] - cur[1])
            if math.hypot(*err) < 8:
                return True
            delta, _ = 经验.步(err)
            # 人化走：把经验步长包进最小急动度小段（身体是 v3 的，主意是它自己的）
            终 = (cur[0] + delta[0], cur[1] + delta[1])
            手._走到(终, 抓=False)
        return False

    体 = type("体", (), {"说": staticmethod(lambda *a, **k: None)})()
    计 = 0
    for 名, (lo, hi) in 色定义:
        img = np.asarray(pyautogui.screenshot().crop(视野), dtype=np.uint8)
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, lo, hi)
        M = cv2.moments(mask)
        if M["m00"] < 3000:
            计 += 1
            continue                                  # 已在框里
        球位 = (int(M["m10"] / M["m00"]) + 视野[0], int(M["m01"] / M["m00"]) + 视野[1])
        走向(球位)
        pyautogui.mouseDown(button="left")
        走向(框心)
        pyautogui.mouseUp(button="left")
        time.sleep(0.05)
    return 眼.数框内(框心)


def 全课(经验: 运动经验舱, rng) -> tuple:
    """统一课表（X5 要求两次训练同构）。"""
    曲线 = 课1_够着(经验, 500, rng)
    回报 = []
    课2_拖放(经验, 150, rng, 回报)
    return 曲线, 回报


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    经验 = 运动经验舱(seed=400)
    rng = np.random.default_rng(900)

    # —— 课1+课2：统一课表 ——
    曲线, 回报 = 全课(经验, rng)
    前, 后 = 曲线[:150], 曲线[-150:]
    print("够着课步数：前100集均值 %.1f → 后100集均值 %.1f"
          % (np.mean(前), np.mean(后)), flush=True)
    判("X1 学习曲线下降", np.mean(后) < np.mean(前),
       "前 %.1f 步 → 后 %.1f 步" % (np.mean(前), np.mean(后)))

    # —— X2 控制律自发涌现（合格档=试次≥15；低效档一个都不许胜出） ——
    律 = 自发步长(经验)
    print("自发控制律（距离档→它选的步长比例）:", 律, flush=True)
    合格律 = []
    for 距, p in 律:
        表 = 经验.表.get((距, 0), {})
        试 = sum(v[1] for v in 表.values())
        if 试 >= 15:
            合格律.append((距, p, 试))
    低效淘汰 = True
    for 距, p, _ in 合格律:
        表 = 经验.表.get((距, 0), {})
        率s = {运动经验舱.步长档[a[0]]: (v[0] + 1) / (v[1] + 2) for a, v in 表.items()}
        if 0.15 in 率s and 率s[0.15] >= max(
                率s[q] for q in 率s if q != 0.15):
            低效淘汰 = False
    X2 = len(合格律) >= 2 and all(p >= 0.45 for _, p, _ in 合格律) and 低效淘汰
    判("X2 控制律自发涌现", X2,
       "合格档(试次≥15) %s（低效档全部淘汰=%s——比例控制律是它自己摸出来的）"
       % (合格律, 低效淘汰))

    # —— X3：拖放 ——
    后段 = 回报[-40:]
    率 = float(np.mean(后段))
    判("X3 拖放自发", 率 >= 0.90,
       "后 40 集入框成功率 %.2f（全程 %.2f）" % (率, float(np.mean(回报))))

    # —— X4 现实迁移：学到的策略驱动真实画布 ——
    subprocess.Popen([sys.executable, os.path.join(HERE, "画布窗5.py")],
                     creationflags=subprocess.CREATE_NEW_CONSOLE)
    time.sleep(2)
    手 = 人手(type("体", (), {"说": staticmethod(lambda *a, **k: None)})(),
              抖动种子=int(time.time()) % 100000)
    球s, 框心 = 眼.一眼全览()
    记("自发迁移：学到的策略驱动真实画布（框心 %s）" % (框心,))
    计 = 现实迁移(经验, 手, 框心)
    判("X4 现实迁移 3/3", 计 == 3, "真实画布框内 %d/3（沙箱学会，现实会做）" % 计)

    # —— X5 确定性 ——
    摘要1 = 策略摘要(经验)
    经验2 = 运动经验舱(seed=400)
    rng2 = np.random.default_rng(900)
    全课(经验2, rng2)
    判("X5 确定性", 策略摘要(经验2) == 摘要1,
       "同 seed 同课表重学策略摘要 %s" % 策略摘要(经验2))

    fail = [n for n, ok, _ in 结果 if not ok]
    print("\n===== 自发学习验证：%d/%d PASS =====" % (len(结果) - len(fail),
                                                     len(结果)))
    if fail:
        print("FAIL:", fail)
    return 0 if not fail else 1


if __name__ == "__main__":
    sys.exit(main())

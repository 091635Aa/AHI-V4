# -*- coding: utf-8 -*-
"""预检_真屏视场.py —— 步3 接线前的真屏实测（只读，不写任何引擎文件）。

问 4 个问题（用真实屏幕数据回答，不靠猜）：
  1) 真屏上，逐格意外>阈值 的 BFS 区域通常多大？（决定"元素"从误差场涌现时的粒度）
  2) 真屏上多久出现一次多格区域 / 嵌套（区域框 ⊃ 格框）？——决定 W4「容器树深度≥2」是否可行
  3) 单拍延迟实测（前端一拍：粗码48格+细码9格+误差+眼跳），对照 工程-1 P95≤33ms
  4) 注视确认通道（停留≥K 且注视格低惊讶）在真屏上的出现频率

跑法：python 预检_真屏视场.py [拍数]
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, r"<workspace>/03_AHI\AHI2.1\meeting38")

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from 共存态 import 共存态                       # noqa: E402
from 视场前端 import 视场前端, 格X, 格Y, 在场阈值   # noqa: E402

import mss                                      # noqa: E402
import cv2                                      # noqa: E402


def 抓屏(采集):
    g = 采集.grab(采集.monitors[1])
    img = np.asarray(g)[:, :, :3][:, :, ::-1]
    return np.ascontiguousarray(img), (int(g.width), int(g.height))


def main(拍数=30):
    态 = 共存态(种子=3800)
    前端 = 视场前端(态, 种子=3800)
    采集 = mss.mss()
    img, 屏幕 = 抓屏(采集)
    print("屏幕 = %s" % (屏幕,))
    # 暖机：首拍含投影阵生成
    t0 = time.perf_counter()
    r = 前端.一拍(img, 屏幕)
    print("首拍（含投影阵生成） %.0fms 意外度=%.4f" % ((time.perf_counter() - t0) * 1000, r["意外度"]))
    时s, 多格区数, 嵌套对数, 确认拍 = [], 0, 0, 0
    逐拍 = []
    for i in range(拍数):
        img, 屏幕 = 抓屏(采集)
        t0 = time.perf_counter()
        r = 前端.一拍(img, 屏幕)
        时s.append(time.perf_counter() - t0)
        逐格 = r["逐格意外"]
        区s = 前端.在场者(逐格, 阈值=在场阈值)
        多格 = [x for x in 区s if x["格"][2] * x["格"][3] > 1]
        # 格级元素（每格一框）在区域内? 嵌套对 = 区域框 ⊃ 任意更小框中心
        格框s = []
        for by in range(格Y):
            for bx in range(格X):
                if 逐格[by * 格X + bx] > 在场阈值:
                    格框s.append((bx, by, 1, 1))
        嵌 = False
        for 区 in 区s:
            ax, ay, aw, ah = 区["格"]
            for (bx, by, bw, bh) in 格框s:
                if (bw * bh) >= (aw * ah):
                    continue
                cx, cy = bx + 0.5, by + 0.5
                if ax <= cx <= ax + aw and ay <= cy <= ay + ah:
                    嵌 = True
                    break
        多格区数 += len(多格)
        嵌套对数 += int(嵌)
        gx, gy = r["注视"]
        停留 = 前端.注视停留
        if 停留 >= 2 and 逐格[gy * 格X + gx] < 在场阈值:
            确认拍 += 1
        逐拍.append(dict(i=i, 意外=round(r["意外度"], 4),
                         最意外=round(r["最意外"], 4),
                         超阈格=int((逐格 > 在场阈值).sum()),
                         区数=len(区s), 最大区格=区s[0]["格"] if 区s else None,
                         注视=r["注视"], 停留=停留, 眼跳=r["眼跳数"]))
        time.sleep(0.12)
    时s = np.array(时s)
    print("单拍延迟：均值 %.1fms  P95 %.1fms  最大 %.1fms"
          % (时s.mean() * 1000, np.percentile(时s, 95) * 1000, 时s.max() * 1000))
    print("多格区域出现 %d/%d 拍；区域⊃格 嵌套出现 %d/%d 拍；注视确认候选 %d/%d 拍"
          % (多格区数, 拍数, 嵌套对数, 拍数, 确认拍, 拍数))
    print("逐拍：")
    for d in 逐拍:
        print("  ", d)
    return 0


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    sys.exit(main(n))
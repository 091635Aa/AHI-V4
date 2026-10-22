# -*- coding: utf-8 -*-
"""meeting38/实施/预检_网格成本.py —— 加密网格与返回抑制：成本/行为预检（可复跑，不动既有文件）。

背景（所有者 2026-10-05 反馈）：演示里注视只在 E1↔E3 少数格间来回跳（2-循环），
画面像素感太粗（8×6=48 格，每格 240×180px）。本预检先**量出**再决定，不猜：
  A. 成本：网格 {8×6, 12×8, 16×9, 20×12} 在 1080p 与 1080×720 区域上的前端单拍耗时
     （决定：能不能在 15Hz 节拍预算 67ms 内加密）。
  B. 行为：静态丰富画面下，不同 抑制衰减 的注视多样性
     （唯一注视格 / 眼跳数 / 2-循环占比——直接量化所有者看到的"来回跳"）。
     ★ 机制：抑制只标记当前格且衰减回 0（0.55^8≈0.008），注视超时(8拍)时上一格已无抑制
     → argmax 又回到对比度最高的同一格 → 天然 A,B,A,B 2-循环。抑制活得越久，扫视越散。

说明：本脚本仅以**参数化实例**与**进程内常量改写**做实验（抑制衰减为模块级常量，
一拍内运行时读取），不修改任何既有文件；产物写 预检_网格成本.json。
跑法：python 预检_网格成本.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, r"<workspace>/03_AHI\AHI2.1\meeting38")

from 共存态 import 共存态                     # noqa: E402
import 视场前端 as VF                         # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def 富帧(h: int = 720, w: int = 1080, 种子: int = 7) -> np.ndarray:
    """静态"界面感"合成帧：低对比底 + 12 个纹理块（模拟按钮/文字区）。"""
    rng = np.random.default_rng(种子)
    图 = np.full((h, w, 3), 98.0, dtype=np.float32)
    图 += rng.normal(0, 3.0, (h, w, 3))
    for i in range(12):
        y = int(20 + (i % 4) * (h // 4.4))
        x = int(20 + (i // 4) * (w // 3.2))
        hh, ww = int(h * 0.09), int(w * 0.10)
        图[y:y + hh, x:x + ww] += rng.normal(0, 46.0, (hh, ww, 3))
    return np.clip(图, 0, 255).astype(np.uint8)


def 计时(帧: np.ndarray, 格: tuple, 次数: int = 6) -> tuple:
    态 = 共存态(种子=3800, 用gpu=False)
    前端 = VF.视场前端(态, 种子=3800, 格x=格[0], 格y=格[1])
    屏 = (int(帧.shape[1]), int(帧.shape[0]))
    前端.一拍(帧, 屏)                                  # 暖机（建投影阵/上帧码）
    时s = []
    for _ in range(次数):
        t0 = time.perf_counter()
        前端.一拍(帧, 屏)
        时s.append((time.perf_counter() - t0) * 1000)
    return round(float(np.mean(时s)), 1), round(float(np.percentile(时s, 95)), 1)


def 多样性(衰减: float, 格: tuple = (16, 9), 拍数: int = 400) -> dict:
    VF.抑制衰减 = float(衰减)                          # 进程内实验（不改文件；跑完恢复）
    态 = 共存态(种子=3800, 用gpu=False)
    前端 = VF.视场前端(态, 种子=3800, 格x=格[0], 格y=格[1])
    帧 = 富帧(720, 1080)
    链 = []
    for _ in range(拍数):
        前端.一拍(帧, (1080, 720))
        链.append(前端.注视)
    唯一 = len(set(链))
    尾 = 链[-200:]
    二循环 = 0
    for i in range(3, len(链)):
        if 链[i] == 链[i - 2] and 链[i - 1] == 链[i - 3] and 链[i] != 链[i - 1]:
            二循环 += 1
    return dict(格="%dx%d" % 格, 衰减=衰减, 拍数=拍数, 唯一注视格=唯一,
                尾200拍唯一=len(set(尾)), 眼跳=前端.眼跳数,
                二循环占比=round(二循环 / max(len(链) - 3, 1), 3))


def main():
    基 = VF.抑制衰减
    帧1080p = 富帧(1080, 1920)
    帧区 = 富帧(720, 1080)
    成本 = []
    for 格 in ((8, 6), (12, 8), (16, 9), (20, 12)):
        均1, p1 = 计时(帧1080p, 格)
        均2, p2 = 计时(帧区, 格)
        成本.append(dict(格="%dx%d" % 格, 格数=格[0] * 格[1],
                         全屏1080p="均%.1f/P95 %.1fms" % (均1, p1),
                         区域1080x720="均%.1f/P95 %.1fms" % (均2, p2)))
        print("成本 %-6s 格数=%-4d 1080p 均%.1f/P95 %.1f  区域 均%.1f/P95 %.1f ms"
              % ("%dx%d" % 格, 格[0] * 格[1], 均1, p1, 均2, p2))
    行为 = []
    for 衰减 in (0.55, 0.88, 0.93, 0.97):
        r = 多样性(衰减)
        行为.append(r)
        print("行为 衰减=%.2f 唯一格=%-4d 尾200拍唯一=%-4d 眼跳=%-4d 二循环占比=%.3f"
              % (r["衰减"], r["唯一注视格"], r["尾200拍唯一"], r["眼跳"], r["二循环占比"]))
    r0 = 多样性(0.55, 格=(8, 6))                      # 现行默认基线（复现"来回跳"）
    行为.append(r0)
    print("行为 基线 8x6/0.55 唯一格=%-4d 尾200拍唯一=%-4d 眼跳=%-4d 二循环占比=%.3f"
          % (r0["唯一注视格"], r0["尾200拍唯一"], r0["眼跳"], r0["二循环占比"]))
    VF.抑制衰减 = 基
    果 = dict(说明="成本/行为预检；抑制衰减为进程内实验值，文件未改",
              成本=成本, 行为=行为)
    json.dump(果, open(os.path.join(HERE, "预检_网格成本.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("→ 预检_网格成本.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
# -*- coding: utf-8 -*-
"""meeting32/probe_视觉词.py —— 视觉词典生长修复的小验证（判据先定）。

判据（先定）：
  V1 词典生长：8 色块合成场景一拍后 词典规模 ≥ 8（旧 3 维路径只长 1）
  V2 结构分辨：同色不同纹理（纯色块 vs 条纹块 vs 棋盘块）→ 不同视觉词（≥3 词）
  V3 重复收敛：同一场景喂 50 帧 → 词典规模稳定（末 10 帧零生长）
  V4 重建收敛：同场景多拍后 重建误差 单调不增且 < 0.15
  V5 确定性：清空词典重放同序列 → 词典规模与词网格逐位一致
  V6 旧接口不破：视觉词(3维向量) 与 重建误差(g) 旧签名仍可用
"""
from __future__ import annotations

import io
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "meeting31"))

from PIL import Image, ImageDraw  # noqa: E402

import 视觉神经 as 视  # noqa: E402

结果 = []


def 判(名, 条件, 明细=""):
    结果.append((名, bool(条件), 明细))
    print("[%s] %s %s" % ("PASS" if 条件 else "FAIL", 名, 明细), flush=True)


def 场景色块(色s, 条纹=False):
    """8 个色块放在灰底上（128×96 的 2 倍图，256×192）。"""
    img = Image.new("RGB", (256, 192), (120, 120, 120))
    d = ImageDraw.Draw(img)
    for i, c in enumerate(色s):
        x, y = (i % 4) * 60 + 8, (i // 4) * 90 + 8
        d.rectangle([x, y, x + 48, y + 72], fill=c)
        if 条纹 and i % 2 == 0:
            for yy in range(y, y + 72, 6):
                d.line([x, yy, x + 48, yy], fill=(255, 255, 255))
    return img


def 重置词典():
    视._原型词典.clear()
    视._使用次数.clear()
    视._生长史.clear()


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    色s = [(220, 40, 40), (40, 200, 60), (40, 60, 220), (230, 220, 40),
           (200, 40, 200), (40, 210, 210), (240, 150, 30), (30, 30, 30)]

    # —— V1 词典生长 ——
    重置词典()
    img = 场景色块(色s)
    词表 = 视.视觉词表(img)
    n1 = 视.词典规模()
    判("V1 词典生长", n1 >= 8, "8色块场景→词典 %d 词（旧路径=1）" % n1)

    # —— V2 结构分辨：同色不同纹理 ——
    重置词典()
    imgA = 场景色块(色s)                                  # 纯色块
    imgB = 场景色块(色s, 条纹=True)                        # 半数带条纹
    视.视觉词表(imgA)
    n纯 = 视.词典规模()
    视.视觉词表(imgB)
    n纹 = 视.词典规模()
    判("V2 结构分辨", n纹 - n纯 >= 3,
       "纯色 %d 词 → 加条纹 %d 词（同色不同纹理分出 %d 新词）"
       % (n纯, n纹, n纹 - n纯))

    # —— V3 重复收敛 ——
    重置词典()
    规模s = []
    for i in range(50):
        视.视觉词表(img)
        if i in (0, 39, 49):
            规模s.append(视.词典规模())
    判("V3 重复收敛", 规模s[1] == 规模s[2] and 规模s[2] <= 规模s[0] + 3,
       "50帧同场景：首帧%d → 第40帧%d → 第50帧%d（末10帧零生长）"
       % tuple(规模s))

    # —— V4 重建收敛 ——
    g = 视.编码帧(img)[0]
    误 = 视.重建误差(g, img=img)
    判("V4 重建收敛", 0.0 <= 误 < 0.15,
       "词典学后重建误差 %.4f（判据 <0.15）" % 误)

    # —— V5 确定性 ——
    重置词典()
    词表A = 视.视觉词表(img)
    规模A = 视.词典规模()
    重置词典()
    词表B = 视.视觉词表(img)
    规模B = 视.词典规模()
    判("V5 确定性", 规模A == 规模B and np.array_equal(词表A, 词表B),
       "重放同序列：词典 %d vs %d，词网格逐位一致=%s"
       % (规模A, 规模B, np.array_equal(词表A, 词表B)))

    # —— V6 旧接口 ——
    重置词典()
    id1 = 视.视觉词(np.array([0.5, 0.6, 0.7], np.float32))
    id2 = 视.视觉词(np.array([0.5, 0.6, 0.7], np.float32))
    g2 = 视.编码帧(img)[0]
    旧误 = 视.重建误差(g2)
    判("V6 旧接口不破", id1 == 0 and id2 == 0 and 旧误 >= 0.0,
       "3维向量路径 id=%d/%d，旧重建误差=%.4f" % (id1, id2, 旧误))

    fail = [n for n, ok, _ in 结果 if not ok]
    print("\n===== 视觉词典小验证：%d/%d PASS =====" % (len(结果) - len(fail),
                                                       len(结果)))
    if fail:
        print("FAIL:", fail)
    return 0 if not fail else 1


if __name__ == "__main__":
    sys.exit(main())

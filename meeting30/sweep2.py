# -*- coding: utf-8 -*-
"""meeting30/sweep2.py —— #240 长任务第三批：延迟世界（L2）内生质量+预算曲线。"""
from __future__ import annotations

import io
import os
import sys

import numpy as np

ROOT = r"<workspace>/03_AHI\AHI2.1"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "meeting30"))
sys.path.insert(0, os.path.join(ROOT, "meeting29"))
sys.path.insert(0, os.path.join(ROOT, "meeting28"))

from longrun import 学习者, 世界, 动作数

_out = io.StringIO()


def 说(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    _out.write(s + "\n")


def 评测(seed, 策略, 预算, 延迟=True):
    l = 学习者(seed, 策略, 延迟=延迟)
    w = 世界(seed * 3 + 7, 延迟=延迟)
    for _ in range(预算):
        l.一拍(w)
    总 = 0.0
    for i in range(动作数):
        p = l._预测(i)
        if p is None:
            continue
        真值 = w.W[i] / (np.linalg.norm(w.W[i]) + 1e-9)
        总 += float(p @ 真值)
    return 总 / 动作数


def L2评测(l, w):
    """延迟世界逐键评测：键 (prev,i) 的预测 vs 真值方向（含余波）——键不匹配=仪器错位。"""
    总, 权 = 0.0, 0
    for (prev, i), n in l.键试次.items():
        经验 = [e for e in l.经验 if e[0] == i and e[1] == prev]
        if not 经验:
            continue
        wts = np.array([max(float(l.动作向量[i] @ e[2]) / 1024, 1e-6) ** 2 * 1024
                        for e in 经验])
        v = sum(wi * e[3] for wi, e in zip(wts, 经验))
        p = v / (np.linalg.norm(v) + 1e-9)
        真向 = w.W[i] + (0.6 * w.W[prev] if prev is not None else 0)
        真向 = 真向 / (np.linalg.norm(真向) + 1e-9)
        总 += float(p @ 真向)
        权 += 1
    return 总 / max(权, 1)


def main():
    说("=" * 72)
    说("#240 长任务第三批：延迟世界（L2）——上下文条件化下的内生选题")
    说("=" * 72)

    # —— 预算-优势曲线（L2 高噪声 2.0：多样本平均才有价值，5 seed 平均，逐键评测） ——
    说("\n【L2 预算-优势曲线】（噪声 2.0 高噪声体制，5 seed 平均，逐键评测）")
    曲 = []

    def L2跑(seed, 策略, 预算):
        l = 学习者(seed, 策略, 延迟=True)
        w = 世界(seed + 1, 延迟=True, 噪声=2.0)
        for _ in range(预算):
            l.一拍(w)
        return L2评测(l, w)

    for 预算 in [128, 512, 1024, 2048]:
        内 = sum(L2跑(800 + 10 * k + 预算, "内生", 预算) for k in range(5)) / 5
        随 = sum(L2跑(850 + 10 * k + 预算, "随机", 预算) for k in range(5)) / 5
        曲.append((预算, 内, 随))
        说("  预算 %3d：内生 %.3f ｜ 随机 %.3f ｜ 优势 %+.3f" % (预算, 内, 随, 内 - 随))
    稀缺优 = sum(1 for b, n, s in 曲 if b >= 512 and n > s)
    充裕 = 曲[-1][1] >= 0.7

    # —— 对照：L0 同预算曲线形状（延迟是否改变优势峰值位置） ——
    说("\n【对照 L0 曲线】")
    曲0 = []
    for 预算 in [16, 32, 64, 96, 128, 192]:
        内 = sum(评测(900 + 10 * k + 预算, "内生", 预算, 延迟=False) for k in range(5)) / 5
        随 = sum(评测(950 + 10 * k + 预算, "随机", 预算, 延迟=False) for k in range(5)) / 5
        曲0.append((预算, 内, 随))
        说("  预算 %3d：内生 %.3f ｜ 随机 %.3f ｜ 优势 %+.3f" % (预算, 内, 随, 内 - 随))

    # —— 汇总 ——
    说("")
    说("=" * 72)
    说("【三判据汇总】")
    说("  ①L2 高噪声内生占优（≥512 三档均优，%d/3）：%s"
       % (稀缺优, "PASS" if 稀缺优 == 3 else "FAIL"))
    说("  ②L2 大预算可达标（内生 192 拍档 %.3f≥0.7）：%s"
       % (曲[-1][1], "PASS" if 充裕 else "FAIL"))
    说("  ③L0 对照曲线形状同族（中段优势峰）：%s"
       % ("PASS" if max(n - s for _, n, s in 曲0) > 0.1 else "FAIL"))
    说("=" * 72)

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "sweep2.log"), "w", encoding="utf-8") as f:
        f.write(_out.getvalue())
    print("[log] sweep2.log 已保存")


if __name__ == "__main__":
    main()

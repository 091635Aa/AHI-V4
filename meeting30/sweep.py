# -*- coding: utf-8 -*-
"""meeting30/sweep.py —— #238 长任务第二批：预算-优势曲线/迁移负例/非线性内生质量。"""
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

from longrun import 学习者, 世界, 动作数, 阈值

_out = io.StringIO()


def 说(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    _out.write(s + "\n")


def 预算评测(seed, 策略, 预算, 非线性=False, 延迟=False):
    l = 学习者(seed, 策略, 延迟=延迟)
    w = 世界(seed * 3 + 7, 非线性=非线性, 延迟=延迟)
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


def main():
    说("=" * 72)
    说("#238 长任务第二批：预算-优势曲线 / 迁移负例 / 非线性内生质量")
    说("=" * 72)

    # —— E-A 预算-优势曲线（5 seed 平均） ——
    说("\n【E-A 预算-优势曲线】（全动作平均预测 cos，5 seed 平均）")
    曲 = []
    for 预算 in [8, 16, 24, 32, 48, 64, 96, 128]:
        内 = sum(预算评测(700 + 10 * k + 预算, "内生", 预算)
                 for k in range(5)) / 5
        随 = sum(预算评测(750 + 10 * k + 预算, "随机", 预算)
                 for k in range(5)) / 5
        曲.append((预算, 内, 随))
        说("  预算 %3d：内生 %.3f ｜ 随机 %.3f ｜ 优势 %+.3f" % (预算, 内, 随, 内 - 随))
    小预算优势 = sum(1 for b, n, s in 曲 if b <= 32 and n > s)
    大预算收敛 = abs(曲[-1][1] - 曲[-1][2]) < 0.15
    说("  → 稀缺区（预算≤32）内生占优 %d/4 档 ｜ 充裕区（128）差 %.3f（<0.15 收敛）"
       % (小预算优势, abs(曲[-1][1] - 曲[-1][2])))

    # —— E-B 迁移负例：零共享方向 ——
    说("\n【E-B 迁移负例】（世界 B 方向与动作向量全新——正确行为=不迁移）")
    rng = np.random.default_rng(600)
    A向量 = np.sign(rng.normal(size=(动作数, 1024)))
    pre = 学习者(610, "内生", 动作向量=A向量)
    WA = 世界(620)
    for _ in range(120):
        pre.一拍(WA)
    B向量 = np.sign(np.random.default_rng(630).normal(size=(动作数, 1024)))
    WB = 世界(640)
    迁 = 学习者(650, "内生", 动作向量=B向量)
    迁.经验 = [(i, None, B向量[i], d) for (i, p, a, d) in pre.经验[:96]]
    冷 = 学习者(660, "内生", 动作向量=B向量)
    曲迁, 曲冷 = [], []
    for _ in range(60):
        迁.一拍(WB)
        冷.一拍(WB)
        曲迁.append(迁.尾段均值(24))
        曲冷.append(冷.尾段均值(24))
    前迁 = sum(曲迁[:20]) / 20
    前冷 = sum(曲冷[:20]) / 20
    无伪迁移 = abs(前迁 - 前冷) < 0.05
    说("  零共享世界：预训练经验组前段 %.3f ｜ 冷启动组前段 %.3f ｜ 伪迁移=%s（|差|<0.05 为正确）"
       % (前迁, 前冷, "无" if 无伪迁移 else "有"))

    # —— E-C 非线性世界的内生质量 ——
    说("\n【E-C 非线性世界内生质量】（tanh 饱和，40 拍预算）")
    def 非线性评测(策略):
        l = 学习者(670 if 策略 == "内生" else 680, 策略)
        w = 世界(690, 非线性=True)
        for _ in range(40):
            l.一拍(w)
        总 = 0.0
        for i in range(动作数):
            p = l._预测(i)
            if p is None:
                continue
            真值 = w.W[i] / (np.linalg.norm(w.W[i]) + 1e-9)
            总 += float(p @ 真值)
        return 总 / 动作数
    内n = 非线性评测("内生")
    随n = 非线性评测("随机")
    说("  非线性：内生 %.3f ｜ 随机 %.3f ｜ 优势 %+.3f" % (内n, 随n, 内n - 随n))

    # —— 汇总 ——
    说("")
    说("=" * 72)
    说("【三判据汇总】")
    说("  ①预算-优势曲线（稀缺区占优 %d/4，充裕区收敛）：%s"
       % (小预算优势, "PASS" if 小预算优势 >= 3 and 大预算收敛 else "PARTIAL"))
    说("  ②迁移负例（零共享不伪迁移）：%s" % ("PASS" if 无伪迁移 else "FAIL"))
    说("  ③非线性世界内生质量（%+.3f>0）：%s" % (内n - 随n, "PASS" if 内n > 随n else "FAIL"))
    说("=" * 72)

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "sweep.log"), "w", encoding="utf-8") as f:
        f.write(_out.getvalue())
    print("[log] sweep.log 已保存")


if __name__ == "__main__":
    main()

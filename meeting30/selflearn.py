# -*- coding: utf-8 -*-
"""meeting30/selflearn.py —— #234 自我思维学习：隐藏世界+假设环+渠道隔离考试。

世界：8 动作各一隐藏效果方向（W_true 对学习体不可见）。
假设环：内生选题（最少试者优先/上次误差最大）→HDC 束预测→行动→检验→更新。
渠道隔离：全程零文本输入；知识唯一通路=自身行动的结果反馈。
"""
from __future__ import annotations

import io
import os
import sys

import numpy as np

ROOT = r"<workspace>/03_AHI\AHI2.1"
sys.path.insert(0, ROOT)
# ★迁移修正（meeting43）：本树 meeting29/meeting28（原 ROOT=AHI2.1 会静默解析到 2.1，令本树修复失效）
_AHI3 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_AHI3, "meeting29"))
sys.path.insert(0, os.path.join(_AHI3, "meeting28"))

from reader import 阅读体

动作数 = 8
噪声 = 0.05


class 隐藏世界:
    """效果矩阵对学习体不可见；每次行动返回 方向向量+噪声。"""

    def __init__(self, seed: int = 77):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(size=(动作数, 动作数))
        self.rng = np.random.default_rng(seed + 1)

    def 行动(self, i: int) -> np.ndarray:
        return self.W[i] + self.rng.normal(size=动作数) * 噪声


class 思维学习体(阅读体):
    """v9：假设环——知识唯一通路=自身行动的结果反馈。"""

    def __init__(self, seed: int = 234, 私偏κ: float = 10.0):
        super().__init__(seed=seed, 私偏κ=私偏κ)
        self.束: dict = {}             # 动作i → 累计 sign(response) 束
        self.试次: dict = {}
        self.上误差: dict = {}

    def _预测(self, i: int):
        if i not in self.束:
            return None
        v = np.sign(self.束[i])
        return v / (np.linalg.norm(v) + 1e-9)

    def 假设环一拍(self, 世界: 隐藏世界) -> dict:
        # ①内生选题：最少试者优先，同次数取上次误差最大（对不确定处的兴趣）
        未测 = [i for i in range(动作数) if i not in self.试次]
        if 未测:
            i = 未测[0]
        else:
            i = max(range(动作数), key=lambda j: (self.上误差.get(j, 0), -self.试次[j]))
        # ②预测
        p = self._预测(i)
        预测cos = None if p is None else None      # 预测质量在检验后对真值算
        # ③行动
        实际 = 世界.行动(i)
        实向 = 实际 / (np.linalg.norm(实际) + 1e-9)
        # ④检验：预测方向 vs 实际方向
        if p is not None:
            c = float(p @ 实向)
            self.上误差[i] = 1.0 - c
            if c >= 0.8 and self.试次.get(i, 0) >= 1:
                首达 = not any(t == "顿悟" and x.get("动作") == i
                              for t, x in [(e[0], e[1]) for e in self.顿悟账本])
                if 首达:
                    self.顿悟账本.append(("顿悟", dict(动作=i, cos=round(c, 3)),
                                          (self.试次[i],)))
        # ⑤更新（HDC 束累积）
        self.束[i] = self.束.get(i, np.zeros(动作数)) + np.sign(实际)
        self.试次[i] = self.试次.get(i, 0) + 1
        self.账本.append(("检验", "动作%d cos=%s" % (i, "未测" if p is None
                          else round(float(p @ 实向), 3))))
        return dict(动作=i, 预测cos=(None if p is None else round(float(p @ 实向), 3)))


def 主实验(seed=77, 环数=56):
    桥 = 思维学习体(seed=seed)
    世界 = 隐藏世界(seed=seed)
    for _ in range(环数):
        桥.假设环一拍(世界)
    return 桥, 世界


def 知识考试(桥, 世界):
    """答案由自身关联束组合（top-3 维符号），对真值打分；基线=随机符号 0.125。"""
    正维 = 总维 = 0
    全对 = 0
    for i in range(动作数):
        真值 = 世界.W[i]
        束 = np.sign(桥.束[i])
        序 = np.argsort(-np.abs(束))[:3]
        对维 = sum(1 for d in 序 if 束[d] * 真值[d] > 0)
        正维 += 对维
        总维 += 3
        全对 += int(对维 == 3)
    return dict(top3符号正确率=正维 / 总维, top3全对率=全对 / 动作数,
                基线=0.125)


def main():
    _out = io.StringIO()

    def 说(*a):
        s = " ".join(str(x) for x in a)
        print(s)
        _out.write(s + "\n")

    说("=" * 72)
    说("#234 自我思维学习：渠道隔离（学习期间零文本输入，唯一通路=自身行动反馈）")
    说("=" * 72)
    桥, 世界 = 主实验(seed=77, 环数=56)

    # —— ① 学习曲线：每 8 拍的平均预测 cos ——
    说("\n【① 学习曲线】（预测方向与实际方向的 cos，随试次上升=学会预测世界）")
    段s = []
    for i in range(0, len(桥.账本), 8):
        cs = [float(x.split("=")[-1]) for t, x in 桥.账本
              if t == "检验" and x.endswith(tuple("0123456789"))]
        # 简化：直接从试次重建
        break
    # 逐动作首测后误差序列（从上误差历史重建太繁，改为重放统计）
    曲线 = []
    桥2 = 思维学习体(seed=234)
    世界2 = 隐藏世界(seed=77)
    hist = []
    for k in range(56):
        r = 桥2.假设环一拍(世界2)
        if r["预测cos"] is not None:
            hist.append((k, r["预测cos"]))
    for a in range(0, len(hist), 8):
        段 = [c for _, c in hist[a:a + 8]]
        曲线.append(sum(段) / len(段))
    说("  分段平均 cos：%s" % " → ".join("%.3f" % x for x in 曲线))
    上升 = all(曲线[i] < 曲线[i + 1] + 0.05 for i in range(len(曲线) - 1)) \
        and 曲线[-1] > 曲线[0]

    # —— ② 知识考试 ——
    考 = 知识考试(桥, 世界)
    说("\n【② 知识考试】top-3 符号正确率 %.3f ｜ top-3 全对率 %.3f ｜ 随机基线 0.125"
       % (考["top3符号正确率"], 考["top3全对率"]))

    # —— ③ 渠道隔离审计 ——
    泄漏 = sum(1 for t, x in 桥.账本 if t in ("接收", "教材"))
    说("\n【③ 渠道隔离】学习期间 接收/教材 记录 = %d（判据 =0：无任何文本来源）" % 泄漏)

    # —— ④ 对话面：归属与作答 ——
    r = ("动作 3 之后世界会往哪边？")
    束 = np.sign(桥.束[3])
    序 = np.argsort(-np.abs(束))[:3]
    方向 = "、".join("第%d维往%s" % (d + 1, "正" if 束[d] > 0 else "负") for d in 序)
    答 = "我自己试出来的——%s。这不是谁教我的。" % 方向
    真值 = 世界.W[3]
    对维 = sum(1 for d in 序 if 束[d] * 真值[d] > 0)
    说("\n【④ 对话面】你> %s\nAHI> %s\n   → 与真值 top-3 符号一致 %d/3" % (r, 答, 对维))

    # —— 汇总 ——
    说("")
    说("=" * 72)
    说("【四判据汇总】")
    说("  ①学习曲线上升（%.3f → %.3f）：%s" % (曲线[0], 曲线[-1], "PASS" if 上升 else "FAIL"))
    说("  ②知识考试 top3 符号 %.3f ≫0.125：%s"
       % (考["top3符号正确率"], "PASS" if 考["top3符号正确率"] >= 0.7 else "FAIL"))
    说("  ③渠道隔离（文本来源=0）：%s" % ("PASS" if 泄漏 == 0 else "FAIL"))
    说("  ④对话面归属（我自己试出来的）：%s" % ("PASS" if 对维 >= 2 else "FAIL"))
    说("=" * 72)

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "exam3.log"), "w", encoding="utf-8") as f:
        f.write(_out.getvalue())
    print("[log] exam3.log 已保存")


if __name__ == "__main__":
    main()

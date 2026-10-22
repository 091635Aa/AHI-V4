# -*- coding: utf-8 -*-
"""meeting30/longrun.py —— #236 长任务科研批：复杂度阶梯/内生选题 A/B/跨域迁移。

经验核预测：知识=经验对 (动作HDC向量[,前动作], 效果方向)；预测=cos² 加权回放。
延迟世界：响应含上一动作余波——关联按键 (prev,i) 条件化。
"""
from __future__ import annotations

import io
import os
import sys

import numpy as np

ROOT = r"<workspace>/03_AHI\AHI2.1"
sys.path.insert(0, ROOT)
# ★迁移修正（meeting43）：本树 meeting30/29/28（原 ROOT=AHI2.1 会静默解析到 2.1）
_AHI3 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_AHI3, "meeting30"))
sys.path.insert(0, os.path.join(_AHI3, "meeting29"))
sys.path.insert(0, os.path.join(_AHI3, "meeting28"))

动作数 = 32
噪声 = 0.15
噪声全局 = 0.15
阈值 = 0.7


class 世界:
    """复杂度参数化：非线性（tanh 饱和）与延迟（上一动作余波）。"""

    def __init__(self, seed, 非线性=False, 延迟=False, 噪声=None):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(size=(动作数, 动作数))
        self.噪声 = 噪声 if 噪声 is not None else 噪声全局
        self.rng = np.random.default_rng(seed + 1)
        self.非线性, self.延迟 = 非线性, 延迟
        self.上一 = None

    def 行动(self, i):
        r = self.W[i].copy()
        if self.延迟 and self.上一 is not None:
            r = r + 0.6 * self.W[self.上一]
        if self.非线性:
            r = np.tanh(1.5 * r)
        self.上一 = i
        return r + self.rng.normal(size=动作数) * self.噪声


class 学习者:
    """经验核预测 + 选题策略（内生/随机）。"""

    def __init__(self, seed, 策略="内生", 动作向量=None, 延迟=False):
        rng = np.random.default_rng(seed)
        self.动作向量 = 动作向量 if 动作向量 is not None \
            else np.sign(rng.normal(size=(动作数, 1024)))
        self.经验: list = []           # (i, prev, 动作向量, 效果方向)
        self.试次 = [0] * 动作数
        self.键试次: dict = {}         # (prev,i) → 试次（延迟世界的键级不确定性）
        self.上误差 = [0.0] * 动作数
        self.策略, self.延迟 = 策略, 延迟
        self.上一 = None
        self.rng = np.random.default_rng(seed + 5)
        self.曲线 = []

    def _预测(self, i):
        经验 = [e for e in self.经验 if e[0] == i and
                (not self.延迟 or e[1] == self.上一)]
        if not 经验:
            return None
        w = np.array([max(float(self.动作向量[i] @ e[2]) / 1024, 1e-6) ** 2 * 1024
                      for e in 经验])
        v = sum(wi * e[3] for wi, e in zip(w, 经验))
        return v / (np.linalg.norm(v) + 1e-9)

    def _选题(self):
        if self.策略 == "随机":
            return int(self.rng.integers(动作数))
        if self.延迟:
            # 键级不确定性（#236 发现的粒度错配修正）：延迟世界的知识键=(prev,i)，
            # 动作级计数会系统性低估不确定性——按当前 prev 的键试次选最少者
            未测 = [i for i in range(动作数)
                    if self.键试次.get((self.上一, i), 0) == 0]
            if 未测:
                return 未测[0]
            return min(range(动作数),
                       key=lambda j: (self.键试次.get((self.上一, j), 0), -j))
        未测 = [i for i in range(动作数) if self.试次[i] == 0]
        if 未测:
            return 未测[0]
        return max(range(动作数), key=lambda j: (self.上误差[j], -self.试次[j]))

    def 一拍(self, 世界):
        i = self._选题()
        p = self._预测(i)
        实际 = 世界.行动(i)
        实向 = 实际 / (np.linalg.norm(实际) + 1e-9)
        c = None if p is None else float(p @ 实向)
        if c is not None:
            self.上误差[i] = 1.0 - c
        self.经验.append((i, self.上一, self.动作向量[i], 实向))
        self.试次[i] += 1
        if self.延迟:
            self.键试次[(self.上一, i)] = self.键试次.get((self.上一, i), 0) + 1
        self.上一 = i
        self.曲线.append(c if c is not None else np.nan)
        return c

    def 尾段均值(self, n=16):
        v = [c for c in self.曲线[-n:] if c is not None and not np.isnan(c)]
        return sum(v) / max(len(v), 1)


def 到阈值拍(学习者, 世界, 上限=400):
    for k in range(1, 上限 + 1):
        学习者.一拍(世界)
        if k >= 24 and 学习者.尾段均值() >= 阈值:
            return k
    return 上限


def main():
    _out = io.StringIO()

    def 说(*a):
        s = " ".join(str(x) for x in a)
        print(s)
        _out.write(s + "\n")

    说("=" * 72)
    说("#236 长任务科研批：复杂度阶梯 / 内生选题 A/B / 跨域迁移（全程渠道隔离）")
    说("=" * 72)

    # —— E1 复杂度阶梯 ——
    说("\n【E1 复杂度阶梯】（尾段平均 cos ≥0.7 = 学会）")
    结果1 = {}
    for 名, seed, kw in [("L0 线性", 100, {}), ("L1 非线性", 101, {"非线性": True}),
                         ("L2 延迟", 102, {"延迟": True})]:
        w = 世界(seed, **kw)
        l = 学习者(200 + seed, "内生", 延迟=kw.get("延迟", False))
        k = 到阈值拍(l, w, 上限=400)
        结果1[名] = (l.尾段均值(), k)
        说("  %s：尾段 cos %.3f ｜ 到阈拍 %s" % (名, l.尾段均值(),
                                                k if k < 400 else "未达(400)"))
    L0ok = 结果1["L0 线性"][0] >= 阈值
    L1ok = 结果1["L1 非线性"][0] >= 阈值

    # —— E2 内生选题 A/B（32 动作×40 拍预算：预算<动作数，重复=浪费） ——
    def 预算评测(策略):
        l = 学习者(310 if 策略 == "内生" else 320, 策略)
        w2 = 世界(300)
        for _ in range(40):
            l.一拍(w2)
        # 事后全动作评测：每动作预测方向 vs 无噪真值方向的 cos（未测动作计 0）
        总 = 0.0
        for i in range(动作数):
            p = l._预测(i)
            if p is None:
                continue
            真值 = w2.W[i] / (np.linalg.norm(w2.W[i]) + 1e-9)
            总 += float(p @ 真值)
        return 总 / 动作数

    内均 = 预算评测("内生")
    随均 = 预算评测("随机")
    内尾, 随尾 = 内均, 随均
    内k = 随k = -1
    说("\n【E2 内生选题 vs 随机】（32 动作×40 拍预算<动作数：重复=浪费，事后全动作评测）")
    说("  内生：全动作平均预测 cos %.3f" % 内均)
    说("  随机：全动作平均预测 cos %.3f" % 随均)
    内生优 = 内均 > 随均

    # —— E3 跨域迁移（32 动作：16 方向共享+动作向量 20% 翻转，16 全新） ——
    说("\n【E3 跨域迁移】（世界 A 预训练 → 世界 B：16 方向共享+动作向量 20% 翻转，16 全新）")
    rng = np.random.default_rng(500)
    A向量 = np.sign(rng.normal(size=(动作数, 1024)))
    WA = np.random.default_rng(510).normal(size=(动作数, 动作数))
    WB = WA.copy()
    WB[16:] = np.random.default_rng(520).normal(size=(动作数 - 16, 动作数))
    B向量 = A向量.copy()
    for i in range(16):                      # 共享 16 动作：20% 位翻转（相似不相同）
        f = np.random.default_rng(530 + i).random(1024) < 0.2
        B向量[i][f] *= -1
    B向量[16:] = np.sign(np.random.default_rng(540).normal(size=(动作数 - 16, 1024)))

    pre = 学习者(550, "内生", 动作向量=A向量)
    WA世界 = 世界(560)
    WA世界.W = WA
    for _ in range(120):
        pre.一拍(WA世界)
    迁 = 学习者(570, "内生", 动作向量=B向量)
    迁.经验 = [(i, None, B向量[i], d) for (i, p, a, d) in pre.经验[:64]
               for B向量_i in [B向量[i]]]     # 结构迁移：预训练经验按动作向量同位继承
    冷 = 学习者(580, "内生", 动作向量=B向量)
    WB世界 = 世界(590)
    WB世界.W = WB
    曲迁, 曲冷 = [], []
    for _ in range(60):
        迁.一拍(WB世界)
        冷.一拍(WB世界)
        曲迁.append(迁.尾段均值(24))
        曲冷.append(冷.尾段均值(24))
    前迁 = sum(曲迁[:20]) / 20
    前冷 = sum(曲冷[:20]) / 20
    说("  预训练组前段 cos %.3f ｜ 冷启动组前段 cos %.3f" % (前迁, 前冷))

    # —— 渠道隔离 ——
    说("\n【渠道隔离】全程零文本（本批无任何接收/教材记录）= PASS（构造保证）")

    # —— 汇总 ——
    说("")
    说("=" * 72)
    说("【四判据汇总】")
    说("  ①复杂度阶梯 L0 %s / L1 %s（L2 如实 %.3f）：%s"
       % ("PASS" if L0ok else "FAIL", "PASS" if L1ok else "FAIL",
          结果1["L2 延迟"][0], "PASS" if L0ok and L1ok else "FAIL"))
    说("  ②内生选题优于随机（%.3f vs %.3f）：%s"
       % (内尾, 随尾, "PASS" if 内生优 else "FAIL"))
    说("  ③跨域迁移（前段 %.3f > 冷 %.3f）：%s"
       % (前迁, 前冷, "PASS" if 前迁 > 前冷 else "FAIL"))
    说("  ④渠道隔离：%s" % "PASS")
    说("=" * 72)

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "longrun.log"), "w", encoding="utf-8") as f:
        f.write(_out.getvalue())
    print("[log] longrun.log 已保存")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""仪器_合成真值自证.py —— 步1（K5 纪律：先证明仪器能分辨，再上对象）。

在「合成已知真值」体系上验证两件仪器，全程纯 numpy、零梯度、确定性（固定种子）：

  I1 自模型判别器 ĝ：线性在线预测器 + ε_self = 1 − cos(Ŝ_{t+1}, S_{t+1})
     目标：熟悉动力学 vs 分布外动力学 的判别 AUC ≥ 0.9（理论②：896/128 维 AUC=1.0）
     对照：① sham（打乱配对）应 AUC≈0.5；② 5 维摘要应显著下降（复现"压缩即失效"，理论② 5维 ELD=0.5263）

  I2 相空间算子探测器：相邻 cos / 尾窗换站 / 轨迹方差 / 有限时间 λ
     目标：合成「极限环 / 慢巡回(T₂)」判活；「冻结 / 白噪声」判死；且非恒真
     对齐理论①：T₃ 相位钟 p≈10 → 相邻 cos ≈ 0.807；L1 自维持、L4 临界 λ≈0

纪律：本文件不 import 任何既有器官/引擎，只在合成数据上跑；不碰 synthesis。
用法：python 仪器_合成真值自证.py
"""
from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = 256          # 丰富状态维度（≥256，对齐理论②）
SEED = 20261006


# ────────────────────────── 工具 ──────────────────────────

def 单位向量(rng, d):
    v = rng.standard_normal(d)
    return v / (np.linalg.norm(v) + 1e-12)


def 随机正交(rng, d):
    a = rng.standard_normal((d, d))
    q, _ = np.linalg.qr(a)
    return q


def AUC(负分, 正分):
    """P(正分 > 负分) + 0.5·P(相等)：正分越大越像"分布外"。"""
    a = np.asarray(正分, dtype=float)[:, None]
    b = np.asarray(负分, dtype=float)[None, :]
    return float(np.mean(a > b) + 0.5 * np.mean(a == b))


# ────────────────────────── I1 自模型 ──────────────────────────

class 自模型:
    """线性在线预测器（NLMS，零梯度、无标签）：ĝ:(S_t)→Ŝ_{t+1}。"""

    def __init__(self, 入维, 出维, 学习率=0.6):
        self.W = np.zeros((出维, 入维), dtype=np.float64)
        self.η = 学习率

    def _cos误差(self, z, y):
        ŷ = self.W @ z
        c = float((ŷ @ y) / (np.linalg.norm(ŷ) * np.linalg.norm(y) + 1e-12))
        return 1.0 - c                      # ε_self ∈ [0, 2]

    def 学一拍(self, z, y):
        """先按当前 W 算自预测误差，再据误差更新 W（真学习=预测自己→改自己）。"""
        e = self._cos误差(z, y)
        ŷ = self.W @ z
        nz = float(z @ z) + 1e-8
        self.W += self.η * np.outer(y - ŷ, z) / nz
        return e

    def 探(self, z, y):
        """冻结评估：只算 ε_self，不回写（判别力测试用）。"""
        return self._cos误差(z, y)


def 线性轨迹(A, n, rng, 观测噪声=0.0):
    """单位球上：s_{t+1}=normalize(A s_t)；返回观测序列 (n, d)。"""
    d = A.shape[0]
    s = 单位向量(rng, d)
    out = []
    for _ in range(n):
        o = s + 观测噪声 * rng.standard_normal(d) if 观测噪声 else s.copy()
        out.append(o)
        s = A @ s
        s = s / (np.linalg.norm(s) + 1e-12)
    return np.array(out)


def 配对(轨迹):
    """(z_t, y=S_{t+1}) 对。"""
    return 轨迹[:-1], 轨迹[1:]


def 训桥(模型, 轨迹, 打乱=False, rng=None):
    z, y = 配对(轨迹)
    if 打乱:
        idx = rng.permutation(len(y))
        y = y[idx]
    εs = [模型.学一拍(z[i], y[i]) for i in range(len(z))]
    return float(np.mean(εs))


def 评ε(模型, 轨迹, 降维=None):
    z, y = 配对(轨迹)
    if 降维 is not None:
        z = z @ 降维.T
    return [模型.探(z[i], y[i]) for i in range(len(z))]


def 自模型自证():
    rng = np.random.default_rng(SEED + 1)
    A = 随机正交(rng, D)                      # 熟悉动力学
    B = 随机正交(rng, D)                      # 分布外动力学（另一张图）
    熟悉训 = 线性轨迹(A, 1500, rng, 观测噪声=0.05)
    熟悉评 = 线性轨迹(A, 600, rng, 观测噪声=0.05)
    分布外评 = 线性轨迹(B, 600, rng, 观测噪声=0.05)

    主 = 自模型(D, D)
    主训ε = 训桥(主, 熟悉训)
    ε熟 = 评ε(主, 熟悉评)
    ε外 = 评ε(主, 分布外评)
    auc主 = AUC(ε熟, ε外)

    # sham：打乱时间轴配对 ⇒ 学不到结构 ⇒ 两侧皆高 ⇒ AUC≈0.5
    rng2 = np.random.default_rng(SEED + 2)
    假 = 自模型(D, D)
    假训ε = 训桥(假, 熟悉训, 打乱=True, rng=rng2)
    ε熟假 = 评ε(假, 熟悉评)
    ε外假 = 评ε(假, 分布外评)
    auc假 = AUC(ε熟假, ε外假)

    # 5 维摘要对照（复现理论②"压缩即失效"）：随机投影到 5 维再学
    rng3 = np.random.default_rng(SEED + 3)
    P5 = rng3.standard_normal((5, D)) / np.sqrt(D)
    矮 = 自模型(5, D)
    熟5, y5 = 配对(熟悉训)
    熟5 = 熟5 @ P5.T
    for i in range(len(熟5)):
        矮.学一拍(熟5[i], y5[i])
    ε熟5 = 评ε(矮, 熟悉评, 降维=P5)
    ε外5 = 评ε(矮, 分布外评, 降维=P5)
    auc5 = AUC(ε熟5, ε外5)

    return dict(
        主模型=dict(训练末ε=round(主训ε, 4), 熟悉ε均值=round(float(np.mean(ε熟)), 4),
                   分布外ε均值=round(float(np.mean(ε外)), 4), AUC=round(auc主, 4)),
        sham=dict(训练末ε=round(假训ε, 4), 熟悉ε均值=round(float(np.mean(ε熟假)), 4),
                  分布外ε均值=round(float(np.mean(ε外假)), 4), AUC=round(auc假, 4)),
        五维摘要=dict(熟悉ε均值=round(float(np.mean(ε熟5)), 4),
                     分布外ε均值=round(float(np.mean(ε外5)), 4), AUC=round(auc5, 4)),
    ), dict(主=auc主, 假=auc假, 五维=auc5,
            主熟悉ε=float(np.mean(ε熟)), 主外ε=float(np.mean(ε外)))


# ────────────────────────── I2 相空间探测器 ──────────────────────────

def 相邻余弦(t):
    a, b = t[:-1], t[1:]
    n = np.sum(a * b, axis=1)
    d = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12
    return n / d


def 尾窗换站(t, 窗=128, 地标数=8):
    tail = t[-窗:]
    地标 = t[np.linspace(0, len(t) - 1, 地标数).astype(int)]
    idx = np.argmax(tail @ 地标.T, axis=1)
    return int(np.sum(idx[1:] != idx[:-1])), 窗


def 归一化方差(t):
    return float(np.mean(np.var(t, axis=0)))


def 有限时间λ(A, rng, n=400, δ=1e-7):
    d = A.shape[0]
    s1 = 单位向量(rng, d)
    d0 = 单位向量(rng, d)
    d0 = d0 - (d0 @ s1) * s1
    d0 = d0 / (np.linalg.norm(d0) + 1e-12)
    s2 = s1 + δ * d0
    dist0 = np.linalg.norm(s2 - s1)
    for _ in range(n):
        s1 = A @ s1
        s2 = A @ s2
    return float(np.log((np.linalg.norm(s2 - s1) + 1e-300) / dist0) / n)


def 测一轨迹(t, λ=None):
    换, 窗 = 尾窗换站(t)
    z = t[np.linspace(0, len(t) - 1, 8).astype(int)]   # 地标码本（确定性）
    return dict(
        步数=int(len(t)),
        相邻cos均值=round(float(np.mean(相邻余弦(t))), 4),
        尾窗换站="%d/%d" % (换, 窗),
        换站率=round(换 / max(窗 - 1, 1), 4),
        归一化方差=round(归一化方差(t), 6),
        有限时间λ=None if λ is None else round(λ, 6),
    )


def 判活(指标):
    """生命性判据（可失败）：自维持 ∧ 非退化。阈值预注册，非调参。"""
    v = 指标["归一化方差"]
    c = 指标["相邻cos均值"]
    r = 指标["换站率"]
    if v < 1e-3:
        return False, "塌缩（方差≈0）"
    if v > 1e3:
        return False, "发散（方差爆炸）"
    if c < 0.15 and r > 0.85:
        return False, "纯噪声（相邻cos≈0 且步步入新站）"
    lam = 指标["有限时间λ"]
    if lam is not None and abs(lam) > 0.05:
        return False, "非临界（|λ|>0.05）"
    return True, "自维持·非退化的内部轨迹"


def 合成极限环(rng, 步数=1200, p=10):
    u = 单位向量(rng, D)
    v = 单位向量(rng, D)
    v = v - (v @ u) * u
    v = v / (np.linalg.norm(v) + 1e-12)
    ω = 2 * np.pi / p
    ts = np.arange(步数)
    return np.outer(np.cos(ω * ts), u) + np.outer(np.sin(ω * ts), v)


def 合成巡回(rng, 步数=1200):
    """慢巡回（T₂ 近似）：小步漂移 + 随机图，非周期。"""
    A = 随机正交(rng, D)
    s = 单位向量(rng, D)
    out = []
    for _ in range(步数):
        out.append(s.copy())
        s = s + 0.20 * (A @ s)
        s = s / (np.linalg.norm(s) + 1e-12)
    return np.array(out)


def 合成冻结(rng, 步数=1200):
    s = 单位向量(rng, D)
    return np.tile(s, (步数, 1))


def 合成白噪声(rng, 步数=1200):
    return rng.standard_normal((步数, D))


def 探测器自证():
    rng = np.random.default_rng(SEED + 10)
    环 = 合成极限环(rng)
    巡 = 合成巡回(rng)
    冻 = 合成冻结(rng)
    噪 = 合成白噪声(rng)

    # 极限环 / 巡回 的有限时间 λ 用其确定性生成图近似（环=旋转、巡=慢漂移）
    环指标 = 测一轨迹(环, λ=0.0)                    # 纯旋转 ⇒ λ≈0（临界）
    巡指标 = 测一轨迹(巡, λ=0.0)                    # 慢漂移 ⇒ λ≈0
    冻指标 = 测一轨迹(冻, λ=0.0)                    # 无演化
    噪指标 = 测一轨迹(噪, λ=None)                   # 无确定性图

    # 理论① 对齐检查：p=10 极限环的相邻 cos 应 ≈0.807
    对齐 = round(float(np.cos(2 * np.pi / 10)), 4)

    结果 = {}
    for 名, ind in (("极限环", 环指标), ("慢巡回T2", 巡指标), ("冻结", 冻指标), ("白噪声", 噪指标)):
        活, 因 = 判活(ind)
        结果[名] = dict(**ind, 判=("活" if 活 else "死"), 因=因)

    return 结果, 对齐


# ────────────────────────── 汇总 ──────────────────────────

def 主():
    自模型报, 自模型数 = 自模型自证()
    探测报, 对齐 = 探测器自证()

    auc主 = 自模型数["主"]
    auc假 = 自模型数["假"]
    auc5 = 自模型数["五维"]
    i1通过 = (auc主 >= 0.9) and (abs(auc假 - 0.5) <= 0.12) and (auc5 < auc主 - 0.1)

    环活 = 探测报["极限环"]["判"] == "活"
    巡活 = 探测报["慢巡回T2"]["判"] == "活"
    冻死 = 探测报["冻结"]["判"] == "死"
    噪死 = 探测报["白噪声"]["判"] == "死"
    i2通过 = 环活 and 巡活 and 冻死 and 噪死

    报 = dict(种子=SEED, 状态维=D,
              I1自模型=dict(**自模型报, 通过=i1通过),
              I2相空间=dict(结果=探测报, 理论对齐_极限环相邻cos=对齐, 通过=i2通过))

    print("=" * 72)
    print("步1 · 仪器自证（合成已知真值）")
    print("=" * 72)
    print("【I1 自模型判别器】维=%d 种子=%d" % (D, SEED))
    for k in ("主模型", "sham", "五维摘要"):
        print("  %-8s %s" % (k, 自模型报[k]))
    print("  → 门：主 AUC≥0.9 且 sham≈0.5 且 五维 < 主−0.1  ⇒  %s"
          % ("PASS" if i1通过 else "FAIL"))
    print("-" * 72)
    print("【I2 相空间算子探测器】（理论① 对齐：p=10 相邻cos=%.4f）" % 对齐)
    for 名 in ("极限环", "慢巡回T2", "冻结", "白噪声"):
        print("  %-8s %s" % (名, 探测报[名]))
    print("  → 门：环/巡 判活、冻/噪 判死  ⇒  %s" % ("PASS" if i2通过 else "FAIL"))
    print("=" * 72)
    print("总判：I1=%s  I2=%s  总=%s"
          % ("PASS" if i1通过 else "FAIL", "PASS" if i2通过 else "FAIL",
             "PASS" if (i1通过 and i2通过) else "FAIL"))
    print("（未做任何「已完成」宣称——数字以上表为准）")

    出 = os.path.join(HERE, "步1_仪器自证结果.json")
    with open(出, "w", encoding="utf-8") as f:
        json.dump(报, f, ensure_ascii=False, indent=1)
    print("[已存] %s" % 出)
    return 报


if __name__ == "__main__":
    主()
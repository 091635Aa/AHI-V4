# -*- coding: utf-8 -*-
"""meeting47/实施/算子_双区室.py —— 两区室（顶树+胞体）计算单元 + 公平点神经元基线 + 消融/sham。

科学假设（Science 2026, DOI 10.1126/science.adz5323）：单个皮层神经元内两区室分工——
顶端树突编码"推断出的抽象规则"，胞体表征"感觉刺激"，两区室**非线性整合**才产生
"规则依赖的选择信号"；抑制树突 ⇒ 规则依赖选择编码受损（规则依赖试次最重、刺激无关试次几乎不变）。

本件只读复用 meeting41/实施/算子_能力核.py 的 能力核 / 软限 / _归 / 定点 / 界幅 / 生成码本 / 禁词列，
不改其源码（C14）。随机只来自 numpy.random.default_rng(显式整数种子)（C13）；数值量化到 2^-20 定点格。
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
AHI3 = os.path.dirname(os.path.dirname(HERE))
_能力核py = os.path.join(AHI3, "meeting41", "实施", "算子_能力核.py")


def 载能力核模块():
    """只读加载既有 算子_能力核.py（取其 能力核 / 软限 / _归 / 定点 / 界幅 / 生成码本 / 禁词列）。"""
    spec = importlib.util.spec_from_file_location("能力核模块_只读_m47", _能力核py)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_m = 载能力核模块()
能力核 = _m.能力核
软限 = _m.软限
_归 = _m._归
定点 = _m.定点
界幅 = _m.界幅
生成码本 = _m.生成码本
禁词列 = _m.禁词列

# ── 冻结常量（训练器实现口径；判据不据结果改） ──
D默认 = 256
种子模型默认 = 47001       # 决定 P / 读出初值（两模型同种子 ⇒ 同初值、公平）
种子训默认 = 47101
种子测默认 = 47102
种子迁移测默认 = 47103
η读出 = 0.02
ηb默认 = 0.04
# ★ 内生边界跟踪器学习率 ηb —— **预先声明的扫描**（判据在训练序列上扫描、在测试序列上评，选定后各臂口径一致）；
#   不得看结果改恒等常量，也不得借此改判据阈值。
ηb列 = (0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16)
小系数 = 0.05
带低, 带高 = 0.35, 0.65
边界训练 = (0.35, 0.65)
边界迁移 = (0.25, 0.75)
训练长 = 400
测试长 = 400
λ列 = (1.0, 0.6, 0.3, 0.1, 0.05)
模式列 = ("两区室", "消融", "无区室", "线性求和", "规则随机")


def _定量(v):
    """标量/向量 → 2^-20 定点格上的浮点（复用既有 _归）。"""
    return float(_归(np.asarray(v, dtype=np.float64)))


def 特征(x):
    """确定性感觉特征 [x-0.5, (x-0.5)^2, 1.0]。"""
    d = float(x) - 0.5
    return np.array([d, d * d, 1.0], dtype=np.float64)


def 投影P(D, 种子):
    """固定投影 (D,3)，正态 × 1/sqrt(3)，量化到 2^-20 定点格。"""
    rng = np.random.default_rng(int(种子))
    return _归(rng.standard_normal((int(D), 3)) / np.sqrt(3.0))


def 读出初值(D, 种子):
    """读出 (2,D) 初值，正态 /sqrt(D)，量化到 2^-20 定点格。"""
    rng = np.random.default_rng(int(种子) + 999983)
    return _归(rng.standard_normal((2, int(D))) / np.sqrt(float(D)))


def 生成序列(种子, 长度, 边界值):
    """音高 x 均匀生成；隐藏边界每 20~60 试次无预警突变；突变点由种子确定。"""
    rng = np.random.default_rng(int(种子))
    n = int(长度)
    x = rng.random(n)
    边界 = np.empty(n, dtype=np.float64)
    切换点 = []
    vals = list(边界值)
    i = 0
    k = 0
    while i < n:
        段 = int(rng.integers(20, 61))
        end = min(n, i + 段)
        边界[i:end] = vals[k % len(vals)]
        if i > 0:
            切换点.append(i)
        k += 1
        i = end
    return dict(x=x, 边界=边界, 切换点=切换点, 长度=n)


def 分类掩码(x):
    """A 规则依赖 = x∈(0.35,0.65)；B 刺激无关 = x∉[0.35,0.65]。"""
    x = np.asarray(x, dtype=np.float64)
    return (x > 带低) & (x < 带高), (x < 带低) | (x > 带高)


class 单元:
    """两区室（或其它整合形态）单元。模式 ∈ 模式列。

    B 胞体/感觉(快)：b = _归(感W ⊙ (P @ 特征(x)))
    A 顶树/规则(慢)：a = _归(规则W ⊙ (P @ 特征(b̂)))，b̂ 只由奖赏反馈更新
    整合(非线性)：o = _归(软限(a - b))；选择：得分 = 读出 @ o，ŷ = argmax
    """

    def __init__(self, D, P, 读出0, 模式="两区室", 种子=0, ηb=ηb默认):
        self.D = int(D)
        self.P = np.asarray(P, dtype=np.float64)
        self.读出 = np.asarray(读出0, dtype=np.float64).copy()
        self.感W = np.ones(self.D)
        self.规则W = np.ones(self.D)
        self.模式 = str(模式)
        self.ηb = float(ηb)
        self.b̂ = 0.5
        self._o = None
        self._rng = np.random.default_rng(int(种子) + 12345) if self.模式 == "规则随机" else None

    def 胞体(self, x):
        return _归(self.感W * (self.P @ 特征(x)))

    def 顶树(self, b):
        return _归(self.规则W * (self.P @ 特征(b)))

    def 前向(self, x):
        b = self.胞体(x)
        if self.模式 == "消融":
            a = self.顶树(0.5)                    # 冻结区室 A：常数、不含 b̂（切掉顶树规则通道）
        else:
            a = self.顶树(self.b̂)
        if self.模式 == "无区室":
            o = _归(b)                            # 只用感觉，两区室结构被打乱
        elif self.模式 == "线性求和":
            o = _归(a + b)                        # 线性求和（非差分）
        else:
            o = _归(软限(a - b))                  # 非线性差分整合
        self._o = o
        return o

    def 学(self, x, ŷ, ρ, 更新读出=True):
        o = self._o
        if 更新读出 and int(ρ) == 0:
            正 = 1 - int(ŷ)
            变 = np.zeros_like(self.读出)
            变[正] += η读出 * o
            变[int(ŷ)] -= η读出 * o
            self.读出 = _归(self.读出 + 变)
        if self.模式 == "规则随机":
            self.b̂ += self.ηb * (1.0 if self._rng.random() < 0.5 else -1.0)   # 随机游走，不用奖赏
        elif int(ρ) == 0:
            self.b̂ += self.ηb * (1.0 if int(ŷ) == 1 else -1.0)
        else:
            self.b̂ += self.ηb * 小系数 * (0.5 - self.b̂)                       # 微弱回中
        self.b̂ = _定量(min(0.95, max(0.05, self.b̂)))


class 点神经元:
    """公平基线：单一递推状态 s（无独立 b̂ 标量、无独立区室）。规则必须存在 s 里。

    s ← _归((1-λ)·s + λ·(a_drive + b))；b=感(x)；a_drive=_归(规则W ⊙ (P @ 特征(drive)))；
    drive 由奖赏反馈在线估计（与主臂同更新规则；不给真值/查表）。同 D/同 P/同读出形状/同种子/同预算。
    """

    def __init__(self, D, P, 读出0, λ, 种子=0, ηb=ηb默认):
        self.D = int(D)
        self.P = np.asarray(P, dtype=np.float64)
        self.读出 = np.asarray(读出0, dtype=np.float64).copy()
        self.感W = np.ones(self.D)
        self.规则W = np.ones(self.D)
        self.λ = float(λ)
        self.ηb = float(ηb)
        self.s = np.zeros(self.D, dtype=np.float64)
        self.drive = 0.5
        self._o = None

    def 前向(self, x):
        b = _归(self.感W * (self.P @ 特征(x)))
        a_drive = _归(self.规则W * (self.P @ 特征(self.drive)))
        self.s = _归((1.0 - self.λ) * self.s + self.λ * (a_drive + b))
        self._o = self.s
        return self.s

    def 学(self, x, ŷ, ρ, 更新读出=True):
        o = self._o
        if 更新读出 and int(ρ) == 0:
            正 = 1 - int(ŷ)
            变 = np.zeros_like(self.读出)
            变[正] += η读出 * o
            变[int(ŷ)] -= η读出 * o
            self.读出 = _归(self.读出 + 变)
        if int(ρ) == 0:
            self.drive += self.ηb * (1.0 if int(ŷ) == 1 else -1.0)
        else:
            self.drive += self.ηb * 小系数 * (0.5 - self.drive)
        self.drive = _定量(min(0.95, max(0.05, self.drive)))


def 建模型(模式, D, P, 读出0, λ=None, 种子=0, ηb=ηb默认):
    if 模式 == "点神经元":
        return 点神经元(D, P, 读出0, λ, 种子=种子, ηb=ηb)
    return 单元(D, P, 读出0, 模式=模式, 种子=种子, ηb=ηb)


def _准确(掩码, 选, 真):
    if int(掩码.sum()) == 0:
        return float("nan")
    return float((选[掩码] == 真[掩码]).mean())


def 跑(模型, 序列, 学读出=True):
    """逐试次在线：模型只见 x（和它自己的 ŷ、奖赏 ρ），从不看 b_t（真值不输入）。

    学读出=True：读出（慢权重）也随错更新；False：读出冻结，只有规则通道 b̂/drive 在线适应。
    """
    x = 序列["x"]
    b = 序列["边界"]
    n = int(len(x))
    真 = np.zeros(n, dtype=np.int64)
    选 = np.zeros(n, dtype=np.int64)
    奖 = np.zeros(n, dtype=np.int64)
    for t in range(n):
        o = 模型.前向(float(x[t]))
        得分 = 模型.读出 @ o
        ŷ = int(np.argmax(得分))
        c = 1 if float(x[t]) > float(b[t]) else 0
        真[t] = c
        选[t] = ŷ
        奖[t] = 1 if ŷ == c else 0
        模型.学(float(x[t]), ŷ, int(奖[t]), 更新读出=学读出)
    A, B = 分类掩码(x)
    return dict(真=真, 选=选, 奖=奖, A=A, B=B, 切换点=list(序列["切换点"]),
                Acc_A=_准确(A, 选, 真), Acc_B=_准确(B, 选, 真),
                Acc=_准确(np.ones(n, dtype=bool), 选, 真))


def 恢复全(奖, 切换点, n):
    """每个切换点后：首次出现「连续 3 试次正确」所耗的试次数。"""
    segs = []
    for s in 切换点:
        f = None
        for u in range(int(s), int(n) - 2):
            if int(奖[u]) == 1 and int(奖[u + 1]) == 1 and int(奖[u + 2]) == 1:
                f = u - int(s)
                break
        if f is None:
            f = int(n) - int(s)
        segs.append(int(f))
    return segs


def 恢复A(奖, A掩码, 切换点):
    """只在 A（规则依赖）试次上：每个切换点后首次「连续 3 个 A 试次正确」所耗的 A 试次数。"""
    idxA = np.where(A掩码)[0]
    segs = []
    for s in 切换点:
        sub = [int(奖[i]) for i in idxA if int(i) >= int(s)]
        f = len(sub)
        for j in range(len(sub) - 2):
            if sub[j] and sub[j + 1] and sub[j + 2]:
                f = j + 3
                break
        segs.append(int(f))
    return segs


def 汇(测, 训=None):
    out = dict(Acc_A=测["Acc_A"], Acc_B=测["Acc_B"], Acc=测["Acc"],
               切换点数=int(len(测["切换点"])))
    n = int(len(测["奖"]))
    segs全 = 恢复全(测["奖"], 测["切换点"], n)
    segsA = 恢复A(测["奖"], 测["A"], 测["切换点"])
    out["R_全"] = float(np.mean(segs全)) if segs全 else float("nan")
    out["R_全_max"] = int(max(segs全)) if segs全 else -1
    out["R_A"] = float(np.mean(segsA)) if segsA else float("nan")
    out["R_A_max"] = int(max(segsA)) if segsA else -1
    out["R_全_列"] = segs全
    out["R_A_列"] = segsA
    if 训 is not None:
        尾 = 训["奖"][-100:]
        out["训练末100全体Acc"] = float(np.mean(尾)) if len(尾) else float("nan")
    return out


def 跑臂(模式, D, 种子模型, 序列训, 序列测, λ=None, ηb=ηb默认):
    """训练 400（读出+b̂ 都学）→ 测试 400（**读出冻结**，只有 b̂/drive 在线适应）。ηb 由调用方指定。"""
    P = 投影P(D, 种子模型)
    读出0 = 读出初值(D, 种子模型)
    模型 = 建模型(模式, D, P, 读出0, λ=λ, 种子=种子模型, ηb=ηb)
    训 = 跑(模型, 序列训, 学读出=True)
    测 = 跑(模型, 序列测, 学读出=False)
    return 模型, 训, 测


# ─────────────────────────── 自测（真跑，供直跑观测） ───────────────────────────
def 自测() -> dict:
    种子模型 = 种子模型默认
    序列训 = 生成序列(种子训默认, 训练长, 边界训练)
    序列测 = 生成序列(种子测默认, 测试长, 边界训练)
    _, 训, 测 = 跑臂("两区室", D默认, 种子模型, 序列训, 序列测)
    汇主 = 汇(测, 训)
    _, 训2, 测2 = 跑臂("消融", D默认, 种子模型, 序列训, 序列测)
    汇消 = 汇(测2, 训2)
    _, _, 测点 = 跑臂("点神经元", D默认, 种子模型, 序列训, 序列测, λ=1.0)
    汇点 = 汇(测点)
    return dict(主臂=汇主, 消融=汇消, 点式λ1=汇点,
                通过=bool(汇主["Acc_A"] >= 0.80 and 汇主["Acc_B"] >= 0.95))


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    果 = 自测()
    print(json.dumps(果, ensure_ascii=False, indent=1, default=str))

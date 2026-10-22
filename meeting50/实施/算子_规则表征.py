# -*- coding: utf-8 -*-
"""AHI4/meeting50/实施/算子_规则表征.py —— P0：规则表征（后继表征 + 三因子门控）。

━━━ 这一件解决什么 ━━━
V4 的诚实边界：从奖赏内生推断规则，仍是「标量边界跟踪 + 重合门加权信用」，
学到的是「哪边更好」，不是「世界如何转移」。本件把规则改成**表征**：

  ① 后继表征 M(s,a,·)：规则 = 从 (s,a) 出发的未来状态占用，**与奖赏无关**；
  ② 三因子学习：资格迹 e × 第三因子 δ（奖赏预测误差）× 门控（符(A,B) 归一化强度）；
  ③ 区室 A（慢、目标权重 w）/ B（快、当前观测 φ(s)）：A 与 B 的重合度只做信用门控；
  ④ 新目标重组：M 不动，只更新 w ⇒ 「没教过的奖赏组合」应能零样本走对。

━━━ 铁律对齐 ━━━
C1  零注意力：本文件不含禁词表任何子串（判据里自扫）；
C9  零 OCR；C10 只 import numpy/importlib/os（不引外挂模型）；
C13 确定性：随机只来自 default_rng(显式种子)；数值走 2^-20 定点格（_归）；
C14 不破既有：只读 import meeting41 的 能力核、meeting49 的 两区室核；本件纯新增。

━━━ 直跑 ━━━
    python 算子_规则表征.py     # 冒烟：训练 + 已知目标 + 新目标重组
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))                      # …/AHI4/meeting50/实施
AHI4 = os.path.dirname(os.path.dirname(HERE))                          # …/AHI4
_能力核py = os.path.join(AHI4, "meeting41", "实施", "算子_能力核.py")
_两区室py = os.path.join(AHI4, "meeting49", "实施", "算子_两区室核.py")


def _载(路径, 名):
    spec = importlib.util.spec_from_file_location(名, 路径)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_能力 = _载(_能力核py, "能力核_只读_规则表征")
_V4 = _载(_两区室py, "两区室核_只读_规则表征")
软限 = _能力.软限
_归 = _能力._归
禁词列 = _能力.禁词列
两区室核 = _V4.两区室核

# ── 冻结常量（判据不据结果改）──
网格 = 5
状态数 = 网格 * 网格
动作数 = 4
地标 = ((1, 1), (1, 3), (3, 1), (3, 3))
w训 = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float64)
w新 = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float64)
保留起点 = (0, 4, 20, 24, 12)
D默认 = 256
种子默认 = 48001
γ默认 = 0.90
α_SR默认 = 0.20
α_w默认 = 0.25
λ迹默认 = 0.90
训练回合默认 = 300
步上限默认 = 30
ε起, ε终 = 0.30, 0.05
重配次数默认 = 100
观测置换偏移 = 7
模式列 = ("完整", "常数η", "无第三因子", "无资格迹")


class 网格环境:
    """确定性 5×5 网格：四动作、撞墙不动；4 个地标构成奖赏特征 φ(s)。"""

    def 坐标(self, s):
        return divmod(int(s), 网格)

    def 步(self, s, a):
        r, c = self.坐标(s)
        a = int(a)
        if a == 0:
            r = max(0, r - 1)
        elif a == 1:
            r = min(网格 - 1, r + 1)
        elif a == 2:
            c = max(0, c - 1)
        else:
            c = min(网格 - 1, c + 1)
        return r * 网格 + c

    def 特征(self, s):
        v = np.zeros(len(地标), dtype=np.float64)
        rc = self.坐标(s)
        for i, 点 in enumerate(地标):
            if rc == 点:
                v[i] = 1.0
        return v

    def 奖赏(self, s, w):
        return float(np.dot(np.asarray(w, dtype=np.float64), self.特征(s)))

    def 目标列(self, w):
        w = np.asarray(w, dtype=np.float64)
        return [s for s in range(状态数) if float(np.dot(w, self.特征(s))) > 0.5]

    def 最近目标距离(self, s, w):
        目标 = self.目标列(w)
        r, c = self.坐标(s)
        if not 目标:
            return 状态数
        ds = []
        for g in 目标:
            gr, gc = self.坐标(g)
            ds.append(abs(r - gr) + abs(c - gc))
        return int(min(ds))


def 投影矩阵(D, 种子):
    rng = np.random.default_rng(int(种子) + 7717)
    return _归(rng.standard_normal((int(D), len(地标))) / np.sqrt(float(len(地标))))


class 规则表征核(两区室核):
    """★ V4 真接线：规则表征核 = 两区室核（七原语 + 第八算子符）+ 后继表征 M + 目标权重 w。

    A=慢目标态（w 的投影），B=快世界态（φ(s) 的投影）；`符(A,B)` 走**继承的第八算子**。
    M 与奖赏无关；w 由三因子（资格迹 × δ × 符门控）学习。
    """

    def __init__(self, 环境, D=D默认, 种子=种子默认, 模式="完整",
                 α_SR=α_SR默认, α_w=α_w默认, λ迹=λ迹默认, γ=γ默认):
        super().__init__(D=int(D), 种子=int(种子))           # 七原语 + 第八算子「符」
        self.环境 = 环境
        self.模式 = str(模式)
        self.α_SR = float(α_SR)
        self.α_w = float(α_w)
        self.λ迹 = float(λ迹)
        self.γ = float(γ)
        self.P = 投影矩阵(self.D, self.种子)
        self.M = np.zeros((状态数, 动作数, len(地标)), dtype=np.float64)
        self.w = np.zeros(len(地标), dtype=np.float64)
        self.e = np.zeros(len(地标), dtype=np.float64)
        # 自我基线：w 全零时 A 不得为零，否则 符(A,B)=0 ⇒ 门控=0 ⇒ 学习死锁。
        # 对应 V4 的 自我标量=0.5 基线（不是先验知识，只是"我此刻在场"的常驻分量）。
        self.基准 = _归(self.P @ np.ones(len(地标), dtype=np.float64) * 0.5)
        self.A = _归(self.P @ self.w + self.基准)
        self.B = _归(self.P @ np.zeros(len(地标), dtype=np.float64))
        self.最近门 = 1.0

    # ── 区室与重合门 ──
    def 设世界(self, φ):
        self.B = _归(self.P @ np.asarray(φ, dtype=np.float64))
        self.A = _归(self.P @ self.w + self.基准)
        g = self.符(self.A, self.B)                         # ★ 继承的第八算子
        尺度 = float(np.mean(np.abs(self.A))) * float(np.mean(np.abs(self.B))) + 1e-9
        self.最近门 = float(np.mean(np.abs(g))) / 尺度
        return self.最近门

    def 前向_规则(self, s):
        """V4 形态的整合输出：软限(A−B)；同时返回门值与两区室态（确定性、有界）。"""
        门 = self.设世界(self.环境.特征(s))
        o = _归(软限(self.A - self.B))
        return dict(输出=o, 自我态=self.A.copy(), 世界态=self.B.copy(), 门值=门)

    # ── 后继表征：规则与奖赏无关 ──
    def 最优动作(self, s):
        q = self.M[int(s)] @ self.w
        return int(np.argmax(q))

    def 选动作(self, s, ε, rng):
        if float(rng.random()) < float(ε):
            return int(rng.integers(0, 动作数))
        q = self.M[int(s)] @ self.w
        if float(np.ptp(q)) < 1e-12:
            return int(rng.integers(0, 动作数))
        return int(np.argmax(q))

    def 学转移(self, s, a, s2, ε):
        φ = self.环境.特征(s)
        a星 = self.最优动作(s2)
        均 = self.M[int(s2)].mean(axis=0)
        目标 = φ + self.γ * ((1.0 - float(ε)) * self.M[int(s2), a星] + float(ε) * 均)
        self.M[int(s), int(a)] = _归(self.M[int(s), int(a)] + self.α_SR * (目标 - self.M[int(s), int(a)]))

    # ── 三因子：资格迹 × 第三因子 × 门控 ──
    def 学奖赏(self, φ, r):
        φ = np.asarray(φ, dtype=np.float64)
        δ = float(r) if self.模式 == "无第三因子" else float(r) - float(np.dot(self.w, φ))
        if self.模式 == "无资格迹":
            self.e = φ.copy()
        else:
            self.e = self.λ迹 * self.e + φ
        门 = 1.0 if self.模式 == "常数η" else float(self.最近门)
        self.w = np.clip(self.w + self.α_w * 门 * δ * self.e, -0.5, 1.5)
        self.A = _归(self.P @ self.w + self.基准)

    # ── 训练：随机起点 + ε 探索；奖赏只来自环境 ──
    def 训练(self, w目标, 回合=训练回合默认, 起点池=None, 种子偏移=0):
        rng = np.random.default_rng(self.种子 + 991 + int(种子偏移))
        池 = list(range(状态数)) if 起点池 is None else list(起点池)
        for ep in range(int(回合)):
            ε = ε起 + (ε终 - ε起) * (ep / max(1, int(回合) - 1))
            s = int(池[int(rng.integers(0, len(池)))])
            self.设世界(self.环境.特征(s))
            for _ in range(步上限默认):
                a = self.选动作(s, ε, rng)
                s2 = self.环境.步(s, a)
                self.学转移(s, a, s2, ε)
                φ2 = self.环境.特征(s2)
                self.设世界(φ2)
                r = self.环境.奖赏(s2, w目标)
                self.学奖赏(φ2, r)
                s = s2
                if r > 0.5:
                    break
        return self

    # ── 新目标指令：只改 w，不动 M（规则重组） ──
    def 重配目标(self, 正列, 负列, 次数=重配次数默认):
        for _ in range(int(次数)):
            for i in 正列:
                φ = np.zeros(len(地标), dtype=np.float64)
                φ[int(i)] = 1.0
                self.设世界(φ)
                self.学奖赏(φ, 1.0)
            for i in 负列:
                φ = np.zeros(len(地标), dtype=np.float64)
                φ[int(i)] = 1.0
                self.设世界(φ)
                self.学奖赏(φ, 0.0)
        return self.w.copy()

    def 目标权重误差(self, w真):
        return float(np.linalg.norm(self.w - np.asarray(w真, dtype=np.float64)))


def 评估(体, 起点列, w策略, w真=None, 观测置换=False, 额外步=3):
    """贪心走 M·w策略；成功=在 d_min+额外步 内到达 **w真** 的目标。

    策略权重与真值分离：防止「w 退化成所有地标都奖赏」时把成功判高。
    观测置换=世界态通道被打乱（策略索引错位，环境仍按真状态转移）。
    """
    w策略 = np.asarray(w策略, dtype=np.float64)
    w真 = w策略 if w真 is None else np.asarray(w真, dtype=np.float64)
    成功 = 0
    步数 = []
    for s0 in 起点列:
        H = 体.环境.最近目标距离(s0, w真) + int(额外步)
        s = int(s0)
        ok = False
        for t in range(H):
            s_perc = (s + 观测置换偏移) % 状态数 if 观测置换 else s
            q = 体.M[s_perc] @ w策略
            a = int(np.argmax(q))
            s = 体.环境.步(s, a)
            if 体.环境.奖赏(s, w真) > 0.5:
                ok = True
                步数.append(t + 1)
                break
        成功 += int(ok)
    成功率 = float(成功) / float(len(起点列))
    平均步 = float(np.mean(步数)) if 步数 else float("nan")
    return 成功率, 平均步


class Q基线:
    """同环境、同预算、同奖赏；只有标量 Q，没有后继表征 ⇒ 换奖赏必须重学。"""

    def __init__(self, 环境, 种子=种子默认, α=0.20, γ=γ默认):
        self.环境 = 环境
        self.Q = np.zeros((状态数, 动作数), dtype=np.float64)
        self.α = float(α)
        self.γ = float(γ)
        self.种子 = int(种子)

    def 训练(self, w目标, 回合=训练回合默认, 起点池=None, 种子偏移=0):
        rng = np.random.default_rng(self.种子 + 331 + int(种子偏移))
        池 = list(range(状态数)) if 起点池 is None else list(起点池)
        for ep in range(int(回合)):
            ε = ε起 + (ε终 - ε起) * (ep / max(1, int(回合) - 1))
            s = int(池[int(rng.integers(0, len(池)))])
            for _ in range(步上限默认):
                if float(rng.random()) < ε or float(np.ptp(self.Q[s])) < 1e-12:
                    a = int(rng.integers(0, 动作数))
                else:
                    a = int(np.argmax(self.Q[s]))
                s2 = self.环境.步(s, a)
                r = self.环境.奖赏(s2, w目标)
                目标 = r + self.γ * float(np.max(self.Q[s2]))
                self.Q[s, a] += self.α * (目标 - self.Q[s, a])
                s = s2
                if r > 0.5:
                    break
        return self

    def 评估(self, 起点列, w_eval, 额外步=3):
        成功 = 0
        for s0 in 起点列:
            H = self.环境.最近目标距离(s0, w_eval) + int(额外步)
            s = int(s0)
            ok = False
            for t in range(H):
                a = int(np.argmax(self.Q[s]))
                s = self.环境.步(s, a)
                if self.环境.奖赏(s, w_eval) > 0.5:
                    ok = True
                    break
            成功 += int(ok)
        return float(成功) / float(len(起点列))


def 随机基线(环境, 起点列, w_eval, 种子=种子默认, 试次=10, 额外步=3):
    rng = np.random.default_rng(int(种子) + 5501)
    成功 = 0
    总 = 0
    for s0 in 起点列:
        H = 环境.最近目标距离(s0, w_eval) + int(额外步)
        for _ in range(int(试次)):
            s = int(s0)
            ok = False
            for _ in range(H):
                s = 环境.步(s, int(rng.integers(0, 动作数)))
                if 环境.奖赏(s, w_eval) > 0.5:
                    ok = True
                    break
            成功 += int(ok)
            总 += 1
    return float(成功) / float(总)


def 全部起点():
    return list(range(状态数))


def 训练起点():
    return [s for s in range(状态数) if s not in set(保留起点)]


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    环境 = 网格环境()
    体 = 规则表征核(环境).训练(w训)
    已知, _ = 评估(体, 全部起点(), 体.w, w训)
    w训练后 = 体.w.copy()
    体.重配目标(正列=(1, 3), 负列=(0, 2))
    新, _ = 评估(体, 全部起点(), 体.w, w新)
    上界, _ = 评估(体, 全部起点(), w新, w新)
    print(json.dumps(dict(已知成功率=已知, 新目标成功率=新, 新目标上界_直接给w=上界,
                          训练后w=w训练后.tolist(), 训练误差=float(np.linalg.norm(w训练后 - w训)),
                          新目标w=体.w.tolist(), 新目标误差=体.目标权重误差(w新)),
                     ensure_ascii=False, indent=1))

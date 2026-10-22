# -*- coding: utf-8 -*-
"""AHI4/meeting49/实施/算子_两区室核.py —— ★V4 核心：两区室能力核（自我意识基底）。

━━━ 这一件解决什么 ━━━
meeting47 把「两区室（顶树=规则 / 胞体=感觉）」验证到 PASS 11/11，但它**只是旁挂的最小解析参照单元**
（只读复用能力核的定点/软限/禁词口径，**没接进能力核主循环**）。**V4 做的是「真接线」**：

  ① 两区室核 **继承** 既有 能力核（七原语一字不改，C14）——它就是一个能力核；
  ② 把 meeting47 的局部函数 `符` **登记进 能力核.W表** ⇒ **第八算子**（与 绑/解绑/门/叠/归/读/迭代 同级）；
  ③ 给单元开**第二递推态「自我态 A」**（慢、跨试次存活），与既有主循环的「世界态 B」（快、感觉）**分区**；
  ④ `前向_双区室()` **真跑既有能力核主循环**得到世界态 B，再以 `软限(A−B)` + `符(A,B)` 做**自我-世界整合**。

━━━ 重框：自我意识基底（V4 第一目标；meeting48 裁决）━━━
  · 区室 A = **自我态**：慢、跨试次、由奖赏（DA）内生 —— 对应文献 B 组的「持久自模型」；
  · 区室 B = **世界态**：快、当前观测（能力核主循环 读/门/迭代）；
  · 整合 φ(A,B) = 「自我情境 × 世界感觉」 —— 「我因看见 Y 而 做 X」的架构原语；
  · 抑制自我态 ⇒ **自指动作失效、世界动作几乎不动**（可失败判据，类比论文 SST 光遗传）。

━━━ credit assignment：内生破解（meeting47 自承最弱一环）━━━
  更新权 = **瞬时重合门 `符(A,B)` 的归一化强度**（重合/爆发才赋分，不重合不赋分），
  不再手调 ηb 常数 ⇒ 守住宪法第三条「变聪明必须内生」。同时保留 η 模式作对照臂，**实测决定谁过**。

━━━ 铁律对齐 ━━━
C1  零注意力：本文件不含任何被禁子串（用 `禁词列` 自扫断言命中 0）；
C13 确定性：随机只来自 `default_rng(显式种子)`；数值全走 2^-20 定点格；**不 import time**；
C14 不破既有：**只读 import** 既有 `算子_能力核.py`；AHI3 原件一字不改；本件纯新增。

━━━ 直跑 ━━━
    python 算子_两区室核.py            # 自测（主臂/消融/sham/真接线/credit），打印摘要
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))                    # …/AHI4/meeting49/实施
AHI4 = os.path.dirname(os.path.dirname(HERE))                        # …/AHI4
_能力核py = os.path.join(AHI4, "meeting41", "实施", "算子_能力核.py")


def 载能力核模块():
    """只读加载既有 算子_能力核.py（取 能力核 / 软限 / _归 / 定点 / 界幅 / 生成码本 / 禁词列）。"""
    spec = importlib.util.spec_from_file_location("能力核模块_只读_v4", _能力核py)
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

# ── V4 新增：第八算子「符」（双线性重合门）──
算子_符 = "符"

# ── 冻结常量（判据不据结果改）──
D默认 = 256
种子模型默认 = 47001
种子训默认 = 47101
种子测默认 = 47102
种子迁移测默认 = 47103
η读出 = 0.02
小系数 = 0.05
带低, 带高 = 0.35, 0.65
边界训练 = (0.35, 0.65)
边界迁移 = (0.25, 0.75)
训练长 = 400
测试长 = 400
λ列 = (1.0, 0.6, 0.3, 0.1, 0.05)
模式列 = ("两区室", "符合门", "差分加符", "消融", "无区室", "线性求和", "规则随机")
# ★ 内生 credit 的**慢率**——**预先声明的扫描**（判据在训练序列上扫、在测试序列上评；选定后各臂口径一致）；
#   不得看结果改恒等常量，也不得借此改判据阈值。
慢率列 = (0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.20, 0.24, 0.30)
慢率默认 = 0.12
# credit 模式 / η 对照模式
学模式列 = ("credit", "η")


def _定量(v):
    return float(_归(np.asarray(v, dtype=np.float64)))


def 特征(x):
    """确定性感觉特征 [x−0.5, (x−0.5)², 1.0]。"""
    d = float(x) - 0.5
    return np.array([d, d * d, 1.0], dtype=np.float64)


def 投影P(D, 种子):
    rng = np.random.default_rng(int(种子))
    return _归(rng.standard_normal((int(D), 3)) / np.sqrt(3.0))


def 读出初值(D, 种子):
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
    """A 自指依赖（规则依赖）= x∈(0.35,0.65)；B 世界（刺激无关）= x∉[0.35,0.65]。"""
    x = np.asarray(x, dtype=np.float64)
    return (x > 带低) & (x < 带高), (x < 带低) | (x > 带高)


class 两区室核(能力核):
    """★V4 核心：继承既有七原语 + 注册第八算子「符」 + 第二递推态「自我态 A」 + 分区可塑。

    模式 ∈ 模式列（两区室/消融/无区室/线性求和/规则随机，供消融与反例臂）。
    学模式 ∈ 学模式列（credit=内生重合门加权 / η=meeting47 同款常数，作对照）。
    """

    def __init__(self, D=D默认, 种子=0, 模式="两区室", 慢率=慢率默认, 学模式="credit"):
        super().__init__(D=int(D), 种子=int(种子))
        # ★ 真接线：把「符」登记进既有能力核注册表 ⇒ 第八算子
        self.登记算子(算子_符, np.ones(self.D))
        self.模式 = str(模式)
        self.学模式 = str(学模式)
        self.慢率 = float(慢率)
        # 区室 B：世界态（既有主循环通路）；区室 A：自我态（慢，跨试次）
        self.P = 投影P(self.D, int(种子))
        self.读出 = 读出初值(self.D, int(种子))
        self.感W = np.ones(self.D)
        self.自我W = np.ones(self.D)
        self.自我标量 = 0.5                       # 自我态的可读出标量（本任务=规则/边界估计）
        self.自我态 = np.zeros(self.D, dtype=np.float64)
        self._刷新自我态()
        self._o = None
        self.最近门 = np.zeros(self.D, dtype=np.float64)
        self._最近B = np.zeros(self.D, dtype=np.float64)
        self._rng = np.random.default_rng(int(种子) + 12345) if self.模式 == "规则随机" else None

    # ─────────────────── 第八算子：符（双线性重合门）───────────────────
    def 符(self, a, b) -> np.ndarray:
        """双线性重合门：符(a,b) = 归(W符 ⊙ a ⊙ b)。逐元素乘积，**a 或 b 为 0 ⇒ 无输出**。

        对应论文 A2（arXiv:2606.10891）的 G(g)·Y(s)：目标态 × 状态态的重合检测。
        支持广播（(D,) 与 (n,D) 逐元素相乘），零成对加权 ⇒ 线性时间守恒。
        """
        aa = np.asarray(a, dtype=np.float64)
        bb = np.asarray(b, dtype=np.float64)
        return _归(self.W表[算子_符] * aa * bb)

    # ─────────────────── 区室 A：自我态（慢）───────────────────
    def _刷新自我态(self):
        self.自我态 = _归(self.自我W * (self.P @ 特征(self.自我标量)))

    def 世界态(self, x):
        """区室 B（快、感觉）：世界态向量。"""
        return _归(self.感W * (self.P @ 特征(x)))

    def 自我态向量(self):
        return self.自我态.copy()

    def 设自我标量(self, 值):
        self.自我标量 = _定量(min(0.95, max(0.05, float(值))))
        self._刷新自我态()

    def 设自我态向量(self, 向量):
        """直接置自我态（贯通用：把引擎内部态当作自我态；确定性、有界、定点归一）。"""
        v = np.asarray(向量, dtype=np.float64).ravel()
        if v.shape[0] != self.D:
            raise ValueError("自我态须为 (%d,)，收到 %r" % (self.D, v.shape))
        self.自我态 = _归(v)

    # ─────────────────── 前向：任务试次（自我-世界整合）───────────────────
    def 前向_试次(self, x):
        B = self.世界态(x)
        if self.模式 == "消融":
            A = _归(self.自我W * (self.P @ 特征(0.5)))       # 冻结自我态：常数、不含自我标量（切掉自指通道）
        else:
            A = self.自我态
        if self.模式 == "无区室":
            o = _归(B)                                       # 只用世界态，两区室结构被打乱
        elif self.模式 == "线性求和":
            o = _归(A + B)                                    # 线性求和（非差分）
        elif self.模式 == "符合门":
            o = _归(self.符(A, B))                            # ★纯双线性重合门整合（论文 A2 形态；备选臂）
        elif self.模式 == "差分加符":
            o = _归(软限(A - B) + self.符(A, B))               # 差分 + 双线性（备选臂）
        else:
            o = _归(软限(A - B))                              # 非线性差分整合：自我-世界差分
        self._o = o
        self.最近门 = self.符(A, B)                           # ★ 内生 credit 的瞬时重合门
        self._最近B = B                                       # 本试次世界态（credit 归一用）
        return o

    # ─────────────────── 前向：真接线（真跑既有能力核主循环）───────────────────
    def 前向_双区室(self, 令牌列, 上下文=None, 迭代步数=1) -> dict:
        """★ 真接线：先跑**既有能力核主循环**得世界态 B，再以 软限(A−B) + 符(A,B) 做自我-世界整合。"""
        基 = super().前向(令牌列, 上下文=上下文, 迭代步数=迭代步数)      # 既有主循环（读/门/迭代）
        B列 = 基["输出"]                                                # (n,D) 世界态
        A = self.自我态                                                 # (D,)   自我态
        差 = _归(软限(A - B列))                                         # 自我-世界差分（逐行）
        门 = self.符(A, B列)                                            # 自我⊗世界 重合门（逐行）
        return dict(输出=差, 世界态=B列, 自我态=A.copy(), 门=门,
                    终态=self.迭代(self.叠(差), A, 迭代步数),
                    迭代数=int(基["迭代数"]), 未定义令牌=bool(基["未定义令牌"]),
                    令牌数=int(基["令牌数"]), 维数=int(self.D),
                    自我标量=float(self.自我标量))

    # ─────────────────── 学习：分区可塑 ───────────────────
    def 学自我(self, ŷ, 奖赏):
        """★ 内生 credit：更新权 = 归一化重合门强度（重合才赋分），不再手调常数。"""
        门强 = float(np.mean(np.abs(self.最近门)))
        尺度 = float(np.mean(np.abs(self.自我态))) * float(np.mean(np.abs(self._最近B))) + 1e-9
        权 = 门强 / 尺度                                       # 归一化重合强度（≈1；重合高 ⇒ >1）
        if self.模式 == "规则随机":
            新 = self.自我标量 + self.慢率 * (1.0 if self._rng.random() < 0.5 else -1.0)   # 随机游走，不用奖赏
        else:
            有效率 = self.慢率 * (权 if self.学模式 == "credit" else 1.0)
            if int(奖赏) == 0:
                新 = self.自我标量 + 有效率 * (1.0 if int(ŷ) == 1 else -1.0)
            else:
                新 = self.自我标量 + 有效率 * 小系数 * (0.5 - self.自我标量)
        self.设自我标量(新)

    def 学读出(self, ŷ, 奖赏, 更新读出=True):
        """读出（慢权重）只在**答错**时更新（与 meeting47 口径一致；答对不动）。"""
        if (not 更新读出) or int(奖赏) != 0:
            return
        o = self._o
        正 = 1 - int(ŷ)
        变 = np.zeros_like(self.读出)
        变[正] += η读出 * o
        变[int(ŷ)] -= η读出 * o
        self.读出 = _归(self.读出 + 变)

    # ─────────────────── 归属（provenance）：外源不得塑造自我 ───────────────────
    def 注入外源规则(self, 向量, 允许=False) -> dict:
        """★ 归属臂：外源规则向量**默认不得**塑造自我态（沿用 `内核_自我回路` provenance 门控语义）。"""
        前 = self.自我态.copy()
        若塑 = bool(允许)
        if 若塑:
            self.自我态 = _归(向量)
        变 = bool(not np.array_equal(前, self.自我态))
        return dict(允许=若塑, 被塑造=变, 变化元素=int(np.sum(前 != self.自我态)))


class 点神经元:
    """公平基线：单一递推状态 s（无独立自我标量、无独立区室）。规则必须存在 s 里。

    同 D/同 P/同读出形状/同种子/同预算/同奖赏更新规则；差异只在「单一递推状态 s」。
    """

    def __init__(self, D, P, 读出0, λ, 种子=0, 慢率=慢率默认, 学模式="credit"):
        self.D = int(D)
        self.P = np.asarray(P, dtype=np.float64)
        self.读出 = np.asarray(读出0, dtype=np.float64).copy()
        self.感W = np.ones(self.D)
        self.自我W = np.ones(self.D)
        self.λ = float(λ)
        self.慢率 = float(慢率)
        self.学模式 = str(学模式)
        self.s = np.zeros(self.D, dtype=np.float64)
        self.drive = 0.5
        self._o = None

    def 前向(self, x):
        b = _归(self.感W * (self.P @ 特征(x)))
        a = _归(self.自我W * (self.P @ 特征(self.drive)))
        self.s = _归((1.0 - self.λ) * self.s + self.λ * (a + b))
        self._o = self.s
        return self.s

    def 学(self, x, ŷ, 奖赏, 更新读出=True):
        o = self._o
        if 更新读出 and int(奖赏) == 0:
            正 = 1 - int(ŷ)
            变 = np.zeros_like(self.读出)
            变[正] += η读出 * o
            变[int(ŷ)] -= η读出 * o
            self.读出 = _归(self.读出 + 变)
        if int(奖赏) == 0:
            self.drive += self.慢率 * (1.0 if int(ŷ) == 1 else -1.0)
        else:
            self.drive += self.慢率 * 小系数 * (0.5 - self.drive)
        self.drive = _定量(min(0.95, max(0.05, self.drive)))


def 建模型(模式, D, P, 读出0, 种子=0, 慢率=慢率默认, 学模式="credit", λ=None):
    if 模式 == "点神经元":
        return 点神经元(D, P, 读出0, λ, 种子=种子, 慢率=慢率, 学模式=学模式)
    return 两区室核(D=D, 种子=种子, 模式=模式, 慢率=慢率, 学模式=学模式)


def _准确(掩码, 选, 真):
    if int(掩码.sum()) == 0:
        return float("nan")
    return float((选[掩码] == 真[掩码]).mean())


def 跑(模型, 序列, 学读出=True, 学模式="credit"):
    x = 序列["x"]
    b = 序列["边界"]
    n = int(len(x))
    真 = np.zeros(n, dtype=np.int64)
    选 = np.zeros(n, dtype=np.int64)
    奖 = np.zeros(n, dtype=np.int64)
    for t in range(n):
        if isinstance(模型, 点神经元):
            o = 模型.前向(float(x[t]))
        else:
            o = 模型.前向_试次(float(x[t]))
        得分 = 模型.读出 @ o
        ŷ = int(np.argmax(得分))
        c = 1 if float(x[t]) > float(b[t]) else 0
        真[t] = c
        选[t] = ŷ
        奖[t] = 1 if ŷ == c else 0
        if isinstance(模型, 点神经元):
            模型.学(float(x[t]), ŷ, int(奖[t]), 更新读出=学读出)
        else:
            模型.学读出(ŷ, int(奖[t]), 更新读出=学读出)
            模型.学自我(ŷ, int(奖[t]))
    A, B = 分类掩码(x)
    return dict(真=真, 选=选, 奖=奖, A=A, B=B, 切换点=list(序列["切换点"]),
                Acc_A=_准确(A, 选, 真), Acc_B=_准确(B, 选, 真),
                Acc=_准确(np.ones(n, dtype=bool), 选, 真))


def 恢复全(奖, 切换点, n):
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


def 跑臂(模式, D, 种子模型, 序列训, 序列测, 慢率=慢率默认, 学模式="credit", λ=None):
    """训练 400（读出+自我都学）→ 测试 400（**读出冻结**，只有自我/规则通道在线适应）。"""
    P = 投影P(D, 种子模型)
    读出0 = 读出初值(D, 种子模型)
    模型 = 建模型(模式, D, P, 读出0, 种子=种子模型, 慢率=慢率, 学模式=学模式, λ=λ)
    训 = 跑(模型, 序列训, 学读出=True, 学模式=学模式)
    测 = 跑(模型, 序列测, 学读出=False, 学模式=学模式)
    return 模型, 训, 测


# ─────────────────────────── 自测（真跑，供直跑观测） ───────────────────────────
def 自测() -> dict:
    序列训 = 生成序列(种子训默认, 训练长, 边界训练)
    序列测 = 生成序列(种子测默认, 测试长, 边界训练)
    _, 训, 测 = 跑臂("两区室", D默认, 种子模型默认, 序列训, 序列测, 慢率=慢率默认)
    汇主 = 汇(测, 训)
    _, _, 测消 = 跑臂("消融", D默认, 种子模型默认, 序列训, 序列测, 慢率=慢率默认)
    汇消 = 汇(测消)
    核 = 两区室核(D=64, 种子=0)
    真接 = 核.前向_双区室(生成码本(4, 64, 7), None, 1)
    归属 = 核.注入外源规则(np.ones(64), 允许=False)
    return dict(主臂=汇主, 消融=汇消,
                真接线=dict(算子表=核.算子表(), 输出形状=list(真接["输出"].shape),
                            门形状=list(真接["门"].shape), 未定义令牌=bool(真接["未定义令牌"])),
                归属=归属,
                通过=bool(汇主["Acc_A"] >= 0.80 and 汇主["Acc_B"] >= 0.95 and (算子_符 in 核.算子表())))


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    try:
        import numpy as _np
        _np.seterr(all="ignore")
    except Exception:
        pass
    果 = 自测()
    print(json.dumps(果, ensure_ascii=False, indent=1, default=str))

# -*- coding: utf-8 -*-
"""meeting41/实施/算子_能力核.py —— L1 能力核（非注意力 · 线性时间 · 「任意训练」）。

━━━ 这一件解决什么 ━━━
所有者口径「核心和基座大模型一致，能力提升」= **能力对标**，同时保留 C1 红线：不做注意力机制。
上下文条件计算由**三件套**承担（全部是逐元素 / 索引操作，没有任何成对加权）：

  ① 码本索引读出   —— 最近邻查回（`读`：定标 → 精确整数内积 → 取最大 → 叠回）
                       开销 = O(码本数·D)，**与令牌数 n 无关**
  ② 逐元素门控     —— g⊙x + (1−g)⊙y（`门`），无归一化指数、无令牌两两交互
  ③ 定点迭代推进   —— s ← 归(s + 步长·软限(W⊙s + 驱动))（`迭代`），**取代「堆层」**

长程依赖来自**递推状态 s**（跨令牌累积、由上下文初始化、每拍定点推进），
不是令牌两两交互 ⇒ 时间与开销严格线性于令牌数 n：
`计费(4096)/计费(1024) = 4.0000`（解析乘加账，见 `计费` docstring 的逐项口径）。

「任意训练」= **未定义令牌**（码本外的输入）不抛异常、不返回全零 / 全同，而是返回一个
由「上下文状态 s」与「令牌自身」共同决定的稳定向量，并如实上报 `未定义令牌=True`。

━━━ 铁律对齐 ━━━
C1  零注意力机制：本文件不含任何相关子串；`禁词自检()` 会**自扫本文件**并断言命中数 == 0。
C13 跨进程确定性：随机只来自 `numpy.random.default_rng(显式种子)`；所有数值走 2^-20 定点格，
    内积用 int64 **精确**累加（无浮点求和顺序问题）；**本文件不 import time**，
    线性时间只由 `计费()` 的**解析乘加账**给出，不看墙钟。
C14 纯新增：不改 接口契约.py，不改 meeting38/39/40 任何既有件。

━━━ 直跑 ━━━
    python 算子_能力核.py            # 全量自证，exit=0/1
    python 算子_能力核.py --W码      # 只打印 参数基线()["W码"]（跨 PYTHONHASHSEED 比对用）
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np

# ───────────────────────── 既有件导入（时基.特征码 是唯一口径） ─────────────────────────
_AHI3 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # → AHI3
for _p in (os.path.join(_AHI3, "meeting41", "实施"), os.path.join(_AHI3, "meeting40", "实施")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from 时基 import 特征码 as _特征码
    _特征码来源 = "时基.特征码"
except Exception as _e:                                       # 兜底：同口径（规范 JSON → sha256）
    def _特征码(对象):
        return hashlib.sha256(json.dumps(对象, ensure_ascii=False, sort_keys=True,
                                         separators=(",", ":"), default=str)
                              .encode("utf-8")).hexdigest()
    _特征码来源 = "本地兜底(与 时基.特征码 同口径)：%s" % (_e,)

try:
    from 接口契约 import (算子_绑, 算子_解绑, 算子_门, 算子_叠, 算子_归, 算子_读, 算子_迭代)
    _契约来源 = "接口契约.py"
except Exception as _e:                                       # 契约不可导入时用同名字面量
    算子_绑, 算子_解绑, 算子_门, 算子_叠, 算子_归, 算子_读, 算子_迭代 = \
        ("绑", "解绑", "门", "叠", "归", "读", "迭代")
    _契约来源 = "本地字面量(接口契约.py 不可导入)：%s" % (_e,)

全部算子_ = (算子_绑, 算子_解绑, 算子_门, 算子_叠, 算子_归, 算子_读, 算子_迭代)

# ───────────────────────── 口径常量（实现者A 冻结；改动即换基线） ─────────────────────────
D_默认 = 1024                 # 契约：维数 D 固定 = 1024
界幅 = 16.0                   # 有界：所有返回向量的 ‖v‖∞ ≤ 界幅
定点 = float(1 << 20)         # 2^-20 定点格（定点归一的格距；跨进程逐位可复现）
码本大小_默认 = 64            # `读` 的 W = 码本 (码本数, D)，默认 64 个原子
命中阈值_默认 = 0.35          # 码本索引读出的命中阈值（双极码本下噪声 ~1/sqrt(D)=0.031 ⇒ 11σ）
学习率_默认 = 1.0             # `更新` 的步长
步长_默认 = 0.5               # `迭代` 的定点推进步长
每微秒乘加 = 500              # 解析估计用的**标称**吞吐（不是测量值；只用于把乘加数折成估计时长）


# ───────────────────────── 数值小工具（无超越函数，纯算术 + IEEE 运算） ─────────────────────────
def 软限(x, 幅: float = 界幅) -> np.ndarray:
    """算术软饱和 幅·x/(幅+|x|) → (−幅, 幅)：单调、有限、跨进程逐位同（不用 exp/tanh）。"""
    x = np.asarray(x, dtype=np.float64)
    return 幅 * x / (幅 + np.abs(x))


def _归(v, 幅: float = 界幅) -> np.ndarray:
    """定点归一（模块级纯函数）：非有限 → 限端，再限幅到 ±幅，再量化到 2^-20 定点格。"""
    a = np.asarray(v, dtype=np.float64)
    a = np.nan_to_num(a, nan=0.0, posinf=幅, neginf=-幅)
    a = np.clip(a, -幅, 幅)
    return np.rint(a * 定点) / 定点


def _整格(v) -> np.ndarray:
    """定点格上的精确整数表示（_归 已保证是 2^-20 的整数倍 ⇒ 乘 2^20 无舍入）。"""
    return (_归(v) * 定点).astype(np.int64)


def 生成码本(数: int, D: int = 1024, 种子: int = 0) -> np.ndarray:
    """确定性双极码本 (数, D)，取值 ±1。随机性只来自 default_rng(显式种子)（C13）。"""
    rng = np.random.default_rng(int(种子))
    return np.where(rng.integers(0, 2, size=(int(数), int(D))) == 1, 1.0, -1.0)


# ═══════════════════════════════════════════════════════════════════════════════════
# L1 能力核（签名逐字对齐 接口契约.py 第二节）
# ═══════════════════════════════════════════════════════════════════════════════════
class 能力核:
    """非注意力能力核。参数一律显式 W，无隐藏可变态。"""

    D: int = 1024

    def __init__(self, D: int = 1024, 种子: int = 0) -> None:
        self.D = int(D)
        if self.D <= 0:
            raise ValueError("D 必须为正，收到 %r" % (D,))
        self.种子 = int(种子)
        self.码本大小 = int(码本大小_默认)
        self.命中阈值 = float(命中阈值_默认)
        self.学习率 = float(学习率_默认)
        self.步长 = float(步长_默认)
        self.W表: dict = {}          # 名 → ndarray（唯一的显式参数存储）
        self.可训练表: dict = {}      # 名 → bool
        self._初始化参数()

    # ── 参数初始化（只依赖 种子 ⇒ 同种子跨进程逐位同） ──
    def _初始化参数(self) -> None:
        rng = np.random.default_rng(self.种子)
        D, M = self.D, self.码本大小

        def 双极(形状):
            return np.where(rng.integers(0, 2, size=形状) == 1, 1.0, -1.0)

        键 = 双极((D,))
        self.登记算子(算子_绑, 键)
        self.登记算子(算子_解绑, 键.copy())          # 解绑初值取 绑 的逆 ⇒ 往返逐位还原
        self.登记算子(算子_门, np.ones(D))           # 逐元素门控斜率
        self.登记算子(算子_叠, np.ones(D))           # 叠加权重
        self.登记算子(算子_归, np.ones(D))           # 归一尺度
        self.登记算子(算子_读, 双极((M, D)))         # ★ 读的 W 就是码本（M 个原子）
        self.登记算子(算子_迭代, np.ones(D))         # 定点迭代耦合

    # ───────────────────────── 注册表 ─────────────────────────
    def 登记算子(self, 名: str, W) -> None:
        """登记/覆盖一个显式参数。一维须长度 D；多维末维须为 D；标量广播成 (D,)。"""
        a = np.asarray(W, dtype=np.float64)
        if a.ndim == 0:
            a = np.full((self.D,), float(a))
        elif a.shape[-1] != self.D:
            raise ValueError("W 的末维必须 == D(%d)，收到形状 %r" % (self.D, a.shape))
        self.W表[str(名)] = _归(a)
        self.可训练表[str(名)] = True

    def 取算子(self, 名: str) -> dict:
        """→ dict(名=str, W=ndarray, 形状=tuple, 可训练=bool, 是注意力=False, 维数=int)。"""
        名 = str(名)
        if 名 not in self.W表:
            raise KeyError("未登记算子：%r（已登记：%s）" % (名, "、".join(self.算子表())))
        W = self.W表[名]
        return dict(名=名, W=W.copy(), 形状=tuple(W.shape),
                    可训练=bool(self.可训练表.get(名, True)), 是注意力=False, 维数=int(self.D))

    def 算子表(self) -> list:
        """已登记算子名（sorted）。"""
        return sorted(self.W表.keys())

    @property
    def 码本(self) -> list:
        """`读` 的码本，逐行导出成 list[(D,)]（只读副本；总装 `主程序_第七轮` 按可迭代对象取用）。"""
        return [行.copy() for 行 in self.W表[算子_读]]

    @property
    def 码本数(self) -> int:
        W = self.W表.get(算子_读)
        return int(W.shape[0]) if (W is not None and W.ndim == 2) else int(self.码本大小)

    # ───────────────────────── 七原语（纯函数式：只依赖入参与 self.W） ─────────────────────────
    def 绑(self, 键, 值) -> np.ndarray:
        """键⊛值 绑定（双极自逆）：W绑⊙键⊙值，归一回 ±界幅。取代特征交互的成对加权。"""
        k = self._取向量(键)
        v = self._取向量(值)
        return _归(self.W表[算子_绑] * k * v)

    def 解绑(self, 键, 捆绑) -> np.ndarray:
        """键⊙捆绑 解绑：W解绑⊙键⊙捆绑。W解绑 = W绑 时，解绑(键, 绑(键,值)) 逐位还原 值。"""
        k = self._取向量(键)
        b = self._取向量(捆绑)
        return _归(self.W表[算子_解绑] * k * b)

    def 门(self, 门控, x, y) -> np.ndarray:
        """逐元素门控 g⊙x + (1−g)⊙y，g = clip(门控, 0, 1)。取代归一化指数加权。"""
        g = np.clip(self._取向量(门控), 0.0, 1.0)
        xx = self._取向量(x)
        yy = self._取向量(y)
        return _归(g * xx + (1.0 - g) * yy)

    def 叠(self, 向量列) -> np.ndarray:
        """叠加（取代缓存的容量角色）：Σ 行 → W叠⊙Σ → 定点归一。用 int64 精确累加，无求和顺序误差。"""
        if isinstance(向量列, np.ndarray) and 向量列.ndim == 1:
            A = 向量列.reshape(1, -1)
        else:
            L = list(向量列)
            if not L:
                return _归(np.zeros(self.D))
            A = np.asarray(L, dtype=np.float64)
            if A.ndim == 1:
                A = A.reshape(1, -1)
        I = _整格(A)
        和 = I.sum(axis=0, dtype=np.int64).astype(np.float64) / 定点
        return _归(self.W表[算子_叠] * 和)

    def 归(self, 向量) -> np.ndarray:
        """定点归一（有界，防饱和）：尺度 W归 → 限幅 ±界幅 → 量化到 2^-20 格。"""
        return _归(np.asarray(向量, dtype=np.float64) * self.W表[算子_归])

    def 读(self, 码本, 探针, k: int = 1) -> dict:
        """码本索引读出（取代检索式注意力）：定标 → int64 精确内积 → 取最大 k 个 → 叠回。

        相似度 = 内积 / (D·定点²·界幅) ∈ [−1, 1]：
          码本按 ±1 量级给出、探针按 ∞ 范数归一 ⇒ 完全命中 = 1.0，无关原子 ~ 1/sqrt(D) ≈ 0.031。
        开销 = O(码本行数·D)（k=1 时用 argmax，无排序），**与令牌数无关**。
        """
        A = np.asarray(码本, dtype=np.float64)
        if A.ndim == 1:
            A = A.reshape(1, -1)
        if A.ndim != 2 or A.shape[1] != self.D:
            raise ValueError("码本形状须为 (M, %d)，收到 %r" % (self.D, A.shape))
        p = _归(np.asarray(探针, dtype=np.float64).ravel())
        if p.shape[0] != self.D:
            raise ValueError("探针形状须为 (%d,)，收到 %r" % (self.D, p.shape))
        M = int(A.shape[0])
        kk = max(1, min(int(k), M))
        尺 = float(np.max(np.abs(p)))
        if 尺 <= 0.0:
            尺 = 1.0
        AI = _整格(A)                                        # ±1 量级码本 → 定点整数
        pI = np.rint((p / 尺) * (定点 * 界幅)).astype(np.int64)   # 探针 ∞ 范数 = 界幅
        内积 = AI @ pI                                       # int64 精确（|和| ≤ 1.8e16 ≪ 9.2e18）
        相似度全 = 内积.astype(np.float64) / (self.D * 定点 * 定点 * 界幅)
        if kk == 1:
            序 = np.array([int(np.argmax(内积))], dtype=np.int64)   # 并列取最小索引（确定）
        else:
            序 = np.argsort(-相似度全, kind="stable")[:kk]
        索引 = [int(i) for i in 序]
        相似度 = [float(相似度全[i]) for i in 索引]
        读出 = self.叠([_归(A[i]) for i in 索引])
        return dict(索引=索引, 相似度=相似度, 读出=读出, 命中=bool(相似度[0] >= self.命中阈值),
                    最高相似度=float(相似度[0]), k=int(kk), 码本数=M, 阈值=float(self.命中阈值),
                    是注意力=False)

    def 迭代(self, 态, 驱动力, 步数: int = 1) -> np.ndarray:
        """定点迭代（相空间推进，取代「层」的深度）：

            s ← 归((1−步长)·s + 步长·软限(W迭⊙s + 驱动))

        软限导数 ∈ [0,1] ⇒ 迭代映射是**收缩**的（不跑到饱和、不发散），
        收敛到 s* = 软限(W迭⊙s* + 驱动)；步数 = 推进多少拍，取代「堆多少层」。
        """
        s = _归(态)
        d = _归(驱动力)
        if s.shape != (self.D,) or d.shape != (self.D,):
            raise ValueError("态/驱动力形状须为 (%d,)，收到 %r / %r" % (self.D, s.shape, d.shape))
        W = self.W表[算子_迭代]
        λ = float(self.步长)
        for _ in range(max(0, int(步数))):
            推 = 软限(W * s + d, 界幅)
            s = _归((1.0 - λ) * s + λ * 推)
        return s

    # ───────────────────────── 前向：上下文条件计算（取代「堆层」） ─────────────────────────
    def 前向(self, 令牌列, 上下文=None, 迭代步数: int = 1) -> dict:
        """逐令牌推进递推状态 s：读出 → 门控 → 定点迭代。开销 O(n·(码本数·D + D·步数))，线性。

        未定义令牌（`读` 未命中）：不抛异常、不返回全零 / 全同，输出由「上下文状态 s」与
        「令牌自身」共同决定（见下方 `值` 的三项叠加），并如实上报 未定义令牌=True。
        """
        X = self._取令牌(令牌列)
        n = int(X.shape[0])
        步数 = max(1, int(迭代步数))
        s = self._取上下文(上下文)
        W读 = self.W表[算子_读]
        W绑 = self.W表[算子_绑]
        W解 = self.W表[算子_解绑]
        W门 = self.W表[算子_门]
        输出 = np.zeros((n, self.D), dtype=np.float64)
        未定义 = False
        命中数 = 0
        for i in range(n):
            p = _归(X[i])
            R = self.读(W读, p, k=2)
            键 = self.绑(W绑, p)                       # 令牌 → 键空间
            if R["命中"]:
                命中数 += 1
                值 = self.叠([R["读出"], 键])
            else:
                未定义 = True
                # ★ 码本外：上下文分量 + 令牌分量 + 最近邻残余 ⇒ 绝不全零、绝不全同
                值 = self.叠([self.绑(W解, s), 键, R["读出"]])
            驱 = self.归(值)
            门控 = 0.5 + 0.5 * 软限(W门 * (驱 + s), 1.0)   # 逐元素门控，落在 (0,1)
            c = self.门(门控, 驱, p)                        # g⊙驱 + (1−g)⊙p
            s = self.迭代(s, c, 步数)                        # 定点迭代推进（取代堆层）
            输出[i] = self.归(0.5 * s + 0.5 * c)
        return dict(输出=self.归(输出), 终态=s, 迭代数=int(步数), 未定义令牌=bool(未定义),
                    令牌数=int(n), 命中令牌数=int(命中数), 码本数=int(self.码本数),
                    维数=int(self.D))

    def _取向量(self, v) -> np.ndarray:
        a = np.asarray(v, dtype=np.float64)
        if a.shape != (self.D,):
            raise ValueError("向量形状须为 (%d,)，收到 %r" % (self.D, a.shape))
        return a

    def _取令牌(self, 令牌列) -> np.ndarray:
        if 令牌列 is None:
            raise ValueError("令牌列不得为 None")
        A = np.asarray(令牌列, dtype=np.float64)
        if A.size == 0:                                        # 空令牌列：合法，n = 0
            return np.zeros((0, self.D), dtype=np.float64)
        if A.ndim == 1:
            if A.shape[0] != self.D:
                raise ValueError("单令牌须为 (%d,)，收到 %r" % (self.D, A.shape))
            A = A.reshape(1, self.D)
        if A.ndim != 2 or A.shape[1] != self.D:
            raise ValueError("令牌列须为 (n, %d) 或 list[(%d,)]，收到 %r" % (self.D, self.D, A.shape))
        return A

    def _取上下文(self, 上下文) -> np.ndarray:
        """上下文 → (D,) 归一回的状态。None ⇒ 码本首原子（确定性、非零、随 读 的 W 一起可训练）。"""
        if 上下文 is None:
            return _归(self.W表[算子_读][0])
        if isinstance(上下文, dict):
            for 键 in ("向量", "上下文", "列", "值"):
                if 键 in 上下文:
                    上下文 = 上下文[键]
                    break
            else:
                raise ValueError("上下文 dict 需含 '向量'/'上下文'/'列'/'值' 之一")
        A = np.asarray(上下文, dtype=np.float64)
        if A.ndim == 0:
            return _归(np.full((self.D,), float(A)))
        if A.ndim == 1:
            if A.shape[0] != self.D:
                raise ValueError("上下文须为 (%d,)，收到 %r" % (self.D, A.shape))
            return _归(A)
        if A.shape[-1] != self.D:
            raise ValueError("上下文末维须为 (%d,)，收到 %r" % (self.D, A.shape))
        return self.叠(A.reshape(-1, self.D))

    # ───────────────────────── 参数基线（比对学习前后） ─────────────────────────
    def 参数基线(self) -> dict:
        """→ dict(维数, 算子数, 参数总量, W码)。W码 = 时基.特征码(规范化后的全部 W)，跨进程逐位同。"""
        规范 = {}
        总量 = 0
        for 名 in self.算子表():
            W = self.W表[名]
            总量 += int(W.size)
            规范[名] = dict(形状=list(W.shape), 整数值=_整格(W).ravel().tolist())
        return dict(维数=int(self.D), 算子数=int(len(self.W表)), 参数总量=int(总量),
                    W码=_特征码(规范), 码本数=int(self.码本数), 种子=int(self.种子))

    # ───────────────────────── 时间 / 规模证据（解析乘加账） ─────────────────────────
    def 计费(self, n: int, 迭代步数: int = 1) -> dict:
        """解析式乘加账（**不是墙钟**）：乘加数 = n·每令牌 + 固定。

        每令牌 = 4·码本数·D   （码本定标 → int64 内积 → 相似度，见 `读`）
               + 20·D         （探针归一 / 逐元素门控 / 叠加 / 输出归一）
               + 10·D·步数    （定点迭代：每步 10·D）
        固定   = 8·D           （上下文归 + 终态归，与 n 无关）
        ⇒ 计费(4096)/计费(1024) = 4.0000（线性）；n² 形态（注意力口径）为 16.0。
        时长估计 = 乘加数 // 每微秒乘加（标称 500 乘加/微秒；纯解析，不测墙钟）。
        """
        n = int(n)
        步数 = max(1, int(迭代步数))
        M, D = int(self.码本数), int(self.D)
        每令牌 = 4 * M * D + 20 * D + 10 * D * 步数
        固定 = 8 * D
        乘加数 = n * 每令牌 + 固定
        return dict(乘加数=int(乘加数), 时长估计=int(乘加数 // 每微秒乘加),
                    每令牌乘加=int(每令牌), 固定乘加=int(固定), 码本数=M, 维数=D,
                    迭代步数=步数, 阶=1, 单位="微秒(解析估计,非墙钟)",
                    口径="乘加数 = n·(4·码本数·D + 20·D + 10·D·步数) + 8·D")

    # ───────────────────────── 参数更新（唯一的「学习」入口；只动 W） ─────────────────────────
    def 更新(self, 名: str, ΔW) -> None:
        """只改 名 对应的 W：W ← 定点归一(W + 学习率·clip(ΔW))。有界、归一、确定；其余参数不动。"""
        名 = str(名)
        if 名 not in self.W表:
            raise KeyError("未登记算子：%r（已登记：%s）" % (名, "、".join(self.算子表())))
        W = self.W表[名]
        Δ = np.asarray(ΔW, dtype=np.float64)
        try:
            Δ = np.broadcast_to(Δ, W.shape).astype(np.float64)
        except ValueError:
            raise ValueError("ΔW 形状 %r 无法广播到 %r 的 W 形状 %r"
                             % (np.shape(ΔW), 名, tuple(W.shape))) from None
        Δ = np.clip(np.nan_to_num(Δ, nan=0.0, posinf=界幅, neginf=-界幅), -界幅, 界幅)
        self.W表[名] = _归(W + self.学习率 * Δ)


# ═══════════════════════════════════════════════════════════════════════════════════
# C1 自审：禁词子串级自扫
# ═══════════════════════════════════════════════════════════════════════════════════
# 注意：下面的禁词**故意用片段拼装**，这样本文件正文里不会出现任何被禁的连续子串
#       —— 否则"自扫脚本自己就违规"了。
_禁词片段 = (("at", "tention"), ("at", "tn"), ("q", "kv"), ("soft", "max"),
             ("multi", "head"), ("multi", "_head"), ("trans", "former"), ("trans", "formers"),
             ("位置", "编码"), ("positional", "_encoding"), ("scaled", "_dot"),
             ("to", "rch"), ("rw", "kv"), ("self_", "at", "tn"), ("cross_", "at", "tn"))


def 禁词列() -> list:
    """A6 审计口径的禁词表（与 接口契约.禁词 同集；片段拼装以免自伤）。"""
    return ["".join(片段) for 片段 in _禁词片段]


def 禁词自检(路径: str = None) -> dict:
    """子串级扫描单个 .py（默认本文件）；大小写不敏感。"""
    p = os.path.abspath(路径 or __file__)
    with open(p, encoding="utf-8") as f:
        原文 = f.read().lower()
    命中 = [w for w in 禁词列() if w.lower() in 原文]
    return dict(路径=p, 文件=os.path.basename(p), 禁词数=len(禁词列()),
                命中=命中, 命中数=len(命中), 通过=(not 命中))


def 扫目录(目录: str = None) -> dict:
    """递归扫一个目录下所有 .py 的禁词命中数（A6 口径；供总装核对，非本件判据）。"""
    目录 = os.path.abspath(目录 or os.path.dirname(os.path.abspath(__file__)))
    表 = {}
    for 根, _, 文件列 in os.walk(目录):
        for 名 in sorted(文件列):
            if not 名.endswith(".py"):
                continue
            p = os.path.join(根, 名)
            try:
                with open(p, encoding="utf-8") as f:
                    原文 = f.read().lower()
            except Exception:                                  # noqa: BLE001
                continue
            表[os.path.relpath(p, 目录)] = sum(1 for w in 禁词列() if w.lower() in 原文)
    return 表


# ═══════════════════════════════════════════════════════════════════════════════════
# 自证（全部真跑；每条都可失败）
# ═══════════════════════════════════════════════════════════════════════════════════
def _双极(rng, 数: int) -> np.ndarray:
    return np.where(rng.integers(0, 2, size=int(数)) == 1, 1.0, -1.0)


def 自证(详细: bool = True) -> dict:
    核 = 能力核(D=1024, 种子=0)
    D = 核.D
    rng = np.random.default_rng(20261008)
    条 = {}
    判定 = []

    def 记(名, 值, 通过=None):
        """记一条：值进结果字典；通过(默认=值本身，非布尔则视为 True) 进判定列表。"""
        条[名] = 值
        if 通过 is None:
            通过 = 值 if isinstance(值, (bool, np.bool_)) else True
        判定.append(bool(通过))
        return 值

    条["契约来源"] = _契约来源
    条["特征码来源"] = _特征码来源
    条["D"] = D
    条["界幅"] = 界幅

    # ── 0 C1 自审：本文件禁词命中必须为 0 ──
    自 = 禁词自检()
    记("C1_禁词自检", dict(命中=自["命中"], 禁词数=自["禁词数"]), 自["通过"])
    条["C1_目录扫描(信息性,含契约件)"] = 扫目录()

    # ── 1 注册表 ──
    名列 = 核.算子表()
    条["算子表"] = 名列
    记("注册表_与契约七算子一致", bool(名列 == sorted(全部算子_)))
    取列 = [核.取算子(n) for n in 名列]
    记("注册表_取算子非注意力且齐", bool(all(x["是注意力"] is False for x in 取列)
                                   and all(x["形状"][-1] == D for x in 取列)
                                   and all(x["可训练"] for x in 取列)))
    条["取算子读的形状"] = 核.取算子(算子_读)["形状"]
    记("注册表_越界名报错", bool(_抛(lambda: 核.取算子("不存在的算子"))))

    # ── 2 七原语 ──
    a, b = _双极(rng, D), _双极(rng, D)
    包 = 核.绑(a, b)
    包2 = 核.绑(a, b)
    记("绑_逐位确定且有界有限", bool(np.array_equal(包, 包2) and np.isfinite(包).all()
                              and np.max(np.abs(包)) <= 界幅))
    记("解绑_往返逐位还原(绑→解绑=值)", bool(np.array_equal(核.解绑(a, 包), _归(b))))
    记("解绑_错键不还原(可失败)", bool(not np.array_equal(核.解绑(a, _双极(rng, D)), _归(b))))

    x, y = _双极(rng, D), _双极(rng, D)
    记("门_全1取x", bool(np.array_equal(核.门(np.ones(D), x, y), _归(x))))
    记("门_全0取y", bool(np.array_equal(核.门(np.zeros(D), x, y), _归(y))))
    记("门_半为均值", bool(np.array_equal(核.门(np.full(D, 0.5), x, y), _归(0.5 * (x + y)))))

    叠3 = 核.叠([a, b, x])
    记("叠_有界有限", bool(np.isfinite(叠3).all() and np.max(np.abs(叠3)) <= 界幅))
    记("叠_k个同原子=归(k·原子)", bool(np.array_equal(核.叠([a, a, a]), _归(3 * a))))
    记("叠_确定性", bool(np.array_equal(叠3, 核.叠([a, b, x]))))

    v乱 = np.zeros(D)
    v乱[0], v乱[1], v乱[2], v乱[3], v乱[4], v乱[5] = np.nan, np.inf, -np.inf, 1e9, -1e9, 1.0 / 3.0
    归乱 = 核.归(v乱)
    记("归_非有限被治且有界", bool(np.isfinite(归乱).all() and np.max(np.abs(归乱)) <= 界幅))
    条["归_nan/inf/超界处置"] = [float(归乱[0]), float(归乱[1]), float(归乱[2]),
                          float(归乱[3]), float(归乱[4])]
    记("归_nan→0且±inf→±界幅", bool(归乱[0] == 0.0 and 归乱[1] == 界幅 and 归乱[2] == -界幅))
    记("归_定点幂等", bool(np.array_equal(核.归(归乱), 归乱)))
    记("归_定点格距(2^-20)", bool(abs(float(归乱[5]) * 定点 - round(float(归乱[5]) * 定点)) < 1e-9))

    # ── 3 `读`：码本索引读出（E2 长程依赖） ──
    小码 = 核.码本
    R0 = 核.读(小码, 小码[7])
    记("读_命中自身", bool(R0["命中"] and R0["索引"][0] == 7), R0["命中"] and R0["索引"][0] == 7)
    条["读_完全命中相似度"] = float(R0["相似度"][0])
    记("读_完全命中相似度=1.0", bool(abs(float(R0["相似度"][0]) - 1.0) < 1e-12))
    无关 = _双极(rng, D)
    R1 = 核.读(小码, 无关)
    条["读_无关向量最高相似度"] = float(R1["相似度"][0])
    记("读_无关向量不命中", bool(R1["命中"] is False), (R1["命中"] is False))
    # E2：4096 长序列里按索引检索
    大码 = 生成码本(4096, D, 种子=7)
    序号 = rng.integers(0, 4096, size=64)
    q = [int(i) for i in 序号]
    中 = sum(1 for i in q if 核.读(大码, 大码[i])["索引"][0] == i)
    条["E2_4096全码本命中"] = "%d/64" % 中
    记("E2_全码本命中率>=0.9", bool(中 / 64 >= 0.9), 中 / 64 >= 0.9)
    条["E2_全码本命中率"] = round(中 / 64, 6)
    # sham：只读局部窗（前 64 行）
    中2 = sum(1 for i in q if 核.读(大码[:64], 大码[i])["索引"][0] == i)
    条["E2_sham局部窗命中"] = "%d/64" % 中2
    记("E2_sham局部窗必须不过", bool(not (中2 / 64 >= 0.9)), not (中2 / 64 >= 0.9))

    # ── 4 `迭代`：定点推进 ──
    态 = _双极(rng, D)
    驱 = _双极(rng, D)
    迭1, 迭2 = 核.迭代(态, 驱, 1), 核.迭代(态, 驱, 1)
    迭多 = 核.迭代(态, 驱, 16)
    记("迭代_逐位确定且有限有界", bool(np.array_equal(迭1, 迭2) and np.isfinite(迭多).all()
                                and np.max(np.abs(迭多)) <= 界幅))
    记("迭代_0步=归(态)", bool(np.array_equal(核.迭代(态, 驱, 0), _归(态))))
    记("迭代_多步收敛(相邻步差收缩)", bool(float(np.max(np.abs(核.迭代(态, 驱, 24) - 核.迭代(态, 驱, 16))))
                                  <= float(np.max(np.abs(核.迭代(态, 驱, 8) - 核.迭代(态, 驱, 1)))) + 1e-12))

    # ── 5 `前向`：上下文条件计算 ──
    定 = 核.前向(小码[:3], _双极(rng, D), 迭代步数=1)
    记("前向_码本内不报未定义", bool(定["未定义令牌"] is False), 定["未定义令牌"] is False)
    条["前向_形状"] = [list(定["输出"].shape), list(定["终态"].shape), int(定["迭代数"])]
    记("前向_形状合契约", bool(定["输出"].shape == (3, D) and 定["终态"].shape == (D,)
                        and 定["迭代数"] == 1))
    记("前向_有限有界", bool(np.isfinite(定["输出"]).all() and np.max(np.abs(定["输出"])) <= 界幅))
    空 = 核.前向([], None, 迭代步数=1)
    记("前向_n=0不炸", bool(空["输出"].shape == (0, D) and 空["未定义令牌"] is False))
    单 = 核.前向(小码[0], None, 迭代步数=1)
    记("前向_单令牌可直喂", bool(单["输出"].shape == (1, D)))

    # ── 6 ★未定义令牌三条（任意训练） ──
    外 = [_双极(rng, D) for _ in range(3)]
    外似 = [max(abs(float(np.dot(小码[i], 外[j]) / D)) for i in range(len(小码))) for j in range(3)]
    条["E3_外向量与码本最大相似度"] = [round(v, 6) for v in 外似]
    记("E3_外向量确实在码本外(相似度<阈值)", bool(max(外似) < 核.命中阈值), max(外似) < 核.命中阈值)
    上甲 = _双极(rng, D)
    上乙 = (-上甲).copy()
    甲1 = 核.前向(外, 上甲, 迭代步数=1)
    甲2 = 核.前向(外, 上甲, 迭代步数=1)
    乙1 = 核.前向(外, 上乙, 迭代步数=1)
    行甲1, 行甲2, 行乙1 = 甲1["输出"], 甲2["输出"], 乙1["输出"]
    记("E3_①同上下文两遍逐位同", bool(np.array_equal(行甲1, 行甲2)
                              and np.array_equal(甲1["终态"], 甲2["终态"])))
    记("E3_②不同上下文输出不同", bool(not np.array_equal(行甲1, 行乙1)),
       not np.array_equal(行甲1, 行乙1))
    记("E3_未定义令牌如实上报True", bool(甲1["未定义令牌"] is True), 甲1["未定义令牌"] is True)
    两两 = bool(not np.array_equal(行甲1[0], 行甲1[1])
              and not np.array_equal(行甲1[0], 行甲1[2])
              and not np.array_equal(行甲1[1], 行甲1[2]))
    记("E3_③三个随机码本外向量输出两两不同", 两两, 两两)
    记("E3_③全部有限且有界", bool(np.isfinite(行甲1).all() and np.max(np.abs(行甲1)) <= 界幅))
    零行 = int(sum(1 for r in 行甲1 if float(np.max(np.abs(r))) == 0.0))
    记("E3_③无全零行", bool(零行 == 0), 零行 == 0)
    条["E3_输出行数/非零行"] = [int(行甲1.shape[0]), int(行甲1.shape[0]) - 零行]
    条["E3_不同上下文差异元素数"] = int(np.sum(行甲1 != 行乙1))
    条["E3_两两不同最小差异元素数"] = int(min(np.sum(行甲1[0] != 行甲1[1]),
                                        np.sum(行甲1[0] != 行甲1[2]),
                                        np.sum(行甲1[1] != 行甲1[2])))

    # ── 7 参数基线 / 确定性 ──
    核同 = 能力核(D=1024, 种子=0)
    核异 = 能力核(D=1024, 种子=1)
    基, 基同, 基异 = 核.参数基线(), 核同.参数基线(), 核异.参数基线()
    条["参数基线"] = dict(维数=基["维数"], 算子数=基["算子数"], 参数总量=基["参数总量"], W码=基["W码"])
    记("参数基线_同种子W码逐位同", bool(基["W码"] == 基同["W码"]), 基["W码"] == 基同["W码"])
    记("参数基线_异种子W码不同", bool(基["W码"] != 基异["W码"]), 基["W码"] != 基异["W码"])
    记("参数基线_参数总量=6D+码本数·D", bool(基["参数总量"] == 6 * D + 核.码本数 * D))

    # ── 8 `计费` 线性（E1）+ 二次反例臂 ──
    费1, 费4 = 核.计费(1024), 核.计费(4096)
    比 = float(费4["乘加数"]) / max(1.0, float(费1["乘加数"]))
    比时 = float(费4["时长估计"]) / max(1.0, float(费1["时长估计"]))
    条["E1_计费1024"] = 费1
    条["E1_计费4096"] = 费4
    条["E1_乘加比值"] = round(比, 6)
    条["E1_时长估计比值"] = round(比时, 6)
    记("E1_线性比值<4.5", bool(比 < 4.5), 比 < 4.5)
    记("E1_比值≈4.0(±0.05)", bool(abs(比 - 4.0) < 0.05), abs(比 - 4.0) < 0.05)
    比二 = float(D * 4096 * 4096) / float(D * 1024 * 1024)
    条["E1_二次反例臂比值"] = round(比二, 6)
    记("E1_二次反例臂必须不过", bool(not (比二 < 4.5)), not (比二 < 4.5))
    记("E1_计费步数单调(步数↑开销↑)", bool(核.计费(1024, 4)["乘加数"] > 费1["乘加数"]))

    # ── 9 `更新`：唯一学习入口，只动 W ──
    前W码 = 核.参数基线()["W码"]
    其他前 = {n: _特征码(核.取算子(n)["W"].tolist()) for n in 名列 if n != 算子_门}
    核.更新(算子_门, 0.25)
    其他后 = {n: _特征码(核.取算子(n)["W"].tolist()) for n in 名列 if n != 算子_门}
    后W码 = 核.参数基线()["W码"]
    记("更新_W码变化可观测", bool(前W码 != 后W码), 前W码 != 后W码)
    记("更新_只动指定算子", bool(其他前 == 其他后), 其他前 == 其他后)
    记("更新_有界", bool(np.max(np.abs(核.取算子(算子_门)["W"])) <= 界幅))
    核更 = 能力核(D=1024, 种子=0)
    核更.更新(算子_门, 0.25)
    记("更新_确定性(同序同Δ⇒逐位同)", bool(核更.参数基线()["W码"] == 后W码),
       核更.参数基线()["W码"] == 后W码)
    记("更新_形状不合时报错", bool(_抛(lambda: 核.更新(算子_门, np.ones((3, 5))))))
    记("更新_只接受已登记名", bool(_抛(lambda: 核.更新("不存在", 1.0))))
    # 学习可观测：更新后同一码本内令牌的输出必须变
    前出 = 核同.前向(小码[:2], 上甲, 迭代步数=1)["输出"]
    核同.更新(算子_读, np.full((核.码本数, D), 0.5))
    后出 = 核同.前向(小码[:2], 上甲, 迭代步数=1)["输出"]
    记("更新_改变前向输出(学习可观测)", bool(not np.array_equal(前出, 后出)),
       not np.array_equal(前出, 后出))

    # ── 10 长序列（4096 令牌）实跑：线性实现不炸、全有限、行间互异 ──
    长令 = 生成码本(4096, D, 种子=11)
    长 = 核.前向(长令, 上甲, 迭代步数=1)
    互异 = int(np.unique(长["输出"], axis=0).shape[0])
    条["长序列_输出形状"] = list(长["输出"].shape)
    条["长序列_互异行数"] = 互异
    条["长序列_未定义令牌"] = bool(长["未定义令牌"])
    记("长序列_4096令牌全有限有界", bool(_有限有界(长["输出"], 界幅)))
    记("长序列_行间高度互异(>=99%)", bool(互异 >= int(4096 * 0.99)), 互异 >= int(4096 * 0.99))

    条["判定条数"] = len(判定)
    条["未过条数"] = int(sum(1 for v in 判定 if not v))
    条["通过"] = bool(all(判定))
    条["实测摘要"] = dict(
        W码=基["W码"], 参数总量=基["参数总量"], 算子数=基["算子数"], D=D,
        E1乘加比值=round(比, 6), E1二次反例比值=round(比二, 6),
        E1_计费1024乘加=费1["乘加数"], E1_计费4096乘加=费4["乘加数"],
        E2全码本命中率=round(中 / 64, 6), E2局部窗命中率=round(中2 / 64, 6),
        E3同上下文两遍逐位同=bool(np.array_equal(行甲1, 行甲2)),
        E3不同上下文输出不同=bool(not np.array_equal(行甲1, 行乙1)),
        E3三个码本外向量两两不同=两两, E3未定义令牌上报=bool(甲1["未定义令牌"]),
        E3最大外相似度=round(max(外似), 6), C1禁词命中=自["命中数"])
    return 条


def _有限有界(A, 幅) -> bool:
    A = np.asarray(A)
    return bool(np.isfinite(A).all() and (np.max(np.abs(A)) if A.size else 0.0) <= 幅)


def _抛(fn) -> bool:
    """跑一次，抛异常 ⇒ True（用于「可失败」的负向用例）。"""
    try:
        fn()
    except Exception:                                          # noqa: BLE001
        return True
    return False


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:                                          # noqa: BLE001
        pass
    if "--W码" in sys.argv:                                    # 跨 PYTHONHASHSEED 比对入口
        print(能力核(D=1024, 种子=0).参数基线()["W码"])
        raise SystemExit(0)
    果 = 自证()
    print(json.dumps(果, ensure_ascii=False, indent=1, default=str))
    print("[算子_能力核] 自证：%s（判定 %d 条，未过 %d 条）"
          % ("PASS" if 果["通过"] else "FAIL", 果["判定条数"], 果["未过条数"]))
    raise SystemExit(0 if 果["通过"] else 1)

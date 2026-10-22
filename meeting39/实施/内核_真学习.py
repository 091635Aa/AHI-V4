# -*- coding: utf-8 -*-
"""内核_真学习.py —— meeting39 步4：真学习闭环（ε_self → 内生偏置 b[]；默认关）。

设计出处（meeting39 会议纪要 §2 共识；专家01 §2.2/§3.2/§3.3/§3.4；专家05 §1.2/§2 步4）：
  · 真学习 = 「预测自己 → 算误差 ε_self → 用误差实时改自己」。步2 已产出 ε_self 并存入
    **独立学习存储**（自模型 ĝ 的 W + 迹）；本步把 ε_self **接回引擎**——但**只准**经既有
    `内生偏置 b[]` 通路（与 κd 同构），**禁止直写** a_prev/i_prev/prev_c/消化债/饱足 等状态量。
  · 记忆 ≠ 学习（所有者硬命题）：记忆 = `codec._段`（记着）；学习 = ĝ 参数 + ε_self 历史（变会）。
    本文件所有新状态（ε史）与 `codec._段` **物理分离**，绝不写入记忆库。
  · 零参数·自我调控（专家01 §3.3）：**不新增人挑阈值**——门限取 ε_self **自身的慢 EMA**
    （自归一，"高于自己平时的水平才算异常"）；增益复用 内生驱动器 既有预注册常数 KAPPA=50.0
    （ε_self 与 ê 同为 [0,2] 量纲），**零新超参**。
  · 不破既有（C14）：唯一覆盖点 = 综合体._引擎步（**只换偏置源、只加挂**）；开关默认关时
    `self.drv` 保持原 内生驱动器 实例 ⇒ 逐位恒等（D1/D9）。本文件不 import 任何器官（冻结）。

时间因果（对齐 内生驱动器 纪律）：`偏置()` 只消费 **t−1 及之前** 的 ε_self（吸收发生在
「本拍动作选择完成后」）——先偏置后动作，可对账。

纪律：C1 禁 Transformer｜C9 禁 OCR｜C10 禁外挂模型当器官｜C13 确定性可复现｜C14 不破既有。
用法（测试）：见 步4_真学习闭环自证.py。
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from 内核_相空间算子T import 顶覆盖_综合体T, 开关默认 as _步3开关默认   # noqa: E402

# 内生驱动器（既有 novelty 偏置发生器，meeting17）——只**复用**其基类与预注册常数
_ROOT = os.path.dirname(os.path.dirname(_HERE))          # AHI3 根（迁移：随文件定位）
_P17 = os.path.join(_ROOT, "meeting17")
if _P17 not in sys.path:
    sys.path.insert(0, _P17)
from m2_intrinsic import 内生驱动器, KAPPA, ALPHA_SLOW          # noqa: E402

开关默认 = False           # ★保护开关默认关（C14/D1：默认关 ⇒ 既有输出逐位恒等）
κ_ε = KAPPA                # 复用 内生驱动器 预注册常数（ε_self∈[0,2] 与 ê 同量纲；零新超参）
ε同构系数 = 0.6             # 与 synthesis._引擎步 κd 的 b 加项系数同构（非新数值，形式对齐）


class 驱动_ε(内生驱动器):
    """内生驱动器 ε 版：在既有 novelty 偏置之上**再加一项** ε_self 驱动的偏置。

    与 κd 同构（synthesis.py:529-530）：ε_self 高于自身慢基线（自模型预测失灵＝不确定）
    ⇒ 偏向 停下想/反刍（b[1]）与 联想检索（b[3]）。
      · 门限 = ε_self 自身的慢 EMA（自归一，零人挑阈值）——对齐专家01 §3.3；
      · 增益 = 内生驱动器既有 κ（零新超参）；
      · ε史 = ★独立学习存储（与记忆库 codec._段 物理分离；所有者硬命题）。
    """

    def __init__(self, n候选: int = 9, α_slow: float = ALPHA_SLOW, κ: float = KAPPA,
                 初值: float = 1.0, α_meta: float = 0.02, γ_m: float = 0.5):
        super().__init__(n候选=n候选, α_slow=α_slow, κ=κ, 初值=初值,
                         α_meta=α_meta, γ_m=γ_m)
        self.ε_自身 = None            # 最近一次吸收的 ε_self（t−1 因果承载）
        self.ε_slow = 0.0             # ε_self 的慢 EMA 基线（初值 0=不预设；第一条 ε 即显"异常"）
        self.ε史: list = []           # ★独立学习存储（非 codec._段）
        self.ε位数 = 0

    @classmethod
    def 提升(cls, 旧: 内生驱动器) -> "驱动_ε":
        """把既有 内生驱动器 实例**原状态搬入** ε 版（不新增语义常数，只加挂）。"""
        新 = cls(旧.n, α_slow=旧.α, κ=旧.κ)
        新.slow = list(旧.slow)
        新.last_ê = list(旧.last_ê)
        新.meta_err = list(旧.meta_err)
        新.α_m = 旧.α_m
        新.γ_m = 旧.γ_m
        新.拍数 = 旧.拍数
        return 新

    # ── 本拍动作选择**完成后**调用：ε_self 进独立学习存储 + 更新慢基线（下一拍的因果原料） ──
    def 吸收ε(self, ε) -> None:
        if ε is None:
            return
        e = float(ε)
        self.ε史.append(e)
        self.ε_自身 = e
        self.ε_slow = self.ε_slow + self.α * (e - self.ε_slow)     # 慢 EMA（α=既有 α_slow）
        self.ε位数 += 1

    # ── 本拍动作选择**前**调用：在既有 novelty 偏置上叠加 ε_self 项 ──
    def 偏置(self) -> list:
        b = list(super().偏置())
        if self.ε_自身 is not None and len(b) > 3:
            项 = κ_ε * max(0.0, self.ε_自身 - self.ε_slow)          # 相对自身基线的"异常量"
            if 项 > 0.0:
                b[1] += 项 * ε同构系数          # 提问位→停下想/反刍（与 κd 同构）
                b[3] += 项 * ε同构系数          # 转移话题位→联想检索（与 κd 同构）
        return b


# ────────────────────────── synthesis 顶覆盖（只加挂；基链一行不改） ──────────────────────────

class 顶覆盖_综合体L(顶覆盖_综合体T):
    """步4 宿主：在步3 宿主之上**再**把 ε_self 经 内生偏置 b[] 接回引擎（默认关）。

    · 学习开关=False（默认）：`self.drv` 保持原 内生驱动器 实例 ⇒ 偏置通路与基类逐位同构
      （_引擎步 只多一次 getattr 守卫，不改任何状态/决策）。
    · 学习开关=True：`self.drv` 换成 驱动_ε（原状态搬入）。每拍 `_引擎步` 末尾吸收本拍 ε_self，
      下一拍 `偏置()` 用它偏向 停下想/联想。**引擎状态推进行一行不改**（不直写 a/i/prev_c）。
    """

    def __init__(self, seed: int = 310, 私偏κ: float = 10.0, 学习开关=None,
                 思核开关=None, 自模型开关=None, 学习: bool = True,
                 状态投影=None, 思核参: dict = None):
        # 先置属性——super().__init__ 期间 _引擎步 可能已被调用（见 synthesis.py:89 注释），
        # 那时 学习开关 已就位但 self.drv 尚未换成 ε 版 ⇒ 自然走原通路。
        self.学习开关 = bool(开关默认 if 学习开关 is None else 学习开关)
        self._ε源 = None                        # 外部 ε 源（单变量法 A4 用；None=用自模型产的 ε）
        super().__init__(seed=seed, 私偏κ=私偏κ, 思核开关=思核开关,
                         自模型开关=自模型开关, 学习=学习,
                         状态投影=状态投影, 思核参=思核参)
        if self.学习开关:
            drv = getattr(self, "drv", None)
            if drv is not None and not isinstance(drv, 驱动_ε):
                self.drv = 驱动_ε.提升(drv)     # 只换偏置源，状态原样搬入

    def _引擎步(self, u):
        rec = super()._引擎步(u)                 # 基链状态推进 + 步2 自模型只读喂入，逐行不变
        drv = getattr(self, "drv", None)
        if self.学习开关 and isinstance(drv, 驱动_ε):
            ε = None
            if self._ε源 is not None:            # A4 单变量法：外部指定 ε（不读自模型）
                ε = self._ε源(self)
            elif self.自模型 is not None and self.自模型.迹:
                ε = self.自模型.迹[-1].get("ε_self")
            if ε is not None:
                drv.吸收ε(ε)                      # 只写独立学习存储；下一拍才影响偏置
        return rec
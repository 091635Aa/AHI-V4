# -*- coding: utf-8 -*-
"""内核_自我回路.py —— meeting39 步5：自我回路融合（自指束留痕 + provenance 门控；默认关）。

设计出处（meeting39 会议纪要 §2 共识；专家03 §2.1/§2.2/§3.0/§3.4；专家05 §2 步5 契约）：
  步5 契约原文：`每 ε_self/深度顿悟 经自指束留痕（复用 _自言）；provenance=来源标记`
              `依赖 步2、步3 ｜ 验收 V4 增强但不破；A4 可测 ｜ 回退 开关置 false`。

  · 自指束 À_t = 束[绑(角色_r, 字段_r)]（专家03 §2.1「此刻的我」；全 ±1 双极、纯 HDC）：
      角色族 + 持久身份锚 a_seal 由**固定档案 seed**的独立 PRNG 派生（同一性靠档案、不靠动力学，
      理论③）；字段 = 引擎当下真值（自态 H双极 / 注意 a星 提升 / ε_self / 消化债 / 饱足）。
  · 每拍把 À_t 记入**独立自我痕迹存储** `自我核.迹`（与 codec._段 物理分离；记忆≠学习）；
      当出现**自指事件**（ε_self 高于其自身慢基线＝"我不理解我自己"）时，经既有 `_自言`
      （类型∈六类、强制 ≥1 账本引用）留一条**可审计的自我痕迹**，文本携带 À_t 派生字段 +
      provenance 来源号（专家03 §4 J2）。
  · provenance 门控（专家03 §2.2 / §4 J5；理论③）：**仅 provenance=自源**的广播能更新自指联想
      记忆 W_self（Hebbian：W += η·ε·outer(À,À)，η 复用既有 readout_head ETA_R，**零新超参**）；
      外源（用户输入）只进内容记忆，`W_self` **逐位不变**（max|Δ|=0）。
  · 不破既有（C14）：唯一覆盖点 = 综合体._引擎步（**只加挂**，不改任何状态推进行）；开关默认关时
      逐位恒等（D1/D9）。本文件不 import 任何器官（眼/耳/手/语义/我 冻结）。

诚实边界（不宣称超出本步）：
  1) 本步只做「留痕 + provenance 门控 + W_self 生长」；**未**把 W_self/ε_self **反哺决策**
     （那是后续步）——故 J1 的「换档案⇒行为分化」在本步只验到**机制层**（换档案 ⇒ À_t 分化），
     行为层分化待反哺接入后成立（如实登记，不冒充）。
  2) 「深度顿悟」的相变涌现＝步6；本步的「自指事件」门＝ε_self 相对**自身慢基线**（自归一，
     零人挑阈值），不是硬编码阈值。
  3) À_t 逐拍入迹（约 4KB/拍）仅供本步判据；默认关，长程内存不在本步范围。

纪律：C1 禁 Transformer｜C9 禁 OCR｜C10 禁外挂模型当器官｜C13 确定性可复现｜C14 不破既有。
用法（测试）：见 步5_自我回路自证.py。
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# 步4 宿主（含步2/步3）；本步在其之上再加挂自我回路（默认关）
from 内核_真学习 import 顶覆盖_综合体L, 驱动_ε, 开关默认 as _步4开关默认, ALPHA_SLOW  # noqa: E402
from 内核_自模型 import ETA_R                                                        # noqa: E402

开关默认 = False            # ★保护开关默认关（C14/D1：默认关 ⇒ 既有输出逐位恒等）
η_self = ETA_R              # W_self Hebbian 步长 = 既有 readout_head ETA_R（#152 冻结；零新超参）
上限范数_self = 1.0         # ★自指联想记忆的范数上限（纯数值稳定；保方向、限幅度，见 更新W）
来源标_自源 = "自源"
来源标_外源 = "外源"
_全部名 = ("状态", "注意", "ε", "债", "饱足")
_标量名 = ("ε", "债", "饱足")


def _双极(rng: np.random.Generator, n: int) -> np.ndarray:
    """确定性双极 ±1 向量（float64）。"""
    return np.where(rng.random(int(n)) < 0.5, -1.0, 1.0).astype(np.float64)


def _余弦(a, b) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class 自我核:
    """瞬时自指束 À_t = 束[绑(角色_r, 字段_r)] + 自指联想记忆 W_self（纯 HDC·零梯度·确定性）。

    · 持久身份锚 a_seal 与角色/码族由**档案 seed** 的独立 PRNG 派生（同一性靠档案，理论③）。
    · W_self 仅在 provenance=自源 时 Hebbian 更新（J5：外源不可塑造我）。
    · 迹 = 独立自我痕迹存储（与 codec._段 物理分离；记忆≠学习）。
    """

    def __init__(self, D: int = 1024, 档案seed: int = 310, α慢: float = None):
        self.D = int(D)
        self.档案seed = int(档案seed)
        rng = np.random.default_rng(self.档案seed)
        self.a_seal = _双极(rng, self.D)                                  # 持久身份锚（谁是它）
        self.角色s = {名: _双极(rng, self.D) for 名 in _全部名}            # 角色常量族（此刻字段的角色）
        self.码s = {名: (_双极(rng, self.D), _双极(rng, self.D))          # 标量字段的 正/负 值码
                   for 名 in _标量名}
        self.W_self = np.zeros((self.D, self.D), dtype=np.float64)         # 自指联想记忆（仅自源）
        self.迹: list = []                                                 # ★独立自我痕迹存储
        self.自源更新次数 = 0
        self.外源次数 = 0
        self.α慢 = float(ALPHA_SLOW if α慢 is None else α慢)
        self.慢 = {名: None for 名 in _标量名}                             # 各标量的自身慢基线（自归一）

    # ── 一拍：算自指束 À_t（含各标量相对自身基线的极性），返回 (À 单位向量, 一致度, 极性表) ──
    def 快拍(self, 状态, 注意, 标量: dict):
        项s = [self.角色s["状态"] * np.sign(np.asarray(状态, dtype=np.float64)),
               self.角色s["注意"] * np.sign(np.asarray(注意, dtype=np.float64))]
        极性 = {}
        for 名 in _标量名:
            v = 标量.get(名)
            if v is None:
                极性[名] = None
                continue
            fv = float(v)
            if self.慢[名] is None:                    # 第一条：以自身为基线，极性取正码
                self.慢[名] = fv
                极 = 0
            else:
                极 = 0 if fv >= self.慢[名] else 1     # 高于自身基线=正码（"比我平时更…"）
                self.慢[名] = float(self.慢[名]) + self.α慢 * (fv - float(self.慢[名]))
            极性[名] = 极
            项s.append(self.角色s[名] * self.码s[名][极])
        累 = 项s[0]
        for t in 项s[1:]:
            累 = 累 + t
        À = np.sign(累)
        À[À == 0.0] = 1.0
        一致度 = _余弦(À, self.a_seal)
        return À, float(一致度), 极性

    # ── 自源 → Hebbian 写自指联想记忆（ε 越大写得越多＝自监督门控；零人挑阈值） ──
    def 更新W(self, À, ε=None) -> None:
        """★范数归一（AHI3 第三轮修复，2026-10-07）：Hebbian 只加不减 ⇒ ‖W_self‖ 随拍数**无界增长**
        （实测 400→3200 拍：12395→66327，每次更新约 +21，近似线性）。自指联想记忆的长时表征
        必须**有界**，否则"我"的强度会变成"活了多久"的函数（时间污染）。
        修法：写完后按 Frobenius 范数归一到 `上限范数_self`。
        · **保方向**：归一不改 W 的方向结构（`W/‖W‖` 与原 W 同向）；
        · **限幅度**：‖W_self‖ ≡ 上限（恒等），故"我"的强度不再随拍数漂移；
        · 零新超参：上限取 1.0（单位范数，标准 Hebbian 尺度）；步长仍用既有 `η_self=ETA_R`。
        副作用（诚实标注）：痕迹里的 `W范数` 将恒为上限值 ⇒ 判据改为验**有界**（而非看它涨）。
        """
        g = 1.0 if ε is None else max(0.0, min(2.0, float(ε)))
        self.W_self = self.W_self + (η_self * g) * np.outer(À, À)
        n = float(np.linalg.norm(self.W_self))
        if n > 1e-12:
            self.W_self = self.W_self * (上限范数_self / n)
        self.自源更新次数 += 1

    # ── 记一拍独立痕迹（与 codec._段 物理分离） ──
    def 记迹(self, 拍, À, 一致度, ε, 来源, 极性) -> None:
        self.迹.append(dict(拍=int(拍), À=np.asarray(À, dtype=np.float32),
                            一致度=float(一致度),
                            ε=(None if ε is None else float(ε)),
                            来源=str(来源), 极性=dict(极性)))
        if str(来源) == 来源标_外源:
            self.外源次数 += 1

    def 摘要(self) -> dict:
        return dict(迹条数=len(self.迹), 自源更新次数=self.自源更新次数,
                    外源次数=self.外源次数, D=self.D, 档案seed=self.档案seed,
                    W范数=round(float(np.linalg.norm(self.W_self)), 4),
                    a_seal范数=round(float(np.linalg.norm(self.a_seal)), 4))


# ────────────────────────── synthesis 顶覆盖（只加挂；基链一行不改） ──────────────────────────

class 顶覆盖_综合体Z(顶覆盖_综合体L):
    """步5 宿主：在步4 宿主之上**再**加挂自我回路（自指束 À_t 留痕 + provenance 门控 W_self）。

    · 自我开关=False（默认）：_引擎步 与步4/基类逐位同构（D1 逐位恒等）。
    · 自我开关=True：每拍算 À_t 记入独立自我痕迹；**仅自源**更新 W_self；自指事件经既有 _自言
      留痕（类型∈六类、≥1 账本引用，文本含 À_t 派生字段 + 来源号）。**引擎状态推进行一行不改**。
    """

    def __init__(self, seed: int = 310, 私偏κ: float = 10.0, 自我开关=None, 档案seed=None,
                 学习开关=None, 思核开关=None, 自模型开关=None, 学习: bool = True,
                 状态投影=None, 思核参: dict = None):
        # 先置属性——super().__init__ 期间 _引擎步 已被调用（见 synthesis.py:89 注释）；
        # 那时 自我开关 已就位但 自我核=None ⇒ 自然跳过（不改任何状态）。
        self.自我开关 = bool(开关默认 if 自我开关 is None else 自我开关)
        self.自我核 = None
        self._来源标 = 来源标_自源                     # provenance：默认自源（自主拍）；用户回合改外源
        self._来源钉 = None                            # ★非 None = 宿主**显式钉住**的来源（不被 步/步_向量 覆盖）
        self._档案seed = (int(seed) if 档案seed is None else int(档案seed))
        super().__init__(seed=seed, 私偏κ=私偏κ, 学习开关=学习开关, 思核开关=思核开关,
                         自模型开关=自模型开关, 学习=学习,
                         状态投影=状态投影, 思核参=思核参)
        if self.自我开关:
            D = int(np.asarray(self.eng.hdc.Φ_act).shape[0])   # 引擎母语空间维（1024）
            self.自我核 = 自我核(D=D, 档案seed=self._档案seed)

    # ── provenance 接线：用户回合=外源（其余自主路径默认自源） ──
    # ★`_来源钉`（AHI3 接线补丁）：宿主用 `钉来源()` 显式指定后，`步/步_向量` 不再覆盖它。
    #   原实现把**每一次** 步/步_向量 都判为外源——但 V3 用 `步_向量(原文=None)` 喂
    #   **它自己看的网页正文**（自主行为），那是自源；不钉住的话「我」永不被自己的观察塑造。
    def 钉来源(self, 标) -> None:
        self._来源钉 = 标
        self._来源标 = (来源标_自源 if 标 is None else 标)

    def 步(self, *a, **k):
        旧 = self._来源标
        self._来源标 = (来源标_外源 if self._来源钉 is None else self._来源钉)
        try:
            return super().步(*a, **k)
        finally:
            self._来源标 = 旧

    def 步_向量(self, *a, **k):
        旧 = self._来源标
        self._来源标 = (来源标_外源 if self._来源钉 is None else self._来源钉)
        try:
            return super().步_向量(*a, **k)
        finally:
            self._来源标 = 旧

    # ── 本拍 ε_self（优先步4 吸收的 ε史；否则步2 自模型；再否则外部 _ε源） ──
    def _本拍ε(self):
        drv = getattr(self, "drv", None)
        if isinstance(drv, 驱动_ε) and drv.ε史:
            return float(drv.ε史[-1])
        zm = getattr(self, "自模型", None)
        if zm is not None and zm.迹:
            e = zm.迹[-1].get("ε_self")
            if e is not None:
                return float(e)
        if self._ε源 is not None:
            return self._ε源(self)
        return None

    def _自我一拍(self, rec) -> None:
        """只加挂：算 À_t → 记独立痕迹 → 仅自源更新 W_self → 自指事件经 _自言 留痕。"""
        核 = self.自我核
        h = (rec.get("模块") or {}).get("hdc") or {}
        S = h.get("H双极")
        if S is None:
            return
        a星 = rec.get("a星")
        Φ = np.asarray(self.eng.hdc.Φ_act, dtype=np.float64)
        A = Φ @ np.asarray(a星, dtype=np.float64)
        ε = self._本拍ε()
        标量 = dict(ε=ε, 债=float(getattr(self, "消化债", 0.0)),
                    饱足=float(getattr(self, "饱足", 1.0)))
        À, 一致度, 极性 = 核.快拍(S, A, 标量)
        来源 = self._来源标
        核.记迹(self.拍, À, 一致度, ε, 来源, 极性)
        if 来源 == 来源标_自源:                        # ★仅自源塑造"我"（J5；理论③）
            核.更新W(À, ε)
        # 每 ε_self 经自指束留痕（契约：每 ε_self/深度顿悟 经自指束留痕；复用既有 _自言）。
        # 类型按"相对自身基线的意外/常态"分型（均为六类之一；零人挑阈值）：
        #   ε 高于自身慢基线＝"结果与预测差在哪" → 复盘；否则＝状态自陈 → 自评。
        if ε is not None and len(self.账本) > 0:
            意外 = (极性.get("ε") == 0)
            类型 = "复盘" if 意外 else "自评"
            b慢 = 核.慢.get("ε")
            a号 = int(rec.get("a星序号", -1))
            文本 = ("自我回路[%s·第%d拍]：自指束一致度 %.3f（对身份锚），注意位 %d，"
                    "ε_self %.3f（自身基线 %s，%s）。" %
                    (来源, int(self.拍), 一致度, a号, ε,
                     ("—" if b慢 is None else "%.3f" % b慢),
                     ("自我意外↑" if 意外 else "常态")))
            self._自言(类型, 文本, (来源, "自我回路"), [len(self.账本) - 1])

    def _引擎步(self, u):
        rec = super()._引擎步(u)                       # 基链状态推进 + 步2/3/4 旁挂，逐行不变
        if self.自我开关 and self.自我核 is not None:
            try:
                self._自我一拍(rec)
            except Exception:
                pass                                   # 旁挂失败绝不打断既有路径
        return rec
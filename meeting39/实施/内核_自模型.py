# -*- coding: utf-8 -*-
"""内核_自模型.py —— meeting39 步2：桥顶自模型 ĝ（默认关的保护开关）。

设计出处（meeting39 会议纪要 §2 共识 2/3/4；专家01 §3.2；专家05 §1.1/§3 步2）：
  · 真学习 = 实时自预测更新：ĝ:(S_t,a_t)→Ŝ_{t+1}，ε_self = 1 − cos(Ŝ_{t+1}, S_{t+1})，
    完全自监督、零梯度、无外部标签（对齐 readout_head.py:59-70 的 NLMS 合规先例）。
  · 记忆 ≠ 学习（所有者硬命题）：记忆 = codec._段（"记着"）；学习 = ĝ 参数 + ε_self（"变会"）。
    两者**物理分离**——本文件所有状态独立于 codec._段，绝不写入记忆库。
  · 不破既有（C14）：唯一覆盖点 = 综合体._引擎步（**只读、只加挂**，不改其任何状态推进行）；
    默认关时逐位恒等（D1/D9）。本步**不改决策**：ĝ 只观测、只入独立学习存储。
  · 唯一合法写入口（步4才启用）= 既有 内生偏置 b[]（与 κd 同构）；本步不动。

时钟域：每 R1 拍对齐 _引擎步（与引擎同域、同拍号，保证事件重放逐位再生）。
状态表征：S_t = H双极 = sign(H_t)（引擎**带记忆的内部态**，hdc_core.py:412/484，1024 维母语空间
          丰富态，**非 5 维摘要**）。键 = S_t（**不做动作绑定**）。

⚠ 步2 实测修正（见 步2_自模型接线自证.md §3）：专家01 原设计 `k_t = S_t ⊛ a_t`（动作条件化）
  在本引擎母语空间**摧毁可预测性**——离线最小二乘实测：`x_t→S_{t+1}` 逐维命中 1.000/ε=0.000，
  而 `S_t⊛a_t→S_{t+1}` 仅 0.600/ε=0.800；`H双极_t→H双极_{t+1}` 命中 1.000/ε=0.0002，`H⊛a→H` 仅 0.754/0.491。
  根因：本引擎 `x_{t+1}=编码u(u_{t+1})` 由**外感输入**决定，(S_t,a_t) 里的动作对未来状态是**纯噪**，
  被绑进键后同一自态裂成多键、记忆互相抵消。故键取**自态本身**（可学习性由实测裁决，不照抄设计）。

⚠ 读出方式实测（AHI3 第三轮，2026-10-07；**两次测量、一次自我纠错，如实记录**）：
  起因：纯自主长跑里 ε_self 停在 ~0.5 不降，我怀疑"读出方式"有问题。
  ① **第一次测量（错）**：我在 `桥.eng.hdc.H` 上量「拷贝预测器」，得 ε≈0.081，
     据此断定 ĝ（ε≈0.52）"比照抄还差 6 倍"，于是把读出改成残差 `Ŝ=sign(S_t+W@k)`。
  ② **第二次测量（对）**：我在**内核实际使用的信号** `rec["模块"]["hdc"]["H双极"]` 上重量，得：
        · 相邻帧位翻转 **38.4%**（不是 2%）——`H双极` 与 `hdc.H` 是**两个不同的信号**，我量错了对象；
        · 拷贝预测器 ε ≈ **0.768**（**不是** 0.081）；
        · 绝对读出 ĝ 的 ε ≈ **0.58~0.64** ⇒ **本来就优于拷贝基线**（它在学真结构，不是"只会照抄"）。
  ③ 同场受控对照（只换读出、其余全同，900 拍）：
        绝对读出  首窗 0.7064 → 末窗 **0.5839**（均 0.6425）
        残差读出  首窗 0.7701 → 末窗 **0.7693**（均 0.7700）= **恰好卡在拷贝基线，学不动**
     根因：残差项 `W@k` 量级远小于自态本身（±1/维），`sign(S_t + 小残差) ≡ S_t` ⇒ 输出被钉在拷贝上。
  ④ **决定**：残差读出**不作为默认**（`读出_默认="绝对"`），保留为可对照的实验臂；
     `auto` 那条"ε 卡住"的真因另有——见下方 ⑤。
  ⑤ 「自主态 ε 不降」的真因（另测）：不是读出问题，而是**自主拍自态变化太小、没有新信息可学**。
     自模型在**有输入**时学得很好（`步2` 实测 训首 0.6256 → 训末 0.0103、AUC=1.0；
     喂连贯语料时 ε 0.73 → 0.06）。故该现象应解读为**如实反映"没东西可学"**，而不是学习坏掉。

纪律：C1 禁 Transformer｜C9 禁 OCR｜C10 禁外挂模型当器官｜C13 确定性可复现｜C14 不破既有。
用法（测试）：见 步2_自模型接线自证.py。
"""
from __future__ import annotations

import os
import sys

import numpy as np

ETA_R = 0.1          # NLMS 步长（沿用 readout_head.py:16 ETA_R，#152 冻结不动）——不新增人挑超参
开关默认 = False     # ★保护开关默认关（C14/D1：默认关 ⇒ 既有输出逐位恒等）
读出_默认 = "绝对"   # ★默认读出（= 既有行为）。"残差" 为实验臂，见文件头「读出方式实测」


def _cos(a, b) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class 自模型ĝ:
    """桥顶自模型：把 (自己的状态 S_t, 自己的动作 a_t) 映射到 自己的下一步状态 Ŝ_{t+1}。

    零梯度·纯乘加·确定性：W 由零初始化（与 readout_head 同款），逐拍仅做局部 delta 更新，
    无 lstsq/solve/pinv/inv。独立学习存储 self.迹（**不是** codec._段）。
    """

    def __init__(self, 更新: bool = True, η0: float = ETA_R, 读出: str = None,
                 残差上限: float = 1.0):
        self.W = None                       # (D, D)，首次见到状态时按 D 惰性零初始化
        self.更新 = bool(更新)              # False ⇒ 冻结臂（A2 的"模拟学习"对照；η≡0）
        self.η0 = float(η0)
        self.读出 = str(读出_默认 if 读出 is None else 读出)   # "绝对"（默认，= 既有行为）| "残差"（实验臂）
        self.残差上限 = float(残差上限)      # 残差项裁剪（±1 双极量纲；纯数值稳定，非语义阈值）
        self._上键 = None                   # k_{t-1}：上一步形成的键（用于预测本步 S_t）
        self._上态 = None                   # ★S_t：残差读出需要它作基线（"照抄上一拍"）
        self.迹: list = []                  # ★独立学习存储：[dict(拍, ε_self)]，与记忆库无关
        self.更新次数 = 0
        self.预测次数 = 0
        self.拷贝次数 = 0                    # 残差项被判为"无信息"、退回拷贝基线的次数

    # ── 预测（读出）：两种读出共用同一个 W，只差**基线** ──
    def _预测(self):
        """返回 (Ŝ, 基线)：`残差` 读出 Ŝ=sign(S_t+W@k)；`绝对` 读出 Ŝ=sign(W@k)（旧行为）。"""
        raw = self.W @ self._上键
        if self.读出 == "绝对" or self._上态 is None:
            return np.sign(raw), None
        残 = np.clip(raw, -self.残差上限, self.残差上限)
        return np.sign(self._上态 + 残), self._上态

    # ── 单拍：算 ε_self → 改自己 → 记本拍键 ──
    def 学一拍(self, 键k, 状态S, 拍: int = -1):
        """键k = k_t；状态S = S_{t+1}（本拍引擎产出的状态）。

        流程（"预测自己 → 误差 → 改自己"）：用上拍键 + 上拍自态预测本拍状态 → ε_self
        → 据误差更新 W → 存本拍键/自态供下拍预测。返回本拍 ε_self（首拍无预测 ⇒ None）。
        """
        S = np.asarray(状态S, dtype=np.float64)
        k = np.asarray(键k, dtype=np.float64)
        if self.W is None:                                   # 惰性 D 初始化（确定性：零矩阵）
            D = int(S.shape[0])
            self.W = np.zeros((D, D), dtype=np.float64)
        ε = None
        if self._上键 is not None:
            Ŝ, 基线 = self._预测()                            # 预测本拍状态
            # ★读出语义：`残差` 时模型学的是「相对上拍自态的变化量」真值 = S − S_t；
            #   `绝对` 时学的是 S 本身（旧行为）。Ŝ_残差 = sign(S_t + W@k)，故残差预测 = Ŝ − S_t。
            if 基线 is None:
                真残差, 预残差 = S, Ŝ
            else:
                真残差, 预残差 = S - 基线, Ŝ - 基线
            self.预测次数 += 1
            ε = 1.0 - _cos(S, Ŝ)                             # ε_self ∈ [0,2]（对**真状态**算，自监督）
            if self.更新:                                    # 真学习臂：误差越大写得越多
                n2 = float(self._上键 @ self._上键) + 1e-6
                self.W += (self.η0 * ε) * np.outer(真残差 - 预残差, self._上键) / n2
                self.更新次数 += 1
                if 基线 is not None and not np.any(真残差):
                    self.拷贝次数 += 1                        # 本拍"世界无变化"⇒ 拷贝基线本就正确
        self._上键 = k
        self._上态 = S                                           # ★供下拍残差基线
        self.迹.append(dict(拍=int(拍), ε_self=(None if ε is None else float(ε))))
        return ε

    # ── 只读查询（判据用；不触碰记忆库） ──
    def ε序列(self) -> list:
        return [d["ε_self"] for d in self.迹 if d["ε_self"] is not None]

    def 摘要(self) -> dict:
        ε = self.ε序列()
        return dict(迹条数=len(self.迹), 更新次数=self.更新次数, 预测次数=self.预测次数,
                    读出=self.读出, 拷贝次数=self.拷贝次数,
                    D=(int(self.W.shape[0]) if self.W is not None else 0),
                    ε均值=(round(float(np.mean(ε)), 4) if ε else None))


# ────────────────────────── synthesis 顶覆盖（只加挂；基链一行不改） ──────────────────────────

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # AHI3 根（迁移：随文件定位）
for _p in (os.path.join(ROOT, "meeting31"), ROOT, os.path.join(ROOT, "meeting30"),
           os.path.join(ROOT, "meeting29"), os.path.join(ROOT, "meeting28")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from synthesis import 综合体   # noqa: E402


class 顶覆盖_综合体(综合体):
    """synthesis 顶覆盖宿主：只覆写 _引擎步，在既有引擎步之后**只读**喂自模型。

    · 关（默认）：自模型=None ⇒ _引擎步 与基类逐位同构（D1/D9 逐位恒等）。
    · 开：只读 rec 的丰富态 x / a星 → 喂 ĝ，写**独立学习存储**；**不改** rec、不改任何状态量、不改决策。
    """

    def __init__(self, seed: int = 310, 私偏κ: float = 10.0, 自模型开关=None,
                 学习=True, 状态投影=None):
        # 先置属性——super().__init__ 期间 _引擎步 已被调用（见 synthesis.py:90 注释），
        # 那时自模型尚未建 ⇒ 自然跳过；构造完成后才建 ĝ（确定性）。
        self.自模型开关 = bool(开关默认 if 自模型开关 is None else 自模型开关)
        self.自模型 = None
        self._学习 = bool(学习)
        # 状态投影=低维摘要对照（如 5 维 ELD）：退化表征下 A1 应失效（理论② AUC≈0.53）
        self.状态投影 = (None if 状态投影 is None
                        else np.asarray(状态投影, dtype=np.float64))
        super().__init__(seed=seed, 私偏κ=私偏κ)
        if self.自模型开关:
            self.自模型 = 自模型ĝ(更新=self._学习)

    def _引擎步(self, u):
        rec = super()._引擎步(u)                     # 基类引擎步：状态推进逐行不变
        if self.自模型 is not None:
            h = None
            try:
                h = (rec.get("模块") or {}).get("hdc") or {}
            except Exception:
                h = {}
            # 自态 S_t = 引擎带记忆的内部态 H双极（母语空间 1024 维双极；非 5 维摘要）
            S = h.get("H双极")
            if S is not None:
                S = np.asarray(S, dtype=np.float64)
                if self.状态投影 is not None:          # 压缩对照：5 维摘要
                    Sp = self.状态投影 @ S
                    self.自模型.学一拍(Sp, Sp, 拍=self.拍)
                else:                                  # 主路：键 = 自态本身（无动作绑定，见文件头 ⚠）
                    self.自模型.学一拍(S, S, 拍=self.拍)   # 独立学习存储；不改 rec/状态/决策
        return rec
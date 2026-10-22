# -*- coding: utf-8 -*-
"""内核_相空间算子T.py —— meeting39 步3：相空间算子 T（默认关·旁挂思考态）。

设计出处（meeting39 会议纪要 §1 专家02；专家05 §2 步3；专家02 §2.2/§3.1）：
  · 「想」= 内部思想相点 z ∈ M 上的一次算子迭代 z←T(z)，**不是** _检索。
  · 复用理论① 的 T₂ 非周期 itinerancy 档：差模竞争 + 轨迹自排斥 + 惯性 + 自适应增益。
  · 每一「思考拍」推进 1 步算子；轨迹写入**独立思考存储**（不是 codec._段）。
  · 默认关（保护开关）：关时 _思考一拍 逐位走基类旧路径（V2–V5 零变）。

⚠ 与理论①源（评测/实验七_内在思想与AHI.py 内在动力学）的两处**经授权偏离**（诚实标注）：
  1) 相空间方向表：本阶段眼/耳/语义冻结（无器官广播向量），故竞争方向取**引擎自身动作
     自由度的母语空间提升** ṽ_j = unit(Φ_act[:, j])（引擎既有固定投影，**零新权重表**，C1/C13 合规）。
  2) 身份回拉（锚定.引力）：文本态无可加载的「身份锚定」载体（无认知引擎），故**省略**该项，
     改以 L2 归一保证有界（‖z‖≡1，不爆不塌）；真正的身份档案接入留待步5（自我回路）。
  其余算子部件（差模共振场 / 自排斥记忆 / 惯性 / 相对门 softmax / 习惯化 / 自适应增益）逐条
  按 实验七 `内在动力学.步` 的数学**等价移植**为纯 numpy（零 torch、零梯度、确定性）。

参数默认取 `results/实验七_内在思想与AHI.json` 的「T2选中」（经 实验八 §113 T2参数引证，未另行调参）。
本文件运行不 import 任何器官；只在测试脚本中被显式加载。

纪律：C1 禁 Transformer｜C9 禁 OCR｜C10 禁外挂模型当器官｜C13 确定性可复现｜C14 不破既有。
"""
from __future__ import annotations

import os
import sys

import numpy as np

# T2 选中参数（实验七 JSON「T2选中」，见 实验八_自主行为.py:113-116）——原样引用，不调参
T2参数 = dict(增益模式="自适应", η_g=0.1, η_h=0.35, β门控=4.0, 惯性=0.35,
             差模=True, γ_共=1.0, κ门=1.0, ρ自排斥=1.5, γ_m=0.2)
g_min, g_max = 0.1, 10.0
增益归一化 = True          # 长程塌缩修复（实验七 增益归一化默认=False 的已知缺陷；此处开启）
开关默认 = False           # ★保护开关默认关（C14/D1：默认关 ⇒ 既有输出逐位恒等）


def _单位(x) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    n = float(np.linalg.norm(x))
    return x / n if n > 1e-12 else x


def _余弦(a, b) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class 思核:
    """相空间算子 T：把「想」承载为单位球面上的确定性迭代（零 Token、零梯度、零外部输入）。

    一步 = 相对门（softmax 竞争）→ 差模共振场 → 轨迹自排斥 → L2 归一 → 惯性(二阶) → 自适应增益。
    """

    def __init__(self, 方向表: dict, 参: dict = None, 增益归一化_=增益归一化):
        参 = dict(T2参数 if 参 is None else 参)
        self.向量表 = {名: _单位(v) for 名, v in 方向表.items()}
        维度集 = {int(v.shape[0]) for v in self.向量表.values()}
        if len(维度集) != 1:
            raise ValueError("各方向维度不一致：%s" % (维度集,))
        self.维度 = int(维度集.pop())
        self.增益模式 = str(参.get("增益模式", "自适应"))
        self.η_g = float(参.get("η_g", 0.1))
        self.η_h = float(参.get("η_h", 0.35))
        self.β门控 = float(参.get("β门控", 4.0))
        self.惯性 = max(0.0, min(0.95, float(参.get("惯性", 0.35))))
        self.差模 = bool(参.get("差模", True))
        self.γ_共 = float(参.get("γ_共", 1.0))
        self.κ门 = float(参.get("κ门", 1.0))
        self.ρ自排斥 = max(0.0, float(参.get("ρ自排斥", 1.5)))
        self.γ_m = max(1e-6, min(1.0, float(参.get("γ_m", 0.2))))
        self.增益归一化 = bool(增益归一化_)

        if self.差模:
            v̄ = np.zeros(self.维度, dtype=np.float64)
            for v in self.向量表.values():
                v̄ = v̄ + v
            v̄ = v̄ / max(1, len(self.向量表))
            self.共模 = _单位(v̄)
            self.竞争表 = {名: _单位(v - v̄) for 名, v in self.向量表.items()}
        else:
            self.共模 = np.zeros(self.维度, dtype=np.float64)
            self.竞争表 = dict(self.向量表)
        self.增益0 = {名: 1.0 for 名 in self.竞争表}      # 初始显著度相等（不引入人挑权重）
        self.迹: list = []                               # ★独立思考存储：[dict(拍, z)]，与记忆库无关

    # ────────────────────────── 算子单步 ──────────────────────────
    def _相对门(self, z, h):
        """softmax_j(β·(cos(z,ṽ_j) − c̄) − κ门·h_j)：门是**有限资源**竞争。"""
        余弦 = {名: _余弦(z, v) for 名, v in self.竞争表.items()}
        if self.增益模式 != "自适应":
            return {名: 1.0 for 名 in self.竞争表}, 余弦
        均值 = sum(余弦.values()) / max(1, len(余弦))
        指数 = {名: self.β门控 * (余弦[名] - 均值) - self.κ门 * float(h.get(名, 0.0))
                for 名 in 余弦}
        最大 = max(指数.values()) if 指数 else 0.0
        权重 = {名: float(np.exp(值 - 最大)) for 名, 值 in 指数.items()}
        总和 = sum(权重.values())
        if 总和 <= 1e-300:
            return {名: 1.0 / max(1, len(权重)) for 名 in 权重}, 余弦
        return {名: 值 / 总和 for 名, 值 in 权重.items()}, 余弦

    def _共振场(self, z, g, 门, 余弦):
        共 = self.γ_共 * self.共模 if self.差模 else np.zeros(self.维度, dtype=np.float64)
        w = 共.copy()
        for 名, v in self.竞争表.items():
            c = 余弦.get(名, _余弦(z, v))
            w = w + float(g.get(名, 1.0)) * float(门.get(名, 1.0)) * c * v
        return w

    def 步(self, z, g, h, z前, 记忆):
        """一次内在迭代。返回 (z新, g新, h新, 记忆新, 信息)。全程无外部输入。"""
        h = dict(h)
        门, 余弦 = self._相对门(z, h)
        w = self._共振场(z, g, 门, 余弦)
        记忆新 = (np.zeros(self.维度, dtype=np.float64) if 记忆 is None
                 else (1.0 - self.γ_m) * 记忆 + self.γ_m * z)
        if self.ρ自排斥 > 0.0 and 记忆 is not None:
            w = w - self.ρ自排斥 * 记忆          # 轨迹自排斥：去过的地方不再想去
        范数 = float(np.linalg.norm(w))
        if 范数 < 1e-12:                          # 退化：共振场相消为零 ⇒ 保持原状态
            return z.copy(), dict(g), h, 记忆新, {"退化": True}
        u = w / 范数
        if self.惯性 > 0.0 and z前 is not None:
            z新 = _单位((1.0 - self.惯性) * u + self.惯性 * z前)
        else:
            z新 = u

        g新 = dict(g)
        h新 = dict(h)
        if self.增益模式 == "自激":
            for 名 in self.竞争表:
                g新[名] = float(min(max(float(g.get(名, 1.0)) * np.exp(
                    self.η_g * (_余弦(z新, self.竞争表[名]) - 0.5)), g_min), g_max))
        elif self.增益模式 == "自适应":
            for 名 in self.竞争表:
                a = float(门[名])
                h旧 = float(h.get(名, 0.0))
                h新[名] = float(h旧 + self.η_h * (a - h旧))
                g新[名] = float(min(max(float(g.get(名, 1.0)) * np.exp(
                    self.η_g * a - self.η_h * h新[名]), g_min), g_max))
            if self.增益归一化:                   # 去共模漂移：只保留器官间相对增益结构
                均值 = sum(g新.values()) / max(1, len(g新))
                if 均值 > 1e-12:
                    for 名 in g新:
                        g新[名] = float(min(max(g新[名] / 均值, g_min), g_max))
        return z新, g新, h新, 记忆新, {"退化": False, "门": 门}

    def 迭代(self, z0, K: int):
        """从 z0 迭代 K 步（一次完整推理链），返回轨迹。"""
        z = _单位(z0)
        z前 = z.copy()
        g = dict(self.增益0)
        h = {名: 0.0 for 名 in self.竞争表}
        记忆 = None
        轨迹 = [z.copy()]
        for _ in range(int(K)):
            旧 = z
            z, g, h, 记忆, _ = self.步(z, g, h, z前, 记忆)
            z前 = 旧
            轨迹.append(z.copy())
        return 轨迹

    # ────────────────────────── 判据接口 ──────────────────────────
    def 站点(self, z) -> str:
        """z 落在哪个方向（argmax cos）——供路径熵/换站统计。"""
        return max(self.竞争表, key=lambda 名: _余弦(z, self.竞争表[名]))

    def 清迹(self) -> None:
        self.迹 = []


# ────────────────────────── synthesis 顶覆盖（只旁挂；基链一行不改） ──────────────────────────

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from 内核_自模型 import 顶覆盖_综合体 as _顶覆盖步2   # noqa: E402  (步2 宿主；默认关时逐位同构基类)


class 顶覆盖_综合体T(_顶覆盖步2):
    """步3 宿主：在步2 宿主之上**再**旁挂相空间算子 T（思考态每拍推进一步）。

    · 思核开关=False（默认）：_思考一拍 完全走步2/基类旧路径（V2–V5 零变）。
    · 思核开关=True：思考态每拍**额外**推进一步 T，轨迹入独立思考存储；**旧检索/收敛路径
      一行不改**（本步只做「旁挂·可探测」，深度/停止规则的交还与步6）。
    """

    def __init__(self, seed: int = 310, 私偏κ: float = 10.0, 思核开关=None,
                 自模型开关=None, 学习: bool = True, 状态投影=None, 思核参: dict = None):
        self.思核开关 = bool(开关默认 if 思核开关 is None else 思核开关)
        self.思核 = None
        self._思核参 = dict(思核参) if 思核参 else None
        super().__init__(seed=seed, 私偏κ=私偏κ, 自模型开关=自模型开关,
                         学习=学习, 状态投影=状态投影)

    # ── 方向表：引擎动作自由度 → 母语空间方向（Φ_act 列单位化；零新权重表） ──
    def _方向表(self) -> dict:
        Φ = getattr(self.eng.hdc, "Φ_act", None)
        if Φ is not None:
            Φ = np.asarray(Φ, dtype=np.float64)
            k = int(Φ.shape[1])
            return {"动%d" % j: _单位(Φ[:, j]) for j in range(k)}
        raise RuntimeError("引擎无 Φ_act：文本态无法在母语空间开方向表（不应发生）")

    def _起思核(self) -> None:
        if self.思核 is None:
            参 = dict(T2参数)
            if self._思核参:
                参.update(self._思核参)
            self.思核 = 思核(self._方向表(), 参=参)

    def _思考一拍(self):                                   # noqa: D401
        思 = self.思考中
        if self.思核开关 and 思 is not None:
            try:
                self._起思核()
                自态 = np.sign(np.asarray(self.eng.hdc.H, dtype=np.float64))
                z = 思.get("_z")
                if z is None:                              # 本思考态首拍：z0 由引擎自态派生
                    z = _单位(自态)
                    思["_z"] = z
                    思["_z前"] = z.copy()
                    思["_记忆"] = None
                    思["_g"] = dict(self.思核.增益0)
                    思["_h"] = {名: 0.0 for 名 in self.思核.竞争表}
                z新, g, h, 记忆新, _ = self.思核.步(
                    思["_z"], 思["_g"], 思["_h"], 思["_z前"], 思["_记忆"])
                思["_z前"] = 思["_z"]
                思["_z"] = z新
                思["_g"] = g
                思["_h"] = h
                思["_记忆"] = 记忆新
                self.思核.迹.append(dict(拍=int(self.拍), z=z新.astype(np.float32)))
            except Exception:
                pass                                       # 旁挂失败绝不打断既有思考路径
        return super()._思考一拍()
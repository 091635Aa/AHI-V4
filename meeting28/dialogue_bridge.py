# -*- coding: utf-8 -*-
"""meeting28/dialogue_bridge.py —— #216 AHI 对话桥：核心引擎×语言。

语言→16 维语义感觉（随机探针 cos 投影）→引擎动力学（弛豫/M/井/agency 全链真实响应）；
引擎动作候选映射为对话动作（陈述/提问/确认/转移）——由引擎 agency 机制（EFE+novelty+
私偏）选择，状态依赖；响应=检索记忆+模板封套+遥测。记忆=编解码器注册段（可持久化）。
诚实边界：语言表面为模板+检索（非生成式），智能来自引擎状态动力学。
"""
from __future__ import annotations

import os
import sys
import pickle

import numpy as np

ROOT = r"<workspace>/03_AHI\AHI2.1"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "meeting15"))
sys.path.insert(0, os.path.join(ROOT, "meeting17"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import lang_codec as _lc
from lang_codec import 语言编解码器, D

动作名 = ["陈述", "提问", "确认", "转移话题"]
模板 = {
    "陈述": "我记得你说过的：『{内容}』。就这一点来看，我的看法是——{遥测}",
    "提问": "你提到『{内容}』，那我想问：这和之前的事有什么关联？({遥测})",
    "确认": "嗯，我记下了：『{内容}』。({遥测})",
    "转移话题": "这个我先放一放。倒是想起一件相关的：『{内容}』。({遥测})",
}


class 对话桥:
    """核心引擎的对话封装（#216）：语言进→引擎动力学→对话动作+检索响应出。"""

    def __init__(self, seed: int = 216, 私偏κ: float = 10.0):
        from ahi_theory.v2_engine import V2引擎
        from m2_intrinsic import 内生驱动器
        from k_engine_v3_12seed_run import _C3流
        import ahi_theory.hdc_core as hdc_core

        hdc_core.HDC_LEVEL = 4
        self.seed = seed
        self.codec = 语言编解码器(seed=seed)
        self.eng = V2引擎(seed=seed)
        self.eng.私有熵开关 = True
        self.eng.hdc.启用M上下文(4)
        self.eng.regime供数 = _C3流
        self.drv = 内生驱动器(n候选=len(self.eng.hdc.动作候选))
        # 语义探针 (16, D)：★动态读取 lang_codec.D（#动态维度伸缩 Phase2）——
        # 缺省 D=1024 与「from lang_codec import D」逐位一致；运行期 set lang_codec.D=新D
        # 后再构造桥，即得新维探针（D→16 语义投影随维重建；引擎输入恒 16 维，不破既有）。
        # #P1 修复：随机高斯探针 → **数据自定语义探针**（已注册段向量的前 16 主方向）。
        # 策略（与本次其余机制一致）：**默认关**（`AHI_语义探针=1` 开启），守 C14「新层默认关 ⇒ 逐位恒等」；
        #   `AHI_旧探针=1` 亦为关（兼容早前登记的回退线）。
        # ★默认（随机探针）下「意」仍进引擎：`精化` 已使 `编码/字袋` 携带共现语义，随机探针只是它的
        #   JL 投影 ⇒ 语义照常进入 16 维感觉（见 meeting43/实施/判据_语义探针.py J3 实测）。
        self._随机探针 = np.random.default_rng(seed + 70001).standard_normal((16, int(_lc.D)))
        self._随机探针 /= np.linalg.norm(self._随机探针, axis=1, keepdims=True)
        self._语义探针开 = (os.environ.get("AHI_语义探针", "") in ("1", "true", "True", "yes")
                        and os.environ.get("AHI_旧探针", "") not in ("1", "true", "True", "yes"))
        self.探针 = self._随机探针
        self._探针段数 = -1
        self.W_priv = np.random.default_rng(seed + 90003).standard_normal((4, 8))
        self.私偏κ = float(私偏κ)
        self.拍 = 0
        self.引导拍 = 3               # 对话域预热（对话轮次短，3 轮后 agency 即参与选择）
        self.历史: list = []          # (轮, 用户文本, 动作, 检索, 响应)
        self.a_prev = self.eng.hdc.动作候选[0].copy()
        self.i_prev = 0
        self.prev_c = None
        self._错误数 = 0

    # —— 语言→感觉 ——
    def _重建探针(self) -> None:
        """#P1：数据自定语义探针——已注册段向量的前 16 个主方向（确定性 SVD）。

        段不足(<2) 或未开（默认）→ 确定性回退随机探针。形状恒 (16, D)（引擎输入恒 16 维 = C14）。
        确定性：SVD 同机同输入一致；再叠加「每行首个非零元素为正」的符号约定，消除 SVD 符号自由度；
        无时间、无全局随机。"""
        if not getattr(self, "_语义探针开", False):
            self.探针 = self._随机探针
            return
        D = int(_lc.D)
        段s = list(getattr(self.codec, "_段向量", None) or [])
        if len(段s) < 2:
            self.探针 = self._随机探针
            return
        X = np.stack([np.asarray(v, dtype=np.float64) for v in 段s])
        X = X - X.mean(axis=0, keepdims=True)
        try:
            _, _, Vt = np.linalg.svd(X, full_matrices=False)
        except np.linalg.LinAlgError:
            self.探针 = self._随机探针
            return
        k = int(min(16, Vt.shape[0]))
        P = np.array(self._随机探针, dtype=np.float64, copy=True)   # 不足 16 主方向→余行沿用随机
        P[:k] = Vt[:k]
        for i in range(16):                                # 确定性符号约定（消 SVD 符号自由度）
            nz = np.flatnonzero(P[i])
            if nz.size and P[i, nz[0]] < 0:
                P[i] = -P[i]
            nrm = float(np.linalg.norm(P[i]))
            if nrm > 0:
                P[i] = P[i] / nrm
        self.探针 = P

    def 感知向量(self, vec: np.ndarray) -> np.ndarray:
        if len(getattr(self.codec, "_段", [])) != self._探针段数:   # 段集变化→惰性重建（确定性）
            self._重建探针()
            self._探针段数 = len(getattr(self.codec, "_段", []))
        cos = self.探针 @ vec / (np.linalg.norm(vec) + 1e-9)
        return (cos * 2.0).astype(np.float32)          # 语义→16 维感觉（×2 信号强度）

    # —— 单轮对话 ——
    def 步(self, 用户文本: str) -> dict:
        vec = self.codec.编码(用户文本)
        u = self.感知向量(vec)
        b = self.drv.偏置() if self.拍 >= self.引导拍 else None
        if b is not None:
            私 = self.eng.生成私有()
            wdir = self.W_priv @ np.asarray(私["q8"], dtype=np.float64)
            动作候选 = self.eng.hdc.动作候选
            b = [float(b[i]) + self.私偏κ * float(np.dot(动作候选[i], wdir))
                 for i in range(len(动作候选))]
        else:
            私 = self.eng.生成私有()
        gate, pred = self.drv.学习门(self.i_prev)
        rec = self.eng.通电步进R1(u, a_prev=self.a_prev, 内生偏置=b, efe_mode=True,
                                  学习门=gate if self.拍 >= self.引导拍 else None,
                                  私有上下文=self.prev_c)
        self.drv.更新(rec["候选ê"])
        动作 = 动作名[int(rec["a星序号"]) % 4]
        self.a_prev = rec["a星"]
        self.i_prev = int(rec["a星序号"])
        # 记忆检索（双通道近邻；检索库=用户文本+语料，响应不入库防自回声）
        # ★先检索旧记忆，后注册新文本——否则查询自相似 1.0 永远命中自身（冒烟实证）
        检索 = self.codec.解码(用户文本) if self.codec._段 else "(尚无记忆)"
        self.codec.注册段(用户文本)                     # 用户文本入记忆
        遥测 = ("能量%.2f/新异%.2f" % (rec["e"] if rec["e"] is not None else -1.0,
                                      float(np.mean(rec["候选ê"]))))
        响应 = 模板[动作].format(内容=检索, 遥测=遥测)
        self.历史.append((self.拍, 用户文本, 动作, 检索, 响应))
        self.拍 += 1
        self.prev_c = int(私["c"])
        if (self.拍 % 20) == 0:
            self.eng.维护R2(q=私["q8"])
        return dict(动作=动作, 检索=检索, 响应=响应, 遥测=遥测,
                    e=rec["e"], ñ=float(np.mean(rec["候选ê"])))

    # —— 语料灌注（训练：无反向传播，纯记忆+井） ——
    def 教(self, 文本: str) -> int:
        n = 0
        for 句 in [s.strip() for s in 文本.replace("。", "。\n").split("\n") if s.strip()]:
            self.codec.注册段(句)
            self.步(句)                                 # 语义进引擎动力学（无响应输出需求）
            n += 1
        return n

    # —— 持久化（#216：确定性重放——引擎同 seed+同输入序列=状态逐位重建；不 pickle 引擎内部） ——
    def 保存(self, 路径: str) -> None:
        import pickle as _p
        with open(路径, "wb") as f:
            _p.dump(dict(seed=self.seed, 输入=[h[1] for h in self.历史]), f)

    @classmethod
    def 载入(cls, 路径: str) -> "对话桥":
        import pickle as _p
        with open(路径, "rb") as f:
            d = _p.load(f)
        桥 = cls(seed=d["seed"])
        for t in d["输入"]:
            桥.步(t)                                        # 确定性重放=状态逐位重建
        return 桥

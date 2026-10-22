# -*- coding: utf-8 -*-
"""meeting38/实施/发声动作与EFE.py —— 步 2b：发声动作 5 轴因子化 + EFE 选动作器（纯新增）。

规格：meeting38/实施/08_步2b_发声动作与EFE规格.md（依据 05 专家方案 §5/§6）。
拆弹账：发声动作朴素 6×8×8×8×8=24576 独立码 → 5 轴 27 档原子+5 角色码（实装 4500 组合）。
前向模型 = HDC one-shot 联想记忆（无梯度；步 2 教训适用：只做成员识别+cleanup）。
嘴/手动作共用同一 EFE 打分器（05 §5.3「扩展互补空间」）。
"""
from __future__ import annotations

import numpy as np

from fsw_kernel import FSW内核, 束, 绑, 相似, D_C

# —— 发声动作 5 轴（离散档；实装档位，组合 6×6×5×5×5=4500） ——
动作轴表 = (
    ("类型", ("陈述", "疑问", "感叹", "呼唤", "强调", "静默")),
    ("强度", ("pp", "p", "m", "mf", "f", "ff")),
    ("音高", ("低", "中低", "中", "中高", "高")),
    ("时长", ("短促", "偏短", "正常", "偏长", "拖长")),
    ("音质", ("气声", "偏气", "正常", "偏紧", "紧喉")),
)
# —— 手动作 4 轴（同打分器的另一效应器实例；档位示例，步 3/5 可扩） ——
手动作轴表 = (
    ("器", ("指", "手", "臂")),
    ("方向", ("上", "下", "左", "右", "前")),
    ("距离", ("近", "中", "远")),
    ("力度", ("轻", "中", "重")),
)

λ默认 = 0.3      # EFE 探索权重（05 §5.2 预登记）


class 动作编码器:
    """离散动作轴 → 动作码（5/4 对绑定的束）。全部 sha256 确定性原子。"""

    def __init__(self, 核: FSW内核 = None, 种子: int = 3800, 前缀: str = "动作"):
        self.核 = 核 if 核 is not None else FSW内核(种子=种子)
        self.D = D_C
        self.前缀 = 前缀
        self.轴表 = 动作轴表 if 前缀 == "动作" else (手动作轴表 if 前缀 == "手动作" else ())
        self._轴s: dict = {}                      # 轴名 → (角色键, {档名: 原子})
        for 轴名, 档s in self.轴表:
            角色键 = "角色|%s%s" % (前缀, 轴名)
            档原子 = {档: self.核.原子("原子|%s|%s%s=%s" % (self.核.种子, 前缀, 轴名, 档))
                      for 档 in 档s}
            self._轴s[轴名] = (角色键, 档原子)
        self.角色原子 = {轴名: self.核.原子(键) for 轴名, (键, _) in self._轴s.items()}

    def 轴档表(self) -> dict:
        """{轴名: 档名列表}（枚举/采样用）。"""
        return {轴名: list(档原子) for 轴名, (_, 档原子) in self._轴s.items()}

    def 编码(self, 选取: dict) -> np.ndarray:
        """选取: {轴名: 档名} → 动作码 = 束(绑(role_x, 档原子_x))。"""
        缺 = set(self._轴s) - set(选取)
        if 缺:
            raise ValueError("动作缺少轴选取: %s" % sorted(缺))
        return 束([绑(self.角色原子[轴名], self._轴s[轴名][1][选取[轴名]])
                   for 轴名, _ in self.轴表], self.D)


class 联想记忆:
    """HDC one-shot 前向模型（向量化最近邻版）。

    08 规格第一版用 sign-捆绑 + unbind cleanup，实测 K=40 时 cleanup 正确率 0.73
    （与步 1/步 2 同一物理学：捆绑聚合信号被 1/√K 串扰淹没）。重写为「最近邻动作
    → 其结果」：机制与步 1 FSW 检索同律（已证 top-1=1.0），无梯度、one-shot 不变，
    且顺带获得近邻泛化（未见动作 → 最近已学动作的结果，ambiguity 如实标高）。
    超出 ~40 对时由 步 1 IVF 扩展（本层 K ≤ 几十，直接 GEMM 即可）。"""

    def __init__(self, 种子: int = 3800):
        self.种子 = 种子
        self.对s: list = []                        # [(a码, o码)]（学习序，确定性）
        self._动作矩阵: np.ndarray = None
        self._结果矩阵: np.ndarray = None

    def 学(self, a码: np.ndarray, o码: np.ndarray) -> None:
        self.对s.append((a码.copy(), o码.copy()))
        self._动作矩阵 = self._结果矩阵 = None

    def _缓存(self) -> tuple:
        if self._动作矩阵 is None or self._动作矩阵.shape[0] != len(self.对s):
            self._动作矩阵 = np.stack([a for a, _ in self.对s], 0)
            self._结果矩阵 = np.stack([o for _, o in self.对s], 0)
        return self._动作矩阵, self._结果矩阵

    def 读批(self, 候选矩阵: np.ndarray) -> tuple:
        """候选矩阵 (C,D) → (最近动作索引 (C,), 该余弦 (C,))。cos = 点积/D（双极）。"""
        if not self.对s:
            return None, None
        动作矩阵, _ = self._缓存()
        S = 候选矩阵 @ 动作矩阵.T / float(D_C)     # (C,K) 一次 GEMM
        索引 = np.argmax(S, axis=1)                # tie 取先学（确定性）
        return 索引, S[np.arange(len(索引)), 索引]

    def 读(self, a码: np.ndarray) -> tuple:
        """→ (最佳结果码 o*, cos*)；无已存结果 → (None, 0.0)。"""
        if not self.对s:
            return None, 0.0
        索引, cos = self.读批(a码.reshape(1, -1))
        _, 结果矩阵 = self._缓存()
        return 结果矩阵[int(索引[0])], float(cos[0])


class EFE选择器:
    """risk + λ·ambiguity 的 argmin（嘴/手候选共用同一实例与同一前向模型）。整批向量化。"""

    def __init__(self, 记忆: 联想记忆 = None, λ: float = λ默认):
        self.记忆 = 记忆 if 记忆 is not None else 联想记忆()
        self.λ = float(λ)

    def 打分(self, a码: np.ndarray, G: np.ndarray) -> dict:
        """单动作 EFE 明细（选择 的单候选包装）。"""
        _, d = self.选择([(0, a码)], G)
        return d

    def 选择(self, 候选, G: np.ndarray):
        """候选: [(标识, a码)] → (标识, 明细)。argmin EFE；并列 → 列表序（确定性）。"""
        if not 候选:
            return None, dict(risk=1.0, ambiguity=1.0, efe=1.0 + self.λ, cos=0.0)
        G = np.asarray(G, np.float32)
        标识s = [标识 for 标识, _ in 候选]
        候选矩阵 = np.stack([a for _, a in 候选], 0)
        if not self.记忆.对s:                      # 空记忆：全候选同分，取首
            return 标识s[0], dict(risk=1.0, ambiguity=1.0,
                                  efe=1.0 + self.λ, cos=0.0)
        索引, 检索cos = self.记忆.读批(候选矩阵)
        _, 结果矩阵 = self.记忆._缓存()
        o星矩阵 = 结果矩阵[索引]                    # (C,D)
        risk = 1.0 - o星矩阵 @ G / float(D_C)
        ambiguity = 1.0 - 检索cos
        efe = risk + self.λ * ambiguity
        星 = int(np.argmin(efe))                   # tie 取列表序首个（确定性）
        return 标识s[星], dict(risk=float(risk[星]), ambiguity=float(ambiguity[星]),
                               efe=float(efe[星]), cos=float(检索cos[星]))

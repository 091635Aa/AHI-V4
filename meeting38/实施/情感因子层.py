# -*- coding: utf-8 -*-
"""meeting38/实施/情感因子层.py —— 步 2：情感/韵律/意图因子化编码层（纯新增）。

规格：meeting38/实施/07_步2_情感韵律因子化规格.md（依据 05 专家方案 §3/§4/§6）。
范围：05 §10.2 落地顺序 1–3 项（建轴 + 编解码往返 + 容量压力）；
EFE/发声动作（4–5 项）与声带对接（6 项）属步 2b/步 4。

设计律（05 §2）：
1. 因子化：不给情绪/韵律组合发独立码，只发因子轴，组合 = 轴取值的绑定束；
2. 角色绑定：与语义/意图同居同一 D=1024 空间，靠 role⊙filler 准正交软隔离；
3. 可变向量：情绪是因子空间按需读出的区域，只有当前态占 live 槽位。

拆弹账（05 §6）：韵律 8×N → 8 轴 + K 窗口；发声动作 24576 → 5 轴（步 2b）；
live 全模态 ≤87 对绑定 < 0.1·D=102 干净容量线。
"""
from __future__ import annotations

import importlib.util
import os

import numpy as np

from fsw_kernel import FSW内核, 束, 绑, 相似, D_C

# 声带.数轴 动态导入（不改其文件，C8 复用）
_AHI3 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # AHI3 根（迁移）
_spec = importlib.util.spec_from_file_location(
    "声带mod", os.path.join(_AHI3, "meeting32", "声带.py"))
_声带mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_声带mod)
数轴 = _声带mod.数轴

# —— 因子表（05 §3.2/§4：名称, 域下界, 域上界） ——
情感因子 = (
    ("V", "效价", -1.0, 1.0),      # 核心：决定 F0 升降、幅度、气声底
    ("A", "唤醒", 0.0, 1.0),       # 核心：语速、F0 摆幅、能量 CV
    ("P", "支配", 0.0, 1.0),       # 核心：幅度、紧喉、起音陡缓
    ("G", "归因", 0.0, 1.0),       # 社会：0=他人/环境 1=自我（怒 vs 内疚）
    ("M", "道德", -1.0, 1.0),      # 社会：正/负评价
    ("X", "暴露", 0.0, 1.0),       # 社会：0=私密 1=公开（羞耻 vs 内疚唯一判别轴）
    ("T", "威胁", 0.0, 1.0),       # 社会：嫉妒(高) vs 羡慕(低)
    ("Nv", "预期违背", 0.0, 1.0),  # 社会：惊讶/顿悟（与声带"新异"同源）
)
韵律因子 = (
    ("基频", 70.0, 400.0), ("幅度", 0.0, 1.0), ("气声", 0.0, 1.0), ("紧喉", 0.0, 1.0),
    ("粗糙", 0.0, 1.0), ("送气", 0.0, 1.0), ("声区", 0.0, 1.0), ("语速", -2.0, 2.0),
)
意图因子 = (("类别", 0.0, 6.0), ("强度", 0.0, 1.0), ("时限", 0.0, 1.0), ("来源", 0.0, 1.0))

# —— 情绪区域表（05 §3.4 的 20 项；区间 → 代表值） ——
# 约定：高=0.85 中=0.5 低=0.15 —=0.5(中性不设) 他人=0 自我=1 +=0.85 −=−0.85
_H, _Z, _L, _N = 0.85, 0.5, 0.15, 0.5
_OTH, _SELF, _POS, _NEG = 0.0, 1.0, 0.85, -0.85
# 情绪名: (V, A, P, G, M, X, T, Nv)   —— 顺序对齐 情感因子
情绪区域表 = {
    "喜悦": (_POS, _Z, _Z, _N, _POS, _N, _L, _L),
    "悲伤": (_NEG, _L, _L, _N, _N, _N, _N, _N),
    "愤怒": (_NEG, _H, _H, _OTH, _NEG, _N, _N, _N),
    "恐惧": (_NEG, _H, _L, _N, _N, _N, _N, _N),
    "惊讶": (_N, _H, _L, _N, _N, _N, _N, _H),
    "羞耻": (_NEG, _H, _L, _SELF, _NEG, _H, _N, _N),
    "内疚": (_NEG, _Z, _Z, _SELF, _NEG, _L, _N, _N),
    "尴尬": (_NEG, _H, _L, _SELF, 0.35, _H, _N, _N),
    "骄傲": (_POS, _Z, _Z, _SELF, _POS, _N, _N, _N),
    "谦卑": (_POS, _L, _L, _SELF, _POS, _N, _N, _N),
    "嫉妒": (_NEG, _Z, _L, _OTH, _NEG, _N, _H, _N),
    "羡慕": (_NEG, _Z, _L, _OTH, _NEG, _N, _L, _N),
    "共情": (_Z, _Z, _L, _OTH, _N, _N, _L, _N),
    "幸灾乐祸": (_POS, _Z, _Z, _OTH, _NEG, _N, _L, _N),
    "感激": (_POS, _Z, _L, _OTH, _POS, _N, _L, _N),
    "轻蔑": (_NEG, _L, _H, _OTH, _NEG, _N, _L, _N),
    "厌恶": (_NEG, _Z, _Z, _OTH, _NEG, _N, _N, _N),
    "惭愧": (_NEG, _L, _L, _SELF, _NEG, _H, _N, _N),
    "优越": (_POS, _L, _H, _SELF, _NEG, _N, _L, _N),
    "敌意": (_NEG, _Z, _H, _OTH, _NEG, _N, _H, _N),
}
临界对 = (("羞耻", "内疚"), ("羡慕", "嫉妒"), ("骄傲", "优越"), ("羞耻", "尴尬"))


class 情感因子层:
    """8 情感轴 + 8 韵律轴 + 4 意图轴的因子化编码/解码（全部复用 数轴+绑+束）。"""

    def __init__(self, 核: FSW内核 = None, 种子: int = 3800, D: int = D_C):
        self.核 = 核 if 核 is not None else FSW内核(种子=种子)
        self.D = int(D)
        def _建轴(表, 前缀):
            return {名: 数轴("%s|%s" % (前缀, 名), self.D, 16, 16, 下, 上)
                    for 名, 下, 上 in 表}
        self.情感轴 = _建轴([(f[0], f[2], f[3]) for f in 情感因子], "情感")
        self.韵律轴 = _建轴(韵律因子, "韵律")
        self.意图轴 = _建轴(意图因子, "意图")
        # 角色码（并入内核原子表 ⇒ 与语义/知识同空间软隔离）；
        # 三张「裸名→原子键」映射：编码束按裸名查轴与角色。
        self._原子s: dict = {}
        def _建角色(表, 前缀):
            return {名: "%s%s" % (前缀, 名) for 名, *_ in 表}
        self.情感角色 = _建角色(情感因子, "情感")
        self.韵律角色 = _建角色(韵律因子, "韵律")
        self.意图角色 = _建角色(意图因子, "意图")
        for 键 in list(self.情感角色.values()) + list(self.韵律角色.values()) \
                + list(self.意图角色.values()):
            self._原子s[键] = self.核.原子("角色|%s" % 键)

    def 槽角色(self, 键: str) -> np.ndarray:
        """通用槽位角色（容量压力测试用超出 20 因子的槽位）。"""
        if 键 not in self._原子s:
            self._原子s[键] = self.核.原子("角色|%s" % 键)
        return self._原子s[键]

    # ── 编码 ──
    def 编码束(self, 组: dict, 轴表: dict, 角色键: dict) -> np.ndarray:
        """组: {裸名: 值}；轴表/角色键: {裸名: 轴/原子键} → 束([绑(role, 轴(值))])。"""
        return 束([绑(self._原子s[角色键[r]], 轴表[r].编码向量(v))
                   for r, v in 组.items()], self.D)

    def 编码情绪(self, 点, 强度: float = None) -> np.ndarray:
        """点: 8 因子 dict（缺省项取 0.5 中性）→ live 情感态束（8 对绑定）。"""
        全 = {名: 0.5 for 名, *_ in 情感因子}
        全.update(点 or {})
        return self.编码束(全, self.情感轴, self.情感角色)

    def 编码情绪区域(self, 名: str) -> np.ndarray:
        return self.编码情绪(dict(zip([f[0] for f in 情感因子], 情绪区域表[名])))

    def 编码韵律(self, 点: dict) -> np.ndarray:
        return self.编码束(点, self.韵律轴, self.韵律角色)

    def 编码意图(self, 点: dict) -> np.ndarray:
        return self.编码束(点, self.意图轴, self.意图角色)

    # ── 回读（解绑 → 档位最近邻 → 值；无 softmax）──
    def 回读(self, E: np.ndarray, 轴表: dict, 角色键: dict) -> dict:
        出 = {}
        for r in 轴表:
            raw = 绑(self._原子s[角色键[r]], E)            # 自逆解绑：F + 串扰
            k = 轴表[r].解码档位(raw)                      # 257 档扫描 O(S·D)
            轴 = 轴表[r]
            值 = 轴.下界 + (轴.上界 - 轴.下界) * (k / 轴.S)
            出[r] = (k, 值, float(相似(raw, 轴._组合原型(k))))
        return 出

    def 回读情绪(self, E: np.ndarray) -> dict:
        return self.回读(E, self.情感轴, self.情感角色)

    def 最近区域(self, E: np.ndarray) -> str:
        """回读 8 因子 → 因子空间最近情绪区域（无 softmax，纯最近邻）。"""
        读 = self.回读情绪(E)
        因子序 = [f[0] for f in 情感因子]
        v = np.array([读[r][1] for r in 因子序])
        最佳, 最距 = None, None
        for 名, 点 in 情绪区域表.items():
            d = float(np.linalg.norm(v - np.asarray(点, np.float64)))
            if 最距 is None or d < 最距 or (d == 最距 and (最佳 is None or 名 < 最佳)):
                最佳, 最距 = 名, d
        return 最佳

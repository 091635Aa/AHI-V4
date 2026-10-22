# -*- coding: utf-8 -*-
"""meeting28/gen_lm.py —— #224 纯 HDC 自回归字符语言模型（对话流的生成核心）。

- 计数关联：二元/三元上下文→下一字符计数（持续学习=在线累加，即学即用）；
- HDC 退避：未见上下文由 字符向量×上下文编码 cos 提供平滑先验（分布式泛化项）；
- 插值打分：score = 3·P3 + 2·P2 + 1·P1 + λ·hdc（零梯度，纯计数+HDC 相似度）；
- 采样：温度由引擎遥测驱动，rng 由 (seed,拍) 派生——同状态确定、异状态不同。
"""
from __future__ import annotations

import numpy as np

BOS = "\u0002"
EOS = "。"


class HDC语言模型:
    def __init__(self, codec, seed: int = 216, λ: float = 0.05):
        self.codec = codec
        self.seed = seed
        self.λ = float(λ)
        self.计数4: dict = {}      # (c1,c2,c3) -> {c:n}（#248 四元退避）
        self.计数3: dict = {}      # (c1,c2) -> {c: n}
        self.计数2: dict = {}      # (c1,)  -> {c: n}
        self.计数1: dict = {}      # 总字频
        self.词汇: set = set()
        self.句集: set = set()     # 已学句（生成自省的真值集）

    # ---------- 持续学习 ----------
    def 学句(self, 句: str) -> int:
        s = [BOS] + [c for c in 句.strip() if c.strip()]
        if len(s) < 2 or not s[-1] == EOS:
            s = s + ([EOS] if s[-1:] != [EOS] else [])
        n = 0
        for i in range(1, len(s)):
            c3 = s[i]
            c2 = s[i - 1]
            c1 = s[i - 2] if i >= 2 else BOS
            if i >= 3:                                    # #248 四元计数
                c0 = s[i - 3]
                self.计数4.setdefault((c0, c1, c2), {})
                self.计数4[(c0, c1, c2)][c3] = \
                    self.计数4[(c0, c1, c2)].get(c3, 0) + 1
            self.计数3.setdefault((c1, c2), {})
            self.计数3[(c1, c2)][c3] = self.计数3[(c1, c2)].get(c3, 0) + 1
            self.计数2.setdefault((c2,), {})
            self.计数2[(c2,)][c3] = self.计数2[(c2,)].get(c3, 0) + 1
            self.计数1[c3] = self.计数1.get(c3, 0) + 1
            self.词汇.update((c3, c2, c1))
            n += 1
        self.句集.add(句.strip())
        return n

    # ---------- HDC 退避先验：下一字符与上下文编码的相容度 ----------
    def _hdc先验(self, 历史: list, c: str) -> float:
        尾 = "".join(历史[-6:]).replace(BOS, "")
        if not 尾:
            return 0.0
        v = self.codec.编码(尾 + c) - self.codec.编码(尾)   # 加入 c 后编码位移
        return float(np.linalg.norm(v)) / 2.0               # ∈[0,~1]：c 与上下文争夺越大越不合

    # ---------- 下一字符分布（插值：3·P3+2·P2+1·P1+λ·hdc） ----------
    def 分布(self, 历史: list) -> dict:
        c1, c2 = 历史[-2], 历史[-1]
        d3 = self.计数3.get((c1, c2), {})
        d2 = self.计数2.get((c2,), {})
        t3 = max(sum(d3.values()), 1)
        t2 = max(sum(d2.values()), 1)
        t1 = max(sum(self.计数1.values()), 1)
        词汇 = sorted(self.词汇 | {EOS})
        p = {}
        for c in 词汇:
            hdc = self._hdc先验(历史, c)
            p[c] = (3.0 * d3.get(c, 0) / t3
                    + 2.0 * d2.get(c, 0) / t2
                    + 1.0 * self.计数1.get(c, 0) / t1
                    + self.λ * max(hdc, 0.0))
        z = sum(p.values())
        return {c: v / z for c, v in p.items()}

    # ---------- 自回归生成（ε-探索约束解码） ----------
    def 生成(self, 种子: str = "", 温度: float = 1.0, rng=None, 最大长: int = 24,
             ε: float = 0.15) -> str:
        """约束解码：默认沿已见三元图行走（层3，流畅）；以概率 ε 退避到已见二元（层2，
        受控偏离=新句组合的来源）；仅当二元也未见过才落到字频（层1）。
        自然终止（采到 EOS）返回带句号文本；被最大长截断时不带（调用方可辨）。"""
        rng = rng if rng is not None else np.random.default_rng(self.seed)
        历史 = [BOS, BOS] + [c for c in 种子 if c.strip()]   # (BOS,BOS)=句首上下文（学句同构）
        out = []
        for _ in range(最大长):
            c1, c2 = 历史[-2], 历史[-1]
            d4 = self.计数4.get(tuple(历史[-3:])) if len(历史) >= 3 else None
            d3 = self.计数3.get((c1, c2), {})
            d2 = self.计数2.get((c2,), {})
            池, 层 = ((d4, 4) if (d4 and rng.random() > 0.5) else
                     (d3, 3) if (d3 and rng.random() > ε) else
                     (d2, 2) if d2 else (d3 or d2 or dict(self.计数1), 1))
            if not 池:
                break
            cs = sorted(池)
            ps = np.array([max(池[c], 1e-12) for c in cs], dtype=np.float64)
            ps = ps ** (1.0 / max(温度, 1e-6))
            ps /= ps.sum()
            c = cs[int(rng.choice(len(cs), p=ps))]
            if c == EOS or c == BOS:
                return "".join(out) + EOS
            out.append(c)
            历史.append(c)
        return "".join(out)

    # ---------- 流畅性代理：文本的三元命中率 ----------
    def 三元命中率(self, 文本: str) -> float:
        s = [c for c in 文本 if c.strip()]
        if len(s) < 3:
            return 1.0
        命中 = sum(1 for i in range(2, len(s))
                   if s[i] in self.计数3.get((s[i - 2], s[i - 1]), {}))
        return 命中 / (len(s) - 2)

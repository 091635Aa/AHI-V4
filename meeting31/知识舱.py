# -*- coding: utf-8 -*-
"""meeting31/知识舱.py —— #265 可插拔知识舱室：隔离/挂载/摘除。

舱=独立知识隔间：own 语言模型（门控/生成）+own 段/向量/字袋+own 检索。
核心库（codec 段+lm_m）保持纯净——舱室知识不混入核心。
摘除=整体剥离（存储清空、检索不可达）；挂载/摘除动作由综合体入账。
"""
from __future__ import annotations

import numpy as np

from gen_lm import HDC语言模型


class 知识舱:
    def __init__(self, 名字: str, codec, seed: int):
        self.名字 = 名字
        self.codec = codec
        self.挂载 = True
        self.行s: list = []
        self.向量s: list = []
        self.袋s: list = []
        self.lm = HDC语言模型(codec, seed=seed)

    # ---------- 写入（仅挂载时） ----------
    def 存行(self, 行: str) -> None:
        self.行s.append(行)
        self.向量s.append(self.codec.编码(行))
        self.袋s.append(self.codec._字袋(行))
        self.lm.学句(行)

    def 命中(self, 行: str) -> float:
        return self.lm.三元命中率(行)

    # ---------- 检索（仅挂载时可达） ----------
    def 检索(self, q: str, k: int = 2) -> list:
        if not self.挂载 or not self.行s:
            return []
        v = self.codec.编码(q)
        vb = self.codec._字袋(q)
        nv = float(np.linalg.norm(v)) + 1e-9
        nb = float(np.linalg.norm(vb)) + 1e-9
        sc = []
        for sv, sb in zip(self.向量s, self.袋s):
            s1 = float(v @ sv) / (nv * float(np.linalg.norm(sv)) + 1e-9)
            s2 = float(vb @ sb) / (nb * float(np.linalg.norm(sb)) + 1e-9)
            sc.append(max(s1, s2))
        idx = sorted(range(len(sc)), key=lambda i: -sc[i])[:k]
        return [(self.行s[i], sc[i]) for i in idx]

    # ---------- 摘除（整体剥离） ----------
    def 摘除(self) -> None:
        self.行s = []
        self.向量s = []
        self.袋s = []
        self.lm = HDC语言模型(self.codec, seed=self.lm.seed)
        self.挂载 = False

    def 挂载恢复(self) -> None:
        self.挂载 = True

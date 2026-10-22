# -*- coding: utf-8 -*-
"""meeting31/writer.py —— #253 长时持续输出：以最近细读的教材语料为条件（轮转题材源）、
逐拍从 lm_m（4 元计数模型）条件采样一句——题材轮转+近期重复拒绝+逐次升温，防 n-gram
吸引子循环（首测实证：链自己尾巴 4 句即枯竭）。

诚实边界：输出=语料统计的重组（作者式续写），不是语义创作；作品不作为知识来源入账。
全部确定性：同 seed+同账本状态+同起点=逐位一致。
"""
from __future__ import annotations

from collections import deque

import numpy as np


class 写作体:
    def __init__(self, 桥, 路径: str):
        self.桥 = 桥
        self.路径 = 路径
        self.句数 = 0
        self.字数 = 0
        self.唯一句: set = set()
        self.最近句 = deque(maxlen=50)     # 近期重复拒绝
        self.题材游标 = 0

    # ---------- 题材源：轮转最近细读的教材账本行 ----------
    def _题材条件(self) -> str:
        题材s = [x for t, x in self.桥.账本 if t == "教材"]
        if not 题材s:
            return ""
        self.题材游标 = (self.题材游标 + 1) % min(len(题材s), 200)
        return 题材s[-1 - self.题材游标][-6:]

    # ---------- 单句条件生成（轮转题材+重复拒绝+逐次升温） ----------
    def 写一句(self) -> int:
        桥 = self.桥
        条件 = self._题材条件()
        if not 条件 or not 桥.lm_m.计数1:
            return 0
        τ0 = float(0.7 + 0.5 * (桥.ñ if 桥.ñ is not None else 1.0))
        for 尝试 in range(5):
            rng = np.random.default_rng((桥.seed, 桥.拍, 21, 尝试))
            s = 桥.lm_m.生成(种子=条件, 温度=τ0 + 0.1 * 尝试, rng=rng, 最大长=48)
            if len(s) >= 6 and s not in self.最近句 and s not in self.唯一句:
                with open(self.路径, "a", encoding="utf-8") as f:
                    f.write(s + "\n")
                self.句数 += 1
                self.字数 += len(s)
                self.唯一句.add(s)
                self.最近句.append(s)
                return len(s)
        return 0                                             # 5 次全重复→本拍静默

    # ---------- 单次最长输出：链式生成直到枯竭（重复拒满/达上限） ----------
    def 单次最长输出(self, 上限句: int = 400) -> dict:
        起句, 起字 = self.句数, self.字数
        静默连发 = 0
        for i in range(上限句):
            n = self.写一句()
            静默连发 = 静默连发 + 1 if n == 0 else 0
            if 静默连发 >= 8:                                # 连续 8 拍无可产新句=枯竭
                break
        return dict(句数=self.句数 - 起句, 字数=self.字数 - 起字,
                    唯一句=len(self.唯一句), 上限句=上限句,
                    触发="枯竭(连续静默8拍)" if 静默连发 >= 8 else "达上限句数")

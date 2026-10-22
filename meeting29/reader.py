# -*- coding: utf-8 -*-
"""meeting29/reader.py —— #232 阅读体：自定节奏阅读+注意门控+自查顿悟。

书放在那里；翻页/重读/联想/停下想由引擎 agency 机制逐拍自选；
细读门控=lm_m 对该行命中率<0.5（预测不了→值得学，已能预测→略过）；
顿悟两类自查：消化（首遇<0.5∧重读≥0.5）、连接（联想找到跨文件的两条强命中）。
"""
from __future__ import annotations

import os
import sys

ROOT = r"<workspace>/03_AHI\AHI2.1"
sys.path.insert(0, ROOT)
# ★迁移修正（meeting43）：本树 meeting29/meeting28（原 ROOT=AHI2.1 会静默解析到 2.1）
_AHI3 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_AHI3, "meeting29"))
sys.path.insert(0, os.path.join(_AHI3, "meeting28"))

import numpy as np

from learner import 学习体
from dialogue_v5 import _关键词集

动作集 = ["继续读", "重读", "联想", "停下想"]
细读门限 = 0.5


class 阅读体(学习体):
    """v8 阅读体（#232）：引擎 agency 自选阅读动作。"""

    def __init__(self, seed: int = 232, 私偏κ: float = 10.0):
        super().__init__(seed=seed, 私偏κ=私偏κ)
        self.位置 = 0
        self.首遇命中: dict = {}       # 位置→首遇 trigram 命中率（自查顿悟的原料）
        self.顿悟账本: list = []       # (类型, 内容, 数字)
        self.阅读统计 = dict(继续读=0, 重读=0, 联想=0, 停下想=0, 细读=0, 略读=0)

    def 细读(self, 行: str, h: float) -> None:
        if self.注册去重(行):
            self.段源[行] = "教材"
        self.lm_m.学句(行)
        self.账本.append(("教材", 行))
        self.阅读统计["细读"] += 1

    def 读一拍(self, 书: list, 文件界: list) -> dict:
        """一拍阅读：引擎 agency 自选动作。文件界=[各行所属文件号]（连接顿悟核对用）。"""
        self.位置 = min(self.位置, max(len(书) - 1, 0))
        行 = 书[self.位置]
        u = self.感知向量(self.codec.编码(行))
        rec = self._引擎步(u)
        self._吃(rec)
        动作 = 动作集[int(rec["a星序号"]) % 4]
        self.阅读统计[动作] += 1
        if 动作 == "继续读":
            h = self.lm_m.三元命中率(行)
            self.首遇命中.setdefault(self.位置, h)
            if h < 细读门限:
                self.细读(行, h)
            else:
                self.阅读统计["略读"] += 1     # 已能预测→略过不学（选择性注意）
            self.位置 += 1
        elif 动作 == "重读":
            self.位置 = max(self.位置 - 3, 0)
            行2 = 书[self.位置]
            h0 = self.首遇命中.get(self.位置, self.lm_m.三元命中率(行2))
            h1 = self.lm_m.三元命中率(行2)
            if h0 < 细读门限 <= h1:
                self.顿悟账本.append(("消化", 行2, (h0, h1)))
                self.首遇命中[self.位置] = h1
        elif 动作 == "联想":
            hits = self._检索(行, k=2)
            if len(hits) == 2 and hits[0][1] >= 0.5 and hits[1][1] >= 0.5:
                段1, 段2 = hits[0][0], hits[1][0]
                位置1 = self.codec._段.index(段1) if 段1 in self.codec._段 else -1
                self.顿悟账本.append(("连接", (段1, 段2), (self.位置,)))
        else:                                    # 停下想：内部生成（不自言输出）
            rng = np.random.default_rng((self.seed, self.拍, 7))
            self.lm.生成(种子="", 温度=1.0, rng=rng, 最大长=16)
        self.拍账本()
        return dict(动作=动作, 位置=self.位置)

    def 拍账本(self) -> None:
        pass                                     # 账本在细读/引擎层已记

    def 读到(self, 书: list, 文件界: list, 最多拍: int) -> dict:
        for _ in range(最多拍):
            if self.位置 >= len(书) - 1 and self.阅读统计["重读"] > 0:
                break
            self.读一拍(书, 文件界)
        return dict(拍=最多拍, 覆盖率=round(self.位置 / max(len(书) - 1, 1), 3),
                    统计=dict(self.阅读统计), 顿悟=len(self.顿悟账本))

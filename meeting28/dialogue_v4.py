# -*- coding: utf-8 -*-
"""meeting28/dialogue_v4.py —— #224 对话体 v4：生成式对话流（纯 HDC 自回归）。

在 v3（三分路由/槽位巩固/自主拍/阈值检索）之上：
- 持续学习：每轮用户文本即学即入 LM（计数关联，零梯度）；
- 生成式表面：连接语由 HDC 语言模型自回归采样（温度=引擎新异度、rng=(seed,拍)）；
- 生成自省：对自己生成的连接语做新异性自检（∉已学句集→自报告『这句是我自己组的』）；
- 自指/未知诚实路径保持 v3 原样（自报告=纯遥测；不知道=不编）。
"""
from __future__ import annotations

import os
import sys

import numpy as np

ROOT = r"<workspace>/03_AHI\AHI2.1"
sys.path.insert(0, ROOT)
# ★迁移修正（meeting43）：本树 meeting28（原 join(ROOT,"meeting28") 会静默解析到 AHI2.1）
_M28 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "meeting28")
sys.path.insert(0, _M28)

from dialogue_v3 import 对话桥v3
from gen_lm import HDC语言模型, BOS, EOS

连接语料 = ("原来如此。\n这样啊，我明白了。\n嗯，我记住了。\n这让我想到一些事。\n"
            "我还在琢磨这件事。\n听你说这些，我觉得挺有意思。\n好，这件事我收下了。\n"
            "原来这里还有一层。\n这样的话就说得通了。\n我把它放进记忆里了。\n"
            "嗯，这个说法我认同。\n有点意思，我接着想。\n原来如此，我又学到一点。\n"
            "这一条我记下了。\n行，这话有道理。\n听起来是一回事。\n嗯。")


class 对话桥v4(对话桥v3):
    """v4 生成式对话桥（#224）。"""

    def __init__(self, seed: int = 216, 私偏κ: float = 10.0):
        super().__init__(seed=seed, 私偏κ=私偏κ)
        self.lm = HDC语言模型(self.codec, seed=seed)
        self.自省记录: list = []      # (生成串, 打标bool)——自省校准的原料
        self.连接语料 = 连接语料
        self.教_v2(连接语料, epochs=1)             # 连接语进引擎+LM（入教史=重放一致）

    # ---------- 持续学习入口 ----------
    def 学句(self, 句: str) -> None:
        self.lm.学句(句)

    def 教_v2(self, 文本, epochs: int = 3, 记录: bool = True) -> dict:
        for 句 in [s.strip() for s in 文本.replace("。", "。\n").split("\n") if s.strip()]:
            self.学句(句)                          # LM 同步学习（计数幂等，重放一致）
        return super().教_v2(文本, epochs=epochs, 记录=记录)

    # ---------- 生成连接语（引擎条件化） ----------
    def 生成连接(self) -> str:
        τ = float(0.7 + 0.5 * (self.ñ if self.ñ is not None else 1.0))
        for 尝试 in range(3):
            rng = np.random.default_rng((self.seed, self.拍, 尝试))
            s = self.lm.生成(种子="", 温度=τ, rng=rng, 最大长=24)
            if 4 <= len(s) <= 20 and s.endswith(EOS):
                新 = s not in self.lm.句集
                self.自省记录.append((s, 新))
                return s + ("（开头这句是我自己组的）" if 新 else "")
        s = "嗯。"
        self.自省记录.append((s, False))
        return s

    # ---------- 问句回答：生成连接语 + 检索事实（覆盖 v2 的模板封套） ----------
    def _答问(self, q: str) -> str:
        hits = self._检索(q, k=2)
        if not hits:
            return "我还没有任何记忆——你先教我一些事，我再回答。"
        (段1, s1) = hits[0]
        段2 = hits[1] if len(hits) > 1 else None
        if s1 < self.阈值:
            return ("这个我还没学过——记忆里最接近的是『%s』，但相似度只有 %.2f"
                    "（阈值 %.2f），我不想瞎猜。" % (段1, s1, self.阈值))
        连接 = self.生成连接()
        if 段2 is not None and 段2[1] >= self.阈值 and 段2[0] != 段1:
            return "%s『%s』；还有一条相关的：『%s』。" % (连接, 段1, 段2[0])
        return "%s『%s』。" % (连接, 段1)

    # ---------- 对话主入口（每轮即学 + 陈述表面生成化） ----------
    def 步(self, 用户文本: str) -> dict:
        self.学句(用户文本)                        # 持续学习：每轮即学即用
        r = super().步(用户文本)
        if r["响应"].startswith("记下了"):
            连接 = self.生成连接()
            r["响应"] = "%s%s" % (连接, r["响应"])
        return r

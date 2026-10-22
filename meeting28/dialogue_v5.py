# -*- coding: utf-8 -*-
"""meeting28/dialogue_v5.py —— #226 声道分离：知道≠经历过。

修复 v4 被所有者发现的声道混淆逻辑错误（用户事实被第一人称断言，如
『我最喜欢下雨天』『我的工作是程序员』）：
- 声道分离：LM 拆 用户声道(lm_u)/自声道(lm_s=lm)——用户文本只入用户声道+记忆库；
  自声道只在自我语料（连接语+澄清句）上训练，结构上采样不到用户独有词；
- 归属引用：记忆段带 源（用户/自我），答问一律『你说过：…』/『我自己提过：…』；
- 澄清通路：『你+吃/养/工作/喜欢/去过』问题 → 澄清『我没有身体…知道不等于经历过』
  +归属引用；生成输出加经验谓词守卫（引号外 我+经验谓词 → 拒绝重生成）。
"""
from __future__ import annotations

import os
import sys

ROOT = r"<workspace>/03_AHI\AHI2.1"
sys.path.insert(0, ROOT)
# ★迁移修正（meeting43）：本树 meeting28（原 join(ROOT,"meeting28") 会静默解析到 AHI2.1）
_M28 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "meeting28")
sys.path.insert(0, _M28)

from dialogue_v4 import 对话桥v4, 连接语料
from gen_lm import HDC语言模型

自我澄清语料 = ("我没有身体——不吃东西、不养宠物，也没有职业经历，这类经历不属于我。\n"
                "我知道不等于经历过。")
经验谓词 = ("我最", "我养", "我吃", "我的工", "我去", "我喜欢", "我不太喜欢",
            "我今天", "我昨天", "我周末")
经验问词 = ("吃", "养", "工作", "去过", "喜欢")


def _引号外文本(s: str) -> str:
    """去掉 『…』 引用后的剩余文本（守卫扫描域）。"""
    out, 深度 = [], 0
    for ch in s:
        if ch == "『":
            深度 += 1
        elif ch == "』":
            深度 = max(深度 - 1, 0)
        elif 深度 == 0:
            out.append(ch)
    return "".join(out)


停用符 = set("？?。，,、！!…—：:；;（）()『』\"\"'")


def _关键词集(s: str) -> set:
    return set(c for c in s if c.strip() and c not in 停用符)


class 对话桥v5(对话桥v4):
    """v5 声道分离对话桥（#226）。"""

    def __init__(self, seed: int = 216, 私偏κ: float = 10.0):
        self._声道 = "自我"                        # __init__ 期间的教学=自声道
        self.段源: dict = {}                       # 记忆段→源（用户/自我）
        super().__init__(seed=seed, 私偏κ=私偏κ)   # v4 init：lm 创建+连接语（走 self.学句→自声道）
        self.教_v2(自我澄清语料, epochs=1)         # 澄清句入自声道+记忆库
        self.lm_s = self.lm                        # 自声道
        self.lm_u = HDC语言模型(self.codec, seed=seed + 1)   # 用户声道（独立）
        for 句 in [s.strip() for s in (连接语料 + 自我澄清语料).replace("。", "。\n").split("\n") if s.strip()]:
            self.段源.setdefault(句, "自我")
        self._声道 = "用户"

    # ---------- 声道路由 ----------
    def 学句(self, 句: str) -> None:
        (self.lm if self._声道 == "自我" else self.lm_u).学句(句)

    def 教_v2(self, 文本, epochs: int = 3, 记录: bool = True) -> dict:
        if 记录:
            self.教史.append((str(文本), int(epochs), self._声道))   # 3 元组带声道
        for 句 in [s.strip() for s in 文本.replace("。", "。\n").split("\n") if s.strip()]:
            self.段源.setdefault(句, self._声道)
        return super().教_v2(文本, epochs=epochs, 记录=False)

    # ---------- 重放（教史带声道；__init__ 已教的自声道条目跳过防计数翻倍） ----------
    def _重放(self, d) -> None:
        教seq = [tuple(x) for x in d.get("教史") or []]
        for e in 教seq:
            if e[0] in (连接语料, 自我澄清语料):
                continue                               # __init__ 已教，跳过防 ×2
            self._声道 = e[2] if len(e) > 2 else "用户"
            self.教_v2(e[0], epochs=int(e[1]), 记录=False)
        self._声道 = "用户"
        for t in d["输入"]:
            if t is None:
                self.自主拍(1)
            else:
                self.步(t)
        self.教史 = 教seq or list(self.教史)

    # ---------- 经验谓词守卫的连接语生成 ----------
    def 生成连接(self) -> str:
        τ = float(0.7 + 0.5 * (self.ñ if self.ñ is not None else 1.0))
        for 尝试 in range(3):
            rng = __import__("numpy").random.default_rng((self.seed, self.拍, 尝试))
            s = self.lm.生成(种子="", 温度=τ, rng=rng, 最大长=24)
            if 4 <= len(s) <= 20 and s.endswith("。"):
                if any(p in s for p in 经验谓词):      # 引号外第一人称经验断言→拒绝
                    continue
                新 = s not in self.lm.句集
                self.自省记录.append((s, 新))
                return s + ("（开头这句是我自己组的）" if 新 else "")
        s = "嗯。"
        self.自省记录.append((s, False))
        return s

    # ---------- 澄清通路：你+经验谓词 → 知道≠经历过 ----------
    经验检索阈值 = 0.35              # 字符集重叠系数口径（与 v3 槽位检测同款）——
                                     # 「宠物↔猫」等零共享字同义仍受 #221 局限约束，
                                     # 此处仅把经验分支检索对齐到集合口径（不声称解决同义）

    def _自报告(self, q: str) -> str:
        if any(w in q for w in 经验问词):
            t = q.replace("你", "我")
            新集 = _关键词集(t)
            段, 最佳 = None, 0.0
            for 旧文本 in self.codec._段:
                if self.段源.get(旧文本, "用户") != "用户":
                    continue                       # 经验分支只在用户声道段检索
                旧集 = _关键词集(旧文本)
                重叠 = len(新集 & 旧集) / max(min(len(新集), len(旧集)), 1)
                if 重叠 > 最佳:
                    段, 最佳 = 旧文本, 重叠
            if 段 is None or 最佳 < self.经验检索阈值:
                段 = None
            base = "我没有身体——不吃东西、不养宠物，也没有职业经历，这类经历不属于我。"
            if 段:
                return base + "我知道的只是你说过：『%s』——知道不等于经历过。" % 段
            return base + "这方面我也还没有相关记忆。"
        return super()._自报告(q)

    # ---------- 答问：归属引用（覆盖 v4 的无归属表面） ----------
    def _答问(self, q: str) -> str:
        hits = self._检索(q, k=2)
        if not hits:
            return "我还没有任何记忆——你先教我一些事，我再回答。"
        (段1, s1) = hits[0]
        段2 = hits[1] if len(hits) > 1 else None
        if s1 < self.阈值:
            return ("这个我还没学过——记忆里最接近的是『%s』，但相似度只有 %.2f"
                    "（阈值 %.2f），我不想瞎猜。" % (段1, s1, self.阈值))
        源 = self.段源.get(段1, "用户")
        引用 = ("你说过" if 源 == "用户" else "我自己提过") + "：『%s』" % 段1
        连接 = self.生成连接()
        if 段2 is not None and 段2[1] >= self.阈值 and 段2[0] != 段1:
            return "%s%s；还有一条相关的：『%s』。" % (连接, 引用, 段2[0])
        return "%s%s。" % (连接, 引用)

    # ---------- 对话入口（陈述标记源=用户） ----------
    def 步(self, 用户文本: str) -> dict:
        from dialogue_v2 import 是问句
        if not 是问句(用户文本):
            self.段源[用户文本] = "用户"
        return super().步(用户文本)

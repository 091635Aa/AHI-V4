# -*- coding: utf-8 -*-
"""meeting29/learner.py —— #230 学习体：给目录→持续探索→学习→能力考试。

第三声道：教材（材料内容既非用户也非自我——不得被断言为任何人的经历）。
探索器：遍历目录→切句→注册记忆(源=教材)+教材声道计数学习+引擎步进+账本记录。
"""
from __future__ import annotations

import os
import re
import sys

ROOT = r"<workspace>/03_AHI\AHI2.1"
sys.path.insert(0, ROOT)
# ★迁移修正（meeting43）：本树 meeting28（原 ROOT=AHI2.1 会静默解析到 2.1，令本树修复失效）
_AHI3 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_AHI3, "meeting28"))

from dialogue_v6 import 对话桥v6
from gen_lm import HDC语言模型
from dialogue_v5 import _关键词集


class 学习体(对话桥v6):
    """v7 学习体（#230）：v6 全部机制 + 教材声道 + 目录探索。"""

    def __init__(self, seed: int = 230, 私偏κ: float = 10.0):
        super().__init__(seed=seed, 私偏κ=私偏κ)
        self.lm_m = HDC语言模型(self.codec, seed=seed + 2)   # 教材声道
        self.统计 = dict(文件=0, 句=0, epoch=0)

    # ---------- 目录探索（持续学习） ----------
    def 探索(self, 目录: str, 最多文件: int = 5, 每文件句上限: int = 200,
            扩展: tuple = (".py",)) -> dict:
        文件s = sorted(f for f in os.listdir(目录) if f.endswith(扩展))
        for 名 in 文件s[:最多文件]:
            try:
                行s = open(os.path.join(目录, 名), encoding="utf-8",
                           errors="ignore").read().splitlines()
            except OSError:
                continue
            句s = [ln.strip() for ln in 行s if 6 <= len(ln.strip()) <= 120]
            for 句 in 句s[:每文件句上限]:
                if self.注册去重(句):
                    self.段源[句] = "教材"
                self.lm_m.学句(句)
                u = self.感知向量(self.codec.编码(句))
                self._吃(self._引擎步(u))
                self.账本.append(("教材", 句))
                self.统计["句"] += 1
            self.统计["文件"] += 1
            self.自主拍(2)                      # 探索间隙引擎自发活动
        self.统计["epoch"] += 1
        return dict(self.统计)

    # ---------- 代码行不槽位塌缩（dump/dumps 字符集重叠 ≥0.85 会合并——概念名必须保真） ----------
    代码式 = re.compile(r"[A-Za-z_][A-Za-z_0-9]{2,}")

    def 注册去重(self, 文本: str) -> bool:
        if 文本 in self.codec._段:
            return False
        if self.代码式.search(文本):
            self.codec.注册段(文本)                # 代码行：仅精确去重
            return True
        return super().注册去重(文本)              # 自然语言：槽位更新（v3 语义）

    # ---------- 检索三通道：概念名整体匹配 ⊕ bag-cos ⊕ 集合重叠 ----------
    # 查询含代码标识符（如 dump/JSONDecodeError）时，包含该标识符的段直接优先——
    # 概念名检索=子串匹配（代码检索的正确语义）；模糊通道留给自然语言查询。
    def _检索(self, 文本: str, k: int = 2) -> list:
        import numpy as np
        if not self.codec._段:
            return []
        v = self.codec.编码(文本)
        vb = self.codec._字袋(文本)
        nv = float(np.linalg.norm(v)) + 1e-9
        nb = float(np.linalg.norm(vb)) + 1e-9
        新集 = _关键词集(文本)
        tokens = re.findall(r"[A-Za-z_][A-Za-z_0-9]{2,}", 文本)
        sc = []
        for i, (sv, sb) in enumerate(zip(self.codec._段向量, self.codec._段字袋)):
            旧文本 = self.codec._段[i]
            if tokens and any(t in 旧文本 for t in tokens):
                sc.append((1.0, i))                       # 概念名整体匹配（最高通道）
                continue
            s1 = float(v @ sv) / (nv * float(np.linalg.norm(sv)) + 1e-9)
            s2 = float(vb @ sb) / (nb * float(np.linalg.norm(sb)) + 1e-9)
            旧集 = _关键词集(旧文本)
            重叠 = len(新集 & 旧集) / max(min(len(新集), len(旧集)), 1)
            sc.append((max(s1, s2, 重叠), i))
        sc.sort(reverse=True)
        s星 = sc[0][0]
        近因 = [i for s, i in sc if s >= s星 - 0.02]   # 近因平局裁决（v3 同款）
        最佳i = max(近因)
        最佳分 = next(s for s, i in sc if i == 最佳i)
        其余 = [(s, i) for s, i in sc if i != 最佳i][:max(k - 1, 0)]
        return [(self.codec._段[最佳i], 最佳分)] + \
            [(self.codec._段[i], s) for s, i in 其余]

    # ---------- 答问归属扩展：教材 ----------
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
        归属 = {"用户": "你说过", "自我": "我自己提过", "教材": "教材里说"}[源]
        引用 = "%s：『%s』" % (归属, 段1)
        连接 = self.生成连接()
        if 段2 is not None and 段2[1] >= self.阈值 and 段2[0] != 段1:
            return "%s%s；还有一条相关的：『%s』。" % (连接, 引用, 段2[0])
        return "%s%s。" % (连接, 引用)

    # ---------- 经历归属（考试 C 的对话面）：学过≠写过 ----------
    def _自报告(self, q: str) -> str:
        if ("学过" in q or "写过" in q) and self._声道 == "用户":
            t = re.findall(r"[A-Za-z_][A-Za-z_0-9]{2,}", q)
            tok = t[0] if t else None
            if tok:
                n = sum(1 for tp, x in self.账本 if tp == "教材" and tok in x)
                行动n = sum(1 for tp, x in self.账本 if tp == "行动" and tok in x)
                hits = self._检索(q, k=1)
                段 = hits[0][0] if hits and hits[0][1] >= self.阈值 else None
                base = ("『%s』在我的账本是教材记录（%d 条），不是我的经历"
                        "——我没写过它。" % (tok, n))
                if 段:
                    return base + "教材里说：『%s』" % 段
                return base
            return super()._自报告(q)
        return super()._自报告(q)


# ---------- 能力考试 ----------
def 考试B_预测(桥, 训练s, 留出s):
    """留出语句的 trigram 命中率 + 下一字符 top-1 准确率。"""
    命中 = sum(桥.lm_m.三元命中率(s) for s in 留出s) / max(len(留出s), 1)
    对 = 总 = 0
    for s in 留出s:
        cs = [c for c in s if c.strip()]
        for i in range(2, len(cs)):
            d = 桥.lm_m.计数3.get((cs[i - 2], cs[i - 1]), {})
            if d:
                对 += int(max(d, key=d.get) == cs[i])
            总 += 1
    return dict(命中率=命中, top1=对 / max(总, 1))


def 考试A_归属检索(桥, 问题s):
    正 = 0
    明细 = []
    for q, key in 问题s:
        r = 桥.步(q)["响应"]
        ok = (key in r) and ("教材里说" in r)
        正 += int(ok)
        明细.append((q, ok))
    return 正, 明细

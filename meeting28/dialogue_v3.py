# -*- coding: utf-8 -*-
"""meeting28/dialogue_v3.py —— #222 对话体 v3：可持续性（同义在线学习/巩固/自主拍）。

四支柱：
- 语义在线学习：对话全程文本（含问句）进共现统计（不进答案库）；巩固=codec.精化
  （字符向量向上下文袋捆绑，纯确定性零拟合）；
- 记忆巩固：注册去重 + 周期性精化重建段缓存 + 阈值重标定（v3.重标定，仪器侧调用）；
- 引擎自主拍：对话间隙零外感+内生偏置自发步进——引擎在无用户时仍活着，拍数与对话轮解耦；
- 继承 v2 三分路由与全部诚实边界（现象学体验不可声称）。
"""
from __future__ import annotations

import os
import sys

import numpy as np

ROOT = r"<workspace>/03_AHI\AHI2.1"
sys.path.insert(0, ROOT)
# ★迁移修正（meeting43）：本树 meeting28（原 join(ROOT,"meeting28") 会把 meeting28 模块静默
#   解析到 AHI2.1，令本树修复失效）。ROOT 保留给引擎依赖（meeting15/17）。
_M28 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "meeting28")
sys.path.insert(0, _M28)

from dialogue_v2 import 对话桥v2


class 对话桥v3(对话桥v2):
    """v3 可持续对话桥（#222）：在 v2 之上加 巩固/共现/自主拍。"""

    巩固周期 = 12                      # 每 12 轮巩固一次

    def __init__(self, seed: int = 216, 私偏κ: float = 10.0):
        super().__init__(seed=seed, 私偏κ=私偏κ)
        self.自主拍数 = 0
        self.轮距巩固 = 0
        self.巩固次数 = 0

    # ---------- 记忆槽位更新（#222：同骨架重复属性=同一槽位，原位替换为最新绑定） ----------
    槽位阈值 = 0.85                   # 字符集重叠系数 ≥0.85 ⇒ 同槽。实测口径：同句改一位
                                      # 数字 9/10=0.90；不同事实 ≤6/9=0.67——干净分离。
                                      # （不用字袋 cos：sign 袋被差异字符随机向量支配，
                                      #  差 1 字的近同句 cos≈0，会漏判/乱判——已证伪）

    def 注册去重(self, 文本: str) -> bool:
        if 文本 in self.codec._段:
            return False
        新集 = set(c for c in 文本 if c.strip())
        for i, 旧文本 in enumerate(self.codec._段):
            旧集 = set(c for c in 旧文本 if c.strip())
            重叠 = len(新集 & 旧集) / max(min(len(新集), len(旧集)), 1)
            if 重叠 >= self.槽位阈值:                  # 同一记忆槽 → 原位更新
                self.codec._段[i] = 文本
                self.codec._段向量[i] = self.codec.编码(文本)
                self.codec._段字袋[i] = self.codec._字袋(文本)
                return True
        self.codec.注册段(文本)
        return True

    # ---------- 巩固（语义精化+段缓存重建） ----------
    def 巩固(self) -> dict:
        r = self.codec.精化(α=0.3, 轮数=2)
        self.轮距巩固 = 0
        self.巩固次数 += 1
        return r

    # ---------- 对话主入口（全程文本进共现；周期巩固） ----------
    def 步(self, 用户文本: str) -> dict:
        self.codec.共现(用户文本)
        self.轮距巩固 += 1
        if self.轮距巩固 >= self.巩固周期:
            self.巩固()
        return super().步(用户文本)

    # ---------- 引擎自主拍（无用户输入时的自发活动；逐拍入历史保重放一致） ----------
    def 自主拍(self, n: int = 5) -> int:
        for _ in range(n):
            self._引擎步(np.zeros(16, dtype=np.float32))
            self.历史.append((self.拍, None, "自主", "", ""))
        self.自主拍数 += int(n)
        return self.自主拍数

    # ---------- 阈值重标定（精化改变相似度尺度后由仪器侧调用） ----------
    def 重标定(self, 无关族, 相关族) -> dict:
        ir = max((self._检索(q, k=1) or [("", -1)])[0][1] for q in 无关族)
        rel = min((self._检索(q, k=1) or [("", -1)])[0][1] for q in 相关族)
        if rel > ir:
            self.阈值 = round(min(max((ir + rel) / 2.0, 0.2), 0.45), 3)
        return dict(无关最大=round(ir, 3), 相关最小=round(rel, 3), 新阈值=self.阈值)

    # ---------- 持久化（重放含共现、周期巩固与自主拍——同输入序列=确定性重建） ----------
    def 保存(self, 路径: str) -> None:
        import pickle as _p
        with open(路径, "wb") as f:
            _p.dump(dict(seed=self.seed, 教史=list(self.教史),
                         输入=[h[1] for h in self.历史]), f)

    def _重放(self, d) -> None:
        教seq = [tuple(x) for x in d.get("教史") or []]
        for 文本, epochs in 教seq:
            self.教_v2(文本, epochs=epochs, 记录=False)
        for t in d["输入"]:
            if t is None:
                self.自主拍(1)                         # 自主拍标记逐拍重放
            else:
                self.步(t)
        self.教史 = 教seq or list(self.教史)    # 教史档案恢复（跨存/载链不断裂）

    @classmethod
    def 载入(cls, 路径: str) -> "对话桥v3":
        import pickle as _p
        with open(路径, "rb") as f:
            d = _p.load(f)
        桥 = cls(seed=d["seed"])
        桥._重放(d)
        return 桥

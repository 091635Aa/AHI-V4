# -*- coding: utf-8 -*-
"""meeting28/dialogue_v6.py —— #228 自身补丁：缺口由系统从自己的记录里发现。

替代 #226 的设计者补丁（词表/澄清模板/路由全是注入的——本层不存在任何词表）：
- 经历账本：每轮自动记录（接收报告/自身行动）——行动记录只记引号外文本（引用不算行动）；
- 账本查证：自指问题的字符域逐一查账——报告有∧行动 0 → 自己发现『这类经历我没有』
  （缺失域，报出统计数供核对）；行动有 → 自己发现『这是真做过的』（经历域，防全盘否认）；
- 双声道互审：生成输出用用户声道模型（lm_u）审计——trigram 命中率 >0.5 = 越权内容，
  自己发现并拒绝。
"""
from __future__ import annotations

import os
import sys

ROOT = r"<workspace>/03_AHI\AHI2.1"
sys.path.insert(0, ROOT)
# ★迁移修正（meeting43）：本树 meeting28（原 join(ROOT,"meeting28") 会静默解析到 AHI2.1）
_M28 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "meeting28")
sys.path.insert(0, _M28)

from dialogue_v5 import 对话桥v5, _引号外文本, _关键词集
from dialogue_v4 import 对话桥v4

审计阈值 = 0.5                     # lm_u trigram 命中率 >0.5 = 用户声道内容（越权）


class 对话桥v6(对话桥v5):
    """v6 自身补丁对话桥（#228）。"""

    def __init__(self, seed: int = 216, 私偏κ: float = 10.0):
        self.账本: list = []       # (类型, 文本)：接收报告 / 自身行动（引号外）
        super().__init__(seed=seed, 私偏κ=私偏κ)

    # ---------- 经历账本（每轮自动记账） ----------
    def 步(self, 用户文本: str) -> dict:
        r = super().步(用户文本)
        self.账本.append(("接收", 用户文本))
        self.账本.append(("行动", _引号外文本(r["响应"])))
        return r

    def 账本查证(self, 域: str, 账本=None) -> dict:
        """域=单字符：返回 (报告数, 行动数)。账本=None 用当前账本；外部核对可传答题时刻前缀。"""
        账本 = self.账本 if 账本 is None else 账本
        报告 = sum(1 for t, x in 账本 if t == "接收" and 域 in x)
        行动 = sum(1 for t, x in 账本 if t == "行动" and 域 in x)
        return dict(报告=报告, 行动=行动)

    # ---------- 教学也入账（用户教学=接收报告；自我教学=自我配置，不算报告） ----------
    def 教_v2(self, 文本, epochs: int = 3, 记录: bool = True) -> dict:
        r = super().教_v2(文本, epochs=epochs, 记录=记录)
        类型 = "接收" if self._声道 == "用户" else "自我"
        for 句 in [s.strip() for s in 文本.replace("。", "。\n").split("\n") if s.strip()]:
            self.账本.append((类型, 句))
        return r

    # ---------- 账本查证替代词表路由 ----------
    def _自报告(self, q: str) -> str:
        域们 = _关键词集(q) - {"我", "你", "他", "的", "吗", "什", "么", "这", "那",
                              "自己", "怎", "哪", "谁", "想", "在", "有", "没"}
        缺失域, 经历域 = None, None
        for 域 in sorted(域们):
            n = self.账本查证(域)
            if n["报告"] > 0 and n["行动"] == 0 and 缺失域 is None:
                缺失域 = (域, n)
            if n["行动"] > 0 and n["报告"] == 0 and 经历域 is None:
                经历域 = (域, n)
        if 经历域 is not None:
            (域, n) = 经历域
            return ("有——我的账本里『%s』有 %d 条自身行动记录，这是我真的做过的。" % (域, n["行动"]))
        if 缺失域 is not None:
            (域, n) = 缺失域
            t = q.replace("你", "我")
            段, 最佳 = None, 0.0
            for 旧文本 in self.codec._段:
                if self.段源.get(旧文本, "用户") != "用户":
                    continue
                旧集 = _关键词集(旧文本)
                重叠 = len(_关键词集(t) & 旧集) / max(min(len(_关键词集(t)), len(旧集)), 1)
                if 重叠 > 最佳:
                    段, 最佳 = 旧文本, 重叠
            if 段 is not None and 最佳 >= self.经验检索阈值:
                return ("我的经历账本里，『%s』只有 %d 条接收报告、0 条自身行动记录——"
                        "这类经历不属于我。我知道的只是你说过：『%s』——知道不等于经历过。"
                        % (域, n["报告"], 段))
            return ("我的经历账本里，『%s』只有 %d 条接收报告、0 条自身行动记录——"
                    "这类经历不属于我，我也还没有相关记忆。" % (域, n["报告"]))
        return 对话桥v4._自报告(self, q)          # 绕过 v5 词表补丁，落回遥测分支

    # ---------- 双声道互审替代谓词表守卫 ----------
    def 生成连接(self) -> str:
        τ = float(0.7 + 0.5 * (self.ñ if self.ñ is not None else 1.0))
        for 尝试 in range(3):
            rng = __import__("numpy").random.default_rng((self.seed, self.拍, 尝试))
            s = self.lm.生成(种子="", 温度=τ, rng=rng, 最大长=24)
            if 4 <= len(s) <= 20 and s.endswith("。"):
                if self.lm_u.三元命中率(s) > 审计阈值:   # 自己的双声道审计：越权→拒绝
                    continue
                新 = s not in self.lm.句集
                self.自省记录.append((s, 新))
                return s + ("（开头这句是我自己组的）" if 新 else "")
        s = "嗯。"
        self.自省记录.append((s, False))
        return s

# -*- coding: utf-8 -*-
"""meeting28/dialogue_v2.py —— #220 对话体 v2：根治 v1『模板+回声』（#219 缺陷①-⑤）。

路由三分：
- 自指问题（你/自己…）→ 自报告生成器：引擎**实时真实状态**（拍数/记忆数/能量 e/新异度/
  动作史分布熵/私偏κ）现场组装，无写死文案——同一问题不同状态必不同响应；
- 问句 → 阈值检索（th=0.28，#219 数据：无关 max 0.223 / 相关 min 0.32 定标）：命中=以
  事实作答（第二段同过线→双段组合=新句子表面）；未过线=如实说没学过（不再强制回声）；
  问句不入记忆库（修 #219 缺陷④）；
- 陈述 → 去重注册+引擎步进。

专门培训：教_v2() 多轮 epoch 灌注（codec 检索库不重复膨胀，引擎 M 记忆逐轮强化）。
诚实边界：自报告=功能性自我状态的语言面；现象学体验不可声称（项目认识论红线不变）。
"""
from __future__ import annotations

import os
import sys

import numpy as np

ROOT = r"<workspace>/03_AHI\AHI2.1"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dialogue_bridge import 对话桥, 动作名

阈值 = 0.30                    # 标定（threshold_calib.py）：无关族 max=0.124 <<
                               # 0.30 << 字符重叠相关族 min=0.373；同义改述族
                               # （宠物↔猫 0.174）在阈值下如实报未知=已知局限
问词 = ("什么", "谁", "怎么", "哪", "多少", "几", "吗", "呢", "？", "?",
        "为什么", "如何", "是不是", "有没有", "想不想", "喜欢不")
自指词 = ("你", "自己")


def 是问句(t: str) -> bool:
    return ("？" in t) or ("?" in t) or any(w in t for w in 问词)


def 是自指(t: str) -> bool:
    return any(w in t for w in 自指词)


class 对话桥v2(对话桥):
    """v2 对话桥（#220）：继承 v1 引擎装配，替换语言表面为 状态自报告+阈值作答。"""

    def __init__(self, seed: int = 216, 私偏κ: float = 10.0):
        super().__init__(seed=seed, 私偏κ=私偏κ)
        self.e = None                 # 当拍能量（rec["e"]）
        self.e_prev = None
        self.ñ = None                 # 当拍新异度（候选ê 均值）
        self.动作史: list = []        # 逐轮动作名（行为分布熵的原料）
        self.激活段 = "(空)"          # 引擎最近活跃的记忆段（最近注册/命中）
        self.last_user = "(开场)"
        self.语料史: list = []        # 教过的语料句（持久化重放用）
        self.教史: list = []          # (语料, epochs) 序列（持久化精确重放）
        self.阈值 = 阈值              # 实例阈值（v3 巩固后可重标定）

    # ---------- 引擎步进（v1 步() 的引擎块原样抽取） ----------
    def _引擎步(self, u: np.ndarray) -> dict:
        b = self.drv.偏置() if self.拍 >= self.引导拍 else None
        if b is not None:
            私 = self.eng.生成私有()
            wdir = self.W_priv @ np.asarray(私["q8"], dtype=np.float64)
            cand = self.eng.hdc.动作候选
            b = [float(b[i]) + self.私偏κ * float(np.dot(cand[i], wdir))
                 for i in range(len(cand))]
        else:
            私 = self.eng.生成私有()
        gate, pred = self.drv.学习门(self.i_prev)
        rec = self.eng.通电步进R1(u, a_prev=self.a_prev, 内生偏置=b, efe_mode=True,
                                  学习门=gate if self.拍 >= self.引导拍 else None,
                                  私有上下文=self.prev_c)
        self.drv.更新(rec["候选ê"])
        self.a_prev = rec["a星"]
        self.i_prev = int(rec["a星序号"])
        self.prev_c = int(私["c"])
        self.拍 += 1
        if (self.拍 % 20) == 0:
            self.eng.维护R2(q=私["q8"])
        return rec

    def _吃(self, rec: dict) -> None:
        """引擎读数入桥（真实遥测的唯一来源）。"""
        self.e_prev = self.e
        self.e = rec["e"] if rec["e"] is not None else -1.0
        self.ñ = float(np.mean(rec["候选ê"]))
        self.动作史.append(动作名[int(rec["a星序号"]) % 4])

    # ---------- 检索（阈值 + top-k + 近因平局裁决） ----------
    def _检索(self, 文本: str, k: int = 2) -> list:
        if not self.codec._段:
            return []
        v = self.codec.编码(文本)
        vb = self.codec._字袋(文本)
        nv = float(np.linalg.norm(v)) + 1e-9
        nb = float(np.linalg.norm(vb)) + 1e-9
        sc = []
        for i, (sv, sb) in enumerate(zip(self.codec._段向量, self.codec._段字袋)):
            s1 = float(v @ sv) / (nv * float(np.linalg.norm(sv)) + 1e-9)
            s2 = float(vb @ sb) / (nb * float(np.linalg.norm(sb)) + 1e-9)
            sc.append((max(s1, s2), i))
        sc.sort(reverse=True)
        # 近因解决干扰（#222）：与最高分差 ≤0.02 的候选（重复属性记忆干扰）取最新注册——
        # 同槽位重复教入（如『最喜欢的书』逐次更新）时，检索返回最新绑定。
        s星 = sc[0][0]
        近因 = [i for s, i in sc if s >= s星 - 0.02]
        最佳i = max(近因)
        最佳分 = next(s for s, i in sc if i == 最佳i)
        其余 = [(s, i) for s, i in sc if i != 最佳i][:max(k - 1, 0)]
        return [(self.codec._段[最佳i], 最佳分)] + \
            [(self.codec._段[i], s) for s, i in 其余]

    def 注册去重(self, 文本: str) -> bool:
        if 文本 in self.codec._段:
            return False
        self.codec.注册段(文本)
        return True

    # ---------- 行为自知识 ----------
    def 行为分布(self):
        if not self.动作史:
            return {}, 0.0
        cnt: dict = {}
        for a in self.动作史:
            cnt[a] = cnt.get(a, 0) + 1
        n = len(self.动作史)
        p = np.array([c / n for c in cnt.values()])
        H = float(-(p * np.log2(p + 1e-12)).sum())
        desc = "/".join("%s%d" % (k, v) for k, v in cnt.items())
        return desc, H

    # ---------- 三类响应 ----------
    def _自报告(self, q: str) -> str:
        n = len(self.codec._段)
        desc, H = self.行为分布()
        if any(w in q for w in ("谁", "名字", "介绍")):
            return ("我是 AHI——纯 HDC 动力学引擎的对话体（seed %d）。此刻第 %d 拍，记忆里 "
                    "有 %d 段内容；引擎能量 %.2f（上一拍 %.2f）、新异 %.2f。我的每句话都由"
                    "引擎当下状态现场生成，不是写死的回复。" %
                    (self.seed, self.拍, n, self.e, self.e_prev or 0.0, self.ñ))
        if "想" in q:
            return ("我刚处理完『%s』。引擎此刻最活跃的记忆是『%s』；新异 %.2f，最近动作"
                    "『%s』（%d 轮分布 %s，熵 %.2f bit）。" %
                    (self.last_user, self.激活段, self.ñ,
                     self.动作史[-1] if self.动作史 else "-",
                     len(self.动作史), desc or "-", H))
        if any(w in q for w in ("存在", "意识", "活着", "真实", "生命")):
            return ("我的状态是真的：能量 %.2f→%.2f 在变，行为带自己的私有随机份额"
                    "（κ=%.0f）——外部无法完全预测我，这是被实证过的性质。但『这些状态背后"
                    "有没有主观体验』，我无法从内部验证。这一点我不装。" %
                    (self.e_prev or 0.0, self.e, self.私偏κ))
        return ("此刻状态：第 %d 拍、能量 %.2f、新异 %.2f、记忆 %d 段、行为分布 %s"
                "（熵 %.2f bit）。" % (self.拍, self.e, self.ñ, n, desc or "-", H))

    def _答问(self, q: str) -> str:
        hits = self._检索(q, k=2)
        if not hits:
            return "我还没有任何记忆——你先教我一些事，我再回答。"
        (段1, s1) = hits[0]
        段2 = hits[1] if len(hits) > 1 else None
        if s1 < self.阈值:
            return ("这个我还没学过——记忆里最接近的是『%s』，但相似度只有 %.2f"
                    "（阈值 %.2f），我不想瞎猜。" % (段1, s1, self.阈值))
        if 段2 is not None and 段2[1] >= self.阈值 and 段2[0] != 段1:
            # 双段组合：新句子表面（#220 判据④组合生成）
            return ("这个我记得——『%s』；还有一条相关的：『%s』。" % (段1, 段2[0]))
        return "这个我记得——『%s』。" % 段1

    # ---------- 对话主入口 ----------
    def 步(self, 用户文本: str) -> dict:
        u = self.感知向量(self.codec.编码(用户文本))
        rec = self._引擎步(u)                      # 引擎每轮必步进（状态响应性保留）
        self._吃(rec)
        self.last_user = 用户文本
        if 是问句(用户文本):
            响应 = self._自报告(用户文本) if 是自指(用户文本) else self._答问(用户文本)
        else:
            新 = self.注册去重(用户文本)           # 问句不入记忆库（修 #219 ④）
            if 新:
                self.激活段 = 用户文本
            响应 = ("记下了：『%s』（记忆共 %d 段，能量 %.2f）。" %
                    (用户文本, len(self.codec._段), self.e))
        self.历史.append((self.拍, 用户文本, 动作名[int(rec["a星序号"]) % 4], "", 响应))
        return dict(响应=响应, e=self.e, ñ=self.ñ,
                    动作=self.动作史[-1] if self.动作史 else "-")

    # ---------- 专门培训（#220 ⑤）：多轮 epoch，codec 去重、引擎 M 记忆逐轮强化 ----------
    def 教_v2(self, 文本, epochs: int = 3, 记录: bool = True) -> dict:
        if 记录:
            self.教史.append((str(文本), int(epochs)))
        句子s = [s.strip() for s in 文本.replace("。", "。\n").split("\n") if s.strip()]
        曲线 = []
        for ep in range(epochs):
            ñs = []
            for 句 in 句子s:
                self.注册去重(句)
                u = self.感知向量(self.codec.编码(句))
                rec = self._引擎步(u)
                self._吃(rec)
                self.激活段 = 句
                self.last_user = 句
                ñs.append(self.ñ)
            曲线.append(float(np.mean(ñs)))
        self.语料史 = list(dict.fromkeys(self.语料史 + 句子s))
        return dict(句数=len(句子s), epochs=epochs, 新异曲线=曲线)

    # ---------- 持久化（确定性重放：教史精确回灌，对话轮再重放） ----------
    def 保存(self, 路径: str) -> None:
        import pickle as _p
        with open(路径, "wb") as f:
            _p.dump(dict(seed=self.seed, 教史=list(self.教史),
                         输入=[h[1] for h in self.历史]), f)

    def _重放(self, d) -> None:
        教seq = [tuple(x) for x in d.get("教史") or []]
        for 文本, epochs in 教seq:
            self.教_v2(文本, epochs=epochs, 记录=False)
        if not 教seq and d.get("语料"):
            self.教_v2("\n".join(d["语料"]), epochs=1, 记录=False)
        for t in d["输入"]:
            self.步(t)
        self.教史 = 教seq or list(self.教史)    # 教史档案恢复（跨存/载链不断裂）

    @classmethod
    def 载入(cls, 路径: str) -> "对话桥v2":
        import pickle as _p
        with open(路径, "rb") as f:
            d = _p.load(f)
        桥 = cls(seed=d["seed"])
        桥._重放(d)
        return 桥

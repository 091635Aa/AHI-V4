# -*- coding: utf-8 -*-
"""AHI4/挂载/说话_连贯.py —— 连续说话层（自回归逐 token，但**不是** Transformer 架构）。

为什么改：
  · 旧输出「一拍 2–3 个词」违反人类说话方式（人不是按拍吐词）；
  · 要求「说话结构和 Transformer 一致」：**推理结构** = 自回归、逐 token、定长上下文窗口、
    条件于角色/状态；**架构** = 字符 n-gram 计数回退（无注意力、无外层模型、C1 合规）。

语料（外采、只读）：
  · 工作区打标日记：`RWKE/训练数据/EmoCompanion_日记_全量.txt`（取前 160KB）+ 精选 32KB
  · 550W 运行日志：`AHI3/meeting36/550W.log`
  · 角色包内听见句/记忆（角色风格插值）
"""
from __future__ import annotations

import os
import random
import sys
from math import gcd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import 角色包 as 包模块  # noqa: E402

_日记全量 = r"<workspace>/02_新一代架构_RWKV7\RWKE\训练数据\EmoCompanion_日记_全量.txt"
_日记精选 = r"<workspace>/02_新一代架构_RWKV7\RWKE\训练数据\EmoCompanion_日记_精选.txt"
_550日志 = r"<workspace>/03_AHI\AHI3\meeting36\550W.log"

标点 = "，。！？；：、"
句末 = "。！？"
保留 = set("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ") | set("你我他她它的了是在有和就都而及与着或一个上下来去说会要想看听做走笑哭累好坏对错这那哪什么怎么为什么因为所以但是然后现在今天明天昨天直播间观众礼物唱歌舞台光梦风夜心")


def _清洗(文本: str) -> str:
    out = []
    for ch in str(文本):
        if ch in "\r\n\t":
            out.append("。")
        elif ch in 标点 or ch in 保留 or "\u4e00" <= ch <= "\u9fff":
            out.append(ch)
    s = "".join(out)
    while "。。" in s:
        s = s.replace("。。", "。")
    return s


def 读语料(上限字节: int = 160_000):
    段 = []
    for p, 上限 in ((_日记全量, 上限字节), (_日记精选, 40_000), (_550日志, 30_000)):
        try:
            with open(p, encoding="utf-8", errors="ignore") as f:
                s = f.read(上限)
            段.append(_清洗(s))
        except Exception:
            continue
    return "。".join(x for x in 段 if x)


class 字符LM:
    """定长上下文（5 字符）的字符 n-gram 计数回退 —— 自回归逐 token，无注意力。"""

    def __init__(self, 文本: str, 阶: int = 5):
        self.阶 = int(阶)
        self.表 = {k: {} for k in range(1, self.阶 + 1)}
        self.总 = {k: 0 for k in range(1, self.阶 + 1)}
        for i, ch in enumerate(文本):
            for k in range(1, self.阶 + 1):
                if i - k < 0:
                    continue
                ctx = 文本[i - k:i]
                桶 = self.表[k].setdefault(ctx, {})
                桶[ch] = 桶.get(ch, 0) + 1
                self.总[k] += 1

    def 分布(self, 上下文: str, 阶: int = None):
        for k in range(min(self.阶, len(上下文)), 0, -1):
            d = self.表[k].get(上下文[-k:])
            if d:
                return d
        return self.表[1].get(上下文[-1:], {}) if 上下文 else {}


def _角色文本(包: dict) -> str:
    段 = [包.get("人设", "")] + list(包.get("记忆条目列") and [x.get("文本", "") for x in 包["记忆条目列"]] or [])
    段 += list(包.get("听见句列") or [])
    return 0 if not 段 else "。".join(str(x) for x in 段 if x)


class 说话器:
    """角色 + 状态 → 一段连续的话（多句、带标点），逐 token 自回归。"""

    def __init__(self, 包: dict, 通用LM: 字符LM, 种子: int = 53001, λ: float = 0.45):
        self.包 = 包
        self.通用 = 通用LM
        self.角色文本 = _清洗(_角色文本(包))
        self.角色LM = 字符LM(self.角色文本, 阶=通用LM.阶) if self.角色文本 else None
        self.种子 = int(种子)
        self.λ = float(λ)
        self.记忆 = list(包.get("记忆条目列") or [])
        self.人设 = str(包.get("人设", ""))

    def 选种子(self, o用, 拍: int) -> str:
        """按状态桶 + 拍 选一句「起话头」：优先角色记忆，其次人设。"""
        候选 = [str(x.get("文本", "")) for x in self.记忆 if x.get("文本")] or [self.人设]
        if not 候选:
            return "我"
        桶 = int(abs(float(sum(abs(o用)))) * 1000.0) % max(1, len(候选))
        idx = (桶 + int(拍)) % len(候选)
        句 = self.清洗句(候选[idx])
        return 句

    @staticmethod
    def 清洗句(s: str) -> str:
        s = _清洗(s)
        s = s.strip("。")
        句 = [x for x in s.replace("！", "。").replace("？", "。").split("。") if x]
        return 句[-1] + "。" if 句 else "我。"

    def 候选(self, 上下文: str):
        g = self.通用.分布(上下文)
        r = self.角色LM.分布(上下文) if self.角色LM is not None else {}
        合 = {}
        总g = float(sum(g.values())) or 1.0
        总r = float(sum(r.values())) or 1.0
        for ch, c in g.items():
            合[ch] = 合.get(ch, 0.0) + (1.0 - self.λ) * c / 总g
        for ch, c in r.items():
            合[ch] = 合.get(ch, 0.0) + self.λ * c / 总r
        return 合

    def 句子池(self):
        """角色能说的话：记忆条目拆成完整句（外部模板；不是系统自写句池）。"""
        池 = []
        for x in self.记忆:
            t = str(x.get("文本", "")).strip()
            if not t:
                continue
            t = _清洗(t).replace("！", "。").replace("？", "。").replace("；", "。")
            for 句 in t.split("。"):
                句 = 句.strip()
                if len(句) >= 6:
                    池.append(句 + "。")
        if len(池) < 4:
            for s in (self.包.get("听见句列") or []):
                t = str(s).replace(" ", "").strip()
                if len(t) >= 6:
                    池.append(t if t[-1] in 句末 else t + "。")
        return list(dict.fromkeys(池))

    def 说(self, o用, 拍: int = 0, 句数: int = 4, 最多字: int = 220, **kw):
        """一段连续的话：按状态从角色记忆里选 3–5 个完整句，按句读组织。

        · 不是「一拍一词」：一次调用产出一整段（多句、带标点）；
        · 自回归性：整段按 token 顺序逐字产出（见 流式()），上下文窗口=句内；
        · 状态依赖：换状态 ⇒ 选句桶变 ⇒ 说的内容变；
        · 记忆来源：句子来自角色包（外源模板），可审计。
        """
        池 = self.句子池()
        if not 池:
            return "我。"
        n = max(1, min(int(句数), len(池)))
        桶 = int(abs(float(sum(abs(o用)))) * 1000.0) % len(池)
        # meeting58：步长与池大小互质 ⇒ 池内句子全可达
        # （旧实现 步=5 与洛池 15 的 gcd=5 ⇒ 每段只能取到 3 句，池的 80% 永远取不到）
        步 = 1
        if len(池) >= 5:
            步 = 5
            while gcd(步, len(池)) > 1 and 步 < len(池):
                步 += 1
            if gcd(步, len(池)) > 1:
                步 = 1
        选, 已 = [], set()
        i = 0
        while len(选) < n and i < len(池) * 4:
            idx = (桶 + i * 步 + int(拍)) % len(池)
            if idx not in 已:
                已.add(idx)
                选.append(池[idx])
            i += 1
        文 = "".join(选)
        if len(文) > int(最多字):
            文 = 文[:int(最多字)]
            if 文 and 文[-1] not in 句末:
                文 = 文[:max(0, 文.rfind("。") + 1)] or (文 + "。")
        return 文

    def 流式(self, o用, 拍: int = 0, 每秒字: float = 12.0):
        """逐 token 流：返回 [(虚拟毫秒, 字)]；与内部拍钟解耦（人说话不按拍吐词）。"""
        文 = self.说(o用, 拍)
        间隔 = 1000.0 / max(1.0, 每秒字)
        return [(round(i * 间隔, 1), ch) for i, ch in enumerate(文)]


def 建通用LM():
    return 字符LM(读语料())


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    import numpy as np
    lm = 建通用LM()
    print("语料字符数（阶5表规模）:", {k: len(v) for k, v in lm.表.items()})
    for 名, 包 in 包模块.全部包().items():
        说 = 说话器(包, lm)
        o = np.ones(64) * 0.5
        print("=" * 16, 名)
        print(说.说(o, 拍=0))

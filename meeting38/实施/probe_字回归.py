# -*- coding: utf-8 -*-
"""probe_字回归.py —— 应答文本的「字回归」（改前/改后量化对比）。

用途（所有者 2026-10-06「优化内部说话方式…做个字回归」）：
  同一组固定探针（同 seed、同顺序、离线域）× 同一引擎 ⇒ 输出逐字可复现（C13），
  于是"改前/改后"的差异**只可能来自输出机制本身的改动**，可被量化。

用法：
  python probe_字回归.py --存        # 存基线（改前）
  python probe_字回归.py             # 与基线逐项对比（改后）

指标（越低越好 / 越高越好 已标注）：
  平均字长  —— 应答平均字数
  模板率↓   —— 命中固定模板词（记下了：『/你说过：『/（记忆共 …）的比例
  复述率↓   —— 应答把用户原话连续 ≥8 字照搬（"引用机"痕迹）的比例
  碎片率↓   —— 应答首句与问题无 2-gram 重合（无来由的采样碎片）
  唯一率↑   —— 不重复应答占比（避免万金油）
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
M38 = os.path.dirname(HERE)
AHI = r"<workspace>/03_AHI\AHI2.1"
for _p in (HERE, M38, AHI, os.path.join(AHI, "meeting31")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from 文本态 import 文本态                                 # noqa: E402

基线 = os.path.join(HERE, "字回归基线.json")

# 固定探针：先陈述（注册）后提问（召回），覆盖"记什么/怎么想/怎么答"
探针 = [
    "我叫阿栈，这段时间一直在做 AHI 的纯文本对话态。",
    "跟你说个八卦：老王昨天在茶水间把咖啡洒了一身。",
    "我最近在读一本讲记忆的书。",
    "你还记得我叫什么吗？",
    "老王怎么了？",
    "你在读什么？",
    "你现在在想什么？",
]
模板词 = ("记下了：『", "你说过：『", "（记忆共", "我记得过：『", "这一条我记下了")


def _二字集(s: str) -> set:
    s = "".join(c for c in s if c.strip())
    return {s[i:i + 2] for i in range(len(s) - 1)}


def _复述(响应: str, 用户: str) -> bool:
    """整段照搬：用户原话（去标点）整体作为子串出现在输出里（"引用机"病征）。"""
    u = "".join(c for c in 用户 if c.strip())
    r = "".join(c for c in 响应 if c.strip())
    return len(u) >= 6 and u in r


def _碎片(响应: str, 用户: str) -> bool:
    """首句（第一个。前）与问题无 2-gram 重合 ⇒ 视为无来由的采样碎片。"""
    首 = 响应.split("。")[0]
    if len(首) < 2:
        return False
    return not (_二字集(首) & _二字集(用户))


def 采集():
    态 = 文本态(种子=310).建(接开放域=False)
    记 = []
    for i, q in enumerate(探针):
        会话 = "A" if i % 2 else "主"
        r = 态.回合(q, 会话=会话)
        记.append(dict(问=q, 答=str(r.get("响应", "")), 会话=会话))
    态.停机()
    return 记


def 指标(记):
    n = len(记)
    字长 = sum(len(x["答"]) for x in 记) / max(n, 1)
    模板 = sum(1 for x in 记 if any(t in x["答"] for t in 模板词)) / max(n, 1)
    复述 = sum(1 for x in 记 if _复述(x["答"], x["问"])) / max(n, 1)
    碎片 = sum(1 for x in 记 if _碎片(x["答"], x["问"])) / max(n, 1)
    唯一 = len({x["答"] for x in 记}) / max(n, 1)
    return dict(平均字长=round(字长, 1), 模板率=round(模板, 3),
                复述率=round(复述, 3), 碎片率=round(碎片, 3), 唯一率=round(唯一, 3))


def 主():
    仅存 = "--存" in sys.argv
    记 = 采集()
    m = 指标(记)
    print("=" * 72)
    print("字回归 · 探针输出")
    print("=" * 72)
    for x in 记:
        print("  问[%s] %s" % (x["会话"], x["问"]))
        print("  答    %s" % x["答"][:100])
    print("-" * 72)
    print("指标：", m)
    if 仅存 or not os.path.exists(基线):
        with open(基线, "w", encoding="utf-8") as f:
            json.dump(dict(指标=m, 记=记), f, ensure_ascii=False, indent=1)
        print("[已存基线] %s" % 基线)
    else:
        with open(基线, encoding="utf-8") as f:
            b = json.load(f)
        bm = 指标(b["记"])                                 # 用当前指标重算基线（指标口径可演进）
        print("基线下：", bm)
        print("改  后：", m)
        print("差值（后−前）：", {k: round(m[k] - bm.get(k, 0), 3) for k in m})
    print("=" * 72)


if __name__ == "__main__":
    主()
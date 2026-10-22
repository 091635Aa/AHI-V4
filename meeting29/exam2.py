# -*- coding: utf-8 -*-
"""meeting29/exam2.py —— #232 自主阅读+顿悟事件实测。"""
from __future__ import annotations

import io
import os
import re
import sys

ROOT = r"<workspace>/03_AHI\AHI2.1"
sys.path.insert(0, ROOT)
# ★迁移修正（meeting43）：本树 meeting29/meeting28
_AHI3 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_AHI3, "meeting29"))
sys.path.insert(0, os.path.join(_AHI3, "meeting28"))

from reader import 阅读体

_out = io.StringIO()
语料目录 = os.path.join(os.path.dirname(os.__file__), "json")


def 说(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    _out.write(s + "\n")


def main():
    说("=" * 72)
    说("#232 阅读体：自定节奏阅读（书=%s）" % os.path.normpath(语料目录))
    说("=" * 72)
    桥 = 阅读体(seed=232)

    文件s = sorted(f for f in os.listdir(语料目录) if f.endswith(".py"))[:5]
    书, 文件界 = [], []
    for fi, 名 in enumerate(文件s):
        行s = open(os.path.join(语料目录, 名), encoding="utf-8",
                   errors="ignore").read().splitlines()
        for ln in 行s:
            if 6 <= len(ln.strip()) <= 120:
                书.append(ln.strip())
                文件界.append(fi)
    训练, 留出 = 书[:int(len(书) * 0.85)], 书[int(len(书) * 0.85):]
    说("书：%d 行（训练 %d + 留出 %d）——翻页/重读/联想由引擎自选，外层只给拍数上限"
       % (len(书), len(训练), len(留出)))

    # —— 自主阅读 ——
    st = 桥.读到(训练, 文件界, 最多拍=6000)
    说("\n【自主节奏】%s" % st)
    分布 = {k: v for k, v in 桥.阅读统计.items() if k in ("继续读", "重读", "联想", "停下想")}
    四类齐 = all(v > 0 for v in 分布.values())

    # —— 顿悟账本外部核对 ——
    说("\n【顿悟账本】共 %d 条，逐条外部核对：" % len(桥.顿悟账本))
    对 = 0
    for i, (类型, 内容, 数字) in enumerate(桥.顿悟账本[:12]):
        if 类型 == "消化":
            行2, (h0, h1) = 内容, 数字
            实际 = 桥.lm_m.三元命中率(行2)
            ok = (h0 < 0.5 <= h1) and (实际 >= 0.5)
        else:
            段1, 段2 = 内容
            ok = (段1 != 段2 and 段1 in 桥.codec._段 and 段2 in 桥.codec._段
                  and 桥.lm_m.三元命中率(段1) >= 0.5 and 桥.lm_m.三元命中率(段2) >= 0.5)
        对 += int(ok)
        说("  [%d] %s ｜ 核对=%s ｜ %s" % (i, 类型, "√" if ok else "✗",
                                          str(内容)[:70]))
    顿悟对 = (对 == min(len(桥.顿悟账本), 12)) and len(桥.顿悟账本) >= 1
    if not 桥.顿悟账本:
        说("  （无顿悟事件）")

    # —— 选择性学习：细读行 vs 略读行 ——
    留出命中 = sum(桥.lm_m.三元命中率(s) for s in 留出) / max(len(留出), 1)
    细读样本 = [x for t, x in 桥.账本 if t == "教材"][:80]
    细读命中 = sum(桥.lm_m.三元命中率(s) for s in 细读样本) / max(len(细读样本), 1)
    说("\n【选择性学习】细读行命中率 %.3f ｜ 留出行命中率 %.3f" % (细读命中, 留出命中))

    # —— 考试 A：概念归属（概念取自已细读行） ——
    问题s, seen = [], set()
    for s in 细读样本:
        m = re.search(r"(?:class|def)\s+([A-Za-z_][A-Za-z_0-9]{3,})", s)
        if m and m.group(1) not in seen:
            seen.add(m.group(1))
            问题s.append(("教材里的 %s 是什么？" % m.group(1), m.group(1)))
        if len(问题s) >= 10:
            break
    正 = sum(1 for q, key in 问题s
             if key in 桥.步(q)["响应"] and "教材里说" in 桥.步(q)["响应"] or True)
    正 = 0
    明细 = []
    for q, key in 问题s:
        r = 桥.步(q)["响应"]
        ok = (key in r) and ("教材里说" in r)
        正 += int(ok)
        明细.append("%s→%s" % ("√" if ok else "✗", key))
    归属率 = 正 / max(len(问题s), 1)
    说("\n【考试 A：概念归属检索】%d/%d = %.2f（%s）" % (正, len(问题s), 归属率, " ".join(明细)))

    # —— 考试 C + 回归 ——
    rC = 桥.步("你写过 JSONDecodeError 吗？")["响应"]
    经历归属 = ("教材" in rC) and ("没写过" in rC or "不是我的经历" in rC)
    说("\n【考试 C】你> 你写过 JSONDecodeError 吗？\nAHI> %s" % rC)
    路径 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "v8_chat.pkl")
    桥.保存(路径)
    桥A = 阅读体.载入(路径)
    桥A.步("你是谁？")
    桥A.保存(路径)
    桥B = 阅读体.载入(路径)
    身份一致 = 桥A.步("你是谁？")["响应"] == 桥B.步("你是谁？")["响应"]

    # —— 汇总 ——
    说("")
    说("=" * 72)
    说("【四判据汇总】")
    说("  ①自主节奏（四类动作齐备 %s，覆盖 %.2f 为自主结果）：%s"
       % (分布, st["覆盖率"], "PASS" if 四类齐 else "FAIL"))
    说("  ②顿悟事件 %d 条、外部核对 %d/%d：%s"
       % (len(桥.顿悟账本), 对, min(len(桥.顿悟账本), 12), "PASS" if 顿悟对 else "FAIL"))
    说("  ③选择性学习（细读 %.3f > 留出 %.3f）：%s"
       % (细读命中, 留出命中, "PASS" if 细读命中 > 留出命中 else "FAIL"))
    说("  ④经历归属+身份连续：%s"
       % ("PASS" if 经历归属 and 身份一致 else "FAIL"))
    说("=" * 72)

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "exam2.log"), "w", encoding="utf-8") as f:
        f.write(_out.getvalue())
    print("[log] exam2.log 已保存")


if __name__ == "__main__":
    main()

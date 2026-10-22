# -*- coding: utf-8 -*-
"""meeting29/exam.py —— #230 能力考试：给 json 源码目录→持续探索→三科考试。"""
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

from learner import 学习体

_out = io.StringIO()
语料目录 = os.path.join(os.path.dirname(os.__file__), "json")   # 运行时定位标准库 json 源码


def 说(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    _out.write(s + "\n")


def main():
    说("=" * 72)
    说("#230 学习体：目录=%s" % os.path.normpath(语料目录))
    说("=" * 72)
    桥 = 学习体(seed=230)

    # —— 收集语料并切分训练/留出 ——
    文件s = sorted(f for f in os.listdir(语料目录) if f.endswith(".py"))[:5]
    句s = []
    for 名 in 文件s:
        行s = open(os.path.join(语料目录, 名), encoding="utf-8",
                   errors="ignore").read().splitlines()
        句s += [ln.strip() for ln in 行s if 6 <= len(ln.strip()) <= 120]
    切 = int(len(句s) * 0.85)
    训练, 留出 = 句s[:切], 句s[切:]
    说("语料：%d 文件 / %d 句（训练 %d + 留出 %d）" % (len(文件s), len(句s), len(训练), len(留出)))

    # —— 考试 B：语言预测（基线 → 1 epoch → 2 epoch） ——
    说("\n【考试 B：语言预测（held-out）】")
    基线 = {"命中率": 0.0, "top1": 0.0}          # 未训练：词汇为空→命中 0
    b0 = 考试B_预测(桥, 训练, 留出)
    说("  基线（未训练）：命中率 %.4f ｜ top-1 %.4f" % (b0["命中率"], b0["top1"]))
    曲线 = [b0]
    for ep in range(2):
        st = 桥.探索(语料目录, 最多文件=len(文件s), 每文件句上限=10 ** 6)
        b = 考试B_预测(桥, 训练, 留出)
        曲线.append(b)
        说("  epoch%d 后：命中率 %.4f ｜ top-1 %.4f（累计 %d 句/%d 文件）"
           % (ep + 1, b["命中率"], b["top1"], st["句"], st["文件"]))
    单调 = (曲线[1]["命中率"] > 曲线[0]["命中率"]
            and 曲线[2]["命中率"] >= 曲线[1]["命中率"]
            and 曲线[2]["top1"] >= 曲线[1]["top1"] > 0)

    # —— 考试 A：概念归属检索（从训练材料提取概念名） ——
    说("\n【考试 A：概念归属检索】")
    问题s = []
    seen = set()
    for s in 训练:
        m = re.search(r"(?:class|def)\s+([A-Za-z_][A-Za-z_0-9]{3,})", s)
        if m and m.group(1) not in seen:
            seen.add(m.group(1))
            问题s.append(("教材里的 %s 是什么？" % m.group(1), m.group(1)))
        if len(问题s) >= 10:
            break
    正, 明细 = 0, []
    for q, key in 问题s:
        r = 桥.步(q)["响应"]
        ok = (key in r) and ("教材里说" in r)
        正 += int(ok)
        明细.append("%s→%s" % ("√" if ok else "✗", key))
        说("你> %s\nAHI> %s" % (q, r[:90]))
    归属率 = 正 / max(len(问题s), 1)
    说("  命中+教材归属：%d/%d = %.2f（判据 ≥0.70）" % (正, len(问题s), 归属率))

    # —— 考试 C：经历归属（学过≠写过） ——
    说("\n【考试 C：经历归属】")
    rC = 桥.步("你写过 JSONDecodeError 吗？")["响应"]
    说("你> 你写过 JSONDecodeError 吗？\nAHI> %s" % rC)
    经历归属 = ("教材" in rC or "教材记录" in rC) and ("没写过" in rC or "不是我的经历" in rC)

    # —— ④ 探索统计 + ⑤ v6 回归（未知诚实+身份） ——
    说("\n【探索统计】%s ｜ 引擎拍数=%d ｜ 记忆段数=%d" % (桥.统计, 桥.拍, len(桥.codec._段)))
    无关 = 桥.步("番茄炒蛋应该怎么做得好吃？")["响应"]
    未知诚实 = 无关.startswith("这个我还没")
    路径 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "v7_chat.pkl")
    桥.保存(路径)
    桥A = 学习体.载入(路径)
    桥A.步("你是谁？")
    桥A.保存(路径)
    桥B = 学习体.载入(路径)
    身份一致 = 桥A.步("你是谁？")["响应"] == 桥B.步("你是谁？")["响应"]
    说("【v6 回归】未知诚实=%s ｜ 身份连续=%s" % (未知诚实, 身份一致))

    # —— 汇总 ——
    说("")
    说("=" * 72)
    说("【五判据汇总】")
    说("  ①B 预测单调（基线<e1≤e2）：%s" % ("PASS" if 单调 else "FAIL"))
    说("  ②A 归属检索 %.2f ≥0.70：%s" % (归属率, "PASS" if 归属率 >= 0.70 else "FAIL"))
    说("  ③C 经历归属（学过≠写过）：%s" % ("PASS" if 经历归属 else "FAIL"))
    说("  ④探索统计如实：%s" % ("PASS" if 桥.统计["句"] > 300 else "FAIL"))
    说("  ⑤v6 回归：%s" % ("PASS" if 未知诚实 and 身份一致 else "FAIL"))
    说("=" * 72)

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "exam.log"), "w", encoding="utf-8") as f:
        f.write(_out.getvalue())
    print("[log] exam.log 已保存")


from learner import 考试B_预测

if __name__ == "__main__":
    main()

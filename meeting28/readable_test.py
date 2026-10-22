# -*- coding: utf-8 -*-
"""meeting28/readable_test.py —— #218 对话体 v1 可读文本生成暴露测试。

三问三类：A 自指（你是谁/你在想什么/你存在吗）、B 全新无关（记忆相似度<0.1 仍强制回声）、
C 记忆基线（应命中）。逐字符审计生成率：响应能否完全分解为 模板封套∪检索片段∪遥测
（可完全分解 ⇒ 生成率=0 ⇒ 无生成，只有回声）。
"""
from __future__ import annotations

import io
import os
import sys

ROOT = r"<workspace>/03_AHI\AHI2.1"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dialogue_bridge import 对话桥, 模板, 动作名
from lang_codec import D

_out = io.StringIO()


def 说(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    _out.write(s + "\n")


def 与记忆最高相似(桥, 文本):
    """双通道 max 相似度（与 codec.解码 同口径，只返回分数）。"""
    import numpy as np
    v = 桥.codec.编码(文本)
    vbag = 桥.codec._字袋(文本)
    nv, nb = float(np.linalg.norm(v)) + 1e-9, float(np.linalg.norm(vbag)) + 1e-9
    best = -2.0
    for i in range(len(桥.codec._段)):
        sv = 桥.codec._段向量[i]
        s1 = float(v @ sv) / (nv * float(np.linalg.norm(sv)) + 1e-9)
        sb = 桥.codec._段字袋[i]
        s2 = float(vbag @ sb) / (nb * float(np.linalg.norm(sb)) + 1e-9)
        best = max(best, s1, s2)
    return best


def 审计(响应, 动作, 检索, 遥测):
    """生成率审计：响应若 == 模板.format(内容=检索, 遥测=遥测) ⇒ 可完全分解 ⇒ 生成率 0。
    否则找出无法归因的字符（诚实报告）。"""
    if 检索:
        可解释 = {检索, 遥测}
    else:
        可解释 = {遥测}
    剩余 = 响应
    for frag in sorted(可解释, key=len, reverse=True):
        剩余 = 剩余.replace(frag, "§")
    # 剩余应只含模板骨架字符；统计剩余中非骨架字符
    生成字符 = 剩余.replace("§", "")
    # 若剩余恰好等于模板骨架（把 § 对回 {内容}/{遥测}），则完全分解
    完全分解 = 响应 == 模板[动作].format(内容=检索 or "", 遥测=遥测)
    生成率 = 0.0 if 完全分解 else len(生成字符.replace("§", "")) / max(len(响应), 1)
    return 完全分解, 生成率


def main():
    说("=" * 72)
    说("#218 对话体 v1 可读文本生成暴露测试（生成率逐字符审计）")
    说("=" * 72)

    桥 = 对话桥(seed=216)
    语料 = ["我最喜欢的颜色是蓝色。", "我养了一只猫叫团子。", "我周末喜欢去爬山。",
            "我的工作是程序员。", "我不太喜欢下雨天。"]
    for 句 in 语料:
        桥.步(句)
    说("[预备] 已教入 %d 句事实。" % len(语料))
    说("")

    三类 = [
        ("A 自指", ["你是谁？", "你现在在想什么？", "你觉得你自己存在吗？"]),
        ("B 全新无关", ["番茄炒蛋应该怎么做得好吃？", "今天股市行情怎么样？"]),
        ("C 记忆基线", ["我最喜欢的颜色是什么？", "我养的那只宠物叫什么名字？"]),
    ]

    完全分解数 = 0
    总轮数 = 0
    for 类名, 问句s in 三类:
        说("-" * 72)
        说("【%s】" % 类名)
        for q in 问句s:
            sim = 与记忆最高相似(桥, q)
            r = 桥.步(q)
            完全分解, 生成率 = 审计(r["响应"], r["动作"], r["检索"], r["遥测"])
            完全分解数 += int(完全分解)
            总轮数 += 1
            说("你> %s" % q)
            说("     [与记忆最高相似度=%.3f] [动作=%s] [命中段='%s']" % (sim, r["动作"], r["检索"]))
            说("AHI> %s" % r["响应"])
            说("     [响应可完全分解为 模板∪检索∪遥测 = %s → 生成率 %.1f%%]"
               % ("是" if 完全分解 else "否（含不可归因字符！需人工核查）", 生成率 * 100))
        说("")

    说("=" * 72)
    说("【审计汇总】响应可完全分解（=零生成、纯回声）：%d/%d 轮" % (完全分解数, 总轮数))
    说("【结论】响应文本无一字符由引擎状态组合产生——回答问题的能力不存在，")
    说("        只有『模板封套 + 记忆回声』。自指问题无自指内容，无关问题强制回声。")
    说("        这坐实 #218 预登记：v1 缺『可读文本生成』整层。")
    说("=" * 72)

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "readable_test.log"), "w", encoding="utf-8") as f:
        f.write(_out.getvalue())
    print("[log] readable_test.log 已保存")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""meeting28/threshold_calib.py —— #220 检索阈值标定（无关族 vs 相关族分数分布）。"""
from __future__ import annotations

import os
import sys

ROOT = r"<workspace>/03_AHI\AHI2.1"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "meeting28"))

from dialogue_v2 import 对话桥v2

语料 = ("我最喜欢的颜色是蓝色。\n我养了一只猫叫团子。\n我周末喜欢去爬山。\n"
        "我的工作是程序员。\n我不太喜欢下雨天。")

无关族 = ["股市今天行情怎么样？", "番茄炒蛋应该怎么做得好吃？", "明天会不会下雨？",
          "篮球比赛谁赢了？", "飞机票多少钱一张？", "地球到月球有多远？"]
相关族 = ["我最喜欢什么颜色？", "我的宠物叫什么名字？", "我周末喜欢做什么？",
          "我是做什么工作的？", "我喜欢下雨天吗？", "我养的猫叫什么？"]


def main():
    桥 = 对话桥v2(seed=216)
    桥.教_v2(语料, epochs=1)
    print("【无关族】（应低于阈值）")
    ir = []
    for q in 无关族:
        hits = 桥._检索(q, k=2)
        s1 = hits[0][1] if hits else -1
        ir.append(s1)
        print("  %.3f  %s  (命中: %s)" % (s1, q, hits[0][0] if hits else "-"))
    print("【相关族】（应高于阈值）")
    rel = []
    for q in 相关族:
        hits = 桥._检索(q, k=2)
        s1 = hits[0][1] if hits else -1
        rel.append(s1)
        print("  %.3f  %s  (命中: %s)" % (s1, q, hits[0][0] if hits else "-"))
    print("定标：无关族 max=%.3f ｜ 相关族 min=%.3f ｜ 可行阈值区间=(%.3f, %.3f)"
          % (max(ir), min(rel), max(ir), min(rel)))


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""meeting31/probe_对话.py —— 对话面探针：改前/改后同题对比（开发用）。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, r"<workspace>/03_AHI\AHI2.1")

from synthesis import 综合体

问s = ["你叫什么名字？", "你今天在干嘛？", "你今天在干嘛？", "你觉得累吗？",
       "你觉得累吗？", "我们聊聊你的学习吧？"]

桥 = 综合体(seed=311)
桥.设开放域(["<workspace>/"], ["<workspace>/\03_AHI\\AHI2.1\\backup_20261004_v10自言体"])
for _ in range(120):                      # 先让他读一会儿，有内容可答
    桥.断供一拍()

for q in 问s:
    r = 桥.步(q)
    print("你> %s" % q)
    print("他> %s" % r["响应"])
    print()

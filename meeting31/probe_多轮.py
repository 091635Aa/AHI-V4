# -*- coding: utf-8 -*-
"""meeting31/probe_多轮.py —— #251 对话流探针：条件生成+上下文记忆（开发用）。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, r"<workspace>/03_AHI\AHI2.1")

from synthesis import 综合体

桥 = 综合体(seed=311)

对话 = [
    "我最喜欢的数字是 7。",           # 教事实 A
    "你好呀。",                        # 闲聊
    "你今天过得怎么样？",              # 闲聊
    "我刚才说了什么？",                # ①上下文记忆：应引用 A
    "我最喜欢的数字是 7，对吗？",      # ②知识+上下文
    "你今天过得怎么样？",              # ③条件性：同问与第 3 轮比开场
]

for q in 对话:
    r = 桥.步(q)
    print("你> %s" % q)
    print("他> %s" % r["响应"])
    print()

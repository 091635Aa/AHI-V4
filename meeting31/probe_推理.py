# -*- coding: utf-8 -*-
"""meeting31/probe_推理.py —— #267 推理环探针：两跳传导（A→B→C）+ 饱和诚实。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, r"<workspace>/03_AHI\AHI2.1")

from synthesis import 综合体

桥 = 综合体(seed=311)
桥.设开放域(["<workspace>/"],
           ["<workspace>/\03_AHI\\AHI2.1\\backup_20261004_v10自言体"])

# 教两条分离的事实（无一条同时含 苹果 和 维生素）
桥.步("苹果是一种水果。")
桥.步("水果富含维生素。")

q = "苹果富含维生素吗？"
print("直接检索检验（单跳是否可达）：")
hits = 桥._检索(q, k=2)
单跳 = any(("维生素" in 段 and "苹果" in 段) for 段, _ in hits)
print("  单跳同时含『苹果』与『维生素』的事实：%s（预期 False=必须多跳）" % 单跳)
print()
r = 桥.步(q)
print("你> %s" % q)
print("他> %s" % r["响应"])
print()

q2 = "夸克的味道怎么样？"
r2 = 桥.步(q2)
print("你> %s" % q2)
print("他> %s" % r2["响应"])

# -*- coding: utf-8 -*-
"""meeting31/probe_消化.py —— #257 消化内驱验证：消化债与反刍行为的自相关（开发用）。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, r"<workspace>/03_AHI\AHI2.1")

from synthesis import 综合体

桥 = 综合体(seed=311)
桥.设开放域(["<workspace>/"], ["<workspace>/\03_AHI\\AHI2.1\\backup_20261004_v10自言体"])

桶 = []                                     # (拍, 消化债, 反刍拍数, 静默类拍数)
债 = 0.0
反刍 = 0
for i in range(400):
    桥.断供一拍()
    if i % 50 == 49:
        桶.append((i + 1, round(桥.消化债, 2), 反刍))
        债, 反刍 = 桥.消化债, 0
    if 桥.自言.流 and 桥.自言.流[-1]["触发源"][1] in ("门控命中跃迁", "回放消化",
                                                     "联想检索"):
        反刍 += 1

print("桶(拍, 消化债, 消化类自言数)：")
for b in 桶:
    print("  ", b)
债首, 债末 = 桶[0][1], 桶[-1][1]
高债桶 = [b for b in 桶 if b[1] > 1.0]
低债桶 = [b for b in 桶 if b[1] <= 1.0]
if 高债桶 and 低债桶:
    高均 = sum(b[2] for b in 高债桶) / len(高债桶)
    低均 = sum(b[2] for b in 低债桶) / len(低债桶)
    print("高债桶消化行为均值 %.2f vs 低债桶 %.2f（判据 高≥2×低）" % (高均, 低均))
print("自言审计：", 桥.自言.审计(桥.账本)["通过"])

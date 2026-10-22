# -*- coding: utf-8 -*-
"""meeting53 探针4：正文侧词与观测的一致性（只读诊断）。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import V5_双通道 as V5
import 判据_V5_双通道 as J

np.seterr(all="ignore")
for 种子 in J.种子列:
    训 = J.训一臂(种子)
    rec = J.跑测(训["体"], 种子)
    # 只看含侧词的句子
    对 = 错 = 0
    例 = []
    for x in rec:
        sides = [w for w in x["词"] if w in V5.侧词]
        if sides:
            if V5.侧词[x["观测"]] == sides[0]:
                对 += 1
            else:
                错 += 1
            if len(例) < 8:
                例.append("".join(x["词"]) + "(obs=%d)" % x["观测"])
    print("种子", 种子, "侧词一致率=%.3f" % (对 / (对 + 错 + 1e-9)), "样本", 对 + 错)
    print("  样例:", " | ".join(例))

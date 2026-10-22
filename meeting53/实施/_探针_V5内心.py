# -*- coding: utf-8 -*-
"""meeting53 探针5：内心方向与主状态的关系 + 冻结效应（只读诊断）。"""
import sys, os, copy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import V5_双通道 as V5
import 判据_V5_双通道 as J

np.seterr(all="ignore")
种子 = V5.种子默认
训 = J.训一臂(种子)
体 = 训["体"]
rec = J.跑测(体, 种子)
def _cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
print("cos(内心, o) 均值=%.3f 最小=%.3f 最大=%.3f" % (
    np.mean([_cos(x["内心"], x["状态"]) for x in rec]),
    np.min([_cos(x["内心"], x["状态"]) for x in rec]),
    np.max([_cos(x["内心"], x["状态"]) for x in rec])))
print("|内心|/|o| 均值=%.3f" % np.mean([np.linalg.norm(x["内心"]) / (np.linalg.norm(x["状态"]) + 1e-9) for x in rec]))
体2 = copy.deepcopy(体)
rec2 = J.跑测(体2, 种子, 内心覆盖="冻结")
变 = np.mean([tuple(rec2[i]["词"]) != tuple(rec[i]["词"]) for i in range(len(rec))])
print("冻结后正文改变率=%.3f 动作改变率=%.3f" % (
    变, np.mean([rec2[i]["动作id"] != rec[i]["动作id"] for i in range(len(rec))])))

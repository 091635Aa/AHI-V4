# -*- coding: utf-8 -*-
"""meeting53 探针3：多脑会议性能（只读诊断）。"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import V5_双通道 as V5

np.seterr(all="ignore")
T = 500
for 名, kw in [("3脑+通信", dict(通信=True)),
               ("3脑-断通信", dict(通信=False)),
               ("单脑全观测", dict(全观测单脑=True)),
               ("单脑部分观测(D64)", dict(部分观测单脑=True))]:
    t0 = time.time()
    果 = V5.跑多脑(拍数=T, 测拍=150, 种子=V5.种子默认, **kw)
    print("%-16s 训练=%.3f 测试=%.3f 后段=%.3f 用时=%.1fs" % (
        名, 果["训练成功率"], 果["测试成功率"], 果["测试后段成功率"], time.time() - t0))

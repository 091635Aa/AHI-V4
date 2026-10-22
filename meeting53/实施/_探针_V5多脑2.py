# -*- coding: utf-8 -*-
"""meeting53 探针6：多脑训练量/学习率扫描（只读诊断）。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import V5_双通道 as V5

np.seterr(all="ignore")
for 拍数 in (600, 1200):
    for η in (0.05, 0.10):
        V5.η动 = η
        for 名, kw in (("3脑+通信", dict(通信=True)), ("3脑-断", dict(通信=False)),
                       ("单脑全观测", dict(全观测单脑=True))):
            果 = V5.跑多脑(拍数=拍数, 测拍=200, 种子=V5.种子默认, **kw)
            print("拍=%4d η=%.2f %-8s 训练=%.3f 测试=%.3f 后段=%.3f" % (
                拍数, η, 名, 果["训练成功率"], 果["测试成功率"], 果["测试后段成功率"]))

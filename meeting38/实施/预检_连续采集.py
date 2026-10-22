# -*- coding: utf-8 -*-
"""预检_连续采集.py —— 定位 dxcam(DDA) 建相机失败的最小前置（诊断用，只读）。

用法：python 预检_连续采集.py [mss|gpu0|gpu1|front|all|none]
逐个叠加前置，观察 dxcam.create 何时开始失败（本机实测：裸进程 OK，全前置 FAIL）。
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, r"<workspace>/03_AHI\AHI2.1\meeting38")
sys.path.insert(0, r"<workspace>/03_AHI\AHI2.1\meeting38\实施")

模式 = sys.argv[1] if len(sys.argv) > 1 else "all"

if 模式 in ("mss", "all"):
    import mss
    _ = mss.mss()

if 模式 in ("gpu0", "all"):
    from 共存态 import 共存态
    _ = 共存态(种子=3800, 用gpu=False)

if 模式 in ("gpu1", "all"):
    from 共存态 import 共存态
    _ = 共存态(种子=3800, 用gpu=True)

if 模式 in ("front", "all"):
    from 共存态 import 共存态
    态 = 共存态(种子=3800)
    from 视场前端 import 视场前端
    _ = 视场前端(态, 种子=3800)

import dxcam
try:
    t0 = time.perf_counter()
    c = dxcam.create(output_idx=0, output_color="RGB")
    f = None
    while f is None and time.perf_counter() - t0 < 1.5:
        f = c.grab()
    print("MARK %s OK frame=%s" % (模式, None if f is None else f.shape))
except Exception as e:
    print("MARK %s FAIL %r" % (模式, repr(e)[:120]))
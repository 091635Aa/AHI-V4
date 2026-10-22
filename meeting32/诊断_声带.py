# -*- coding: utf-8 -*-
"""meeting32/诊断_声带.py —— D1 F0 倍频误差与 通过=False 根因诊断。"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "meeting28"))

from 声带 import 声带器官, FS, 每帧样本  # noqa: E402

带 = 声带器官(seed=300, D=1024, 目录=os.path.join(HERE, "声音"))
v1 = np.sign(np.random.default_rng(1).standard_normal(1024)).astype(np.float32)
遥测 = dict(能量=0.5, 新异=0.5, 消化债=0.0, 饱足=0.5)

r = 带.发声(v1, "妈妈", 遥测)
print("自检:", r["自检"])
帧 = 带.音节帧("妈妈", dict(带.韵律(遥测), _ñ=0.5), 带.分位读数(v1))
目标 = np.array(帧["f0"])
声源 = 帧["声源"]
print("帧数:", len(目标))
print("f0 帧:", ["%.0f" % f for f in 目标])
print("声源 :", list(声源))
wf = r["波形"]

浊帧 = [i for i, s in enumerate(声源) if s == 1 and 目标[i] > 0]
段s, 起0, 前 = [], None, None
for i in 浊帧:
    if 前 is None or i != 前 + 1:
        if 起0 is not None:
            段s.append((起0, 前))
        起0 = i
    前 = i
if 起0 is not None:
    段s.append((起0, 前))
print("浊段:", 段s)

for a, b in 段s:
    长 = (b - a + 1) * 每帧样本
    段 = min(长, 2048)
    中心 = a * 每帧样本 + 长 // 2
    起窗 = max(中心 - 段 // 2, 0)
    x = wf[起窗:起窗 + 段].astype(np.float64)
    x = x - x.mean()
    ac = np.correlate(x, x, "full")[段 - 1:]
    ac /= (ac[0] + 1e-12)
    滞后低, 滞后高 = int(FS / 400), int(FS / 70)
    窗 = ac[滞后低:滞后高]
    峰 = int(np.argmax(窗)) + 滞后低
    f测 = FS / 峰
    f标 = 目标[(a + b) // 2]
    print("段(%d,%d) f标=%.1f f测=%.1f(滞后%d) ac[峰]=%.3f ac[峰/2]=%.3f 误差=%.3f"
          % (a, b, f标, f测, 峰, ac[峰],
             ac[峰 // 2] if 峰 >= 2 * 滞后低 else -1,
             abs(f测 - f标) / f标))
    # 打印 ac 局部峰列表
    局部峰 = [(i + 滞后低, ac[i + 滞后低]) for i in range(1, len(窗) - 1)
             if 窗[i] > 窗[i - 1] and 窗[i] > 窗[i + 1] and 窗[i] > 0.3]
    print("  局部峰(lag,ac):", [(l, round(a2, 3)) for l, a2 in 局部峰[:8]])

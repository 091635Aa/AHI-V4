# -*- coding: utf-8 -*-
"""标定探针：同帧下 引擎_v5(并联) 与 引擎_v6(串联) 的增益差 → 定 v6 标定常数。"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, r"<workspace>/03_AHI\AHI4\meeting61\实施")
sys.path.insert(0, r"<workspace>/03_AHI\AHI4\meeting58\实施")

import 引擎_v5 as E5  # noqa: E402
import 引擎_v6 as E6  # noqa: E402

Fs = (825.0, 1305.0, 2714.0, 3654.0, 4698.0)
n = 30


def 帧(Fs_, 幅=0.85):
    return dict(f0=[120.0] * n, f1=[Fs_[0]] * n, f2=[Fs_[1]] * n, f3=[Fs_[2]] * n,
                f4=[Fs_[3]] * n, f5=[Fs_[4]] * n, 幅=[幅] * n, 气声=[0.02] * n,
                声源=[1] * n, 噪带=[None] * n, 送气=[0.0] * n, 紧喉=[0.25] * n,
                粗糙=[0.03] * n, 噪形=[0] * n)


for 名, mod in (("v5并联", E5), ("v6串联", E6)):
    原 = mod._源标定
    mod._源标定 = 1.0
    x, 元 = mod.合成(帧(Fs), 音色=None, 种子=300)
    r = float(np.sqrt(np.mean(x ** 2)))
    print("%-8s 标定=1 → RMS %.5f 峰 %.4f | 原标定 %.3f ⇒ RMS %.5f"
          % (名, r, float(np.max(np.abs(x))), 原, r * 原))
    mod._源标定 = 原

# 噪声道：确认两边在 噪标定 相同时电平是否一致
噪帧 = dict(f0=[180.0] * 20, f1=[300.0] * 20, f2=[1800.0] * 20, f3=[2600.0] * 20,
           f4=[3500.0] * 20, f5=[4500.0] * 20, 幅=[0.8] * 20, 气声=[0.0] * 20,
           声源=[0] * 20, 噪带=[(6800.0, 2200.0)] * 20, 送气=[1.0] * 20,
           紧喉=[0.3] * 20, 粗糙=[0.0] * 20, 噪形=[0] * 20)
for 名, mod in (("v5并联", E5), ("v6串联", E6)):
    x, _ = mod.合成(噪帧, 音色=None, 种子=300)
    print("%-8s 擦音 s：RMS %.5f 峰 %.4f" % (名, float(np.sqrt(np.mean(x ** 2))),
                                          float(np.max(np.abs(x)))))

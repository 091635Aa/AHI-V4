# -*- coding: utf-8 -*-
"""探针：串联（Klatt/全极点）声道 |H| 在各峰处的相对电平 —— 判定「F2/F3 被压掉」是模型错还是真实语音本来如此。

对照基准（真人语音的常识值）：
  无损耗声道各共振峰峰高**相等**；经声门流 −12dB/oct + 唇辐射 +6dB/oct 后，
  辐射谱的峰高按 −6dB/oct 递减 ⇒ 峰高 ≈ −6·log2(F_j/F_1) dB（相对 F1）。
所以 /a/(F1 825→F2 1305) 应约 −2.5dB；/i/(F1 363→F2 2349) 应约 −17.7dB。
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

MEET61 = r"<workspace>/03_AHI\AHI4\meeting61"
sys.path.insert(0, os.path.join(MEET61, "实施"))
sys.path.insert(0, os.path.join(os.path.dirname(MEET61), "meeting58", "实施"))

import 引擎_v4 as E4  # noqa: E402
import 音韵_v5 as Y  # noqa: E402
from 接口契约_角色声线 import 声线档案  # noqa: E402

FS = 44100
带 = E4.默认带宽


def H_串联(f, Fs):
    tot = np.ones(len(np.atleast_1d(f)), dtype=np.complex128)
    for j in range(5):
        A, B, C = E4.谐振器系数(Fs[j], 带[j])
        z = np.exp(-1j * 2.0 * math.pi * np.atleast_1d(f) / FS)
        tot = tot * (A / (1.0 - B * z - C * z ** 2))
    return tot


with open(os.path.join(MEET61, "产物", "声线档案", "EmoCompanion.json"), encoding="utf-8") as f:
    档 = 声线档案.从dict(json.load(f))
s = 档.共振峰缩放


def 目标(韵):
    目 = Y.韵母表[韵][0][-1]
    return (目[0] * s[0], 目[1] * s[1], 目[2] * s[2], 3500.0 * s[2], 4500.0 * s[2])


print("=" * 108)
print("%-4s %s" % ("韵", "各峰 |H|（dB，相对 F1 峰）          低频100Hz  预测(−6dB/oct) 与预测差"))
for 韵 in ("a", "o", "e", "i", "u", "v", "iii"):
    Fs = 目标(韵)
    f = np.array(list(Fs) + [100.0])
    Hv = np.abs(H_串联(f, Fs))
    db = 20.0 * np.log10(Hv / Hv[0])
    预测 = [-6.0 * math.log2(Fs[j] / Fs[0]) for j in range(5)]
    差 = [round(db[j] - 预测[j], 1) for j in range(5)]
    print("%-4s F1 %6.1f | F2 %6.1f | F3 %6.1f | F4 %6.1f | F5 %6.1f | 低频 %6.1f | 预测 %s | 差 %s"
          % (韵, db[0], db[1], db[2], db[3], db[4], db[5],
             " ".join("%6.1f" % x for x in 预测), " ".join("%6.1f" % x for x in 差)))
    print("     目标F %s" % " ".join("%6.0f" % x for x in Fs))
print("=" * 108)

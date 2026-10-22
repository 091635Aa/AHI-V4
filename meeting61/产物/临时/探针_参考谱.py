# -*- coding: utf-8 -*-
"""探针：参考原声（歌唱/清唱）的低频到底来自哪里 —— 决定 E5 带分布判据是否可信。"""
from __future__ import annotations

import glob
import json
import math
import os
import sys

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

FS = 44100
MEET61 = r"<workspace>/03_AHI\AHI4\meeting61"


def 带(x, lo, hi):
    n, hop = 4096, 2048
    k = 1 + (len(x) - n) // hop
    idx = np.arange(n)[None, :] + hop * np.arange(k)[:, None]
    P = np.mean(np.abs(np.fft.rfft(x[idx] * np.hanning(n)[None, :], axis=1)) ** 2, axis=0)
    P[0] = 0.0
    f = np.fft.rfftfreq(n, 1.0 / FS)
    T = float(P.sum()) or 1.0
    return float(P[(f >= lo) & (f < hi)].sum() / T), f, P


print("%-22s %7s %7s %7s %7s %7s %7s %7s %8s %8s" %
      ("文件", "0-60", "60-200", "200-500", "0-500", "500-1k", "1-2k", "2-4k", "重心Hz", "峰Hz"))
for 名 in ("EmoCompanion", "SpeakerA", "洛天依"):
    for p in sorted(glob.glob(os.path.join(MEET61, "素材", 名, "ref44_*.wav"))):
        fs, x = wavfile.read(p)
        x = np.asarray(x, dtype=np.float64)
        if x.ndim > 1:
            x = x.mean(axis=1)
        if np.max(np.abs(x)) > 1.2:
            x = x / 32768.0
        if fs != FS:
            g = math.gcd(int(fs), FS)
            x = resample_poly(x, FS // g, fs // g)
        b0, f, P = 带(x, 0, 60)
        b1, _, _ = 带(x, 60, 200)
        b2, _, _ = 带(x, 200, 500)
        b3, _, _ = 带(x, 0, 500)
        b4, _, _ = 带(x, 500, 1000)
        b5, _, _ = 带(x, 1000, 2000)
        b6, _, _ = 带(x, 2000, 4000)
        重心 = float((f * P).sum() / max(P.sum(), 1e-30))
        峰 = float(f[int(np.argmax(P))])
        print("%-22s %7.4f %7.4f %7.4f %7.4f %7.4f %7.4f %7.4f %8.0f %8.1f"
              % (os.path.basename(p), b0, b1, b2, b3, b4, b5, b6, 重心, 峰))

print("-" * 108)
# 合成句对照（v6 现状配置）
sys.path.insert(0, os.path.join(MEET61, "实施"))
sys.path.insert(0, os.path.join(os.path.dirname(MEET61), "meeting58", "实施"))
import 韵律_v6 as H5  # noqa: E402
import 引擎_v6 as E6  # noqa: E402
from 接口契约_角色声线 import 声线档案  # noqa: E402

with open(os.path.join(MEET61, "产物", "声线档案", "EmoCompanion.json"), encoding="utf-8") as fh:
    档 = 声线档案.从dict(json.load(fh))
H5.人类韵律_v5(档, 300).音色()  # 预热
for i, 句 in enumerate(("你好，我是EmoCompanion。", "今天的风很轻，适合慢慢地说一会儿话。")):
    x = H5.人类韵律_v5(档, 300 + i).说(句)[0]
    b0, f, P = 带(x, 0, 60)
    b1, _, _ = 带(x, 60, 200)
    b2, _, _ = 带(x, 200, 500)
    b3, _, _ = 带(x, 0, 500)
    b4, _, _ = 带(x, 500, 1000)
    b5, _, _ = 带(x, 1000, 2000)
    b6, _, _ = 带(x, 2000, 4000)
    重心 = float((f * P).sum() / max(P.sum(), 1e-30))
    峰 = float(f[int(np.argmax(P))])
    print("%-22s %7.4f %7.4f %7.4f %7.4f %7.4f %7.4f %7.4f %8.0f %8.1f"
          % ("v6合成:" + 句[:8], b0, b1, b2, b3, b4, b5, b6, 重心, 峰))

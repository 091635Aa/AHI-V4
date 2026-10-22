# -*- coding: utf-8 -*-
"""探针：逐韵母在 v6 下的谱包络峰位与带分布 —— 找「哪些韵母还是模糊」。"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

MEET61 = r"<workspace>/03_AHI\AHI4\meeting61"
sys.path.insert(0, os.path.join(MEET61, "实施"))
sys.path.insert(0, os.path.join(os.path.dirname(MEET61), "meeting58", "实施"))

import 韵律_v6 as H5  # noqa: E402
import 音韵_v5 as Y  # noqa: E402
from 接口契约_角色声线 import 声线档案  # noqa: E402

FS = 44100
with open(os.path.join(MEET61, "产物", "声线档案", "EmoCompanion.json"), encoding="utf-8") as f:
    档 = 声线档案.从dict(json.load(f))
    档低 = 档.派生(基频=120.0)
音 = H5.人类韵律_v5(档, 300).音色()


def 带(x):
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 4096:
        return {}
    n, hop = 4096, 2048
    k = 1 + (len(x) - n) // hop
    idx = np.arange(n)[None, :] + hop * np.arange(k)[:, None]
    P = np.mean(np.abs(np.fft.rfft(x[idx] * np.hanning(n)[None, :], axis=1)) ** 2, axis=0)
    P[0] = 0.0
    f = np.fft.rfftfreq(n, 1.0 / FS)
    T = float(P.sum()) or 1.0
    return {k2: round(float(P[(f >= a) & (f < b)].sum() / T), 3)
            for k2, (a, b) in {"0-500": (0, 500), "500-1k": (500, 1000), "1-2k": (1000, 2000),
                               "2-4k": (2000, 4000), "4-8k": (4000, 8000)}.items()}


def 包络峰(x):
    x = np.asarray(x, dtype=np.float64) - float(np.mean(x))
    X = np.fft.rfft(x * np.hanning(len(x)))
    cep = np.fft.irfft(np.log(np.abs(X) + 1e-12))
    L = max(4, int(0.0022 * FS))
    cep[L:-L] = 0.0
    env = np.exp(np.fft.rfft(cep).real)
    f = np.fft.rfftfreq(len(x), 1.0 / FS)
    m = (f >= 150.0) & (f <= 3500.0)
    return float(f[m][int(np.argmax(env[m]))])


# 字 → 期望 F1/F2（乘 EmoCompanion 缩放 1.1/1.044）
样 = [("衣", "i1"), ("乌", "u1"), ("迂", "v1"), ("啊", "a1"), ("哦", "o1"), ("鹅", "e1"),
     ("儿", "er2"), ("好", "hao3"), ("我", "wo3"), ("是", "shi4"), ("远", "yuan3"),
     ("林", "lin2"), ("风", "feng1"), ("很", "hen3"), ("青", "qing1"), ("和", "he2"),
     ("慢", "man4"), ("地", "di4"), ("说", "shuo1"), ("天", "tian1")]
print("=" * 112)
print("%-4s %-6s %8s %8s %8s | %s" % ("字", "韵", "F1目标", "F2目标", "包络峰", "带分布(0-500/500-1k/1-2k/2-4k/4-8k)"))
for 字, 拼 in 样:
    s = Y.解析音节(拼)
    目 = s["目标s"][-1]
    f1t, f2t = 目[0] * 1.1, 目[1] * 1.044
    帧 = H5.人类韵律_v5(档低, 300).建帧(字)
    idx = [i for i, t in enumerate(帧["声源"]) if int(t) == 1]
    a0 = (idx[len(idx) // 3] if idx else 0) * 441
    w, _, _ = H5.人类韵律_v5(档低, 300).说(字)
    seg = w[a0:a0 + int(0.12 * FS)]
    峰 = 包络峰(seg) if len(seg) > 512 else 0.0
    print("%-4s %-6s %8.0f %8.0f %8.0f | %s"
          % (字, s["韵"], f1t, f2t, 峰, json.dumps(带(seg), ensure_ascii=False)))
print("=" * 112)

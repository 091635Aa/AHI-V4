# -*- coding: utf-8 -*-
"""LPC 正确性自检：已知 AR(2) 极点能否复原（决定「共振峰实测」指标是否可用）。"""
import sys
import numpy as np

sys.dont_write_bytecode = True


def _lpc(ac, p):
    a = np.zeros(p + 1)
    a[0] = 1.0
    e = float(ac[0]) + 1e-12
    for i in range(1, p + 1):
        累 = ac[i] + float(np.dot(a[1:i], ac[i - 1:0:-1]))
        k = -累 / e
        新 = a.copy()
        for j in range(1, i):
            新[j] = a[j] + k * a[i - j]
        新[i] = k
        a = 新
        e *= max(1.0 - k * k, 1e-9)
    return a


def 极点(段, sr, 阶=16):
    ac = np.correlate(段, 段, "full")[len(段) - 1:len(段) + 阶]
    a = _lpc(ac, 阶)
    rt = np.roots(a)
    rt = rt[np.imag(rt) > 0]
    fr = np.angle(rt) * sr / (2 * np.pi)
    bw = -np.log(np.maximum(np.abs(rt), 1e-9)) * sr / np.pi
    return sorted((round(float(x)), round(float(b))) for x, b in zip(fr, bw)
                  if 90 < x < sr / 2 - 200 and b < 600)


sr = 16000
print("A. 纯 AR(2) 极点还原（期望 pole≈target）")
for f0, r in ((800, 0.99), (1500, 0.99), (2500, 0.99)):
    th = 2 * np.pi * f0 / sr
    a1 = -2 * r * np.cos(th)
    a2 = r * r
    rng = np.random.default_rng(0)
    n = 6000
    y = np.zeros(n)
    for i in range(2, n):
        y[i] = -a1 * y[i - 1] - a2 * y[i - 2] + 0.05 * rng.standard_normal()
    段 = y[1000:1000 + 512] * np.hamming(512)
    print("  target %4d -> %s" % (f0, 极点(段, sr)[:3]))

print("\nB. 本合成器 wav：同一文本、已知帧参数共振峰，实测是否稳定")
import os
import wave

HERE = os.path.dirname(os.path.abspath(__file__))
WAVDIR = os.path.join(HERE, "产物", "发声")


def 读(p):
    with wave.open(p, "rb") as w:
        n = w.getnframes()
        return (np.frombuffer(w.readframes(n), dtype="<i2").astype(np.float64)
                / 32768.0)


for 名 in sorted(os.listdir(WAVDIR))[-3:]:
    p = os.path.join(WAVDIR, 名)
    if not os.path.exists(p):
        continue
    x = 读(p)
    win, hop = 512, 160
    E = np.array([np.sum(x[i * hop:i * hop + win] ** 2)
                  for i in range(1 + (len(x) - win) // hop)])
    门 = E > 0.05 * E.max()
    F = [[], [], [], []]
    for i in np.where(门)[0]:
        段 = x[i * hop:i * hop + win] * np.hamming(win)
        p_ = 极点(段, sr, 16)
        if len(p_) >= 4:
            for j in range(4):
                F[j].append(p_[j][0])
    print("  %s  浊帧 %d  F1/F2/F3/F4 中位 %s  带 %s" %
          (名, int(门.sum()),
           [round(float(np.median(z)), 0) for z in F[:3]] if F[0] else None,
           [round(float(np.median(z)), 0) for z in F[3]] if F[0] else None))

# -*- coding: utf-8 -*-
"""meeting59 / refs-web：点查探针——指定时刻的谱峰/谐波梳 F0/停顿区底噪成分
用法: python 工具_点查.py <in.wav> t1 t2 ...
"""
import sys
import numpy as np
from scipy.signal import get_window

sys.path.insert(0, __file__.rsplit("\\", 1)[0])
from 工具_核验分析 import read_wav, db  # noqa


def frame_at(x, sr, tc, win=0.080):
    n = int(win * sr)
    i = int(tc * sr - n / 2)
    i = max(0, min(len(x) - n, i))
    return x[i:i + n]


def comb_f0(seg, sr, fmin=70.0, fmax=700.0):
    n = len(seg)
    w = get_window('hann', n)
    S = np.abs(np.fft.rfft(seg * w)) / (np.sum(w) / 2)
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    best = (0.0, 0.0)
    for f0 in np.arange(fmin, fmax, 0.5):
        ks = np.arange(1, int(7000 / f0) + 1)
        idx = np.round(ks * f0 * n / sr).astype(int)
        idx = idx[(idx > 2) & (idx < len(S) - 1)]
        if len(idx) < 3:
            continue
        # 每个谐波取 ±1 bin 最大
        val = np.maximum.reduce([S[np.clip(idx - 1, 0, len(S) - 1)],
                                 S[idx], S[np.clip(idx + 1, 0, len(S) - 1)]])
        score = float(np.sum(val * ks ** -0.5))
        if score > best[0]:
            best = (score, f0)
    return best


def peaks_in(seg, sr, fmax=4000, top=12):
    n = len(seg)
    w = get_window('hann', n)
    S = np.abs(np.fft.rfft(seg * w)) / (np.sum(w) / 2)
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    keep = freqs <= fmax
    S, freqs = S[keep], freqs[keep]
    loc = np.where((S[1:-1] > S[:-2]) & (S[1:-1] >= S[2:]))[0] + 1
    order = loc[np.argsort(S[loc])[::-1][:top]]
    return sorted([(float(freqs[i]), float(20 * np.log10(S[i] + 1e-12))) for i in order])


def main():
    src = sys.argv[1]
    sr, x = read_wav(src)
    print("file=%s sr=%d dur=%.3f" % (src, sr, len(x) / sr))
    for tc in [float(a) for a in sys.argv[2:]]:
        seg = frame_at(x, sr, tc)
        sc, f0 = comb_f0(seg, sr)
        print("\n--- t=%.2fs  rms=%.1f dBFS  combF0=%.1f Hz (score %.4g)" % (tc, db(np.sqrt((seg ** 2).mean())), f0, sc))
        pk = peaks_in(seg, sr, 4000, 10)
        print("   top peaks: " + " | ".join("%.0fHz %.0fdB" % p for p in pk))
        if f0 > 0:
            print("   f0/k Hz: " + " ".join("%.0f" % (f0 / k) for k in range(1, 11)))


if __name__ == "__main__":
    main()

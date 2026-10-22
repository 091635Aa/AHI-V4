# -*- coding: utf-8 -*-
"""meeting59 / refs-web：局部谱图放大（判断 F0 谐波间距 / 伴奏低频 / 颤音）
用法: python 工具_放大谱图.py <in.wav> <out.png> <t0> <t1> [--fmax 1500]
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import get_window

sys.path.insert(0, __file__.rsplit("\\", 1)[0])
from 工具_核验分析 import read_wav  # noqa


def main():
    src, out, t0, t1 = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
    fmax = 1500
    for i, a in enumerate(sys.argv):
        if a == "--fmax" and i + 1 < len(sys.argv):
            fmax = float(sys.argv[i + 1])
    sr, x = read_wav(src)
    i0, i1 = int(t0 * sr), int(t1 * sr)
    x = x[i0:i1]
    n, hop = 2048, 64
    m = 1 + (len(x) - n) // hop
    idx = np.arange(n)[None, :] + hop * np.arange(m)[:, None]
    fr = x[idx] * get_window('hann', n)
    S = 20 * np.log10(np.abs(np.fft.rfft(fr, axis=1)) + 1e-9)
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    keep = freqs <= fmax
    tspec = t0 + (np.arange(m) * hop + n / 2) / sr
    fig, ax = plt.subplots(2, 1, figsize=(16, 9), gridspec_kw={"height_ratios": [3, 1]})
    ax[0].imshow(S[:, keep].T, aspect="auto", origin="lower",
                 extent=[tspec[0], tspec[-1], 0, fmax], cmap="magma",
                 vmin=np.percentile(S, 25), vmax=np.percentile(S, 99.7))
    ax[0].set_ylabel("Hz")
    ax[0].set_title("%s  %.2f-%.2f s" % (src.rsplit("\\", 1)[-1], t0, t1))
    # 时间平均谱(帧能量 >p50 的帧) -> 看谐波间距
    rms = np.sqrt((fr ** 2).mean(axis=1))
    act = rms > np.percentile(rms, 60)
    avg = S[act].mean(axis=0)
    ax[1].plot(freqs[keep], avg[keep], lw=0.8)
    ax[1].set_xlim(0, fmax)
    ax[1].set_xlabel("Hz (avg over active frames)")
    ax[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=95)
    print("[PNG]", out)


if __name__ == "__main__":
    main()

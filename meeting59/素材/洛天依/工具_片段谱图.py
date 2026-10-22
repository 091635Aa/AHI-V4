# -*- coding: utf-8 -*-
"""meeting59 / refs-web：交付片段 QA 谱图（每段: 波形 + 0-6kHz 谱图 + F0）
用法: python 工具_片段谱图.py <out.png> <wav...>
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import get_window

sys.path.insert(0, __file__.rsplit("\\", 1)[0])
from 工具_核验分析 import read_wav, f0_track  # noqa


def main():
    out, files = sys.argv[1], sys.argv[2:]
    n = len(files)
    fig, axes = plt.subplots(n, 1, figsize=(16, 2.6 * n))
    if n == 1:
        axes = [axes]
    for ax, p in zip(axes, files):
        sr, x = read_wav(p)
        dur = len(x) / sr
        t, f0, vr, rms = f0_track(x, sr)
        ax.plot(np.arange(len(x)) / sr, x, lw=0.25, color="#444")
        ax.set_xlim(0, dur)
        ax.set_ylabel("wave", fontsize=7)
        ax2 = ax.twinx()
        m, hop, nw = 1 + (len(x) - 1024) // 160, 160, 1024
        idx = np.arange(nw)[None, :] + hop * np.arange(m)[:, None]
        S = 20 * np.log10(np.abs(np.fft.rfft(x[idx] * get_window('hann', nw), axis=1)) + 1e-9)
        fq = np.fft.rfftfreq(nw, 1.0 / sr)
        keep = fq <= 6000
        ax2.imshow(S[:, keep].T, aspect="auto", origin="lower",
                   extent=[0, dur, 0, 6000], cmap="magma",
                   vmin=np.percentile(S, 40), vmax=np.percentile(S, 99.7), alpha=0.85)
        v = np.isfinite(f0) & (vr > 0.42)
        ax2.scatter(t[v], f0[v], s=2, c="#00e5ff")
        ax2.set_ylim(0, 6000)
        ax2.set_yticks([])
        ax.set_title("%s  (%.2fs)" % (p.replace("\\", "/").rsplit("/", 1)[-1], dur), fontsize=9)
        ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(out, dpi=80)
    print("[PNG]", out)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""meeting59 / refs-web：参考音频可视化核验（波形 + 谱图 + F0/能量），纯 numpy/scipy/matplotlib
用法: python 工具_可视化.py <in.wav> <out.png> [--title 标题]
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import get_window

sys.path.insert(0, __file__.rsplit("\\", 1)[0])
from 工具_核验分析 import read_wav, f0_track, db  # noqa


def main():
    src, out = sys.argv[1], sys.argv[2]
    title = ""
    for i, a in enumerate(sys.argv):
        if a == "--title" and i + 1 < len(sys.argv):
            title = sys.argv[i + 1]
    sr, x = read_wav(src)
    t, f0, vr, rms = f0_track(x, sr)
    dur = len(x) / sr
    # 谱图
    n, hop = 1024, 160
    m = 1 + (len(x) - n) // hop
    idx = np.arange(n)[None, :] + hop * np.arange(m)[:, None]
    fr = x[idx] * get_window('hann', n)
    S = 20 * np.log10(np.abs(np.fft.rfft(fr, axis=1)) + 1e-9)
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    tspec = (np.arange(m) * hop + n / 2) / sr
    fig, ax = plt.subplots(3, 1, figsize=(18, 10), sharex=False)
    ax[0].plot(np.arange(len(x)) / sr, x, lw=0.3, color="#333")
    ax[0].set_ylabel("waveform")
    ax[0].set_xlim(0, dur)
    ax[0].set_title(title or src)
    ax[1].imshow(S.T, aspect="auto", origin="lower", extent=[0, dur, 0, 4000],
                 cmap="magma", vmin=np.percentile(S, 30), vmax=np.percentile(S, 99.5))
    v = np.isfinite(f0) & (vr > 0.42)
    ax[1].scatter(t[v], np.clip(f0[v], 0, 4000), s=1.5, c="#00e5ff", alpha=0.7,
                  label="F0 (autocorr)")
    ax[1].set_ylim(0, 4000)
    ax[1].set_ylabel("Hz")
    ax[1].legend(loc="upper right", fontsize=7)
    ax[2].plot(t, f0, lw=0.8, color="#1f77b4")
    ax[2].plot(t, np.clip(rms, 1e-6, None) * 0 + db(rms) + 200, lw=0.6, color="#999",
               label="RMS dB + 200")
    ax[2].set_ylabel("F0 Hz")
    ax[2].set_xlabel("s")
    ax[2].set_ylim(0, min(1200, np.nanpercentile(f0[np.isfinite(f0)], 99) * 1.2 if np.isfinite(f0).any() else 600))
    ax[2].legend(loc="upper right", fontsize=7)
    ax[2].set_xlim(0, dur)
    for a in ax:
        a.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out, dpi=90)
    print("[PNG]", out)


if __name__ == "__main__":
    main()

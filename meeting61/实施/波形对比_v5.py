# -*- coding: utf-8 -*-
"""meeting61/实施/波形对比_v5.py —— 参考 / v4 / v5 的波形包络 + 频谱图对比。

输出：产物/修复版v5/波形对比_v4_v5.png
说明：参考片段与合成文本不同，**只比谱形/包络/清浊比例，不做逐字对齐**。
"""
from __future__ import annotations

import os
import sys

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

HERE = os.path.dirname(os.path.abspath(__file__))
MEET61 = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FS = 44100
秒 = 3.0


def 读wav(p):
    fs, x = wavfile.read(p)
    x = np.asarray(x, dtype=np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if np.max(np.abs(x)) > 1.2:
        x = x / 32768.0
    if fs != FS:
        g = int(np.gcd(fs, FS))
        x = resample_poly(x, FS // g, fs // g)
    return x[:int(秒 * FS)]


def 包络(x, n=441):
    k = max(1, len(x) // n)
    e = np.array([np.sqrt(np.mean(x[i * n:(i + 1) * n] ** 2)) for i in range(k)])
    return 20.0 * np.log10(np.maximum(e, 1e-7)), n / FS


def 主():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    项 = [
        ("参考 · EmoCompanion（不同文本，仅比谱形）",
         os.path.join(MEET61, "素材", "EmoCompanion", "ref44_EmoCompanion_01.wav")),
        ("v4 · 组C EmoCompanion",
         os.path.join(MEET61, "产物", "测试组", "组C_v4韵律", "角色_EmoCompanion.wav")),
        ("v5 · 组V5 EmoCompanion",
         os.path.join(MEET61, "产物", "测试组", "组V5_三角色", "角色_EmoCompanion.wav")),
    ]
    fig, axs = plt.subplots(len(项), 2, figsize=(13, 8.2),
                            gridspec_kw=dict(width_ratios=[1, 1.25]))
    for r, (标题, 路径) in enumerate(项):
        x = 读wav(路径)
        edb, 帧秒 = 包络(x)
        t = np.arange(len(edb)) * 帧秒
        axs[r, 0].plot(t, edb, lw=0.8, color="#2b6cb0")
        axs[r, 0].set_ylim(-70, 0)
        axs[r, 0].set_ylabel("包络 dBFS")
        axs[r, 0].set_xlabel("秒")
        axs[r, 0].set_title(标题, fontsize=10)
        axs[r, 0].grid(alpha=0.25)
        axs[r, 1].specgram(x, NFFT=1024, Fs=FS, noverlap=768, cmap="magma",
                           vmin=-120, vmax=-20)
        axs[r, 1].set_ylim(0, 8000)
        axs[r, 1].set_ylabel("Hz")
        axs[r, 1].set_xlabel("秒")
        axs[r, 1].set_title("频谱图 0–8kHz", fontsize=10)
    fig.suptitle("参考 / v4 / v5：包络与频谱（v5 = 音韵重建 + 协同发音 + 声门流源）",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = os.path.join(MEET61, "产物", "修复版v5", "波形对比_v4_v5.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=140)
    print("[OK]", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(主())

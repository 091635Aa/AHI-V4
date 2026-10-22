# -*- coding: utf-8 -*-
"""meeting61/实施/波形对比_v6.py —— 波形级证据图（v6 交付用）。

产出两张 PN，放在 产物/修复版v6/：
  波形对比_v5_v6.png   参考 / v5 / v6：包络 + 频谱图 + 长期平均谱（标 F1/F2）
  元音包络_v5_v6.png   6 个单元音：v5（峰被 300Hz 低频平台拖走） vs v6（峰回到共振峰）
口径：参考与合成文本不同，只比谱形/包络，不做逐字对齐（C13 确定性，纯 numpy/matplotlib）。
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

HERE = os.path.dirname(os.path.abspath(__file__))
MEET61 = os.path.dirname(HERE)
AHI4 = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(AHI4, "meeting58", "实施"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import 韵律_v6 as H5  # noqa: E402
import 引擎_v5 as E5  # noqa: E402
import 引擎_v6 as E6  # noqa: E402
from 接口契约_角色声线 import 声线档案  # noqa: E402

FS = 44100
秒 = 3.0
出目录 = os.path.join(MEET61, "产物", "修复版v6")
os.makedirs(出目录, exist_ok=True)


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


def 长期谱(x, n=4096, hop=2048):
    k = max(1, 1 + (len(x) - n) // hop)
    idx = np.arange(n)[None, :] + hop * np.arange(k)[:, None]
    P = np.mean(np.abs(np.fft.rfft(x[idx] * np.hanning(n)[None, :], axis=1)) ** 2, axis=0)
    f = np.fft.rfftfreq(n, 1.0 / FS)
    P = np.maximum(P, 1e-30)
    return f, 10.0 * np.log10(P / float(np.max(P)))


def 谱包络(x):
    x = np.asarray(x, dtype=np.float64) - float(np.mean(x))
    X = np.fft.rfft(x * np.hanning(len(x)))
    cep = np.fft.irfft(np.log(np.abs(X) + 1e-12))
    L = max(4, int(0.0022 * FS))
    cep[L:-L] = 0.0
    env = np.exp(np.fft.rfft(cep).real)
    f = np.fft.rfftfreq(len(x), 1.0 / FS)
    return f, 20.0 * np.log10(np.maximum(env, 1e-30))


def 图_三角色():
    项 = [
        ("参考 · EmoCompanion（歌唱/清唱，文本不同）", os.path.join(MEET61, "素材", "EmoCompanion", "ref44_EmoCompanion_01.wav")),
        ("v5 · EmoCompanion（并联共振峰组）", os.path.join(MEET61, "产物", "测试组", "组V6_三角色", "角色_EmoCompanion.wav")),
    ]
    # 组V6_三角色 的角色文件是 v5 目录里的旧名，这里用 v5 的组目录确保取到 v5 波形
    项[1] = ("v5 · EmoCompanion（并联共振峰组）",
            os.path.join(MEET61, "产物", "测试组", "组V5_三角色", "角色_EmoCompanion.wav"))
    项.append(("v6 · EmoCompanion（Klatt 串联声道）",
              os.path.join(MEET61, "产物", "测试组", "组V6_三角色", "角色_EmoCompanion.wav")))
    fig, axs = plt.subplots(len(项), 3, figsize=(15.5, 8.6),
                            gridspec_kw=dict(width_ratios=[1, 1.1, 1.15]))
    for r, (标题, 路径) in enumerate(项):
        x = 读wav(路径)
        edb, 帧秒 = 包络(x)
        axs[r, 0].plot(np.arange(len(edb)) * 帧秒, edb, lw=0.8, color="#2b6cb0")
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
        f, db = 长期谱(x)
        m = f <= 5000
        axs[r, 2].plot(f[m], db[m], lw=1.1, color="#c05621")
        for hv, c in ((825.0, "#2f855a"), (1305.0, "#805ad5")):
            axs[r, 2].axvline(hv, color=c, ls="--", lw=0.9, alpha=0.8)
        axs[r, 2].set_ylim(-75, 2)
        axs[r, 2].set_xlabel("Hz")
        axs[r, 2].set_ylabel("长期平均谱 dB")
        axs[r, 2].set_title("长期平均谱（虚线 = EmoCompanion /a/ 的 F1 825 / F2 1305）", fontsize=10)
        axs[r, 2].grid(alpha=0.25)
    fig.suptitle("参考 / v5 / v6 波形级对比（v6 = 浊音道改 Klatt 串联：谱峰从 300Hz 低频平台回到共振峰）",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    out = os.path.join(出目录, "波形对比_v5_v6.png")
    fig.savefig(out, dpi=135)
    plt.close(fig)
    print("[OK]", out)


def 图_元音包络():
    with open(os.path.join(MEET61, "产物", "声线档案", "EmoCompanion.json"), encoding="utf-8") as f:
        档 = 声线档案.从dict(json.load(f))
    档低 = 档.派生(基频=120.0)
    音 = H5.人类韵律_v5(档低, 300).音色()
    样 = (("啊", "a1", 825.0), ("哦", "o1", 528.0), ("鹅", "e1", 572.0),
         ("衣", "i1", 330.0), ("乌", "u1", 352.0), ("迂", "v1", 330.0))
    fig, axs = plt.subplots(2, 3, figsize=(15.0, 7.4))
    for i, (字, 拼, f1t) in enumerate(样):
        s = 拼
        帧 = H5.人类韵律_v5(档低, 300).建帧(字)
        idx = [k for k, t in enumerate(帧["声源"]) if int(t) == 1]
        a0 = (idx[len(idx) // 3] if idx else 0) * 441
        x6, _, _ = H5.人类韵律_v5(档低, 300).说(字)
        x5 = E5.合成(帧, 音色=音, 种子=300)[0]
        seg6 = x6[a0:a0 + int(0.12 * FS)]
        seg5 = x5[a0:a0 + int(0.12 * FS)]
        f6, e6 = 谱包络(seg6)
        f5, e5 = 谱包络(seg5)
        m = f6 <= 4000
        ax = axs[i // 3, i % 3]
        ax.plot(f5[m], e5[m], lw=1.2, color="#c53030", label="v5 并联")
        ax.plot(f6[m], e6[m], lw=1.2, color="#2b6cb0", label="v6 串联")
        ax.axvline(f1t, color="#2f855a", ls="--", lw=1.0)
        ax.set_title("%s（/a/ F1=%.0fHz 虚线）" % (字, f1t), fontsize=10)
        ax.set_xlabel("Hz")
        ax.set_ylabel("谱包络 dB")
        ax.set_ylim(-40, 5)
        ax.grid(alpha=0.25)
        if i == 0:
            ax.legend(fontsize=8)
    fig.suptitle("单元音谱包络：v5 的最大点在 ~300Hz（低频平台），v6 回到 F1/F2 共振峰区", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    out = os.path.join(出目录, "元音包络_v5_v6.png")
    fig.savefig(out, dpi=135)
    plt.close(fig)
    print("[OK]", out)


def 主():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    图_三角色()
    图_元音包络()
    return 0


if __name__ == "__main__":
    raise SystemExit(主())

# -*- coding: utf-8 -*-
"""meeting61/实施/实验_共振峰频谱.py —— 用「解析传递函数 × 谐波电平」精确对照声道架构。

为什么要换量法：倒谱包络在窄窗里找峰会撞到窗边（诊断_元音_v5 的 F1 误差有一部分是
量法artifact）。这里改成**耳听的等价物**——逐谐波电平：
    输出(k·f0) = |H_声源(k·f0)| × |H_声道(k·f0)|
|H_声道| 用双二阶的解析传递函数直接算，无 FFT/倒谱噪声。

要回答的问题：
  1) 元音的谱峰（最高谐波）到底落在哪？落在 F1 上才是「有元音」；
  2) F1 相对低频平台（k=1..3）突出多少 dB？
  3) F1/F2/F3 谐波电平的先后次序是否符合「F1>F2>F3」？
架构候选同 实验_共振峰架构.py：A 并联(v5现状) / B 串联 / C 串联+逐级增益 / D 并联中心归一。
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import 引擎_v4 as E4  # noqa: E402
import 引擎_v5 as E5  # noqa: E402

FS = 44100
元音 = {
    "a": (825.0, 1305.0, 2714.0, 3654.0, 4698.0),
    "o": (528.0, 887.0, 2610.0, 3445.0, 4385.0),
    "e": (572.0, 1305.0, 2610.0, 3602.0, 4646.0),
    "i": (330.0, 2349.0, 3132.0, 3967.0, 5011.0),
    "u": (352.0, 752.0, 2610.0, 3550.0, 4489.0),
}
带宽 = E4.默认带宽
F0 = 120.0


def _H(F, BW, f):
    """双二阶谐振器（A=1−B−C，DC 增益 1）的解析频响。"""
    A, B, C = E4.谐振器系数(F, BW)
    z = np.exp(-1j * 2.0 * math.pi * np.asarray(f, dtype=np.float64) / FS)
    return A / (1.0 - B * z - C * z ** 2)


def H_并联(f, Fs, 增益):
    tot = np.zeros(len(np.atleast_1d(f)), dtype=np.complex128)
    for j in range(5):
        tot += 增益[j] * _H(Fs[j], 带宽[j], f)
    return tot


def H_串联(f, Fs, 增益=(1.0,) * 5):
    tot = np.ones(len(np.atleast_1d(f)), dtype=np.complex128)
    for j in range(5):
        tot *= 增益[j] * _H(Fs[j], 带宽[j], f)
    return tot


def H_并联中心归一(f, Fs, 增益):
    f = np.atleast_1d(f)
    tot = np.zeros(len(f), dtype=np.complex128)
    for j in range(5):
        g = 1.0 / max(abs(_H(Fs[j], 带宽[j], Fs[j])), 1e-9)
        tot += 增益[j] * g * _H(Fs[j], 带宽[j], f)
    return tot


架构 = {
    "A_并联_v5现状": lambda f, Fs: H_并联(f, Fs, (1.0, 0.55, 0.20, 0.09, 0.04)),
    "B_串联": lambda f, Fs: H_串联(f, Fs),
    "C_串联+逐级增益": lambda f, Fs: H_串联(f, Fs, (1.0, 1.0, 0.70, 0.50, 0.35)),
    "D_并联中心归一": lambda f, Fs: H_并联中心归一(f, Fs, (1.0, 0.55, 0.30, 0.18, 0.10)),
}


def 源谱(f):
    """声门流 + 6kHz 低通 + 单次唇辐射，在给定频点处的幅度（解析）。"""
    n = int(0.30 * FS)
    表 = E5.声门流表(1024, 0.3)
    相 = np.mod(np.cumsum(np.full(n, F0 / FS)), 1.0)
    idx = np.minimum((相 * 1024).astype(np.int64), 1023)
    x = 表[idx]
    b, a = E4.butter(2, 6000.0 / (FS * 0.5), btype="low")
    x = E4.lfilter(b, a, x)
    辐 = np.empty_like(x)
    辐[0] = x[0]
    辐[1:] = x[1:] - x[:-1]
    X = np.abs(np.fft.rfft(辐 * np.hanning(n)))
    fr = np.fft.rfftfreq(n, 1.0 / FS)
    # 按谐波位置取峰，再线性插值成任意频点的连续估计
    ks = np.arange(1, int(fr[-1] / F0))
    fs = ks * F0
    amp = np.array([np.max(X[(fr >= c - 12) & (fr <= c + 12)]) if np.any(
        (fr >= c - 12) & (fr <= c + 12)) else 0.0 for c in fs])
    return np.interp(np.asarray(f), fs, amp, left=amp[0], right=0.0)


def 量(f, Fs, H):
    """在谐波上算输出电平，报：最高谐波、F1/F2/F3 电平、低频平台、F1 突出度。"""
    ls = np.arange(1, 60) * F0
    ls = ls[ls <= 8000.0]
    Hv = np.abs(H(ls, Fs))
    Sv = 源谱(ls)
    out = Hv * Sv
    out = out / max(float(np.max(out)), 1e-12)
    db = 20.0 * np.log10(np.maximum(out, 1e-12))
    低 = float(np.mean(db[:3]))
    峰 = float(ls[int(np.argmax(out))])
    附 = {}
    for k in (1, 2, 3):
        c = Fs[k - 1]
        fq = min(ls, key=lambda x: abs(x - c))
        i = int(np.argmin(np.abs(ls - c)))
        附["F%d" % k] = dict(谐波=int(fq), dB=round(float(db[i]), 1))
    # 谐波里 F1 附近的最大电平，相对低频平台
    i1 = (ls >= Fs[0] * 0.8) & (ls <= Fs[0] * 1.2)
    突 = float(np.max(db[i1]) - 低) if np.any(i1) else None
    return dict(最高谐波Hz=峰, 低频平台dB=round(低, 1),
                F1突出度dB=round(突, 1) if 突 is not None else None,
                谐波电平=附,
                次序=("%d>%d>%d" % tuple(
                    sorted(((1, 附["F1"]["dB"]), (2, 附["F2"]["dB"]), (3, 附["F3"]["dB"])),
                           key=lambda z: -z[1]))[0][0],
                    ) if False else "F1>F2>F3" if (附["F1"]["dB"] > 附["F2"]["dB"] > 附["F3"]["dB"])
                else "否")


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    报 = {}
    print("=" * 112)
    print("耳听等价量：输出(k·f0)=|H源|×|H声道|，归一化到最大值。F0=%dHz" % F0)
    print("=" * 112)
    for 元 in 元音:
        Fs = 元音[元]
        print("[%s] 目标 %s" % (元, " ".join("%.0f" % t for t in Fs[:3])))
        for 名, H in 架构.items():
            r = 量(0.0, Fs, H)
            报.setdefault(名, {})[元] = r
            print("   %-16s 最高谐波 %5.0f | 低频台 %6.1f | F1突出 %6.1f | "
                  "F1 %5.0f(%5.1f) F2 %5.0f(%5.1f) F3 %5.0f(%5.1f) | F1>F2>F3 %s"
                  % (名, r["最高谐波Hz"], r["低频平台dB"], r["F1突出度dB"] or 0,
                     r["谐波电平"]["F1"]["谐波"], r["谐波电平"]["F1"]["dB"],
                     r["谐波电平"]["F2"]["谐波"], r["谐波电平"]["F2"]["dB"],
                     r["谐波电平"]["F3"]["谐波"], r["谐波电平"]["F3"]["dB"],
                     r["次序"]))
        print("-" * 112)
    out = os.path.join(os.path.dirname(HERE), "产物", "临时", "实验_共振峰频谱.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(报, f, ensure_ascii=False, indent=1)
    print("JSON:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# -*- coding: utf-8 -*-
"""meeting69/实施/合成_S0刺激.py —— S0 合成刺激（自研 DSP 真值；确定性；无外部模型）。

生成 S0 预注册（03）所需刺激与 ground truth：
  VOT 序列 / 塞擦 burst→frication / 擦音 5 类 / 声调 T1–T4 / 稳态元音 / 掩蔽探针 / 边界 click
全部 22.05 kHz 单声道 16-bit；真值随 wav 落 manifest.json。
"""
from __future__ import annotations

import json
import math
import sys
import wave
from pathlib import Path

import numpy as np

SR = 22050
HERE = Path(__file__).resolve().parent
M68 = HERE.parent
出 = M68 / "产物" / "S0" / "刺激"
种子 = 6900


def 写wav(路, x):
    x = np.asarray(x, dtype=np.float32)
    p = float(np.abs(x).max()) if x.size else 0.0
    if p > 0:
        x = x / p * 0.9
    pcm = np.clip(x, -1, 1)
    with wave.open(str(路), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((pcm * 32767).astype(np.int16).tobytes())


def 谐振(x, f, bw=80.0):
    """2 阶共振器（formant 近似）。"""
    import scipy.signal as sps
    r = math.exp(-math.pi * bw / SR)
    θ = 2 * math.pi * f / SR
    b = [1.0 - r]
    a = [1.0, -2 * r * math.cos(θ), r * r]
    return sps.lfilter(b, a, x)


def 元音(f0, 时长, 形位=(700.0, 1200.0, 2500.0), 振幅=0.5):
    n = int(时长 * SR)
    t = np.arange(n) / SR
    if np.isscalar(f0):
        φ = 2 * np.pi * f0 * t
    else:
        φ = 2 * np.pi * np.cumsum(f0) / SR
    # 声门脉冲：相位过零处脉冲（近似 LF）
    脉 = np.zeros(n)
    过 = np.where(np.diff(np.sign(np.sin(φ))) < 0)[0]
    脉[过] = 1.0
    源 = 脉
    for i, f in enumerate(形位):
        y = 谐振(源, f, 80.0 + 20 * i)
        if i == 0:
            声 = y
        else:
            声 = 声 + 0.7 * y
    声 = np.tanh(声 / (np.abs(声).max() + 1e-9) * 2.0) * 振幅
    return 声.astype(np.float32)


def 噪声段(时长, 中心=None, 带宽=None, 振幅=0.5, 高通=None, 低通=None, 种子=0):
    rs = np.random.RandomState(种子)
    n = int(时长 * SR)
    x = rs.randn(n)
    import scipy.signal as sps
    if 中心 is not None and 带宽 is not None:
        lo = max(20, 中心 - 带宽 / 2) / (SR / 2); hi = min(SR / 2 - 100, 中心 + 带宽 / 2) / (SR / 2)
        sos = sps.butter(4, [lo, hi], btype="band", output="sos")
        x = sps.sosfilt(sos, x)
    elif 高通 or 低通:
        if 高通 and 低通:
            sos = sps.butter(4, [高通 / (SR / 2), 低通 / (SR / 2)], btype="band", output="sos")
        elif 高通:
            sos = sps.butter(4, 高通 / (SR / 2), btype="high", output="sos")
        else:
            sos = sps.butter(4, 低通 / (SR / 2), btype="low", output="sos")
        x = sps.sosfilt(sos, x)
    x = x / (np.abs(x).max() + 1e-9) * 振幅
    return x.astype(np.float32)


def 爆破(中心=3500.0, 带宽=2500.0, 时长=0.012, 种子=1):
    x = 噪声段(时长, 中心, 带宽, 振幅=0.9, 种子=种子)
    env = np.exp(-np.linspace(0, 6, len(x)))
    return (x * env).astype(np.float32)


def 送气(时长, 种子=2):
    return 噪声段(时长, 高通=500, 低通=4500, 振幅=0.35, 种子=种子)


def 拼接(*段):
    return np.concatenate([np.asarray(a, np.float32) for a in 段]).astype(np.float32)


def 静默(时长):
    return np.zeros(int(时长 * SR), np.float32)


def F0轮廓(型):
    # 半音近似 5 度标记（5=高，1=低），映射到 Hz：T1 55→130/130；T2 35→100→130；T3 214→100→85→120；T4 51→130→85
    if 型 == "T1":
        return np.full(50, 130.0)
    if 型 == "T2":
        return np.linspace(100, 130, 50)
    if 型 == "T3":
        return np.concatenate([np.linspace(100, 85, 25), np.linspace(85, 120, 25)])
    if 型 == "T4":
        return np.linspace(130, 85, 50)


def 主():
    出.mkdir(parents=True, exist_ok=True)
    清 = []
    真 = []

    def 存(名, x, gt):
        路 = 出 / 名
        写wav(路, x)
        真.append(dict(文件=名, 路径=str(路), **gt))
        清.append(名)

    # ① VOT 序列：爆破 + 送气(vot) + 元音
    for vot_ms in (0, 5, 10, 20, 30, 40, 60, 80, 100):
        burst = 爆破()
        asp = 送气(vot_ms / 1000.0) if vot_ms > 0 else np.zeros(0, np.float32)
        vow = 元音(120.0, 0.18)
        x = 拼接(burst, asp, vow)
        t0 = 0.3
        x = 拼接(静默(t0), x)
        存(f"vot_{vot_ms:03d}.wav", x,
            dict(类="VOT", vot_ms=vot_ms, burst秒=t0, 浊起秒=t0 + 0.012 + vot_ms / 1000.0))
    # ② 塞擦：爆破 + 间隔(gap) + 摩擦 + 元音
    for gap_ms in (10, 20, 30):
        burst = 爆破(中心=4000, 带宽=2500)
        gap = 静默(gap_ms / 1000.0)
        fric = 噪声段(0.08, 中心=4000, 带宽=3000, 振幅=0.6, 种子=3)
        vow = 元音(120.0, 0.15)
        x = 拼接(burst, gap, fric, vow)
        t0 = 0.3
        x = 拼接(静默(t0), x)
        存(f"塞擦_gap{gap_ms:02d}.wav", x,
            dict(类="塞擦", gap_ms=gap_ms, burst秒=t0, 摩擦起秒=t0 + 0.012 + gap_ms / 1000.0))
    # ③ 擦音 5 类
    擦 = {"s": 6500, "ɕ": 5500, "ʂ": 4000, "x": 2000, "f": None}
    for 名, f in 擦.items():
        for k in range(5):                     # 5 次重复（不同噪声种子）供闭合式分类
            if f is None:
                x = 噪声段(0.15, 高通=1500, 低通=8000, 振幅=0.6, 种子=4 + k)
            else:
                x = 噪声段(0.15, 中心=f, 带宽=max(800, f * 0.3), 振幅=0.6, 种子=4 + k)
            x = 拼接(静默(0.3), x, 静默(0.1))
            存(f"擦音_{名}_{k}.wav", x, dict(类="擦音", 标签=名, 中心Hz=f, 重复=k))
    # ④ 声调 T1–T4
    for 型 in ("T1", "T2", "T3", "T4"):
        f0 = np.concatenate([np.repeat(F0轮廓(型), SR // 100)])  # 50 点 × 10ms = 0.5s
        vow = 元音(f0, 0.5)
        x = 拼接(静默(0.3), vow)
        存(f"声调_{型}.wav", x, dict(类="声调", 标签=型, f0轮廓=[float(v) for v in f0[::50]]))
    # ⑤ 稳态元音 2 s（适应）
    vow = 元音(120.0, 2.0)
    x = 拼接(静默(0.2), vow)
    存("稳态元音.wav", x, dict(类="适应", 稳态起秒=0.2, 时长秒=2.0))
    # ⑥ 掩蔽探针：1k 纯音 + 掩蔽噪声（Δf = 0.5/1/2/4 ERB）
    for 档 in (0.5, 1.0, 2.0, 4.0):
        n = int(0.4 * SR)
        t = np.arange(n) / SR
        probe = 0.2 * np.sin(2 * np.pi * 1000 * t)
        中心 = 1000 + 档 * 132.0
        mask = 噪声段(0.4, 中心=中心, 带宽=max(200, 档 * 132), 振幅=0.5, 种子=5)
        x = 拼接(静默(0.3), probe + mask)
        存(f"掩蔽_d{档:.1f}.wav", x, dict(类="掩蔽", 档=档, 探针Hz=1000))
    # ⑦ 边界：静默 + click + 静默 + 爆破
    click = 爆破(中心=3000, 带宽=4000, 时长=0.004, 种子=6)
    burst = 爆破()
    x = 拼接(静默(0.3), click, 静默(0.3), burst, 静默(0.3))
    存("边界.wav", x, dict(类="边界", click秒=0.3, burst秒=0.6 + 0.004))
    (M68 / "产物" / "S0" / "manifest.json").write_text(
        json.dumps(dict(工具="合成_S0刺激.py", 种子=种子, 采样率=SR, 刺激数=len(真), 真值=真),
                   ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"已生成 {len(真)} 个刺激 → {出}")
    print("manifest:", M68 / "产物" / "S0" / "manifest.json")
    return 0


if __name__ == "__main__":
    sys.exit(主())

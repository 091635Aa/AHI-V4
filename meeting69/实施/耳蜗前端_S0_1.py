# -*- coding: utf-8 -*-
"""meeting69/实施/耳蜗前端_S0_1.py —— 耳蜗前端 v0.2（S0.1 承重件；自研 DSP；确定性；C10）。

= `耳蜗前端_S0.py` 的**只读副本 + 仅两处修正**（见 `06_预注册_S0.1_F0与适应修正.md`）：
  ① F0 估计器：25 ms 自相关 → **YIN 型**（40 ms 窗 / 2.5 ms 跳 / 窗中心 / CMNDF θ=0.15 /
     抛物线细化 / 能量+非周期度浊音门），见 `F0轨迹`；
  ② 适应实现：新增三变体（V1 除式5/100 = S0 现状；V2 除式5/200；V3 减式 v0 同形），
     见 `适应`；默认 V3（按 06 §4 选择规则选中，见 07 证据）。
其余（64 带 4 阶 gammatone、半波整流、1 ms 包络、立方根压缩、事件层、log-mel 对照在判据件）
与 S0 件逐行相同，保证 J1/J2/J3/J5 可比。

用法：
  & $py 耳蜗前端_S0_1.py 分析 "x.wav" [--变体 V1|V2|V3] --出 out.json
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
import time
import wave
from pathlib import Path

import numpy as np

SR = 22050
频带数 = 64
FMIN, FMAX = 60.0, 7800.0
包络步 = 22          # ≈1 ms（22050/22≈1002 Hz）
F0跳 = SR // 400     # 2.5 ms
F0窗ms = 40          # 40 ms（S0 的 25 ms 在 85 Hz 只有 2.1 周期）
F0最低, F0最高 = 60.0, 500.0
F0阈 = 0.15          # YIN 教科书默认；见 06 §3
F0非周期阈 = 0.35    # 浊音判决上界；见 06 §3
快适应 = 0.005
慢适应 = 0.100
适应变体集 = ("V1", "V2", "V3")


def ERB(f):
    return 24.7 * (4.37 * f / 1000.0 + 1.0)


def 中心频率():
    def pos(f):
        return math.log(4.37 * f / 1000.0 + 1.0)
    a, b = pos(FMIN), pos(FMAX)
    return np.array([(math.exp(a + (b - a) * i / (频带数 - 1)) - 1.0) * 1000.0 / 4.37
                     for i in range(频带数)])


def 读wav(路):
    with wave.open(str(路), "rb") as w:
        n = w.getnframes(); sr = w.getframerate(); ch = w.getnchannels()
        x = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32) / 32768.0
    if ch == 2:
        x = x.reshape(-1, 2).mean(axis=1)
    if sr != SR:
        import scipy.signal as sps
        g = math.gcd(int(sr), SR)
        x = sps.resample_poly(x, SR // g, int(sr) // g).astype(np.float32)
    return np.ascontiguousarray(x)


def 滤波组(x):
    import scipy.signal as sps
    fs = 中心频率()
    out = np.zeros((len(fs), len(x)), dtype=np.float32)
    for i, f in enumerate(fs):
        bw = 1.019 * ERB(f)
        r = math.exp(-2 * math.pi * bw / SR)
        θ = 2 * math.pi * f / SR
        sec = [1.0, 0.0, 0.0, 1.0, -2 * r * math.cos(θ), r * r]
        sos = np.array([sec] * 4, dtype=np.float64)
        z = np.exp(1j * θ)
        A1 = 1.0 - 2.0 * r * math.cos(θ) * z + r * r * z * z
        增益 = (1.0 / max(abs(A1), 1e-12)) ** 4
        out[i] = (sps.sosfilt(sos, x) / 增益).astype(np.float32)
    return out, fs


def 包络(x):
    # 半波整流 + 1 ms 低通（因果）
    x = np.maximum(x, 0.0)
    α = 1.0 / (SR * 0.001)
    y = np.empty_like(x)
    acc = 0.0
    for i in range(len(x)):          # 22k 步 × 长度；60 s ≈ 1.3M（可接受）
        acc += (x[i] - acc) * α
        y[i] = acc
    return y


def 适应(压缩, 变体="V3"):
    """06 §4 三变体；时间常数按 D2 固定 5/100（V2 慢态 200），只换组合规则。"""
    if 变体 not in 适应变体集:
        raise ValueError(f"未知适应变体: {变体}")
    T = 压缩.shape[1]
    if 变体 == "V3":
        h0 = np.zeros(压缩.shape[0], np.float64)
    else:
        h0 = np.maximum(np.median(压缩, axis=1) * 0.1, 1e-6)
    慢 = 0.200 if 变体 == "V2" else 慢适应
    af = np.zeros(压缩.shape[0], np.float64)
    as_ = np.zeros(压缩.shape[0], np.float64)
    kf = 1.0 / (快适应 * 1000.0)
    ks = 1.0 / (慢 * 1000.0)
    out = np.zeros(压缩.shape, np.float32)
    for t in range(T):
        c = 压缩[:, t]
        af += (c - af) * kf
        as_ += (c - as_) * ks
        if 变体 == "V3":
            out[:, t] = np.maximum(c - (0.6 * af + 0.4 * as_), 0.0).astype(np.float32)
        else:
            out[:, t] = (c / (af + as_ + h0)).astype(np.float32)
    return out, h0


def F0轨迹(x):
    """YIN 型 F0（06 §3）。返回 (t=窗中心秒, f0=Hz；0=清音/静音)。"""
    import scipy.signal as sps
    lp = sps.butter(4, 1200 / (SR / 2.0), btype="low", output="sos")
    y = sps.sosfilt(lp, np.asarray(x, np.float64)).astype(np.float64)
    W = int(F0窗ms * SR / 1000)
    hop = F0跳
    n = 1 + max(0, (len(y) - W) // hop)
    if n <= 0:
        return np.zeros(0), np.zeros(0)
    τmin = max(2, int(SR / F0最高))
    τmax = min(W - 2, int(SR / F0最低))
    帧s = np.lib.stride_tricks.sliding_window_view(y, W)[: n * hop: hop][: n]
    f0 = np.zeros(n, np.float64)
    ásper = np.ones(n, np.float64)
    rmsv = np.zeros(n, np.float64)
    分块 = 2048
    ts = np.arange(τmax + 1)
    for s in range(0, n, 分块):
        e = min(n, s + 分块)
        fr = 帧s[s:e].astype(np.float64)
        fr -= fr.mean(axis=1, keepdims=True)          # 去均值；不加窗（YIN）
        m = e - s
        F = np.fft.rfft(fr, n=2 * W, axis=1)
        R = np.fft.irfft(F * np.conj(F), n=2 * W, axis=1)[:, :W]
        E = np.concatenate([np.zeros((m, 1)), np.cumsum(fr * fr, axis=1)], axis=1)
        总 = E[:, W:W + 1]
        e0 = E[:, W - ts]                              # Σ_{j<W-τ} x_j²
        e1 = 总 - E[:, ts]                             # Σ_{j≥τ} x_j²
        d = np.maximum(e0 + e1 - 2 * R[:, :τmax + 1], 0.0)
        d[:, 0] = 0.0
        cs = np.cumsum(d, axis=1)
        dd = np.ones_like(d)
        dd[:, 1:] = d[:, 1:] * ts[1:] / (cs[:, 1:] + 1e-12)   # CMNDF
        mid = dd[:, τmin:τmax]
        left = dd[:, τmin - 1:τmax - 1]
        right = dd[:, τmin + 1:τmax + 1]
        局部极小 = (mid < F0阈) & (mid <= left) & (mid <= right)
        有 = 局部极小.any(axis=1)
        首个 = np.argmax(局部极小, axis=1) + τmin
        全局 = np.argmin(mid, axis=1) + τmin
        cand = np.where(有, 首个, 全局)
        idx = np.arange(m)
        ásper[s:e] = dd[idx, cand]
        ak = np.clip(cand, 1, τmax - 1)
        a1 = dd[idx, ak - 1]; b1 = dd[idx, ak]; c1 = dd[idx, ak + 1]
        δ = np.clip(0.5 * (a1 - c1) / (a1 - 2 * b1 + c1 + 1e-12), -0.5, 0.5)
        滞后 = ak + δ
        f0[s:e] = np.where(滞后 > 0, SR / 滞后, 0.0)
        rmsv[s:e] = np.sqrt(np.mean(fr * fr, axis=1))
    门 = (rmsv > max(1e-4, 0.05 * float(rmsv.max()))) & (ásper < F0非周期阈)
    f0 = np.where(门, f0, 0.0)
    t = (np.arange(n) * hop + W / 2.0) / SR               # 窗中心
    return t, f0


def 事件(场, 场未适, f0t, f0):
    """场 [64, T]（1 ms）→ 事件（1 ms 定位）。与 S0 件逐行相同（先于读数固定）。"""
    import scipy.signal as sps
    T = 场.shape[1]
    fl = np.diff(np.log(场 + 1e-9), axis=1)
    flux = np.concatenate([[0.0], np.sum(np.maximum(fl, 0.0), axis=0)])
    高 = 场未适[:, 32:].mean(axis=1)
    低 = 场未适[:, :16].mean(axis=1)
    中 = 场[8:32, :].mean(axis=0)
    ev = []
    # ① 爆破：高频能量在 ≤5 ms 内跃升，且离开前 20 ms 的基线；1 ms 定位=跃升起点
    import scipy.ndimage as ndi
    基线 = np.roll(ndi.median_filter(高, size=21), 5)
    跃 = 高 / (基线 + 1e-9)
    峰s, _ = sps.find_peaks(跃, height=2.0, distance=5)
    for t in 峰s:
        ev.append(dict(type="爆破", t=round(float(t) / 1000.0, 4), 强度=round(float(跃[t]), 4)))
    # ② 摩擦：高频持续 ≥30 ms 且该段 F0 为 0（非周期），前 120 ms 内有爆破
    f0_1ms = np.interp(np.arange(T), f0t * 1000.0, f0) if len(f0t) > 1 else np.zeros(T)
    段存 = [False] * T
    for t in range(T):
        if 高[t] > 1.2 * 低[t] and 高[t] > np.median(高) + 1.0 * np.std(高) and f0_1ms[t] == 0:
            段存[t] = True
    t = 0
    while t < T:
        if 段存[t]:
            s = t
            while t < T and 段存[t]:
                t += 1
            if t - s >= 30:
                有爆 = any(e["type"] == "爆破" and 0 <= s - int(round(e["t"] * 1000)) <= 120 for e in ev)
                ev.append(dict(type="摩擦", t=round(float(s) / 1000.0, 4),
                               强度=round(float(高[s]), 4), 持续ms=t - s, 前有爆破=bool(有爆)))
        else:
            t += 1
    # ③ 浊音起：爆破后 300 ms 内，带 8–24（≈250–1000 Hz）能量最陡上升（1 ms 定位）
    for e0 in [e for e in ev if e["type"] == "爆破"]:
        t0 = int(round(e0["t"] * 1000)); t1 = min(T, t0 + 300)
        if t1 <= t0 + 8:
            continue
        低中 = 场未适[t0:t1, 8:24].mean(axis=1).astype(np.float64)
        差 = np.diff(低中)
        k = int(np.argmax(差[3:])) + 3 if len(差) > 3 else int(np.argmax(差))
        ev.append(dict(type="浊音起", t=round((t0 + k) / 1000.0, 4), 强度=round(float(差[k]), 6)))
    # 浊音止：F0 轨迹由浊→清（保留，S0 不判）
    if len(f0t) > 3:
        v = f0 > 0
        for i in range(1, len(v)):
            if not v[i] and v[i - 1]:
                ev.append(dict(type="浊音止", t=round(float(f0t[i]), 4), 强度=1.0))
    # ④ 元音核：低中频平滑能量的显著局部极大（≥100 ms 间隔、prominence 高）
    包 = np.convolve(中, np.ones(25) / 25, mode="same")
    峰s2, _ = sps.find_peaks(包, distance=100, height=np.median(包) + np.std(包),
                             prominence=2.0 * np.std(包))
    for t in 峰s2:
        ev.append(dict(type="元音核", t=round(float(t) / 1000.0, 4), 强度=round(float(包[t]), 4)))
    # 同类/异类最小间隔：异类 5 ms、同类 8 ms（保留更强或更早）
    ev.sort(key=lambda e: (e["t"], -e["强度"]))
    滤 = []
    for e in ev:
        ok = True
        for p in 滤:
            gap = (e["t"] - p["t"]) * 1000.0
            if e["type"] == p["type"] and 0 <= gap < 8:
                ok = False; break
            if e["type"] != p["type"] and 0 <= gap < 5:
                ok = False; break
        if ok:
            滤.append(e)
    return 滤


def 分析(路, 变体="V3"):
    x = 读wav(路)
    t0 = time.time()
    样本, fs = 滤波组(x)
    包 = 包络(样本)                                  # (64, n)
    n1 = len(x) // 包络步
    包1 = 包[:, : n1 * 包络步].reshape(包.shape[0], n1, 包络步).mean(axis=2)   # 1 ms
    压缩 = np.power(包1, 0.4)
    场, h0 = 适应(压缩, 变体)                        # (64, T) 1 ms
    场未适 = 压缩.T
    f0t, f0 = F0轨迹(x)
    ev = 事件(场, 场未适, f0t, f0)
    T = 场.shape[1]
    场5 = 场[:, : T // 5 * 5].reshape(64, T // 5, 5).mean(axis=2).T.astype(np.float32)
    sha = hashlib.sha256(np.ascontiguousarray(场).tobytes()).hexdigest()
    return dict(采样率=SR, 频带数=频带数, 中心频率=[round(float(v), 1) for v in fs],
                场=场.T.astype(np.float32), 场未适=压缩.T.astype(np.float32), 场5ms=场5, 事件=ev,
                F0=dict(t=[round(float(v), 4) for v in f0t], f0=[round(float(v), 2) for v in f0]),
                时长秒=round(len(x) / SR, 3), sha256=sha, 适应变体=变体,
                实时率=round((time.time() - t0) / max(len(x) / SR, 1e-9), 3),
                h0=[round(float(v), 6) for v in h0[:4]])


def 主(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="meeting69 耳蜗前端 v0.2（S0.1）")
    子 = ap.add_subparsers(dest="命令", required=True)
    p = 子.add_parser("分析"); p.add_argument("wav"); p.add_argument("--变体", default="V3", choices=适应变体集)
    p.add_argument("--出", default=None)
    参 = ap.parse_args(argv)
    r = 分析(参.wav, 变体=参.变体)
    精简 = dict(时长秒=r["时长秒"], 频带数=r["频带数"], 事件数=len(r["事件"]),
                事件=r["事件"][:20], sha256=r["sha256"], 实时率=r["实时率"],
                适应变体=r["适应变体"], F0浊帧=sum(1 for v in r["F0"]["f0"] if v > 0),
                事件率=round(len(r["事件"]) / max(r["时长秒"], 1e-9), 2))
    print(json.dumps(精简, ensure_ascii=False, indent=1))
    if 参.出:
        Path(参.出).write_text(json.dumps(r, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(主())

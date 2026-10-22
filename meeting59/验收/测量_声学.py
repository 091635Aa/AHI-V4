# -*- coding: utf-8 -*-
"""meeting59 独立验收 · 声学测量库（qa 自建）

口径：纯 numpy/scipy 自写；**不 import** lead 的 `实施/拟合_参考声线.py`、
`实施/韵律_人类.py`、`实施/合成_测试音频.py`，也不 import `声带.py`（只借其常量语义，
本文件自带 FS=16000 / 带边界 / 韵母表数值仅用于自测），保证测量与实现解耦。

实现项（对应简报 §4）：
  F0        : 自相关法，40ms/10ms 帧，70–500Hz，抛物线插值，倍频离群剔除
  7 带包络 : 32ms/10ms，7 带（0-250/250-500/500-1k/1k-2k/2k-3k/3k-4k/4k-8k）log 能量，
             逐帧减总能量（增益不变；先减均值再余弦 = R1 口径，见 R1_立场 §3 M-B）
  共振峰    : 倒谱 lifter(30) 平滑对数谱的**局部峰**（按突出度取），F1/F2/F3 中位 + MAD
  能量/停顿 : 20ms/10ms 能量 dB；自适应阈值；静音段 ≥120ms 记为停顿；
             长停顿（≥300ms）视作句界（本测量内部的显式假设）
  半音距    : 12*log2(f1/f2)
  声纹代理  : 17 维 = 7 带均值 + 7 带帧间 std + F0 中位(st) + F0 std(st) + 谱倾斜
  回放互相关: FFT 归一化最大互相关（Cauchy-Schwarz 归一，≤1）
  确定性    : 文件 sha256、两趟同值

CLI:
  python 测量_声学.py a.wav b.wav ... [--json out.json]
  python 测量_声学.py --目录 某目录 [--json out.json]
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

# ── 常量（与 声带.py 语义一致，但不 import） ──────────────────────────
FS默认 = 16000
F0范围 = (70.0, 500.0)
带边界 = (0.0, 250.0, 500.0, 1000.0, 2000.0, 3000.0, 4000.0, 8000.0)
带名 = ("0-250", "250-500", "500-1k", "1k-2k", "2k-3k", "3k-4k", "4k-8k")
倒谱lifter = 30
禁import = ("torch", "transformers", "rwkv", "pyttsx3", "onnxruntime", "tts",
           "tensorflow", "jax", "paddle", "whisper", "funasr", "rvc", "fairseq",
           "espeak", "edge_tts", "sovits", "vits", "hifigan", "vocoder")
视频后缀 = (".mp4", ".mov", ".mkv", ".avi", ".webm", ".flv", ".wmv", ".m4v")


# ══════════════════════════════════════════════════════════════════
# 基础 IO / 分帧
# ══════════════════════════════════════════════════════════════════
def 读wav(路径: str):
    """读 wav → (sr, 单声道 float64 ∈[-1,1])，纯 scipy。"""
    from scipy.io import wavfile

    sr, x = wavfile.read(路径)
    x = np.asarray(x)
    if x.dtype == np.uint8:
        x = (x.astype(np.float64) - 128.0) / 128.0
    elif x.dtype == np.int16:
        x = x.astype(np.float64) / 32768.0
    elif x.dtype == np.int32:
        x = x.astype(np.float64) / 2147483648.0
    elif np.issubdtype(x.dtype, np.floating):
        x = x.astype(np.float64)
    else:  # pragma: no cover
        raise TypeError("不支持的样本类型: %s" % x.dtype)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return int(sr), np.ascontiguousarray(x, dtype=np.float64)


def sha256文件(路径: str) -> str:
    h = hashlib.sha256()
    with open(路径, "rb") as f:
        for 块 in iter(lambda: f.read(1 << 20), b""):
            h.update(块)
    return h.hexdigest()


def 分帧矩阵(x, sr, 窗秒, 跳秒, 加窗=True):
    L = max(2, int(round(窗秒 * sr)))
    H = max(1, int(round(跳秒 * sr)))
    if len(x) < L:
        x = np.concatenate([x, np.zeros(L - len(x))])
    n = 1 + (len(x) - L) // H
    帧 = x[np.arange(L)[None, :] + H * np.arange(n)[:, None]]
    if 加窗:
        帧 = 帧 * np.hanning(L)[None, :]
    return 帧, L, H


def 能量包络(x, sr=FS默认, 帧秒=0.020, 跳秒=0.010):
    """返回 (帧中心秒, 帧能量 dB)。"""
    帧, L, H = 分帧矩阵(x, sr, 帧秒, 跳秒)
    e = np.mean(帧 ** 2, axis=1)
    edb = 10.0 * np.log10(e + 1e-20)
    t = (np.arange(len(e)) * H + L / 2.0) / sr
    return t, edb


# ══════════════════════════════════════════════════════════════════
# F0：自相关
# ══════════════════════════════════════════════════════════════════
def F0轨迹(x, sr=FS默认, f0范围=F0范围, 窗秒=0.040, 跳秒=0.010,
           自相关阈=0.30, 能量相对阈dB=30.0):
    帧, L, H = 分帧矩阵(x, sr, 窗秒, 跳秒)
    帧 = 帧 - 帧.mean(axis=1, keepdims=True)
    nfft = 1 << int(math.ceil(math.log2(max(L, 512))))
    S = np.fft.rfft(帧, n=nfft, axis=1)
    r = np.fft.irfft(np.abs(S) ** 2, n=nfft, axis=1)[:, :L]
    r0 = r[:, 0].copy()
    均方 = r0 / float(L)
    rn = r / np.maximum(r0, 1e-18)[:, None]
    lagmin = max(2, int(math.floor(sr / f0范围[1])))
    lagmax = min(L - 2, int(math.ceil(sr / f0范围[0])))
    段 = rn[:, lagmin:lagmax + 1]
    n = 段.shape[0]
    中 = 段[:, 1:-1]
    左 = 段[:, :-2]
    右 = 段[:, 2:]
    候选 = (中 >= 左) & (中 > 右) & (中 > 0.0)
    峰 = np.max(段, axis=1)
    f0 = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        c = np.nonzero(候选[i])[0]
        if c.size == 0:
            continue
        vals = 段[i, c]
        m = float(vals.max())
        if m < max(自相关阈, 0.85 * float(峰[i])):
            continue
        ok = c[vals >= 0.85 * m]           # 取最靠前（最高频）的近似峰，抑次谐波
        j = int(ok[0]) + 1                 # 段内下标 → 全局 lag 偏移 +1
        lag = float(lagmin + j)
        a = float(段[i, j - 1]); b = float(段[i, j]); cc = float(段[i, j + 1])
        den = a - 2.0 * b + cc
        if abs(den) > 1e-12:
            lag += 0.5 * (a - cc) / den
        if lag > 0:
            f0[i] = sr / lag
    e_db = 10.0 * np.log10(均方 + 1e-20)
    有声 = np.isfinite(f0) & (峰 >= 自相关阈) & (e_db > e_db.max() - 能量相对阈dB)
    t = (np.arange(n) * H + L / 2.0) / sr
    return {"时间": t, "f0": f0, "有声": 有声, "自相关峰": 峰, "帧能量dB": e_db}


def F0统计(轨迹, 倍频离群st=12.0):
    f0 = np.asarray(轨迹["f0"], dtype=np.float64)
    有声 = np.asarray(轨迹["有声"], dtype=bool) & np.isfinite(f0)
    vals = f0[有声]
    out = {"帧数": int(len(f0)), "有声帧": int(vals.size),
           "有声帧比例": float(vals.size / max(len(f0), 1)),
           "中位Hz": float("nan"), "中位st": float("nan"),
           "p5st": float("nan"), "p95st": float("nan"), "跨度st": float("nan"),
           "std_st": float("nan"), "倍频离群帧比例": float("nan"),
           "中位自相关峰": float("nan")}
    if vals.size < 3:
        return out
    中位 = float(np.median(vals))
    st = 12.0 * np.log2(vals / 55.0)
    中位st = 12.0 * math.log2(中位 / 55.0)
    离 = np.abs(st - float(np.median(st))) > 倍频离群st
    st2 = st[~离] if (~离).sum() >= 3 else st
    p5, p95 = np.percentile(st2, [5.0, 95.0])
    out.update({"中位Hz": 中位, "中位st": float(中位st),
                "p5st": float(p5), "p95st": float(p95), "跨度st": float(p95 - p5),
                "std_st": float(np.std(st2)),
                "倍频离群帧比例": float(离.mean()),
                "中位自相关峰": float(np.median(np.asarray(轨迹["自相关峰"])[有声]))})
    return out


def 半音距(f1, f2):
    f1 = float(f1); f2 = float(f2)
    if not (math.isfinite(f1) and math.isfinite(f2)) or f1 <= 0 or f2 <= 0:
        return float("nan")
    return 12.0 * math.log2(f1 / f2)


# ══════════════════════════════════════════════════════════════════
# 7 带 log 包络 / 谱倾斜 / 17 维声纹代理
# ══════════════════════════════════════════════════════════════════
def 七带包络(x, sr=FS默认, 窗秒=0.032, 跳秒=0.010, 频带=带边界, 能量相对阈dB=35.0):
    帧, L, H = 分帧矩阵(x, sr, 窗秒, 跳秒)
    帧 = 帧 - 帧.mean(axis=1, keepdims=True)
    能量dB = 10.0 * np.log10(np.mean(帧 ** 2, axis=1) + 1e-20)
    选中 = 能量dB > (能量dB.max() - 能量相对阈dB)
    if 选中.sum() < 2:
        选中 = 能量dB >= np.percentile(能量dB, 50.0)
    nfft = 1 << int(math.ceil(math.log2(max(L, 512))))
    P = np.abs(np.fft.rfft(帧, n=nfft, axis=1)) ** 2
    freqs = np.fft.rfftfreq(nfft, 1.0 / sr)
    nb = len(频带) - 1
    带能 = np.empty((P.shape[0], nb), dtype=np.float64)
    for i in range(nb):
        lo, hi = 频带[i], 频带[i + 1]
        m = (freqs >= lo) & (freqs <= hi) if i == nb - 1 else (freqs >= lo) & (freqs < hi)
        带能[:, i] = P[:, m].mean(axis=1) if m.any() else 1e-20
    dB = 10.0 * np.log10(带能 + 1e-20)
    相对 = dB - dB.mean(axis=1, keepdims=True)      # 逐帧减总能量 ⇒ 增益不变
    带均值 = 相对[选中].mean(axis=0)
    带std = 相对[选中].std(axis=0)
    # R1 口径（e_i/Σe 后取 log）与上式只差每帧一个常数；R1 的带余弦先减均值，
    # 故两者余弦完全一致（见 R1_立场 §3 M-B）。
    ltas = 10.0 * np.log10(P[选中].mean(axis=0) + 1e-20)
    mk = (freqs >= 250.0) & (freqs <= 6000.0)
    if mk.sum() >= 8:
        倾斜 = float(np.polyfit(np.log2(freqs[mk]), ltas[mk], 1)[0])
    else:
        倾斜 = float("nan")
    中心 = np.array([math.sqrt(max(频带[i], 1.0) * 频带[i + 1]) for i in range(nb)])
    return {"带均值dB": 带均值, "带stddB": 带std, "倾斜dB每倍频程": 倾斜,
            "带中心Hz": 中心, "选中帧": int(选中.sum()), "总帧": int(len(选中))}


def 余弦(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return float("nan")
    na = float(np.linalg.norm(a)); nb = float(np.linalg.norm(b))
    if na < 1e-9 or nb < 1e-9:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def 带余弦(a, b):
    """R1 口径：先减各自均值再余弦（对增益/通道常数不变）。"""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return float("nan")
    return 余弦(a - a.mean(), b - b.mean())


def 七带包络LTA(x, sr=FS默认, 窗秒=0.032, 跳秒=0.010, 频带=带边界):
    """**对照口径**：log-of-mean 的 7 带长时平均谱（= lead `声学工具.带包络` 的实现口径）。

    与 `七带包络`（R1 口径：逐帧相对 dB → 帧间平均）不同：本函数先对功率跨帧取平均再取 log。
    两者都增益不变，但帧加权不同；4–8k 带可差 20–30 dB（见报告 §2.2b）。
    """
    帧, L, H = 分帧矩阵(x, sr, 窗秒, 跳秒, 加窗=False)
    均 = np.sqrt(np.mean(帧 ** 2, axis=1) + 1e-30)
    噪门 = max(1e-4, 0.20 * float(np.percentile(均, 85)))
    选中 = 均 >= 噪门
    if 选中.sum() < 2:
        选中 = 均 >= np.percentile(均, 50.0)
    帧w, L2, _ = 分帧矩阵(x, sr, 窗秒, 跳秒, 加窗=True)
    nfft = 1 << int(math.ceil(math.log2(max(L2, 512))))
    P = np.abs(np.fft.rfft(帧w, n=nfft, axis=1)) ** 2
    freqs = np.fft.rfftfreq(nfft, 1.0 / sr)
    nb = len(频带) - 1
    累积 = np.zeros(nb)
    for i in range(nb):
        lo, hi = 频带[i], 频带[i + 1]
        m = (freqs >= lo) & (freqs <= hi) if i == nb - 1 else (freqs >= lo) & (freqs < hi)
        累积[i] = float(np.mean(P[选中][:, m])) if m.any() else 1e-20
    v = 10.0 * np.log10(累积 + 1e-20)
    return {"LTA_dB": v - v.mean(), "选中帧": int(选中.sum()), "总帧": int(len(均))}


def 声纹17(包络, f0统):
    return np.concatenate([
        np.asarray(包络["带均值dB"], dtype=np.float64),
        np.asarray(包络["带stddB"], dtype=np.float64),
        np.array([f0统["中位st"], f0统["std_st"], 包络["倾斜dB每倍频程"]], dtype=np.float64),
    ])


# ══════════════════════════════════════════════════════════════════
# 倒谱 lifter(30) 共振峰（局部峰 + 突出度）
# ══════════════════════════════════════════════════════════════════
def 倒谱共振峰(x, sr=FS默认, 窗秒=0.025, 跳秒=0.010, lifter=倒谱lifter,
               能量相对阈dB=30.0, 上限=4200.0, 突出度dB=2.0):
    """倒谱 lifter(30) 平滑谱包络的局部峰，按突出度归 F1/F2/F3。

    取峰规则（冻结）：F1=区域[200,1300]内突出度最大的局部峰；
    F2=区域[max(F1+250,500),3400]内突出度最大的局部峰；
    F3=区域[max(F2+250,1400),4200]内突出度最大的局部峰。
    说明：该合成器的级联谐振器 + 声源 -12dB/oct 使高共振峰可能只有 -40…-50dB，
    局部峰可测性用「有效帧比例」如实报告；有效帧不足时中位不可靠。
    """
    from scipy.signal import find_peaks

    帧, L, H = 分帧矩阵(x, sr, 窗秒, 跳秒)
    帧 = 帧 - 帧.mean(axis=1, keepdims=True)
    能量dB = 10.0 * np.log10(np.mean(帧 ** 2, axis=1) + 1e-20)
    选中 = 能量dB > (能量dB.max() - 能量相对阈dB)
    if 选中.sum() < 2:
        选中 = 能量dB >= np.percentile(能量dB, 50.0)
    nfft = 1 << int(math.ceil(math.log2(max(L, 512))))
    mag = np.abs(np.fft.rfft(帧, n=nfft, axis=1))
    floor = mag.max(axis=1, keepdims=True) * 1e-3 + 1e-12     # -60dB 底
    logmag = 20.0 * np.log10(np.maximum(mag, floor))           # dB 域（尺度不影响峰位）
    ceps = np.fft.irfft(logmag / 20.0, n=nfft, axis=1)         # 实倒谱（ln 域）
    lift = ceps.copy()
    if 1 <= lifter < nfft // 2 - 1:
        lift[:, lifter + 1: nfft - lifter] = 0.0
    env = np.real(np.fft.rfft(lift, n=nfft, axis=1)) * 20.0   # dB 域平滑包络
    freqs = np.fft.rfftfreq(nfft, 1.0 / sr)
    mm = (freqs >= 150.0) & (freqs <= 上限)
    em = env[:, mm]
    fm = freqs[mm]
    F1s, F2s, F3s = [], [], []
    for i in np.nonzero(选中)[0]:
        pk, 属 = find_peaks(em[i], prominence=突出度dB, distance=2)
        if len(pk) == 0:
            continue
        pf = fm[pk]
        pp = np.asarray(属["prominences"], dtype=np.float64)

        def 取(lo, hi):
            m = (pf >= lo) & (pf <= hi)
            if not m.any():
                return float("nan")
            cand = np.nonzero(m)[0]
            return float(pf[cand[int(np.argmax(pp[cand]))]])

        f1 = 取(200.0, 1300.0)
        if not math.isfinite(f1):
            continue
        f2 = 取(max(f1 + 250.0, 500.0), 3400.0)
        f3 = 取((f2 + 250.0) if math.isfinite(f2) else (f1 + 500.0), 4200.0)
        F1s.append(f1)
        if math.isfinite(f2):
            F2s.append(f2)
        if math.isfinite(f3):
            F3s.append(f3)

    def 统计(v):
        v = np.asarray(v, dtype=np.float64)
        if v.size == 0:
            return {"中位": float("nan"), "MAD": float("nan"), "有效帧": 0}
        m = float(np.median(v))
        return {"中位": m, "MAD": float(np.median(np.abs(v - m))), "有效帧": int(v.size)}

    s1 = 统计(F1s); s2 = 统计(F2s); s3 = 统计(F3s)
    选 = int(选中.sum())
    return {"F1中位": s1["中位"], "F2中位": s2["中位"], "F3中位": s3["中位"],
            "F1MAD": s1["MAD"], "F2MAD": s2["MAD"], "F3MAD": s3["MAD"],
            "有效帧": s1["有效帧"], "F2有效帧": s2["有效帧"], "F3有效帧": s3["有效帧"],
            "选中帧": 选,
            "F1有效比": float(s1["有效帧"] / max(选, 1)),
            "F2有效比": float(s2["有效帧"] / max(选, 1)),
            "F3有效比": float(s3["有效帧"] / max(选, 1))}


# ══════════════════════════════════════════════════════════════════
# 能量 / 停顿 / 句界 / 句末 / 像人代理
# ══════════════════════════════════════════════════════════════════
def 停顿检测(x, sr=FS默认, 帧秒=0.020, 跳秒=0.010, 最短秒=0.120, 句界秒=0.300):
    t, edb = 能量包络(x, sr, 帧秒, 跳秒)
    n = len(edb)
    噪底 = float(np.percentile(edb, 5.0))
    峰值 = float(np.percentile(edb, 95.0))
    最大 = float(edb.max())
    if 峰值 - 噪底 < 8.0:                    # 近乎连续，无真正静音
        阈值 = 最大 - 40.0
    else:
        阈值 = 噪底 + max(10.0, 0.30 * (峰值 - 噪底))
        阈值 = min(阈值, 最大 - 12.0)
    from scipy.ndimage import median_filter

    有声 = median_filter((edb > 阈值).astype(np.uint8), size=3).astype(bool)
    有idx = np.nonzero(有声)[0]
    out = {"阈值dB": float(阈值), "噪底dB": 噪底, "峰值dB": 峰值,
           "停顿列表": [], "句间停顿": [], "句内停顿": [],
           "最长停顿秒": float("nan"), "停顿总秒": 0.0,
           "语音段": [], "句级语音段": [], "语音段数": 0, "句数": 0,
           "有声帧": int(有idx.size), "总帧": int(n)}
    if 有idx.size == 0:
        return out
    首, 尾 = int(有idx[0]), int(有idx[-1])
    停顿 = []
    i = 首
    while i <= 尾:
        if 有声[i]:
            i += 1
            continue
        j = i
        while j <= 尾 and not 有声[j]:
            j += 1
        段长 = (j - i) * 跳秒
        if 段长 >= 最短秒:
            停顿.append({"开始秒": float((i * 跳秒)), "结束秒": float(j * 跳秒),
                         "时长秒": float(段长)})
        i = j

    def 切段(分隔):
        segs = []
        起 = float(首 * 跳秒)
        for p in 分隔:
            segs.append({"开始秒": 起, "结束秒": float(p["开始秒"])})
            起 = float(p["结束秒"])
        segs.append({"开始秒": 起, "结束秒": float(尾 * 跳秒)})
        return [s for s in segs if s["结束秒"] - s["开始秒"] >= 0.060] or segs

    语音段 = 切段(停顿)
    句间 = [p for p in 停顿 if p["时长秒"] >= 句界秒]
    句内 = [p for p in 停顿 if 最短秒 <= p["时长秒"] < 句界秒]
    句级 = 切段(句间)
    out.update({"停顿列表": 停顿, "句间停顿": 句间, "句内停顿": 句内,
                "最长停顿秒": float(max(p["时长秒"] for p in 停顿)) if 停顿 else 0.0,
                "停顿总秒": float(sum(p["时长秒"] for p in 停顿)),
                "语音段": 语音段, "语音段数": len(语音段),
                "句级语音段": 句级, "句数": len(句级)})
    return out


def 音节峰(t, edb, 平滑帧=5, 最小间隔秒=0.075, 突出度dB=2.0):
    from scipy.ndimage import uniform_filter1d
    from scipy.signal import find_peaks

    if len(edb) < 5:
        return np.array([], dtype=np.float64)
    e = uniform_filter1d(edb, size=平滑帧)
    距 = max(1, int(round(最小间隔秒 / max(t[1] - t[0], 1e-9))))
    idx, 属性 = find_peaks(e, distance=距, prominence=突出度dB)
    return t[idx]


def 句末分析(停顿信息, 峰时间, 轨迹):
    """节律代理（句级）：句末音节时长比、句末降调。不依赖人工标注，方法显式：
    句末音节时长 ≈ 句尾 − 最后两个声核中点（音节边界≈相邻声核中点），
    除以句内中位声核间隔；句末降调 = 句末 150ms F0 中位 − 句内 F0 中位（半音）。
    """
    句级 = 停顿信息.get("句级语音段") or 停顿信息["语音段"]
    结果 = {"每句": [], "句末时长比": float("nan"), "句末时长比最大": float("nan"),
            "句末降调st": float("nan"), "句数": len(句级)}
    f0 = np.asarray(轨迹["f0"]); tt = np.asarray(轨迹["时间"])
    有声 = np.asarray(轨迹["有声"]) & np.isfinite(f0)
    比们 = []; 降们 = []
    for s in 句级:
        起, 终 = float(s["开始秒"]), float(s["结束秒"])
        if 终 - 起 < 0.150:
            continue
        ps = np.asarray([p for p in 峰时间 if 起 <= p <= 终])
        m = 有声 & (tt >= 起) & (tt <= 终)
        f0句 = f0[m]
        句内st = float(np.median(12.0 * np.log2(f0句 / 55.0))) if f0句.size >= 3 else float("nan")
        条 = {"开始秒": 起, "结束秒": 终, "时长秒": 终 - 起,
              "音节峰数": int(ps.size), "句内F0st": 句内st,
              "句内停顿数": int(sum(1 for p in 停顿信息["停顿列表"]
                                  if 起 <= p["开始秒"] and p["结束秒"] <= 终))}
        if ps.size >= 3:
            间隔 = np.diff(ps)
            间隔 = 间隔[间隔 > 0.050]
            if 间隔.size >= 1:
                中位间隔 = float(np.median(间隔))
                末时长 = 终 - 0.5 * (ps[-1] + ps[-2])
                条.update({"中位声核间隔秒": 中位间隔,
                          "句末音节时长秒": float(末时长),
                          "句末时长比": float(末时长 / 中位间隔) if 中位间隔 > 0 else float("nan"),
                          "末间隔秒": float(ps[-1] - ps[-2])})
                比们.append(条["句末时长比"])
        if f0句.size >= 3:
            m末 = m & (tt >= 终 - 0.150)
            f0末 = f0[m末]
            if f0末.size >= 2:
                st末 = float(np.median(12.0 * np.log2(f0末 / 55.0)))
                if math.isfinite(句内st):
                    条["句末降调st"] = st末 - 句内st
                    降们.append(条["句末降调st"])
        结果["每句"].append(条)
    if 比们:
        结果["句末时长比"] = float(np.median(比们))
        结果["句末时长比最大"] = float(max(比们))
    if 降们:
        结果["句末降调st"] = float(np.median(降们))
    return 结果


def F0句界重置(停顿信息, 轨迹):
    """在每个长停顿(句界)前后各取 200ms 有声帧，比较 F0 中位（半音）。"""
    f0 = np.asarray(轨迹["f0"]); tt = np.asarray(轨迹["时间"])
    有声 = np.asarray(轨迹["有声"]) & np.isfinite(f0)
    结果 = {"每界": [], "重置界数": 0, "重置中位st": float("nan"), "重置比例": float("nan")}
    deltas = []
    for p in 停顿信息["句间停顿"]:
        起, 终 = float(p["开始秒"]), float(p["结束秒"])
        m前 = 有声 & (tt >= 起 - 0.200) & (tt <= 起)
        m后 = 有声 & (tt >= 终) & (tt <= 终 + 0.200)
        if m前.sum() < 3 or m后.sum() < 3:
            结果["每界"].append({"开始秒": 起, "结束秒": 终, "Δst": float("nan")})
            continue
        前 = float(np.median(12.0 * np.log2(f0[m前] / 55.0)))
        后 = float(np.median(12.0 * np.log2(f0[m后] / 55.0)))
        d = 后 - 前
        deltas.append(d)
        结果["每界"].append({"开始秒": 起, "结束秒": 终, "前st": 前, "后st": 后, "Δst": d,
                             "重置": bool(d >= 1.0)})
    if deltas:
        结果["重置界数"] = int(sum(1 for d in deltas if d >= 1.0))
        结果["重置中位st"] = float(np.median(deltas))
        结果["重置比例"] = float(np.mean([d >= 1.0 for d in deltas]))
    return 结果


def 能量调制(t, edb, 停顿信息=None):
    """能量包络调制：全帧与语音帧两个口径都报（语音帧剔除数字静音）。"""
    e = np.asarray(edb, dtype=np.float64)
    阈值 = e.max() - 40.0
    if 停顿信息 is not None and math.isfinite(停顿信息.get("阈值dB", float("nan"))):
        阈值 = float(停顿信息["阈值dB"])
    语音 = e > 阈值
    ev = e[语音] if 语音.sum() >= 2 else e
    return {"std_dB_全帧": float(np.std(e)),
            "std_dB_语音帧": float(np.std(ev)),
            "范围_dB_语音帧": float(np.percentile(ev, 95) - np.percentile(ev, 5)),
            "中位_dB_语音帧": float(np.median(ev)),
            "语音帧数": int(语音.sum()), "总帧数": int(len(e)),
            "是否非恒定": bool(np.std(ev) > 0.5)}


# ══════════════════════════════════════════════════════════════════
# 回放互相关
# ══════════════════════════════════════════════════════════════════
def 最大规整互相关(a, b, 截秒=None, sr=FS默认):
    from scipy.signal import fftconvolve

    a = np.asarray(a, dtype=np.float64).copy()
    b = np.asarray(b, dtype=np.float64).copy()
    if 截秒 is not None and sr:
        a = a[: int(截秒 * sr)]
        b = b[: int(截秒 * sr)]
    a -= a.mean(); b -= b.mean()
    den = math.sqrt(float(np.dot(a, a)) * float(np.dot(b, b)))
    if den <= 0.0:
        return 0.0
    c = fftconvolve(a, b[::-1], mode="full")
    return float(np.abs(c).max() / den)


# ══════════════════════════════════════════════════════════════════
# 汇总
# ══════════════════════════════════════════════════════════════════
def 汇总波形(x, sr, 路径=None, 名称=None):
    """对内存波形做全量测量（供阶段B把多段参考拼接后测量）。"""
    x = np.asarray(x, dtype=np.float64)
    时长 = len(x) / float(sr)
    轨迹 = F0轨迹(x, sr)
    f0统 = F0统计(轨迹)
    包络 = 七带包络(x, sr)
    共振 = 倒谱共振峰(x, sr)
    停顿 = 停顿检测(x, sr)
    t, edb = 能量包络(x, sr)
    峰 = 音节峰(t, edb)
    句末 = 句末分析(停顿, 峰, 轨迹)
    重置 = F0句界重置(停顿, 轨迹)
    调制 = 能量调制(t, edb, 停顿)
    声纹 = 声纹17(包络, f0统)
    读数 = {
        "路径": os.path.abspath(路径) if 路径 else "内存",
        "文件名": 名称 or (os.path.basename(路径) if 路径 else "内存"),
        "sha256": sha256文件(路径) if (路径 and os.path.isfile(路径)) else None,
        "采样率": int(sr), "时长秒": float(时长),
        "F0": f0统, "七带": {
            "带名": list(带名),
            "带均值dB": [float(v) for v in 包络["带均值dB"]],
            "带stddB": [float(v) for v in 包络["带stddB"]],
            "倾斜dB每倍频程": float(包络["倾斜dB每倍频程"]),
            "选中帧": int(包络["选中帧"]), "总帧": int(包络["总帧"]),
        },
        "共振峰": 共振,
        "停顿": {k: 停顿[k] for k in ("阈值dB", "噪底dB", "峰值dB", "最长停顿秒",
                                    "停顿总秒", "语音段数", "句数", "有声帧", "总帧")},
        "停顿列表": 停顿["停顿列表"],
        "句间停顿": 停顿["句间停顿"],
        "句内停顿": 停顿["句内停顿"],
        "句末": 句末,
        "F0句界": 重置,
        "能量调制": 调制,
        "声纹17": [float(v) for v in 声纹],
    }
    return {"读数": 读数, "内部": {"包络均值": 包络["带均值dB"].copy(),
                                "声纹17": 声纹.copy(), "轨迹": 轨迹, "能量dB": edb}}


def 汇总(路径, sr=None, 需互相关=False, 最大秒=None, 计时=True):
    开始 = time.time()
    sr, x = 读wav(路径)
    if 最大秒 is not None:
        x = x[: int(最大秒 * sr)]
    汇 = 汇总波形(x, sr, 路径=路径)
    汇["读数"]["计时秒"] = float(time.time() - 开始) if 计时 else None
    return 汇


def 可序列化(汇):
    return 汇["读数"]


def 打印读数(读数):
    f0 = 读数["F0"]; 峰 = 读数["共振峰"]; 停 = 读数["停顿"]
    print("── %s  (%.2fs, sha256=%s)" % (读数["文件名"], 读数["时长秒"], 读数["sha256"][:16]))
    print("   F0: 中位 %.2fHz = %.2fst  跨度 %.2fst (p5=%.2f p95=%.2f)  std=%.2fst  有声帧 %.1f%%"
          % (f0["中位Hz"], f0["中位st"], f0["跨度st"], f0["p5st"], f0["p95st"],
             f0["std_st"], 100.0 * f0["有声帧比例"]))
    print("   7带: " + " ".join("%.1f" % v for v in 读数["七带"]["带均值dB"]) +
          "  倾斜 %.2f dB/oct" % 读数["七带"]["倾斜dB每倍频程"])
    print("   共振峰: F1 %.0f±%.0f(%.0f%%)  F2 %.0f±%.0f(%.0f%%)  F3 %.0f±%.0f  Hz"
          % (峰["F1中位"], 峰["F1MAD"], 100 * 峰["F1有效比"],
             峰["F2中位"], 峰["F2MAD"], 100 * 峰["F2有效比"],
             峰["F3中位"], 峰["F3MAD"]))
    print("   停顿: %d 段(≥120ms) 最长 %.2fs  句界 %d / 句内 %d  语音段 %d 句数 %d"
          % (len(读数["停顿列表"]), 停["最长停顿秒"],
             len(读数["句间停顿"]), len(读数["句内停顿"]), 停["语音段数"], 停["句数"]))
    print("   像人: 句末时长比 %.2f  句末降调 %.2fst  F0句界重置 %d/%d 界  能量std(语音) %.1fdB"
          % (读数["句末"].get("句末时长比", float("nan")),
             读数["句末"].get("句末降调st", float("nan")),
             读数["F0句界"]["重置界数"], len(读数["F0句界"]["每界"]),
             读数["能量调制"]["std_dB_语音帧"]))


def main(argv=None):
    ap = argparse.ArgumentParser(description="meeting59 qa 独立声学测量")
    ap.add_argument("文件", nargs="*", help="wav 路径")
    ap.add_argument("--目录", default=None, help="对目录下 *.wav 批量测")
    ap.add_argument("--json", default=None, help="结果 JSON 输出路径")
    ap.add_argument("--最大秒", type=float, default=None)
    args = ap.parse_args(argv)
    列表 = list(args.文件)
    if args.目录:
        列表 += sorted(glob.glob(os.path.join(args.目录, "*.wav")))
    if not 列表:
        ap.print_help()
        return 2
    结果 = []
    for p in 列表:
        try:
            r = 汇总(p, 最大秒=args.最大秒)
            打印读数(r["读数"])
            结果.append(可序列化(r))
        except Exception as e:  # noqa: BLE001
            print("!! %s 失败: %s" % (p, e))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(结果, f, ensure_ascii=False, indent=1)
        print("→ %s" % args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())

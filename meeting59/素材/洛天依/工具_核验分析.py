# -*- coding: utf-8 -*-
"""meeting59 / refs-web：第三方参考音频声学核验（纯 numpy/scipy，无神经网络）
用法:
  python 工具_核验分析.py <in.wav> [--json out.json]
输出: 帧级 CSV 旁路(同名 .frames.csv)、终端报告、可选 JSON
口径: [实测] 全部由本脚本计算; 不做主观判断。
"""
import sys, json, os
import numpy as np
from scipy.io import wavfile
from scipy.signal import get_window

SR_TARGET = 16000


def read_wav(path):
    sr, x = wavfile.read(path)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if x.dtype == np.int16:
        x = x.astype(np.float64) / 32768.0
    elif x.dtype == np.int32:
        x = x.astype(np.float64) / 2147483648.0
    elif x.dtype == np.uint8:
        x = (x.astype(np.float64) - 128.0) / 128.0
    return int(sr), np.asarray(x, dtype=np.float64)


def frame_signal(x, win, hop):
    n = int(round(win))
    h = int(round(hop))
    if len(x) < n:
        return np.zeros((0, n))
    m = 1 + (len(x) - n) // h
    idx = np.arange(n)[None, :] + h * np.arange(m)[:, None]
    return x[idx]


def f0_track(x, sr, win=0.040, hop=0.010, fmin=60.0, fmax=900.0, vth=0.42):
    """NSDF(McLeod/YIN 同族) 批量 F0 跟踪。
    n(τ) = 2·ac(τ)/[e0(τ)+e1(τ)]；取"第一个 ≥0.9·max 的局部极大"为周期(抗次谐波)；
    抛物线插值。返回 (t, f0, 周期性, 帧 rms)。
    """
    n = int(round(win * sr)); h = int(round(hop * sr))
    fr = frame_signal(x, n, h)
    m = fr.shape[0]
    if m == 0:
        return np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0)
    fr = fr - fr.mean(axis=1, keepdims=True)
    rms = np.sqrt((fr ** 2).mean(axis=1))
    tau_min = max(2, int(np.floor(sr / fmax)))
    tau_max = min(n - 2, int(np.ceil(sr / fmin)))
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    F = np.fft.rfft(fr, nfft, axis=1)
    ac = np.fft.irfft(F * np.conj(F), nfft, axis=1)[:, :n]      # 线性自相关
    c = np.cumsum(fr ** 2, axis=1)
    tot = c[:, -1:]
    taus = np.arange(tau_min, tau_max + 1)
    # e0(τ)=Σ_{j=0}^{n-1-τ} x_j² = c[n-1-τ]；e1(τ)=Σ_{j=τ}^{n-1} x_j² = tot - c[τ-1]
    e0 = c[:, (n - 1 - taus)]
    prev = np.concatenate([np.zeros((m, 1)), c], axis=1)[:, taus]   # = c[τ-1]
    e1 = tot - prev
    mt = e0 + e1 + 1e-20
    nd = 2.0 * ac[:, taus] / mt
    # 局部极大
    if nd.shape[1] < 3:
        return np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0)
    ismax = (nd[:, 1:-1] > nd[:, :-2]) & (nd[:, 1:-1] >= nd[:, 2:])
    orig = np.arange(1, nd.shape[1] - 1)
    vals = nd[:, 1:-1]
    vmax = vals.max(axis=1)
    ok = np.where(ismax & (vals >= 0.88 * vmax[:, None]), orig[None, :],
                  np.iinfo(np.int32).max)
    idx = ok.argmin(axis=1)
    found = ok.min(axis=1) != np.iinfo(np.int32).max
    sel = np.where(found, idx, np.argmax(vals, axis=1))
    val = vals[np.arange(m), sel]
    # 抛物线插值(在该局部极大上)
    L = sel + 1       # nd 中的索引
    a = np.take_along_axis(nd, np.clip(L - 1, 0, nd.shape[1] - 1)[:, None], axis=1)[:, 0]
    b = np.take_along_axis(nd, L[:, None], axis=1)[:, 0]
    d = np.take_along_axis(nd, np.clip(L + 1, 0, nd.shape[1] - 1)[:, None], axis=1)[:, 0]
    den = a - 2 * b + d
    shift = np.where(np.abs(den) > 1e-12, 0.5 * (a - d) / den, 0.0)
    shift = np.clip(shift, -1, 1) * found
    tau = taus[np.clip(sel + 1, 0, len(taus) - 1)] + shift
    f0 = sr / tau
    f0[rms < 1e-6] = np.nan
    return (np.arange(m) * h + n / 2.0) / sr, f0, val, rms


def db(v):
    return 20.0 * np.log10(np.maximum(v, 1e-12))


def ltas_7band(x, sr):
    """7 带 log 能量(与 meeting58 指标口径接近): 100-300/300-600/600-1200/1200-2400/2400-4000/4000-6000/6000-8000"""
    bands = [(100, 300), (300, 600), (600, 1200), (1200, 2400), (2400, 4000), (4000, 6000), (6000, 8000)]
    n = 1024
    hop = 256
    fr = frame_signal(x, n, hop)
    if fr.shape[0] == 0:
        return np.zeros(len(bands))
    w = get_window('hann', n)
    S = np.abs(np.fft.rfft(fr * w, axis=1)) ** 2
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    out = []
    for lo, hi in bands:
        sel = (freqs >= lo) & (freqs < hi)
        e = S[:, sel].sum(axis=1)
        out.append(float(np.mean(10 * np.log10(np.maximum(e, 1e-20)) + 1e-12)))
    return np.array(out)


def spectral_flatness(x, sr, win=1024, hop=256):
    fr = frame_signal(x, win, hop)
    if fr.shape[0] == 0:
        return np.zeros(0)
    w = get_window('hann', win)
    S = np.abs(np.fft.rfft(fr * w, axis=1)) ** 2 + 1e-20
    gm = np.exp(np.mean(np.log(S), axis=1))
    am = np.mean(S, axis=1)
    return gm / am


def reverb_probe(x, sr, t, env_db):
    """混响代理:
    - 衰减斜率: 找能量急降帧(每 10ms 一帧), 取其后续 30 帧做 dB 线性拟合, 斜率 dB/s
    - 包络自相关: log 包络 lag 0.1~0.6s 的最大相关(混响/平滑会把相关性抬向正值)
    """
    env = env_db - np.median(env_db)
    d = np.diff(env)
    th = -6.0
    idx = np.where(d < th)[0]
    slopes = []
    for i in idx:
        seg = env[i + 1:i + 31]
        if len(seg) >= 20 and np.all(np.isfinite(seg)):
            k = np.arange(len(seg))
            A = np.vstack([k, np.ones_like(k)]).T
            coef, *_ = np.linalg.lstsq(A, seg, rcond=None)
            slopes.append(coef[0] * 100.0)  # dB/s (hop=10ms)
    slopes = np.array(slopes) if slopes else np.array([np.nan])
    e = env - env.mean()
    hop = t[1] - t[0] if len(t) > 1 else 0.01
    ac = {}
    for lag_s in (0.05, 0.1, 0.2, 0.3, 0.4, 0.6):
        L = int(round(lag_s / hop))
        if L < len(e):
            a, b = e[:-L], e[L:]
            ac[lag_s] = float(np.corrcoef(a, b)[0, 1]) if len(a) > 10 else np.nan
    return {
        "drop_slope_dBps_median": float(np.nanmedian(slopes)),
        "drop_slope_dBps_p25": float(np.nanpercentile(slopes, 25)) if slopes.size else np.nan,
        "n_drops": int(len(slopes)),
        "env_autocorr": {str(k): v for k, v in ac.items()},
    }


def bgm_probe(x, sr, t, fr, rms):
    """伴奏/背景音乐代理:
    - 低能量帧(静音/停顿)里是否仍有宽带稳态能量(纯人声录音应有很低底噪)
    - 时间维度上"稳态谱峰": 逐帧幅度谱取 10 分位(即该频点在绝大多数时间都有的能量) -> 稳态窄峰
    - 低频(<120Hz)稳态能量: 人声 F0 常 >150Hz, 持续低频多为 BGM 低音
    """
    n = 1024
    hop = 256
    w = get_window('hann', n)
    S = np.abs(np.fft.rfft(fr * w, axis=1)) / (np.sum(w) * 0.5)
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    # 与本函数谱帧同帧数(1024/256), 避免与 40ms F0 帧错位
    rmsdb = 20 * np.log10(np.maximum(np.sqrt((fr ** 2).mean(axis=1)), 1e-12))
    q = np.percentile(rmsdb, 8)
    quiet = rmsdb <= q
    quiet_spec = S[quiet].mean(axis=0) if quiet.any() else np.zeros_like(S[0])
    # 稳态谱 = 时间 10 分位
    steady = np.percentile(S, 10, axis=0)
    tot = np.sum(steady ** 2) + 1e-20
    low = (freqs < 120)
    mid = (freqs >= 120) & (freqs < 2000)
    # 窄峰检测: 稳态谱上的局部峰(相邻 ±3 bin 均值 >3 倍)
    peaks = []
    for i in range(4, len(steady) - 4):
        nb = (steady[i - 4:i].mean() + steady[i + 1:i + 5].mean()) / 2 + 1e-20
        if steady[i] > 6 * nb and steady[i] > 1e-4:
            peaks.append((float(freqs[i]), float(steady[i] / nb)))
    peaks = sorted(peaks, key=lambda z: -z[1])[:6]
    return {
        "quiet_frame_rms_db_p08": float(q),
        "quiet_frame_spectral_centroid_hz": float(
            np.sum(freqs * quiet_spec) / (np.sum(quiet_spec) + 1e-20)),
        "steady_low_energy_ratio": float(np.sum(steady[low] ** 2) / tot),
        "steady_mid_energy_ratio": float(np.sum(steady[mid] ** 2) / tot),
        "steady_narrow_peaks": peaks,
        "quiet_vs_active_band_ratio_db": float(
            10 * np.log10((np.sum(quiet_spec ** 2) + 1e-20) / (np.sum(S[~quiet].mean(axis=0) ** 2) + 1e-20))
            if quiet.any() and (~quiet).any() else np.nan),
    }


def speech_vs_sing(t, f0, vr):
    """说话 vs 歌唱 代理: 浊音段 F0 半音稳定性 / 稳定平台比例 / 颤音(4-8Hz)强度"""
    voiced = np.isfinite(f0) & (vr > 0.42)
    if voiced.sum() < 20:
        return {"n_voiced": int(voiced.sum())}
    sem = 12 * np.log2(f0[voiced] / 55.0)
    vf = f0[voiced]
    hop = t[1] - t[0]
    # 连续浊音段
    runs = []
    i = 0
    v = voiced.copy()
    while i < len(v):
        if v[i]:
            j = i
            while j < len(v) and v[j]:
                j += 1
            if j - i >= 5:
                runs.append((i, j))
            i = j
        else:
            i += 1
    run_lens = np.array([(b - a) * hop for a, b in runs]) if runs else np.array([0.0])
    # 半音稳定平台: |Δsemi| < 0.6 st / 40ms
    dsemi = np.abs(np.diff(sem)) / (1.0)  # 每帧(10ms)
    stable = float(np.mean(dsemi < 0.6 * (hop / 0.04))) if len(dsemi) else np.nan
    # 颤音: 对 F0 轮廓 detrend 后 4-8Hz 能量占比
    s = sem - np.convolve(sem, np.ones(51) / 51, mode='same')
    F = np.abs(np.fft.rfft(s))
    frq = np.fft.rfftfreq(len(s), hop)
    vib = (frq >= 4) & (frq <= 8)
    return {
        "n_voiced": int(voiced.sum()),
        "voiced_ratio": float(voiced.mean()),
        "f0_semitone_p5": float(np.percentile(sem, 5)),
        "f0_semitone_median": float(np.median(sem)),
        "f0_semitone_p95": float(np.percentile(sem, 95)),
        "f0_semitone_span_p5_p95": float(np.percentile(sem, 95) - np.percentile(sem, 5)),
        "longest_voiced_run_s": float(run_lens.max()) if len(run_lens) else 0.0,
        "median_voiced_run_s": float(np.median(run_lens)) if len(run_lens) else 0.0,
        "frac_runs_gt_0p4s": float(np.mean(run_lens > 0.4)) if len(run_lens) else 0.0,
        "pitch_stable_frac": stable,
        "vibrato_4_8Hz_energy_frac": float(np.sum(F[vib] ** 2) / (np.sum(F ** 2) + 1e-20)),
    }


def main():
    if len(sys.argv) < 2:
        print("usage: 工具_核验分析.py <in.wav> [--json out.json] [--csv out.csv]")
        return 1
    path = sys.argv[1]
    out_json = None
    out_csv = None
    for i, a in enumerate(sys.argv):
        if a == "--json" and i + 1 < len(sys.argv):
            out_json = sys.argv[i + 1]
        if a == "--csv" and i + 1 < len(sys.argv):
            out_csv = sys.argv[i + 1]
    sr, x = read_wav(path)
    dur = len(x) / sr
    t, f0, vr, rms = f0_track(x, sr)
    env_db = db(rms)
    voiced = np.isfinite(f0) & (vr > 0.42)
    f0v = f0[voiced]
    sem = 12 * np.log2(f0v / 55.0) if f0v.size else np.zeros(0)
    # 段能量分布: 每 0.5s 一段
    seg_n = int(0.5 * sr)
    seg_db = []
    for i in range(0, len(x) - seg_n + 1, seg_n):
        s = x[i:i + seg_n]
        seg_db.append(float(db(np.sqrt((s * s).mean()))))
    seg_db = np.array(seg_db)
    rep = {
        "file": os.path.abspath(path),
        "sr": sr,
        "duration_s": float(dur),
        "n_samples": int(len(x)),
        "peak_dbfs": float(db(np.max(np.abs(x)))),
        "rms_dbfs_global": float(db(np.sqrt(np.mean(x * x)))),
        "frame_rms_db_p10_p50_p90": [float(np.percentile(env_db, 10)),
                                     float(np.percentile(env_db, 50)),
                                     float(np.percentile(env_db, 90))],
        "seg0p5s_energy_db_p10_p50_p90": [float(np.percentile(seg_db, 10)),
                                          float(np.percentile(seg_db, 50)),
                                          float(np.percentile(seg_db, 90))],
        "seg0p5s_energy_db_seq": [round(v, 2) for v in seg_db],
        "voiced_ratio": float(voiced.mean()),
        "f0_hz_p5_median_p95": [float(np.percentile(f0v, 5)), float(np.median(f0v)),
                                float(np.percentile(f0v, 95))] if f0v.size else [None] * 3,
        "f0_hz_min_max": [float(np.min(f0v)), float(np.max(f0v))] if f0v.size else [None] * 2,
        "f0_semitone_span_p5_p95": float(np.percentile(sem, 95) - np.percentile(sem, 5))
        if sem.size else None,
        "speech_vs_sing": speech_vs_sing(t, f0, vr),
        "bgm_probe": bgm_probe(x, sr, t, frame_signal(x, 1024, 256), rms),
        "reverb_probe": reverb_probe(x, sr, t, env_db),
        "ltas_7band_db": [round(v, 2) for v in ltas_7band(x, sr)],
        "spectral_flatness_mean": float(np.mean(spectral_flatness(x, sr))),
        "n_frames": int(len(t)),
    }
    print(json.dumps({k: v for k, v in rep.items() if k != "seg0p5s_energy_db_seq"},
                     ensure_ascii=False, indent=1))
    print("\n[0.5s 段能量 dB 序列]")
    for i in range(0, len(seg_db), 10):
        print("  %5.1fs: " % (i * 0.5) + " ".join("%6.1f" % v for v in seg_db[i:i + 10]))
    if out_csv:
        with open(out_csv, "w", encoding="utf-8") as f:
            f.write("t_s,f0_hz,vr,rms_db\n")
            for i in range(len(t)):
                f.write("%.3f,%.2f,%.3f,%.2f\n" % (t[i], f0[i] if np.isfinite(f0[i]) else -1, vr[i], env_db[i]))
        print("\n[帧级 CSV] %s" % out_csv)
    if out_json:
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(rep, f, ensure_ascii=False, indent=1)
        print("[JSON] %s" % out_json)
    return 0


if __name__ == "__main__":
    sys.exit(main())

# -*- coding: utf-8 -*-
"""meeting59 / refs-web：深查——谱白化谐波梳 F0 / 包络调制谱(说话 vs 歌唱) / 低频稳态(伴奏-嗡声)检测
用法: python 工具_深查.py <mono16k.wav> <out_prefix>
"""
import sys, json
import numpy as np
from scipy.io import wavfile
from scipy.signal import get_window, butter, sosfiltfilt
from scipy.ndimage import uniform_filter1d

sys.path.insert(0, __file__.rsplit("\\", 1)[0])
from 工具_侧信号与F0 import comb_matrix  # noqa


def read_mono(path):
    sr, x = wavfile.read(path)
    x = x.astype(np.float64) / (32768.0 if x.dtype == np.int16 else 1.0)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return int(sr), x


def frames(x, win, hop):
    n, h = int(win), int(hop)
    m = 1 + (len(x) - n) // h
    idx = np.arange(n)[None, :] + h * np.arange(m)[:, None]
    return x[idx], n


def main():
    src, pref = sys.argv[1], sys.argv[2]
    sr, x = read_mono(src)
    dur = len(x) / sr
    # --- 带通 150-4000 再算 F0, 去掉低频嗡声/贝斯 ---
    sos = butter(4, [150.0, min(4000.0, sr / 2 - 200)], btype='band', fs=sr, output='sos')
    xh = sosfiltfilt(sos, x)
    win, hop = int(0.05 * sr), int(0.01 * sr)
    fr, n = frames(xh, win, hop)
    w = get_window('hann', n)
    S = np.abs(np.fft.rfft((fr - fr.mean(axis=1, keepdims=True)) * w, axis=1))
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    # 谱白化: 每帧除以 ~120Hz 滑动平均包络
    bw = freqs[1] - freqs[0]
    k = max(3, int(round(120.0 / bw)))
    env = uniform_filter1d(S, size=k, axis=1, mode='nearest') + 1e-9
    Sw = S / env
    lo = freqs >= 150
    cands, A = comb_matrix(n, sr, 150.0, 900.0, 1.0, 4000.0)
    # 白化谱做梳状评分
    Sc = Sw @ A.T
    Sc = Sc / (np.sqrt((Sw[:, lo] ** 2).sum(axis=1))[:, None] + 1e-12)
    best = Sc.argmax(axis=1)
    sal = Sc[np.arange(len(best)), best]
    # 显著性: 最佳/中位
    saln = sal / (np.median(Sc, axis=1) + 1e-12)
    f0 = cands[best].astype(float)
    rmsdb = 20 * np.log10(np.sqrt((fr ** 2).mean(axis=1)) + 1e-12)
    gate = (rmsdb > max(-55.0, np.percentile(rmsdb, 12) + 3.0)) & (saln > 3.0) & (sal > 0.02)
    f0[~gate] = np.nan
    t = (np.arange(len(f0)) * hop + win / 2) / sr
    ok = np.isfinite(f0)
    sem = 12 * np.log2(f0[ok] / 55.0)
    # 中值滤波后的 F0 用于段落判断
    f0s = f0.copy()
    runs = []
    i = 0
    while i < len(ok):
        if ok[i]:
            j = i
            while j < len(ok) and ok[j]:
                j += 1
            runs.append((i, j)); i = j
        else:
            i += 1
    rl = np.array([(b - a) * 0.01 for a, b in runs])
    # 每段 F0 稳定性(半音标准差)
    stab = []
    for a, b in runs:
        if b - a >= 10:
            ss = 12 * np.log2(f0[a:b] / 55.0)
            stab.append((float(np.std(ss)), float(np.median(f0[a:b])), (b - a) * 0.01))
    stab = np.array(stab) if stab else np.zeros((0, 3))
    out = {
        "file": src, "dur": dur, "sr": sr,
        "f0_whitened": {
            "n_voiced": int(ok.sum()), "voiced_ratio": float(ok.mean()),
            "median_hz": float(np.median(f0[ok])) if ok.any() else None,
            "p5_hz": float(np.percentile(f0[ok], 5)) if ok.any() else None,
            "p95_hz": float(np.percentile(f0[ok], 95)) if ok.any() else None,
            "semitone_span_p5_p95": float(np.percentile(sem, 95) - np.percentile(sem, 5)) if ok.any() else None,
            "frame_df0_semitone_median": float(np.median(np.abs(np.diff(sem)))) if ok.sum() > 50 else None,
            "frame_df0_semitone_p90": float(np.percentile(np.abs(np.diff(sem)), 90)) if ok.sum() > 50 else None,
        },
        "voiced_runs": {
            "n": int(len(rl)), "median_s": float(np.median(rl)) if len(rl) else 0,
            "p90_s": float(np.percentile(rl, 90)) if len(rl) else 0,
            "max_s": float(rl.max()) if len(rl) else 0,
            "frac_gt_0p4s": float(np.mean(rl > 0.4)) if len(rl) else 0,
            "frac_gt_1p0s": float(np.mean(rl > 1.0)) if len(rl) else 0,
        },
        "run_f0_std_semitone_median": float(np.median(stab[:, 0])) if len(stab) else None,
    }
    # --- 包络调制谱 (说话 4-6Hz 峰 vs 歌唱 1-3Hz) ---
    sos_e = butter(4, 30.0, btype='low', fs=sr, output='sos')
    ec = sosfiltfilt(sos_e, np.sqrt(xh ** 2))
    ec = np.log(np.maximum(ec, 1e-9))
    ec = ec - uniform_filter1d(ec, size=int(0.5 * sr), mode='nearest')
    E = np.sqrt(np.mean(np.array([ec[i:i + 512] ** 2 for i in range(0, len(ec) - 512, 256)]), axis=1))
    # 用 8Hz 重采样的包络做调制 FFT
    dec = int(sr / 64)
    e2 = ec[::dec]
    e2 = e2 - e2.mean()
    F = np.abs(np.fft.rfft(e2 * np.hanning(len(e2))))
    frq = np.fft.rfftfreq(len(e2), dec / sr)
    def band(l, h):
        s = (frq >= l) & (frq < h)
        return float(np.sum(F[s] ** 2))
    tot = band(0.5, 20) + 1e-20
    out["envelope_modulation"] = {
        "1_3Hz_frac": band(1, 3) / tot, "3_5Hz_frac": band(3, 5) / tot,
        "4_6Hz_frac": band(4, 6) / tot, "5_10Hz_frac": band(5, 10) / tot,
        "peak_hz": float(frq[np.argmax(np.where((frq > 0.5) & (frq < 15), F, 0))]),
    }
    # --- 低频稳态检测 (60-160Hz 窄带谱随时间) ---
    sos_l = butter(4, [60.0, 200.0], btype='band', fs=sr, output='sos')
    xl = sosfiltfilt(sos_l, x)
    nl = int(0.10 * sr)
    fl, _ = frames(xl, nl, int(0.02 * sr))
    SL = np.abs(np.fft.rfft(fl * get_window('hann', nl), axis=1))
    fL = np.fft.rfftfreq(nl, 1.0 / sr)
    peakidx = SL[:, (fL > 70) & (fL < 180)].argmax(axis=1)
    fLsel = fL[(fL > 70) & (fL < 180)]
    pk = fLsel[peakidx]
    out["lowband_60_200"] = {
        "peak_hz_median": float(np.median(pk)),
        "peak_hz_p10_p90": [float(np.percentile(pk, 10)), float(np.percentile(pk, 90))],
        "rms_db_p50": float(np.median(20 * np.log10(np.sqrt((fl ** 2).mean(axis=1)) + 1e-12))),
        "concentration": float(np.median(SL.max(axis=1) / (SL.mean(axis=1) + 1e-12))),
    }
    print(json.dumps(out, ensure_ascii=False, indent=1))
    np.save(pref + ".f0w.npy", np.vstack([t, f0, saln]))
    np.save(pref + ".lowf.npy", np.vstack([np.arange(len(pk)) * 0.02, pk]))
    # 浊音段表(>=0.25s) 供选段
    tbl = []
    for a, b in runs:
        if (b - a) * 0.01 >= 0.25:
            ss = f0[a:b]
            tbl.append({"t0": round(float(t[a]), 3), "t1": round(float(t[b - 1]), 3),
                        "dur": round((b - a) * 0.01, 2),
                        "f0_med": round(float(np.nanmedian(ss)), 1),
                        "f0_std_st": round(float(np.nanstd(12 * np.log2(ss / 55.0))), 2)})
    print("\n[voiced runs >=0.25s] n=%d" % len(tbl))
    for r in tbl[:80]:
        print("  %6.2f-%6.2f (%4.2fs) f0med=%5.0f st_std=%.2f" % (r["t0"], r["t1"], r["dur"], r["f0_med"], r["f0_std_st"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())

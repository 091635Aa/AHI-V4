# -*- coding: utf-8 -*-
"""meeting59 / refs-web：立体声侧信号 BGM 检测 + 谐波梳 F0 全程跟踪（批量矩阵化）
用法: python 工具_侧信号与F0.py <stereo_44k.wav> <mono_16k.wav> <out_prefix>
"""
import sys, json
import numpy as np
from scipy.io import wavfile
from scipy.signal import get_window


def read_stereo(path):
    sr, x = wavfile.read(path)
    x = x.astype(np.float64) / (32768.0 if x.dtype == np.int16 else 1.0)
    if x.ndim == 1:
        x = np.stack([x, x], axis=1)
    return int(sr), x


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
    return x[idx], n, h


def comb_matrix(n, sr, fmin, fmax, step=1.0, voice_lim=5000.0, spread=1):
    """(n_cand, n_bins) 谐波梳矩阵: 每个候选 f0 在 k*f0 处加权 ks^-0.5"""
    nfft_bins = n // 2 + 1
    cands = np.arange(fmin, fmax + 1e-9, step)
    A = np.zeros((len(cands), nfft_bins), dtype=np.float64)
    for ci, f0 in enumerate(cands):
        ks = np.arange(1, int(min(voice_lim, sr / 2 - 100) / f0) + 1)
        idx = np.round(ks * f0 * n / sr).astype(int)
        bad = (idx < 2) | (idx >= nfft_bins - 1)
        ks, idx = ks[~bad], idx[~bad]
        w = ks ** -0.5
        for d in range(-spread, spread + 1):
            ii = np.clip(idx + d, 0, nfft_bins - 1)
            np.add.at(A[ci], ii, w)
    # 行归一: 否则低 f0 因谐波数多而系统性占优(次谐波误判)
    s = A.sum(axis=1, keepdims=True)
    A = A / np.maximum(s, 1e-12)
    return cands, A


def main():
    src_st, src_mono, pref = sys.argv[1], sys.argv[2], sys.argv[3]
    sr, X = read_stereo(src_st)
    dur = len(X) / sr
    L, R = X[:, 0], X[:, 1]
    mid, side = 0.5 * (L + R), 0.5 * (L - R)
    win, hop = int(0.05 * sr), int(0.01 * sr)
    fm, n, h = frames(mid, win, hop)
    fs, _, _ = frames(side, win, hop)
    rm = np.sqrt((fm ** 2).mean(axis=1)); rs = np.sqrt((fs ** 2).mean(axis=1))
    dB = lambda v: 20 * np.log10(np.maximum(v, 1e-12))
    ratio = dB(rs) - dB(rm)
    q = np.percentile(dB(rm), 10)
    quiet = dB(rm) <= q + 2.0
    out = {
        "stereo_file": src_st, "sr": sr, "dur": dur,
        "mid_rms_db_p10_p50_p90": [float(np.percentile(dB(rm), 10)), float(np.percentile(dB(rm), 50)),
                                    float(np.percentile(dB(rm), 90))],
        "side_minus_mid_db_p10_p50_p90": [float(np.percentile(ratio, 10)), float(np.median(ratio)),
                                          float(np.percentile(ratio, 90))],
        "quiet_frames_n": int(quiet.sum()),
        "quiet_side_minus_mid_db_median": float(np.median(ratio[quiet])) if quiet.any() else None,
        "quiet_mean_rms_db": float(np.median(dB(rm)[quiet])) if quiet.any() else None,
    }
    # ---- F0: 16k mono, 批量谐波梳 ----
    sr2, x2 = read_mono(src_mono)
    fwin, fhop = int(0.05 * sr2), int(0.01 * sr2)
    fr, n2, _ = frames(x2, fwin, fhop)
    fmn = fr.mean(axis=1, keepdims=True)
    fr = fr - fmn
    w = get_window('hann', n2)
    S = np.abs(np.fft.rfft(fr * w, axis=1))
    freqs = np.fft.rfftfreq(n2, 1.0 / sr2)
    band = freqs <= 8000
    base = np.sqrt((S[:, band] ** 2).mean(axis=1)) + 1e-12
    cands, A = comb_matrix(n2, sr2, 110.0, 900.0, 1.0, 5000.0)
    Sc = S @ A.T  # (n_frames, n_cand)
    Sc = Sc / base[:, None]
    best = Sc.argmax(axis=1)
    sal = Sc[np.arange(len(best)), best]
    f0 = cands[best].astype(float)
    rmsdb = 20 * np.log10(np.sqrt((fr ** 2).mean(axis=1)) + 1e-12)
    gate = rmsdb > max(-55.0, np.percentile(rmsdb, 12) + 3.0)
    f0[~(gate & (sal > 0.35))] = np.nan
    t = (np.arange(len(f0)) * fhop + fwin / 2) / sr2
    ok = np.isfinite(f0)
    sem = 12 * np.log2(f0[ok] / 55.0)
    out["combF0"] = {
        "n_frames": int(len(f0)), "n_voiced": int(ok.sum()),
        "median_hz": float(np.median(f0[ok])), "p5_hz": float(np.percentile(f0[ok], 5)),
        "p95_hz": float(np.percentile(f0[ok], 95)),
        "semitone_span_p5_p95": float(np.percentile(sem, 95) - np.percentile(sem, 5)),
        "frame_df0_semitone_median": float(np.median(np.abs(np.diff(sem)))) if ok.sum() > 50 else None,
        "frame_df0_semitone_p90": float(np.percentile(np.abs(np.diff(sem)), 90)) if ok.sum() > 50 else None,
    }
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
    out["voiced_runs"] = {
        "n": int(len(rl)), "median_s": float(np.median(rl)) if len(rl) else 0.0,
        "p90_s": float(np.percentile(rl, 90)) if len(rl) else 0.0,
        "max_s": float(rl.max()) if len(rl) else 0.0,
        "frac_gt_0p4s": float(np.mean(rl > 0.4)) if len(rl) else 0.0,
    }
    print(json.dumps(out, ensure_ascii=False, indent=1))
    np.save(pref + ".f0.npy", np.vstack([t, f0, sal]))
    np.save(pref + ".env.npy", np.vstack([t, rmsdb]))
    return 0


if __name__ == "__main__":
    sys.exit(main())

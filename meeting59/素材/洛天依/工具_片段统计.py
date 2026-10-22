# -*- coding: utf-8 -*-
"""meeting59 / refs-web：片段清单统计（manifest 用）——F0/LTAS/倾斜/浊音率/静音率/sha256
用法: python 工具_片段统计.py <out.json> <wav1> <wav2> ...
"""
import sys, json, os, hashlib
import numpy as np

sys.path.insert(0, __file__.rsplit("\\", 1)[0])
from 工具_核验分析 import read_wav, f0_track, ltas_7band, db  # noqa

BANDS = ["100-300", "300-600", "600-1200", "1200-2400", "2400-4000", "4000-6000", "6000-8000"]
BAND_EDGES = [(100, 300), (300, 600), (600, 1200), (1200, 2400), (2400, 4000), (4000, 6000), (6000, 8000)]


def ltas_density(x, sr, n=1024, hop=256):
    """谱密度 LTAS: 每带取 |X|² 的带内均值(不受带宽影响), 便于算谱倾斜 dB/oct"""
    from scipy.signal import get_window
    fr = np.lib.stride_tricks.sliding_window_view(x, n)[::hop]
    w = get_window('hann', n)
    S = np.abs(np.fft.rfft(fr * w, axis=1)) ** 2
    f = np.fft.rfftfreq(n, 1.0 / sr)
    out = []
    for lo, hi in BAND_EDGES:
        sel = (f >= lo) & (f < hi)
        out.append(10 * np.log10(np.mean(S[:, sel]) + 1e-20))
    return np.array(out)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def stat(path):
    sr, x = read_wav(path)
    t, f0, vr, rms = f0_track(x, sr)
    voiced = np.isfinite(f0) & (vr > 0.42)
    f0v = f0[voiced]
    rmsdb = db(rms)
    ltas = ltas_7band(x, sr)
    ltasd = ltas_density(x, sr)
    # 谱倾斜: 谱密度 LTAS 对 log2(f) 线性拟合 dB/oct
    fc = np.array([200, 450, 900, 1800, 3200, 5000, 7000], dtype=float)
    A = np.vstack([np.log2(fc), np.ones_like(fc)]).T
    coef, *_ = np.linalg.lstsq(A, ltasd, rcond=None)
    return {
        "file": os.path.basename(path),
        "size_bytes": os.path.getsize(path),
        "sample_rate": sr,
        "duration_s": round(len(x) / sr, 3),
        "n_samples": int(len(x)),
        "sha256": sha256(path),
        "peak_dbfs": round(float(db(np.max(np.abs(x)))), 2),
        "rms_dbfs": round(float(db(np.sqrt(np.mean(x * x)))), 2),
        "silence_lt_-55db_frac": round(float(np.mean(rmsdb < -55.0)), 3),
        "voiced_frac": round(float(voiced.mean()), 3),
        "periodicity_median": round(float(np.median(vr[voiced])), 3) if voiced.any() else None,
        "f0_hz_p5_p50_p95": [round(float(np.percentile(f0v, 5)), 1),
                              round(float(np.median(f0v)), 1),
                              round(float(np.percentile(f0v, 95)), 1)] if f0v.size else None,
        "f0_semitone_span_p5_p95": round(float(12 * np.log2(np.percentile(f0v, 95) /
                                                            np.percentile(f0v, 5))), 2) if f0v.size else None,
        "frames_df0_st_median": round(float(np.median(np.abs(np.diff(12 * np.log2(f0v / 55.0))))), 3)
        if f0v.size > 50 else None,
        "ltas_7band_db": {b: round(float(v), 2) for b, v in zip(BANDS, ltas)},
        "ltas_density_7band_db": {b: round(float(v), 2) for b, v in zip(BANDS, ltasd)},
        "ltas_slope_db_per_oct": round(float(coef[0]), 2),
    }


def main():
    out = sys.argv[1]
    rows = [stat(p) for p in sys.argv[2:]]
    print(json.dumps(rows, ensure_ascii=False, indent=1))
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    tot = sum(r["duration_s"] for r in rows)
    print("[total] %d files, %.2fs" % (len(rows), tot))


if __name__ == "__main__":
    main()

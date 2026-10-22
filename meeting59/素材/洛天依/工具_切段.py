# -*- coding: utf-8 -*-
"""meeting59 / refs-web：参考片段切取（纯 scipy：去直流 + 5ms 淡入淡出 + s16 落盘；不降噪不增强）
用法:
  python 工具_切段.py list  <in.wav> [--dur 4.0] [--top 12] [--n 4]
  python 工具_切段.py cut   <in.wav> <out_dir> <prefix> <seg1> [seg2 ...]   # seg = t0:t1
"""
import sys, os
import numpy as np
from scipy.io import wavfile
from scipy.signal import get_window
from scipy.ndimage import uniform_filter1d

SR = 16000


def read_wav(path):
    sr, x = wavfile.read(path)
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = x.astype(np.float64) / (32768.0 if x.dtype == np.int16 else 1.0)
    return int(sr), x


def env_frames(x, sr, win=0.040, hop=0.010):
    n, h = int(win * sr), int(hop * sr)
    m = 1 + (len(x) - n) // h
    idx = np.arange(n)[None, :] + h * np.arange(m)[:, None]
    fr = x[idx]
    rms = np.sqrt((fr ** 2).mean(axis=1))
    return rms, h / sr


def candidates(x, sr, dur, top=12, n=4, min_gap=0.5):
    rms, hop = env_frames(x, sr)
    db = 20 * np.log10(np.maximum(rms, 1e-12))
    win = int(round(dur / hop))
    if win >= len(db):
        return []
    act = db > max(-60.0, np.percentile(db, 20) + 12.0)
    act_r = np.convolve(act.astype(float), np.ones(win) / win, mode="valid")
    # 局部能量稳定度: 段内 dB 的标准差(越小越稳)
    # 用前缀和快速求均值/方差
    c1 = np.concatenate([[0], np.cumsum(db)])
    c2 = np.concatenate([[0], np.cumsum(db ** 2)])
    k = np.arange(len(db) - win + 1)
    s1 = c1[k + win] - c1[k]
    s2 = c2[k + win] - c2[k]
    mean = s1 / win
    var = np.maximum(s2 / win - mean ** 2, 0)
    std = np.sqrt(var)
    # 静音段: dB < -55 的比例
    quiet = (db < -55).astype(float)
    cq = np.concatenate([[0], np.cumsum(quiet)])
    quiet_r = (cq[k + win] - cq[k]) / win
    # 削波率: 段内 |x|>0.999 的样点占比
    clip = (np.abs(x) > 0.999).astype(float)
    cc = np.concatenate([[0], np.cumsum(clip)])
    hop_s = int(round(hop * sr))
    win_s = int(round(dur * sr))
    ks = k * hop_s
    clip_r = (cc[ks + win_s] - cc[ks]) / win_s
    score = act_r - 2.0 * quiet_r - 0.02 * std - 50.0 * clip_r
    order = np.argsort(score)[::-1]
    picks = []
    for i in order:
        t0 = i * hop
        if t0 + dur > len(x) / sr:
            continue
        if any(abs(t0 - p[0]) < dur + min_gap for p in picks):
            continue
        picks.append((float(t0), float(act_r[i]), float(quiet_r[i]), float(mean[i]), float(std[i]), float(score[i])))
        if len(picks) >= top:
            break
    return picks


def write_seg(x, sr, t0, t1, out, fade=0.005):
    i0, i1 = int(round(t0 * sr)), int(round(t1 * sr))
    seg = x[i0:i1].copy()
    seg = seg - seg.mean()
    nf = int(round(fade * sr))
    if len(seg) > 2 * nf:
        r = np.linspace(0, 1, nf)
        seg[:nf] *= r
        seg[-nf:] *= r[::-1]
    peak = np.max(np.abs(seg))
    if peak > 0.999:
        seg = seg / peak * 0.999
    wavfile.write(out, sr, np.round(seg * 32767.0).astype(np.int16))
    return len(seg), float(peak)


def main():
    mode = sys.argv[1]
    src = sys.argv[2]
    if mode == "list":
        dur = 4.0
        top = 12
        for i, a in enumerate(sys.argv):
            if a == "--dur":
                dur = float(sys.argv[i + 1])
            if a == "--top":
                top = int(sys.argv[i + 1])
        sr, x = read_wav(src)
        print("file=%s dur=%.2fs  候选段(%.1fs)" % (src, len(x) / sr, dur))
        for t0, ar, qr, mean, std, sc in candidates(x, sr, dur, top):
            print("  t0=%7.2f  active=%.2f quiet=%.2f mean=%6.1fdB std=%4.1f score=%5.2f" % (
                t0, ar, qr, mean, std, sc))
    elif mode == "cut":
        out_dir, prefix = sys.argv[3], sys.argv[4]
        sr, x = read_wav(src)
        os.makedirs(out_dir, exist_ok=True)
        info = []
        for j, s in enumerate(sys.argv[5:], 1):
            a, b = s.split(":")
            t0, t1 = float(a), float(b)
            out = os.path.join(out_dir, "%s_%02d.wav" % (prefix, j))
            n, pk = write_seg(x, sr, t0, t1, out)
            info.append({"file": os.path.basename(out), "t0": t0, "t1": t1,
                         "dur": round(t1 - t0, 3), "n": n, "peak_before_norm": round(pk, 5)})
            print("[cut] %s  %.3f-%.3f (%.2fs) peak=%.3f" % (out, t0, t1, t1 - t0, pk))
        print(json_dump(info))


def json_dump(o):
    import json
    return json.dumps(o, ensure_ascii=False)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""meeting71/实施/发现_S1c.py —— S1 极速探针（mel 代理耳）：先拿信号，耳蜗版在 S1b 后台跑。

12 条语料 → librosa log-mel（80 带，10 ms）→ 能量事件切分 → 每段 mel 统计 + F0 中位 + 能量 + 时长
→ KMeans 网格 + 两次种子 ARI（可复现性）。文字标签仅作诊断。
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import librosa

HERE = Path(__file__).resolve().parent
M69 = HERE.parent.parent / "meeting69"
E = Path(r"<path>\AHI_m71")
SR = 22050


def 载(名, 路):
    s = importlib.util.spec_from_file_location(名, 路)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


耳 = 载("耳蜗前端_S0_2", M69 / "实施" / "耳蜗前端_S0_2.py")


def 主():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, silhouette_score
    meta = json.loads((HERE.parent.parent / "meeting66" / "产物" / "数据_对齐asr" / "meta.json").read_text(encoding="utf-8"))
    样本 = {r["id"]: r for r in meta["样本"]}
    ids = sorted(样本)[:12]
    特征 = []
    t0 = time.time()
    for sid in ids:
        x = 耳.读wav(Path(r"<path>\SpeakerA7代模型训练数据") / 样本[sid]["wav"])
        S = np.abs(librosa.stft(np.asarray(x, np.float64), n_fft=1024, hop_length=220, win_length=551,
                                window="hann", center=True)) ** 2
        M = np.log(np.maximum(librosa.feature.melspectrogram(S=S, sr=SR, n_fft=1024, n_mels=80,
                                                             fmin=0.0, fmax=SR / 2.0), 1e-10)).T   # (T,80)
        e = np.sqrt(np.mean(np.exp(M), axis=1))
        e = np.convolve(e, np.ones(3) / 3, mode="same")
        thr = max(0.08 * float(np.percentile(e, 95)), 1e-4)
        cand = [i for i in range(1, len(e) - 1) if e[i] >= e[i - 1] and e[i] >= e[i + 1] and e[i] > thr]
        keep = []
        for i in cand:
            if not keep or (i - keep[-1]) * 0.01 >= 0.05:
                keep.append(i)
            elif e[i] > e[keep[-1]]:
                keep[-1] = i
        if len(keep) < 2:
            continue
        边界 = [0]
        for a, b in zip(keep[:-1], keep[1:]):
            边界.append(int(a + np.argmin(e[a:b + 1])))
        边界.append(len(e) - 1)
        t, f0 = 耳.F0轨迹(x)
        for a0, b0 in zip(边界[:-1], 边界[1:]):
            if (b0 - a0) * 0.01 < 0.05 or (b0 - a0) * 0.01 > 0.45:
                continue
            seg = M[a0:b0]
            m = seg.mean(0); s = seg.std(0)
            v = f0[(t >= a0 * 0.01) & (t <= b0 * 0.01)]
            v = v[v > 0]
            f = np.concatenate([m, s, [float(np.median(v)) / 300.0 if len(v) else 0.0,
                                       float(e[a0:b0].mean()), (b0 - a0) * 0.01]])
            特征.append(f)
    X = np.asarray(特征, np.float64)
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    print("单元", X.shape, flush=True)
    报 = dict(工具="发现_S1c.py", 时间=time.strftime("%Y-%m-%d %H:%M:%S"), 样本=len(ids), 单元=int(len(X)), 网格={})
    for k in (12, 16, 24, 32, 48, 64):
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
        km2 = KMeans(n_clusters=k, n_init=10, random_state=1).fit(X)
        ari = float(adjusted_rand_score(km.labels_, km2.labels_))
        sil = float(silhouette_score(X, km.labels_)) if k < len(X) else 0.0
        报["网格"][str(k)] = dict(ARI两次种子=round(ari, 4), 轮廓=round(sil, 4))
        print("| k", k, "ARI", round(ari, 4), "sil", round(sil, 4), flush=True)
    报["耗时秒"] = round(time.time() - t0, 1)
    (E / "S1b" / "发现结果_mel.json").write_text(json.dumps(报, ensure_ascii=False, indent=1), encoding="utf-8")
    print("完成", 报["耗时秒"], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(主())

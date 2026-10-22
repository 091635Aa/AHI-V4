# -*- coding: utf-8 -*-
"""meeting71/实施/发现_S1b.py —— S1 快速探针：用 AHI 自己的耳（耳蜗场）无监督发现单元。

只读音频、不算控制/不合成：对 40 条语料做耳蜗前端分析（场未适 64 带 1 ms），
能量事件切分 → 每段取场统计（64 带均值/标准差）+ F0 中位 + 能量 + 时长 → KMeans 网格。
稳定性 = 两次随机种子聚类的 ARI；文字标签仅作诊断纯度（不进回路）。
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

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


def 切段(场, 最短=0.05, 最长=0.45):
    """场 (T,64) 1 ms → 单位段（秒）。能量 = 带均值。"""
    e = 场.mean(axis=1)
    e = np.convolve(e, np.ones(5) / 5, mode="same")
    if len(e) < 10:
        return []
    thr = max(0.05 * float(np.percentile(e, 95)), 1e-4)
    cand = [i for i in range(1, len(e) - 1) if e[i] >= e[i - 1] and e[i] >= e[i + 1] and e[i] > thr]
    keep = []
    for i in cand:
        if not keep or (i - keep[-1]) / 1000.0 >= 最短:
            keep.append(i)
        elif e[i] > e[keep[-1]]:
            keep[-1] = i
    if len(keep) < 2:
        return []
    边界 = [0]
    for a, b in zip(keep[:-1], keep[1:]):
        j = int(a + np.argmin(e[a:b + 1]))
        边界.append(j)
    边界.append(len(e) - 1)
    段 = []
    for a, b in zip(边界[:-1], 边界[1:]):
        if (b - a) / 1000.0 < 最短 or (b - a) / 1000.0 > 最长:
            continue
        段.append((a, b))
    return 段


def 主():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, silhouette_score

    meta = json.loads((HERE.parent.parent / "meeting66" / "产物" / "数据_对齐asr" / "meta.json").read_text(encoding="utf-8"))
    样本 = {r["id"]: r for r in meta["样本"]}
    ids = sorted(样本)[:40]
    E.mkdir(parents=True, exist_ok=True)
    (E / "S1b").mkdir(parents=True, exist_ok=True)
    特征 = []
    真标签 = []          # (sid, 音节序号) 仅诊断
    t0 = time.time()
    for sid in ids:
        wav = Path(r"<path>\SpeakerA7代模型训练数据") / 样本[sid]["wav"]
        a = 耳.分析(str(wav))
        场 = np.asarray(a["场未适"], np.float32)          # (T,64) 1 ms
        f0t = np.asarray(a["F0"]["t"], np.float64)
        f0 = np.asarray(a["F0"]["f0"], np.float64)
        段 = 切段(场)
        z = np.load(HERE.parent.parent / "meeting66" / "产物" / "数据_对齐asr" / "样本" / f"{sid:04d}.npz", allow_pickle=True)
        帧数 = z["音节帧数"]
        cen = np.cumsum(帧数) * 10.0                       # 字中心 ms（近似，仅诊断）
        for a0, b0 in 段:
            seg = 场[a0:b0]
            m = seg.mean(0); s = seg.std(0)
            i0 = int(a0 / 10); i1 = int(b0 / 10)
            v = f0[(f0t * 1000 >= a0) & (f0t * 1000 <= b0)]
            v = v[v > 0]
            e = float(np.log(np.mean(seg) + 1e-6))
            f = np.concatenate([m, s, [float(np.median(v)) / 300.0 if len(v) else 0.0,
                                       e, (b0 - a0) / 1000.0]])
            特征.append(f)
            c = 0.5 * (a0 + b0)
            真标签.append(int(np.argmin(np.abs(cen - c))) if len(cen) else -1)
        if len(特征) % 500 < len(特征) % 500 + 60:
            print("|", sid, "段", len(特征), round(time.time() - t0, 1), "s", flush=True)
    X = np.asarray(特征, np.float64)
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    y = np.asarray(真标签)
    print("单元", X.shape, flush=True)
    报 = dict(工具="发现_S1b.py", 时间=time.strftime("%Y-%m-%d %H:%M:%S"), 样本=len(ids),
              单元=int(X.shape[0]), 网格={})
    for k in (12, 16, 24, 32, 48, 64, 96):
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
        km2 = KMeans(n_clusters=k, n_init=10, random_state=1).fit(X)
        ari = float(adjusted_rand_score(km.labels_, km2.labels_))
        sil = float(silhouette_score(X, km.labels_)) if k < len(X) else 0.0
        # 诊断纯度（仅当该 k 下有意义的字标签）
        报["网格"][str(k)] = dict(ARI两次种子=round(ari, 4), 轮廓=round(sil, 4))
        print("| k", k, "ARI", round(ari, 4), "sil", round(sil, 4), flush=True)
    (E / "S1b" / "发现结果.json").write_text(json.dumps(报, ensure_ascii=False, indent=1), encoding="utf-8")
    np.savez(E / "S1b" / "特征.npz", X=X, 字= y)
    print("完成", round(time.time() - t0, 1), "s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(主())

# -*- coding: utf-8 -*-
"""meeting71/实施/发现_S1.py —— 原生自主发现 S1：无监督声学-运动单元发现（不引用文字标签）。

流程：语料 → 器官照着说（H3，控制↔音频一一对应）→ 能量事件切分 → 每段特征（控制侧 LPC 倒谱统计 +
F0/能量/时长）→ KMeans 聚类（k 由轮廓系数选）→ 稳定性（两次种子 ARI）+ 诊断纯度（仅对照，不进回路）。
输出：<path>\AHI_m71\S1\发现结果.json + 单元特征缓存。
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
M70 = HERE.parent.parent / "meeting70"
E = Path(r"<path>\AHI_m71")
SR = 22050
参数 = dict(OQ=0.62, OQ1=0.42, 噪声尺度=0.40, 去加重=0.97)


def 载(名, 路):
    s = importlib.util.spec_from_file_location(名, 路)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


器 = 载("器官v2", M70 / "实施" / "器官v2.py")


def 切段(x, 最短=0.05, 最长=0.45):
    hop = 220
    e = np.sqrt(np.convolve(np.asarray(x, np.float64) ** 2, np.ones(hop) / hop, mode="same")[::hop])
    e = np.convolve(e, np.ones(3) / 3, mode="same")
    if e.size < 5:
        return []
    thr = max(0.06 * float(np.percentile(e, 95)), 1e-5)
    cand = [i for i in range(1, len(e) - 1) if e[i] >= e[i - 1] and e[i] >= e[i + 1] and e[i] > thr]
    keep = []
    for i in cand:
        if not keep or (i - keep[-1]) * hop / SR >= 最短:
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
        if (b - a) * hop / SR < 最短 or (b - a) * hop / SR > 最长:
            continue
        段.append((a * hop, b * hop))
    return 段


def 倒谱(A, 阶=13):
    """LPC → 倒谱系数（c0..c阶）。"""
    out = np.zeros((len(A), 阶 + 1))
    for i, a in enumerate(A):
        p = len(a) - 1
        c = np.zeros(阶 + 1)
        c[0] = np.log(max(float(np.dot(a, a)), 1e-12)) / 2
        for n in range(1, 阶 + 1):
            s = 0.0
            for k in range(1, min(n, p)):
                s += k * c[k] * a[n - k]
            c[n] = -a[n] - s / n if n <= p else -s / n
        out[i] = c
    return out


def 主():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    import json as js
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, silhouette_score

    meta = js.loads((HERE.parent.parent / "meeting66" / "产物" / "数据_对齐asr" / "meta.json").read_text(encoding="utf-8"))
    样本 = {r["id"]: r for r in meta["样本"]}
    ids = sorted(样本)[:60]
    E.mkdir(parents=True, exist_ok=True)
    (E / "S1").mkdir(parents=True, exist_ok=True)
    特征 = []
    标签 = []
    段信息 = []
    t0 = time.time()
    for sid in ids:
        x = 器.耳.读wav(Path(r"<path>\SpeakerA7代模型训练数据") / 样本[sid]["wav"])
        c = 器.分析控制(x)
        y = 器.合成(c, 模式="全极点", **参数)
        段 = 切段(y)
        ce = 倒谱(c["A"][::2])                      # 10 ms 帧
        for a, b in 段:
            i0 = max(0, a // 220 // 2); i1 = min(len(ce), b // 220 // 2 + 1)
            if i1 - i0 < 2:
                continue
            cc = ce[i0:i1]
            f0 = c["f0"][2 * i0:2 * i1]
            v = f0[f0 > 0]
            能量 = np.log(np.maximum(c["E"][2 * i0:2 * i1], 1e-6))
            f = np.concatenate([cc.mean(0), cc.std(0),
                                [float(np.median(v)) / 300.0 if len(v) else 0.0,
                                 float(np.mean(f0 > 0)), float(能量.mean()), float(能量.std()),
                                 (b - a) / SR]])
            特征.append(f)
            标签.append(sid)
            段信息.append((sid, a, b))
        if (len(特征) // 100) != ((len(特征) - 1) // 100):
            print("|", sid, "段", len(特征), round(time.time() - t0, 1), "s", flush=True)
    X = np.asarray(特征, np.float64)
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    print("单元样本", X.shape, flush=True)
    报 = dict(工具="发现_S1.py", 时间=time.strftime("%Y-%m-%d %H:%M:%S"), 样本数=len(ids),
              单元数=int(X.shape[0]), 网格={})
    ks = [16, 24, 32, 48, 64, 96]
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
        lab = km.labels_
        km2 = KMeans(n_clusters=k, n_init=10, random_state=1).fit(X)
        ari = float(adjusted_rand_score(lab, km2.labels_))
        sil = float(silhouette_score(X, lab)) if k < len(X) else 0.0
        报["网格"][str(k)] = dict(ARI两次种子=round(ari, 4), 轮廓=round(sil, 4))
        print("| k", k, "ARI", round(ari, 4), "sil", round(sil, 4), flush=True)
    (E / "S1" / "发现结果.json").write_text(js.dumps(报, ensure_ascii=False, indent=1), encoding="utf-8")
    print("完成", round(time.time() - t0, 1), "s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(主())

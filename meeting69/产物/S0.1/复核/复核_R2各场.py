# -*- coding: utf-8 -*-
"""附加探针 4：J4 的 R²=0.55 是在『场未适』上测的；在选中的 V3『场』上重测同一探针。
另：J6 判据所用窗口在 V3 下的信息量（用于评估 S1 载体风险）。不改任何阈值。"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

M69 = Path(r"<workspace>/03_AHI\AHI4\meeting69")
复核 = M69 / "产物" / "S0.1" / "复核"


def 载(名, 路):
    s = importlib.util.spec_from_file_location(名, 路)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


前 = 载("f", M69 / "实施" / "耳蜗前端_S0_1.py")
清 = json.loads((M69 / "产物" / "S0" / "manifest.json").read_text(encoding="utf-8"))["真值"]
声调 = [g for g in 清 if g["类"] == "声调"]


def 岭(特征s, f0s, 折):
    X = np.asarray(特征s, np.float64); y = np.asarray(f0s, np.float64)
    R2 = []
    for 型 in sorted(set(折)):
        训 = 折 != 型; 测 = ~训
        if 测.sum() < 5 or 训.sum() < 5:
            continue
        μ = X[训].mean(axis=0); σ = X[训].std(axis=0) + 1e-9
        Xn = (X - μ) / σ
        A = np.concatenate([Xn, np.ones((len(Xn), 1))], axis=1)
        w = np.linalg.solve(A[训].T @ A[训] + 1e-2 * np.eye(A.shape[1]), A[训].T @ y[训])
        pred = A[测] @ w
        ss = float(np.sum((y[测] - y[测].mean()) ** 2))
        R2.append(1.0 - float(np.sum((y[测] - pred) ** 2)) / ss if ss > 1e-9 else 0.0)
    return round(float(np.mean(R2)), 4), [round(float(v), 4) for v in R2]


def 测(键, 变体):
    特 = []; 靶 = []; 折 = []
    for g in 声调:
        r = 前.分析(g["路径"], 变体=变体)
        f = r[键].astype(np.float64)
        nf = len(f)
        gtf = np.interp((np.arange(nf) * 0.001) - 0.3, np.linspace(0, 0.5, len(g["f0轮廓"])), g["f0轮廓"])
        sel = np.arange(nf) >= 310
        特.append(f[sel]); 靶.append(gtf[sel]); 折.append([g["标签"]] * int(sel.sum()))
    X = np.concatenate(特); y = np.concatenate(靶)
    折v = np.concatenate([np.array(z, dtype=object) for z in 折])
    return 岭(X, y, 折v)


出 = {}
for 键, 变体, 名 in (("场未适", "V3", "场未适(V3,未适应)"), ("场", "V3", "场(V3,选中)"),
                     ("场", "V1", "场(V1,除式5/100)"), ("场", "V2", "场(V2,除式5/200)")):
    r2, 折 = 测(键, 变体)
    出[名] = dict(R2=r2, 逐折=折)
    出[名]["特征尺度"] = None

# 场量级对比（同一声调）
g = 声调[0]
量 = {}
for v in ("V1", "V3"):
    r = 前.分析(g["路径"], 变体=v)
    量[v] = dict(场均值=round(float(np.abs(r["场"]).mean()), 6),
                 场未适均值=round(float(np.abs(r["场未适"]).mean()), 6),
                 比=round(float(np.abs(r["场"]).mean() / (np.abs(r["场未适"]).mean() + 1e-12)), 4))
出["量级"] = 量
出["说明"] = "同一探针（4 折留一声调、岭 λ=1e-2、sel≥0.31s）在不同场上的 R²；未改任何阈值"

(复核 / "复核_R2_各场.json").write_text(json.dumps(出, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps({k: v for k, v in 出.items() if k != "说明"}, ensure_ascii=False, indent=1))

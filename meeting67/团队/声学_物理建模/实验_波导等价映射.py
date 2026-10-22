# -*- coding: utf-8 -*-
"""meeting67/团队/声学_物理建模/实验_波导等价映射.py —— KL 波导 ↔ LPC 全极点的等价映射标定。

动机（R1 §4.3/R3）：探针里 P3 用「结点反射系数 = Levinson k、唇端闭合反射 +1」的朴素映射，
实测共振峰 LPC [2078,3661,4802] Hz vs KL [3295,7321,8097] Hz ⇒ **不等价**。

本件不改探针脚本，只做**离线标定实验**：在候选端点/结点口径里找能把 1/A(z) 的冲激响应
复现出来的组合。判据 = 归一化 IR 误差 与 前 3 共振峰偏差。

候选结构（p 段管，p−1 个内部结点，结点反射 c_i 取 Levinson k_i 或其子集）：
  端点： (r_G 声门反射, r_L 唇端反射) ∈ {(0,+1), (0,−1), (0,k_p), (+1,0), (+1,−1), (+1,+1), (−1,0)}
  结点： c_i = k_i (i=1..p−1)  ；或 c_i = k_{i+1}（错位一位）
  输出： 唇端前向波 F_out（压力口径，乘 (1+r_L)）

用法：
  & "<workspace>/\03_AHI\\.venv\\Scripts\\python.exe" "…\\实验_波导等价映射.py"
"""
from __future__ import annotations

import importlib.util
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import scipy.signal as ss

HERE = Path(__file__).resolve().parent
出目录 = HERE / "产物"


def _载探针():
    spec = importlib.util.spec_from_file_location("探针", str(HERE / "探针_渲染上限.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def 波导IR(k列: np.ndarray, 结点取: str, rG: float, rL: float, L: int = 2000) -> np.ndarray:
    """单帧常系数 KL 波导冲激响应。结点取: k全(p 个结点=p+1 段) / k(p−1) / k错位。"""
    p = len(k列)
    if 结点取 == "k全":
        c = np.asarray(k列, dtype=float)
    elif 结点取 == "k":
        c = np.asarray(k列[:p - 1], dtype=float)
    elif 结点取 == "k错位":
        c = np.asarray(k列[1:p], dtype=float)
    else:
        raise ValueError(结点取)
    nj = len(c)
    c = np.clip(c, -0.97, 0.97)
    F = np.zeros(nj)
    B = np.zeros(nj)
    y = np.zeros(L)
    for n in range(L):
        源 = 1.0 if n == 0 else 0.0
        F[0] = 源 + rG * B[0]          # 声门端反射后再注入（rG=0 即匹配）
        Fout = np.empty(nj)
        Bout = np.empty(nj)
        for i in range(nj):
            Fout[i] = (1.0 + c[i]) * F[i] - c[i] * B[i]
            Bout[i] = c[i] * F[i] + (1.0 - c[i]) * B[i]
        F[1:] = Fout[:-1]
        B[:-1] = Bout[1:]
        B[nj - 1] = rL * Fout[nj - 1]
        y[n] = (1.0 + rL) * Fout[nj - 1]
    return y


def 共振峰(yt, 个数=3, 上界=5000.0):
    H = np.abs(np.fft.rfft(yt, 4096))
    频 = np.fft.rfftfreq(4096, 1.0 / 22050.0)
    from scipy.signal import find_peaks
    idx, _ = find_peaks(H, distance=max(int(150 / (频[1] - 频[0])), 1))
    if idx.size == 0:
        return []
    序 = idx[np.argsort(H[idx])[::-1][:个数]]
    return sorted(round(float(频[i]), 1) for i in 序)


def _k到A(k列: np.ndarray) -> np.ndarray:
    """反射系数 → A(z)=1+Σa_j z^-j（Levinson 逆向递推）。"""
    p = len(k列)
    a = np.zeros(p + 1)
    a[0] = 1.0
    for i in range(1, p + 1):
        a旧 = a.copy()
        for j in range(1, i):
            a[j] = a旧[j] + k列[i - 1] * a旧[i - j]
        a[i] = k列[i - 1]
    return a


def 合成帧(个数: int = 9, 阶: int = 18) -> list:
    """确定性合成反射系数（|k|<0.9，模拟共振峰结构的 k 分布）→ [(名, 帧, A, k)]。"""
    rng = np.random.default_rng(20261021)
    出 = []
    for i in range(个数):
        k = rng.uniform(-0.85, 0.85, 阶) * np.linspace(1.0, 0.35, 阶) ** 0.5
        A = _k到A(k)
        if np.all(np.isfinite(A)):
            出.append(("合成%d" % i, 0, A, k))
    return 出


def 主(条数: int = 3) -> dict:
    P = _载探针()
    候选 = []
    for 结点取, (rG, rL) in itertools.product(
            ("k全", "k", "k错位"),
            [(0.0, 1.0), (0.0, -1.0), (0.0, "kp"), (1.0, 0.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 0.0)]):
        候选.append((结点取, rG, rL))
    报 = dict(说明="KL 波导端点/结点口径标定；判据=归一化 IR 误差 + 共振峰偏差",
              逐帧=[], 排名=[])
    累 = {c: [] for c in 候选}
    帧s = []
    源名 = "真语料"
    try:
        for 行号, 起点 in P.片段表[:条数]:
            x = P.读片段(行号, 起点)
            控 = P.提取控制(x)
            for t in (控["帧数"] // 3, 控["帧数"] // 2, (2 * 控["帧数"]) // 3):
                帧s.append((行号, t, 控["A"][t], 控["k"][t]))
    except Exception as e:
        源名 = "合成AR(p)（L: 不可用时的离线替代：%s）" % type(e).__name__
        帧s = 合成帧()
    报["控制源"] = 源名
    if True:
        for 行号, t, A, k列 in 帧s:
            if not np.all(np.isfinite(A)) or not np.all(np.isfinite(k列)):
                continue
            ir = ss.lfilter([1.0], A, np.eye(1, 2000, 0)[0])
            能 = float(np.sum(ir ** 2) + 1e-18)
            fL = 共振峰(ir)
            行 = dict(行号=行号, 帧=t, LPC共振峰=fL)
            for (结点取, rG, rL) in 候选:
                rLv = float(k列[-1]) if rL == "kp" else float(rL)
                y = 波导IR(k列, 结点取, float(rG), rLv)
                若有限 = bool(np.all(np.isfinite(y)) and np.abs(y).max() < 1e6)
                err = float(np.sum((y - ir) ** 2) / 能) if 若有限 else float("inf")
                # 增益最优（最小二乘标量）后再算相对误差，去掉纯幅度差
                if 若有限 and np.sum(y ** 2) > 0:
                    g = float(np.sum(ir * y) / np.sum(y ** 2))
                    errg = float(np.sum((g * y - ir) ** 2) / 能)
                else:
                    errg = float("inf")
                名 = "%s|rG=%+.0f,rL=%s" % (结点取, rG, ("kp" if rL == "kp" else "%+.0f" % rL))
                行[名] = dict(有限=若有限, 归一误差=round(err, 4) if np.isfinite(err) else None,
                              增益后误差=round(errg, 4) if np.isfinite(errg) else None,
                              共振峰=共振峰(y) if 若有限 else None)
                累[(结点取, rG, rL)].append(errg)
            报["逐帧"].append(行)
    排名 = []
    for c, vs in 累.items():
        if vs:
            排名.append(dict(口径="%s|rG=%+.0f,rL=%s" % (c[0], c[1], ("kp" if c[2] == "kp" else "%+.0f" % c[2])),
                            增益后误差中位=round(float(np.median(vs)), 4)))
    排名.sort(key=lambda r: r["增益后误差中位"])
    报["排名"] = 排名
    出目录.mkdir(parents=True, exist_ok=True)
    (出目录 / "波导等价映射.json").write_text(
        json.dumps(报, ensure_ascii=False, indent=1), encoding="utf-8")
    print("=== 候选口径排名（增益后归一 IR 误差，中位；越小越等价）===")
    for r in 排名:
        print("  %-28s %s" % (r["口径"], r["增益后误差中位"]))
    print("[落盘] %s" % (出目录 / "波导等价映射.json"))
    return 报


if __name__ == "__main__":
    sys.exit(0 if 主() else 0)


# -*- coding: utf-8 -*-
"""可行性检验：把「自身自由能动量」注入 m 的不动点基座后，线性读出能否预测 y；并仿真 verify_P1 的确切校准流程。"""
import sys
sys.path.insert(0, "/root/AHI")
import numpy as np
from ahi_theory.brain import Brain, R_REGIONS, D_REGION, K_SCHEMA

def auc(score, y):
    p, n = score[y == 1], score[y == 0]
    if len(p) == 0 or len(n) == 0: return float("nan")
    return float((p[:, None] > n[None, :]).mean() + 0.5 * (p[:, None] == n[None, :]).mean())

def verify_P1_pipeline(F, M):
    """完全复刻 verify_all.verify_P1 的校准实现（不改任何阈值）。"""
    _all = {"F": F, "m": M}
    _cut = 1500
    tr_tr = {k: v[:_cut] for k, v in _all.items()}
    _hold = {k: v[_cut:] for k, v in _all.items()}
    _F_all = _all["F"]
    _y_all = np.zeros(len(_F_all)); _y_all[:-1] = (_F_all[1:] > _F_all[:-1]).astype(float)
    Ftr = tr_tr["F"]; Str = tr_tr["m"]; ytr = _y_all[:_cut]
    Xtr = np.column_stack([Str, np.ones(len(Str))])
    mu_, sd_ = Xtr.mean(0), Xtr.std(0) + 1e-9
    Xs = (Xtr - mu_) / sd_
    _w0 = np.linalg.lstsq(Xs, ytr, rcond=None)[0]
    score_tr = Xs @ _w0
    qs = np.unique(np.quantile(score_tr, np.linspace(0, 1, 11)))
    if len(qs) < 3: qs = np.linspace(score_tr.min(), score_tr.max() + 1e-9, 11)
    idx_tr = np.clip(np.digitize(score_tr, qs[1:-1]), 0, len(qs) - 2)
    table = np.array([ytr[idx_tr == b].mean() if (idx_tr == b).sum() > 0 else float(ytr.mean())
                      for b in range(len(qs) - 1)])
    wts = np.array([max((idx_tr == b).sum(), 1) for b in range(len(qs) - 1)], dtype=float)
    vals = table.copy(); changed = True
    while changed:
        changed = False
        for b in range(len(vals) - 1):
            if vals[b] > vals[b + 1]:
                tot = wts[b] + wts[b + 1]
                vals[b] = vals[b + 1] = (vals[b]*wts[b] + vals[b+1]*wts[b+1]) / tot
                wts[b] = wts[b + 1] = tot; changed = True
    table = vals
    tr = _hold
    F_ = tr["F"]; S = tr["m"]; y = _y_all[_cut:]
    Sp = np.column_stack([S, np.ones(len(S))])
    sc = ((Sp - mu_) / sd_) @ _w0
    ib = np.clip(np.digitize(sc, qs[1:-1]), 0, len(table) - 1)
    conf = table[ib]; certainty = y
    edges = np.quantile(conf, np.linspace(0, 1, 11)); edges = np.unique(edges)
    if len(edges) < 3: edges = np.linspace(conf.min(), conf.max() + 1e-9, 11)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, len(edges) - 2)
    ece = 0.0
    for b in range(len(edges) - 1):
        mm = idx == b
        if mm.sum() > 0: ece += mm.mean() * abs(conf[mm].mean() - certainty[mm].mean())
    _p, _n = conf[certainty == 1], conf[certainty == 0]
    auc_c = float((_p[:, None] > _n[None, :]).mean()) if len(_p) and len(_n) else float("nan")
    return float(ece), auc_c, float(np.mean(np.abs(score_tr))), len(edges)-1

seed = 0
b = Brain(seed=seed)
tr = b.run(3000, seed=seed + 21, env_rho=0.85)
F = tr["F"]; T = len(F); z = tr["h"].reshape(T, -1); M0 = tr["m"]
print("【m 的动态范围】 |m| mean=%.4f  P(|m|>0.95)=%.4f  std over time mean=%.4f" %
      (np.abs(M0).mean(), np.mean(np.abs(M0) > 0.95), M0.std(0).mean()))

dF = np.diff(F)
dF_prev = np.concatenate([[0.0], dF])[:T]   # dF_prev[t] = F[t]-F[t-1]

print()
print("【仿真：m 注入 w_mom*dF_prev 后，线性读出对 y 的 AUC 及 verify_P1 校准结果】")
print(f"  {'w_mom':>8} {'AUC(m->y)全体':>14} {'tr':>7} {'ho':>7} {'ECE':>8} {'AUC(conf)':>10} {'箱数':>5}")
for w_mom in (0.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0):
    base = b.Wm_in @ z.T + b.w_slow * tr["h_slow"].T + w_mom * dF_prev[None, :]
    m = np.zeros(K_SCHEMA)
    Mv = np.zeros((T, K_SCHEMA))
    for t in range(T):
        mm = m.copy()
        for _ in range(60):
            mm = np.tanh(b.Wm @ mm + base[:, t])
        m = mm; Mv[t] = mm
    y = np.zeros(T); y[:-1] = (F[1:] > F[:-1]).astype(float)
    yv = y[:-1]
    X = np.column_stack([Mv, np.ones(T)]); Xv = X[:-1]
    w = np.linalg.lstsq(Xv, yv, rcond=None)[0]
    ece, a_c, _, nb = verify_P1_pipeline(F, Mv)
    print(f"  {w_mom:8.0f} {auc(Xv@w, yv):14.4f} {auc(Xv[:1500]@w, yv[:1500]):7.4f} "
          f"{auc(Xv[1500:]@w, yv[1500:]):7.4f} {ece:8.4f} {a_c:10.4f} {nb:5d}")

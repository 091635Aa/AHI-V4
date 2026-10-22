
# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, "/root/AHI")
import numpy as np
from ahi_theory.brain import Brain, R_REGIONS, D_REGION, K_SCHEMA

def auc(score, y):
    p, n = score[y == 1], score[y == 0]
    if len(p) == 0 or len(n) == 0: return float("nan")
    return float((p[:, None] > n[None, :]).mean() + 0.5 * (p[:, None] == n[None, :]).mean())

def P1pipe(F, M, verbose=False):
    _cut = 1500
    _y_all = np.zeros(len(F)); _y_all[:-1] = (F[1:] > F[:-1]).astype(float)
    Xtr = np.column_stack([M[:_cut], np.ones(_cut)])
    mu_, sd_ = Xtr.mean(0), Xtr.std(0) + 1e-9
    Xs = (Xtr - mu_) / sd_
    _w0 = np.linalg.lstsq(Xs, _y_all[:_cut], rcond=None)[0]
    score_tr = Xs @ _w0
    qs = np.unique(np.quantile(score_tr, np.linspace(0, 1, 11)))
    if len(qs) < 3: qs = np.linspace(score_tr.min(), score_tr.max() + 1e-9, 11)
    idx_tr = np.clip(np.digitize(score_tr, qs[1:-1]), 0, len(qs) - 2)
    table = np.array([_y_all[:_cut][idx_tr == b].mean() if (idx_tr == b).sum() > 0 else float(_y_all[:_cut].mean())
                      for b in range(len(qs) - 1)])
    wts = np.array([max((idx_tr == b).sum(), 1) for b in range(len(qs) - 1)], dtype=float)
    vals = table.copy(); ch = True
    while ch:
        ch = False
        for b in range(len(vals) - 1):
            if vals[b] > vals[b+1]:
                tot = wts[b]+wts[b+1]
                vals[b] = vals[b+1] = (vals[b]*wts[b]+vals[b+1]*wts[b+1])/tot
                wts[b] = wts[b+1] = tot; ch = True
    table = vals
    S = M[_cut:]; y = _y_all[_cut:]
    sc = ((np.column_stack([S, np.ones(len(S))]) - mu_)/sd_) @ _w0
    ib = np.clip(np.digitize(sc, qs[1:-1]), 0, len(table)-1)
    conf = table[ib]
    edges = np.unique(np.quantile(conf, np.linspace(0, 1, 11)))
    if len(edges) < 3: edges = np.linspace(conf.min(), conf.max()+1e-9, 11)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, len(edges)-2)
    ece = 0.0
    for b in range(len(edges)-1):
        mm = idx == b
        if mm.sum() > 0: ece += mm.mean()*abs(conf[mm].mean()-y[mm].mean())
    _p, _n = conf[y == 1], conf[y == 0]
    a_c = float((_p[:, None] > _n[None, :]).mean()) if len(_p) and len(_n) else float("nan")
    if verbose:
        print("   [score] train mean=%.4f sd=%.4f | holdout mean=%.4f sd=%.4f" % (score_tr.mean(), score_tr.std(), sc.mean(), sc.std()))
        print("   [qs] ", np.round(qs, 4))
        print("   [table] ", np.round(table, 4))
        print("   [holdout bin 占用] ", np.bincount(ib, minlength=len(table)))
        print("   [conf 唯一值] ", np.unique(np.round(conf, 4)))
        print("   [y holdout 基率] %.4f" % y.mean())
    return float(ece), a_c, len(edges)-1, auc(score_tr, _y_all[:_cut]), auc(sc, y)

seed = 0
b = Brain(seed=seed)
tr = b.run(3000, seed=seed + 21, env_rho=0.85)
F = tr["F"]; T = len(F); z = tr["h"].reshape(T, -1)
dF = np.diff(F); dFp = np.concatenate([[0.0], dF])[:T]
# 自身自由能的**去趋势动量**（相对自身近期基线）
ema = np.zeros(T); a = 0.02
for t in range(1, T): ema[t] = (1-a)*ema[t-1] + a*dFp[t]
momz = dFp - ema

def build(w_mom, feat):
    base = b.Wm_in @ z.T + b.w_slow * tr["h_slow"].T + w_mom * feat[None, :]
    m = np.zeros(K_SCHEMA); Mv = np.zeros((T, K_SCHEMA))
    for t in range(T):
        mm = m.copy()
        for _ in range(60): mm = np.tanh(b.Wm @ mm + base[:, t])
        m = mm; Mv[t] = mm
    return Mv

# 校正索引：base_t 用 z_{t-1}, h_slow_{t-1}
base0 = b.Wm_in @ np.roll(z, 1, axis=0).T + b.w_slow * np.roll(tr["h_slow"], 1, axis=0).T
m = np.zeros(K_SCHEMA); Mv = np.zeros((T, K_SCHEMA))
for t in range(T):
    mm = m.copy()
    for _ in range(60): mm = np.tanh(b.Wm @ mm + base0[:, t])
    m = mm; Mv[t] = mm
print("【索引校验】重放 m 与真实 m 的最大偏差 =", np.abs(Mv - tr["m"]).max().round(8))

print()
print("【动量特征自身的 AUC（用已知特征预测 y）】")
y = np.zeros(T); y[:-1] = (F[1:] > F[:-1]).astype(float)
yv = y[:-1]
for nm, f in (("dF_prev", dFp), ("去趋势动量 mom", momz)):
    print(f"  {nm:<16} 全体AUC={auc(f[:-1], yv):.4f}  训练段={auc(f[:1500], yv[:1500]):.4f}  留出={auc(f[1500:-1], yv[1500:]):.4f}"
          f"  mean_tr={f[:1500].mean():+.6f} mean_ho={f[1500:-1].mean():+.6f}")

print()
print("【注入 m 后跑 verify_P1 确切流程】")
for nm, f in (("dF_prev", dFp), ("momz", momz)):
    print(f" --- 特征 = {nm} ---")
    print(f"  {'w':>8} {'ECE':>9} {'AUC(conf)':>10} {'箱':>4} {'AUCscore_tr':>12} {'AUCscore_ho':>12}")
    for w in (0.0, 300.0, 3000.0, 20000.0, 100000.0):
        Mv = build(w, f)
        ece, ac, nb, a1, a2 = P1pipe(F, Mv)
        print(f"  {w:8.0f} {ece:9.4f} {ac:10.4f} {nb:4d} {a1:12.4f} {a2:12.4f}")

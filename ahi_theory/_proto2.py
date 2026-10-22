
# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, "/root/AHI")
import numpy as np
from ahi_theory.brain import Brain, R_REGIONS, D_REGION, K_SCHEMA

def auc(score, y):
    p, n = score[y == 1], score[y == 0]
    if len(p) == 0 or len(n) == 0: return float("nan")
    return float((p[:, None] > n[None, :]).mean() + 0.5 * (p[:, None] == n[None, :]).mean())

seed = 0
b = Brain(seed=seed)
tr = b.run(3000, seed=seed + 21, env_rho=0.85)
h, g, s, L = tr["h"], tr["g"], tr["s"], tr["L"]
T = len(h)
gs = np.zeros((T, R_REGIONS)); prev = np.zeros(R_REGIONS)
for t in range(T):
    gs[t] = 0.97 * prev + 0.03 * b.gs_amp * np.tanh(b.Wgs @ g[t]); prev = gs[t]

e = np.zeros(T)
for t in range(T - 1):
    ctx = np.linalg.norm(h[t], axis=1)
    s_n = np.tanh(b.Ws @ s[t] + b.Ws_in @ np.concatenate([g[t], ctx]))
    raw = (b.Wg @ np.concatenate([ctx, s_n])) / np.sqrt(R_REGIONS + K_SCHEMA) + b.bg
    graw = 0.5 + 0.5 * np.tanh((raw + b.gs_weight * gs[t]) * (b.gain_scale + 0.6 * L[t][0]))
    g_n = 0.85 * graw + 0.15 * g[t]
    gate = (b.GL @ L[t]).reshape(R_REGIONS, 3); hp = np.empty_like(h[t]); hc = b.Wc @ h[t]
    for i in range(R_REGIONS):
        inp = b.Bin[i] @ (g_n[i] * hc[i].ravel()[:D_REGION])
        exc, inh, tau = gate[i]; leak = 0.2 * (1.0 + 0.3 * tau)
        hp[i] = (1 - leak) * h[t][i] + leak * np.tanh(b.A[i] @ h[t][i] + inp * (1.0 + 0.5 * exc) - 0.3 * inh * h[t][i])
    e[t + 1] = float(np.mean((h[t + 1] - hp) ** 2))
e[0] = e[1]

cx = 0.5 * (tr["m"] ** 2).sum(1) / K_SCHEMA
pe = tr["F"] - cx
F0 = tr["F"]
y0 = np.zeros(T); y0[:-1] = (F0[1:] > F0[:-1]).astype(float)

print("【对照】原始 F 目标下 e_t 的 AUC =", round(auc(e[:-1], y0[:-1]), 4))

print()
print("【F' = cx + pe + w*e 各种 w 下的目标可预测性】")
hdr = f"  {'w_self':>9} {'基率tr':>7} {'基率ho':>7} {'AUC(e_t)全体':>12} {'AUC(e_t)ho':>11} {'线性读出 tr':>11} {'ho':>8} {'Fmin':>7} {'Fmax':>7}"
print(hdr)
for w in (0.0, 500.0, 1000.0, 2000.0, 4000.0):
    Fp = cx + pe + w * e
    y = np.zeros(T); y[:-1] = (Fp[1:] > Fp[:-1]).astype(float)
    yv = y[:-1]
    a_all = auc(e[:-1], yv); a_ho = auc(e[1500:-1], yv[1500:])
    X = np.column_stack([e, tr["m"], g, tr["h_slow"]])[:-1]
    mu, sd = X[:1500].mean(0), X[:1500].std(0) + 1e-12
    Xs = (X - mu) / sd
    Fd = np.column_stack([Xs[:1500], np.ones(1500)])
    lam = 1.0
    w_ = np.linalg.solve(Fd.T @ Fd + lam * np.eye(Fd.shape[1]), Fd.T @ yv[:1500])
    sc_ho = np.column_stack([Xs[1500:], np.ones(len(Xs) - 1500)]) @ w_
    print(f"  {w:9.0f} {yv[:1500].mean():7.3f} {yv[1500:].mean():7.3f} {a_all:12.4f} {a_ho:11.4f} "
          f"{auc(Fd @ w_, yv[:1500]):11.4f} {auc(sc_ho, yv[1500:]):8.4f} {Fp.min():7.4f} {Fp.max():7.4f}")

print()
print("【若 F' = cx + w*e （自预测误差作为 inaccuracy，去掉外部驱动的 pred_err）】")
for w in (1000.0, 2000.0, 4000.0):
    Fp = cx + w * e
    y = np.zeros(T); y[:-1] = (Fp[1:] > Fp[:-1]).astype(float)
    yv = y[:-1]
    X = np.column_stack([e, tr["m"], g, tr["h_slow"]])[:-1]
    mu, sd = X[:1500].mean(0), X[:1500].std(0) + 1e-12
    Xs = (X - mu) / sd
    Fd = np.column_stack([Xs[:1500], np.ones(1500)])
    w_ = np.linalg.solve(Fd.T @ Fd + np.eye(Fd.shape[1]), Fd.T @ yv[:1500])
    sc_ho = np.column_stack([Xs[1500:], np.ones(len(Xs) - 1500)]) @ w_
    print(f"  w={w:6.0f} 基率tr={yv[:1500].mean():.3f} ho={yv[1500:].mean():.3f} "
          f"AUC(e_t)={auc(e[:-1], yv):.4f}  读出 tr={auc(Fd @ w_, yv[:1500]):.4f} ho={auc(sc_ho, yv[1500:]):.4f}")

print()
print("【e_t 对未来 e 的预测（AR 结构）】")
print(f"  corr(e_t, e_t+1) = {np.corrcoef(e[:-1], e[1:])[0,1]:+.4f}")
print(f"  corr(de_t, de_t+1) = {np.corrcoef(np.diff(e)[:-1], np.diff(e)[1:])[0,1]:+.4f}")
print(f"  用 e_t 线性预测 de_t 的 R2 = ", end="")
Xe = np.column_stack([e[:-2], np.ones(T-2)]); de = np.diff(e)[:T-2]
ww = np.linalg.lstsq(Xe, de, rcond=None)[0]; pr = Xe @ ww
print(round(float(1 - ((de-pr)**2).sum()/((de-de.mean())**2).sum()), 4))

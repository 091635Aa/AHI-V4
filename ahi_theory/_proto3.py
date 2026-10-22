
# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, "/root/AHI")
import numpy as np
from ahi_theory.brain import Brain, K_SCHEMA

def auc(score, y):
    p, n = score[y == 1], score[y == 0]
    if len(p) == 0 or len(n) == 0: return float("nan")
    return float((p[:, None] > n[None, :]).mean() + 0.5 * (p[:, None] == n[None, :]).mean())

seed = 0
b = Brain(seed=seed)
tr = b.run(3000, seed=seed + 21, env_rho=0.85)
F = tr["F"]; T = len(F)
dF = np.diff(F)                       # dF[i] = F[i+1]-F[i]
y = np.zeros(T); y[:-1] = (dF > 0).astype(float)     # y[t] = 1[dF[t]>0]
yv = y[:-1]                            # 长度 T-1, 索引 t=0..T-2 对应 dF[t]
prev_dF = np.concatenate([[np.nan], dF])[:-1]        # prev_dF[t] = dF[t-1]

print("【核心检验】用已知的 dF_{t-1} 预测 y_t = 1[dF_t>0]")
ok = ~np.isnan(prev_dF)
print(f"  全体 AUC = {auc(prev_dF[ok], yv[ok]):.4f}")
for s, e_ in ((1, 1400), (1400, 2999)):
    m = np.arange(s, e_)
    print(f"  t={s}-{e_}: 基率={yv[m].mean():.3f}  AUC={auc(prev_dF[m], yv[m]):.4f}")

print()
print("【滞后关系】")
for lag in (1, 2, 3, 5, 10):
    a = dF[:len(dF)-lag]; bb = yv[lag:lag+len(a)]
    print(f"  用 dF_{{t-{lag}}} 预测 y_t : AUC = {auc(a, bb):.4f}   corr={np.corrcoef(a, bb)[0,1]:+.4f}")

print()
print("【dF 的统计稳定性】")
for s in range(0, 2800, 700):
    seg = dF[s:s+700]
    print(f"  t={s}-{s+700}: mean={seg.mean():+.6f} std={seg.std():.6f} ac1={np.corrcoef(seg[:-1],seg[1:])[0,1]:+.4f} P(>0)={np.mean(seg>0):.3f}")

print()
print("【线性读出：m + h_slow + g + dF_{t-1} -> y, 训练0-1500 测试1500-2998】")
def ridge_auc(X, yv, cut=1500, lam=1.0):
    mu, sd = X[:cut].mean(0), X[:cut].std(0) + 1e-12
    Xs = (X - mu)/sd
    Fd = np.column_stack([Xs[:cut], np.ones(cut)])
    w = np.linalg.solve(Fd.T@Fd + lam*np.eye(Fd.shape[1]), Fd.T@yv[:cut])
    Fh = np.column_stack([Xs[cut:], np.ones(len(Xs)-cut)])
    return auc(Fd@w, yv[:cut]), auc(Fh@w, yv[cut:])

pd_ = np.nan_to_num(prev_dF, nan=0.0)
cands = {
    "m 仅":            tr["m"][:-1],
    "m+h_slow":        np.column_stack([tr["m"], tr["h_slow"]])[:-1],
    "m+g+h_slow":      np.column_stack([tr["m"], tr["g"], tr["h_slow"]])[:-1],
    "dF_prev 仅":      pd_[:-1, None],
    "m+dF_prev":       np.column_stack([tr["m"], pd_])[:-1],
    "m+h_slow+dF_prev":np.column_stack([tr["m"], tr["h_slow"], pd_])[:-1],
}
for k, X in cands.items():
    a1, a2 = ridge_auc(X, yv)
    print(f"  {k:<20} 训练={a1:.4f}  留出={a2:.4f}")

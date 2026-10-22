
# -*- coding: utf-8 -*-
"""P1 诊断复现：自我状态是否预测「下一步我是否会变差」"""
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
F = tr["F"]
y = np.zeros(len(F)); y[:-1] = (F[1:] > F[:-1]).astype(float)

print("=" * 90)
print("目标 y_t = 1[F_{t+1} > F_t]   （下一步我是否会变差）")
print(f"  F: min={F.min():.4f} max={F.max():.4f} mean={F.mean():.4f}")
print(f"  y 全体基率 = {y.mean():.4f}   前1500 = {y[:1500].mean():.4f}  后1500 = {y[1500:].mean():.4f}")
dF = np.diff(F)
print(f"  dF: mean={dF.mean():+.5f} std={dF.std():.5f} ac1={np.corrcoef(dF[:-1], dF[1:])[0,1]:+.4f}")
print(f"  F  自相关 ac1={np.corrcoef(F[:-1], F[1:])[0,1]:+.4f}")

feats = {}
feats["m (16)"] = tr["m"]
feats["h (288)"] = tr["h"].reshape(len(F), -1)
feats["h_slow (16)"] = tr["h_slow"]
feats["g (12)"] = tr["g"]
feats["s (16)"] = tr["s"]
feats["conf (12)"] = tr["conf"]
feats["L (5)"] = tr["L"]
feats["m+h"] = np.column_stack([tr["m"], tr["h"].reshape(len(F), -1)])
feats["m+h_slow"] = np.column_stack([tr["m"], tr["h_slow"]])
feats["m+g+h_slow"] = np.column_stack([tr["m"], tr["g"], tr["h_slow"]])
feats["全部"] = np.column_stack([tr["m"], tr["h"].reshape(len(F), -1), tr["g"], tr["h_slow"], tr["s"]])

print("\n" + "=" * 90)
print("【单特征对 y 的 AUC】（AUC=0.5 无信息；<0.5 反向）")
for k, X in feats.items():
    print(f"  {k:<18} AUC={auc(X[-1-1500:-1, 0] if X.shape[1]==1 else X[:-1,0], y[:-1]):.4f}  (仅首维)")

def lin_auc(X, y, label):
    X = X[:-1]
    yv = y[:-1]
    Fd = np.column_stack([X, np.ones(len(X))])
    w = np.linalg.lstsq(Fd, yv, rcond=None)[0]
    s = Fd @ w
    a_all = auc(s, yv)
    a_tr = auc(s[:1500], yv[:1500])
    a_ho = auc(s[1500:], yv[1500:])
    # 训练段拟合、留出段测试
    Xtr = np.column_stack([X[:1500], np.ones(1500)])
    wtr = np.linalg.lstsq(Xtr, yv[:1500], rcond=None)[0]
    s_ho = np.column_stack([X[1500:], np.ones(len(X)-1500)]) @ wtr
    a_gen = auc(s_ho, yv[1500:])
    print(f"  {label:<18} 全体={a_all:.4f}  训练段={a_tr:.4f}  留出(in-sample)={a_ho:.4f}  训练->留出={a_gen:.4f}")

print("\n" + "=" * 90)
print("【线性读出对 y 的 AUC】")
for k, X in feats.items():
    lin_auc(X, y, k)

print("\n" + "=" * 90)
print("【F 分解】")
pred_err = []
complexity = []
b2 = Brain(seed=seed)
tr2 = b2.run(3000, seed=seed + 21, env_rho=0.85)
print(f"  F 变化量占比: complexity = 0.5*sum(m^2)/K")
cx = 0.5 * (tr2["m"] ** 2).sum(1) / K_SCHEMA
print(f"  complexity: mean={cx.mean():.5f} std={cx.std():.5f}  d_std={np.diff(cx).std():.5f}")
print(f"  pred_err  = F - complexity: mean={(F-cx).mean():.5f} std={(F-cx).std():.5f} d_std={np.diff(F-cx).std():.5f}")
print(f"  corr(complexity, pred_err) = {np.corrcoef(cx, F-cx)[0,1]:+.4f}")
print(f"  dF 的可预测性：ac1(dF)={np.corrcoef(np.diff(F)[:-1], np.diff(F)[1:])[0,1]:+.4f}")

print("\n" + "=" * 90)
print("【m 的各维对 y 的 AUC（找最好的一维）】")
best = sorted([(abs(auc(tr["m"][:-1, i], y[:-1]) - 0.5), i, auc(tr["m"][:-1, i], y[:-1])) for i in range(K_SCHEMA)], reverse=True)
for d, i, a in best[:5]:
    print(f"  m[{i:2d}] AUC={a:.4f}")

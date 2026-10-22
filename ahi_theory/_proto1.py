
# -*- coding: utf-8 -*-
"""离线原型：用引擎自身已知动力学做一步前向自预测，看预测误差是否携带「下一步变差」的信息。"""
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
# 重放 g_slow（确定性，由 g 完全决定）
gs = np.zeros((T, R_REGIONS)); prev = np.zeros(R_REGIONS)
for t in range(T):
    gs[t] = 0.97 * prev + 0.03 * b.gs_amp * np.tanh(b.Wgs @ g[t]); prev = gs[t]

def forward_predict(t):
    ctx = np.linalg.norm(h[t], axis=1)
    s_n = np.tanh(b.Ws @ s[t] + b.Ws_in @ np.concatenate([g[t], ctx]))
    x = np.concatenate([ctx, s_n])
    raw = (b.Wg @ x) / np.sqrt(len(x)) + b.bg
    graw = 0.5 + 0.5 * np.tanh((raw + b.gs_weight * gs[t]) * (b.gain_scale + 0.6 * L[t][0]))
    g_n = 0.85 * graw + 0.15 * g[t]
    gate = (b.GL @ L[t]).reshape(R_REGIONS, 3)
    hp = np.empty_like(h[t]); hc = b.Wc @ h[t]
    for i in range(R_REGIONS):
        inp = b.Bin[i] @ (g_n[i] * hc[i].ravel()[:D_REGION])
        exc, inh, tau = gate[i]
        leak = 0.2 * (1.0 + 0.3 * tau)
        drive = np.tanh(b.A[i] @ h[t][i] + inp * (1.0 + 0.5 * exc) - 0.3 * inh * h[t][i])
        hp[i] = (1 - leak) * h[t][i] + leak * drive
    return hp, g_n

gp_err = []; e = np.zeros(T); res = np.zeros((T, R_REGIONS, D_REGION))
for t in range(T - 1):
    hp, g_n = forward_predict(t)
    res[t + 1] = h[t + 1] - hp
    e[t + 1] = float(np.mean(res[t + 1] ** 2))
    gp_err.append(np.abs(g_n - g[t + 1]).mean())
e[0] = e[1]
print("【自预测校验】预测的 g_{t+1} 与真实 g_{t+1} 的平均绝对差 =", np.mean(gp_err).round(6))
print("【自预测校验】e 均值 =", e.mean().round(8), " 标准差 =", e.std().round(8))
print("  残差范数 / h 范数 =", (np.linalg.norm(res[1:]) / np.linalg.norm(h[1:])).round(5))

print()
print("【e 的时间结构】")
for st in range(0, 3000, 500):
    print(f"  t={st}-{st+500}: e mean={e[st:st+500].mean():.8f} std={e[st:st+500].std():.8f}")
de = np.diff(e)
print(f"  ac1(e) = {np.corrcoef(e[:-1], e[1:])[0,1]:+.4f}   ac1(de) = {np.corrcoef(de[:-1], de[1:])[0,1]:+.4f}")
print(f"  std(e)={e.std():.8f}  std(de)={de.std():.8f}")

# 新的 F'
cx = 0.5 * (tr["m"] ** 2).sum(1) / K_SCHEMA
pe = tr["F"] - cx
for w in (0.0, 200.0, 500.0, 1000.0, 2000.0):
    Fp = cx + pe + w * e
    y = np.zeros(T); y[:-1] = (Fp[1:] > Fp[:-1]).astype(float)
    yy = y[:-1]
    # 用 e_t 直接预测
    a_e = auc(e[:-1], yy)
    # 用 (e_t, de_t, m_t) 线性读出：训练 0-1500, 测试 1500-2998
    X = np.column_stack([e, np.concatenate([[0], de[:-1]]), tr["m"], g, tr["h_slow"]])
    X = X[:-1]; yv = yy
    mu, sd = X[:1500].mean(0), X[:1500].std(0) + 1e-12
    Xs = (X - mu) / sd
    Fd = np.column_stack([Xs[:1500], np.ones(1500)])
    # 岭回归
    lam = 1.0
    A_ = Fd.T @ Fd + lam * np.eye(Fd.shape[1])
    w_ = np.linalg.solve(A_, Fd.T @ yv[:1500])
    sc_ho = np.column_stack([Xs[1500:], np.ones(len(Xs) - 1500)]) @ w_
    a_ho = auc(sc_ho, yv[1500:])
    a_tr = auc(Fd @ w_, yv[:1500])
    print(f"  w_self={w:7.1f}: 基率={yv.mean():.3f}  AUC(e_t)={a_e:.4f}  线性读出 训练={a_tr:.4f} 留出={a_ho:.4f}   F'范围[{Fp.min():.3f},{Fp.max():.3f}]")

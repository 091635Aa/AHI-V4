
# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, "/root/AHI")
import numpy as np
from ahi_theory.brain import Brain, K_SCHEMA

def auc(s, y):
    p, n = s[y == 1], s[y == 0]
    if len(p)==0 or len(n)==0: return float('nan')
    return float((p[:, None] > n[None, :]).mean() + 0.5*(p[:, None] == n[None, :]).mean())

print("=== 纯净基线：关闭全部自预测通道（w_mom=w_fwd=w_surp=w_self=0）===")
b = Brain(seed=0, w_mom=0.0, w_fwd=0.0, w_surp=0.0, w_self=0.0)
t = b.run(3000, seed=21, env_rho=0.85)
F = t["F"]; T = len(F)
y = np.zeros(T); y[:-1] = (F[1:] > F[:-1]).astype(float); yv = y[:-1]
dF = np.diff(F); dFp = np.concatenate([[0.0], dF])[:T]
ema = np.zeros(T); a = 0.02
for i in range(1, T): ema[i] = (1-a)*ema[i-1] + a*dFp[i]
mom = dFp - ema

print(f"  F: mean={F.mean():.5f} std={F.std():.5f} min={F.min():.5f} max={F.max():.5f}")
print(f"  self_err: mean={t['self_err'].mean():.4e} std={t['self_err'].std():.4e}  (前50步均值 {t['self_err'][:50].mean():.4e})")
cx = 0.5*(t['m']**2).sum(1)/K_SCHEMA
print(f"  cx(m): mean={cx.mean():.4f}   pred_err: mean={(F-cx).mean():.4f}")
print(f"  |m| mean={np.abs(t['m']).mean():.4f}  P(|m|>0.95)={np.mean(np.abs(t['m'])>0.95):.4f}")
print(f"  y 基率={yv.mean():.4f} (前1500 {yv[:1500].mean():.4f} / 后1499 {yv[1500:].mean():.4f})")
print(f"  dF: std={dF.std():.6f}  ac1={np.corrcoef(dF[:-1],dF[1:])[0,1]:+.4f}")
for s0 in (0,700,1400,2100):
    seg=dF[s0:s0+700]; print(f"    t={s0}-{s0+700}: mean={seg.mean():+.7f} std={seg.std():.6f} ac1={np.corrcoef(seg[:-1],seg[1:])[0,1]:+.4f} P(>0)={np.mean(seg>0):.3f}")
print(f"  AUC(dF_prev->y)={auc(dFp[:-1],yv):.4f}   AUC(mom->y)={auc(mom[:-1],yv):.4f}   AUC(-mom->y)={auc(-mom[:-1],yv):.4f}")
print(f"  corr(dF_prev, y)={np.corrcoef(dFp[:-1],yv)[0,1]:+.4f}  corr(mom,y)={np.corrcoef(mom[:-1],yv)[0,1]:+.4f}")

def lin(X, yv, lam=0.0):
    mu, sd = X[:1500].mean(0), X[:1500].std(0)+1e-12
    Xs = (X-mu)/sd
    A_ = Xs[:1500].T@Xs[:1500] + lam*np.eye(Xs.shape[1])
    w = np.linalg.solve(A_, Xs[:1500].T@yv[:1500])
    return auc(Xs[:1500]@w, yv[:1500]), auc(Xs[1500:]@w, yv[1500:])
print()
print("  【当前 m 对 y 的线性读出】tr=%.4f ho=%.4f" % lin(t['m'][:-1], yv))

print()
print("=== 自预测误差 e 的时间结构（u_scale=0.01）===")
e = t['self_err'][:,0]
print(f"  e: mean={e.mean():.4e} std={e.std():.4e} ac1={np.corrcoef(e[:-1],e[1:])[0,1]:+.4f}")
print(f"  分段均值: " + " ".join(f"{e[s:s+500].mean():.3e}" for s in range(0,3000,500)))
print(f"  前20步: {np.round(e[:20],10)}")

print()
print("=== 前向自预测是否真的准（诊断：e 是否只由外部输入 u 决定）===")
# 复算 u
rng = np.random.default_rng(21); u_env = np.zeros(24); Us=[]
for i in range(3000):
    u_env = 0.85*u_env + np.sqrt(1-0.85**2)*rng.normal(size=24); Us.append(u_env*0.01)
Us = np.array(Us)
un = np.linalg.norm(Us, axis=1)
print(f"  corr(e_t, |u_t|^2) = {np.corrcoef(e, un**2)[0,1]:+.4f}")
print(f"  e_t 与 |u_t|^2 的比值均值 = {np.mean(e/(un**2+1e-30)):.4f}")

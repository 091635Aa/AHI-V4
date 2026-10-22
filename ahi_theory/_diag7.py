
# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, "/root/AHI")
import numpy as np
from ahi_theory.brain import Brain, K_SCHEMA
def auc(s, y):
    p, n = s[y==1], s[y==0]
    if len(p)==0 or len(n)==0: return float('nan')
    return float((p[:,None]>n[None,:]).mean()+0.5*(p[:,None]==n[None,:]).mean())

b = Brain(seed=0, w_mom=0.0, w_fwd=0.0, w_surp=0.0, w_self=0.0)
tr = b.run(3000, seed=21, env_rho=0.85)
F=tr["F"]; T=len(F)
y=np.zeros(T); y[:-1]=(F[1:]>F[:-1]).astype(float); yv=y[:-1]
C = 0.5*(tr["m"]**2).sum(1)/K_SCHEMA; P = F - C
dvC = np.concatenate([[0.0],np.diff(C)])[:T]
dvP = np.concatenate([[0.0],np.diff(P)])[:T]
dvF = np.concatenate([[0.0],np.diff(F)])[:T]

def momz(dv, a=0.02):
    e1=np.zeros(T); e2=np.zeros(T)
    for i in range(1,T):
        e1[i]=(1-a)*e1[i-1]+a*dv[i]
        e2[i]=((1-a)*e2[i-1]+a*abs(dv[i])) if e2[i-1]>0 else abs(dv[i])
    return (dv-e1)/(e2+1e-12)

n = T-2
def ev(feat):
    X = feat[:n]; Y = yv[1:1+n]
    return auc(X, Y), auc(X[:1500], Y[:1500]), auc(X[1500:], Y[1500:])

print("【ΔF 分解：哪一部分携带可预测性】(特征 lag1 -> y_t)")
for nm, dv in (("d_complexity", dvC), ("d_pred_err", dvP), ("d_F", dvF)):
    a1,a2,a3 = ev(dv)
    print(f"  {nm:<14} std={dv.std():.6f} ac1={np.corrcoef(dv[1:-1],dv[2:])[0,1]:+.4f}  AUC={a1:.4f} tr={a2:.4f} ho={a3:.4f}")

print()
print("【去趋势+自归一化 momz】")
for nm, dv in (("d_complexity", dvC), ("d_pred_err", dvP), ("d_F", dvF)):
    a1,a2,a3 = ev(momz(dv))
    print(f"  {nm:<14} momz AUC={a1:.4f} tr={a2:.4f} ho={a3:.4f}")

print()
print("【其它自主动力学量的动量】")
se = tr["self_err"][:,0]; dvse = np.concatenate([[0.0],np.diff(se)])[:T]
a1,a2,a3 = ev(se); print(f"  self_err 本身 AUC={a1:.4f} tr={a2:.4f} ho={a3:.4f}")
a1,a2,a3 = ev(momz(se)); print(f"  momz(self_err) AUC={a1:.4f} tr={a2:.4f} ho={a3:.4f}")
a1,a2,a3 = ev(P); print(f"  pred_err 本身 AUC={a1:.4f} tr={a2:.4f} ho={a3:.4f}")
for nm, v in (("|h|", np.linalg.norm(tr["h"],axis=2).mean(1)), ("g", tr["g"].mean(1))):
    dv = np.concatenate([[0.0],np.diff(v)])[:T]
    a1,a2,a3 = ev(momz(dv)); print(f"  momz({nm}) AUC={a1:.4f} tr={a2:.4f} ho={a3:.4f}")
print()
print(f"  corr(momz(dF), momz(dP)) = {np.corrcoef(momz(dvF), momz(dvP))[0,1]:+.4f}")
print(f"  corr(momz(dF), momz(dC)) = {np.corrcoef(momz(dvF), momz(dvC))[0,1]:+.4f}")
print(f"  corr(momz(dP), momz(dC)) = {np.corrcoef(momz(dvP), momz(dvC))[0,1]:+.4f}")

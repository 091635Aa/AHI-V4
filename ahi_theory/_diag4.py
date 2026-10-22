
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
t = b.run(3000, seed=21, env_rho=0.85)
F = t["F"]; T=len(F)
y = np.zeros(T); y[:-1]=(F[1:]>F[:-1]).astype(float)
yv = y[:-1]
dF = np.diff(F); dFp = np.concatenate([[0.0],dF])[:T]
ema=np.zeros(T); a=0.02
for i in range(1,T): ema[i]=(1-a)*ema[i-1]+a*dFp[i]
mom = dFp-ema
print("【mom / dFp 的分段可预测性（原始特征）】")
for nm, f in (("dF_prev", dFp[:-1]), ("mom", mom[:-1])):
    print(f"  {nm}: 全体={auc(f,yv):.4f}  train={auc(f[:1500],yv[:1500]):.4f}  hold={auc(f[1500:],yv[1500:]):.4f}"
          f"  std_tr={f[:1500].std():.6f} std_ho={f[1500:].std():.6f} mean_tr={f[:1500].mean():+.6f} mean_ho={f[1500:].mean():+.6f}")

print()
print("【m 对 mom 的编码保真度】")
M = t["m"]
X = np.column_stack([M[:-1], np.ones(T-1)])
for lam in (0.0, 10.0):
    A_ = X.T@X + lam*np.eye(X.shape[1]); w = np.linalg.solve(A_, X.T@mom[:-1]); pr = X@w
    r2 = 1-((mom[:-1]-pr)**2).sum()/((mom[:-1]-mom[:-1].mean())**2).sum()
    r2tr = 1-((mom[:1500]-pr[:1500])**2).sum()/((mom[:1500]-mom[:1500].mean())**2).sum()
    r2ho = 1-((mom[1500:-1]-pr[1500:])**2).sum()/((mom[1500:-1]-mom[1500:-1].mean())**2).sum()
    print(f"  lam={lam:5.1f}: R2(m->mom)={r2:.4f}  分段 R2: tr={r2tr:.4f} ho={r2ho:.4f}"
          f"  slope_tr={np.polyfit(mom[:1500],pr[:1500],1)[0]:.3f} slope_ho={np.polyfit(mom[1500:-1],pr[1500:],1)[0]:.3f}"
          f"  AUC(pr) tr={auc(pr[:1500],yv[:1500]):.4f} ho={auc(pr[1500:],yv[1500:]):.4f}")

print()
print("【mom 与 y 的关系在两段是否一致】")
for s0,s1 in ((0,1500),(1500,2999)):
    mm = mom[s0:s1]; yy = y[s0:s1]
    q = np.quantile(mm, np.linspace(0,1,6))
    idx = np.clip(np.digitize(mm, q[1:-1]),0,4)
    print(f"  段[{s0}:{s1}] P(y=1|mom五分位) = {np.round([yy[idx==b].mean() for b in range(5)],3)}")

print()
print("【F 分段】")
for s0 in range(0,3000,300):
    print(f"  t={s0:4d}: F={F[s0:s0+300].mean():.5f} dF={dF[s0:s0+299].mean():+.7f} mom_std={mom[s0:s0+300].std():.6f} y率={y[s0:s0+300].mean():.3f}")

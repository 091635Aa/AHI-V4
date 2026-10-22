
# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, "/root/AHI")
import numpy as np
from ahi_theory.brain import Brain, K_SCHEMA
def auc(s, y):
    p, n = s[y==1], s[y==0]
    if len(p)==0 or len(n)==0: return float('nan')
    return float((p[:,None]>n[None,:]).mean()+0.5*(p[:,None]==n[None,:]).mean())

print("【无反馈 (w_mom=0) 下，归一化动量本身是否还是好信号】")
b = Brain(seed=0, w_mom=0.0, w_fwd=0.0, w_surp=0.0, w_self=0.0)
t = b.run(3000, seed=21, env_rho=0.85)
F=t["F"]; T=len(F)
y=np.zeros(T); y[:-1]=(F[1:]>F[:-1]).astype(float); yv=y[:-1]
dFp=np.concatenate([[0.0],np.diff(F)])[:T]
a=0.02
e1=np.zeros(T); e2=np.zeros(T)
for i in range(1,T):
    e1[i]=(1-a)*e1[i-1]+a*dFp[i]
    e2[i]=((1-a)*e2[i-1]+a*abs(dFp[i])) if e2[i-1]>0 else abs(dFp[i])
raw = dFp-e1
nrm = raw/(e2+1e-12)
print(f"  raw  mom AUC: tr={auc(raw[:1500],yv[:1500]):.4f} ho={auc(raw[1500:-1],yv[1500:]):.4f}")
print(f"  norm mom AUC: tr={auc(nrm[:1500],yv[:1500]):.4f} ho={auc(nrm[1500:-1],yv[1500:]):.4f}")
print(f"  norm mom: std_tr={nrm[:1500].std():.3f} std_ho={nrm[1500:-1].std():.3f} 范围[{nrm.min():.2f},{nrm.max():.2f}]")
print(f"  dF_abs_ema: tr_mean={e2[:1500].mean():.6f} ho_mean={e2[1500:].mean():.6f}")

print()
print("【反馈强度扫描：w_mom 对 F 与 mom 的影响（w_self=0）】")
for w in (0.0, 0.25, 0.5, 1.0, 2.0):
    b = Brain(seed=0, w_mom=w, w_fwd=0.0, w_surp=0.0, w_self=0.0)
    t = b.run(3000, seed=21, env_rho=0.85)
    F=t["F"]; T=len(F)
    y=np.zeros(T); y[:-1]=(F[1:]>F[:-1]).astype(float); yv=y[:-1]
    dFp=np.concatenate([[0.0],np.diff(F)])[:T]
    e1=np.zeros(T); e2=np.zeros(T)
    for i in range(1,T):
        e1[i]=(1-a)*e1[i-1]+a*dFp[i]
        e2[i]=((1-a)*e2[i-1]+a*abs(dFp[i])) if e2[i-1]>0 else abs(dFp[i])
    raw=dFp-e1; nrm=raw/(e2+1e-12)
    print(f"  w={w:4.2f}: F std={F.std():.5f} | 存于 t['mom'] 的 AUC tr={auc(t['mom'][:1500,0],yv[:1500]):.4f} ho={auc(t['mom'][1500:-1,0],yv[1500:]):.4f}"
          f" | 离线 norm AUC tr={auc(nrm[:1500],yv[:1500]):.4f} ho={auc(nrm[1500:-1],yv[1500:]):.4f}")
    # m 对 mom 的编码（考虑一步滞后）
    M=t["m"]
    for lag in (0,1):
        mm = t["mom"][:,0]
        pr = M[:-1]@b.u_mom
        tgt = mm[lag:lag+len(pr)]
        print(f"        lag={lag}: corr(m·u_mom, mom_{{t-lag}}) = {np.corrcoef(pr, tgt)[0,1]:+.4f}")

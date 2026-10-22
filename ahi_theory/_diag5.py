
# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, "/root/AHI")
import numpy as np
from ahi_theory.brain import Brain, K_SCHEMA
def auc(s, y):
    p, n = s[y==1], s[y==0]
    if len(p)==0 or len(n)==0: return float('nan')
    return float((p[:,None]>n[None,:]).mean()+0.5*(p[:,None]==n[None,:]).mean())

for w_mom in (1.0, 4.0, 16.0, 64.0):
    b = Brain(seed=0, w_mom=w_mom, w_fwd=0.0, w_surp=0.0, w_self=0.0)
    t = b.run(3000, seed=21, env_rho=0.85)
    F=t["F"]; T=len(F)
    y=np.zeros(T); y[:-1]=(F[1:]>F[:-1]).astype(float); yv=y[:-1]
    mom = t["mom"][:-1,0]
    M = t["m"][:-1]
    print(f"=== w_mom={w_mom} ===")
    print(f"  mom: std_tr={mom[:1500].std():.4f} std_ho={mom[1500:].std():.4f} mean_tr={mom[:1500].mean():+.4f} mean_ho={mom[1500:].mean():+.4f}")
    print(f"  mom AUC: tr={auc(mom[:1500],yv[:1500]):.4f} ho={auc(mom[1500:],yv[1500:]):.4f}  sign(mom) AUC: tr={auc(np.sign(mom[:1500]),yv[:1500]):.4f} ho={auc(np.sign(mom[1500:]),yv[1500:]):.4f}")
    print(f"  |m| mean={np.abs(M).mean():.4f}  P(|m|>0.95)={np.mean(np.abs(M)>0.95):.4f}  m 每维 std 均值={M.std(0).mean():.4f}")
    # m 与 u_mom 的投影
    proj = M @ b.u_mom
    print(f"  m·u_mom: std_tr={proj[:1500].std():.4f} std_ho={proj[1500:].std():.4f}  corr(proj,mom)={np.corrcoef(proj,mom)[0,1]:+.4f}")
    for lam in (0.0, 10.0):
        X=np.column_stack([M,np.ones(len(M))])
        A_=X.T@X+lam*np.eye(X.shape[1]); w=np.linalg.solve(A_,X.T@mom); pr=X@w
        r2tr=1-((mom[:1500]-pr[:1500])**2).sum()/((mom[:1500]-mom[:1500].mean())**2).sum()
        r2ho=1-((mom[1500:]-pr[1500:])**2).sum()/((mom[1500:]-mom[1500:].mean())**2).sum()
        # 训练段拟合 -> 留出段
        Xt=np.column_stack([M[:1500],np.ones(1500)])
        wt=np.linalg.solve(Xt.T@Xt+lam*np.eye(Xt.shape[1]), Xt.T@mom[:1500])
        pr_ho=np.column_stack([M[1500:],np.ones(len(M)-1500)])@wt
        r2x=1-((mom[1500:]-pr_ho)**2).sum()/((mom[1500:]-mom[1500:].mean())**2).sum()
        print(f"   lam={lam:4.1f}: R2 in-sample tr={r2tr:.4f} ho={r2ho:.4f} | 训练->留出 R2={r2x:.4f} slope_tr={np.polyfit(mom[:1500],pr[:1500],1)[0]:.3f} slope_ho(拟合)= {np.polyfit(mom[1500:],pr[1500:],1)[0]:.3f}  AUC(训练->留出读出)={auc(pr_ho,yv[1500:]):.4f}")
    print()

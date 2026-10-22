
# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, "/root/AHI")
import numpy as np
from ahi_theory.brain import Brain, K_SCHEMA

def auc(s, y):
    p, n = s[y == 1], s[y == 0]
    return float((p[:, None] > n[None, :]).mean() + 0.5*(p[:, None] == n[None, :]).mean())

b = Brain(seed=0, w_mom=9e4, w_fwd=1.4, w_surp=1.0, w_self=6e3)
t = b.run(3000, seed=21, env_rho=0.85)
F = t["F"]; T = len(F)
y = np.zeros(T); y[:-1] = (F[1:] > F[:-1]).astype(float); yv = y[:-1]
dFp = np.concatenate([[0.0], np.diff(F)])[:T]

print("【m 的饱和与编码能力】")
M = t["m"]
print(f"  |m| mean={np.abs(M).mean():.4f}  P(|m|>0.95)={np.mean(np.abs(M)>0.95):.4f}")
print(f"  mom 对 y 的 AUC = {auc(t['mom'][:-1,0], yv):.4f}")
print(f"  dF_prev 对 y 的 AUC = {auc(dFp[:-1], yv):.4f}")

# 线性读出 m->mom 的 R2
X = np.column_stack([M[:-1], np.ones(T-1)])
w = np.linalg.lstsq(X, t["mom"][:-1,0], rcond=None)[0]
pred = X @ w
r2 = 1-((t['mom'][:-1,0]-pred)**2).sum()/((t['mom'][:-1,0]-t['mom'][:-1,0].mean())**2).sum()
print(f"  线性读出 m -> mom 的 R2 = {r2:.4f}；该读出对 y 的 AUC = {auc(pred, yv):.4f}")
w2 = np.linalg.lstsq(X, dFp[:-1], rcond=None)[0]; pred2 = X @ w2
r22 = 1-((dFp[:-1]-pred2)**2).sum()/((dFp[:-1]-dFp[:-1].mean())**2).sum()
print(f"  线性读出 m -> dF_prev 的 R2 = {r22:.4f}；该读出对 y 的 AUC = {auc(pred2, yv):.4f}")

print()
print("【m 的基座各项对 m 的贡献（幅度）】")
z = t["h"].reshape(T,-1)
terms = {
  "Wm_in@z": b.Wm_in @ z.T,
  "w_slow*h_slow": b.w_slow * t["h_slow"].T,
  "w_fwd*Wm_fwd@zpred": b.w_fwd * (b.Wm_fwd @ t["z_pred"].reshape(T,-1).T),
}
for k,v in terms.items():
    print(f"  {k:<22} mean|.|={np.abs(v).mean():.4f}  std={v.std():.4f}")
print(f"  {'e_rel/pi_rel 项':<22} 值范围 [{b.w_surp*(b.Wm_surp@np.array([0,1])).min():.3f}, {b.w_surp*(b.Wm_surp@np.array([3,1])).max():.3f}]")
print(f"  {'w_mom*mom*u_mom':<22} std={np.std(b.w_mom*t['mom'][:,0,None]*b.u_mom[None,:], axis=0).mean():.4f}")
print(f"  mom 值域 [{t['mom'][:,0].min():.5f}, {t['mom'][:,0].max():.5f}]  std={t['mom'][:,0].std():.6f}")

print()
print("【对 y 的单特征 AUC（线性读出，训练0-1500/留出1500-2998）】")
def lin(X, yv, lam=0.0):
    mu, sd = X[:1500].mean(0), X[:1500].std(0)+1e-12
    Xs = (X-mu)/sd
    A = Xs[:1500].T@Xs[:1500] + lam*np.eye(Xs.shape[1])
    w = np.linalg.solve(A, Xs[:1500].T@yv[:1500])
    return auc(Xs[:1500]@w, yv[:1500]), auc(Xs[1500:]@w, yv[1500:])
for nm, X in (("m(16)", M[:-1]), ("m+mom", np.column_stack([M[:-1], t["mom"][:-1,0]])),
              ("mom(1)", t["mom"][:-1,0:1]), ("dFp(1)", dFp[:-1,None])):
    a1, a2 = lin(X, yv, 0.0)
    print(f"  {nm:<10} tr={a1:.4f} ho={a2:.4f}")

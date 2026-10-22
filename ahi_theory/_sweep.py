
# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, "/root/AHI")
import numpy as np
from ahi_theory.brain import Brain, K_SCHEMA
def auc(s, y):
    p, n = s[y==1], s[y==0]
    if len(p)==0 or len(n)==0: return float('nan')
    return float((p[:,None]>n[None,:]).mean()+0.5*(p[:,None]==n[None,:]).mean())

def P1(t):
    _cut = 1500
    F_all = t["F"]; n = len(F_all)
    y_all = np.zeros(n); y_all[:-1] = (F_all[1:] > F_all[:-1]).astype(float)
    Str = t["m"][:_cut]; ytr = y_all[:_cut]
    Xtr = np.column_stack([Str, np.ones(_cut)]); mu_, sd_ = Xtr.mean(0), Xtr.std(0)+1e-9
    Xs = (Xtr-mu_)/sd_; w0 = np.linalg.lstsq(Xs, ytr, rcond=None)[0]; s_tr = Xs@w0
    qs = np.unique(np.quantile(s_tr, np.linspace(0,1,11)))
    if len(qs)<3: qs = np.linspace(s_tr.min(), s_tr.max()+1e-9, 11)
    it = np.clip(np.digitize(s_tr, qs[1:-1]), 0, len(qs)-2)
    tb = np.array([ytr[it==b].mean() if (it==b).sum()>0 else float(ytr.mean()) for b in range(len(qs)-1)])
    wts = np.array([max((it==b).sum(),1) for b in range(len(qs)-1)], float); vals=tb.copy(); ch=True
    while ch:
        ch=False
        for b in range(len(vals)-1):
            if vals[b]>vals[b+1]:
                tot=wts[b]+wts[b+1]; vals[b]=vals[b+1]=(vals[b]*wts[b]+vals[b+1]*wts[b+1])/tot
                wts[b]=wts[b+1]=tot; ch=True
    tb = vals
    S = t["m"][_cut:]; y = y_all[_cut:]
    sc = ((np.column_stack([S, np.ones(len(S))])-mu_)/sd_)@w0
    ib = np.clip(np.digitize(sc, qs[1:-1]), 0, len(tb)-1); conf = tb[ib]
    edges = np.unique(np.quantile(conf, np.linspace(0,1,11)))
    if len(edges)<3: edges = np.linspace(conf.min(), conf.max()+1e-9, 11)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, len(edges)-2)
    ece = sum((idx==b).mean()*abs(conf[idx==b].mean()-y[idx==b].mean()) for b in range(len(edges)-1) if (idx==b).sum()>0)
    p_, n_ = conf[y==1], conf[y==0]
    ac = float((p_[:,None] > n_[None,:]).mean()) if len(p_) and len(n_) else float('nan')
    return float(ece), ac, len(edges)-1, auc(s_tr,ytr), auc(sc,y), float(y.mean())

print("动量自归一化后，扫 w_mom（w_fwd=w_surp=w_self=0）")
print(f"  {'w_mom':>8} | {'ECE':>8} {'AUC(conf)':>10} {'箱':>3} {'raw_tr':>7} {'raw_ho':>7} {'基率':>6} | PASS")
for w_mom in (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0):
    b = Brain(seed=0, w_mom=w_mom, w_fwd=0.0, w_surp=0.0, w_self=0.0)
    t = b.run(3000, seed=21, env_rho=0.85)
    ece, ac, nb, a1, a2, br = P1(t)
    print(f"  {w_mom:8.1f} | {ece:8.4f} {ac:10.4f} {nb:3d} {a1:7.4f} {a2:7.4f} {br:6.3f} | {'✅' if (ece<0.05 and ac>0.6) else '❌'}")

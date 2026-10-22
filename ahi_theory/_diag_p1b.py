
# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, "/root/AHI")
import numpy as np
from ahi_theory.brain import Brain, R_REGIONS, D_REGION, K_SCHEMA

seed = 0
b = Brain(seed=seed)
tr = b.run(3000, seed=seed + 21, env_rho=0.85)
F = tr["F"]; m = tr["m"]
cx = 0.5 * (m ** 2).sum(1) / K_SCHEMA
pe = F - cx
y = np.zeros(len(F)); y[:-1] = (F[1:] > F[:-1]).astype(float)

print("【F 的时间结构】每 200 步的均值")
for s in range(0, 3000, 200):
    seg = slice(s, s + 200)
    print(f"  t={s:4d}-{s+200:4d}  F={F[seg].mean():.5f}  cx={cx[seg].mean():.5f}  pe={pe[seg].mean():.5f}  y涨率={y[seg].mean():.3f}  dF={np.diff(F[seg]).mean():+.6f}")

print()
print("【h_slow 幅值 & L 的时间结构】")
print("  |h_slow| 前100:", np.abs(tr['h_slow'][:100]).mean().round(4), " 后100:", np.abs(tr['h_slow'][-100:]).mean().round(4))
print("  L 前100:", tr['L'][:100].mean(0).round(4), " 后100:", tr['L'][-100:].mean(0).round(4))

print()
print("【分段基率】")
for s in range(0, 3000-500, 500):
    print(f"  t={s}-{s+500}: y基率={y[s:s+500].mean():.3f}  dF mean={np.diff(F[s:s+500]).mean():+.6f}")

print()
print("【dF 自相关在不同段】")
dF = np.diff(F)
for s in (0, 750, 1500, 2250):
    seg = dF[s:s+700]
    print(f"  t={s}-{s+700}: ac1={np.corrcoef(seg[:-1], seg[1:])[0,1]:+.4f}  std={seg.std():.6f}")

print()
print("【若用 dF_t 预测 y_t=1[dF_t>0]】")
print("  完美信息上限 = 1.0 (trivially)")
print("  用 dF_{t-1} 预测 y_t 的 AUC（延迟一步可达性）:")
p, n = dF[:-1][y[:-1]==1], dF[:-1][y[:-1]==0]
print("   AUC(dF_{t-1} -> y_t) =", float((p[:,None] > n[None,:]).mean()))
for s in (0, 1500):
    dd = dF[s:s+1490]; yy = y[s+1:s+1491]
    pp, nn = dd[yy==1], dd[yy==0]
    print(f"   段 {s}: AUC={float((pp[:,None]>nn[None,:]).mean()):.4f}")

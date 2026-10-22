# -*- coding: utf-8 -*-
"""H 项汇总：两趟逐位同、Δ/g/R_drop/R_DiD、bootstrap CI、TOST、轴C 条件性。只读 H_pass*.json。"""
import os, sys, io, json, itertools, statistics
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
p1 = json.load(open(os.path.join(BASE, "H_轴A_C_pass1.json"), encoding="utf-8"))
p2 = json.load(open(os.path.join(BASE, "H_轴A_C_pass2.json"), encoding="utf-8"))
rng = np.random.default_rng(20261018)

def key(r):
    return (r["种子偏"], r["轴"], r["条件"])

d1 = {key(r): r for r in p1["rows"]}
d2 = {key(r): r for r in p2["rows"]}
keys = sorted(set(d1) & set(d2))

# 1) 两趟逐位同（指纹）
fp_same = [d1[k]["指纹"] == d2[k]["指纹"] for k in keys]
val_same = [abs(d1[k]["一评"] - d2[k]["一评"]) < 1e-12 and abs(d1[k]["二评"] - d2[k]["二评"]) < 1e-12 for k in keys]
# 2) 不拆臂跨轴重复（A vs C 同种子同条件）应逐位同
dup = {}
for k in keys:
    off, ac, cond = k
    if cond == "不拆":
        dup.setdefault(off, []).append(d1[k]["指纹"])
dup_same = all(len(set(v)) == 1 for v in dup.values())

def get(off, ac, cond, field, use=1):
    r = (d1 if use == 1 else d2).get((off, ac, cond))
    return None if r is None else r[field]

seeds = sorted({r["种子偏"] for r in p1["rows"]})
分析 = {}
all_rows = []
for off in seeds:
    基 = get(off, "A", "不拆", "一评")   # 不拆臂在 A/C 两轴重复且应逐位同；先验一个
    g = get(off, "A", "不拆", "二评") - 基
    for ac in ("A", "C"):
        基ac = get(off, ac, "不拆", "一评")
        降 = get(off, ac, "拆", "一评")
        回 = get(off, ac, "拆", "二评")
        if 降 is None:
            continue
        Δ = 基ac - 降; G = 回 - 降
        if abs(基ac - 基) > 1e-12:
            raise SystemExit("不拆臂跨轴不一致：off=%s axis=%s %.12f vs %.12f" % (off, ac, 基ac, 基))
        all_rows.append(dict(off=off, axis=ac, 基=基ac, 降=降, 回=回, g=g, Δ=Δ, G=G,
                             R_drop=(G / Δ if abs(Δ) > 1e-9 else None),
                             R_base=(回 / 基ac if abs(基ac) > 1e-9 else None)))
    分析["g_per_seed"] = {str(off): g for off in seeds}

g_mean = float(np.mean([get(off, "A", "不拆", "二评") - get(off, "A", "不拆", "一评") for off in seeds]))
for ac in ("A", "C"):
    rows = [r for r in all_rows if r["axis"] == ac]
    Δs = np.array([r["Δ"] for r in rows], dtype=float)
    Gs = np.array([r["G"] for r in rows], dtype=float)
    bases = np.array([r["基"] for r in rows], dtype=float)
    n = len(Δs)
    # bootstrap mean CI
    def boot_mean_ci(x, n_boot=10000, lo=2.5, hi=97.5):
        idx = rng.integers(0, len(x), size=(n_boot, len(x)))
        means = x[idx].mean(axis=1)
        return float(np.percentile(means, lo)), float(np.percentile(means, hi))
    mean_Δ = float(Δs.mean()); ciΔ = boot_mean_ci(Δs)
    # R_drop / R_DiD 仅在"先降 Δ≥+0.05"的 seed 上有定义（蓝队口径；负 Δ = 提升，不适用）
    valid = Δs >= 0.05
    R_drop = (Gs[valid] / Δs[valid]) if valid.any() else np.array([])
    R_diD = ((Gs[valid] - g_mean) / Δs[valid]) if valid.any() else np.array([])
    def stats(R):
        if len(R) == 0:
            return dict(n=0)
        lo, hi = boot_mean_ci(R)
        lo90, hi90 = boot_mean_ci(R, lo=5, hi=95)
        return dict(n=int(len(R)), mean=float(R.mean()), median=float(np.median(R)),
                    ci95=[lo, hi], ci90=[lo90, hi90],
                    pass_ge_080=bool(R.mean() >= 0.80 and lo >= 0.80),
                    weld_TOST_upper_lt_090=bool(hi90 < 0.90))
    # Spearman(基, Δ) + 精确置换 p
    def rank(x):
        order = np.argsort(np.argsort(x))
        return order.astype(float)
    rb, rd = rank(bases), rank(Δs)
    rho = float(np.corrcoef(rb, rd)[0, 1])
    cnt = 0; tot = 0
    for perm in itertools.permutations(range(n)):
        if n <= 9:
            rp = np.array(perm, dtype=float)
            r = float(np.corrcoef(rp, rd)[0, 1])
            if abs(r) >= abs(rho) - 1e-12:
                cnt += 1
            tot += 1
    p_exact = cnt / max(1, tot)
    hi_mask = bases >= np.median(bases)
    分析[ac] = dict(n=n, mean_Δ=mean_Δ, ci95_Δ=ciΔ, seeds_Δ_ge_005=int(valid.sum()),
                    seeds_Δ_pos_ge_005=int((Δs >= 0.05).sum()),
                    R_drop_R_DiD_applicable=bool((Δs >= 0.05).sum() >= 3),
                    R_drop=stats(R_drop), R_DiD=stats(R_diD),
                    spearman_rho=rho, permutation_p_exact=p_exact,
                    high_group_mean_Δ=float(Δs[hi_mask].mean()), low_group_mean_Δ=float(Δs[~hi_mask].mean()),
                    per_seed=[{k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()} for r in rows])
分析["g_mean"] = g_mean
分析["two_pass"] = dict(n_runs=len(keys), fingerprint_all_same=bool(all(fp_same)),
                        values_all_same=bool(all(val_same)), no_ablate_dup_same=bool(dup_same),
                        pass1_wall_s=p1["total_wall_s"], pass2_wall_s=p2["total_wall_s"],
                        pass1_mem_end=p1["mem_end"], pass2_mem_end=p2["mem_end"])
分析["protocol"] = dict(每跑拍数=p1["每跑拍数"], 续训拍=p1["续训拍"], seeds=seeds,
                        note="路径②：每臂独立重训、不复用快照；拆件在训练前激活")

json.dump(分析, open(os.path.join(BASE, "H_轴A_C_汇总.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps({k: v for k, v in 分析.items() if k in ("A", "C", "g_mean", "two_pass")},
                 ensure_ascii=False, indent=1, default=float))

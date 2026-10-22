# -*- coding: utf-8 -*-
"""H 项：轴A/轴C 9 seeds 配对重训（路径②：每臂独立重训、不复用快照）。只读既有件，不改。
每跑 1,570 拍 = 预热250+训450+评250+续训370+评250；D=64；两趟由外部调用 --pass 1/2。
用法: python H_axisAC.py <pass 1|2> <out.json> [--smoke]
"""
import os
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[k] = "1"
os.environ["PYTHONHASHSEED"] = "0"
import sys, io, json, time, gc, importlib.util, itertools, statistics
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np
import psutil

BASE = r"<workspace>/03_AHI\AHI4\meeting51\实施"
spec = importlib.util.spec_from_file_location("J判据", os.path.join(BASE, "判据_千层骨架.py"))
J = importlib.util.module_from_spec(spec)
spec.loader.exec_module(J)

pass_no = int(sys.argv[1])
outpath = sys.argv[2]
smoke = "--smoke" in sys.argv
SEEDS = [0, 11, 21, 31, 41, 51, 61, 71, 81][: (1 if smoke else 9)]
续训 = 370
AXES = [("轴A分区", "A"), ("轴C区室", "C")]
if smoke:
    AXES = [("轴A分区", "A")]
CONDS = ["不拆", "拆"]

# 预注册执行顺序（固定随机表，防顺序效应）
order = [(off, ax, ac, cond) for off in SEEDS for ax, ac in AXES for cond in CONDS]
rng_order = np.random.default_rng(20261017 + pass_no)
order = [order[i] for i in rng_order.permutation(len(order))]

p = psutil.Process()
def mem():
    mi = p.memory_info()
    return {"rss_MB": round(mi.rss/1048576, 2), "commit_MB": round(mi.private/1048576, 2)}

t_all = time.perf_counter()
rows = []
for (off, 轴名, 轴码, cond) in order:
    t0 = time.perf_counter()
    核 = J.造核(种子偏=off)                       # 每臂独立重训（路径②）
    if cond == "拆":
        核.拆件(轴名)                              # 拆件在训练前激活
    序主 = J.造序(J.种子环境主)
    序测 = J.造序(J.种子环境主 + 5, 全长=J.测 + 1)
    J.训练(核, 序主)                               # 预热250 + 训450
    e1 = J.评(核, 序测)["总"]                      # 基 / 降
    J.训练(核, 序主, 预热拍=0, 训拍=续训)          # 续训 370
    e2 = J.评(核, 序测)["总"]                      # g / 回
    fp = 核.指纹()
    wall = time.perf_counter() - t0
    rows.append(dict(种子偏=off, 种子=int(J.种子模型)+off, 轴=轴码, 轴名=轴名, 条件=cond,
                     一评=round(float(e1), 6), 二评=round(float(e2), 6),
                     指纹=fp, wall_s=round(wall, 2)))
    del 核
    gc.collect()
    print("run off=%d axis=%s cond=%s e1=%.4f e2=%.4f fp=%s wall=%.1fs" % (off, 轴码, cond, e1, e2, fp[:12], wall), flush=True)

total_s = time.perf_counter() - t_all
res = {"pass": pass_no, "smoke": smoke, "续训拍": 续训, "每跑拍数": 250 + 450 + 250 + 续训 + 250,
       "seeds": SEEDS, "n_runs": len(rows), "rows": rows,
       "total_wall_s": round(total_s, 1), "mem_end": mem(),
       "thread_env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")}}

# ── 指标（每 pass 内在数据上算；CI/置换在最终汇总脚本做）──
def get(off, ac, cond, field):
    for r in rows:
        if r["种子偏"] == off and r["轴"] == ac and r["条件"] == cond:
            return r[field]
    return None

per_seed = {}
for off in SEEDS:
    基 = get(off, "A", "不拆", "一评")   # 与 C 的不拆臂应逐位同
    g = get(off, "A", "不拆", "二评") - 基
    for ac in sorted({r["轴"] for r in rows}):
        降 = get(off, ac, "拆", "一评")
        回 = get(off, ac, "拆", "二评")
        Δ = 基 - 降
        G = 回 - 降
        per_seed.setdefault(ac, []).append(dict(off=off, 基=round(基,6), 降=round(降,6), 回=round(回,6),
                                                g=round(g,6), Δ=round(Δ,6), G=round(G,6),
                                                R_drop=round(G/Δ,4) if abs(Δ) > 1e-9 else None,
                                                R_base=round(回/基,4) if abs(基) > 1e-9 else None))
res["per_seed"] = per_seed
json.dump(res, open(outpath, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("WROTE", outpath, "runs", len(rows), "wall_s", round(total_s,1))

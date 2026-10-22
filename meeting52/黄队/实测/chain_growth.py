# -*- coding: utf-8 -*-
"""全链 D=1024 2000 拍增长曲线（GC 开启 vs 关闭），只读。"""
import os
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[k] = "1"
import sys, io, json, time, gc
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import psutil, numpy as np
sys.path.insert(0, r"<workspace>/03_AHI\AHI4\meeting49\实施")
import 贯通_多核心 as G

mode = sys.argv[1]  # "gc_on" | "gc_off"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
outpath = sys.argv[3]
p = psutil.Process()
def mem():
    mi = p.memory_info()
    return {"rss_MB": round(mi.rss/1048576, 2), "commit_MB": round(mi.private/1048576, 2),
            "peak_wset_MB": round(mi.peak_wset/1048576, 2), "peak_commit_MB": round(mi.peak_pagefile/1048576, 2)}
res = {"mode": mode, "N": N, "baseline": mem()}
t0 = time.perf_counter(); 双 = G.造双区室(seed=310); res["build_s"] = round(time.perf_counter()-t0, 2)
G.一拍(双)
res["after_warmup"] = mem()
if mode == "gc_off":
    gc.collect(); gc.disable()
samples = []
t_run = time.perf_counter()
for i in range(1, N+1):
    G.一拍(双)
    if i % 100 == 0:
        samples.append(dict(t=i, **mem()))
res["run_s"] = round(time.perf_counter()-t_run, 2)
res["samples"] = samples
res["end"] = mem()
res["gc_enabled"] = gc.isenabled()
json.dump(res, open(outpath, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(mode, "N", N, "baseline_commit", res["baseline"]["commit_MB"], "end", res["end"], "build", res["build_s"], "run", res["run_s"])
print("samples_commit:", [s["commit_MB"] for s in samples])

# -*- coding: utf-8 -*-
"""黄队 R3 测量探针：五元组 × 两 horizon × threads=1。只读运行，不写既有件。
用法: python probe.py <config> <horizon> <out.json>
config: qb64 | qb256 | chain1024 | torch_only | torch_chain1024
"""
import os
# ---- 必须在 import numpy 之前锁线程 ----
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
          "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[k] = "1"
os.environ["PYTHONHASHSEED"] = "0"

import sys, io, json, time, gc, subprocess, statistics, threading, platform
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import psutil
import numpy as np

cfg = sys.argv[1]
horizon = int(sys.argv[2])
outpath = sys.argv[3]
p = psutil.Process()

def mem():
    mi = p.memory_info()
    return {
        "rss_MB": round(mi.rss / 1048576, 2),
        "private_commit_MB": round(mi.private / 1048576, 2),
        "peak_wset_MB": round(mi.peak_wset / 1048576, 2),
        "peak_commit_MB": round(mi.peak_pagefile / 1048576, 2),
        "pagefile_MB": round(mi.pagefile / 1048576, 2),
    }

def smi():
    try:
        o = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.free",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout.strip()
        used, free = [int(x) for x in o.split(",")]
        return {"nvidia_used_MiB": used, "nvidia_free_MiB": free}
    except Exception as e:
        return {"nvidia_error": str(e)}

def cpu_state():
    return {"cpu_percent_0.3s": psutil.cpu_percent(interval=0.3),
            "python_proc_count": sum(1 for q in psutil.process_iter(["name"])
                                     if (q.info.get("name") or "").lower().startswith("python"))}

res = {"config": cfg, "horizon": horizon, "python": sys.version.split()[0],
       "thread_env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
       "cpu_count": psutil.cpu_count(), "arch": platform.machine()}

res["concurrency_start"] = cpu_state()
res["device_start"] = smi()
res["baseline_before"] = mem()   # 进程刚起、未 import numpy
t_import0 = time.perf_counter()

# ---- 构建 ----
if cfg.startswith("qb"):
    sys.path.insert(0, r"<workspace>/03_AHI\AHI4\meeting51\实施")
    import 算子_千层骨 as q
    D = 64 if cfg == "qb64" else 256
    t0 = time.perf_counter()
    kern = q.千层核(D=D, 种子=7)
    build_s = time.perf_counter() - t0
    res["build_s"] = round(build_s, 3)
    def step():
        kern.前向一拍(None)
    warm = 3
elif cfg in ("chain1024", "torch_chain1024"):
    if cfg == "torch_chain1024":
        t0 = time.perf_counter(); import torch; torch.set_num_threads(1)
        res["torch_import_s"] = round(time.perf_counter() - t0, 3)
        res["torch_version"] = torch.__version__
    sys.path.insert(0, r"<workspace>/03_AHI\AHI4\meeting49\实施")
    import 贯通_多核心 as G
    t0 = time.perf_counter()
    双 = G.造双区室(seed=310)
    build_s = time.perf_counter() - t0
    res["build_s"] = round(build_s, 3)
    res["engine_D"] = int(双["D"])
    def step():
        G.一拍(双)
    warm = 1
elif cfg == "torch_only":
    t0 = time.perf_counter(); import torch; torch.set_num_threads(1)
    res["torch_import_s"] = round(time.perf_counter() - t0, 3)
    res["torch_version"] = torch.__version__
    build_s = 0.0
    res["build_s"] = 0.0
    def step():
        return None
    warm = 0
else:
    raise SystemExit("unknown config " + cfg)

res["import_plus_build_s"] = round(time.perf_counter() - t_import0, 3)
res["after_build"] = mem()

# 预热
for _ in range(warm):
    step()
res["after_warmup"] = mem()

# ---- 采样线程（5ms）----
stop = [False]
maxes = {"rss_MB": 0.0, "private_commit_MB": 0.0}
def sampler():
    while not stop[0]:
        mi = p.memory_info()
        if mi.rss > maxes["rss_MB"] * 1048576:
            maxes["rss_MB"] = mi.rss / 1048576
        if mi.private > maxes["private_commit_MB"] * 1048576:
            maxes["private_commit_MB"] = mi.private / 1048576
        time.sleep(0.005)

gc.collect(); gc.disable()
th = threading.Thread(target=sampler, daemon=True); th.start()
tick_times = []
t_run0 = time.perf_counter()
for i in range(horizon):
    t = time.perf_counter()
    step()
    tick_times.append((time.perf_counter() - t) * 1000)
run_s = time.perf_counter() - t_run0
stop[0] = True; th.join(); gc.enable()

tick_times.sort()
def pct(q):
    return tick_times[min(len(tick_times) - 1, int(q * len(tick_times)))]
res["run_s"] = round(run_s, 3)
res["ticks_per_s"] = round(horizon / run_s, 2) if run_s > 0 else None
res["tick_ms"] = {"median": round(statistics.median(tick_times), 2),
                  "min": round(tick_times[0], 2),
                  "p90": round(pct(0.90), 2),
                  "max": round(tick_times[-1], 2)} if tick_times else None

# 采样峰值与结束状态
res["sampled_peak"] = {k: round(v, 2) for k, v in maxes.items()}
res["end"] = mem()
res["device_end"] = smi()
res["device_delta_used_MiB"] = (res["device_end"].get("nvidia_used_MiB", 0)
                                - res["device_start"].get("nvidia_used_MiB", 0))
res["concurrency_end"] = cpu_state()

if cfg in ("torch_chain1024", "torch_only"):
    try:
        import torch
        res["allocator"] = {
            "max_memory_allocated_MB": round(torch.cuda.max_memory_allocated() / 1048576, 2) if torch.cuda.is_available() else None,
            "max_memory_reserved_MB": round(torch.cuda.max_memory_reserved() / 1048576, 2) if torch.cuda.is_available() else None,
            "cuda_available": bool(torch.cuda.is_available()),
            "is_initialized": bool(torch.cuda.is_initialized()),
        }
    except Exception as e:
        res["allocator"] = {"error": str(e)}
else:
    res["allocator"] = {"note": "numpy-only; torch not imported", "max_memory_allocated_MB": 0.0}

# 五元组汇总（判定用）
res["five_tuple"] = {
    "peak_wset_MB": res["end"]["peak_wset_MB"],
    "peak_commit_MB": res["end"]["peak_commit_MB"],
    "peak_rss_sampled_MB": res["sampled_peak"]["rss_MB"],
    "device_delta_MiB": res["device_delta_used_MiB"],
    "allocator_max_alloc_MB": res["allocator"].get("max_memory_allocated_MB", 0.0),
}

with open(outpath, "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=1)
print(json.dumps(res["five_tuple"], ensure_ascii=False),
      "| build", res["build_s"], "s | run", res["run_s"], "s | tick", res["tick_ms"])

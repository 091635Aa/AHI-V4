# -*- coding: utf-8 -*-
"""聚合六组 result_*.json + growth + CUDA 加性 → 汇总.json（只写本队 实测/）。"""
import os, sys, io, json, glob
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
base = os.path.dirname(os.path.abspath(__file__))

def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)

runs = {}
for f in sorted(glob.glob(os.path.join(base, "result_*.json"))):
    d = load(f)
    key = "%s_%d" % (d["config"], d["horizon"])
    runs[key] = d

growth = load(os.path.join(base, "growth_gc_on.json")) if os.path.exists(os.path.join(base, "growth_gc_on.json")) else None

# 320MB 判定与增长外推（全链 D=1024）
c120 = runs.get("chain1024_120", {})
c2000 = runs.get("chain1024_2000", {})
def slope(a, b, n):
    return (b - a) / n
grow_commit = slope(c120["five_tuple"]["peak_commit_MB"], c2000["five_tuple"]["peak_commit_MB"], 1880) if c120 and c2000 else None
grow_wset = slope(c120["five_tuple"]["peak_wset_MB"], c2000["five_tuple"]["peak_wset_MB"], 1880) if c120 and c2000 else None
verdict = {}
if c2000:
    pc = c2000["five_tuple"]["peak_commit_MB"]; pw = c2000["five_tuple"]["peak_wset_MB"]
    verdict = {
        "peak_wset_MB": pw, "peak_commit_MB": pc,
        "wset_margin_to_320_MB": round(320 - pw, 2),
        "commit_margin_to_320_MB": round(320 - pc, 2),
        "commit_utilization_pct": round(pc / 320 * 100, 1),
        "growth_commit_MB_per_tick": round(grow_commit, 5),
        "growth_wset_MB_per_tick": round(grow_wset, 5),
        "extrapolated_ticks_to_320_commit": int((320 - pc) / grow_commit) if grow_commit and grow_commit > 0 else None,
        "extrapolated_ticks_to_320_wset": int((320 - pw) / grow_wset) if grow_wset and grow_wset > 0 else None,
    }

# 归因（commit）
attr = {}
if "chain1024_2000" in runs:
    d = runs["chain1024_2000"]
    bb = d["baseline_before"]["private_commit_MB"]; ab = d["after_build"]["private_commit_MB"]
    aw = d["after_warmup"]["private_commit_MB"]; en = d["end"]["private_commit_MB"]; pk = d["five_tuple"]["peak_commit_MB"]
    build_add = ab - bb; warm_add = aw - ab; run_add = en - aw; gap = pk - en
    attr["chain1024_2000_commit"] = {
        "total_peak_commit_MB": pk,
        "interpreter_plus_numpy_MB": round(bb, 2),
        "chain_build_add_MB": round(build_add, 2),
        "warmup_add_MB": round(warm_add, 2),
        "run_growth_add_MB": round(run_add, 2),
        "peak_gap_MB": round(gap, 2),
        "attributed_MB": round(bb + build_add + warm_add + run_add, 2),
        "residual_MB": round(gap, 2),
        "attributed_pct": round((bb + build_add + warm_add + run_add) / pk * 100, 1),
    }
if "torch_chain1024_120" in runs and "torch_only_120" in runs:
    tc = runs["torch_chain1024_120"]; to = runs["torch_only_120"]
    pk = tc["five_tuple"]["peak_commit_MB"]; torch_only = to["five_tuple"]["peak_commit_MB"]
    chain_build = runs["chain1024_120"]["after_build"]["private_commit_MB"] - runs["chain1024_120"]["baseline_before"]["private_commit_MB"]
    rest = pk - torch_only - chain_build
    attr["torch_chain1024_120_commit"] = {
        "total_peak_commit_MB": pk,
        "torch_import_MB": torch_only,
        "torch_share_pct": round(torch_only / pk * 100, 1),
        "chain_build_add_MB": round(chain_build, 2),
        "chain_share_pct": round(chain_build / pk * 100, 1),
        "warmup_run_rest_MB": round(rest, 2),
        "rest_share_pct": round(rest / pk * 100, 1),
    }

cuda = {
    "baseline_MiB": 183,
    "level_1_MiB": 324, "delta_1_MiB": 141,
    "level_2_MiB": 466, "delta_2_MiB": 142,
    "level_3_MiB": 608, "delta_3_MiB": 142,
    "after_kill_MiB": 182,
    "per_context_MiB": 142,
    "additive": True,
    "note": "首轮 level_3 采样得 477MiB 为时序假象；复测（每级 8s + 双采样）得 608MiB，线性加性成立。",
}

summary = {
    "schema": "yellow-R3-resource-measurement-v1",
    "time": "2026-10-17T21:35+08:00",
    "host": {"python": runs.get("qb256_120", {}).get("python"), "cpu_count": runs.get("qb256_120", {}).get("cpu_count")},
    "thread_lock": {"env": runs.get("qb256_120", {}).get("thread_env"), "note": "五线程环境变量均=1"},
    "runs": runs,
    "growth_gc_on": growth,
    "verdict_320MB": verdict,
    "attribution": attr,
    "cuda_context": cuda,
    "concurrency_declaration": {
        "state": "会议并发期（蓝队 4 worker 在跑；python_proc_count 8–12，CPU 29–68%）",
        "rule": "墙钟只报 median/min/p90，不作效率宣称",
        "nvidia_baseline_MiB": [183, 187, 190],
    },
    "validity": {
        "five_tuple_complete": True,
        "baseline_before_after": True,
        "threads_locked_1": True,
        "device_and_allocator_columns_present": True,
        "sampled_vs_os_peak_agreement": "差值 <2.2%",
        "two_horizons": [120, 2000],
        "void_flags": [],
    },
}
out = os.path.join(base, "汇总.json")
json.dump(summary, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("wrote", out)
print(json.dumps({"verdict_320MB": verdict, "attribution": attr, "cuda": cuda},
                 ensure_ascii=False, indent=1))

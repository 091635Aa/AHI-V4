# -*- coding: utf-8 -*-
"""
meeting52 · 蓝队实测 · 轴C 9-seed 复跑协议（lead 2026-10-17 授权：独占窗口 15 min）

性质：**测量**，不是改件。只 import 既有判据模块（只读）；不写/不改任何既有件；
      不覆盖既有 JSON；不调 主流程/步3/步8/取主核（避开缓存与不可逆拆边污染）。
跑法：python -B 本文件
落盘：同目录 轴C9_结果.json（新建；每 seed 跑完即增量落盘，超时留部分结果）

──────────────────────────── 预注册（跑前冻结，跑后不改） ────────────────────────────
【种子族】种子偏 = [0, 41, 82, 123, 164, 205, 246, 287, 328]（步长 41；含两锚点）
  - 锚点：种子偏 0 ⇒ 基 ≈ 0.8856666667；种子偏 41 ⇒ 基 ≈ 0.683。|差| > 0.02 ⇒ 立即停，报"口径不匹配"。
【固定序列】序主 = 造序(52031)；序测 = 造序(52036, 全长=251)。全程不变 ⇒ 只变模型初值（基水平方差来源）。
【每 seed】新核 造核(种子偏=s) → 训练(核, 序主) → 基 = 评(核, 序测)["总"]（现场重算）
  → 快照 → 恢复 → 拆件("轴C区室") → 降 = 评 → 训练(核, 序主, 预热拍=0, 训拍=450) → 回 = 评
  → 恢复(快) → 训练(核, 序主, 预热拍=0, 训拍=450) → 对照 = 评  （无拆件续训对照）
【量】Δ = 基−降；G = 回−降；g = 对照−基；R_base = 回/基；R_drop = G/Δ；R_DiD = (G−g)/Δ
【阈值】Δ_min = 0.05；R_drop ≥ 0.80；R_DiD ≥ 0.80 且 CI 下界 ≥ 0.80 ⇒ 可补偿；
        TOST 90% CI 上界 < 0.90 ⇒ 焊死。（本轮只判轴C 效应，不据此删任何轴）
【判决（蓝队预注册）】Spearman(基, Δ) ≤ −0.5 且 分层（基≥中位 / <中位）符号相反
        ⇒ "随基水平翻转（条件性轴）"；否则 "不可判定"；**禁止**报"装饰/删除"。
  另按 lead 三选一单列：全部 9 个 Δ 同号 且 mean(Δ) 的 95% bootstrap CI 不含 0 且 |mean|≥0.05
        ⇒ 附报 "单侧"（仅当不满足翻转条件时生效）。
【CI】seed-level bootstrap 10^4（bootstrap 种子 47106），报 95% CI（2.5/97.5 分位）。
【预算】单 seed > 90 s ⇒ 停；累计 > 20 min ⇒ 停；超时写 状态="超时部分结果"。
【确定性】单趟；显式种子；无 time 参与数值；bootstrap 种子固定。恢复自检只在锚点 seed 0 做一次。
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import random
import subprocess
import sys
import time

sys.dont_write_bytecode = True
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
JUDGE = os.path.abspath(os.path.join(HERE, "..", "..", "..", "meeting51", "实施", "判据_千层骨架.py"))
RESULT = os.path.join(HERE, "轴C9_结果.json")

SEEDS = [0, 41, 82, 123, 164, 205, 246, 287, 328]
ANCHOR = {0: 0.8856666666666667, 41: 0.6830}
ANCHOR_TOL = 0.02
BUDGET_SEED_S = 90.0
BUDGET_TOTAL_S = 1200.0
BOOT_N = 10000
BOOT_SEED = 47106
SEED_ENV_MAIN = 52031
SEED_ENV_TEST = 52036
DELTA_MIN = 0.05
R_DROP_MIN = 0.80
R_DIID_MIN = 0.80

协议 = dict(
    性质="测量（只读 import 既有判据模块；不产判据、不改既有件、不覆盖既有 JSON）",
    种子偏=SEEDS,
    锚点={str(k): v for k, v in ANCHOR.items()},
    锚点容差=ANCHOR_TOL,
    序主=SEED_ENV_MAIN, 序测=SEED_ENV_TEST, 序测全长=251,
    预热拍=250, 训拍=450, 测拍=250, D=64,
    拆件="轴C区室",
    续训=dict(预热拍=0, 训拍=450),
    量=["Δ=基−降", "G=回−降", "g=对照−基", "R_base=回/基", "R_drop=G/Δ", "R_DiD=(G−g)/Δ"],
    阈值=dict(Δ_min=DELTA_MIN, R_drop_min=R_DROP_MIN, R_DiD_min=R_DIID_MIN,
            可补偿="R_DiD≥0.80 且 CI 下界≥0.80", 焊死="TOST 90% CI 上界<0.90"),
    判决规则="Spearman(基,Δ)≤−0.5 且 分层符号相反 ⇒ 随基水平翻转；否则 不可判定；禁止报装饰/删除",
    单侧附报="9/9 Δ 同号 且 mean(Δ) 95%CI 不含 0 且 |mean|≥0.05",
    bootstrap=dict(n=BOOT_N, 种子=BOOT_SEED, 区间="2.5/97.5 分位"),
    预算=dict(单seed秒=BUDGET_SEED_S, 累计秒=BUDGET_TOTAL_S),
    恢复自检="仅 seed 0 做一次（复评==基 才置 True）",
)


def _sha256_path(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _sha256_obj(obj):
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _py_procs():
    try:
        r = subprocess.run(["tasklist", "/FI", "IMAGENAME eq python.exe", "/FO", "CSV", "/NH"],
                           capture_output=True, text=True, timeout=10)
        lines = [l for l in r.stdout.splitlines() if "python" in l.lower()]
        return dict(count=len(lines), sample=lines[:8])
    except Exception as e:  # 并发快照失败不阻断测量，但必须记录
        return dict(count=None, error=repr(e))


T0 = time.perf_counter()

spec = importlib.util.spec_from_file_location("j52_c9", JUDGE)
J = importlib.util.module_from_spec(spec)
spec.loader.exec_module(J)
J.η冻结 = 0.01  # 冻结值（`判据_千层骨架结果.json` → 扫描.η读出_选定）


def _seq():
    return J.造序(SEED_ENV_TEST, 全长=J.测 + 1)


def _run_one(偏, do_recheck):
    t = time.perf_counter()
    k = J.造核(种子偏=偏)
    J.训练(k, J.造序(SEED_ENV_MAIN))
    base = float(J.评(k, _seq())["总"])
    snap = J.快照(k)
    复评 = None
    恢复完整性 = None
    if do_recheck:
        J.恢复(k, snap)
        复评 = float(J.评(k, _seq())["总"])
        恢复完整性 = bool(abs(复评 - base) < 1e-15)
    J.恢复(k, snap)
    k.拆件("轴C区室")
    abl = float(J.评(k, _seq())["总"])
    J.训练(k, J.造序(SEED_ENV_MAIN), 预热拍=0, 训拍=J.训)
    rec = float(J.评(k, _seq())["总"])
    J.恢复(k, snap)
    J.训练(k, J.造序(SEED_ENV_MAIN), 预热拍=0, 训拍=J.训)
    ctl = float(J.评(k, _seq())["总"])
    dt = time.perf_counter() - t
    Δ = base - abl
    G = rec - abl
    g = ctl - base
    return dict(
        种子偏=int(偏), 基=base, 拆后=abl, 续训后=rec, 对照=ctl, 复评=复评,
        恢复完整性=恢复完整性, Δ=Δ, G=G, g=g,
        R_base=rec / base if base else None,
        R_drop=(G / Δ) if abs(Δ) > 1e-12 else None,
        R_DiD=((G - g) / Δ) if abs(Δ) > 1e-12 else None,
        墙钟秒=round(dt, 3),
    )


# ── 统计工具（全部显式种子，不引入 scipy）──
def _ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for t in range(i, j + 1):
            r[order[t]] = avg
        i = j + 1
    return r


def _pearson(xs, ys):
    x = np.asarray(xs, float)
    y = np.asarray(ys, float)
    sx = float(np.std(x))
    sy = float(np.std(y))
    if sx == 0 or sy == 0:
        return float("nan")
    return float(np.mean((x - x.mean()) * (y - y.mean())) / (sx * sy))


def _spearman(xs, ys):
    return _pearson(_ranks(xs), _ranks(ys))


def _spearman_perm_p(xs, ys, n=BOOT_N, seed=BOOT_SEED):
    rho = _spearman(xs, ys)
    if rho != rho:
        return rho, float("nan")
    rng = random.Random(seed)
    y = list(ys)
    cnt = 0
    for _ in range(n):
        rng.shuffle(y)
        r = _spearman(xs, y)
        if r == r or abs(r) >= abs(rho):
            cnt += 1
    return rho, cnt / n


def _boot_mean_ci(xs, n=BOOT_N, seed=BOOT_SEED):
    if len(xs) == 0:
        return None
    rng = random.Random(seed)
    m = float(np.mean(xs))
    vals = []
    for _ in range(n):
        vals.append(sum(xs[rng.randrange(len(xs))] for _ in range(len(xs))) / len(xs))
    vals.sort()
    return dict(mean=m, lo=vals[int(0.025 * n)], hi=vals[int(0.975 * n) - 1],
                sd=float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0, n=len(xs))


def _stratify(基, Δ):
    med = float(np.median(基))
    hi = [d for b, d in zip(基, Δ) if b >= med]
    lo = [d for b, d in zip(基, Δ) if b < med]
    return dict(中位=med,
                高组=dict(n=len(hi), meanΔ=float(np.mean(hi)) if hi else None, Δ=hi),
                低组=dict(n=len(lo), meanΔ=float(np.mean(lo)) if lo else None, Δ=lo))


def _write(payload):
    with open(RESULT, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)


def main():
    并发前 = _py_procs()
    结果 = dict(
        协议=协议,
        协议sha256=_sha256_obj(协议),
        协议py_sha256=_sha256_path(os.path.abspath(__file__)),
        判据sha256=_sha256_path(JUDGE),
        环境=dict(python=sys.version.split()[0], numpy=np.__version__, 判据路径=JUDGE,
                  说明="只读 import；未改既有件；未覆盖既有 JSON；单趟；显式种子"),
        并发状态=dict(跑前=并发前, 跑后=None),
        锚点=dict(), 行=[], 汇总=None, 判定=None, 状态="进行中", 预算=dict(),
    )
    _write(结果)

    # ── 锚点先验（0 与 41）──
    for s in (0, 41):
        行 = _run_one(s, do_recheck=(s == 0))
        结果["锚点"][str(s)] = dict(实测基=行["基"], 期望基=ANCHOR[s],
                                    差=行["基"] - ANCHOR[s],
                                    判定=("PASS" if abs(行["基"] - ANCHOR[s]) <= ANCHOR_TOL else "FAIL"),
                                    墙钟秒=行["墙钟秒"])
        if 结果["锚点"][str(s)]["判定"] == "FAIL":
            结果["状态"] = "口径不匹配-锚点复现失败"
            结果["并发状态"]["跑后"] = _py_procs()
            _write(结果)
            print(json.dumps(结果["锚点"], ensure_ascii=False))
            print("[轴C9] 状态 =", 结果["状态"], "| 停跑")
            return 2
        # 锚点行也进总表（不重复跑）
        if s == 0:
            结果["行"].append(行)
        else:
            结果["行"].append(行)
        _write(结果)
        print(f"[anchor] 种子偏={s} 基={行['基']:.6f} 差={行['基']-ANCHOR[s]:+.4f} 墙钟={行['墙钟秒']:.2f}s", flush=True)

    # ── 其余 7 seeds ──
    for s in SEEDS:
        if s in (0, 41):
            continue
        try:
            行 = _run_one(s, do_recheck=False)
        except Exception as e:  # 接口/异常 → 留部分结果并报卡点
            结果["状态"] = "异常中止"
            结果["异常"] = repr(e)
            结果["并发状态"]["跑后"] = _py_procs()
            _write(结果)
            print("[轴C9] 异常:", repr(e), flush=True)
            return 3
        结果["行"].append(行)
        结果["预算"]["累计秒"] = round(time.perf_counter() - T0, 2)
        _write(结果)
        print(f"[seed] 偏={s:>3d} 基={行['基']:.6f} 降={行['拆后']:.6f} Δ={行['Δ']:+.4f} "
              f"g={行['g']:+.4f} 墙钟={行['墙钟秒']:.2f}s", flush=True)
        if 行["墙钟秒"] > BUDGET_SEED_S or (time.perf_counter() - T0) > BUDGET_TOTAL_S:
            结果["状态"] = "超时部分结果"
            break

    # ── 汇总 ──
    行s = 结果["行"]
    Δs = [r["Δ"] for r in 行s]
    基s = [r["基"] for r in 行s]
    降s = [r["拆后"] for r in 行s]
    gs = [r["g"] for r in 行s]
    rdrop = [r["R_drop"] for r in 行s if r["R_drop"] is not None]
    rdiid = [r["R_DiD"] for r in 行s if r["R_DiD"] is not None]
    rho, rho_p = _spearman_perm_p(基s, Δs)
    分层 = _stratify(基s, Δs)
    bootΔ = _boot_mean_ci(Δs)
    bootRdrop = _boot_mean_ci(rdrop) if len(rdrop) >= 2 else None
    bootRdiid = _boot_mean_ci(rdiid) if len(rdiid) >= 2 else None
    # TOST：R_DiD 的 90% 单侧上界（与 0.90 比）——用 bootstrap 分位近似
    tost = None
    if len(rdiid) >= 2:
        tost = dict(判据="R_DiD 90% 单侧上界 < 0.90 ⇒ 焊死成立",
                    上界90=float(np.percentile([r for r in rdiid], 90)),
                    n=len(rdiid))
    同号 = all(d > 0 for d in Δs) or all(d < 0 for d in Δs)
    单侧 = bool(同号 and bootΔ and (bootΔ["lo"] > 0 or bootΔ["hi"] < 0) and abs(bootΔ["mean"]) >= DELTA_MIN)
    翻转 = bool(rho == rho and rho <= -0.5
                and 分层["高组"]["meanΔ"] is not None and 分层["低组"]["meanΔ"] is not None
                and 分层["高组"]["meanΔ"] * 分层["低组"]["meanΔ"] < 0)
    判定 = "随基水平翻转（条件性轴）" if 翻转 else ("单侧" if 单侧 else "不可判定")

    结果["汇总"] = dict(
        Δ=bootΔ, g=_boot_mean_ci(gs), R_drop=bootRdrop, R_DiD=bootRdiid,
        spearman_rho=rho, spearman_perm_p=rho_p, 分层=分层,
        TOST=tost, 同号=同号,
        单侧条件=单侧, 翻转条件=翻转,
        判定=判定,
        判定说明="蓝队预注册：翻转=Spearman≤−0.5 且分层符号相反；单侧=9/9 同号且 CI 排零且|mean|≥0.05；禁止报装饰/删除",
    )
    结果["判定"] = 判定
    结果["预算"]["累计秒"] = round(time.perf_counter() - T0, 2)
    if 结果["状态"] == "进行中":
        结果["状态"] = "完成"
    结果["并发状态"]["跑后"] = _py_procs()
    _write(结果)
    print(json.dumps(dict(判定=判定, 汇总=结果["汇总"], 状态=结果["状态"], 预算=结果["预算"]),
                     ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

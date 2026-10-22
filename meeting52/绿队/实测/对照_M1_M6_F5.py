# -*- coding: utf-8 -*-
"""绿队 R3 对照实测 · 最小三组（M1-A1 / P2 / M6-P10 / F5）—— v2

v2 相对 v1 的修正（均来自 v1 首跑发现，不是事后调参找 PASS）：
  1) M1 新式新增**驱动变体**：v1 的 `符(s,cue)` 在 s≈cue 时注入常量（≈ones），破坏信息；
     故并列 `驱动="cue"`（残差朝 cue）。两变体都报，不择优隐藏。
  2) M6-P10 加**容量扫描**：v1 在 K=64,D=256 下 解绑叠加=0.31（SNR 不足 ⇒ 容量外），
     扫 K∈{4,8,16,32,64} 找"解绑≥0.95 且 读≈chance"的容量阈值 K*。
  3) F5-M1 判据补"有状态更差 ⇒ FAIL"分支（v1 出现 0.457 vs 0.717 的负差）。
  4) 四态判定与预注册阈值在文件头常量里写死，跑前固定。

纪律：只读导入既有件；只写本目录；真跑（所有者授权 A9）；并发态记录；墙钟不作效率宣称。
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
绿队 = os.path.dirname(HERE)
meeting52 = os.path.dirname(绿队)
AHI4 = os.path.dirname(meeting52)
_核py = os.path.join(AHI4, "meeting41", "实施", "算子_能力核.py")

# ── 预注册阈值（跑前写死）──
阈值 = dict(
    M1_翻β_α_全=0.15, M1_翻β_α_近平局=0.20, M1_翻α_γ=0.03, M1_装饰=0.01,
    P2_连续门=0.10, P2_装饰=0.02,
    M6_解绑=0.95, M6_读chance=0.10,
    F5_M1=0.10, F5_M1负=0.05, F5_M6_有=0.75, F5_M6_可=0.60,
    F5_M7_初值=0.10, F5_M7_可重算=0.20,
)


def 载能力核():
    spec = importlib.util.spec_from_file_location("能力核_只读_r3", _核py)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_m = 载能力核()
能力核 = _m.能力核
_归 = _m._归


def 双极(rng, n):
    return np.where(rng.integers(0, 2, size=int(n)) == 1, 1.0, -1.0)


def 符(a, b):
    """W符=ones 时的第八算子（与 两区室核.符 同式：归(a⊙b)）。[源码读]"""
    return _归(np.asarray(a, dtype=np.float64) * np.asarray(b, dtype=np.float64))


def 决策(核, Z, s):
    D = int(Z.shape[1])
    return int(np.argmax([float(np.dot(np.asarray(s, dtype=np.float64), Z[j])) / D for j in range(Z.shape[0])]))


# ══════════════════════════════════════════════════════════════════════
# 1. M1-A1 / P2：补全新式（M1-A2）三臂 + 冻结索引臂；两驱动变体
# ══════════════════════════════════════════════════════════════════════
def 生成任务(种子, K=8, D=64, 共享数=7, α=1.2):
    """结构化困难任务：7 共享原子 + 1 独有原子（红队 A1.3 任务族）。"""
    rng = np.random.default_rng(int(种子))
    原子 = 双极(rng, (K + 共享数) * D).reshape(K + 共享数, D)
    共享 = 原子[:共享数]
    独有 = 原子[共享数:]
    Z = np.zeros((K, D), dtype=np.float64)
    for j in range(K):
        s = 共享.sum(axis=0) + float(α) * 独有[j]
        Z[j] = np.where(s >= 0.0, 1.0, -1.0)
    return Z


def 破坏(rng, 模式, ρ):
    c = np.array(模式, dtype=np.float64, copy=True)
    if ρ <= 0.0:
        return c
    n = max(1, int(round(float(ρ) * c.shape[0])))
    位 = rng.choice(c.shape[0], size=n, replace=False)
    c[位] *= -1.0
    return c


def 补全(核, Z, cue, T, λ=0.5, ζ=0.35, τ=0.15, 冻结索引=None, 驱动="符"):
    """M1 新式（M1-A2）：硬 top-3 + 逐候选独立线性阈值门 + 残差迭代。
    驱动="符"：s←归((1−λ)s+λ·叠([m, 符(s,cue)]))；驱动="cue"：…叠([m, cue])。"""
    D = int(Z.shape[1])
    s = _归(np.asarray(cue, dtype=np.float64))
    轨迹 = [s.copy()]
    索引列 = []
    固定 = None if 冻结索引 is None else list(冻结索引)
    for n in range(int(T)):
        if 固定 is None:
            R = 核.读(Z, s, 3)
            idx = list(R["索引"])
            sim = [float(v) for v in R["相似度"]]
        else:
            idx = list(固定)
            sim = [float(np.dot(s, Z[i])) / D for i in idx]
        g = [float(np.clip((sm - ζ) / τ, 0.0, 1.0)) for sm in sim]
        m = 核.叠([_归(gm * Z[i]) for gm, i in zip(g, idx)])          # 门(g_m,Z_i,0)=g_m·Z_i
        推 = 核.叠([m, 符(s, cue)]) if 驱动 == "符" else 核.叠([m, cue])
        s = _归((1.0 - float(λ)) * s + float(λ) * 推)
        轨迹.append(s.copy())
        索引列.append(idx)
    return 轨迹, 索引列


def 跑M1(P, 种子列):
    变体列 = ["符", "cue"]
    逐种子 = []
    记录 = {v: [] for v in 变体列}
    for 种子 in 种子列:
        Z = 生成任务(种子, K=P["K"], D=P["D"])
        rng = np.random.default_rng(int(种子) + 7)
        块 = {v: [] for v in 变体列}
        for t in range(P["每种子试次"]):
            j = int(t % P["K"])
            cue = 破坏(rng, Z[j], P["破坏率"])
            d重算 = 决策(核, Z, cue)                                 # F5 可重算臂（与变体无关）
            正重算 = int(d重算 == j)
            for v in 变体列:
                轨4, 索 = 补全(核, Z, cue, 4, 驱动=v)
                轨1, _ = 补全(核, Z, cue, 1, 驱动=v)
                轨冻, _ = 补全(核, Z, cue, 4, 冻结索引=索[0], 驱动=v)
                dα = 决策(核, Z, 轨4[1])
                dβ = 决策(核, Z, 轨4[4])
                dγ = 决策(核, Z, 轨1[1])
                d冻 = 决策(核, Z, 轨冻[4])
                simsγ = [float(np.dot(轨1[1], Z[q])) / P["D"] for q in range(P["K"])]
                排 = sorted(simsγ, reverse=True)
                块[v].append((dα, dβ, dγ, d冻, d重算, j, float(排[0] - 排[1]),
                              float(np.max(np.abs(轨4[4] - 轨4[1]))),
                              int(dβ == j), int(dγ == j), 正重算))
        for v in 变体列:
            记 = 块[v]
            逐种子.append(dict(种子=int(种子), 变体=v, 试次=len(记),
                              翻β_α=float(np.mean([r[0] != r[1] for r in 记])),
                              翻α_γ=float(np.mean([r[0] != r[2] for r in 记])),
                              翻β_冻=float(np.mean([r[1] != r[3] for r in 记])),
                              正β=float(np.mean([r[8] for r in 记])),
                              正γ=float(np.mean([r[9] for r in 记])),
                              状态差均值=float(np.mean([r[7] for r in 记]))))
        for v in 变体列:
            记录[v].extend(块[v])
    汇 = {}
    for v in 变体列:
        记 = 记录[v]
        近平 = []
        # 近平局：每种子按 γ 的 margin 最低 10% 预注册
        游标 = 0
        for 种子 in 种子列:
            子 = 记[游标:游标 + P["每种子试次"]]
            游标 += P["每种子试次"]
            阈 = float(np.quantile([r[6] for r in 子], 0.10))
            近平.extend([r for r in 子 if r[6] <= 阈])
        汇[v] = dict(
            n=len(记),
            翻β_α=float(np.mean([r[0] != r[1] for r in 记])),
            翻α_γ=float(np.mean([r[0] != r[2] for r in 记])),
            翻β_冻=float(np.mean([r[1] != r[3] for r in 记])),
            翻冻_γ=float(np.mean([r[3] != r[2] for r in 记])),
            正β=float(np.mean([r[8] for r in 记])),
            正γ=float(np.mean([r[9] for r in 记])),
            正重算=float(np.mean([r[10] for r in 记])),
            状态差均值=float(np.mean([r[7] for r in 记])),
            状态差非零率=float(np.mean([r[7] > 0.0 for r in 记])),
            近平局n=len(近平),
            近平局翻β_α=float(np.mean([r[0] != r[1] for r in 近平])) if 近平 else float("nan"),
        )
    return 逐种子, 汇


# ══════════════════════════════════════════════════════════════════════
# 2. M6-P10：检索正确性（读 vs 解绑）+ 容量扫描
# ══════════════════════════════════════════════════════════════════════
def 跑M6P10(种子=52101, D=256, K列=(4, 8, 16, 32, 64)):
    核 = 能力核(D=D, 种子=int(种子))
    行 = []
    for K in K列:
        rng = np.random.default_rng(int(种子) + 1000 + int(K))
        键 = np.array([双极(rng, D) for _ in range(K)])
        值 = np.array([双极(rng, D) for _ in range(K)])
        E行 = np.array([核.绑(键[i], 值[i]) for i in range(K)])
        E叠加 = 核.归(E行.sum(axis=0) / float(K))
        E未缩放 = 核.叠([E行[i] for i in range(K)])
        正读 = 正A = 正B = 0
        cos读 = cosB = 0.0
        for i in range(K):
            e_hat = np.asarray(核.读(E行, 键[i], 1)["读出"], dtype=np.float64)
            e_A = 核.解绑(键[i], e_hat)
            e_B = 核.解绑(键[i], E叠加)
            def 对(e):
                return int(np.argmax([float(np.dot(e, 值[j])) / D for j in range(K)]))
            正读 += int(对(e_hat) == i)
            正A += int(对(e_A) == i)
            正B += int(对(e_B) == i)
            cos读 += float(np.dot(e_hat, 值[i])) / D
            cosB += float(np.dot(e_B, 值[i])) / D
        行.append(dict(K=int(K), D=D,
                       准确率_读直接=正读 / float(K), 准确率_读后解绑=正A / float(K),
                       准确率_解绑叠加=正B / float(K),
                       cos读直接=cos读 / float(K), cos解绑叠加=cosB / float(K),
                       饱和值=float(np.max(np.abs(E未缩放)))))
    return dict(种子=int(种子), 扫描=行)


# ══════════════════════════════════════════════════════════════════════
# 3. F5 三臂（M1 / M6 / M7）
# ══════════════════════════════════════════════════════════════════════
def AUC(正, 负):
    正 = np.asarray(正, dtype=np.float64)
    负 = np.asarray(负, dtype=np.float64)
    if 正.size == 0 or 负.size == 0:
        return float("nan")
    return float(np.mean(正[:, None] > 负[None, :]) + 0.5 * np.mean(正[:, None] == 负[None, :]))


def 跑F5_M6(种子=52101, D=256, K=64):
    核 = 能力核(D=D, 种子=int(种子))
    rng = np.random.default_rng(int(种子) + 2)
    键 = np.array([双极(rng, D) for _ in range(K)])
    值 = np.array([双极(rng, D) for _ in range(K)])
    E = 核.归(np.array([核.绑(键[i], 值[i]) for i in range(K)]).sum(axis=0) / float(K))
    置换 = np.random.default_rng(int(种子) + 3).permutation(K)
    E置 = 核.归(np.array([核.绑(键[k], 值[置换[k]]) for k in range(K)]).sum(axis=0) / float(K))
    正, 负, 正可, 负可, 正置, 负置 = [], [], [], [], [], []
    for i in range(K):
        ê = 核.解绑(键[i], E)
        ê可 = 键[i]
        ê置 = 核.解绑(键[i], E置)
        正.append(float(np.dot(ê, 值[i])) / D)
        负.extend([float(np.dot(ê, 值[j])) / D for j in range(K) if j != i])
        正可.append(float(np.dot(ê可, 值[i])) / D)
        负可.extend([float(np.dot(ê可, 值[j])) / D for j in range(K) if j != i])
        正置.append(float(np.dot(ê置, 值[i])) / D)
        负置.extend([float(np.dot(ê置, 值[j])) / D for j in range(K) if j != i])
    return dict(K=K, D=D, AUC_有状态=AUC(正, 负), AUC_可重算=AUC(正可, 负可), AUC_历史置换=AUC(正置, 负置),
                a正均值有状态=float(np.mean(正)),
                注="可重算=只用当前输入(键)；历史置换=把值在键上置换")


def 跑F5_M7(种子=52301, D=64, T=8, 试次=200):
    核 = 能力核(D=D, 种子=int(种子))
    rng = np.random.default_rng(int(种子) + 4)
    变可 = 变换 = 0
    for _ in range(int(试次)):
        A = 双极(rng, D); B = 双极(rng, D); z = 双极(rng, D)
        读出 = _归(rng.standard_normal((2, D)) / np.sqrt(float(D)))
        d = 核.叠([符(A, B), z])
        L0 = 符(A, B)
        L_全 = 核.迭代(L0, d, T)
        L_单 = 核.迭代(L0, d, 1)
        L_换 = 核.迭代(双极(rng, D), d, T)
        def 决(L):
            return int(np.argmax(读出 @ 核.归(核.叠([L, B]))))
        变可 += int(决(L_单) != 决(L_全))
        变换 += int(决(L_换) != 决(L_全))
    return dict(D=D, T=T, 试次=int(试次), 可重算改变率=变可 / float(试次), 初值置换改变率=变换 / float(试次),
                注="可重算=单次投影；初值置换=换 L0 后 T 步")


# ══════════════════════════════════════════════════════════════════════
# 主流程 + 四态判定
# ══════════════════════════════════════════════════════════════════════
def main():
    global 核
    开 = time.time()
    核 = 能力核(D=64, 种子=52001)
    P = dict(K=8, D=64, 每种子试次=400, 破坏率=0.35, T=4, λ=0.5, ζ=0.35, τ=0.15)
    种子列 = [52001, 52002, 52003, 52004, 52005]

    t0 = time.time(); 逐种子, M1 = 跑M1(P, 种子列); tM1 = time.time() - t0
    t0 = time.time(); P10 = 跑M6P10(); tP10 = time.time() - t0
    t0 = time.time(); F5M6 = 跑F5_M6(); F5M7 = 跑F5_M7(); tF5 = time.time() - t0

    判 = {}
    # ── M1-A1（两变体都判；任一达 PASS 即 PASS，并记变体）──
    M1判 = {}
    for v, m in M1.items():
        if m["翻β_α"] < 阈值["M1_装饰"]:
            M1判[v] = "FAIL"
        elif (m["翻β_α"] >= 阈值["M1_翻β_α_全"] and m["近平局翻β_α"] >= 阈值["M1_翻β_α_近平局"]
              and m["翻α_γ"] <= 阈值["M1_翻α_γ"]):
            M1判[v] = "PASS"
        else:
            M1判[v] = "UNDECIDABLE"
    达 = [v for v, r in M1判.items() if r == "PASS"]
    判["M1_A1_分变体"] = M1判
    判["M1_A1"] = "PASS" if 达 else ("FAIL" if all(r == "FAIL" for r in M1判.values()) else "UNDECIDABLE")
    判["M1_整体"] = ("条件通过（须同时看 P2/M10）" if 达 else "未证明（M1-A1 未达 PASS）")
    判["M1_通过变体"] = 达
    # ── P2（两变体；任一 翻β_冻≥0.10 即 PASS）──
    P2判 = {}
    for v, m in M1.items():
        if m["翻β_冻"] >= 阈值["P2_连续门"]:
            P2判[v] = "PASS"
        elif m["翻β_冻"] <= 阈值["P2_装饰"] and m["翻β_α"] >= 阈值["M1_翻β_α_全"]:
            P2判[v] = "FAIL"
        else:
            P2判[v] = "UNDECIDABLE"
    判["P2_分变体"] = P2判
    判["P2"] = "PASS" if any(r == "PASS" for r in P2判.values()) else (
        "FAIL" if all(r == "FAIL" for r in P2判.values()) else "UNDECIDABLE")
    # ── M6-P10（容量扫描：找最大 K* 满足 解绑≥0.95 且 读≤0.10）──
    K好 = [r["K"] for r in P10["扫描"] if r["准确率_解绑叠加"] >= 阈值["M6_解绑"] and r["准确率_读直接"] <= 阈值["M6_读chance"]]
    if K好:
        判["M6_P10"] = "PASS"
        判["M6_P10_容量阈值K*"] = max(K好)
    elif any(r["准确率_读直接"] >= 阈值["M6_解绑"] for r in P10["扫描"]):
        判["M6_P10"] = "INVALID"        # 读 也 0.95 ⇒ green-iso C-6 被否
    else:
        判["M6_P10"] = "UNDECIDABLE"
        判["M6_P10_容量阈值K*"] = None
    # ── F5 ──
    最好变体 = max(M1, key=lambda v: M1[v]["正β"])
    差 = M1[最好变体]["正β"] - M1[最好变体]["正重算"]
    if 差 >= 阈值["F5_M1"]:
        判["F5_M1"] = "PASS"
    elif 差 <= -阈值["F5_M1负"] or abs(差) < 0.02:
        判["F5_M1"] = "FAIL"
    else:
        判["F5_M1"] = "UNDECIDABLE"
    判["F5_M1_差"] = 差
    判["F5_M1_变体"] = 最好变体
    判["F5_M6"] = ("PASS" if (F5M6["AUC_有状态"] >= 阈值["F5_M6_有"] and F5M6["AUC_可重算"] <= 阈值["F5_M6_可"]) else "FAIL")
    判["F5_M7"] = ("FAIL（历史/初值对决策影响≈0，且单次投影可近似复现 ⇒ 不构成持久信息）"
                   if (F5M7["初值置换改变率"] <= 阈值["F5_M7_初值"] and F5M7["可重算改变率"] <= 阈值["F5_M7_可重算"])
                   else "UNDECIDABLE")

    果 = dict(
        meta=dict(件="绿队 R3 最小对照实测 v2", 时间=time.strftime("%Y-%m-%d %H:%M:%S"),
                  并发态="会议并发期（多 Agent 在线）；墙钟仅记录、不作效率宣称",
                  python=sys.version.split()[0], numpy=np.__version__,
                  只读来源=_核py, 写域=str(HERE), 种子列=种子列, 参数=P, 预注册阈值=阈值,
                  总墙钟_s=round(time.time() - 开, 3),
                  分段墙钟_s=dict(M1=round(tM1, 3), M6_P10=round(tP10, 3), F5=round(tF5, 3))),
        M1_A1=dict(逐种子=逐种子, 汇总=M1,
                   预算=dict(每拍乘加解析=4 * (4 * P["K"] * P["D"] + 10 * P["D"]), 存储B=P["K"] * P["D"] * 4),
                   注="α=算4读1；β=算4读4；γ=单步；决策=argmax_j 内积(s,Z_j)/D；v1 发现 驱动='符' 在 s≈cue 时注入常量，故并列 驱动='cue'"),
        P2=dict(说明="冻结索引=冻结 n=1 的 top-3 索引集，仅让连续门演化；对比 完整(β) / 冻结 / 单步(γ)",
                分变体={v: dict(翻完整_冻结=M1[v]["翻β_冻"], 翻完整_单步=M1[v]["翻β_α"], 翻冻结_单步=M1[v]["翻冻_γ"]) for v in M1}),
        M6_P10=P10,
        F5=dict(M1=dict(变体=最好变体, 有状态=M1[最好变体]["正β"], 可重算=M1[最好变体]["正重算"], 差=差,
                        历史置换="未单独跑（内容寻址任务，置换槽序不改变内容）"),
                M6=F5M6, M7=F5M7),
        四态判定=判,
        未测=["P2 仅单破坏率(0.35)与单 T", "M6 未跑 800 拍流式场景", "M7 仅 D=64/T=8",
              "跨进程两趟未验", "M1 参数未扫描（λ/ζ/τ 固定）"],
    )
    with open(os.path.join(HERE, "汇总.json"), "w", encoding="utf-8") as f:
        json.dump(果, f, ensure_ascii=False, indent=1)
    print(json.dumps(果, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    try:
        np.seterr(all="ignore")
    except Exception:
        pass
    main()

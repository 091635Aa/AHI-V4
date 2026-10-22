# -*- coding: utf-8 -*-
"""绿队 R3 补臂 · C1 残差门控（M1 最小候选；主控 2026-10-17 开窗批准）

预注册（见 `M1_重设计一页纸.md` §2/§3）：
  更新式：ρ=‖m−s‖∞；u=clip((ρ−δ0)/δ1,0,1)；s←归(s+η·u·(m−s))
  判据：翻转 βvsα ≥0.15（全样本）/ ≥0.20（近平局，γ margin 最低 10%）；P2 完整vs冻结 ≥0.10；
        召回 ≥ 直接检索+0.05；翻转 <0.01 判装饰。
  must-fail：冻结索引臂 / 单步投影臂 / 随机槽臂 / 无门臂(g≡0)。
  参数：三段切分——验证段种子 54001–54003 选参，测试段 54004–54006 只评一次。

纪律：只读导入既有件；只写本目录；并发期（红/蓝/黄 R3 在线）**只作正确性判定、墙钟不宣称**。
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

阈值 = dict(翻转全=0.15, 翻转近平=0.20, 装饰=0.01, P2=0.10, 直接检索增益=0.05)
验证种子 = [54001, 54002, 54003]
测试种子 = [54004, 54005, 54006]
网格 = [dict(λ=l, δ0=d0, δ1=0.2, η=e, T=4)
        for l in (0.3, 0.5, 0.7) for d0 in (0.01, 0.05) for e in (0.5, 0.8)]


def 载能力核():
    spec = importlib.util.spec_from_file_location("能力核_只读_r3c", _核py)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


_m = 载能力核()
能力核 = _m.能力核
_归 = _m._归


def 双极(rng, n):
    return np.where(rng.integers(0, 2, size=int(n)) == 1, 1.0, -1.0)


def 生成任务(种子, K=8, D=64, 共享数=7, α=1.2):
    rng = np.random.default_rng(int(种子))
    原子 = 双极(rng, (K + 共享数) * D).reshape(K + 共享数, D)
    共享, 独有 = 原子[:共享数], 原子[共享数:]
    Z = np.zeros((K, D))
    for j in range(K):
        s = 共享.sum(axis=0) + float(α) * 独有[j]
        Z[j] = np.where(s >= 0.0, 1.0, -1.0)
    return Z


def 破坏(rng, 模式, ρ):
    c = np.array(模式, dtype=np.float64, copy=True)
    n = max(1, int(round(float(ρ) * c.shape[0])))
    c[rng.choice(c.shape[0], size=n, replace=False)] *= -1.0
    return c


def 补全C1(核, Z, cue, T, λ, δ0, δ1, η, 冻结索引=None, 无门=False):
    """C1 残差门控：s ← 归(s + η·u·(m−s))；u 由残差内生。"""
    D = int(Z.shape[1])
    s = _归(np.asarray(cue, dtype=np.float64))
    轨迹 = [s.copy()]; 索 = []
    固定 = None if 冻结索引 is None else list(冻结索引)
    for n in range(int(T)):
        if 固定 is None:
            R = 核.读(Z, s, 3); idx = list(R["索引"]); sim = [float(v) for v in R["相似度"]]
        else:
            idx = list(固定); sim = [float(np.dot(s, Z[i])) / D for i in idx]
        g = [float(np.clip((sm - 0.35) / 0.15, 0.0, 1.0)) for sm in sim]
        if 无门:
            m = np.zeros(D)
        else:
            m = 核.叠([_归(gm * Z[i]) for gm, i in zip(g, idx)])
        ρ = float(np.max(np.abs(m - s)))
        u = float(np.clip((ρ - δ0) / δ1, 0.0, 1.0))
        s = _归(s + float(η) * u * (m - s))
        轨迹.append(s.copy()); 索.append(idx)
    return 轨迹, 索


def 决策(核, Z, s):
    D = int(Z.shape[1])
    sims = [float(np.dot(s, Z[j])) / D for j in range(Z.shape[0])]
    return int(np.argmax(sims)), sims


def 跑段(种子列, 配置, 段名, 每种子=400, 破坏率=0.35, 选参=False):
    """返回：所有试次记录 + 汇总。选参段只算 β/α/γ/直接；测试段另跑 must-fail。"""
    记 = []
    for 种子 in 种子列:
        Z = 生成任务(种子)
        rng = np.random.default_rng(int(种子) + 11)
        for t in range(每种子):
            j = int(t % 8); cue = 破坏(rng, Z[j], 破坏率)
            轨4, 索 = 补全C1(核, Z, cue, 配置["T"], 配置["λ"], 配置["δ0"], 配置["δ1"], 配置["η"])
            轨1, _ = 补全C1(核, Z, cue, 1, 配置["λ"], 配置["δ0"], 配置["δ1"], 配置["η"])
            dβ, _ = 决策(核, Z, 轨4[-1])
            dα, _ = 决策(核, Z, 轨4[1])
            dγ, simsγ = 决策(核, Z, 轨1[1])
            排 = sorted(simsγ, reverse=True); margin = float(排[0] - 排[1])
            d直, _ = 决策(核, Z, cue)
            行 = dict(j=j, dβ=dβ, dα=dα, dγ=dγ, d直=d直, margin=margin,
                      正β=int(dβ == j), 正γ=int(dγ == j), 正直=int(d直 == j),
                      状态差=float(np.max(np.abs(轨4[-1] - 轨4[1]))))
            if not 选参:
                轨冻, _ = 补全C1(核, Z, cue, 配置["T"], 配置["λ"], 配置["δ0"], 配置["δ1"], 配置["η"], 冻结索引=索[0])
                轨无门, _ = 补全C1(核, Z, cue, 配置["T"], 配置["λ"], 配置["δ0"], 配置["δ1"], 配置["η"], 无门=True)
                Zr = 生成任务(int(种子) + 777)                    # 随机槽臂：同形状随机模式
                轨随, _ = 补全C1(核, Zr, cue, 配置["T"], 配置["λ"], 配置["δ0"], 配置["δ1"], 配置["η"])
                行.update(d冻=决策(核, Z, 轨冻[-1])[0],
                          d无门=决策(核, Z, 轨无门[-1])[0],
                          d随机=决策(核, Zr, 轨随[-1])[0])
            记.append(行)
    汇 = dict(段=段名, n=len(记),
              翻β_α=float(np.mean([r["dβ"] != r["dα"] for r in 记])),
              翻α_γ=float(np.mean([r["dα"] != r["dγ"] for r in 记])),
              正β=float(np.mean([r["正β"] for r in 记])),
              正γ=float(np.mean([r["正γ"] for r in 记])),
              正直=float(np.mean([r["正直"] for r in 记])),
              状态差均值=float(np.mean([r["状态差"] for r in 记])))
    近平 = []
    游标 = 0
    for _ in 种子列:
        子 = 记[游标:游标 + 每种子]; 游标 += 每种子
        阈 = float(np.quantile([r["margin"] for r in 子], 0.10))
        近平.extend([r for r in 子 if r["margin"] <= 阈])
    汇["近平局n"] = len(近平)
    汇["近平局翻β_α"] = float(np.mean([r["dβ"] != r["dα"] for r in 近平])) if 近平 else float("nan")
    if not 选参:
        汇.update(
            翻β_冻=float(np.mean([r["dβ"] != r["d冻"] for r in 记])),
            翻β_无门=float(np.mean([r["dβ"] != r["d无门"] for r in 记])),
            正随机=float(np.mean([r["d随机"] == r["j"] for r in 记])),
            正无门=float(np.mean([r["d无门"] == r["j"] for r in 记])),
        )
    return 汇, 记


def main():
    global 核
    核 = 能力核(D=64, 种子=54001)
    果 = dict(meta=dict(件="C1 残差门控（M1 最小候选）", 时间=time.strftime("%Y-%m-%d %H:%M:%S"),
                        并发态="红/蓝/黄 R3 在线；只作正确性判定，墙钟不宣称",
                        python=sys.version.split()[0], numpy=np.__version__,
                        只读来源=_核py, 验证种子=验证种子, 测试种子=测试种子,
                        预注册阈值=阈值, 网格=网格))
    # ── 验证段选参 ──
    表 = []
    for i, cfg in enumerate(网格):
        汇, _ = 跑段(验证种子, cfg, "验证段", 每种子=200, 选参=True)
        表.append(dict(配置编号=i, **cfg, 翻β_α=汇["翻β_α"], 正β=汇["正β"], 正直=汇["正直"],
                       近平局翻β_α=汇["近平局翻β_α"]))
    果["验证段"] = 表
    可选 = [r for r in 表 if r["翻β_α"] >= 阈值["翻转全"]]
    选中 = max(可选 if 可选 else 表, key=lambda r: r["正β"])
    果["选中配置"] = 选中
    cfg = dict(λ=选中["λ"], δ0=选中["δ0"], δ1=选中["δ1"], η=选中["η"], T=int(选中["T"]))
    # ── 测试段（只评一次） ──
    汇, 记 = 跑段(测试种子, cfg, "测试段", 每种子=400, 选参=False)
    果["测试段"] = 汇
    # ── 四态判定（预注册） ──
    P2过 = 汇["翻β_冻"] >= 阈值["P2"]
    召回过 = 汇["正β"] >= 汇["正直"] + 阈值["直接检索增益"]
    翻转过 = 汇["翻β_α"] >= 阈值["翻转全"]
    近平过 = 汇["近平局翻β_α"] >= 阈值["翻转近平"]
    装饰 = 汇["翻β_α"] < 阈值["装饰"]
    随机槽过 = 汇["正随机"] <= 0.25
    # 无门臂：s←s+η(0−s) 只是把 s 按比例缩小；argmax 对缩放不变 ⇒ 决策不可能变。
    # 这是 must-fail 臂自身的设计失效（记 INVALID），不能据此判"门无效"。
    无门尺度不变 = abs(汇["正无门"] - 汇["正直"]) < 1e-9
    无门标 = ("INVALID（决策对 s 缩放不变 ⇒ must-fail 臂设计失效，不能据此判门无效）"
             if 无门尺度不变 else
             ("符合" if (汇["正无门"] <= 0.25 and 汇["翻β_无门"] >= 0.50) else "不符合"))
    判 = dict(
        翻转_全="过" if 翻转过 else "不过", 翻转_近平局="过" if 近平过 else "不过",
        P2="过" if P2过 else "不过", 召回_优于直接检索="过" if 召回过 else "不过",
        随机槽_mustfail="符合" if 随机槽过 else "不符合",
        无门_mustfail=无门标,
        装饰="是" if 装饰 else "否",
    )
    if 装饰 or not 召回过:
        四 = "FAIL"
    elif 翻转过 and 近平过 and P2过 and 随机槽过 and 无门标 == "符合":
        四 = "PASS"
    else:
        四 = "UNDECIDABLE"
    果["四态判定"] = 四
    果["判定明细"] = 判
    果["对直接检索"] = dict(正β=汇["正β"], 正直=汇["正直"], 增益=汇["正β"] - 汇["正直"])
    with open(os.path.join(HERE, "C1_结果.json"), "w", encoding="utf-8") as f:
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

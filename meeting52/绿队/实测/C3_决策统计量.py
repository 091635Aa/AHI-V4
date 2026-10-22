# -*- coding: utf-8 -*-
"""绿队 · C3 单臂实测：S1「与 cue 重合」决策统计量（主控 2026-10-17 批准）

预注册见 `C3_spec.md`：
  PASS 需：S1 召回 ≥ S0+0.05 且 ≥ 直接检索+0.05；翻转(S1,S0)≥0.15；
          翻转 βvsα(S1) ≥0.15（近平局 ≥0.20）；随机槽 ≤0.25；尺度不变自查逐位同。
  FAIL：翻转(S1,S0)<0.02 或 S1 召回 ≤ 直接检索。
机制固定：C1 残差门控（λ=0.3, δ0=0.01, δ1=0.2, η=0.8, T=4；C1 已排除，仅借其状态轨迹测统计量）。
纪律：只读既有件；只写本目录；并发期只作正确性、墙钟不宣称。
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

阈值 = dict(召回增益=0.05, 翻转统计量=0.15, 翻转全=0.15, 翻转近平=0.20, 随机槽=0.25, 等价=0.02)
测试种子 = [54004, 54005, 54006]
机制 = dict(λ=0.3, δ0=0.01, δ1=0.2, η=0.8, T=4)
每种子 = 400
破坏率 = 0.35


def 载能力核():
    spec = importlib.util.spec_from_file_location("能力核_只读_r3c3", _核py)
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
        Z[j] = np.where(共享.sum(axis=0) + float(α) * 独有[j] >= 0.0, 1.0, -1.0)
    return Z


def 破坏(rng, 模式, ρ):
    c = np.array(模式, dtype=np.float64, copy=True)
    n = max(1, int(round(float(ρ) * c.shape[0])))
    c[rng.choice(c.shape[0], size=n, replace=False)] *= -1.0
    return c


def 补全C1(核, Z, cue, cfg):
    """C1 残差门控（同 R3 §8 配置）。"""
    D = int(Z.shape[1])
    s = _归(np.asarray(cue, dtype=np.float64)); 轨 = [s.copy()]
    for _ in range(int(cfg["T"])):
        R = 核.读(Z, s, 3)
        g = [float(np.clip((float(v) - 0.35) / 0.15, 0.0, 1.0)) for v in R["相似度"]]
        m = 核.叠([_归(gm * Z[i]) for gm, i in zip(g, R["索引"])])
        ρ = float(np.max(np.abs(m - s)))
        u = float(np.clip((ρ - cfg["δ0"]) / cfg["δ1"], 0.0, 1.0))
        s = _归(s + float(cfg["η"]) * u * (m - s)); 轨.append(s.copy())
    return 轨


def 分S0(核, Z, s):
    D = int(Z.shape[1]); return [float(np.dot(s, Z[j])) / D for j in range(Z.shape[0])]


def 分S1(核, Z, s, cue):
    """S1：与 cue 重合 = <符(s,Z_j), cue>/D。"""
    D = int(Z.shape[1])
    return [float(np.dot(_归(s * Z[j]), cue)) / D for j in range(Z.shape[0])]


def 跑():
    记 = []
    for 种子 in 测试种子:
        Z = 生成任务(种子); Zr = 生成任务(int(种子) + 777)
        rng = np.random.default_rng(int(种子) + 11)
        for t in range(每种子):
            j = int(t % 8); cue = 破坏(rng, Z[j], 破坏率)
            轨4 = 补全C1(核, Z, cue, 机制); 轨1 = 补全C1(核, Z, cue, dict(机制, T=1))
            sβ, sα, sγ = 轨4[-1], 轨4[1], 轨1[1]
            # S0
            dS0β = int(np.argmax(分S0(核, Z, sβ))); dS0α = int(np.argmax(分S0(核, Z, sα)))
            dS0γ = int(np.argmax(分S0(核, Z, sγ)))
            # S1
            vS1β = 分S1(核, Z, sβ, cue); vS1α = 分S1(核, Z, sα, cue); vS1γ = 分S1(核, Z, sγ, cue)
            dS1β = int(np.argmax(vS1β)); dS1α = int(np.argmax(vS1α)); dS1γ = int(np.argmax(vS1γ))
            排 = sorted(vS1γ, reverse=True); margin = float(排[0] - 排[1])
            d直 = int(np.argmax(分S0(核, Z, cue)))
            # 随机槽（S1 在随机模式上）
            轨随 = 补全C1(核, Zr, cue, 机制); d随 = int(np.argmax(分S1(核, Zr, 轨随[-1], cue)))
            # 尺度不变自查：s→2s 后 S1 决策必须同
            dS1倍 = int(np.argmax(分S1(核, Z, 2.0 * sβ, cue)))
            记.append(dict(j=j, margin=margin,
                           dS1β=dS1β, dS1α=dS1α, dS0β=dS0β, dS0α=dS0α, d直=d直, d随=d随,
                           尺度同=int(dS1倍 == dS1β),
                           正S1β=int(dS1β == j), 正S0β=int(dS0β == j), 正直=int(d直 == j)))
    汇 = dict(n=len(记),
              召回_S1=float(np.mean([r["正S1β"] for r in 记])),
              召回_S0=float(np.mean([r["正S0β"] for r in 记])),
              召回_直接=float(np.mean([r["正直"] for r in 记])),
              翻转_S1_S0=float(np.mean([r["dS1β"] != r["dS0β"] for r in 记])),
              翻转β_α_S1=float(np.mean([r["dS1β"] != r["dS1α"] for r in 记])),
              翻转β_α_S0=float(np.mean([r["dS0β"] != r["dS0α"] for r in 记])),
              随机槽_S1=float(np.mean([r["d随"] == r["j"] for r in 记])),
              尺度不变率=float(np.mean([r["尺度同"] for r in 记])))
    近平 = []
    游标 = 0
    for _ in 测试种子:
        子 = 记[游标:游标 + 每种子]; 游标 += 每种子
        阈 = float(np.quantile([r["margin"] for r in 子], 0.10))
        近平.extend([r for r in 子 if r["margin"] <= 阈])
    汇["近平局n"] = len(近平)
    汇["近平局翻转β_α_S1"] = float(np.mean([r["dS1β"] != r["dS1α"] for r in 近平])) if 近平 else float("nan")
    return 汇


def main():
    global 核
    核 = 能力核(D=64, 种子=54004)
    t0 = time.time()
    汇一 = 跑()
    汇二 = 跑()
    同 = (汇一 == 汇二)
    判 = dict(
        召回_增益_S1_S0=汇一["召回_S1"] - 汇一["召回_S0"],
        召回_增益_S1_直接=汇一["召回_S1"] - 汇一["召回_直接"],
    )
    过召回 = (判["召回_增益_S1_S0"] >= 阈值["召回增益"] and 判["召回_增益_S1_直接"] >= 阈值["召回增益"])
    过翻转统计 = 汇一["翻转_S1_S0"] >= 阈值["翻转统计量"]
    过迭代 = 汇一["翻转β_α_S1"] >= 阈值["翻转全"] and 汇一["近平局翻转β_α_S1"] >= 阈值["翻转近平"]
    过随机槽 = 汇一["随机槽_S1"] <= 阈值["随机槽"]
    过尺度 = abs(汇一["尺度不变率"] - 1.0) < 1e-12
    判.update(召回="过" if 过召回 else "不过", 统计量真改="过" if 过翻转统计 else "不过",
              迭代可见="过" if 过迭代 else "不过", 随机槽="符合" if 过随机槽 else "不符合",
              尺度不变="通过（逐位同）" if 过尺度 else "失败（实现 bug）")
    if (not 过翻转统计) or (汇一["召回_S1"] <= 汇一["召回_直接"]):
        四 = "FAIL"
    elif 过召回 and 过翻转统计 and 过迭代 and 过随机槽 and 过尺度:
        四 = "PASS"
    else:
        四 = "UNDECIDABLE"
    果 = dict(meta=dict(件="C3 单臂 · S1 与 cue 重合", 时间=time.strftime("%Y-%m-%d %H:%M:%S"),
                        并发态="R3 并发期；只作正确性判定，墙钟不宣称",
                        测试种子=测试种子, 每种子=每种子, 机制=机制, 预注册阈值=阈值,
                        墙钟_s=round(time.time() - t0, 3), 两趟逐位同=bool(同)),
              汇总=汇一, 判定明细=判, 四态判定=四)
    with open(os.path.join(HERE, "C3_结果.json"), "w", encoding="utf-8") as f:
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

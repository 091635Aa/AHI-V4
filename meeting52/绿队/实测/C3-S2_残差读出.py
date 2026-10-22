# -*- coding: utf-8 -*-
"""绿队 · C3 单臂 S2：残差读出（`⟨s−cue, Z_j⟩/D`；主控 2026-10-17 开窗批准）

预注册（主控口径）：
  翻转 βvsα ≥0.15（近平局 ≥0.20）；P2（完整 vs 冻结索引，S2 口径）≥0.10；
  召回 S2 ≥ 直接检索+0.05；随机槽 ≤0.25；实现自查；两趟逐位同。
  注：S2 是**仿射**统计量（含 `⟨cue,Z_j⟩` 常数项），对 s 尺度不不变 ⇒
      方向型统计量的"尺度不变自查"对 S2 **不适用**；改为"尺度缩放特性记录"+"cue 替换自查"。
机制固定：C1 残差门控（λ=0.3, δ0=0.01, δ1=0.2, η=0.8, T=4；仅借其状态轨迹）。
纪律：只读既有件；只写本目录；并发期墙钟不宣称。
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

阈值 = dict(召回增益=0.05, 翻转全=0.15, 翻转近平=0.20, P2=0.10, 随机槽=0.25, 等价=0.02)
测试种子 = [54004, 54005, 54006]
机制 = dict(λ=0.3, δ0=0.01, δ1=0.2, η=0.8, T=4)
每种子 = 400
破坏率 = 0.35


def 载能力核():
    spec = importlib.util.spec_from_file_location("能力核_只读_r3c3s2", _核py)
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


def 补全C1(核, Z, cue, cfg, 冻结索引=None):
    D = int(Z.shape[1]); s = _归(np.asarray(cue, dtype=np.float64)); 轨 = [s.copy()]; 索 = []
    固定 = None if 冻结索引 is None else list(冻结索引)
    for _ in range(int(cfg["T"])):
        if 固定 is None:
            R = 核.读(Z, s, 3); idx = list(R["索引"]); sim = [float(v) for v in R["相似度"]]
        else:
            idx = list(固定); sim = [float(np.dot(s, Z[i])) / D for i in idx]
        g = [float(np.clip((sm - 0.35) / 0.15, 0.0, 1.0)) for sm in sim]
        m = 核.叠([_归(gm * Z[i]) for gm, i in zip(g, idx)])
        ρ = float(np.max(np.abs(m - s)))
        u = float(np.clip((ρ - cfg["δ0"]) / cfg["δ1"], 0.0, 1.0))
        s = _归(s + float(cfg["η"]) * u * (m - s)); 轨.append(s.copy()); 索.append(idx)
    return 轨, 索


def 分S0(核, Z, s):
    D = int(Z.shape[1]); return [float(np.dot(s, Z[j])) / D for j in range(Z.shape[0])]


def 分S2(核, Z, s, cue):
    """S2 残差读出：⟨s−cue, Z_j⟩/D。"""
    D = int(Z.shape[1]); d = np.asarray(s, dtype=np.float64) - np.asarray(cue, dtype=np.float64)
    return [float(np.dot(d, Z[j])) / D for j in range(Z.shape[0])]


def 跑():
    记 = []
    for 种子 in 测试种子:
        Z = 生成任务(种子); Zr = 生成任务(int(种子) + 777)
        rng = np.random.default_rng(int(种子) + 11)
        for t in range(每种子):
            j = int(t % 8); cue = 破坏(rng, Z[j], 破坏率)
            轨4, 索 = 补全C1(核, Z, cue, 机制)
            轨冻, _ = 补全C1(核, Z, cue, 机制, 冻结索引=索[0])
            轨1, _ = 补全C1(核, Z, cue, dict(机制, T=1))
            sβ = 轨4[-1]; sα = 轨4[1]; sγ = 轨1[1]
            vS2β = 分S2(核, Z, sβ, cue); vS2α = 分S2(核, Z, sα, cue); vS2γ = 分S2(核, Z, sγ, cue)
            dS2β = int(np.argmax(vS2β)); dS2α = int(np.argmax(vS2α)); dS2γ = int(np.argmax(vS2γ))
            dS2冻 = int(np.argmax(分S2(核, Z, 轨冻[-1], cue)))
            排 = sorted(vS2γ, reverse=True); margin = float(排[0] - 排[1])
            dS0β = int(np.argmax(分S0(核, Z, sβ)))
            d直 = int(np.argmax(分S0(核, Z, cue)))
            # 随机槽（S2 口径）
            轨随, _ = 补全C1(核, Zr, cue, 机制)
            d随 = int(np.argmax(分S2(核, Zr, 轨随[-1], cue)))
            # 自查①：重复计算逐位一致；自查②：尺度缩放特性（S2 仿射 ⇒ 允许改变）；自查③：cue 替换必须改决策
            dS2β重复 = int(np.argmax(分S2(核, Z, 轨4[-1], cue)))
            dS2倍 = int(np.argmax(分S2(核, Z, 2.0 * sβ, cue)))
            cue2 = 破坏(rng, Z[(j + 1) % 8], 破坏率)
            dS2换线 = int(np.argmax(分S2(核, Z, sβ, cue2)))
            记.append(dict(j=j, margin=margin,
                           dS2β=dS2β, dS2α=dS2α, dS2冻=dS2冻, dS0β=dS0β, d直=d直, d随=d随,
                           自查重复=int(dS2β重复 == dS2β), 自查尺度变=int(dS2倍 != dS2β),
                           自查换线变=int(dS2换线 != dS2β),
                           正S2β=int(dS2β == j), 正S0β=int(dS0β == j), 正直=int(d直 == j)))
    汇 = dict(n=len(记),
              召回_S2=float(np.mean([r["正S2β"] for r in 记])),
              召回_S0=float(np.mean([r["正S0β"] for r in 记])),
              召回_直接=float(np.mean([r["正直"] for r in 记])),
              翻转_S2_S0=float(np.mean([r["dS2β"] != r["dS0β"] for r in 记])),
              翻转β_α_S2=float(np.mean([r["dS2β"] != r["dS2α"] for r in 记])),
              P2_翻完整_冻结_S2=float(np.mean([r["dS2β"] != r["dS2冻"] for r in 记])),
              随机槽_S2=float(np.mean([r["d随"] == r["j"] for r in 记])),
              自查_重复一致率=float(np.mean([r["自查重复"] for r in 记])),
              自查_尺度缩放改变率=float(np.mean([r["自查尺度变"] for r in 记])),
              自查_换线索改变率=float(np.mean([r["自查换线变"] for r in 记])))
    近平 = []
    游标 = 0
    for _ in 测试种子:
        子 = 记[游标:游标 + 每种子]; 游标 += 每种子
        阈 = float(np.quantile([r["margin"] for r in 子], 0.10))
        近平.extend([r for r in 子 if r["margin"] <= 阈])
    汇["近平局n"] = len(近平)
    汇["近平局翻转β_α_S2"] = float(np.mean([r["dS2β"] != r["dS2α"] for r in 近平])) if 近平 else float("nan")
    return 汇


def main():
    global 核
    核 = 能力核(D=64, 种子=54004)
    t0 = time.time()
    汇一 = 跑(); 汇二 = 跑()
    同 = (汇一 == 汇二)
    过召回 = (汇一["召回_S2"] >= 汇一["召回_直接"] + 阈值["召回增益"])
    过迭代 = (汇一["翻转β_α_S2"] >= 阈值["翻转全"] and 汇一["近平局翻转β_α_S2"] >= 阈值["翻转近平"])
    过P2 = 汇一["P2_翻完整_冻结_S2"] >= 阈值["P2"]
    过随机 = 汇一["随机槽_S2"] <= 阈值["随机槽"]
    过自查 = (abs(汇一["自查_重复一致率"] - 1.0) < 1e-12 and 汇一["自查_换线索改变率"] > 0.50)
    等价 = 汇一["翻转_S2_S0"] < 阈值["等价"]
    判 = dict(召回_优于直接检索="过" if 过召回 else "不过",
              迭代可见="过" if 过迭代 else "不过",
              P2="过" if 过P2 else "不过",
              随机槽="符合" if 过随机 else "不符合",
              自查="过" if 过自查 else "不过",
              尺度不变="不适用（S2 为仿射统计量；实测尺度缩放改变率=%.3f，属特性）" % 汇一["自查_尺度缩放改变率"],
              统计量等价="是" if 等价 else "否")
    if 等价 or 汇一["召回_S2"] <= 汇一["召回_直接"]:
        四 = "FAIL"
    elif 过召回 and 过迭代 and 过P2 and 过随机 and 过自查:
        四 = "PASS"
    else:
        四 = "UNDECIDABLE"
    果 = dict(meta=dict(件="C3 单臂 · S2 残差读出", 时间=time.strftime("%Y-%m-%d %H:%M:%S"),
                        并发态="R3 并发期；只作正确性判定，墙钟不宣称",
                        测试种子=测试种子, 每种子=每种子, 机制=机制, 预注册阈值=阈值,
                        墙钟_s=round(time.time() - t0, 3), 两趟逐位同=bool(同)),
              汇总=汇一, 判定明细=判, 四态判定=四)
    with open(os.path.join(HERE, "C3-S2_结果.json"), "w", encoding="utf-8") as f:
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

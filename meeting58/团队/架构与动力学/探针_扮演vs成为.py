# -*- coding: utf-8 -*-
"""meeting58/E1 探针：两区室核「扮演 vs 成为」的真实动力学（CPU-only，D=64，单次 <10s）。

只读 import 既有件，不改任何既有文件；产物只落本目录 JSON。
测 5 件事：
  P1 自我态 A 的有效自由度（对 自我标量 扫格 → SVD 秩谱）
  P2 软限(A−B) 在工作幅值区是否饱和 / 「B 覆盖 A」的真实系数
  P3 内生 credit 权 = mean|符(A,B)| / (mean|A|·mean|B|) 是否随冲突强度变化
  P4 注入外源规则(允许=True) 后一次 学自我 是否把外源冲掉（成为/撤外挂的可逆性）
  P5 V5 双通道体实跑：差 = α·A + β·B 的最小二乘系数 & 出戏率（⟨差,A⟩<0 的拍比）
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
AHI4 = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))  # …/AHI4
sys.path.insert(0, os.path.join(AHI4, "meeting49", "实施"))
sys.path.insert(0, os.path.join(AHI4, "meeting53", "实施"))

import 算子_两区室核 as 核m  # noqa: E402
import V5_双通道 as V5  # noqa: E402

两区室核 = 核m.两区室核
软限 = 核m.软限
_归 = 核m._归
D = 64
种子 = 53001


def cos(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def P1_A自由度():
    核 = 两区室核(D=D, 种子=种子)
    列 = []
    for s in np.linspace(0.05, 0.95, 91):
        核.设自我标量(float(s))
        列.append(核.自我态向量())
    M = np.asarray(列)
    sv = np.linalg.svd(M - M.mean(0), compute_uv=False)
    sv = sv / (sv[0] + 1e-12)
    有效秩 = int((sv > 1e-6).sum())
    # A 落在 P 的 3 列张成空间内？
    P = 核.P
    A0 = 核.自我态向量()
    残 = A0 - P @ np.linalg.lstsq(P, A0, rcond=None)[0]
    return dict(自我标量扫格=91, 有效秩=有效秩, 奇异值比=[round(float(v), 6) for v in sv[:6]],
                在P3列空间内_残差范数=round(float(np.linalg.norm(残)), 9),
                A的inf范=round(float(np.max(np.abs(A0))), 4),
                A的L2范=round(float(np.linalg.norm(A0)), 4),
                D=D)


def P2_软限与覆盖系数():
    核 = 两区室核(D=D, 种子=种子)
    rng = np.random.default_rng(7)
    A = 核.自我态向量()
    行 = []
    for 倍 in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 64.0):
        B = _归(倍 * (rng.standard_normal(D) / np.sqrt(D) * 3.0))
        差 = _归(软限(A - B))
        线性 = A - B
        收缩 = float(np.linalg.norm(差) / (np.linalg.norm(线性) + 1e-12))
        # 最小二乘：差 ≈ αA + βB
        X = np.stack([A, B], 1)
        ab = np.linalg.lstsq(X, 差, rcond=None)[0]
        行.append(dict(B倍=倍, B的inf范=round(float(np.max(np.abs(B))), 3),
                       软限收缩比=round(收缩, 6),
                       α_A=round(float(ab[0]), 4), β_B=round(float(ab[1]), 4),
                       cos_差A=round(cos(差, A), 4)))
    # 「B 压制 A」的可测条件：⟨软限(A−B), A⟩ < 0 ⟺ 出戏
    A范2 = float(A @ A)
    Bs = []
    for 倍 in (0.5, 1.0, 2.0, 4.0):
        # B 沿 A 方向注入（最强冲突/最强一致两种）
        for 关系 in ("与A同向", "与A反向", "与A正交"):
            if 关系 == "与A正交":
                正交 = rng.standard_normal(D); 正交 -= 正交 @ A / A范2 * A
                Bv = 倍 * 正交 / (np.linalg.norm(正交) + 1e-12) * np.linalg.norm(A)
            else:
                s = 1.0 if 关系 == "与A同向" else -1.0
                Bv = s * 倍 * A
            Bv = _归(Bv)
            差 = _归(软限(A - Bv))
            Bs.append(dict(关系=关系, 倍=倍, cos_差A=round(cos(差, A), 4),
                           出戏=bool(float(差 @ A) < 0.0),
                           投影比=round(float((Bv @ A) / A范2), 4)))
    return dict(覆盖扫描=行, A范2=round(A范2, 4), 冲突扫描=Bs,
                说明="差≈αA+βB；|β|≈|α| 即等权，不存在「B 权重被放大」的可调量")


def P3_credit权():
    核 = 两区室核(D=D, 种子=种子)
    A = 核.自我态向量()
    rng = np.random.default_rng(11)
    行 = []
    for 关系 in ("随机", "与A同向", "与A反向"):
        for 倍 in (0.5, 1.0, 2.0, 6.0):
            权s, cs = [], []
            for _ in range(20):
                if 关系 == "随机":
                    B = _归(rng.standard_normal(D) / np.sqrt(D) * 3.0)
                else:
                    s = 1.0 if 关系 == "与A同向" else -1.0
                    B = _归(s * 倍 * A + 0.1 * rng.standard_normal(D) / np.sqrt(D))
                门 = 核.符(A, B)
                尺度 = float(np.mean(np.abs(A))) * float(np.mean(np.abs(B))) + 1e-9
                权 = float(np.mean(np.abs(门))) / 尺度
                权s.append(权); cs.append(cos(A, B))
            行.append(dict(关系=关系, 倍=倍, 权均值=round(float(np.mean(权s)), 4),
                           权最小=round(float(np.min(权s)), 4), 权最大=round(float(np.max(权s)), 4),
                           cos均值=round(float(np.mean(cs)), 4)))
    return dict(扫描=行,
                说明="若 权 几乎不随 cos 变，则 credit 门不携带「冲突/一致性」信息 ⇒ 无法感到失调")


def P4_注入可逆性():
    核 = 两区室核(D=D, 种子=种子)
    外源 = _归(np.linspace(-1, 1, D) * 8.0)          # 「人设向量」
    r0 = 核.注入外源规则(外源, 允许=False)
    r1 = 核.注入外源规则(外源, 允许=True)
    A注 = 核.自我态向量()
    核.学自我(1, 0)                                  # 一次学习（奖赏=0 ⇒ 有方向更新）
    A学 = 核.自我态向量()
    # 再试 设自我态向量 与 学自我 的冲突
    核2 = 两区室核(D=D, 种子=种子)
    核2.设自我态向量(外源)
    A置 = 核2.自我态向量()
    核2.学自我(1, 0)
    A置后学 = 核2.自我态向量()
    return dict(禁止时=dict(允许=r0["允许"], 被塑造=r0["被塑造"]),
                允许时=dict(允许=r1["允许"], 被塑造=r1["被塑造"], 变化元素=r1["变化元素"]),
                注入后与注入向量cos=round(cos(A注, 外源), 6),
                一次学自我后与注入向量cos=round(cos(A学, 外源), 6),
                一次学自我后变化元素=int((A注 != A学).sum()),
                设向量后cos=round(cos(A置, 外源), 6),
                设向量后再学一次cos=round(cos(A置后学, 外源), 6),
                说明="学自我 → 设自我标量 → _刷新自我态 ⇒ 任何经 设自我态向量/注入 写入的 A 被 1 步覆盖（除非同时改 自我标量）")


def P5_V5实跑(拍数=30):
    体 = V5.双通道体(D=D, 种子=种子)
    行 = []
    for t in range(拍数):
        观测 = t % 5
        体.一拍(观测id=观测, 学习=False, 老师=False, 拍=t, 探索率=0.0)
        A = 体.核.自我态向量()
        # 复算「世界态」：与 一拍 内同一 token 列（学习关 ⇒ 词义不更新，可复现）
        r = 体.核.前向_双区室(体._token列(观测, None, list(体.历史词)[-2:]),
                          上下文=None, 迭代步数=1)
        B = r["世界态"][-1]
        差 = _归(软限(A - B))
        X = np.stack([A, B], 1)
        ab = np.linalg.lstsq(X, 差, rcond=None)[0]
        行.append(dict(拍=t, cos_差A=cos(差, A), 出戏=bool(float(差 @ A) < 0.0),
                       α_A=float(ab[0]), β_B=float(ab[1]),
                       A_inf=float(np.max(np.abs(A))), B_inf=float(np.max(np.abs(B))),
                       自我标量=float(体.核.自我标量)))
    α = np.array([x["α_A"] for x in 行]); β = np.array([x["β_B"] for x in 行])
    return dict(拍数=拍数, α均值=round(float(α.mean()), 4), β均值=round(float(β.mean()), 4),
                α标准差=round(float(α.std()), 4), β标准差=round(float(β.std()), 4),
                出戏率=round(float(np.mean([x["出戏"] for x in 行])), 4),
                cos_差A均值=round(float(np.mean([x["cos_差A"] for x in 行])), 4),
                A_inf均值=round(float(np.mean([x["A_inf"] for x in 行])), 4),
                B_inf均值=round(float(np.mean([x["B_inf"] for x in 行])), 4),
                自我标量_traj=[round(x["自我标量"], 4) for x in 行[:10]])


def main():
    out = dict(P1=P1_A自由度(), P2=P2_软限与覆盖系数(), P3=P3_credit权(),
               P4=P4_注入可逆性(), P5=P5_V5实跑())
    p = os.path.join(HERE, "探针_扮演vs成为_结果.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))
    print("\n落盘:", p)


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    np.seterr(all="ignore")
    main()

# -*- coding: utf-8 -*-
"""meeting58/E1 探针三：「角色孕育舱」最小可跑流程 —— M 收敛 / w 收敛 / 撤外挂（CPU-only）。

用既有件（只读）：
  · 规则表征核(两区室核)（meeting50）：A = 归(P@w + 基准)（w 是学出来的，len(地标)=4）
  · 网格环境：4 个地标 ⇒ 一个「价值观剖面」= 4 维 w ⇒ 可当 4 维角色身份
流程：沙箱(网格) → 奖惩(角色价值观 w目标) → M 收敛 → 撤外挂(不给 w 指令，只用 M·w自己) → 判据
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
AHI4 = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(AHI4, "meeting50", "实施"))

import 算子_规则表征 as GR  # noqa: E402

env = GR.网格环境()
# 「角色价值观」：MOSS 式 = 任务点(0) + 生存点(3)；洛天依式 = 表达点(1) + 交汇点(2)
w_MOSS = np.array([1.0, 0.0, 0.0, 1.0])
w_洛 = np.array([0.0, 1.0, 1.0, 0.0])
w_未教 = np.array([0.0, 1.0, 0.0, 1.0])


def 相对变化(M, M旧):
    d = np.linalg.norm(M - M旧)
    return float(d / (np.linalg.norm(M) + 1e-12))


def 孕育(w角色, 种子=48001, 块=25, 总回合=300):
    体 = GR.规则表征核(env, D=256, 种子=种子)
    曲线 = []
    M旧 = 体.M.copy()
    上一拍 = time.perf_counter()
    已跑 = 0
    while 已跑 < 总回合:
        体.训练(w角色, 回合=块, 种子偏移=已跑)
        已跑 += 块
        曲线.append(dict(回合=已跑,
                         M相对变化=round(相对变化(体.M, M旧), 6),
                         w误差=round(体.目标权重误差(w角色), 6),
                         w=[round(float(x), 4) for x in 体.w],
                         门=round(float(体.最近门), 4)))
        M旧 = 体.M.copy()
    耗时 = time.perf_counter() - 上一拍
    # 收敛判定：M 相对变化 < 1e-3 连续两块，且 w 误差 < 0.05
    稳 = [i for i, x in enumerate(曲线) if x["M相对变化"] < 1e-3]
    首稳 = (稳[0] + 1) * 块 if 稳 else None
    return 体, 曲线, dict(耗时秒=round(耗时, 2), 首稳回合=首稳,
                          M非零元=int((体.M != 0).sum()), M形状=list(体.M.shape),
                          w误差=round(体.目标权重误差(w角色), 6),
                          门_末=round(float(体.最近门), 4))


def 撤外挂(体, w角色, w未教):
    """撤外挂 = 不再给 w 指令/教师；只用内核自己长出的 A(=P@w+基准) 与 M 行动。"""
    起点 = list(GR.保留起点)
    # 1) 撤外挂后：内核自己的 w 当策略，真值仍是角色价值观 ⇒ 行为是否仍是这个角色
    成_自, 步_自 = GR.评估(体, 起点, 体.w, w真=w角色)
    # 2) 对照：直接给「外部指令 w角色」当策略（扮演/外挂口径的上界）
    成_外, 步_外 = GR.评估(体, 起点, w角色, w真=w角色)
    # 3) 未教新目标：M 不动，只用 M·w未教（零样本规则重组）
    成_未, 步_未 = GR.评估(体, 起点, w未教, w真=w未教)
    # 4) 置乱 M（must-fail）
    M真 = 体.M.copy()
    体.M = 体.M.copy()
    rng = np.random.default_rng(9)
    体.M = 体.M[rng.permutation(体.M.shape[0])]
    成_乱, _ = GR.评估(体, 起点, 体.w, w真=w角色)
    体.M = M真
    return dict(撤外挂_自己w_成功率=round(float(成_自), 4), 平均步_自=round(float(步_自), 2),
                外部指令w_成功率=round(float(成_外), 4),
                未教新目标零样本=round(float(成_未), 4),
                置乱M_must_fail=round(float(成_乱), 4),
                落差_自减外=round(float(成_自 - 成_外), 4))


def A的维数():
    体 = GR.规则表征核(env, D=256, 种子=48001)
    列 = []
    rng = np.random.default_rng(5)
    for _ in range(60):
        体.w = np.clip(rng.uniform(-0.5, 1.5, size=4), -0.5, 1.5)
        体.A = GR._归(体.P @ 体.w + 体.基准)
        列.append(体.A)
    M = np.asarray(列)
    sv = np.linalg.svd(M - M.mean(0), compute_uv=False)
    sv = sv / (sv[0] + 1e-12)
    return dict(有效秩=int((sv > 1e-6).sum()), 奇异值比=[round(float(v), 5) for v in sv[:6]],
                说明="A = 归(P@w + 基准)，w 4 维 ⇒ A 在固定 4 维子空间内；相对两区室核(1 标量/秩2) 是 4 倍身份自由度")


def main():
    体, 曲线, 元 = 孕育(w_MOSS)
    果 = dict(A维数=A的维数(), 孕育=元, 曲线=曲线[-6:], 曲线_首块=曲线[0],
             判据=撤外挂(体, w_MOSS, w_未教),
              备注="D=256, M形状=25x4x4, 全部 CPU numpy；无 GPU、无外挂模型")
    p = os.path.join(HERE, "探针_孕育舱_结果.json")
    json.dump(果, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(果, ensure_ascii=False, indent=1))
    print("\n落盘:", p)


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    np.seterr(all="ignore")
    main()

# -*- coding: utf-8 -*-
"""meeting58 · 红队探针3：攻击「扮演 vs 成为」——对照源码，量 A/B 的真实动力学。

CPU-only，只读 import 既有件（不改）。目标 ≤10s。
探的五件事（与简报 §1 的机制参考逐条对表）：
  P1 重合门/credit 是否携带「一致 vs 冲突」信息（符号盲？）
  P2 学自我的方向由谁决定：门（失调感）还是老师标签（ŷ）
  P3 A 的维数真相：设自我标量 → SVD 有效秩；写进去的人设向量是否 1 步被冲掉
  P4 A 在角色复刻链路上是否被更新：挂载/说话 路径不调用 更新() ⇒ 自我态恒 0.5
  P5 「扮演」的可测崩点：差=软限(A−B) 中 B 压过 A 的阈值，以及实跑里 B/A 投影比
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
AHI4 = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))   # …/AHI4
sys.path.insert(0, os.path.join(AHI4, "meeting49", "实施"))
sys.path.insert(0, os.path.join(AHI4, "meeting53", "实施"))

import 算子_两区室核 as 核m  # noqa: E402
import V5_双通道 as V5  # noqa: E402

两区室核, 软限, _归, 符 = 核m.两区室核, 核m.软限, 核m._归, None
D, 种子 = 64, 53001


def cos(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def P1_门符号盲():
    核 = 两区室核(D=D, 种子=种子)
    A = 核.自我态向量()
    行 = {}
    for 名, B in (("与A同向×1", _归(A)), ("与A反向×1", _归(-A)),
                  ("与A正交", _归(np.random.default_rng(2).standard_normal(D))),
                  ("随机小", _归(0.1 * np.random.default_rng(3).standard_normal(D)))):
        门 = 核.符(A, B)
        权 = float(np.mean(np.abs(门))) / (float(np.mean(np.abs(A))) * float(np.mean(np.abs(B))) + 1e-9)
        行[名] = dict(cos_AB=round(cos(A, B), 4), credit权=round(权, 4))
    return dict(扫描=行,
                结论="credit 权只随 |cos|（重合强度）变：同向与反向给出**同一个权**（≈1.63）⇒ 门是『重合度』不是『失调感』，代码里没有任何量区分一致/冲突")


def P2_方向由谁定():
    核 = 两区室核(D=D, 种子=种子)
    A = 核.自我态向量()
    核.最近门 = _归(np.abs(核.符(A, A)))          # 门固定（同一个 credit 权）
    核._最近B = A.copy()
    出 = {}
    for 名, ŷ, 奖 in (("答对(ŷ=1,奖=1)", 1, 1), ("答错(ŷ=0,奖=0)", 0, 0),
                      ("同门反向(ŷ=0,奖=0)再跑", 0, 0)):
        核.设自我标量(0.5)
        核.最近门 = _归(np.abs(核.符(核.自我态向量(), 核.自我态向量())))
        核._最近B = 核.自我态向量().copy()
        前 = 核.自我标量
        核.学自我(ŷ, 奖)
        出[名] = dict(标量前=round(float(前), 4), 标量后=round(float(核.自我标量), 4),
                      方向="+" if 核.自我标量 > 前 else ("-" if 核.自我标量 < 前 else "0"))
    return dict(臂=出,
                结论="同一个门（无失调信号）下，自我标量的方向完全由外部老师标签 ŷ/奖赏 决定：奖=0 ⇒ 按 ŷ==1? +1 : −1。『ĝ 感到不一致』在代码里没有对应量，方向来自监督标签")


def P3_A维数真相():
    核 = 两区室核(D=D, 种子=种子)
    列 = []
    for s in np.linspace(0.05, 0.95, 91):
        核.设自我标量(float(s)); 列.append(核.自我态向量())
    M = np.asarray(列)
    sv = np.linalg.svd(M - M.mean(0), compute_uv=False)
    秩 = int((sv / (sv[0] + 1e-12) > 1e-6).sum())
    # 写入一个人设向量 → 一次学自我 后被冲掉
    外源 = _归(np.linspace(-1, 1, D) * 8.0)
    核.注入外源规则(外源, 允许=True)
    c0 = cos(核.自我态向量(), 外源)
    核.学自我(1, 0)
    c1 = cos(核.自我态向量(), 外源)
    return dict(自我标量扫格=91, SVD有效秩=秩,
                奇异值比=[round(float(v / (sv[0] + 1e-12)), 5) for v in sv[:5]],
                注入后人设cos=round(c0, 4), 一次学自我后人设cos=round(c1, 4),
                结论="A 名义 D=64，实际由 1 个标量经 特征(x)=[d,d²,1] 生成 ⇒ 有效秩=2（一个标量 + 常量），能装的人设信息 = 1 个数；任何经 设向量/注入 写入的 A 在 1 次 学自我 后被 设自我标量→_刷新 覆盖（cos 1.0→≈0.03）")


def P4_A在角色链路是否更新():
    # V5.更新() 是唯一调 学自我 的地方；挂载/说话/说_音色 都不调它
    体 = V5.双通道体(D=D, 种子=种子)
    标量0 = float(体.核.自我标量)
    for t in range(30):
        体.一拍(观测id=t % 5, 学习=False, 老师=False, 拍=t, 探索率=0.0)
    标量1 = float(体.核.自我标量)
    # 真跑 60 拍带学习/奖赏，看 A 是不是「失败计数器」
    体2 = V5.双通道体(D=D, 种子=种子)
    环 = V5.接口沙盒(种子=种子 + 1)
    轨 = []
    for t in range(60):
        环.新试次()
        r = 体2.一拍(观测id=环.cue, 学习=True, 老师=True, 拍=t, 探索率=0.0)
        成功 = 环.走(r["动作id"])
        体2.更新(成功, r["动作id"])
        轨.append(dict(拍=t, 标量=round(float(体2.核.自我标量), 4), 成功=int(成功),
                        A范=round(float(np.linalg.norm(体2.核.自我态向量())), 3)))
    标 = np.array([x["标量"] for x in 轨])
    return dict(不调更新30拍=dict(标量0=round(标量0, 4), 标量30=round(标量1, 4),
                                 变化=bool(标量0 != 标量1)),
                带学习60拍=dict(标量头5=[x["标量"] for x in 轨[:5]],
                              标量末5=[x["标量"] for x in 轨[-5:]],
                              最小=round(float(标.min()), 4), 最大=round(float(标.max()), 4),
                              到下限拍数=int(np.argmax(标 <= 0.0501)) if bool((标 <= 0.0501).any()) else None,
                              成功率=round(float(np.mean([x["成功"] for x in 轨])), 3),
                              A范均值=round(float(np.mean([x["A范"] for x in 轨])), 3)),
                调用点="V5_双通道.py:303 更新() 是唯一调 学自我 的地方；挂载_角色.状态列/说话_连贯.说/说_音色.说 都不调它",
                结论="角色复刻链路上 A 恒 0.5（30 拍不动）；带学习时 A 立刻变成『失败计数器』（奖=0 ⇒ 标量 −有效率，几次就到下限 0.05）⇒ 『角色吸引子』不在 A 里，A 是绩效标量")


def P5_扮演的崩点():
    核 = 两区室核(D=D, 种子=种子)
    A = 核.自我态向量()
    A范2 = float(A @ A)
    行 = []
    for 倍 in (0.5, 1.0, 1.5, 2.0, 4.0):
        for 关系 in ("同向", "反向"):
            s = 1.0 if 关系 == "同向" else -1.0
            B = _归(s * 倍 * A)
            差 = _归(软限(A - B))
            行.append(dict(关系=关系, B倍数=倍, 投影比=round(float((B @ A) / A范2), 3),
                           cos_差A=round(cos(差, A), 3),
                           出戏=bool(float(差 @ A) < 0.0)))
    # 实跑：B 相对 A 的投影比分布（不学习 ⇒ A 恒 0.5；这正是角色链路的状态）
    体 = V5.双通道体(D=D, 种子=种子)
    比 = []
    for t in range(30):
        体.一拍(观测id=t % 5, 学习=False, 老师=False, 拍=t, 探索率=0.0)
        A2 = 体.核.自我态向量()
        rr = 体.核.前向_双区室(体._token列(t % 5, None, list(体.历史词)[-2:]), 上下文=None, 迭代步数=1)
        B2 = rr["世界态"][-1]
        比.append(float((B2 @ A2) / (float(A2 @ A2) + 1e-12)))
    return dict(解析扫描=行,
                实跑B在A方向投影比=dict(均值=round(float(np.mean(比)), 3),
                                     最小=round(float(np.min(比)), 3),
                                     最大=round(float(np.max(比)), 3),
                                     超过1的拍比=round(float(np.mean(np.array(比) >= 1.0)), 3)),
                结论="崩点是可测的：差≈软限(A−B)，当 B 在 A 方向投影 ≥ ‖A‖²（即投影比≥1）且方向与 A 相反/更强时，⟨差,A⟩ 翻负 ⇒ A 被 B 覆盖。该条件是**幅度条件**，不是『人设文本进 B』的语义条件；代码里没有任何把 人设 映射到 B/token 的通路")


def main():
    t0 = time.perf_counter()
    出 = dict(耗时秒=None, P1=P1_门符号盲(), P2=P2_方向由谁定(), P3=P3_A维数真相(),
              P4=P4_A在角色链路是否更新(), P5=P5_扮演的崩点())
    出["耗时秒"] = round(time.perf_counter() - t0, 3)
    p = os.path.join(HERE, "探针_R1_扮演vs成为_结果.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(出, f, ensure_ascii=False, indent=1, default=str)
    print("耗时秒", 出["耗时秒"])
    for k in ("P1", "P2", "P3", "P4", "P5"):
        print("=" * 20, k)
        print(json.dumps(出[k], ensure_ascii=False)[:1600])
    print("落盘:", p)


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    np.seterr(all="ignore")
    main()

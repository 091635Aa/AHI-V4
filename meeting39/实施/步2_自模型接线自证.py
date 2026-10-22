# -*- coding: utf-8 -*-
"""步2_自模型接线自证.py —— meeting39 步2 验收（可失败）。

判据（出处：专家05 §2 步2 门 + §3.1 D1/D9 + §3.2 A1/A2）：
  D1  保护开关：新内核 自模型开关=False ⇒ 既有输出**逐位恒等**（对照臂=基类 综合体）  → max|Δ|=0
  D9  字回归  ：本步只新增文件、基链与入口未改 ⇒ 由 probe_字回归.py 单独复跑（差值全 0）
  A1  自模型可学习（真机）：ε_self 在"熟悉动力学"<"分布外动力学"，判别 AUC≥0.7
        对照①：冻结臂（η≡0，模拟学习）AUC≈0.5；对照②：5 维摘要 AUC≈0.5（压缩即失效）
  A2  真学习 vs 模拟学习（真机）：学习臂 ε_self 末窗 < 首窗（显著下降）；冻结臂不降

纪律：纯 numpy、零梯度、确定性（固定种子）；直接走 _引擎步（自模型接线所在层），
      不 import 任何器官/眼耳手（冻结）。不碰 codec._段 语义（记忆库）。
用法：python 步2_自模型接线自证.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))   # AHI3 根（迁移：随文件定位；HERE=meeting39/实施）
for _p in (HERE, os.path.join(ROOT, "meeting31"), os.path.join(ROOT, "meeting38", "实施"), ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from 内核_自模型 import 自模型ĝ, 顶覆盖_综合体, 开关默认   # noqa: E402
from synthesis import 综合体                                # noqa: E402

SEED = 20261006

熟悉句s = ["今天的风很轻。", "我把桌子擦干净了。", "窗外有一棵树。", "水杯放在手边。",
          "我慢慢翻了一页书。", "灯一直亮着。", "他坐在对面没说话。", "雨点敲着玻璃。"]


def AUC(负分, 正分):
    """P(正分>负分)+0.5·P(相等)：正分（分布外 ε）越大越像"陌生"。"""
    a = np.asarray(正分, dtype=float)[:, None]
    b = np.asarray(负分, dtype=float)[None, :]
    if a.size == 0 or b.size == 0:
        return float("nan")
    return float(np.mean(a > b) + 0.5 * np.mean(a == b))


def 跑引擎(桥, u, n):
    for _ in range(n):
        桥._引擎步(u)


def 段ε(桥, a, b):
    return [d["ε_self"] for d in 桥.自模型.迹[a:b] if d["ε_self"] is not None]


# ────────────────────────── D1 保护开关：逐位恒等 ──────────────────────────

def 判D1():
    基 = 综合体(seed=310)
    关 = 顶覆盖_综合体(seed=310, 自模型开关=False)          # 默认关
    序列 = 熟悉句s + 熟悉句s[:3]
    for s in 序列:
        基.步(s)
        关.步(s)
    比 = dict(
        拍=(基.拍 == 关.拍),
        账本条数=(len(基.账本) == len(关.账本)),
        账本逐条=(基.账本 == 关.账本),
        自言条数=(len(基.自言.流) == len(关.自言.流)),
        自言文本逐条=([e["文本"] for e in 基.自言.流] == [e["文本"] for e in 关.自言.流]),
        e序列逐位=(list(基.eng.e序列) == list(关.eng.e序列)),
        记忆段数=(len(基.codec._段) == len(关.codec._段)),
        a_prev逐位=bool(np.array_equal(基.a_prev, 关.a_prev)),
        i_prev=(基.i_prev == 关.i_prev),
        prev_c=(基.prev_c == 关.prev_c),
        关时自模型为空=(关.自模型 is None),
    )
    通过 = all(比.values())
    return dict(逐项=比, 通过=通过, 默认开关=bool(开关默认))


# ────────────────────────── A1/A2：真机自模型可学习性 ──────────────────────────

def 跑臂(学习=True, 状态投影=None, 种子=311, n训=120, n评=40):
    臂 = 顶覆盖_综合体(seed=种子, 自模型开关=True, 学习=学习, 状态投影=状态投影)
    u维 = int(臂.探针.shape[0])                       # 感知向量输出维（16），引擎输入维
    熟悉u = [臂.感知向量(臂.codec.编码(s)) for s in (熟悉句s * 20)[:n训]]
    rng = np.random.default_rng(SEED + 7)
    陌u = [rng.standard_normal(u维).astype(np.float32) for _ in range(n评)]   # 分布外感觉（16 维）
    # Phase T：熟悉训练
    i0 = len(臂.自模型.迹)
    for u in 熟悉u:
        臂._引擎步(u)
    i1 = len(臂.自模型.迹)
    # Phase E熟：留出熟悉（同域·已见句）
    for s in (熟悉句s * 20)[:n评]:
        臂._引擎步(臂.感知向量(臂.codec.编码(s)))
    i2 = len(臂.自模型.迹)
    # Phase E外：分布外（随机向量）
    for u in 陌u:
        臂._引擎步(u)
    i3 = len(臂.自模型.迹)
    return dict(训=段ε(臂, i0, i1), 熟=段ε(臂, i1, i2), 外=段ε(臂, i2, i3),
                ε全=臂.自模型.ε序列(), 摘要=臂.自模型.摘要(),
                D=int(臂.自模型.摘要()["D"]), 桥=臂)


def 判A1A2():
    学 = 跑臂(学习=True)
    冻 = 跑臂(学习=False)
    D = 学["D"]
    P5 = np.random.default_rng(SEED + 11).standard_normal((5, D)) / np.sqrt(D)
    五 = 跑臂(学习=True, 状态投影=P5)

    def 首末(ε, k=20):
        if len(ε) < 2 * k:
            return (float(np.mean(ε[:max(1, len(ε) // 4)])),
                    float(np.mean(ε[-max(1, len(ε) // 4):])))
        return float(np.mean(ε[:k])), float(np.mean(ε[-k:]))

    学首, 学末 = 首末(学["训"])
    冻首, 冻末 = 首末(冻["训"])
    auc学 = AUC(学["熟"], 学["外"])
    auc冻 = AUC(冻["熟"], 冻["外"])
    auc五 = AUC(五["熟"], 五["外"])

    # A2：学习臂显著下降（末<首），冻结臂不降
    a2 = (学末 < 学首 - 1e-6) and (abs(冻末 - 冻首) < 1e-6)
    # A1：学习臂 AUC≥0.7 且 冻结/五维 ≈0.5（|·−0.5|≤0.12）
    a1 = (auc学 >= 0.7) and (abs(auc冻 - 0.5) <= 0.12) and (abs(auc五 - 0.5) <= 0.12)

    报 = dict(
        学习臂=dict(训首ε=round(学首, 4), 训末ε=round(学末, 4),
                   熟ε均值=round(float(np.mean(学["熟"])), 4),
                   外ε均值=round(float(np.mean(学["外"])), 4), AUC=round(auc学, 4),
                   更新次数=学["摘要"]["更新次数"], D=学["摘要"]["D"]),
        冻结臂=dict(训首ε=round(冻首, 4), 训末ε=round(冻末, 4),
                   熟ε均值=round(float(np.mean(冻["熟"])), 4),
                   外ε均值=round(float(np.mean(冻["外"])), 4), AUC=round(auc冻, 4),
                   更新次数=冻["摘要"]["更新次数"]),
        五维臂=dict(熟ε均值=round(float(np.mean(五["熟"])), 4),
                   外ε均值=round(float(np.mean(五["外"])), 4), AUC=round(auc五, 4)),
    )
    return dict(报=报, A1=bool(a1), A2=bool(a2), 状态维=D)


# ────────────────────────── 汇总 ──────────────────────────

def 主():
    d1 = 判D1()
    a = 判A1A2()
    a1, a2 = a["A1"], a["A2"]
    报 = dict(种子=SEED, 开关默认=bool(开关默认),
              D1=d1, A组=dict(**a["报"], A1=a1, A2=a2, 状态维=a["状态维"]),
              门=dict(D1=d1["通过"], A1=a1, A2=a2))
    报["总判"] = bool(d1["通过"] and a1 and a2)

    print("=" * 74)
    print("步2 · 自模型 ĝ 接线自证（默认关 · 只读 · 不改决策）")
    print("=" * 74)
    print("【D1 保护开关】关时逐位恒等（对照=基类 综合体）")
    for k, v in d1["逐项"].items():
        print("   %-14s %s" % (k, v))
    print("   → %s" % ("PASS" if d1["通过"] else "FAIL"))
    print("-" * 74)
    print("【A 组·真机】状态维=%d" % a["状态维"])
    for k in ("学习臂", "冻结臂", "五维臂"):
        print("   %-6s %s" % (k, a["报"][k]))
    print("   → A1（AUC≥0.7 且 冻/五≈0.5）：%s ｜ A2（学降/冻不降）：%s"
          % ("PASS" if a1 else "FAIL", "PASS" if a2 else "FAIL"))
    print("=" * 74)
    print("总判：%s" % ("PASS" if 报["总判"] else "FAIL"))
    print("（D9 字回归由 probe_字回归.py 单独复跑；本步未改既有 .py，故应差值全 0）")

    出 = os.path.join(HERE, "步2_结果.json")
    with open(出, "w", encoding="utf-8") as f:
        json.dump(报, f, ensure_ascii=False, indent=1)
    print("[已存] %s" % 出)
    return 报


if __name__ == "__main__":
    主()
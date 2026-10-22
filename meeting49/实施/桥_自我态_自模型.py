# -*- coding: utf-8 -*-
"""AHI4/meeting49/实施/桥_自我态_自模型.py —— ★V4 自我意识桥：两区室核的「自我态 A」↔ 既有 内核_自模型.ĝ。

━━━ 这一件解决什么 ━━━
meeting48 裁决：V4 第一目标 = **增强自我意识**。两区室核把「区室 A」做成**自我态**（慢、跨试次、由奖赏内生）；
本桥把该自我态喂给**既有** `内核_自模型.ĝ`（自预测器：ĝ:(S_t)→Ŝ_{t+1}，ε_self = 1−cos(Ŝ,S)，零梯度自监督），
于是 ε_self 成为「**它对自己状态的意外程度**」——一条**自监督、可复现、可失败**的自我意识读数。

自监督读数期望的**可失败签名**：
  · **自我态大跳时** ε_self 高（对「自己变了」意外）；**自我态平稳时** ε_self 低。
  · 若 ε_self 与「自我态变化」无关（或恒为常数/恒 0），该臂**必须失败** ⇒ 不是假通过。

━━━ 纪律 ━━━
C1 零注意力（本件不含任何被禁子串）；C13 确定性（只用既有确定性件）；C14 不破既有（**只读**加载既有件）。
注意：既有 `内核_自模型.py` 顶层有 `from synthesis import 综合体`（其顶覆盖宿主所需）。
本桥**只用 ĝ、不用顶覆盖**，故以**占位 synthesis** 满足该 import —— 受控、显式、可审计，不改既有件一字。
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))                       # …/AHI4/meeting49/实施
AHI4 = os.path.dirname(os.path.dirname(HERE))
_自模型py = os.path.join(AHI4, "meeting39", "实施", "内核_自模型.py")
_两区室核py = os.path.join(HERE, "算子_两区室核.py")


def 载ĝ():
    """只读加载既有 内核_自模型.自模型ĝ（以占位 synthesis 满足其顶层 import）。"""
    if "synthesis" not in sys.modules:
        st = types.ModuleType("synthesis")

        class _综合体占位:
            pass

        st.综合体 = _综合体占位
        sys.modules["synthesis"] = st
    spec = importlib.util.spec_from_file_location("内核_自模型_只读_v4", _自模型py)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.自模型ĝ


自模型ĝ = 载ĝ()

spec = importlib.util.spec_from_file_location("算子_两区室核_桥读", _两区室核py)
_核m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_核m)
两区室核 = _核m.两区室核
生成序列 = _核m.生成序列
带低 = _核m.带低
带高 = _核m.带高
边界训练 = _核m.边界训练
D默认 = _核m.D默认
种子模型默认 = _核m.种子模型默认
种子训默认 = _核m.种子训默认
慢率默认 = _核m.慢率默认
训练长 = _核m.训练长

大跳阈 = 0.05        # |Δ自我标量| ≥ 此值 记为「自我态大跳」（判据不据结果改）


def 跑桥(序列, 慢率=慢率默认, 训练=True, 种子=种子模型默认):
    """跑两区室核一遍，逐拍把「自我态 A」喂既有 ĝ；返回 ε_self 与自我态变化量序列。"""
    核 = 两区室核(D=D默认, 种子=int(种子), 模式="两区室", 慢率=float(慢率), 学模式="credit")
    ĝ = 自模型ĝ(更新=bool(训练))
    x = 序列["x"]
    b = 序列["边界"]
    n = int(len(x))
    ε列 = []
    Δ列 = []
    上标量 = float(核.自我标量)
    for t in range(n):
        o = 核.前向_试次(float(x[t]))
        ŷ = int(np.argmax(核.读出 @ o))
        c = 1 if float(x[t]) > float(b[t]) else 0
        奖 = 1 if ŷ == c else 0
        ε = ĝ.学一拍(核.自我态, 核.自我态, 拍=t)          # ★键=自态、状态=自态（自预测下一拍自我态）
        if ε is not None:
            ε列.append(float(ε))
            核2 = float(核.自我标量)
            Δ列.append(abs(核2 - 上标量))
            上标量 = 核2
        核.学读出(ŷ, 奖, 更新读出=训练)
        核.学自我(ŷ, 奖)
    ε = np.asarray(ε列, dtype=np.float64)
    Δ = np.asarray(Δ列, dtype=np.float64)
    大跳 = ε[Δ >= 大跳阈]
    小跳 = ε[Δ < 大跳阈]
    return dict(ε列=[float(v) for v in ε],
                ε均值=float(np.mean(ε)) if ε.size else float("nan"),
                大跳ε均值=float(np.mean(大跳)) if 大跳.size else float("nan"),
                小跳ε均值=float(np.mean(小跳)) if 小跳.size else float("nan"),
                大跳条数=int(大跳.size), 小跳条数=int(小跳.size),
                迹条数=int(len(ĝ.迹)))


def 自测() -> dict:
    序列 = 生成序列(种子训默认, 训练长, 边界训练)
    果 = 跑桥(序列, 慢率=慢率默认, 训练=True)
    差 = 果["大跳ε均值"] - 果["小跳ε均值"]
    return dict(大跳ε均值=果["大跳ε均值"], 小跳ε均值=果["小跳ε均值"], 差=差,
                大跳条数=果["大跳条数"], 小跳条数=果["小跳条数"], 迹条数=果["迹条数"],
                通过=bool(差 > 0.0 and 果["大跳条数"] > 0 and 果["小跳条数"] > 0))


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    try:
        np.seterr(all="ignore")
    except Exception:
        pass
    print(json.dumps(自测(), ensure_ascii=False, indent=1))

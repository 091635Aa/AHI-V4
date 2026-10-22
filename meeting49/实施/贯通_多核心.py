# -*- coding: utf-8 -*-
"""AHI4/meeting49/实施/贯通_多核心.py —— ★V4「多核心贯通」：把 两区室核 与 六内核宿主 接成**一个可运行系统**。

━━━ 这一件解决什么 ━━━
V4 之前，`两区室核`（能力核）+ `内核宿主`（六内核：自模型ĝ/思核T/真学习/自我回路/顿悟/画面）
各自判据都能过，但**没接在一起跑**（"玩具级"）。本件做**真贯通**：

  · 引擎侧：`内核宿主.造桥()` 造**真 synthesis**链（非占位）→ 六内核全在链上；
  · 认知核：`两区室核`（继承能力核 + 第八算子「符」+ 自我态 A）当**认知核**；
  · 每拍：
      ① `桥.步_向量(输入向量)` 驱引擎（真路径，六内核在链上，ĝ 自监督喂自态）；
      ② `核.前向_双区室([输入向量])` 走**自我-世界整合**（世界态 B = 既有主循环；自我态 A = 慢态）；
      ③ 自我态 A ← `软限(A + 输出)`（**递归自更新**：自我由「自我×世界」整合缓慢长成）—— 有界、定点、确定；
      ④ provenance：`设来源(桥, 自源=True)` ⇒ **它自己的经历可塑「我」**（外源默认不塑造，见 注入外源规则）。

━━━ 纪律 ━━━
C1 零注意力；C13 确定性（随机只来自 default_rng(显式种子)；输入流逐拍由种子定）；C14 不破既有（只读加载既有件）。
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))                    # …/AHI4/meeting49/实施
AHI4 = os.path.dirname(os.path.dirname(HERE))
_宿主py目录 = os.path.join(AHI4, "meeting38", "实施")
_核py = os.path.join(HERE, "算子_两区室核.py")
D默认 = 1024
种子默认 = 310
输入种子基 = 7000


def 载内核宿主():
    """只读加载既有 内核宿主（真 synthesis 链；六内核全开）。"""
    if _宿主py目录 not in sys.path:
        sys.path.insert(0, _宿主py目录)
    spec = importlib.util.spec_from_file_location("内核宿主_只读_v4", os.path.join(_宿主py目录, "内核宿主.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_核spec = importlib.util.spec_from_file_location("算子_两区室核_贯通读", _核py)
_核m = importlib.util.module_from_spec(_核spec)
_核spec.loader.exec_module(_核m)
两区室核 = _核m.两区室核
_归 = _核m._归
软限 = _核m.软限
算子_符 = _核m.算子_符


def 输入向量(t: int, D: int = D默认) -> np.ndarray:
    """确定性的「感觉」输入流：逐拍由种子定的双极向量（±1）。"""
    rng = np.random.default_rng(输入种子基 + int(t))
    return np.where(rng.integers(0, 2, size=int(D)) == 1, 1.0, -1.0)


def 造双区室(seed: int = 种子默认, D: int = None, 慢率: float = 0.20):
    """造一个**贯通系统**：六内核宿主桥 + 两区室核（认知核）。"""
    宿主 = 载内核宿主()
    桥 = 宿主.造桥(seed=int(seed))
    d = int(D) if D is not None else _引擎维(桥)
    核 = 两区室核(D=d, 种子=int(seed), 模式="两区室", 慢率=float(慢率), 学模式="credit")
    return dict(桥=桥, 核=核, 宿主=宿主, 拍=0, D=d)


def _引擎维(桥) -> int:
    """取引擎母语空间维（探针.shape[1]）；取不到回退 D默认。"""
    try:
        return int(桥.探针.shape[1])
    except Exception:
        return int(D默认)


def 一拍(双: dict, u: np.ndarray = None) -> dict:
    """一拍贯通：引擎自主一步 + 双区室自我-世界整合 + 递归自更新自我态。"""
    桥, 核, 宿主 = 双["桥"], 双["核"], 双["宿主"]
    if u is None:
        u = 输入向量(双["拍"], 双["D"])
    u = np.asarray(u, dtype=np.float64).ravel()
    # ① 引擎真路径（六内核在链上）；自源经历可塑「我」
    宿主.设来源(桥, True)
    桥.步_向量(u.astype(np.float32), 原文=None)
    # ② 双区室整合（世界态 B = 既有主循环；自我态 A = 慢态）
    R = 核.前向_双区室(u.reshape(1, -1), None, 1)
    # ③ 递归自更新：自我态 ← 软限(自我态 + 输出)（有界、定点、确定）
    核.设自我态向量(_归(软限(核.自我态 + R["输出"][0])))
    双["拍"] += 1
    return dict(拍=双["拍"], 门均值=float(np.mean(np.abs(R["门"]))),
                输出范数=float(np.max(np.abs(R["输出"]))),
                自我态范数=float(np.max(np.abs(核.自我态))),
                未定义令牌=bool(R["未定义令牌"]))


def 跑(双: dict, 拍数: int = 120) -> dict:
    """跑 N 拍贯通；返回逐步读数与最终内核态。异常**如实上报**（不吞）。"""
    迹 = []
    错 = None
    for _ in range(max(1, int(拍数))):
        try:
            迹.append(一拍(双))
        except Exception as ex:                                   # noqa: BLE001
            错 = repr(ex)
            break
    核 = 双["核"]
    return dict(已跑=int(len(迹)), 异常=错, 迹=迹,
                内核态=双["宿主"].内核态(双["桥"]),
                自我态有限=bool(np.isfinite(核.自我态).all()),
                自我态界=float(np.max(np.abs(核.自我态))))


def 自测(拍数: int = 120, seed: int = 种子默认) -> dict:
    双 = 造双区室(seed=seed)
    果 = 跑(双, 拍数)
    宿 = 果["内核态"]
    六层 = [k for k in 宿.keys() if k not in ("挂载",)]
    全挂 = bool(宿.get("挂载") and all(宿.get(k) is not None for k in ("自模型", "思核", "真学习", "自我回路", "顿悟涌现", "画面")))
    通行 = bool(果["异常"] is None and 全挂 and 果["自我态有限"] and 果["自我态界"] <= 16.0)
    return dict(拍数=int(拍数), 已跑=果["已跑"], 异常=果["异常"],
                内核层=六层, 六内核全挂=全挂,
                自我态有限=果["自我态有限"], 自我态界=round(果["自我态界"], 6),
                自模型=宿.get("自模型"), 自我回路=宿.get("自我回路"), 顿悟涌现=宿.get("顿悟涌现"),
                末拍=果["迹"][-1] if 果["迹"] else None,
                通过=通行)


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    print(json.dumps(自测(), ensure_ascii=False, indent=1, default=str))

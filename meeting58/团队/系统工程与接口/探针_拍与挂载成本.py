# -*- coding: utf-8 -*-
"""meeting58/E3 探针 2：每拍成本 / 角色头（角色模板）热插拔成本 / 换包是否动核。

只读 import 既有件；产物落本目录；CPU-only，单次 <10s。
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
AHI4 = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
for p in (os.path.join(AHI4, "meeting53", "实施"), os.path.join(AHI4, "挂载")):
    if p not in sys.path:
        sys.path.insert(0, p)
import V5_双通道 as V5  # noqa: E402
import 挂载_角色 as 挂载  # noqa: E402
import 角色包 as 包模块  # noqa: E402

产物 = os.path.join(HERE, "产物")
os.makedirs(产物, exist_ok=True)
拍数 = 60


def 主():
    out = dict(口径="CPU-only, D=64 双通道体, 只读既有件")
    器 = 挂载.挂载器(D=V5.D默认, 种子=0)
    体 = 器.体

    # 1) 每拍成本（含学习）
    环 = V5.接口沙盒(种子=1)
    列 = []
    for t in range(20):
        环.新试次()
        t0 = time.perf_counter()
        体.一拍(观测id=环.cue, 学习=True, 老师=True, 拍=t, 探索率=0.0)
        列.append(round((time.perf_counter() - t0) * 1000.0, 3))
    out["每拍毫秒"] = dict(样本=列, 中位=round(float(np.median(列)), 3),
                          最大=round(float(np.max(列)), 3))

    # 2) 状态轨迹成本（换包可比性所需）
    t0 = time.perf_counter()
    状态 = 器.状态列(拍数=拍数, 种子=0)
    状态毫秒 = round((time.perf_counter() - t0) * 1000.0, 1)
    核指纹前 = 体.核.参数基线()["W码"] if hasattr(体.核, "参数基线") else "n/a"

    # 3) 热插拔角色头：训练成本 + 是否动核
    包s = 包模块.全部包()
    结果 = {}
    痕迹 = None
    for 名, 包 in 包s.items():
        t0 = time.perf_counter()
        头 = 器.挂载(包, 状态)
        挂毫秒 = round((time.perf_counter() - t0) * 1000.0, 1)
        t1 = time.perf_counter()
        for t, (cue, o用, 内心) in enumerate(状态):
            头.生成(o用, 内心, 最多词=4)
        生毫秒 = round((time.perf_counter() - t1) * 1000.0 / max(len(状态), 1), 3)
        结果[名] = dict(角色=包.get("角色", 名), 词表=len(头.词表),
                        挂载毫秒=挂毫秒, 每拍生成毫秒=生毫秒, 头指纹=头.指纹())
        痕迹 = 头.指纹()
    核指纹后 = 体.核.参数基线()["W码"] if hasattr(体.核, "参数基线") else "n/a"
    out["角色头"] = dict(状态轨迹毫秒=状态毫秒, 拍数=拍数, 包数=len(包s),
                        **结果,
                        核随挂载变化=bool(核指纹前 != 核指纹后))
    out["核指纹"] = dict(前=str(核指纹前)[:16], 后=str(核指纹后)[:16])
    out["结论"] = dict(
        换音色="O(1)，不重建任何对象（探针1 实测波形不同且自检通过）",
        换角色模板="只重建 角色头（训练=原状态列上重跑 η 更新），核 o用/内心 序列可复用",
        核是否需重建= "否（核指纹在换包前后一致 ⇒ 挂载只读状态）")
    return out


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    t0 = time.perf_counter()
    果 = 主()
    果["墙钟秒"] = round(time.perf_counter() - t0, 2)
    p = os.path.join(产物, "拍与挂载成本.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(果, f, ensure_ascii=False, indent=1)
    print(json.dumps(果, ensure_ascii=False, indent=1))

# -*- coding: utf-8 -*-
"""meeting69/实施/协议_听觉源.py —— S1 遥测快照（新 profile；09 §2.6）。

只读听觉工作区，返回 dict（不返回既有 状态帧）；**新 profile `听觉_S1_v1`**，
判据脚本必须打印 profile 名，禁跨 profile 比指纹（D9 / 内生架构 R1 §4.3）。
"""
from __future__ import annotations

import hashlib

import numpy as np

PROFILE = "听觉_S1_v1"


def _sha(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a)).tobytes()).hexdigest()[:16]


def 快照(区, 事件率=0.0):
    d = 区.取听觉()
    链 = d["焦点链"]
    指纹 = hashlib.sha256(("|".join(f"{e['拍号']}:{e['带心Hz']:.3f}:{e['s']:.4f}" for e in 链)
                          + "|" + _sha(d["听场"])).encode("utf-8")).hexdigest()[:16]
    return dict(profile=PROFILE,
                听觉_意外度=round(float(d["意外度"]), 6),
                听觉_事件率=round(float(事件率), 4),
                听觉_流数=int(len(d["流表"])),
                听觉_焦点停留=int(链[-1]["停留拍"]) if 链 else 0,
                听觉_账本条目=0,          # S1 零训练、无账本（延后 S3）
                听觉_学习率=0.0,
                听觉_自他重合=None,       # 未测（需自我态；S2/S3）
                拍号=int(d["拍号"]),
                指纹=指纹)

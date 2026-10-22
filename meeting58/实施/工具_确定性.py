# -*- coding: utf-8 -*-
"""meeting58/实施/工具_确定性.py —— 确定性工具（纯函数，无全局可变态）。

只被 meeting58 新件 import；不 import 任何外部模型。
"""
from __future__ import annotations

import hashlib

import numpy as np


def 指纹(*部件, 长: int = 16) -> str:
    h = hashlib.sha256()
    for p in 部件:
        h.update(repr(p).encode("utf-8"))
        h.update(b"|")
    return h.hexdigest()[:长]


def 确定性向量(键: str, 维: int) -> np.ndarray:
    """由 sha256 流生成 ±1 双极向量（与 seed 无关的显式确定性）。"""
    维 = int(维)
    need = (维 + 7) // 8
    buf = b""
    盐 = str(键).encode("utf-8")
    while len(buf) < need:
        buf += hashlib.sha256(盐 + b"|" + bytes([len(buf) & 0xFF])).digest()
    bits = np.frombuffer(buf[:need], dtype=np.uint8)
    return (((bits[:, None] >> np.arange(8)[None, :]) & 1).reshape(-1)[:维] * 2 - 1).astype(np.float64)


_投影缓存: dict = {}


def 确定性投影(源: np.ndarray, 目标维: int, 键: str = "状态投影") -> np.ndarray:
    """D核 → D声 的确定性投影：每行独立 sha256 双极权重，输出归一化。

    只依赖 (源, 目标维, 键)，两趟逐位同；缓存只读派生矩阵（不改结果）。
    """
    v = np.asarray(源, dtype=np.float64).ravel()
    D = int(目标维)
    ck = (v.size, D, str(键))
    M = _投影缓存.get(ck)
    if M is None:
        基 = 确定性向量("投影|%s|%d|%d" % (键, v.size, D), v.size * D)
        M = 基.reshape(D, v.size) / np.sqrt(max(v.size, 1))
        _投影缓存[ck] = M
    return M @ v


def 归一(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v)) + 1e-12
    return v / n


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    a = np.arange(8, dtype=float)
    p1 = 确定性投影(a, 1024)
    p2 = 确定性投影(a, 1024)
    print("投影维度", p1.shape, "两趟逐位同", np.array_equal(p1, p2), "指纹", 指纹(p1.round(6)))

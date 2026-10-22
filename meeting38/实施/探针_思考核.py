# -*- coding: utf-8 -*-
"""meeting38/实施/探针_思考核.py —— V3 模块一：思考核**只读探针**（纯新增）。

职责（所有者 2026-10-06）：
  「思考可通过探针观测（日志/内部降维投影），但**不进输出流**。」

本模块是**只读**观测器：
  · 读思考槽（含「内」层深度回响，脑写）与自我槽的情感态向量；
  · 把思考核的激活模式用**固定随机投影**（确定性、无训练、无梯度、非 Transformer）降维到
    k 维，写成 NDJSON 日志（每行一条：t/拍/类/摘要/投影/模）。
  · **绝不**调用 态.排队说 / 态.写它说 / 态.写它想 —— 故对输出流零污染。

红线：C1（无 attention/softmax/梯度）｜C13（投影矩阵由 seed 确定性生成，同输入同输出）。
"""
from __future__ import annotations

import json
import os

import numpy as np


class 思考核探针:
    """只读：把思考核激活降维投影到日志。不写任何输出槽。"""

    def __init__(self, 路径: str, 维: int = 4, 种子: int = 3800, D: int = 1024):
        self.路径 = str(路径)
        self.k = int(维)
        self.种子 = int(种子)
        self.D = int(D)
        # 固定随机投影 P（k×D，±1）——确定性、无训练（ródé 投影的第一行思路，纯 numpy）
        rng = np.random.default_rng(int(种子))
        self.P = np.sign(rng.standard_normal((self.k, self.D))).astype(np.float32)
        self.计数 = 0
        d = os.path.dirname(os.path.abspath(self.路径))
        if d:
            os.makedirs(d, exist_ok=True)

    # ── 降维投影（只读计算；越界安全）──
    def 投影(self, 激活向量) -> list:
        try:
            v = np.asarray(激活向量, dtype=np.float32).reshape(-1)
        except Exception:
            return [0.0] * self.k
        if v.shape[0] != self.D:
            # 维度不匹配（扩容后）→ 重采样投影矩阵行数对齐：确定性、按当前 D 重建
            rng = np.random.default_rng((self.种子, int(v.shape[0])))
            P = np.sign(rng.standard_normal((self.k, int(v.shape[0])))).astype(np.float32)
        else:
            P = self.P
        y = P @ v / float(v.shape[0])
        return [round(float(x), 4) for x in y]

    # ── 写一条观测（NDJSON；只写文件，不碰任何槽）──
    def 记(self, 拍: int, 类: str, 文: str, 激活向量=None, 编码器=None) -> dict:
        激活 = 激活向量
        if 激活 is None and 编码器 is not None and 文:
            try:
                激活 = 编码器.文本码(str(文)[:64])
            except Exception:
                激活 = None
        if 激活 is None:
            激活 = _文本伪向量(str(文), self.D)
        投 = self.投影(激活)
        行 = dict(t=round(_秒(), 3), 拍=int(拍 or 0), 类=str(类),
                  摘要=str(文)[:60], 投影=投,
                  模=round(float(np.linalg.norm(np.asarray(激活, np.float32).reshape(-1))), 3))
        try:
            with open(self.路径, "a", encoding="utf-8") as f:
                f.write(json.dumps(行, ensure_ascii=False) + "\n")
        except Exception:
            pass
        self.计数 += 1
        return 行

    # ── 便捷：扫思考槽 + 自我槽（只读），批量写 ──
    def 扫(self, 态, 上限: int = 40) -> list:
        """读 态.取思考(只要外=False)（含内层）+ 态.取自我() 的情感向量；逐条投影落日志。
        绝无写操作。返回本次写入的行。"""
        写 = []
        编码 = getattr(态, "编码", None)
        E = None
        try:
            E = (态.取自我() or {}).get("向量")
        except Exception:
            E = None
        try:
            思考s = 态.取思考(上限=上限, 只要外=False)
        except Exception:
            思考s = []
        for x in 思考s:
            写.append(self.记(拍=0, 类=x.get("类", "?"), 文=x.get("文", ""),
                             激活向量=E, 编码器=编码))
        return 写


def _文本伪向量(s: str, D: int) -> np.ndarray:
    """无编码器时的确定性伪向量（sha256 展开，±1）。仅用于探针可视化，不参与任何决策。"""
    import hashlib
    h = hashlib.sha256(s.encode("utf-8")).digest()
    need = D
    out = np.empty(D, np.float32)
    i = 0
    base = h
    while i < D:
        blk = hashlib.sha256(base).digest()
        for b in blk:
            if i >= D:
                break
            out[i] = 1.0 if (b & 1) else -1.0
            i += 1
        base = blk
    return out


def _秒() -> float:
    import time
    return time.time()
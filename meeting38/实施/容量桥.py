# -*- coding: utf-8 -*-
"""meeting38/实施/容量桥.py —— 步 3b：把步 0-2b 的容量内核挂进自我线程（纯新增桥模块）。

职责（09 规格 §2.5）：
- 自言/事件条目 → FSW 长时库记录（知识/经历可检索化；文本元数据留在桥内，HDC 只存地址）
- 综合体遥测 → 情感因子层 V/A/P 粗值 → 情感态束（步 4/5 再接情绪槽语义，现为占位映射）
- 检索 API：自我线程复盘/脑候选评估用（返回库记录元数据）

★语义检索真通路（AHI3 修复，2026-10-07；审计 P0-1/「不理解语义」根因）：
  原先 `查经历` 直接调 `核.检索内容`，而 FSW 记录束 = 随机原子绑定的束（类/拍/源/纹），
  与查询文本码**几乎正交** → 检索落在噪声底。实测（`共存态.编码`，1024 维）：
      「json 同域」对旧知 cos = 0.45~0.79（文本码直比），而走 FSW 只有 0.031~0.047；
      「晚饭/天气」等**无关**文本走 FSW 反而拿到 0.035/−0.018 —— **语义完全丢失**。
  根因不是编码器（文本码本身语义良好），而是**桥没存语义向量**。
  现修：`记经历(..., 向量=…)` 存**语义向量**；`查经历` 优先走 `查经历语义()`（余弦，真语义），
  并**保留** FSW 原通路为 `查经历FSW()`（对照/兼容，行为逐位不变）。
  这是把已有能力接通，**不是**新增外部模型（C1/C10 守住）。

设计律：确定性（原子/记录全 sha256+种子）；live 束只做粗读广播（07 §11 裁决）；
零引擎改动（本桥被 器官_我.py 调用，不 import 引擎文件）。
"""
from __future__ import annotations

import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from fsw_kernel import FSW内核
from 情感因子层 import 情感因子层


class 容量桥:
    def __init__(self, 种子: int = 3800, 编码器=None):
        self.核 = FSW内核(种子=种子)
        self.层 = 情感因子层(核=self.核, 种子=种子)
        self.编码器 = 编码器                       # ★可选：给了它，记经历可自编码语义向量
        self._文本: dict = {}                      # rid → 文本元数据（HDC 外，可审计）
        self._向量: dict = {}                      # rid → 语义向量（★真语义检索通路的原料）
        self._下一条 = 0
        self.统计 = dict(记经历=0, 查询=0, 语义查询=0, FSW查询=0, 自编码=0)

    # ── 经历入库（自言/事件条目）──
    def 记经历(self, 拍: int, 类型: str, 文本: str, 来源: str = "自我线程",
               向量=None) -> int:
        """条目 → 8 字段内记录（类型/拍号/来源/文本指纹），文本全文留桥内元数据。

        ★`向量`：可选，该条目的**语义向量**（如 `共存态.编码.文本码(文本)`）。
          给了它，这条经历就能被 `查经历()` 语义检索到；不给则只有 FSW 地址通路。
        """
        摘 = "".join(ch for ch in str(文本) if ch.strip())[:24]
        字段 = [("类", 类型),
                ("拍", "拍%06d" % int(拍)),
                ("源", 来源),
                # C13 确定性：禁用内建 hash()（每进程加盐，跨进程不可复现）→ sha256
                ("纹", hashlib.sha256(摘.encode("utf-8")).hexdigest()[:8])]
        rid = self.核.写记录(字段, 句柄=self._下一条)
        self._文本[rid] = dict(拍=拍, 类型=类型, 文本=str(文本), 来源=来源)
        v = 向量
        if v is None and self.编码器 is not None:   # ★自编码：有编码器就自动补语义向量
            try:
                v = self.编码器.文本码(str(文本)[:512])
                self.统计["自编码"] += 1
            except Exception:
                v = None
        if v is not None:                           # ★语义向量入库（真语义检索的原料）
            try:
                import numpy as _np
                self._向量[rid] = _np.asarray(v, dtype=_np.float32).ravel()
            except Exception:
                pass
        self._下一条 += 1
        self.统计["记经历"] += 1
        return rid

    # ── 读①：语义检索（★真通路；余弦，确定性，纯 numpy）──
    def 查经历语义(self, 查询向量, 相似下限: float = 0.0) -> dict:
        """查询向量 → 与库内**语义向量**做余弦，取最大者（tie 取 rid 升序）。

        这是「理解语义」的真通路：编码器本来就把语义编进了 1024 维文本码
        （实测同域 0.45~0.79 / 无关 0.06）。`相似下限` 之下视为未命中（返回 None）。
        """
        import numpy as np
        q = np.asarray(查询向量, dtype=np.float32).ravel()
        nq = float(np.linalg.norm(q))
        if nq <= 0.0 or not self._向量:
            return None
        ids = sorted(self._向量)
        码本 = np.stack([self._向量[i] for i in ids], 0)
        范 = np.linalg.norm(码本, axis=1)
        分 = (码本 @ q) / np.maximum(范 * nq, 1e-9)
        序 = int(np.argmax(分))                    # 平票取首个 = rid 升序（确定性）
        s = float(分[序])
        if s < float(相似下限):
            return None
        元 = dict(self._文本.get(ids[序], {}))
        元["相似"] = s
        元["通路"] = "语义"
        return 元

    # ── 读②：FSW 地址/内容检索（原通路，逐位不变；作对照与大规模候选）──
    def 查经历FSW(self, 查询束) -> dict:
        r = self.核.检索内容(查询束, k=8)
        if r.get("top1") is None:
            return None
        元 = dict(self._文本.get(r["top1"], {}))
        元["相似"] = r.get("细相似")
        元["通路"] = "FSW"
        return 元

    def 查经历(self, 查询, 相似下限: float = 0.0, 通路: str = "自动") -> dict:
        """检索经历。`查询` 可以是**语义向量**（推荐）或 FSW 查询束。

        通路（`通路="自动"`，默认）：
          ① 库里有语义向量 且 查询维度与之匹配 → **语义余弦**（真语义）；
          ② 否则 → FSW 原通路（行为与 AHI2.1 逐位一致）。
        `通路="FSW"` 可强制走原通路（对照实验/回归用）。
        """
        self.统计["查询"] += 1
        if 通路 != "FSW" and self._向量:
            try:
                import numpy as np
                q = np.asarray(查询, dtype=np.float32).ravel()
                d = next(iter(self._向量.values())).shape[0]
                if q.shape[0] == d:
                    self.统计["语义查询"] += 1
                    return self.查经历语义(q, 相似下限=相似下限)
            except Exception:
                pass
        self.统计["FSW查询"] += 1
        return self.查经历FSW(查询)

    # ── 遥测 → 情感粗值（占位映射，步 4/5 接情绪槽语义后替换）──
    def 遥测情感态(self, 能量=None, 新异=0.0, 消化债=0.0, 饱足=1.0):
        e = float(能量) if 能量 is not None else 0.5
        点 = dict(
            V=max(-1.0, min(1.0, 0.6 * (float(饱足) - float(消化债)))),
            A=max(0.0, min(1.0, 0.3 + 0.4 * float(新异))),
            P=max(0.0, min(1.0, 0.3 + 0.4 * min(e, 1.5))),
        )                                          # G/M/X/T/Nv 留 0.5 中性
        E = self.层.编码情绪(点)
        区 = self.层.最近区域(E)
        return E, 点, 区

    def 库大小(self) -> int:
        return self.核.库大小()

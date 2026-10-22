# -*- coding: utf-8 -*-
"""meeting32/绑定.py —— #340 视语舱：视觉词↔语言共现绑定（看到什么↔说什么）。

链路：场景图 → 视觉神经.视觉词表 → 词 id 直方图 → HDC 场景向量（词袋级）；
     文本 → lang_codec.编码 → 文本向量。
     共现即学：learn(场景, 文本) 一对入舱，即学即检索（双向）。
词袋级绑定的天性：物体挪位置 → 直方图不变 → 场景向量不变 = 位置不变性（判据 B1）。
"""
from __future__ import annotations

import hashlib
import sys

import numpy as np

HERE = __import__("os").path.dirname(__file__)
sys.path.insert(0, HERE)
sys.path.insert(0, HERE + r"\..\meeting31")
sys.path.insert(0, HERE + r"\..\meeting28")

import 视觉神经 as 视  # noqa: E402
from lang_codec import 语言编解码器  # noqa: E402

D = 1024


def _词基(词id: int) -> np.ndarray:
    h = hashlib.sha256(("视词|%d" % 词id).encode("utf-8")).digest()
    buf = b""
    while len(buf) < D // 8:
        buf += hashlib.sha256(buf + h).digest()
    bits = np.frombuffer(buf[:D // 8], dtype=np.uint8)
    return ((bits[:, None] >> np.arange(8)[None, :]) & 1).reshape(-1)[:D].astype(
        np.float32) * 2 - 1


class 视语舱:
    """视觉词↔语言绑定舱（即学即检索，双向）。"""

    def __init__(self, codec: 语言编解码器 = None):
        self.codec = codec or 语言编解码器(seed=216)
        self.舱: list = []               # dict(场景向量, 文本向量, 文本, 词数)

    def 场景向量(self, img) -> np.ndarray:
        """图→视觉词直方图→HDC 场景向量（词袋级，位置不变）。"""
        词表 = 视.视觉词表(img)
        ids, 计数 = np.unique(词表.flatten(), return_counts=True)
        v = np.zeros(D, dtype=np.float64)
        for wid, n in zip(ids, 计数):
            v += _词基(int(wid)) * float(n)
        return (v / max(np.linalg.norm(v), 1e-9)).astype(np.float32)

    def learn(self, img, 文本: str) -> int:
        场景向量 = self.场景向量(img)
        文本向量 = self.codec.编码(文本)
        self.codec.注册段(文本)
        self.舱.append(dict(场景向量=场景向量, 文本向量=文本向量, 文本=文本,
                            词数=视.词典规模()))
        return len(self.舱) - 1

    def 看图说文(self, img, k: int = 1) -> list:
        """场景→最像的学过的场景→其文本（视觉词袋最近邻）。"""
        q = self.场景向量(img)
        打分 = sorted(((float(q @ 条["场景向量"]), 条["文本"])
                      for 条 in self.舱), reverse=True)
        return [dict(文本=t, 相似度=round(c, 4)) for c, t in 打分[:k]]

    def 闻文找景(self, 文本: str, k: int = 1) -> list:
        """文本→最像的学过的文本→其场景词数（语言→视觉回指）。"""
        v = self.codec.编码(文本)
        vn = float(np.linalg.norm(v)) + 1e-9
        打分 = []
        for 条 in self.舱:
            sv = 条["文本向量"]
            cos = float(v @ sv) / (vn * float(np.linalg.norm(sv)) + 1e-9)
            打分.append((cos, 条["词数"], 条["文本"]))
        打分.sort(reverse=True)
        return [dict(相似度=round(c, 4), 场景词数=w, 文本=t) for c, w, t in 打分[:k]]

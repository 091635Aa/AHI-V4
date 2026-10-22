# -*- coding: utf-8 -*-
"""meeting69/实施/听觉工作区.py —— S1 听觉工作区（独立锁；纯追加；09 §2.6）。

不触碰既有契约/状态帧/七路契约（C14）；字段与 `02` §2 一致的最小集。
"""
from __future__ import annotations

import threading

import numpy as np


class 听觉工作区:
    def __init__(self):
        self._锁 = threading.Lock()
        self.拍号 = 0
        self.听场 = np.zeros(1024, np.float32)
        self.焦点 = None            # dict(b0, s0, 带心Hz, 带宽oct, 停留拍)
        self.误差场 = np.zeros(1024, np.float32)
        self.意外度 = 0.0
        self.逐格意外 = np.zeros(192, np.float32)
        self.焦点链 = []            # list of dict
        self.流表 = {}
        self.事件表 = []
        self.波形环 = []

    def 写听觉(self, **kw):
        with self._锁:
            for k, v in kw.items():
                setattr(self, k, v)

    def 取听觉(self):
        with self._锁:
            return dict(拍号=self.拍号, 听场=self.听场.copy(), 焦点=self.焦点,
                        误差场=self.误差场.copy(), 意外度=self.意外度,
                        逐格意外=self.逐格意外.copy(), 焦点链=list(self.焦点链),
                        流表=dict(self.流表), 事件表=list(self.事件表))

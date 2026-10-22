# -*- coding: utf-8 -*-
"""meeting38/器官_我.py —— 第 7 路「自我线程」（步 3b；05 架构专家 §2.2 契约）。

- 专属线程跑 meeting31 综合体（自言空间+思维学习体）——并发铁律：阻塞内核专属线程。
- 单写 ◆自我槽（提案/遥测/向量/自言摘要/类型覆盖/审计/自体拍）。
- 消费 ◆事件槽（手/眼/耳回流）→ 容量桥 FSW 长时库（经历可检索化）。
- 脑在本步不读提案（3c 再接）；本器官不改任何既有器官/槽语义。

节奏：事件消费 2Hz；综合体拍 0.5Hz（重，专属线程内节流）。
"""
from __future__ import annotations

import os
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
AHI = os.path.dirname(HERE)                                 # AHI3 根（迁移：随文件定位）
for p in (HERE, os.path.join(HERE, "实施"), AHI, os.path.join(AHI, "meeting31")):
    if p not in sys.path:
        sys.path.insert(0, p)

from 容量桥 import 容量桥                                    # noqa: E402


class 我器:
    def __init__(self, 态, 账=None, 种子: int = 310, 挂内核: bool = True):
        self.态 = 态
        self.账 = 账
        self.自体拍 = 0
        self.停 = threading.Event()
        self.桥 = None                                       # meeting31 综合体（惰性起）
        # ★编码器接入：容量桥入库自动补语义向量 → 查经历可走真语义通路（见 容量桥 文件头）
        self.容量 = 容量桥(种子=3800, 编码器=getattr(态, "编码", None)) if 挂内核 else None
        self.统计 = dict(自体拍数=0, 事件消费=0, 库记录=0, 内核异常=0)

    # ── 内核（meeting31 综合体）惰性启动：失败不拖垮七路（A11 守门）──
    def _起内核(self):
        if self.桥 is not None:
            return True
        try:
            from synthesis import 综合体, 隐藏世界            # noqa: E402  (meeting31)
            桥 = 综合体(seed=310)
            # 知识源最小初始化（照 acceptance 冷启动：教材=标准库 json 源码 + 隐藏世界）
            语料目录 = os.path.join(os.path.dirname(os.__file__), "json")
            教材文件s = sorted(f for f in os.listdir(语料目录) if f.endswith(".py"))
            书, 界 = [], []
            for 名 in 教材文件s:
                行s = [ln.strip() for ln in open(os.path.join(语料目录, 名),
                      encoding="utf-8", errors="ignore").read().splitlines()
                      if 6 <= len(ln.strip()) <= 120]
                界 += [len(书) + k for k in range(len(行s))]
                书 += 行s
            桥.设教材(书, 界)
            桥.设世界(隐藏世界(seed=77, 动作数=8, 噪声=0.05))
            self.桥 = 桥
            return True
        except Exception as ex:
            self.统计["内核异常"] += 1
            if self.账:
                self.账.记("自我线程内核启动失败", 错=repr(ex))
            self.桥 = False                                  # False=已尝试且失败，不再重试
            return False

    def _拍自我槽(self):
        """采综合体遥测 → 情感粗值 → 写自我槽。"""
        遥测, 自言摘要, 类型覆盖, 提案, 审计 = {}, "", {}, None, {}
        E = None
        if self.桥:
            try:
                遥测 = dict(拍=getattr(self.桥, "拍", 0),
                            能量=getattr(self.桥, "e", None),
                            消化债=getattr(self.桥, "消化债", 0.0),
                            饱足=getattr(self.桥, "饱足", 1.0))
                流 = getattr(self.桥.自言, "流", [])
                最近 = 流[-6:] if 流 else []
                自言摘要 = "；".join(getattr(x, "文本", str(x))[:40] for x in 最近)
                类型覆盖 = self.桥.自言.类型覆盖()
                审计 = {"流长": len(流)}
                提案 = (最近[-1].文本[:60] if 最近 and
                        getattr(最近[-1], "类型", "") == "计划" else None)
                E, _, _ = self.容量.遥测情感态(
                    能量=遥测.get("能量"), 新异=float(getattr(self.桥, "ñ", 0.0) or 0.0),
                    消化债=遥测.get("消化债", 0.0), 饱足=遥测.get("饱足", 1.0))
            except Exception as ex:
                self.统计["内核异常"] += 1
                遥测 = dict(错误=repr(ex))
        self.态.写自我(提案=提案, 遥测=遥测, 向量=E, 自言摘要=自言摘要,
                       类型覆盖=类型覆盖, 审计=审计, 自体拍=self.自体拍)

    def 一拍(self):
        """自我线程单拍：槽刷新（先）→ 事件消费 → 综合体拍（重，最后，允许阻塞本线程）。"""
        self.自体拍 += 1
        self.统计["自体拍数"] = self.自体拍
        # ① 先刷自我槽（首次 import 综合体可能耗数秒，槽内须先有粗态）
        self._拍自我槽()
        # ② 事件回流 → 容量桥长时库
        for ev in self.态.取事件(上限=32):
            self.统计["事件消费"] += 1
            if self.容量:
                self.容量.记经历(拍=ev.get("拍", 0),
                                 类型=str(ev.get("类", "事件")),
                                 文本=str(ev.get("文本") or ev.get("类", "")),
                                 来源=str(ev.get("源", "事件槽")))
        # ③ 综合体拍（0.5Hz 节流；专属线程内允许阻塞——并发铁律下不挡其它六路）
        if self.自体拍 % 2 == 0 and self._起内核():
            try:
                self.桥.断供一拍()
            except Exception as ex:
                self.统计["内核异常"] += 1
                if self.账:
                    self.账.记("综合体拍异常", 错=repr(ex))
        self.统计["库记录"] = self.容量.库大小() if self.容量 else 0

    def 运行(self, 到点=None, 周期: float = 0.5):
        while 到点 is None or not 到点.is_set():
            t0 = time.perf_counter()
            self.一拍()
            剩 = 周期 - (time.perf_counter() - t0)
            if 剩 > 0:
                self.停.wait(剩)

    def 停机(self):
        self.停.set()

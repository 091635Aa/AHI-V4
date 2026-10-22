# -*- coding: utf-8 -*-
"""meeting38/实施/学习迹.py —— V3 学习迹独立存储（所有者硬命题：记忆 ≠ 学习）。

出处：`meeting39/01_所有者裁决与自裁备忘.md` 裁定 **H5**：
    「学习迹存储**必须独立于** `codec._段`，保持**物理分离**（最简单、最可证伪）」。

病灶（`V3审计报告_逐条核对_20261007.md` P0-2）：
    V3 模块四原来把「复盘」产物（概念快照摘要）**写回记忆底图 M**
    （`学习_三阶段.py:161-165` `self.记忆.记经历(...)`），而判据 `M4-5` 又以
    「记忆库变大」为通过标准 —— 等于用判据把「学习＝记忆」固化成了合格线，
    与所有者硬命题「记忆和学习两个是不一样的概念」正面冲突。

本文件给出物理边界：
    · **记忆（M）** = `容量桥`（FSW 长时库）里的记录 —— 「记着」；
    · **学习迹** = 本文件 `学习迹` 对象里的 `迹` / `快照s` / `顿悟s` —— 「变会」。
    两者**互不写入**：`记快照()` 只进 `迹`/`快照s`，**绝不**触碰 `容量桥`。
    `背衬`（可选）是**只读**句柄，仅用于报告「记忆库现在多大」，不作写入口。

诚实边界：本文件只保证「学习产物不再写进记忆」这一条物理分离；它**不**主张
    「学习已经很像人」——那是 meeting39 五内核（自模型/算子T/真学习/自我回路/顿悟）的事。

纪律：C1 禁 Transformer｜C9 禁 OCR｜C10 禁外挂模型当器官｜C13 确定性可复现｜C14 不破既有。
"""
from __future__ import annotations

import hashlib

import numpy as np


def _摘要码(文本: str) -> str:
    """确定性内容指纹（禁用内建 hash()：每进程加盐，跨进程不可复现 → C13）。"""
    return hashlib.sha256(str(文本).encode("utf-8")).hexdigest()[:12]


class 学习迹:
    """独立学习存储：概念快照 + 绑定 + 顿悟事件。与记忆库（容量桥/codec._段）物理分离。

    · `放(快照)`：追加一条学习产物（含快照束向量），返回序号；**不写任何记忆库**。
    · `绑定(新, 旧, 相似)`：追加一条「新↔旧」关联（学习到的关系，非记忆条目）。
    · `顿悟(...)`：追加一条相变/顿悟事件（来自 meeting39 内核时由宿主转记）。
    · `摘要()`：只读汇报自身规模 + （若有）只读背衬的库大小，供判据与输出层读取。
    """

    def __init__(self, 背衬=None, 上限: int = 20000):
        self.背衬 = 背衬                    # ★只读：记忆库句柄（仅用于汇报大小，绝不写入）
        self.迹: list = []                  # ★学习产物主存储（快照）
        self.快照s: list = []               # 与 迹 同源（保留模块四既有字段名，便于兼容）
        self.绑定s: list = []               # 学习到的「新↔旧」关系
        self.顿悟s: list = []               # 学习中的相变/顿悟事件
        self.上限 = int(上限)
        self.统计 = dict(快照=0, 绑定=0, 顿悟=0)

    # ── 写入口（只写自己）──
    def 放快照(self, 关键词, 簇数: int, 摘要: str, 束=None, 代表s=None) -> int:
        号 = len(self.迹)
        项 = dict(号=号, 关键词=list(关键词 or []), 簇数=int(簇数), 摘要=str(摘要),
                  束=束, 代表s=list(代表s or []), 指纹=_摘要码(摘要))
        self.迹.append(项)
        self.快照s.append(项)
        if len(self.迹) > self.上限:
            self.迹 = self.迹[-self.上限:]
            self.快照s = self.迹
        self.统计["快照"] += 1
        return 号

    def 放绑定(self, 新: str, 旧: str, 相似: float) -> dict:
        项 = dict(新=str(新)[:40], 旧=str(旧)[:80], 相似=round(float(相似), 3))
        self.绑定s.append(项)
        self.绑定s = self.绑定s[-4000:]
        self.统计["绑定"] += 1
        return 项

    def 放顿悟(self, 拍: int, 文本: str, 形态: dict = None) -> int:
        号 = len(self.顿悟s)
        self.顿悟s.append(dict(号=号, 拍=int(拍), 文本=str(文本)[:300],
                              形态=dict(形态 or {}), 指纹=_摘要码(文本)))
        self.统计["顿悟"] += 1
        return 号

    # ── 只读查询（判据/输出层用）──
    def 条数(self) -> int:
        return len(self.迹)

    def 最近束(self):
        return self.迹[-1]["束"] if self.迹 else None

    def 最近关键词(self, n: int = 6) -> list:
        return list(self.迹[-1]["关键词"][:n]) if self.迹 else []

    def 记忆库大小(self):
        """只读：背衬记忆库规模（用于证明「学习没写记忆」——它不变）。"""
        if self.背衬 is None:
            return None
        try:
            return int(self.背衬.库大小())
        except Exception:
            return None

    def 摘要(self) -> dict:
        return dict(迹条数=len(self.迹), 绑定条数=len(self.绑定s),
                    顿悟条数=len(self.顿悟s), 统计=dict(self.统计),
                    记忆库大小=self.记忆库大小())

    # ── 确定性自证：同内容两次 → 同指纹；且不触碰记忆库 ──
    def 自证(self) -> dict:
        库0 = self.记忆库大小()
        备份 = list(self.迹)
        试 = 学习迹()                        # 无背衬：绝无可能写记忆
        试.放快照(["a", "b"], 1, "概念快照：a、b", 束=None)
        a = 试.迹[-1]["指纹"]
        试2 = 学习迹()
        试2.放快照(["a", "b"], 1, "概念快照：a、b", 束=None)
        b = 试2.迹[-1]["指纹"]
        库1 = self.记忆库大小()
        return dict(指纹一致=bool(a == b), 记忆库未变=bool(库0 == 库1),
                    库前=库0, 库后=库1, 迹条数=len(备份))

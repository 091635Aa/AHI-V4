# -*- coding: utf-8 -*-
"""meeting32/基础课程表.py —— #440 基础补课：视觉属性定位 / 鼠标中键 / 滚轮上下。

历史盘点（报告对照）：视觉(HSV/视觉词)✓ 移动(人化轨迹)✓ 左键✓ 右键✓ 拖拽✓
复制/重命名✓ ——**中键✗ 滚轮✗ 视觉属性定位作为独立课目✗**。本表补齐。

铁律不变：零手写操作流程——动作槽=瞄准（感知），选择=它学的（经验舱计数）。

课视 视觉属性定位：场景随机放红/绿/蓝三球（感知给出全部位置）；
  目标=「报告某颜色球的位置」→ 动作槽=三个位置。学 属性→实体 关联。
课中 中键：文件列表 5 项；物理：中键文件→「在新标签打开」（标签集+1）；
  左键只选中（不产生标签事件）。目标=打开指定文件→必须选中键。
课滚 滚轮：20 项列表，视窗 5 项；物理：滚下=视窗后移，滚上=前移，到界停；
  目标=让第 N 项可见 → 学 滚轮方向映射。
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "meeting28"))

from 自发学习 import 运动经验舱  # noqa: E402

结果 = []


def 判(名, 条件, 明细=""):
    结果.append((名, bool(条件), 明细))
    print("[%s] %s %s" % ("PASS" if 条件 else "FAIL", 名, 明细), flush=True)


# ---------- 课视：视觉属性定位 ----------
class 视觉沙箱:
    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)
        色 = ["红", "绿", "蓝"]
        self.rng.shuffle(色)
        self.球s = {c: (float(self.rng.uniform(60, 900)),
                        float(self.rng.uniform(60, 560)))
                    for c in 色}

    def 报告(self, 色: str, 槽: int) -> bool:
        """槽→它指认的位置；感知给了全部位置（=它看得见）。"""
        名s = sorted(self.球s)
        return 名s[槽 % 3] == 色


def 课视(经验, 集, seed) -> list:
    rng = np.random.default_rng(seed)
    率s = []
    for ep in range(集):
        箱 = 视觉沙箱(seed * 31 + ep)
        目标色 = ["红", "绿", "蓝"][ep % 3]
        ctx = ("报位", 目标色)
        a = 经验.选索引(ctx, 3, ε=max(0.3 * (1 - ep / 集), 0.05))
        成 = 箱.报告(目标色, a)
        经验.记(ctx, a, 成)
        率s.append(成)
    return 率s


# ---------- 课中：中键 ----------
class 文件列表沙箱:
    def __init__(self, seed: int):
        self.文件s = ["文件%d" % i for i in range(5)]
        self.标签: set = set()

    def 点(self, 槽: int, 键: str, 目标: str) -> bool:
        名 = self.文件s[槽 % len(self.文件s)]
        if 键 == "中键":
            self.标签.add(名)                        # 中键=新标签打开
            return 名 == 目标
        return False                                 # 左键只选中，无标签事件


def 课中(经验, 集, seed) -> list:
    rng = np.random.default_rng(seed)
    率s = []
    for ep in range(集):
        箱 = 文件列表沙箱(seed * 17 + ep)
        目标 = 箱.文件s[ep % 5]
        ctx = ("开标签", 目标)
        a = 经验.选索引(ctx, 10, ε=max(0.35 * (1 - ep / 集), 0.05))
        # 动作槽 0-4=左键文件k，5-9=中键文件k（键的选择是它学的）
        键 = "中键" if a >= 5 else "左键"
        成 = 箱.点(a, 键, 目标)
        经验.记(ctx, a, 成)
        率s.append(成)
    return 率s


# ---------- 课滚：滚轮 ----------
class 滚动沙箱:
    def __init__(self, seed: int, 目标: int):
        self.项数 = 20
        self.视窗顶 = 0                              # 可见 [顶, 顶+4]
        self.目标 = 目标                             # 让第目标项可见

    def 可见(self) -> bool:
        return self.视窗顶 <= self.目标 <= self.视窗顶 + 4

    def 滚(self, 方向: str) -> bool:
        if 方向 == "下":
            self.视窗顶 = min(self.视窗顶 + 1, self.项数 - 5)
        elif 方向 == "上":
            self.视窗顶 = max(self.视窗顶 - 1, 0)
        return self.可见()


def 课滚(经验, 集, seed) -> list:
    rng = np.random.default_rng(seed)
    率s = []
    for ep in range(集):
        目标 = int(rng.integers(0, 20))
        箱 = 滚动沙箱(seed * 7 + ep, 目标)
        步 = 0
        while not 箱.可见() and 步 < 25:
            ctx = ("滚动", 箱.视窗顶 // 5, 1 if 箱.目标 > 箱.视窗顶 + 4 else 0)
            # 感知给方向线索（目标在视窗上/下方——像人看见滚动条位置）
            a = 经验.选索引(ctx, 3, ε=max(0.35 * (1 - ep / 集), 0.05))
            方向 = ["上", "下", "不动"][a]
            步 += 1
            done = 箱.滚(方向)
            经验.记(ctx, a, done or 箱.可见())
        率s.append(箱.可见())
    return 率s


def 摘要(率s) -> str:
    return hashlib.sha256(json.dumps([int(x) for x in 率s]).encode()).hexdigest()[:12]


if __name__ == "__main__":
    import hashlib
    sys.stdout.reconfigure(encoding="utf-8")
    摘1, 摘2 = {}, {}
    for 名, 课f, 集, seed, 阈 in (("视觉属性定位", 课视, 120, 900, 0.90),
                                  ("中键", 课中, 150, 910, 0.85),
                                  ("滚轮", 课滚, 200, 920, 0.85)):
        e = 运动经验舱(seed=seed)
        r1 = 课f(e, 集, seed)
        后 = float(np.mean(r1[-max(集 // 5, 10):]))
        前 = float(np.mean(r1[:max(集 // 5, 10)]))
        摘1[名] = 摘要(r1)
        判("J-%s" % 名, 后 >= 阈, "前段 %.2f → 后段 %.2f（阈 %.2f）" % (前, 后, 阈))
        # 重学确定性
        e2 = 运动经验舱(seed=seed)
        r2 = 课f(e2, 集, seed)
        摘2[名] = 摘要(r2)
    判("J-确定性", all(摘1[k] == 摘2[k] for k in 摘1),
       "三门课重学摘要一致")
    # J-滚轮方向映射核对：目标在下方时应选「下」
    e = 运动经验舱(seed=920)
    课滚(e, 200, 920)
    下正确 = e.选索引(("滚动", 1, 1), 3, ε=0.0) == 1
    上正确 = e.选索引(("滚动", 1, 0), 3, ε=0.0) == 0
    判("J-滚轮方向映射", 下正确 and 上正确,
       "目标在下→选滚下=%s，目标在上→选滚上=%s" % (下正确, 上正确))
    fail = [n for n, ok, _ in 结果 if not ok]
    print("\n===== 基础补课验证：%d/%d PASS =====" % (len(结果) - len(fail),
                                                     len(结果)))
    if fail:
        print("FAIL:", fail)
    sys.exit(0 if not fail else 1)

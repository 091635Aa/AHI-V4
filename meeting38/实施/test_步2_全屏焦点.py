# -*- coding: utf-8 -*-
"""实施/test_步2_全屏焦点.py —— 视觉重启 步2 判据（V1 全屏 + V2 焦点加权）。

依据：`步5_文本对话态与全屏视场_设计.md` §2（视野全局化）+ §7 V1/V2。
  V1 全屏：眼采集尺寸 == 全屏；`区域=None`；裁剪路径不可达（默认即整屏）
  V2 焦点加权：焦点邻域评分增益 > 1（焦点格 1.2×，远处 → 1.0），且**周边仍出候选**（不减信息）

单测纯内存：眼器的取帧用桩（不触真屏/不开相机）；V2 用真实 视场前端（纯 numpy）。
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
根 = os.path.dirname(HERE)
for p in (根, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import 器官_眼 as 眼mod                        # noqa: E402
from 共存态 import 共存态                       # noqa: E402
import 视场前端 as 前端mod                      # noqa: E402


# ── 桩：连续采集（返回固定整屏帧）+ mss（降级路径，本测不用）──
class _假连续:
    def __init__(self, 帧):
        self.帧 = 帧

    def 取(self):
        return (self.帧, time.time(), True)

    def 起(self):
        return True

    def 停(self):
        pass

    def 状态(self):
        return {}


class _假mss:
    def __init__(self, 帧):
        self.monitors = [None, {"w": int(帧.shape[1]), "h": int(帧.shape[0])}]
        self._帧 = 帧

    def grab(self, mon):
        return self._帧


def 判V1(态) -> list:
    屏 = np.zeros((1080, 1920, 3), dtype=np.uint8)
    屏[:, :960] = 200                          # 半边亮，供变化率/元素有输入
    眼mod.连续采集 = lambda: _假连续(屏)
    眼mod.mss_开 = lambda: _假mss(屏)
    眼 = 眼mod.眼器(态, 台账=None)
    img, 屏幕 = 眼._截屏()
    判 = []
    判.append(("V1-a 区域默认 None（整屏）", 眼.区域 is None, "区域=%r" % (眼.区域,)))
    判.append(("V1-b 取帧尺寸 == 全屏",
               (int(img.shape[1]), int(img.shape[0])) == (1920, 1080),
               "帧=%dx%d" % (img.shape[1], img.shape[0])))
    判.append(("V1-c 屏幕 == 全屏", tuple(屏幕) == (1920, 1080), "屏幕=%s" % (屏幕,)))
    判.append(("V1-d 裁剪路径不可达（_裁 恒等）",
               np.array_equal(眼._裁(屏), 屏), "恒等=%s" % np.array_equal(眼._裁(屏), 屏)))
    眼.一轮()
    v = 态.取视觉()
    判.append(("V1-e 视觉槽.屏幕 == 全屏", tuple(v.get("屏幕", ())) == (1920, 1080),
               "槽屏幕=%s" % (v.get("屏幕"),)))
    return 判


def 判V2(态) -> list:
    前端 = 前端mod.视场前端(态, 种子=3800)
    gx, gy = 前端.注视                            # 默认 (8,4)
    判 = []
    增 = 前端.焦点增益场()
    焦点值 = float(增[gy * 前端.格x + gx])
    远角 = float(增[前端.格x * 前端.格y - 1])
    判.append(("V2-a 焦点处增益 == 1.2", abs(焦点值 - 1.2) < 1e-4, "焦点增益=%.4f" % 焦点值))
    判.append(("V2-b 远处增益 → 1.0", abs(远角 - 1.0) < 1e-3 and 增.max() <= 1.2 + 1e-4,
               "远角=%.4f max=%.4f" % (远角, 增.max())))

    # 构造：焦点格 意外 0.9、远角格 意外 1.0 → 无增益时 argmax=远角；有增益时被抬成焦点格
    逐格 = np.zeros(前端.格x * 前端.格y, dtype=np.float32)
    逐格[gy * 前端.格x + gx] = 0.9
    远i = 8 * 前端.格x + 15                        # (15,8) 远角
    逐格[远i] = 1.0
    新 = 前端.下一注视(逐格)
    判.append(("V2-c 乘性增益生效（焦点格夺回 argmax）", tuple(新) == (gx, gy),
               "下一注视=%s（期望 %s）" % (tuple(新), (gx, gy))))

    旧 = 前端mod.焦点增益
    前端mod.焦点增益 = 0.0                          # 关闭增益作对照
    新0 = 前端.下一注视(逐格)
    前端mod.焦点增益 = 旧                           # 还原
    判.append(("V2-d 对照（无增益时远角胜）", tuple(新0) == (15, 8),
               "无增益下一注视=%s（期望 (15,8)）" % (tuple(新0),)))

    # 周边不减信息：远角出超阈意外 → 在场者仍给出该处连通区（非裁剪）
    逐格2 = np.zeros(前端.格x * 前端.格y, dtype=np.float32)
    逐格2[远i] = 1.0
    区s = 前端.在场者(逐格2)
    命中 = any(r["格"][0] <= 15 < r["格"][0] + r["格"][2]
               and r["格"][1] <= 8 < r["格"][1] + r["格"][3] for r in 区s)
    判.append(("V2-e 周边仍出候选（不减信息）", 命中 and len(区s) >= 1,
               "周边区数=%d 命中=%s" % (len(区s), 命中)))
    return 判


def 主():
    态 = 共存态(种子=3800, 用gpu=False)
    try:
        判 = 判V1(态) + 判V2(态)
    finally:
        pass
    全过 = True
    for 名, 过, 证 in 判:
        print("[%s] %s  %s" % ("过" if 过 else "败", 名, 证))
        全过 = 全过 and 过
    print("\n==== V1/V2 全屏+焦点加权总判：%s ====" % ("全过" if 全过 else "未过"))
    return 全过


if __name__ == "__main__":
    sys.exit(0 if 主() else 1)
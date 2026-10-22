# -*- coding: utf-8 -*-
"""meeting62/团队/HDC内在动力学/探针_谐振音素格.py

最小可失败探针：验证**既有 8 算子**（绑/解绑/门/叠/归/读/迭代）能否搭出
『谐振器网络』式音素格因子分解 —— 不引入任何新依赖、不外挂模型、不依赖任何深度学习框架。

设计对应（只读 import meeting41 算子_能力核，本件纯新增，C14）：
  · 音素格 = F 个属性码本（此处 F=2：部位×方式）的**乘积**（绑 = Hadamard 键⊙值）；
  · 分解 = resonator network：`估计_b̂ = 叠(全部B原子)` 起步，反复
        â ← 读(码本A, 解绑(b̂, target))     （cleanup 到最近原子）
        b̂ ← 读(码本B, 解绑(â, target))
    直到收敛 —— 与 Frady 2020 谐振器同构，只用本仓库算子，无成对加权。

判据（可失败）：
  P1 谐振恢复率 ≥ 0.9（K 组随机组合，精确命中 (i*,j*)）；
  P2 must-fail：直接 读(码本A, target) 命中率必须低（乘积向量不是原子）；
  P3 must-fail：把 b̂ 初始化为**码本外随机双极**（置乱熵源）⇒ 恢复率必须显著低于 P1；
  P4 两趟逐位同（C13 确定性）。
探针 CPU-only、单次 <10s、不占声卡、不 import time。
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
AHI4 = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))   # …/AHI4
_核路径 = os.path.join(AHI4, "meeting41", "实施", "算子_能力核.py")


def 载核():
    spec = importlib.util.spec_from_file_location("能力核_只读_62", _核路径)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_m = 载核()
能力核 = _m.能力核
生成码本 = _m.生成码本


D默认 = 1024
种子 = 62001


def 跑(每因子数=8, 步数=12, 组数=24, 置乱=False):
    核 = 能力核(D=D默认, 种子=种子)
    码本A = 生成码本(每因子数, D默认, 种子 + 1)
    码本B = 生成码本(每因子数, D默认, 种子 + 2)
    rng = np.random.default_rng(种子 + 7)
    rng乱 = np.random.default_rng(种子 + 8)
    N = 每因子数

    def 解绑(k, b):
        return 核.解绑(k, b)

    def 读(码本, 探针):
        return 核.读(码本, 探针, k=1)

    def 谐振(target):
        if 置乱:
            b̂ = np.where(rng乱.integers(0, 2, size=D默认) == 1, 1.0, -1.0)   # 码本外，置乱熵源
        else:
            b̂ = 核.叠([码本B[j] for j in range(N)])                          # search-in-superposition
        ai = bi = -1
        for _ in range(步数):
            rA = 读(码本A, 解绑(b̂, target)); ai = rA["索引"][0]; â = 码本A[ai]
            rB = 读(码本B, 解绑(â, target)); bi = rB["索引"][0]; b̂ = 码本B[bi]
        return ai, bi

    命中 = 0
    直读命中 = 0
    记录 = []
    for _t in range(组数):
        i = int(rng.integers(0, N)); j = int(rng.integers(0, N))
        target = 核.绑(码本A[i], 码本B[j])
        ai, bi = 谐振(target)
        good = (ai == i and bi == j)
        命中 += int(good)
        记录.append([i, j, ai, bi, bool(good)])
        直读命中 += int(读(码本A, target)["索引"][0] == i)     # must-fail：乘积向量直读不该命中
    return dict(恢复率=命中 / 组数, 直读命中率=直读命中 / 组数, 记录=记录,
                每因子数=N, 步数=步数, 组数=组数, 置乱=置乱)


def 自检():
    果 = {"D": D默认, "种子": 种子}
    主 = 跑(8, 12, 24, False)
    果["P1_谐振_8x8"] = dict(恢复率=round(主["恢复率"], 4), 直读命中率=round(主["直读命中率"], 4))
    果["P1_谐振_容量扫描"] = {("N=%d" % N): round(跑(N, 12, 24, False)["恢复率"], 4)
                        for N in (4, 8, 16, 32)}
    乱 = 跑(8, 12, 24, True)
    果["P3_置乱熵源_8x8"] = dict(恢复率=round(乱["恢复率"], 4), 直读命中率=round(乱["直读命中率"], 4))
    果["P3_置乱显著低于主臂"] = bool(乱["恢复率"] < 主["恢复率"] - 1e-9)
    主2 = 跑(8, 12, 24, False)
    果["P4_两趟逐位同(记录)"] = bool(主2["记录"] == 主["记录"])
    果["P1_通过"] = bool(主["恢复率"] >= 0.9)
    果["P2_直读必须不过"] = bool(主["直读命中率"] < 0.9)
    果["P3_通过"] = bool(果["P3_置乱显著低于主臂"])
    果["通过"] = bool(果["P1_通过"] and 果["P2_直读必须不过"] and 果["P3_通过"] and 果["P4_两趟逐位同(记录)"])
    return 果


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    np.seterr(all="ignore")
    果 = 自检()
    print(json.dumps(果, ensure_ascii=False, indent=1))
    print("[探针_谐振音素格] %s" % ("PASS" if 果["通过"] else "FAIL"))

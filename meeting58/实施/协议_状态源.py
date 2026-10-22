# -*- coding: utf-8 -*-
"""meeting58/实施/协议_状态源.py —— 内核侧适配：只读状态帧（D核=64 真实 V5 双通道）。

解耦边界：
  · 本件只读内核（V5_双通道 / 两区室核），**不 import 声层**；
  · 产出的 `状态帧` 是不可变值（bytes + 指纹），行层/声层都只能读；
  · 声层要 D=1024，由 `工具_确定性.确定性投影` 在**声层内部**完成，不改内核维度。

注：`快照()` 的 D核 是内核真实维度（64），不是声带的 1024；任何实现不得把两者混用。
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
AHI4 = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(AHI4, "meeting53", "实施"))

from 接口契约_角色声线 import 状态帧, 契约版本, 契约错误, E1007  # noqa: E402
from 工具_确定性 import 指纹  # noqa: E402

V5 = None


def _载V5():
    global V5
    if V5 is None:
        import V5_双通道 as _m
        V5 = _m
    return V5


class 双通道状态源:
    """真实内核状态源：双通道体（两区室核）× 接口沙盒，预跑轨迹后只读快照。"""

    协议版本 = 契约版本

    def __init__(self, 种子: int = None, 拍数: int = 240, D: int = 64):
        m = _载V5()
        self.种子 = int(种子 if 种子 is not None else m.种子默认)
        self.D核 = int(D)
        self.体 = m.双通道体(D=self.D核, 种子=self.种子)
        self.环 = m.接口沙盒(种子=self.种子 + 1)
        self._帧列 = []
        for t in range(int(拍数)):
            self.环.新试次()
            self.体.一拍(观测id=self.环.cue, 学习=False, 老师=False, 拍=t, 探索率=0.0)
            self._帧列.append(状态帧.建(
                拍=t, 观测=int(self.环.cue), o用=self.体.o用, 内心=self.体.内心,
                自我标量=float(self.体.自我标量), 惊喜=float(self.体.惊喜),
                遥测=dict(能量=0.5, 新异=float(abs(self.体.惊喜)), 消化债=0.0, 饱足=0.5)))
        self._轨迹 = 指纹(*[f.指纹 for f in self._帧列])
        self._关 = False

    def 快照(self, 拍: int | None = None) -> 状态帧:
        if self._关:
            raise 契约错误(E1007, "状态源已关闭")
        t = 0 if 拍 is None else int(拍) % len(self._帧列)
        return self._帧列[t]

    def 轨迹指纹(self) -> str:
        return self._轨迹

    def 关(self) -> None:
        self._关 = True


class 合成状态源:
    """判据用替身：确定性合成轨迹（不依赖内核），用于隔离「渲染层」单独验收。"""

    协议版本 = 契约版本

    def __init__(self, 拍数: int = 240, D核: int = 64, 种子: int = 7):
        self.D核 = int(D核)
        self.种子 = int(种子)
        rng = np.random.default_rng(self.种子)
        self._帧列 = []
        for t in range(int(拍数)):
            a = rng.standard_normal(self.D核)
            b = rng.standard_normal(self.D核)
            a = np.sign(a) * np.abs(a) ** 0.5
            b = np.sign(b) * np.abs(b) ** 0.5
            self._帧列.append(状态帧.建(
                拍=t, 观测=t % 5, o用=a, 内心=b, 自我标量=float(0.5 + 0.2 * rng.standard_normal()),
                惊喜=float(abs(rng.standard_normal()) * 0.1),
                遥测=dict(能量=float(rng.random()), 新异=float(rng.random()),
                         消化债=float(rng.random() * 10), 饱足=float(rng.random()))))
        self._轨迹 = 指纹(*[f.指纹 for f in self._帧列])

    def 快照(self, 拍: int | None = None) -> 状态帧:
        t = 0 if 拍 is None else int(拍) % len(self._帧列)
        return self._帧列[t]

    def 轨迹指纹(self) -> str:
        return self._轨迹

    def 关(self) -> None:
        pass


if __name__ == "__main__":
    import sys as _s
    try:
        _s.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    源 = 双通道状态源(种子=53001, 拍数=8)
    f = 源.快照(0)
    print("D核", f.D核, "指纹", f.指纹, "轨迹", 源.轨迹指纹())
    print("o用前5", f.数组()[:5])
    print("两趟同", 源.快照(0).指纹 == f.指纹)
    print("合成源 D核", 合成状态源(拍数=4).D核)

# -*- coding: utf-8 -*-
"""test_步4_演播台.py —— 步4 全屏只读实况演播台 判据（V-全屏 / V-只读 / V-思考外层 / V-ESC）

不依赖真窗、不真动鼠标、可复跑：用**假态**（记录任何写访问）+ **假眼**驱动渲染器，
以静态源码审计 + 运行期写访问拦截双重证明「只读」；用 建窗=False 单测 ESC 口径。
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
根 = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, 根)

from 演播台 import 演播渲染, 演播窗, 思考行, _ESC按下   # noqa: E402

源 = open(os.path.join(HERE, "演播台.py"), encoding="utf-8").read()

写名s = ["态.写视场", "态.写视觉", "态.写动作", "态.写意图", "态.写思考", "态.排队说",
         "态.写用户说", "态.写它说", "态.写它想", "态.写脑内", "态.置醒", "态.写语义",
         "态.写听觉", "态.写自我", "态.写对话"]


class 假眼:
    def __init__(self, 帧):
        self.最近帧 = 帧
        self.轮 = 7
        self.采集后端 = "假"
        class _前: 格x, 格y = 16, 9
        self.前端 = _前()
        class _连:
            @staticmethod
            def 状态():
                return {"帧年龄ms": 12}
        self.连续 = _连()


class 假态:
    """只提供 取* 读口；任何 写* 访问被 __getattr__ 记进 self.写访问（判据用）。"""

    def __init__(self):
        self.写访问 = []
        self.思考s = [
            dict(t=1.0, 类="看", 文="眼前 6 件东西（内部回响）", 层="内"),
            dict(t=2.0, 类="想", 文="我该点左上那个方块", 层="外"),
            dict(t=3.0, 类="招", 文="手往 (300,200) 移过去", 层="外"),
            dict(t=4.0, 类="预", 文="点了它可能变色", 层="外"),
            dict(t=5.0, 类="果", 文="刚才那次没变", 层="内"),
        ]
        self.gx, self.gy = 16, 9
        self.逐格 = np.zeros(self.gx * self.gy, dtype=np.float32)
        self.逐格[3 * self.gx + 4] = 0.82            # 一个惊讶格

    def 取视场(self):
        return dict(注视=(4, 3, 2), 逐格意外=self.逐格, 意外度=0.31, 眼跳数=5,
                    注视链=[(4, 3), (4, 3), (6, 5)])

    def 取视觉(self):
        return dict(屏幕=(1920, 1080),
                    元素s=[dict(句柄="E0", 框=(300, 200, 120, 60))])

    def 取动作(self):
        return dict(状态="闲", 动作计数=0, 轨迹=[], 光标=(400, 300))

    def 取意图(self):
        return dict(当前=dict(目标点=(300, 230)), 版本=1)

    def 取思考(self, 上限: int = 40, 只要外: bool = True):
        xs = [dict(x) for x in self.思考s]
        if 只要外:
            xs = [x for x in xs if x.get("层", "外") == "外"]
        return xs[-int(上限):]

    def __getattr__(self, name):                     # 只在正常查找失败时触发
        if str(name).startswith("写") or str(name) in ("排队说", "置醒"):
            self.写访问.append(name)
            raise AssertionError("演播台 不得写槽：%s" % name)
        raise AttributeError(name)


def 项(名, 过, 证据):
    print("  [%s] %-16s %s" % ("PASS" if 过 else "FAIL", 名, 证据))
    return bool(过)


def main():
    过s = []
    屏 = (1920, 1080)
    假 = 假态()
    眼 = 假眼(np.full((1080, 1920, 3), 90, np.uint8))
    渲 = 演播渲染(假, 眼, 尺寸=屏)

    # ── V-全屏 ──
    过s.append(项("全屏-a 尺寸", (渲.W, 渲.H) == 屏,
                  "画布 %dx%d == 屏 %s" % (渲.W, 渲.H, 屏)))
    画 = 渲.画()
    dw, dh = int(1920 * 渲.s), int(1080 * 渲.s)
    触边 = (dw == 渲.ww) or (dh == 渲.wh)
    过s.append(项("全屏-b 整屏不裁", (dw <= 渲.ww and dh <= 渲.wh and 触边 and 渲.s > 0),
                  "帧缩放 %dx%d ≤ 世界 %dx%d（触边=%s s=%.3f）" % (dw, dh, 渲.ww, 渲.wh, 触边, 渲.s)))
    过s.append(项("全屏-c 源码", ("眼器(态, None)" in 源) and ("区域=量区域" not in 源),
                  "眼全屏构造在、无区域裁剪"))

    # ── V-只读 ──
    命中 = [t for t in 写名s if t in 源]
    过s.append(项("只读-a 源码审计", len(命中) == 0, "写槽命中=%s" % (命中 or "无")))
    过s.append(项("只读-b 运行期", (len(假.写访问) == 0 and isinstance(画, np.ndarray)
                                   and 画.shape == (1080, 1920, 3)),
                  "写访问=%d 画.shape=%s" % (len(假.写访问), tuple(画.shape))))

    # ── V-思考外层 ──
    行 = 思考行(假, 64)
    只外 = all(x["层"] == "外" for x in 行)
    过s.append(项("思考-外层 过滤", (只外 and len(行) == 3 and "看" not in [x["类"] for x in 行]),
                  "行=%s（内层『看/果』已滤）" % [x["类"] for x in 行]))
    过s.append(项("思考-外层 源码", "只要外=True" in 源, "画() 调 取思考(只要外=True)"))

    # ── V-ESC ──
    窗 = 演播窗(渲, 建窗=False)
    窗.按ESC()
    过s.append(项("ESC-a 绑定口径", (窗.停.is_set() and 窗.退出因 == "ESC"),
                  "停=%s 因=%s" % (窗.停.is_set(), 窗.退出因)))
    窗2 = 演播窗(渲, 建窗=False)
    窗2.按键q()
    过s.append(项("ESC-b q 退出", (窗2.停.is_set() and 窗2.退出因 == "q"),
                  "因=%s" % 窗2.退出因))
    过s.append(项("ESC-c 系统级轮询", isinstance(_ESC按下(), bool), "_ESC按下() 可调用"))

    ok = all(过s)
    print("\n步4 演播台判据：%d/%d %s" % (sum(过s), len(过s), "全过" if ok else "有失败"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
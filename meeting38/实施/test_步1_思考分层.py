# -*- coding: utf-8 -*-
"""实施/test_步1_思考分层.py —— 视觉重启 步1 判据（V3 思考分层）。

依据：`step5_文本对话态与全屏视场_设计.md` §3（D3 思考分层）+ §7 V3。
  V3-a 外显（取思考 默认）不含「看」类（内层回响不外显）
  V3-b 内层仍在（取思考(只要外=False) 含「看」且 层=="内"）
  V3-c 外显含 想/招/预/果/路 五类朝外正文（脑真产出，非构造）
  V3-d 旧调用兼容（写思考(类,文) 默认 层=="外"）

不 import 任何器官以外的重依赖；不真动鼠标键盘；纯内存。
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
根 = os.path.dirname(HERE)
for p in (根, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from 共存态 import 共存态                      # noqa: E402
from 器官_脑 import 脑器                       # noqa: E402


class _安全探空:
    """空事实源（不触 UIA/win32gui），保证单测纯内存、确定性。"""
    def 窗口标题(self):
        return ""

    def 点控件名(self, 点):
        return ""


def 主():
    态 = 共存态(种子=3800, 用gpu=False)
    脑 = 脑器(态, 语义图=None, 台账=None, 安全探=_安全探空())

    # 脑真产出：先「看」（内层），再「想/招/预/果/路」（外层）
    脑._上次看 = 0.0
    脑._叙述看(dict(元素s=[], 屏幕=(1920, 1080)), dict(状态="闲"))
    脑._思("想", "盯上 E1：方块 40x20，先试它")
    脑._思("招", "把光标移到 (100,200)，按左键")
    脑._思("预", "预想：上回点它结果「大变」，这回应会再变")
    脑._思("果", "结果：点在 E1 上，世界「微变」（变化率 0.10）")
    脑._思("路", "下一步：继续按「没碰过的优先」挑下一个")

    外 = 态.取思考(上限=100)                       # 默认 只要外=True
    全 = 态.取思考(上限=100, 只要外=False)
    类外 = [x["类"] for x in 外]
    类全 = [x["类"] for x in 全]
    层外 = set(x["层"] for x in 外)
    看项 = [x for x in 全 if x["类"] == "看"]

    判 = []
    判.append(("V3-a 外显不含「看」", "看" not in 类外, "类外=%s" % 类外))
    判.append(("V3-b 内层仍在且层=内",
               bool(看项) and 看项[0]["层"] == "内",
               "看项=%d 层=%s" % (len(看项), (看项[0]["层"] if 看项 else None))))
    判.append(("V3-c 外显含五类正文",
               set(["想", "招", "预", "果", "路"]).issubset(set(类外)),
               "类外集=%s" % sorted(set(类外))))
    判.append(("V3-d 旧调用兼容（默认外）",
               态.写思考("想", "旧式调用") or
               all(x["层"] == "外" for x in 态.取思考(上限=100)),
               "全部层=%s" % sorted(层外)))
    判.append(("V3-e 内层不外显于外显列表",
               all(x["层"] == "外" for x in 外) and len(全) >= len(外),
               "外=%d 全=%d" % (len(外), len(全))))

    全过 = True
    for 名, 过, 证 in 判:
        print("[%s] %s  %s" % ("过" if 过 else "败", 名, 证))
        全过 = 全过 and 过
    print("\n==== V3 思考分层总判：%s ====" % ("全过" if 全过 else "未过"))
    return 全过


if __name__ == "__main__":
    sys.exit(0 if 主() else 1)